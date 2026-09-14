//! A host interpreter for the IR — the reference the GPU is checked against.
//!
//! The point of computing the reference from the **same IR** the back end compiles, rather
//! than from a hand-written CPU version of the kernel, is that the two cannot drift. A
//! hand-written reference tests whether two people understood the source the same way. This
//! tests the back end, which is the thing that has never been validated.
//!
//! Every operation here is the IEEE one PTX emits: `add.rn`, `mul.rn`, `fma.rn`. `f32::mul_add`
//! is a true fused multiply-add, so it matches `fma.rn.f32` bit for bit — which is why the
//! comparison can demand exact equality rather than a tolerance. **A tolerance would hide
//! exactly the code-generation bugs this exists to catch.**

use std::collections::BTreeMap;

use crate::ast::{BinOp, Ty};
use crate::ir::{KernelIr, Op, RegId};

#[derive(Debug, thiserror::Error)]
pub enum EvalError {
    #[error("no buffer named `{0}` was supplied")]
    MissingBuffer(String),
    #[error("no scalar named `{0}` was supplied")]
    MissingScalar(String),
    #[error("buffer `{name}` has {got} elements, expected at least {want}")]
    ShortBuffer {
        name: String,
        got: usize,
        want: usize,
    },
}

/// Host-side inputs, keyed by parameter name.
#[derive(Debug, Clone, Default)]
pub struct Inputs {
    pub scalars: BTreeMap<String, f32>,
    pub buffers: BTreeMap<String, Vec<f32>>,
}

/// Run every element of the kernel on the host, returning the drained buffers.
///
/// Element order is irrelevant: v1 has no cross-element communication, which is exactly why
/// one thread per element is a legal schedule in the first place.
pub fn eval(ir: &KernelIr, n: usize, inputs: &Inputs) -> Result<Inputs, EvalError> {
    for p in &ir.params {
        match p.ty {
            Ty::BufF32 => {
                let b = inputs
                    .buffers
                    .get(&p.name)
                    .ok_or_else(|| EvalError::MissingBuffer(p.name.clone()))?;
                if b.len() < n {
                    return Err(EvalError::ShortBuffer {
                        name: p.name.clone(),
                        got: b.len(),
                        want: n,
                    });
                }
            }
            Ty::F32 => {
                if !inputs.scalars.contains_key(&p.name) {
                    return Err(EvalError::MissingScalar(p.name.clone()));
                }
            }
            Ty::U32 => {}
        }
    }

    let mut out = inputs.clone();
    for i in 0..n {
        let mut regs: BTreeMap<RegId, f32> = BTreeMap::new();
        for op in &ir.ops {
            let v = match op {
                Op::Load { buffer, .. } => out
                    .buffers
                    .get(buffer)
                    .ok_or_else(|| EvalError::MissingBuffer(buffer.clone()))?[i],
                Op::Param { name, .. } => *out
                    .scalars
                    .get(name)
                    .ok_or_else(|| EvalError::MissingScalar(name.clone()))?,
                Op::Const { value, .. } => *value as f32,
                Op::Bin { op, lhs, rhs, .. } => {
                    let a = regs[lhs];
                    let b = regs[rhs];
                    match op {
                        BinOp::Add => a + b,
                        BinOp::Sub => a - b,
                        BinOp::Mul => a * b,
                        BinOp::Div => a / b,
                    }
                }
                // A true FMA: one rounding, matching `fma.rn.f32`. Writing `a * b + c` here
                // would round twice and disagree with the GPU on the last bit.
                Op::Fma { a, b, c, .. } => regs[a].mul_add(regs[b], regs[c]),
                Op::Neg { src, .. } => -regs[src],
            };
            regs.insert(op.dst(), v);
        }
        // Every drain lands after the whole body, so a kernel that reads y and writes y sees
        // the old value throughout the element, exactly as the generated code does.
        for (buffer, reg) in &ir.drains {
            let v = regs[reg];
            out.buffers
                .get_mut(buffer)
                .ok_or_else(|| EvalError::MissingBuffer(buffer.clone()))?[i] = v;
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ir::lower, parse::parse};

    fn saxpy() -> KernelIr {
        let src = "machine sm_120\n\nkernel saxpy(n: u32, a: f32, x: [f32], y: [f32])\n    stream x : dram -> reg\n    stream y : dram -> reg, drain\n    at reg:
        y = a * x + y\n";
        let u = parse(src).unwrap();
        lower(&u, &u.kernels[0]).unwrap()
    }

    #[test]
    fn saxpy_computes_saxpy() {
        let ir = saxpy();
        let mut inputs = Inputs::default();
        inputs.scalars.insert("a".into(), 2.0);
        inputs.buffers.insert("x".into(), vec![1.0, 2.0, 3.0]);
        inputs.buffers.insert("y".into(), vec![10.0, 20.0, 30.0]);
        let out = eval(&ir, 3, &inputs).unwrap();
        assert_eq!(out.buffers["y"], vec![12.0, 24.0, 36.0]);
        assert_eq!(out.buffers["x"], vec![1.0, 2.0, 3.0], "x is not drained");
    }

    #[test]
    fn the_reference_fuses_exactly_as_the_generated_code_does() {
        // A case where one rounding and two roundings differ. If this ever needs a tolerance
        // to pass, the host reference and the device have stopped agreeing on the operation.
        let ir = saxpy();
        let mut inputs = Inputs::default();
        let (a, x, y) = (1.0f32 + f32::EPSILON, 1.0f32 + f32::EPSILON, -1.0f32);
        inputs.scalars.insert("a".into(), a);
        inputs.buffers.insert("x".into(), vec![x]);
        inputs.buffers.insert("y".into(), vec![y]);
        let out = eval(&ir, 1, &inputs).unwrap();
        assert_eq!(out.buffers["y"][0], a.mul_add(x, y));
    }

    #[test]
    fn a_missing_input_is_named_rather_than_defaulted_to_zero() {
        let ir = saxpy();
        let mut inputs = Inputs::default();
        inputs.buffers.insert("x".into(), vec![1.0]);
        inputs.buffers.insert("y".into(), vec![1.0]);
        let e = eval(&ir, 1, &inputs).unwrap_err();
        assert!(e.to_string().contains("`a`"), "{e}");
    }

    #[test]
    fn a_buffer_shorter_than_the_element_count_is_refused() {
        let ir = saxpy();
        let mut inputs = Inputs::default();
        inputs.scalars.insert("a".into(), 1.0);
        inputs.buffers.insert("x".into(), vec![1.0, 2.0]);
        inputs.buffers.insert("y".into(), vec![1.0, 2.0]);
        let e = eval(&ir, 4, &inputs).unwrap_err();
        assert!(e.to_string().contains("expected at least 4"), "{e}");
    }
}
