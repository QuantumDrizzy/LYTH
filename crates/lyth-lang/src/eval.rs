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

use crate::ast::{BinOp, ReduceOp, Ty};
use crate::ir::{KernelIr, Op, RegId};

#[derive(Debug, thiserror::Error)]
pub enum EvalError {
    #[error("no buffer named `{0}` was supplied")]
    MissingBuffer(String),
    #[error("no scalar named `{0}` was supplied")]
    MissingScalar(String),
    #[error("no value for the extent `{0}`; a rank-2 kernel needs every extent it walks")]
    MissingExtent(String),
    #[error("a block of zero threads reduces nothing")]
    ZeroBlock,
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
    /// Values for the `u32` extents a rank-2 kernel walks. Empty at rank 1, where the element
    /// count is the only length there is and arrives as `n`.
    pub extents: BTreeMap<String, u32>,
}

/// Run every element of the kernel on the host, returning the drained buffers.
///
/// Element order is irrelevant: v1 has no cross-element communication, which is exactly why
/// one thread per element is a legal schedule in the first place.
pub fn eval(ir: &KernelIr, n: usize, inputs: &Inputs) -> Result<Inputs, EvalError> {
    eval_with_launch(ir, n, inputs, n.div_ceil(256).max(1), 256)
}

/// As [`eval`], with the launch shape the device will use.
///
/// The shape matters only for a reduction, and there it matters completely. Under grid-stride
/// a thread first folds its own elements in sequence and only then does the block tree run, so
/// the result depends on **both** the grid and the block. Float addition is not associative, so
/// the host has to walk that same two-stage order or the bit-exact check fails on a correct
/// kernel and the failure looks like a code-generation bug.
pub fn eval_with_launch(
    ir: &KernelIr,
    n: usize,
    inputs: &Inputs,
    grid: usize,
    block: usize,
) -> Result<Inputs, EvalError> {
    // Before the parameter check, not after: a contraction sizes each buffer by its own shape
    // -- `a` is `m x k` where `c` is `m x n` -- and the check below asks every buffer for `n`
    // elements, which is a rule from the time when one linear index addressed them all.
    if ir.contract.is_some() {
        return eval_contraction(ir, inputs);
    }

    for p in &ir.params {
        match p.ty {
            // Every buffer width validates the same way: this is about how many elements the
            // caller supplied, not how wide one is. The rounding that a narrow buffer implies
            // lives at the store, not here (ADR-0024 step 2).
            Ty::BufF32 | Ty::BufF16 | Ty::BufBF16 => {
                let b = inputs
                    .buffers
                    .get(&p.name)
                    .ok_or_else(|| EvalError::MissingBuffer(p.name.clone()))?;
                // A reduction target holds one value per block, not one per element. Demanding
                // `n` of it would force the caller to allocate `n` slots for `n / block` results.
                let want = match &ir.reduction {
                    Some(r) if r.into == p.name => grid,
                    _ => n,
                };
                if b.len() < want {
                    return Err(EvalError::ShortBuffer {
                        name: p.name.clone(),
                        got: b.len(),
                        want,
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

    if block == 0 || grid == 0 {
        return Err(EvalError::ZeroBlock);
    }
    let mut out = inputs.clone();

    // Per-element values the reduction will consume, in element order.
    let mut reduced: Vec<f32> = Vec::new();
    if ir.reduction.is_some() {
        reduced.reserve(n);
    }

    // Registers are dense small integers, so the environment is a Vec indexed by RegId,
    // allocated once and reused. It was a BTreeMap built fresh per element, which is one heap
    // allocation and a tree of comparisons for every element: 67 million of them made a sweep
    // over launch shapes take longer than every kernel in it put together.
    let n_regs = ir
        .ops
        .iter()
        .map(|o| o.dst() as usize + 1)
        .max()
        .unwrap_or(0);
    let mut regs: Vec<f32> = vec![0.0; n_regs];

    // Resolve buffer names to indices once, rather than hashing a String per element.
    let buffer_order: Vec<String> = out.buffers.keys().cloned().collect();
    let index_of = |name: &str| -> Option<usize> { buffer_order.iter().position(|b| b == name) };
    let mut columns: Vec<Vec<f32>> = buffer_order
        .iter()
        .map(|b| out.buffers[b].clone())
        .collect();
    let loads: Vec<(RegId, usize)> = ir
        .ops
        .iter()
        .filter_map(|op| match op {
            Op::Load { dst, buffer } => index_of(buffer).map(|i| (*dst, i)),
            _ => None,
        })
        .collect();
    let drain_cols: Vec<(usize, RegId)> = ir
        .drains
        .iter()
        .filter_map(|(b, r)| index_of(b).map(|i| (i, *r)))
        .collect();
    let _ = &loads;
    let _ = &drain_cols;

    // How each buffer's element index is computed from the linear one.
    //
    // At rank 1 they coincide. At rank 2 the linear index decomposes row-major -- the order
    // `lyth-ptx` emits as `div.u32` and `rem.u32` -- and each buffer applies its own
    // permutation and its own row length. A transpose reads one operand down a column, and
    // this is where the host reproduces that rather than approximating it.
    struct Walk {
        /// `None` at rank 1.
        rank2: Option<(bool, usize)>,
    }
    let space_cols = ir
        .space
        .as_ref()
        .map(|sp| {
            inputs
                .extents
                .get(&sp.extents[1])
                .copied()
                .ok_or_else(|| EvalError::MissingExtent(sp.extents[1].clone()))
                .map(|v| v as usize)
        })
        .transpose()?;
    let walks: Vec<Walk> = buffer_order
        .iter()
        .map(|name| {
            let Some(sp) = ir.space.as_ref() else {
                return Ok(Walk { rank2: None });
            };
            let Some(st) = ir.streams.iter().find(|s| s.buffer == *name) else {
                // A reduction target has no element address.
                return Ok(Walk { rank2: None });
            };
            let shape = &ir
                .params
                .iter()
                .find(|p| p.name == *name)
                .expect("a streamed buffer is a parameter")
                .shape;
            let row = inputs
                .extents
                .get(&shape[1])
                .copied()
                .ok_or_else(|| EvalError::MissingExtent(shape[1].clone()))? as usize;
            // `[j, i]` against `space i, j` means the outer index comes second.
            let swap = st.index[0] != sp.vars[0];
            Ok(Walk {
                rank2: Some((swap, row)),
            })
        })
        .collect::<Result<_, EvalError>>()?;

    let at = |col: usize, k: usize| -> usize {
        match walks[col].rank2 {
            None => k,
            Some((swap, row)) => {
                let cols = space_cols.expect("rank 2 resolved its space extent");
                let (i, j) = (k / cols, k % cols);
                let (outer, inner) = if swap { (j, i) } else { (i, j) };
                outer * row + inner
            }
        }
    };

    // clippy would rather this were an iterator, but the body indexes several columns at the
    // same position and writes back into one of them; an index is the honest expression of that.
    #[allow(clippy::needless_range_loop)]
    for i in 0..n {
        for op in &ir.ops {
            let v = match op {
                Op::Load { dst, buffer } => {
                    let col =
                        index_of(buffer).ok_or_else(|| EvalError::MissingBuffer(buffer.clone()))?;
                    let _ = dst;
                    columns[col][at(col, i)]
                }
                Op::Param { name, .. } => *out
                    .scalars
                    .get(name)
                    .ok_or_else(|| EvalError::MissingScalar(name.clone()))?,
                Op::Const { value, .. } => *value as f32,
                Op::Bin { op, lhs, rhs, .. } => {
                    let a = regs[*lhs as usize];
                    let b = regs[*rhs as usize];
                    match op {
                        BinOp::Add => a + b,
                        BinOp::Sub => a - b,
                        BinOp::Mul => a * b,
                        BinOp::Div => a / b,
                    }
                }
                // A true FMA: one rounding, matching `fma.rn.f32`. Writing `a * b + c` here
                // would round twice and disagree with the GPU on the last bit.
                Op::Fma { a, b, c, .. } => {
                    regs[*a as usize].mul_add(regs[*b as usize], regs[*c as usize])
                }
                Op::Neg { src, .. } => -regs[*src as usize],
            };
            regs[op.dst() as usize] = v;
        }
        // Every drain lands after the whole body, so a kernel that reads y and writes y sees
        // the old value throughout the element, exactly as the generated code does.
        for (col, reg) in &drain_cols {
            // The write is the same address as a read would be: a drained buffer at rank 2
            // is walked by its own permutation, which for a transpose is the whole point.
            columns[*col][at(*col, i)] = regs[*reg as usize];
        }
        if let Some(r) = &ir.reduction {
            reduced.push(regs[r.value as usize]);
        }
    }

    for (name, col) in buffer_order.iter().zip(columns) {
        out.buffers.insert(name.clone(), col);
    }

    if let Some(r) = &ir.reduction {
        let target = out
            .buffers
            .get_mut(&r.into)
            .ok_or_else(|| EvalError::MissingBuffer(r.into.clone()))?;
        if target.len() < grid {
            return Err(EvalError::ShortBuffer {
                name: r.into.clone(),
                got: target.len(),
                want: grid,
            });
        }
        // Two stages, in this order, because that is what the generated loop does:
        //   1. each thread folds its own strided elements in sequence
        //   2. the block's threads are combined by the tree
        // Swapping them, or folding a block's elements contiguously, gives a different
        // answer -- not a different last bit. See ADR-0011.
        let stride = grid * block;
        for (b, slot) in target.iter_mut().enumerate().take(grid) {
            let mut slots: Vec<f32> = Vec::with_capacity(block);
            for t in 0..block {
                let mut acc = r.op.identity();
                let mut i = b * block + t;
                while i < n {
                    acc = r.op.combine(acc, reduced[i]);
                    i += stride;
                }
                slots.push(acc);
            }
            *slot = tree_reduce(&mut slots, r.op);
        }
    }
    round_narrow_outputs(ir, &mut out);
    Ok(out)
}


/// The host oracle for a contraction (ADR-0018 step 3).
///
/// A separate walk rather than a flag on the main one, because the main one is built around a
/// rule the contraction breaks: every buffer's index is a permutation of the space variables,
/// so one linear element index addresses them all. `a[i, p]` names an axis the space does not
/// iterate, and there is no linear index that reaches it.
///
/// **The accumulation order is the contract, not an implementation detail.** The device walks
/// `p` from 0 upwards -- an outer loop over tile steps and an inner loop over the terms within
/// a step, which is ascending `p` overall -- and float addition is not associative, so this
/// walks the same order. It is also why the emitter does not fuse the multiply and the add
/// into an `fma`: one rounding against two is a different answer, and the two would have to
/// change together.
fn eval_contraction(ir: &KernelIr, inputs: &Inputs) -> Result<Inputs, EvalError> {
    let c = ir.contract.as_ref().expect("dispatched on a contraction");
    let sp = ir
        .space
        .as_ref()
        .ok_or_else(|| EvalError::MissingExtent("space".into()))?;
    let extent = |name: &str| -> Result<usize, EvalError> {
        inputs
            .extents
            .get(name)
            .copied()
            .map(|v| v as usize)
            .ok_or_else(|| EvalError::MissingExtent(name.to_string()))
    };
    let rows = extent(&sp.extents[0])?;
    let cols = extent(&sp.extents[1])?;
    let depth = extent(&c.extent)?;

    let mut out = inputs.clone();
    let n_regs = ir
        .ops
        .iter()
        .map(|o| o.dst() as usize + 1)
        .max()
        .unwrap_or(0);
    let mut regs: Vec<f32> = vec![0.0; n_regs];

    // Each streamed buffer, resolved once: its two index variables and its own row length.
    struct Addr {
        buffer: String,
        outer: String,
        inner: String,
        row: usize,
    }
    let mut addrs: Vec<Addr> = Vec::new();
    for st in &ir.streams {
        let shape = &ir
            .params
            .iter()
            .find(|p| p.name == st.buffer)
            .expect("a streamed buffer is a parameter")
            .shape;
        addrs.push(Addr {
            buffer: st.buffer.clone(),
            outer: st.index[0].clone(),
            inner: st.index[1].clone(),
            row: extent(&shape[1])?,
        });
    }

    let (drain_name, drain_reg) = ir
        .drains
        .first()
        .cloned()
        .ok_or_else(|| EvalError::MissingBuffer("a contraction drains one buffer".into()))?;
    for a in &addrs {
        let b = inputs
            .buffers
            .get(&a.buffer)
            .ok_or_else(|| EvalError::MissingBuffer(a.buffer.clone()))?;
        let want = match (a.outer.as_str(), a.inner.as_str()) {
            _ if a.buffer == drain_name => rows * cols,
            _ => {
                // Whichever two of {rows, cols, depth} this buffer is indexed by.
                let size = |v: &str| -> usize {
                    if v == sp.vars[0] {
                        rows
                    } else if v == sp.vars[1] {
                        cols
                    } else {
                        depth
                    }
                };
                size(&a.outer) * size(&a.inner)
            }
        };
        if b.len() < want {
            return Err(EvalError::ShortBuffer {
                name: a.buffer.clone(),
                got: b.len(),
                want,
            });
        }
    }

    let mut result = vec![0.0f32; rows * cols];
    for i in 0..rows {
        for j in 0..cols {
            let mut acc = c.op.identity();
            for p in 0..depth {
                let coord = |v: &str| -> usize {
                    if v == sp.vars[0] {
                        i
                    } else if v == sp.vars[1] {
                        j
                    } else {
                        p
                    }
                };
                for op in &ir.ops {
                    let v = match op {
                        Op::Load { buffer, .. } => {
                            let a = addrs
                                .iter()
                                .find(|a| a.buffer == *buffer)
                                .ok_or_else(|| EvalError::MissingBuffer(buffer.clone()))?;
                            let col = inputs
                                .buffers
                                .get(buffer)
                                .ok_or_else(|| EvalError::MissingBuffer(buffer.clone()))?;
                            col[coord(&a.outer) * a.row + coord(&a.inner)]
                        }
                        Op::Param { name, .. } => *inputs
                            .scalars
                            .get(name)
                            .ok_or_else(|| EvalError::MissingScalar(name.clone()))?,
                        Op::Const { value, .. } => *value as f32,
                        Op::Bin { op, lhs, rhs, .. } => {
                            let a = regs[*lhs as usize];
                            let b = regs[*rhs as usize];
                            match op {
                                BinOp::Add => a + b,
                                BinOp::Sub => a - b,
                                BinOp::Mul => a * b,
                                BinOp::Div => a / b,
                            }
                        }
                        Op::Fma { a, b, c, .. } => {
                            regs[*a as usize].mul_add(regs[*b as usize], regs[*c as usize])
                        }
                        Op::Neg { src, .. } => -regs[*src as usize],
                    };
                    regs[op.dst() as usize] = v;
                }
                acc = c.op.combine(acc, regs[drain_reg as usize]);
            }
            result[i * cols + j] = acc;
        }
    }

    let drain_row = addrs
        .iter()
        .find(|a| a.buffer == drain_name)
        .map(|a| a.row)
        .unwrap_or(cols);
    let target = out
        .buffers
        .get_mut(&drain_name)
        .ok_or_else(|| EvalError::MissingBuffer(drain_name.clone()))?;
    for i in 0..rows {
        for j in 0..cols {
            target[i * drain_row + j] = result[i * cols + j];
        }
    }
    round_narrow_outputs(ir, &mut out);
    Ok(out)
}

/// The same tree the generated PTX walks: halve the stride, combine `slots[t]` with
/// `slots[t + stride]`, repeat.
///
/// For `sum` the order is the whole point, not an implementation detail: floating-point
/// addition is not associative, so a different tree gives a different bit pattern. For `max`
/// and `min` the order cannot matter -- they select an operand and never round -- and this
/// function reproduces the tree anyway. The evaluator is the oracle for what the GPU does,
/// and an oracle that agrees for a reason the operator happens to supply is one operator away
/// from disagreeing silently. See ADR-0013.
fn tree_reduce(slots: &mut [f32], op: ReduceOp) -> f32 {
    let mut stride = slots.len() / 2;
    while stride > 0 {
        for t in 0..stride {
            slots[t] = op.combine(slots[t], slots[t + stride]);
        }
        stride /= 2;
    }
    slots.first().copied().unwrap_or(op.identity())
}

/// Round every narrow buffer to the width the device will store it at.
///
/// One pass at the end rather than a rounding at each write site, because there are several
/// write sites -- the elementwise drains, the reduction target, the contraction's own path --
/// and a rounding that lives in three places is a rounding that will eventually live in two.
///
/// `crates/lyth-lang/src/half.rs` is the implementation, and it is the one checked bit for bit
/// against `cvt.rn.f16.f32` on this device (ADR-0024 decision 3). The oracle rounds where the
/// device rounds -- at the store, never at the arithmetic, which stays f32 throughout.
fn round_narrow_outputs(ir: &KernelIr, out: &mut Inputs) {
    for p in &ir.params {
        let f: fn(f32) -> f32 = match p.ty {
            Ty::BufF16 => crate::half::round_f16,
            Ty::BufBF16 => crate::half::round_bf16,
            _ => continue,
        };
        if let Some(b) = out.buffers.get_mut(&p.name) {
            for v in b.iter_mut() {
                *v = f(*v);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ir::lower, parse::parse};

    fn saxpy() -> KernelIr {
        let src = "machine sm_120\n\nkernel saxpy(n: u32, a: f32, x: [f32; n], y: [f32; n])\n    stream x : dram -> reg\n    stream y : dram -> reg, drain\n    at reg:
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

    fn dot_ir() -> KernelIr {
        let src = "machine sm_120

kernel dot(n: u32, x: [f32; n], y: [f32; n], partial: [f32; blocks])
    stream x : dram -> reg
    stream y : dram -> reg
    reduce sum p : reg -> smem -> dram into partial
    at reg:
        p = x * y
";
        let u = parse(src).unwrap();
        lower(&u, &u.kernels[0]).unwrap()
    }

    #[test]
    fn the_tree_actually_computes_the_dot_product() {
        // Exact powers of two, so no rounding can hide a wrong sum behind a plausible one.
        let ir = dot_ir();
        let mut inputs = Inputs::default();
        let x: Vec<f32> = (0..8).map(|i| (1 << i) as f32).collect();
        let y: Vec<f32> = vec![1.0; 8];
        let want: f32 = x.iter().zip(&y).map(|(a, b)| a * b).sum();
        inputs.buffers.insert("x".into(), x);
        inputs.buffers.insert("y".into(), y);
        inputs.buffers.insert("partial".into(), vec![0.0; 1]);
        let out = eval_with_launch(&ir, 8, &inputs, 1, 8).unwrap();
        assert_eq!(out.buffers["partial"][0], want, "1+2+...+128 = 255");
        assert_eq!(want, 255.0);
    }

    #[test]
    fn a_ragged_tail_contributes_the_identity_and_nothing_else() {
        // 10 elements in blocks of 4: three blocks, the last one only half full. Threads with
        // no element must add zero, not whatever the slot held.
        let ir = dot_ir();
        let mut inputs = Inputs::default();
        inputs.buffers.insert("x".into(), vec![1.0; 10]);
        inputs.buffers.insert("y".into(), vec![1.0; 10]);
        inputs.buffers.insert("partial".into(), vec![-99.0; 3]);
        let out = eval_with_launch(&ir, 10, &inputs, 3, 4).unwrap();
        let p = &out.buffers["partial"];
        assert_eq!(p[0], 4.0);
        assert_eq!(p[1], 4.0);
        assert_eq!(p[2], 2.0, "the last block holds two elements, not four");
        assert_eq!(p.iter().take(3).sum::<f32>(), 10.0);
    }

    #[test]
    fn the_tree_order_is_the_one_the_device_walks_not_a_left_fold() {
        // Values chosen so the two orders disagree in the last bit. If this ever passes with
        // a sequential sum, the host reference has stopped matching the generated tree and
        // the bit-exact check has quietly become a tolerance.
        let ir = dot_ir();
        // The tree pairs (x0,x2) and (x1,x3); a sequential sum folds left to right. Here the
        // two give 2.0 and 1.0: the large pair cancels first in the tree, so both ones survive,
        // while the left fold loses one of them into 1e8 before the cancellation happens.
        let big = 1.0e8f32;
        let x = vec![big, 1.0, -big, 1.0];
        let mut inputs = Inputs::default();
        inputs.buffers.insert("x".into(), x.clone());
        inputs.buffers.insert("y".into(), vec![1.0; 4]);
        inputs.buffers.insert("partial".into(), vec![0.0; 1]);
        let out = eval_with_launch(&ir, 4, &inputs, 1, 4).unwrap();

        let tree = (x[0] + x[2]) + (x[1] + x[3]);
        let left_fold = ((x[0] + x[1]) + x[2]) + x[3];
        assert_ne!(
            tree, left_fold,
            "the test values must distinguish the orders"
        );
        assert_eq!(tree, 2.0);
        assert_eq!(left_fold, 1.0);
        assert_eq!(out.buffers["partial"][0], tree);
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
