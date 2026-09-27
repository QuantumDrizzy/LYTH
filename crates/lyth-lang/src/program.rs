//! `main:` resolved against the kernel it names. ADR-0025 step 4.
//!
//! The parser accepts `run saxpy(n = 4096, a = 2.0)` without knowing whether `saxpy` exists,
//! whether it takes an `a`, or whether 4096 is a legal element count. This is where those
//! become answers, and every one of them is a refusal rather than a default: a `main` that
//! quietly supplies a missing argument is a file that no longer says what it runs.
//!
//! The result is what a back end needs to emit a program and what the host oracle needs to
//! evaluate the same thing — deliberately one structure, because the entire claim of this
//! target is that the two agree bit for bit, and two descriptions of the same launch is the
//! shape of defect ADR-0019 catalogued.

use std::collections::BTreeMap;

use crate::ast::{Main, Ty, Unit, BLOCKS};
use crate::ir::KernelIr;
use crate::lex::Span;

/// A `main` that has been checked against its kernel.
#[derive(Debug, Clone, PartialEq)]
pub struct Program {
    /// Elements the kernel walks: the bound at rank 1, the product of the extents at rank 2.
    pub n: u32,
    /// Every `u32` extent by name, which a rank-2 oracle needs to decompose a linear index.
    pub extents: BTreeMap<String, u32>,
    pub scalars: BTreeMap<String, f32>,
    pub prints: Vec<PrintRange>,
    /// When a name is present, those f32s are the buffer. Absent names use `lyth_lang::inputs`,
    /// which is what every existing program does. The host oracle must be given the same bytes.
    pub buffers: BTreeMap<String, Vec<f32>>,
}

/// Half-open, and resolved: `print y` has become `y[0:n]`.
#[derive(Debug, Clone, PartialEq)]
pub struct PrintRange {
    pub buffer: String,
    pub lo: u32,
    pub hi: u32,
}

#[derive(Debug, thiserror::Error, PartialEq)]
pub enum ProgramError {
    #[error("{span}: `main` runs `{name}`, and this file defines no kernel called that. It defines: {known}")]
    NoSuchKernel {
        span: Span,
        name: String,
        known: String,
    },
    #[error("{span}: `{name}` is not a parameter of kernel `{kernel}`")]
    NoSuchParam {
        span: Span,
        name: String,
        kernel: String,
    },
    #[error("{span}: `{name}` is a buffer. Buffers are not passed in: this language has no way to read data, so their contents are the deterministic generator in `lyth_lang::inputs` — the same one the oracle uses")]
    BufferArgument { span: Span, name: String },
    #[error("{span}: `run {kernel}` gives no value for `{name}`. Every u32 and f32 parameter needs one; a default here would be a number the file does not contain")]
    MissingArgument {
        span: Span,
        kernel: String,
        name: String,
    },
    #[error("{span}: `{name} = {value}` appears twice")]
    RepeatedArgument {
        span: Span,
        name: String,
        value: f64,
    },
    #[error("{span}: `{name} = {value}` is a u32 extent and {value} is not a whole number at least 1")]
    NotAnExtent {
        span: Span,
        name: String,
        value: f64,
    },
    #[error("{span}: `print {name}` names no buffer of kernel `{kernel}`. Only a buffer leaves the machine — a scalar parameter is already in the file")]
    PrintNotABuffer {
        span: Span,
        name: String,
        kernel: String,
    },
    #[error("{span}: `print {name}[{lo}:{hi}]` runs past its {len} elements")]
    PrintOutOfRange {
        span: Span,
        name: String,
        lo: u32,
        hi: u32,
        len: u32,
    },
}

/// Check a `main` against the kernel it names.
pub fn resolve(unit: &Unit, main: &Main, ir: &KernelIr) -> Result<Program, ProgramError> {
    if main.kernel != ir.name {
        return Err(ProgramError::NoSuchKernel {
            span: main.kernel_span,
            name: main.kernel.clone(),
            known: unit
                .kernels
                .iter()
                .map(|k| k.name.as_str())
                .collect::<Vec<_>>()
                .join(", "),
        });
    }

    let mut extents: BTreeMap<String, u32> = BTreeMap::new();
    let mut scalars: BTreeMap<String, f32> = BTreeMap::new();
    for (name, value, span) in &main.args {
        let Some(p) = ir.params.iter().find(|p| &p.name == name) else {
            return Err(ProgramError::NoSuchParam {
                span: *span,
                name: name.clone(),
                kernel: ir.name.clone(),
            });
        };
        let repeated = match p.ty {
            Ty::U32 => {
                // An extent is a count, and a count is a whole number. `n = 4096.5` is a
                // question about rounding that the file should answer, not the compiler.
                if *value < 1.0 || value.fract() != 0.0 {
                    return Err(ProgramError::NotAnExtent {
                        span: *span,
                        name: name.clone(),
                        value: *value,
                    });
                }
                extents.insert(name.clone(), *value as u32).is_some()
            }
            Ty::F32 => scalars.insert(name.clone(), *value as f32).is_some(),
            _ => {
                return Err(ProgramError::BufferArgument {
                    span: *span,
                    name: name.clone(),
                })
            }
        };
        if repeated {
            return Err(ProgramError::RepeatedArgument {
                span: *span,
                name: name.clone(),
                value: *value,
            });
        }
    }

    for p in &ir.params {
        let given = match p.ty {
            Ty::U32 => extents.contains_key(&p.name),
            Ty::F32 => scalars.contains_key(&p.name),
            _ => continue,
        };
        if !given {
            return Err(ProgramError::MissingArgument {
                span: main.span,
                kernel: ir.name.clone(),
                name: p.name.clone(),
            });
        }
    }

    let n = element_count(ir, &extents);

    let mut prints = Vec::new();
    for pr in &main.prints {
        let Some(p) = ir
            .params
            .iter()
            .find(|p| p.name == pr.buffer && p.ty.is_buffer())
        else {
            return Err(ProgramError::PrintNotABuffer {
                span: pr.span,
                name: pr.buffer.clone(),
                kernel: ir.name.clone(),
            });
        };
        // A reduction target is sized by the launch, not by the problem: `[f32; blocks]`. One
        // block here, because the block is the register (ADR-0025 step 3), so it holds one
        // value and `print partial` is one element rather than `n` of them.
        let len = if p.shape.iter().any(|d| d == BLOCKS) {
            1
        } else {
            n
        };
        let (lo, hi) = pr.range.unwrap_or((0, len));
        if hi > len {
            return Err(ProgramError::PrintOutOfRange {
                span: pr.span,
                name: pr.buffer.clone(),
                lo,
                hi,
                len,
            });
        }
        prints.push(PrintRange {
            buffer: pr.buffer.clone(),
            lo,
            hi,
        });
    }

    Ok(Program {
        n,
        extents,
        scalars,
        prints,
        buffers: BTreeMap::new(),
    })
}

/// How many elements the kernel walks.
///
/// At rank 2 it is the product of the extents, because the space is walked row-major and a
/// linear index covers it. At rank 1 there is no `space` clause and the bound is whatever
/// extent the buffers are declared with — the same `n` the kernel signature already names.
fn element_count(ir: &KernelIr, extents: &BTreeMap<String, u32>) -> u32 {
    if let Some(space) = &ir.space {
        return space
            .extents
            .iter()
            .filter_map(|e| extents.get(e))
            .product();
    }
    ir.params
        .iter()
        .filter(|p| p.ty.is_buffer())
        .flat_map(|p| p.shape.iter())
        .filter(|d| *d != BLOCKS)
        .filter_map(|d| extents.get(d))
        .copied()
        .next()
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parse;

    const SRC: &str = "machine unibit\n\n\
         kernel saxpy(n: u32, a: f32, x: [f32; n], y: [f32; n])\n    \
         intensity 0.1667\n    \
         stream x : dram -> reg\n    \
         stream y : dram -> reg, drain\n    \
         at reg:\n        \
         y = a * x + y\n\n\
         main:\n    \
         run saxpy(n = 4096, a = 2.0)\n    \
         print y[0:8]\n";

    fn go(src: &str) -> Result<Program, ProgramError> {
        let unit = parse(src).expect("parses");
        let ir = crate::ir::lower(&unit, &unit.kernels[0]).expect("lowers");
        resolve(&unit, unit.main.as_ref().expect("has a main"), &ir)
    }

    #[test]
    fn a_main_resolves_to_a_launch() {
        let p = go(SRC).expect("resolves");
        assert_eq!(p.n, 4096);
        assert_eq!(p.scalars["a"], 2.0);
        assert_eq!(p.extents["n"], 4096);
        assert_eq!(
            p.prints,
            vec![PrintRange {
                buffer: "y".into(),
                lo: 0,
                hi: 8
            }]
        );
    }

    #[test]
    fn print_with_no_range_is_the_whole_buffer() {
        let p = go(&SRC.replace("print y[0:8]", "print y")).expect("resolves");
        assert_eq!(p.prints[0].hi, 4096);
    }

    #[test]
    fn a_missing_scalar_is_refused_rather_than_defaulted() {
        // The one that matters most. A `main` that says nothing about `a` and a compiler that
        // supplies 2.0 produce a program whose output no reader of the file can predict --
        // and the oracle would have to guess the same number to agree with it.
        let e = go(&SRC.replace("(n = 4096, a = 2.0)", "(n = 4096)")).unwrap_err();
        assert!(matches!(e, ProgramError::MissingArgument { ref name, .. } if name == "a"), "{e}");
        assert!(e.to_string().contains("would be a number the file does not contain"));
    }

    #[test]
    fn an_argument_the_kernel_does_not_take_is_refused() {
        let e = go(&SRC.replace("a = 2.0", "a = 2.0, b = 1.0")).unwrap_err();
        assert!(matches!(e, ProgramError::NoSuchParam { ref name, .. } if name == "b"), "{e}");
    }

    #[test]
    fn a_buffer_cannot_be_passed_in() {
        // Because there is nothing to pass. This language has no way to read data, so a
        // buffer's contents are the deterministic generator and saying otherwise in the file
        // would be a promise the machine cannot keep.
        let e = go(&SRC.replace("a = 2.0", "a = 2.0, x = 1.0")).unwrap_err();
        assert!(matches!(e, ProgramError::BufferArgument { .. }), "{e}");
        assert!(e.to_string().contains("no way to read data"));
    }

    #[test]
    fn an_extent_that_is_not_a_count_is_refused() {
        for bad in ["n = 4096.5", "n = 0"] {
            let e = go(&SRC.replace("n = 4096", bad)).unwrap_err();
            assert!(matches!(e, ProgramError::NotAnExtent { .. }), "{bad}: {e}");
        }
    }

    #[test]
    fn printing_past_the_end_is_refused() {
        let e = go(&SRC.replace("print y[0:8]", "print y[4090:4100]")).unwrap_err();
        assert!(matches!(e, ProgramError::PrintOutOfRange { len: 4096, .. }), "{e}");
    }

    #[test]
    fn printing_something_that_is_not_a_buffer_is_refused() {
        // A scalar is already in the file; printing it would tell the reader nothing the
        // source does not say two lines above.
        let e = go(&SRC.replace("print y[0:8]", "print a")).unwrap_err();
        assert!(matches!(e, ProgramError::PrintNotABuffer { .. }), "{e}");
    }

    #[test]
    fn a_reduction_target_is_one_element_long_not_n() {
        // Sized by the launch, not by the problem: `[f32; blocks]`, and the block here is the
        // register, so there is one of them. `print partial` is one value.
        let src = "machine unibit\n\n\
             kernel sum(n: u32, x: [f32; n], partial: [f32; blocks])\n    \
             intensity 0.25\n    \
             stream x : dram -> reg\n    \
             reduce sum v : reg -> dram into partial\n    \
             at reg:\n        \
             v = x\n\n\
             main:\n    \
             run sum(n = 4096)\n    \
             print partial\n";
        let p = go(src).expect("resolves");
        assert_eq!(p.prints[0], PrintRange { buffer: "partial".into(), lo: 0, hi: 1 });

        let e = go(&src.replace("print partial", "print partial[0:8]")).unwrap_err();
        assert!(matches!(e, ProgramError::PrintOutOfRange { len: 1, .. }), "{e}");
    }

    #[test]
    fn a_rank_two_space_walks_the_product_of_its_extents() {
        let src = "machine unibit\n\n\
             kernel add2(rows: u32, cols: u32, a: [f32; rows, cols], b: [f32; rows, cols], \
             c: [f32; rows, cols])\n    \
             space i, j : rows, cols\n    \
             intensity 0.0833\n    \
             stream a : dram -> reg\n    \
             stream b : dram -> reg\n    \
             stream c : dram -> reg, drain\n    \
             at reg:\n        \
             c[i, j] = a[i, j] + b[i, j]\n\n\
             main:\n    \
             run add2(rows = 16, cols = 32)\n    \
             print c[0:4]\n";
        let p = go(src).expect("resolves");
        assert_eq!(p.n, 512);
    }
}
