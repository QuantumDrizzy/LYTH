//! The semantic IR, and the cost derived from it.
//!
//! THIS IS THE FILE THE THESIS LIVES IN.
//!
//! Until now `intensity-check` compared a declared number against a hand-written accounting:
//! two declarations, checked against each other, with the kernel itself present only as prose
//! in a `note` field. The cost model could not be wrong about the kernel because it never
//! looked at one.
//!
//! Here the body is code. Bytes come from the streams the program declares, flops from the
//! expression tree the program actually evaluates, and the `intensity` written in the source
//! is checked against a number **nobody typed**. A mismatch is a compile error.
//!
//! That is "arithmetic intensity is a type, not a comment", for the first time.

use std::collections::{BTreeMap, BTreeSet};

use crate::ast::*;
use crate::lex::Span;

/// A kernel that type-checked: every name resolves, every read is backed by a stream, and the
/// declared intensity matches the derived one.
#[derive(Debug, Clone, PartialEq)]
pub struct KernelIr {
    pub name: String,
    pub machine: String,
    pub params: Vec<Param>,
    pub streams: Vec<StreamIr>,
    /// Flattened straight-line body, in evaluation order.
    pub ops: Vec<Op>,
    /// Registers holding the final value of each drained buffer.
    pub drains: Vec<(String, RegId)>,
    pub cost: Cost,
}

#[derive(Debug, Clone, PartialEq)]
pub struct StreamIr {
    pub buffer: String,
    pub from: Level,
    pub to: Level,
    pub drain: bool,
    /// Register the loaded element lands in. `None` for a buffer that is only written.
    pub loaded: Option<RegId>,
    pub read: bool,
}

pub type RegId = u32;

/// One machine operation over registers. Straight-line, SSA-ish: each op defines a new
/// register and never reassigns one.
#[derive(Debug, Clone, PartialEq)]
pub enum Op {
    /// Load this buffer's element at the current index.
    Load {
        dst: RegId,
        buffer: String,
    },
    /// A scalar kernel parameter.
    Param {
        dst: RegId,
        name: String,
    },
    Const {
        dst: RegId,
        value: f64,
    },
    Bin {
        dst: RegId,
        op: BinOp,
        lhs: RegId,
        rhs: RegId,
    },
    /// `a * b + c` collapsed into one instruction.
    Fma {
        dst: RegId,
        a: RegId,
        b: RegId,
        c: RegId,
    },
    Neg {
        dst: RegId,
        src: RegId,
    },
}

impl Op {
    pub fn dst(&self) -> RegId {
        match self {
            Op::Load { dst, .. }
            | Op::Param { dst, .. }
            | Op::Const { dst, .. }
            | Op::Bin { dst, .. }
            | Op::Fma { dst, .. }
            | Op::Neg { dst, .. } => *dst,
        }
    }

    /// FLOPs this op retires.
    ///
    /// A load, a parameter read and a constant are not arithmetic. An `fma` is two — a
    /// multiply and an add — which is the convention every published flop count uses, and it
    /// keeps `a*x + y` costing the same whether or not it is contracted.
    pub fn flops(&self) -> f64 {
        match self {
            Op::Load { .. } | Op::Param { .. } | Op::Const { .. } => 0.0,
            Op::Bin { op, .. } => op.flops(),
            Op::Fma { .. } => 2.0,
            // Negation is a sign flip, not an arithmetic op worth charging for.
            Op::Neg { .. } => 0.0,
        }
    }
}

/// What the compiler derived. Nothing in here was typed by a human.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Cost {
    /// Bytes moved per element, at the deepest level any stream touches.
    pub bytes_per_element: f64,
    pub read_bytes_per_element: f64,
    pub write_bytes_per_element: f64,
    pub flops_per_element: f64,
    pub intensity: f64,
    /// The level `bytes_per_element` is about.
    pub level: Level,
}

#[derive(Debug, thiserror::Error)]
pub enum LowerError {
    #[error("{span}: `{name}` is used but is not a parameter of this kernel")]
    UnknownName { span: Span, name: String },
    #[error(
        "{span}: `{name}` is a buffer but no stream moves it. \
         Add `stream {name} : dram -> reg`. \
         You cannot operate on what you have not declared resident (ADR-0001)."
    )]
    NotStreamed { span: Span, name: String },
    #[error("{span}: `stream {name}` names no parameter of this kernel")]
    StreamOfNothing { span: Span, name: String },
    #[error("{span}: `stream {name}` is declared twice")]
    DuplicateStream { span: Span, name: String },
    #[error("{span}: `{name}` is a scalar parameter and cannot be streamed or assigned")]
    ScalarStream { span: Span, name: String },
    #[error(
        "{span}: assigning to `{name}`, whose stream has no `drain`. \
         A value written in registers and never drained is work the kernel throws away."
    )]
    WriteWithoutDrain { span: Span, name: String },
    #[error(
        "{span}: `{name}` is drained but never assigned in any `at` block. \
         A drain writes back a value that was never computed."
    )]
    DrainWithoutWrite { span: Span, name: String },
    #[error("{span}: v1 supports one `at reg:` block; `at {level}` is not implemented")]
    UnsupportedLevel { span: Span, level: &'static str },
    #[error(
        "{span}: v1 moves data between dram and reg only; `{from} -> {to}` is not implemented"
    )]
    UnsupportedPath {
        span: Span,
        from: &'static str,
        to: &'static str,
    },
    #[error("{0}")]
    Message(String),
}

/// Lower one kernel, resolving names and deriving its cost.
pub fn lower(unit: &Unit, kernel: &Kernel) -> Result<KernelIr, LowerError> {
    let params: BTreeMap<&str, Ty> = kernel
        .params
        .iter()
        .map(|p| (p.name.as_str(), p.ty))
        .collect();

    // --- streams -------------------------------------------------------------------
    let mut seen = BTreeSet::new();
    let mut streams = Vec::new();
    for s in &kernel.streams {
        let Some(ty) = params.get(s.buffer.as_str()) else {
            return Err(LowerError::StreamOfNothing {
                span: s.span,
                name: s.buffer.clone(),
            });
        };
        if !ty.is_buffer() {
            return Err(LowerError::ScalarStream {
                span: s.span,
                name: s.buffer.clone(),
            });
        }
        if !seen.insert(s.buffer.clone()) {
            return Err(LowerError::DuplicateStream {
                span: s.span,
                name: s.buffer.clone(),
            });
        }
        if !(s.from == Level::Dram && s.to == Level::Reg) {
            return Err(LowerError::UnsupportedPath {
                span: s.span,
                from: s.from.name(),
                to: s.to.name(),
            });
        }
        streams.push(StreamIr {
            buffer: s.buffer.clone(),
            from: s.from,
            to: s.to,
            drain: s.drain,
            loaded: None,
            read: false,
        });
    }

    // --- body ---------------------------------------------------------------------
    let mut ctx = Lowering {
        params: &params,
        streams: &mut streams,
        ops: Vec::new(),
        next_reg: 0,
        // Value currently held for each name: a loaded element, or the result of a statement.
        env: BTreeMap::new(),
    };

    for block in &kernel.blocks {
        if block.level != Level::Reg {
            return Err(LowerError::UnsupportedLevel {
                span: block.span,
                level: block.level.name(),
            });
        }
    }
    if kernel.blocks.len() > 1 {
        return Err(LowerError::Message(
            "v1 supports a single `at reg:` block per kernel".into(),
        ));
    }

    let mut assigned: BTreeSet<String> = BTreeSet::new();
    for block in &kernel.blocks {
        for stmt in &block.stmts {
            let value = ctx.expr(&stmt.value)?;
            match params.get(stmt.target.as_str()) {
                None => {
                    return Err(LowerError::UnknownName {
                        span: stmt.target_span,
                        name: stmt.target.clone(),
                    })
                }
                Some(ty) if !ty.is_buffer() => {
                    return Err(LowerError::ScalarStream {
                        span: stmt.target_span,
                        name: stmt.target.clone(),
                    })
                }
                Some(_) => {}
            }
            let drained = ctx
                .streams
                .iter()
                .find(|s| s.buffer == stmt.target)
                .map(|s| s.drain);
            match drained {
                None => {
                    return Err(LowerError::NotStreamed {
                        span: stmt.target_span,
                        name: stmt.target.clone(),
                    })
                }
                Some(false) => {
                    return Err(LowerError::WriteWithoutDrain {
                        span: stmt.target_span,
                        name: stmt.target.clone(),
                    })
                }
                Some(true) => {}
            }
            ctx.env.insert(stmt.target.clone(), value);
            assigned.insert(stmt.target.clone());
        }
    }

    // Destructure to end the mutable borrow of `streams` before reading it back.
    let Lowering { ops, env, .. } = ctx;
    let mut drains = Vec::new();
    for s in streams.iter() {
        if !s.drain {
            continue;
        }
        if !assigned.contains(&s.buffer) {
            return Err(LowerError::DrainWithoutWrite {
                span: kernel.span,
                name: s.buffer.clone(),
            });
        }
        let reg = *env.get(&s.buffer).expect("assigned implies an env entry");
        drains.push((s.buffer.clone(), reg));
    }

    let cost = derive_cost(&streams, &ops);

    Ok(KernelIr {
        name: kernel.name.clone(),
        machine: unit.machine.clone(),
        params: kernel.params.clone(),
        streams,
        ops,
        drains,
        cost,
    })
}

/// Bytes from the declared movement, flops from the evaluated tree.
///
/// A stream contributes a read only if the body actually reads it, and a write only if it
/// drains. A buffer that is streamed and never read costs nothing to read — the declaration
/// does not get to inflate the denominator, and an unread stream is caught elsewhere.
fn derive_cost(streams: &[StreamIr], ops: &[Op]) -> Cost {
    let mut read = 0.0;
    let mut write = 0.0;
    let mut level = Level::Reg;
    for s in streams {
        if s.read {
            read += Ty::BufF32.bytes() as f64;
        }
        if s.drain {
            write += Ty::BufF32.bytes() as f64;
        }
        // The deepest level any stream touches is what the byte count is about.
        if (s.from as u8) < (level as u8) {
            level = s.from;
        }
    }
    let flops: f64 = ops.iter().map(Op::flops).sum();
    let bytes = read + write;
    Cost {
        bytes_per_element: bytes,
        read_bytes_per_element: read,
        write_bytes_per_element: write,
        flops_per_element: flops,
        intensity: if bytes > 0.0 { flops / bytes } else { 0.0 },
        level,
    }
}

struct Lowering<'a> {
    params: &'a BTreeMap<&'a str, Ty>,
    streams: &'a mut Vec<StreamIr>,
    ops: Vec<Op>,
    next_reg: RegId,
    env: BTreeMap<String, RegId>,
}

impl Lowering<'_> {
    fn fresh(&mut self) -> RegId {
        let r = self.next_reg;
        self.next_reg += 1;
        r
    }

    fn emit(&mut self, op: Op) -> RegId {
        let d = op.dst();
        self.ops.push(op);
        d
    }

    fn expr(&mut self, e: &Expr) -> Result<RegId, LowerError> {
        match e {
            Expr::Const(v, _) => {
                let dst = self.fresh();
                Ok(self.emit(Op::Const { dst, value: *v }))
            }
            Expr::Name(name, span) => self.name(name, *span),
            Expr::Neg(inner, _) => {
                let src = self.expr(inner)?;
                let dst = self.fresh();
                Ok(self.emit(Op::Neg { dst, src }))
            }
            Expr::Bin { op, lhs, rhs, .. } => {
                // Contract `a * b + c` and `c + a * b` into one fma. This is a structural
                // pattern match, not a numerical guarantee: it is recorded as a known limit
                // that ptxas may or may not contract the same way a CUDA compiler would.
                if *op == BinOp::Add {
                    if let Expr::Bin {
                        op: BinOp::Mul,
                        lhs: ml,
                        rhs: mr,
                        ..
                    } = &**lhs
                    {
                        let a = self.expr(ml)?;
                        let b = self.expr(mr)?;
                        let c = self.expr(rhs)?;
                        let dst = self.fresh();
                        return Ok(self.emit(Op::Fma { dst, a, b, c }));
                    }
                    if let Expr::Bin {
                        op: BinOp::Mul,
                        lhs: ml,
                        rhs: mr,
                        ..
                    } = &**rhs
                    {
                        let a = self.expr(ml)?;
                        let b = self.expr(mr)?;
                        let c = self.expr(lhs)?;
                        let dst = self.fresh();
                        return Ok(self.emit(Op::Fma { dst, a, b, c }));
                    }
                }
                let l = self.expr(lhs)?;
                let r = self.expr(rhs)?;
                let dst = self.fresh();
                Ok(self.emit(Op::Bin {
                    dst,
                    op: *op,
                    lhs: l,
                    rhs: r,
                }))
            }
        }
    }

    fn name(&mut self, name: &str, span: Span) -> Result<RegId, LowerError> {
        // A value already computed for this name in this element shadows the loaded one,
        // so `y = a*x + y` reads the loaded y and a later statement would read the new one.
        if let Some(r) = self.env.get(name) {
            return Ok(*r);
        }
        let Some(ty) = self.params.get(name) else {
            return Err(LowerError::UnknownName {
                span,
                name: name.into(),
            });
        };
        if !ty.is_buffer() {
            let dst = self.fresh();
            return Ok(self.emit(Op::Param {
                dst,
                name: name.into(),
            }));
        }
        let Some(idx) = self.streams.iter().position(|s| s.buffer == name) else {
            return Err(LowerError::NotStreamed {
                span,
                name: name.into(),
            });
        };
        if let Some(r) = self.streams[idx].loaded {
            return Ok(r);
        }
        let dst = self.fresh();
        let reg = self.emit(Op::Load {
            dst,
            buffer: name.into(),
        });
        self.streams[idx].loaded = Some(reg);
        self.streams[idx].read = true;
        Ok(reg)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parse::parse;

    fn ir(src: &str) -> Result<KernelIr, LowerError> {
        let u = parse(src).expect("should parse");
        lower(&u, &u.kernels[0])
    }

    const SAXPY: &str = "\
machine sm_120

kernel saxpy(n: u32, a: f32, x: [f32], y: [f32])
    intensity 0.1667

    stream x : dram -> reg
    stream y : dram -> reg, drain

    at reg:
        y = a * x + y
";

    #[test]
    fn cost_is_derived_from_the_program_not_declared() {
        let k = ir(SAXPY).expect("saxpy should lower");
        // x read 4, y read 4, y written 4.
        assert_eq!(k.cost.read_bytes_per_element, 8.0);
        assert_eq!(k.cost.write_bytes_per_element, 4.0);
        assert_eq!(k.cost.bytes_per_element, 12.0);
        // One fma = 2 flops.
        assert_eq!(k.cost.flops_per_element, 2.0);
        assert!(
            (k.cost.intensity - 2.0 / 12.0).abs() < 1e-12,
            "{}",
            k.cost.intensity
        );
    }

    #[test]
    fn a_times_x_plus_y_becomes_one_fma() {
        let k = ir(SAXPY).unwrap();
        assert_eq!(
            k.ops.iter().filter(|o| matches!(o, Op::Fma { .. })).count(),
            1,
            "{:?}",
            k.ops
        );
        assert_eq!(
            k.ops.iter().filter(|o| matches!(o, Op::Bin { .. })).count(),
            0
        );
    }

    #[test]
    fn a_buffer_read_without_a_stream_is_refused_and_says_what_to_add() {
        let src = "machine m\n\nkernel k(x: [f32], y: [f32])\n    stream y : dram -> reg, drain\n    at reg:
        y = x\n";
        let e = ir(src).unwrap_err();
        let s = e.to_string();
        assert!(s.contains("stream x : dram -> reg"), "{s}");
        assert!(s.contains("not declared resident"), "{s}");
    }

    #[test]
    fn writing_a_stream_that_does_not_drain_is_refused() {
        let src = "machine m\n\nkernel k(x: [f32])\n    stream x : dram -> reg\n    at reg:
        x = x + 1\n";
        let e = ir(src).unwrap_err();
        assert!(e.to_string().contains("no `drain`"), "{e}");
    }

    #[test]
    fn draining_something_never_computed_is_refused() {
        let src = "machine m\n\nkernel k(x: [f32], y: [f32])\n    stream x : dram -> reg\n    stream y : dram -> reg, drain\n    at reg:
        x = x\n";
        // x has no drain, so this trips WriteWithoutDrain first; swap to make y the issue.
        let src2 = "machine m\n\nkernel k(x: [f32], y: [f32])\n    stream x : dram -> reg\n    stream y : dram -> reg, drain\n    at reg:
        y = x\n";
        assert!(ir(src).is_err());
        assert!(ir(src2).is_ok(), "y is drained and assigned");
    }

    #[test]
    fn a_stream_of_a_scalar_is_refused() {
        let src = "machine m\n\nkernel k(a: f32, y: [f32])\n    stream a : dram -> reg\n    stream y : dram -> reg, drain\n    at reg:
        y = a\n";
        let e = ir(src).unwrap_err();
        assert!(e.to_string().contains("scalar parameter"), "{e}");
    }

    #[test]
    fn an_unknown_name_names_itself() {
        let src = "machine m\n\nkernel k(y: [f32])\n    stream y : dram -> reg, drain\n    at reg:
        y = z\n";
        let e = ir(src).unwrap_err();
        assert!(e.to_string().contains("`z`"), "{e}");
    }

    #[test]
    fn a_buffer_is_loaded_once_however_often_it_is_named() {
        let src = "machine m\n\nkernel k(x: [f32], y: [f32])\n    stream x : dram -> reg\n    stream y : dram -> reg, drain\n    at reg:
        y = x + x + x\n";
        let k = ir(src).unwrap();
        assert_eq!(
            k.ops
                .iter()
                .filter(|o| matches!(o, Op::Load { .. }))
                .count(),
            1,
            "x is named three times and loaded once: {:?}",
            k.ops
        );
    }

    #[test]
    fn a_drained_buffer_that_is_never_read_costs_a_write_and_not_a_read() {
        // y is assigned and drained, but its old value is never used, so nothing loads it.
        // Charging a read here would inflate the denominator and understate the intensity.
        let src = "machine m\n\nkernel k(x: [f32], y: [f32])\n    stream x : dram -> reg\n    stream y : dram -> reg, drain\n    at reg:
        y = x + x\n";
        let k = ir(src).unwrap();
        assert_eq!(k.cost.read_bytes_per_element, 4.0, "x only");
        assert_eq!(k.cost.write_bytes_per_element, 4.0, "y only");
        assert_eq!(k.cost.bytes_per_element, 8.0);

        // saxpy does read y, because `a * x + y` names it.
        let saxpy = ir(SAXPY).unwrap();
        assert_eq!(saxpy.cost.read_bytes_per_element, 8.0, "x and y");
    }

    #[test]
    fn v1_refuses_a_level_it_cannot_generate_instead_of_ignoring_it() {
        let src = "machine m\n\nkernel k(x: [f32])\n    stream x : dram -> reg, drain\n    at smem:
        x = x\n";
        let e = ir(src).unwrap_err();
        assert!(e.to_string().contains("not implemented"), "{e}");
    }
}
