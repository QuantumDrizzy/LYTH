//! The shape of a `.lyth` source file.
//!
//! Movement is declared before arithmetic, and arithmetic is subordinate to it. That ordering
//! is the whole inversion ADR-0001 argues for, so it is enforced by the grammar rather than by
//! a lint: there is nowhere in this tree to put an operation that is not at a level a stream
//! has already reached.

use crate::lex::Span;

#[derive(Debug, Clone, PartialEq)]
pub struct Unit {
    /// `machine sm_120` — names the machine file this source is checked against.
    pub machine: String,
    pub machine_span: Span,
    pub kernels: Vec<Kernel>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Kernel {
    pub name: String,
    pub span: Span,
    pub params: Vec<Param>,
    /// The declared arithmetic intensity, checked against the one derived from the body.
    pub declared_intensity: Option<f64>,
    pub intensity_span: Option<Span>,
    pub streams: Vec<StreamDecl>,
    /// `None` is rank 1: buffers are walked at the loop index and the body names no indices.
    pub space: Option<SpaceDecl>,
    /// At most one in v1. See ADR-0011.
    pub reductions: Vec<ReduceDecl>,
    /// `at <level>:` blocks, in source order.
    pub blocks: Vec<Block>,
}

/// `reduce sum p : reg -> smem -> dram into partial`
///
/// A movement declaration like `stream`, not a statement: it says which value is combined,
/// through which levels it travels, and where the block's result lands. The path is written
/// out rather than implied by the word `reduce`, because this language does not infer
/// movement -- it checks that the arithmetic fits what was declared.
#[derive(Debug, Clone, PartialEq)]
pub struct ReduceDecl {
    pub op: ReduceOp,
    /// The value being combined. A local defined in the body, not a buffer.
    pub source: String,
    /// Levels the partial travels through, as written.
    pub path: Vec<Level>,
    /// Buffer receiving one value per block.
    pub into: String,
    pub span: Span,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReduceOp {
    Sum,
    Max,
    Min,
}

impl ReduceOp {
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "sum" => Some(ReduceOp::Sum),
            "max" => Some(ReduceOp::Max),
            "min" => Some(ReduceOp::Min),
            _ => None,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            ReduceOp::Sum => "sum",
            ReduceOp::Max => "max",
            ReduceOp::Min => "min",
        }
    }

    /// The value a thread with no element contributes, so that it can take part in the tree
    /// instead of branching out and leaving the shared array uninitialised.
    pub fn identity(self) -> f32 {
        match self {
            ReduceOp::Sum => 0.0,
            // The max of an empty set. A block whose threads all fall past `n` writes this to
            // its partial, and a caller combining partials with `max` is unaffected by it --
            // the same property that makes 0.0 safe for `sum`.
            ReduceOp::Max => f32::NEG_INFINITY,
            ReduceOp::Min => f32::INFINITY,
        }
    }

    /// FLOPs one combine retires.
    ///
    /// **`max` and `min` retire none.** They are a compare-and-select, not arithmetic: no
    /// vendor's FLOP/s figure counts them and no published flop count for a reduction counts
    /// its comparisons. Charging them would place the kernel against a compute ceiling
    /// measured with FMA, which is the one comparison ADR-0005 forbids.
    ///
    /// [KNOWN LIMIT] Zero flops is not zero time. Each combine still issues, and `bar.sync`
    /// between rounds costs what it costs. This is an arithmetic count, not an instruction
    /// count. See ADR-0013.
    pub fn flops(self) -> f64 {
        match self {
            ReduceOp::Sum => 1.0,
            ReduceOp::Max | ReduceOp::Min => 0.0,
        }
    }

    /// Combine two values exactly as the generated PTX does.
    ///
    /// This is the oracle's contract, and the signed-zero branch is why it cannot be
    /// `a.max(b)`.
    ///
    /// **`f32::max` is not a deterministic function of its inputs when both are zero.** It
    /// lowers to `llvm.maxnum`, which is specified to return *either* operand when they
    /// compare equal, and `-0.0 == 0.0`. Measured on this machine: constant-folded at compile
    /// time it yields `+0.0`, executed at run time it yields `-0.0`. The same source, two
    /// answers, decided by the optimiser.
    ///
    /// PTX `max.f32` has no such freedom: it returns `+0.0`, which is what IEEE 754-2019
    /// `maximumNumber` specifies. So the branch below is not the evaluator deferring to the
    /// hardware over the host language -- it is the evaluator refusing to be built on
    /// unspecified behaviour. Without it, 135 of 65536 block partials differed from the
    /// device. See ADR-0013 and `examples/signed-zero.lyth`.
    pub fn combine(self, a: f32, b: f32) -> f32 {
        match self {
            ReduceOp::Sum => a + b,
            ReduceOp::Max => {
                if a == 0.0 && b == 0.0 {
                    // Both zero, differing only in sign: take the positive one.
                    if a.is_sign_negative() {
                        b
                    } else {
                        a
                    }
                } else {
                    a.max(b)
                }
            }
            ReduceOp::Min => {
                if a == 0.0 && b == 0.0 {
                    if a.is_sign_negative() {
                        a
                    } else {
                        b
                    }
                } else {
                    a.min(b)
                }
            }
        }
    }

    /// Whether combining is associative on the bit patterns, not merely in mathematics.
    ///
    /// `max` and `min` select an operand and never round, so a tree's result is the same bit
    /// pattern for every tree shape, block size and grid. Floating-point addition is not
    /// associative, so `sum`'s result depends on the shape of the tree that produced it.
    ///
    /// Nothing in the compiler reorders a reduction yet. This records the property so that
    /// the optimisation, when it arrives, has a document to point at rather than an
    /// assumption to make. See ADR-0013.
    pub fn is_reorderable(self) -> bool {
        match self {
            ReduceOp::Sum => false,
            ReduceOp::Max | ReduceOp::Min => true,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Param {
    pub name: String,
    pub ty: Ty,
    /// A buffer's extents, named: `[f32; n]` is `["n"]`, `[f32; rows, cols]` is
    /// `["rows", "cols"]`. Empty for a scalar.
    ///
    /// The names are `u32` parameters of the same kernel, or the reserved extent `blocks`.
    /// Shape lives on the parameter rather than inside `Ty` so that `Ty` stays `Copy` and so
    /// that "buffer of f32" and "how long it is" stay separable, which is what lets a
    /// reduction target be sized by the launch instead of by the problem.
    pub shape: Vec<String>,
    pub span: Span,
}

/// The extent of a reduction target: one element per block, so its length is a fact about the
/// launch and not about the problem.
///
/// Spelled in the source rather than inferred from `reduce ... into partial`, because a caller
/// who sizes that buffer by the element count writes past the end of it, and the place to say
/// so is the declaration the caller reads.
pub const BLOCKS: &str = "blocks";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Ty {
    /// A scalar passed by value in the launch parameters.
    U32,
    F32,
    /// A buffer of f32, passed as a device pointer. `[f32]` in source.
    BufF32,
}

impl Ty {
    pub fn bytes(self) -> u32 {
        match self {
            Ty::U32 | Ty::F32 => 4,
            // The pointer is 8 bytes; the element it addresses is 4.
            Ty::BufF32 => 4,
        }
    }

    pub fn is_buffer(self) -> bool {
        matches!(self, Ty::BufF32)
    }

    pub fn name(self) -> &'static str {
        match self {
            Ty::U32 => "u32",
            Ty::F32 => "f32",
            Ty::BufF32 => "[f32]",
        }
    }
}

/// `stream x : dram -> reg, drain`
#[derive(Debug, Clone, PartialEq)]
pub struct StreamDecl {
    /// The buffer parameter this stream moves.
    pub buffer: String,
    pub from: Level,
    pub to: Level,
    /// Written back at the end of the element. A stream without `drain` is read-only.
    pub drain: bool,
    pub span: Span,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Level {
    Dram,
    L2,
    Smem,
    Reg,
}

impl Level {
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "dram" => Some(Level::Dram),
            "l2" => Some(Level::L2),
            "smem" => Some(Level::Smem),
            "reg" => Some(Level::Reg),
            _ => None,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Level::Dram => "dram",
            Level::L2 => "l2",
            Level::Smem => "smem",
            Level::Reg => "reg",
        }
    }
}

/// `at reg:` and the statements under it.
#[derive(Debug, Clone, PartialEq)]
pub struct Block {
    pub level: Level,
    pub span: Span,
    pub stmts: Vec<Stmt>,
}

/// `space i, j : rows, cols`
///
/// The index space, named. A kernel without one is rank 1 and walks its buffers at the loop
/// index, which is every kernel written before ADR-0015. A kernel with one names its indices
/// so the body can permute them, and the extents so the compiler knows how far each runs.
///
/// Declared rather than inferred from the buffers, for the reason movement is declared: the
/// shape of a buffer says how long it is, not which order a kernel walks it in, and those are
/// different facts. Two kernels over the same buffers can disagree about the second.
#[derive(Debug, Clone, PartialEq)]
pub struct SpaceDecl {
    /// Index variables, **outermost first**.
    pub vars: Vec<String>,
    /// The extent each one runs over, naming a `u32` parameter.
    pub extents: Vec<String>,
    pub span: Span,
}

/// `<target> = <expr>`. The only statement there is.
#[derive(Debug, Clone, PartialEq)]
pub struct Stmt {
    pub target: String,
    /// `b[j, i] = ...` gives `["j", "i"]`. Empty when the kernel is rank 1 and the target is
    /// written at the loop index.
    pub target_index: Vec<String>,
    pub target_span: Span,
    pub value: Expr,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Expr {
    /// A buffer element at the current index, or a scalar parameter.
    Name(String, Span),
    /// `a[i, j]`: a buffer element at a permutation of the index variables.
    ///
    /// v1 allows a permutation and nothing else -- no offsets, no arithmetic on an index. An
    /// offset brings the halo problem at the edges with it, and that is its own decision.
    At {
        buffer: String,
        index: Vec<String>,
        span: Span,
    },
    Const(f64, Span),
    Bin {
        op: BinOp,
        lhs: Box<Expr>,
        rhs: Box<Expr>,
        span: Span,
    },
    Neg(Box<Expr>, Span),
}

impl Expr {
    pub fn span(&self) -> Span {
        match self {
            Expr::Name(_, s)
            | Expr::At { span: s, .. }
            | Expr::Const(_, s)
            | Expr::Bin { span: s, .. }
            | Expr::Neg(_, s) => *s,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinOp {
    Add,
    Sub,
    Mul,
    Div,
}

impl BinOp {
    pub fn symbol(self) -> &'static str {
        match self {
            BinOp::Add => "+",
            BinOp::Sub => "-",
            BinOp::Mul => "*",
            BinOp::Div => "/",
        }
    }

    /// FLOPs charged for one application.
    ///
    /// Add, subtract and multiply are one each. **Divide is counted as one too**, which is a
    /// deliberate understatement: a single-precision divide is several instructions on this
    /// hardware, so a kernel full of divides will measure as doing more arithmetic than this
    /// says. Counting it as one keeps the number comparable with every published flop count,
    /// which all do the same. Recorded as a known limit rather than silently assumed.
    pub fn flops(self) -> f64 {
        1.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The disagreement ADR-0013 found on sm_120, pinned so it cannot come back.
    ///
    /// These fail if anyone simplifies `combine` back to `a.max(b)`, whose result for a pair
    /// of zeros is whatever the optimiser felt like that day.
    ///
    /// Note what is *not* asserted here: the value of `f32::max(-0.0, 0.0)`. It is
    /// unspecified, so pinning it would make this test depend on the thing the code exists to
    /// avoid depending on.
    #[test]
    fn combine_orders_signed_zeros_the_way_the_hardware_does() {
        let max = ReduceOp::Max;
        let min = ReduceOp::Min;

        // Both arguments are equal under ==, so the sign bit is the only observable.
        assert!(max.combine(-0.0, 0.0).is_sign_positive());
        assert!(max.combine(0.0, -0.0).is_sign_positive());
        assert!(min.combine(-0.0, 0.0).is_sign_negative());
        assert!(min.combine(0.0, -0.0).is_sign_negative());

        // A pair of the same zero keeps its sign rather than acquiring one.
        assert!(max.combine(-0.0, -0.0).is_sign_negative());
        assert!(min.combine(0.0, 0.0).is_sign_positive());

    }

    #[test]
    fn combine_returns_the_non_nan_operand() {
        // Measured against sm_120 by examples/not-a-number.lyth: neither side propagates.
        let max = ReduceOp::Max;
        assert_eq!(max.combine(f32::NAN, 3.0), 3.0);
        assert_eq!(max.combine(3.0, f32::NAN), 3.0);
        assert!(max.combine(f32::NAN, f32::NAN).is_nan());
    }

    #[test]
    fn the_identity_cannot_win_its_own_reduction() {
        // A block with no elements must not change the answer.
        let max = ReduceOp::Max;
        let min = ReduceOp::Min;
        assert_eq!(max.combine(max.identity(), -1e30), -1e30);
        assert_eq!(min.combine(min.identity(), 1e30), 1e30);
        assert_eq!(ReduceOp::Sum.combine(ReduceOp::Sum.identity(), 7.0), 7.0);
    }

    #[test]
    fn only_selection_is_reorderable_and_only_selection_is_free() {
        for op in [ReduceOp::Max, ReduceOp::Min] {
            assert!(op.is_reorderable(), "{} selects, so order cannot matter", op.name());
            assert_eq!(op.flops(), 0.0, "{} is not arithmetic", op.name());
        }
        assert!(!ReduceOp::Sum.is_reorderable());
        assert_eq!(ReduceOp::Sum.flops(), 1.0);
    }
}

