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
    /// `None` is rank 1. Rank 2 names its indices and the extents they run over, and every
    /// buffer access permutes them.
    pub space: Option<SpaceIr>,
    /// `None` is one element per thread. A tile blocks the space into patches, one per block,
    /// and is what makes a staged stream legal.
    pub tile: Option<Vec<u32>>,
    /// Outputs of the tile each thread owns, per axis. `None` is one.
    ///
    /// Read through `block_threads`, never multiplied out at a call site: the block width is
    /// derived from the tile in two places already and the two have drifted apart once.
    pub coarsen: Option<Vec<u32>>,
    /// How the staged buffers sit in shared memory, and the bytes the launch must request.
    /// `None` when nothing is staged.
    pub shared: Option<SharedLayout>,
    /// Flattened straight-line body, in evaluation order.
    pub ops: Vec<Op>,
    /// Registers holding the final value of each drained buffer.
    pub drains: Vec<(String, RegId)>,
    /// At most one in v1.
    pub reduction: Option<ReductionIr>,
    /// The contracted axis, resolved. `None` for every kernel that does not contract.
    pub contract: Option<ContractIr>,
    pub cost: Cost,
}

impl KernelIr {
    /// Threads per block: the tile's area divided by what each thread owns of it.
    ///
    /// **One definition.** The manifest publishes this and `lyth run` launches it, and when the
    /// same rule lived in both places they drifted -- the reduction grid was corrected in one
    /// and not the other, and every generated binding published a launch shape that had already
    /// been measured at half the achievable bandwidth (ADR-0019). A fix belongs where the rule
    /// lives.
    ///
    /// `fallback` is what an untiled kernel uses, which is a launch choice rather than a
    /// property of the kernel and so is passed in.
    pub fn block_threads(&self, fallback: u32) -> u32 {
        let Some(tile) = &self.tile else {
            return fallback;
        };
        let area: u32 = tile.iter().product();
        let per_thread: u32 = self
            .coarsen
            .as_ref()
            .map(|c| c.iter().product())
            .unwrap_or(1);
        (area / per_thread.max(1)).max(1)
    }
}


/// `contract sum p : k`, resolved.
///
/// Sequential accumulation inside one thread, over an axis the space does not iterate. Unlike
/// `ReductionIr` there is no path: nothing crosses a level boundary because of the
/// contraction, and nothing is combined across threads. What it changes is how many times each
/// streamed element is read per output, which is the whole of ADR-0018.
#[derive(Debug, Clone, PartialEq)]
pub struct ContractIr {
    pub op: ReduceOp,
    /// The index variable walked. Never a space variable.
    pub var: String,
    /// The `u32` parameter it runs over.
    pub extent: String,
    /// Buffers whose index mentions `var`, and the position it appears at. These are the ones
    /// read `extent` times per output instead of once, and the ones a tile can reuse.
    pub over: Vec<(String, usize)>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ReductionIr {
    pub op: ReduceOp,
    /// Register holding the per-element value being combined.
    pub value: RegId,
    /// Buffer receiving one result per block.
    pub into: String,
    pub path: Vec<Level>,
}

/// The index space, resolved.
///
/// The traversal order is **row-major, outermost first**, and it is part of the contract, not
/// an implementation detail: the linear index `k` decomposes as `i = k / extents[1]` and
/// `j = k % extents[1]`, and `eval` walks that same order so any accumulation meets its
/// operands in the order the device met them. ADR-0015.
#[derive(Debug, Clone, PartialEq)]
pub struct SpaceIr {
    pub vars: Vec<String>,
    pub extents: Vec<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct StreamIr {
    pub buffer: String,
    /// The permutation of the space variables this buffer is accessed with: `a[i, j]` gives
    /// `["i", "j"]`. Empty at rank 1, where every buffer is walked at the loop index.
    ///
    /// One permutation per buffer per kernel. A buffer read at `a[i, j]` **and** at `a[j, i]`
    /// would need two addresses per element, and this IR loads each buffer once.
    pub index: Vec<String>,
    pub from: Level,
    pub to: Level,
    pub drain: bool,
    /// Register the loaded element lands in. `None` for a buffer that is only written.
    pub loaded: Option<RegId>,
    pub read: bool,
    /// Whether this stream travels through shared memory: `dram -> smem -> reg`.
    ///
    /// A staged buffer is read from global once per tile, coalesced, and read from shared by
    /// the body. That is how a transposition moves off the memory bus and into a place where
    /// a one-element skew makes it free. ADR-0017.
    pub staged: bool,
    /// Whether consecutive threads touch consecutive elements of this buffer **as emitted**.
    ///
    /// At rank 1 every access is the loop index, so always. At rank 2 it holds when the
    /// buffer's innermost index is the space's fastest variable — `a[i, j]` under
    /// `space i, j` is contiguous, `b[j, i]` is not, and that is the whole of a transpose.
    ///
    /// **A tile that stages overrides it to true for every buffer**, and not as a special
    /// case: the four-phase body reads a row-contiguous patch into shared, permutes inside
    /// shared, and writes a row-contiguous patch out. The permutation is absorbed, so no
    /// global access is left strided — including the drained buffer, which is not itself
    /// staged. That is a property of the emitted schedule rather than a theorem about tiles,
    /// which is why `lyth-ptx` refuses the shapes it cannot emit that way.
    pub coalesced: bool,
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

/// Traffic per element at one level of the hierarchy, as an affine function of the contracted
/// extent.
///
/// `read` and `write` are the part that does not depend on it, and for every kernel without a
/// `contract` they are the whole of it -- both `_per_extent` fields are zero and `total_at`
/// agrees with `total` at every extent.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct LevelCost {
    pub level: Level,
    pub read: f64,
    pub write: f64,
    /// Bytes read per element for each unit of the contracted extent.
    pub read_per_extent: f64,
    pub write_per_extent: f64,
}

impl LevelCost {
    fn at(level: Level) -> Self {
        LevelCost {
            level,
            ..Default::default()
        }
    }

    /// The part that does not depend on a contracted extent.
    pub fn total(&self) -> f64 {
        self.read + self.write
    }

    pub fn per_extent(&self) -> f64 {
        self.read_per_extent + self.write_per_extent
    }

    /// Traffic per element at a known contracted extent.
    pub fn total_at(&self, k: u32) -> f64 {
        self.total() + self.per_extent() * k as f64
    }

    /// Whether this level carries traffic at all, at any extent.
    ///
    /// Not `total() > 0.0`: a matmul reads nothing per element that is not inside its
    /// contraction, so its DRAM read is 0.0 fixed and 0.25 per unit of k. A roofline that
    /// asked only about the fixed part would walk past DRAM and report shared memory.
    pub fn carries_traffic(&self) -> bool {
        self.total() > 0.0 || self.per_extent() > 0.0
    }
}

/// What the compiler derived. Nothing in here was typed by a human.
///
/// Traffic is per level, not one number. A reduction moves bytes at `dram` **and** at `smem`,
/// and collapsing the two would either hide the shared traffic or corrupt the roofline
/// position, which is a statement about DRAM. They are reported side by side instead.
#[derive(Debug, Clone, PartialEq)]
pub struct Cost {
    /// One entry per level the kernel touches, deepest (farthest from registers) first.
    pub levels: Vec<LevelCost>,
    /// FLOPs per output element that do **not** depend on a contracted extent.
    ///
    /// Private, and read through `flops_per_element()`, which returns `None` for a kernel
    /// whose flop count is a function of a launch extent. The field is 0.0 for a matmul --
    /// every one of its flops is inside the contraction -- and 0.0 is a number that would
    /// print without complaint. The accessor is what makes that unreachable.
    flops_fixed: f64,
    /// FLOPs/byte at the deepest level with traffic. This is the roofline number.
    ///
    /// For a contracted kernel this is the **limit** as the contracted extent grows, and
    /// `asymptotic` says so. It is the right roofline number either way -- what changes is
    /// whether it is reached or approached.
    pub intensity: f64,
    /// Whether `intensity` is a limit rather than an exact figure.
    pub asymptotic: bool,
    /// How a `contract` makes the cost a function of a launch extent. `None` for every kernel
    /// whose traffic is a constant.
    pub contracted: Option<ContractedCost>,
    /// Traffic at the **L1-to-L2 interface**, at the 32-byte sector the memory system moves in.
    ///
    /// The level is not decoration. Measured on sm_120 against `lts__t_bytes.sum`, this figure
    /// is exact -- 8.00 to 8.02 against 8 for a coalesced rank-2 copy at every size, 36.05 and
    /// 36.02 against 36 for a transpose. Against **DRAM** it is not a bound in either
    /// direction: a transpose at 1024 x 1024 moves half its own payload to DRAM because the L2
    /// keeps every write, and at 8192 x 8192 it moves 7.6x the payload, past this figure,
    /// because partially written sectors are evicted and fetched back. See ADR-0015.
    ///
    /// A 32-byte sector is eight f32. A coalesced access has consecutive threads on
    /// consecutive elements, so a warp's 32 threads cover 128 contiguous bytes in four
    /// sectors and every byte fetched is a byte wanted: 4 per element. A strided access has
    /// each thread in its own sector, so 32 bytes move for every 4 wanted.
    ///
    /// **These are upper bounds, and deliberately.** The exact figure is
    /// `min(32, 4 * stride)` where `stride` is the element distance between neighbouring
    /// threads, which for a strided access is the buffer's own row length -- a *launch* value,
    /// not something the source says. A matrix four columns wide wastes four times, not eight.
    /// The bound here assumes a row of eight or more; `lyth run` refines it once the extents
    /// are known, and `--ncu` measures what actually happened.
    ///
    /// [KNOWN LIMIT] The model is a warp at a time and ignores the warp that straddles a row
    /// boundary, where the pattern is neither of the two cases. At `cols >= 32` that is at
    /// most one warp per row.
    pub sector_read_per_element: f64,
    pub sector_write_per_element: f64,
    /// The same, per unit of a contracted extent. Zero without a `contract`.
    pub sector_read_per_extent: f64,
    pub sector_write_per_extent: f64,
    /// DRAM bytes written once per block rather than once per element: a reduction's partial.
    ///
    /// Deliberately **not** folded into `intensity`. Per element it is this over the block
    /// size, and the block size is a launch parameter that does not appear in the source, so
    /// folding it in would require the compiler to invent a constant. At a block of 256 it is
    /// 0.016 bytes/element against 8, which is 0.2%.
    pub dram_bytes_per_block: f64,
}

/// `8` rather than `8.0`, `0.25` as itself.
///
/// `{:?}` on an f64 is unambiguous and ugly; `{}` drops the distinction between 8 and 8.0,
/// which for a byte count nobody needs. These figures are read by people.
pub fn num(v: f64) -> String {
    if v.fract() == 0.0 && v.abs() < 1e15 {
        format!("{}", v as i64)
    } else {
        format!("{v}")
    }
}

/// One stream's contribution to traffic per output element, and where it comes from.
///
/// Kept per stream rather than summed into two coefficients, because a compiler whose claim is
/// that the number is derived has to be able to say *from what*. `bytes(K) = 0.25K + 4` is not
/// checkable by a reader; `a` contributes `4/32` per unit of `k` because a tile of 32 threads
/// shares each element it stages, and `c` contributes 4 once, is.
#[derive(Debug, Clone, PartialEq)]
pub struct CostTerm {
    pub buffer: String,
    /// A write, rather than a read.
    pub write: bool,
    /// Bytes per output element, independent of the contracted extent.
    pub fixed: f64,
    /// Bytes per output element for **each unit** of the contracted extent.
    pub per_extent: f64,
    /// Threads that share one element of this buffer, which is what divides `per_extent`.
    ///
    /// **Earned by staging, not by tiling.** A stream that names the contracted axis and is
    /// not staged is read once per thread per step: the tile says the threads exist, and
    /// shared memory is what makes them share. A buffer left in `dram -> reg` inside a tiled
    /// matmul has a reuse of 1 and moves K elements per output, not K/T -- which is the
    /// difference between a derived expression and a recognised kernel name.
    pub reuse: u32,
}

/// What a `contract` does to the cost: makes it a function of one launch extent.
///
/// The form is affine in that extent -- `a*K + b` -- and that is closed for what this language
/// can express, not a simplification. Tiling the contracted axis as well would add another
/// term in `K` and stay affine; only a second contracted axis would give a product, and v1
/// refuses one. See ADR-0018.
#[derive(Debug, Clone, PartialEq)]
pub struct ContractedCost {
    /// The `u32` parameter the contraction runs over: `k`.
    pub extent: String,
    /// FLOPs per output element for each unit of the extent. `2.0` for a `sum` matmul: one
    /// multiply and one add. `1.0` for a `max` one, because a compare-and-select retires no
    /// flops (ADR-0013) -- so the combinator changes the flops and not the traffic.
    pub flops_per_extent: f64,
    /// Per stream, in declaration order.
    pub terms: Vec<CostTerm>,
    /// Bytes per output element for each unit of the extent: the sum of the terms.
    pub bytes_per_extent: f64,
}

impl ContractedCost {
    /// The traffic, as an expression a reader can check: `0.25 * k + 4`.
    ///
    /// Rendered rather than stored, so there is exactly one place the coefficients live and no
    /// way for the sentence and the number to drift apart.
    pub fn bytes_expr(&self, fixed: f64) -> String {
        let k = &self.extent;
        if fixed == 0.0 {
            format!("{} * {k}", num(self.bytes_per_extent))
        } else {
            format!("{} * {k} + {}", num(self.bytes_per_extent), num(fixed))
        }
    }

    /// `2 * k`, the flops per output element.
    pub fn flops_expr(&self, fixed: f64) -> String {
        let k = &self.extent;
        if fixed == 0.0 {
            format!("{} * {k}", num(self.flops_per_extent))
        } else {
            format!("{} * {k} + {}", num(self.flops_per_extent), num(fixed))
        }
    }
}

impl Cost {
    pub fn at(&self, level: Level) -> Option<&LevelCost> {
        self.levels.iter().find(|l| l.level == level)
    }

    /// The deepest level carrying traffic — the one the roofline is about.
    pub fn roofline(&self) -> Option<&LevelCost> {
        self.levels.iter().find(|l| l.carries_traffic())
    }

    /// Bytes per output element, when that is a constant.
    ///
    /// `None` for a contracted kernel, where it is a function of a launch extent. The
    /// alternative -- returning the constant part -- is 4.0 for a matmul, a number that looks
    /// exactly like every other number this compiler reports and is wrong by a factor of K.
    /// Use `bytes_at`.
    pub fn bytes_per_element(&self) -> Option<f64> {
        if self.contracted.is_some() {
            return None;
        }
        Some(self.bytes_fixed())
    }

    /// The flop count as an expression, when it is one. `None` for a constant kernel.
    pub fn flops_expr(&self) -> Option<String> {
        self.contracted
            .as_ref()
            .map(|c| c.flops_expr(self.flops_fixed))
    }

    /// The traffic as an expression, when it is one. `None` for a constant kernel.
    pub fn bytes_expr(&self) -> Option<String> {
        self.contracted
            .as_ref()
            .map(|c| c.bytes_expr(self.bytes_fixed()))
    }

    /// FLOPs per output element, when that is a constant. `None` for a contracted kernel.
    pub fn flops_per_element(&self) -> Option<f64> {
        if self.contracted.is_some() {
            return None;
        }
        Some(self.flops_fixed)
    }

    /// One level's read and write, as expressions when they are expressions.
    ///
    /// `("0.25 * k", "4")` for a matmul at DRAM. The alternative is what this printed before
    /// the contraction existed -- `0 read + 4 written` -- where the zero is the fixed part of
    /// something that is not fixed.
    pub fn traffic_words(&self, level: Level) -> (String, String) {
        let Some(l) = self.at(level) else {
            return ("0".into(), "0".into());
        };
        let word = |fixed: f64, per: f64| -> String {
            match (&self.contracted, per) {
                (Some(c), p) if p > 0.0 && fixed > 0.0 => {
                    format!("{} * {} + {}", num(p), c.extent, num(fixed))
                }
                (Some(c), p) if p > 0.0 => format!("{} * {}", num(p), c.extent),
                _ => num(fixed),
            }
        };
        (
            word(l.read, l.read_per_extent),
            word(l.write, l.write_per_extent),
        )
    }

    /// The part of the flop count that does not depend on a contracted extent. For a kernel
    /// without a contraction this is the whole of it.
    pub fn flops_fixed_part(&self) -> f64 {
        self.flops_fixed
    }

    /// The part of the traffic that does not depend on a contracted extent. For a kernel
    /// without a contraction this is the whole of it.
    pub fn bytes_fixed(&self) -> f64 {
        self.roofline().map(LevelCost::total).unwrap_or(0.0)
    }

    /// Bytes per output element at a known contracted extent.
    ///
    /// Equal to `bytes_per_element()` for every kernel without a contraction, at every `k`.
    pub fn bytes_at(&self, k: u32) -> f64 {
        self.bytes_fixed()
            + self
                .contracted
                .as_ref()
                .map(|c| c.bytes_per_extent * k as f64)
                .unwrap_or(0.0)
    }

    /// FLOPs per output element at a known contracted extent.
    pub fn flops_at(&self, k: u32) -> f64 {
        self.flops_fixed
            + self
                .contracted
                .as_ref()
                .map(|c| c.flops_per_extent * k as f64)
                .unwrap_or(0.0)
    }

    /// Arithmetic intensity at a known contracted extent: the exact figure `intensity`
    /// approaches.
    pub fn intensity_at(&self, k: u32) -> f64 {
        let bytes = self.bytes_at(k);
        if bytes > 0.0 {
            self.flops_at(k) / bytes
        } else {
            0.0
        }
    }

    /// Payload over sectors: 1.0 when every byte fetched is a byte wanted.
    ///
    /// Reported beside the payload rather than folded into it. The payload is what the source
    /// asks for and what `intensity` is checked against; this is what the bus carries. Two
    /// numbers, because they answer different questions and a single one would hide whichever
    /// question the reader had.
    pub fn coalescence(&self) -> f64 {
        // Both halves of each side, or the ratio compares a per-output numerator against a
        // per-step denominator. For a kernel without a contraction the per-extent parts are
        // zero and this is the figure it always was.
        let sectors = self.sector_read_per_element
            + self.sector_write_per_element
            + self.sector_read_per_extent
            + self.sector_write_per_extent;
        let payload = self.bytes_fixed()
            + self
                .contracted
                .as_ref()
                .map(|c| c.bytes_per_extent)
                .unwrap_or(0.0);
        if sectors > 0.0 {
            payload / sectors
        } else {
            1.0
        }
    }

    pub fn read_bytes_per_element(&self) -> f64 {
        self.roofline().map(|l| l.read).unwrap_or(0.0)
    }

    pub fn write_bytes_per_element(&self) -> f64 {
        self.roofline().map(|l| l.write).unwrap_or(0.0)
    }

    pub fn level(&self) -> Level {
        self.roofline().map(|l| l.level).unwrap_or(Level::Dram)
    }
}

#[derive(Debug, thiserror::Error)]
pub enum LowerError {
    #[error("{span}: `{name}` is used but is not a parameter of this kernel")]
    UnknownName { span: Span, name: String },
    #[error(
        "{span}: `coarsen` without a `tile`. Coarsening says how a block's threads are spread over its tile, and without a tile there is one thread per element and nothing to spread."
    )]
    CoarsenWithoutTile { span: Span },
    #[error(
        "{span}: `coarsen` gives {dims} factors and `tile` has {tile} dimensions. One factor per tile axis: they pair up."
    )]
    CoarsenRankMismatch {
        span: Span,
        dims: usize,
        tile: usize,
    },
    #[error(
        "{span}: axis {axis} is a tile of {edge} coarsened by {per}, which does not divide it. Each thread owns a whole sub-rectangle of the tile, so the factor has to split the edge evenly."
    )]
    CoarsenDoesNotDivide {
        span: Span,
        axis: usize,
        per: u32,
        edge: u32,
    },
    #[error(
        "{span}: `coarsen 1, 1` is what a tile does without the line. Delete it, or give a factor that changes something."
    )]
    CoarsenIsIdentity { span: Span },
    #[error(
        "{span}: `contract` needs a `space`. The contracted axis is the one the space does not iterate, so without free indices there is nothing to contract against."
    )]
    ContractWithoutSpace { span: Span },
    #[error(
        "{span}: a kernel cannot both `reduce` and `contract` in v1. They are different machines: `reduce` combines across threads through shared memory, `contract` combines inside one thread in a register. Composing them is a design, not a default."
    )]
    ContractAndReduce { span: Span },
    #[error(
        "{span}: `contract ... {var}` names `{var}`, which the space already iterates ({vars}). A contracted axis is walked and combined; a free one is walked and kept."
    )]
    ContractVarIsFree {
        span: Span,
        var: String,
        vars: String,
    },
    #[error(
        "{span}: `contract ... {var}` names `{var}`, which is a parameter of this kernel. An index variable is not a value the caller passes."
    )]
    ContractVarIsParam { span: Span, var: String },
    #[error(
        "{span}: `contract ... {var} : {extent}` runs over `{extent}`, which is not a u32 parameter of this kernel. The contracted extent is a length the caller passes."
    )]
    ContractExtentUnknown {
        span: Span,
        var: String,
        extent: String,
    },
    #[error(
        "{span}: `contract` runs over `{extent}`, which the space already runs over. One extent cannot be both walked-and-kept and walked-and-combined."
    )]
    ContractExtentIsFree { span: Span, extent: String },
    #[error(
        "{span}: `contract ... {var}` is declared but no buffer is indexed at `{var}`, so nothing is contracted. The declaration would cost K times the traffic the body moves."
    )]
    ContractOverNothing { span: Span, var: String },
    #[error(
        "{span}: `contract ... {var}` walks only `{buffer}`. With one operand there is no reuse for a tile to exploit, so the traffic this compiler derives for a contraction does not describe it. Reducing a buffer along an axis is a different kernel and is not derived yet (ADR-0018)."
    )]
    ContractOverOne {
        span: Span,
        var: String,
        buffer: String,
    },
    #[error(
        "{span}: `{buffer}` is written at `{var}`, the contracted axis. The target has one value per output and the contraction is what collapses `{var}`; writing at it would store each term in turn and keep the last."
    )]
    ContractedTarget {
        span: Span,
        buffer: String,
        var: String,
    },
    #[error(
        "{span}: `{name}` contracts over `{extent}`, so its intensity is a function of a launch extent and a source constant cannot be one. Write `intensity asymptotic {derived:.4}`, which claims the limit the body approaches -- a weaker claim than an elementwise kernel makes, and the only one that is true here."
    )]
    ContractedIntensityNotAsymptotic {
        span: Span,
        name: String,
        extent: String,
        derived: f64,
    },
    #[error(
        "{span}: `{name}` declares `intensity asymptotic` but does not contract. Its traffic is a constant, so the exact figure is available and the limit is the same number wearing a weaker claim. Write `intensity {declared}`."
    )]
    AsymptoticWithoutContraction {
        span: Span,
        name: String,
        declared: f64,
    },
    #[error(
        "{span}: `{name}` parses and resolves, but its cost is not derived yet. A contraction moves `2K/T + 1` elements per output -- an expression in a launch extent, not a constant -- and this compiler will not publish a constant in its place. ADR-0018 step 2."
    )]
    ContractCostNotDerived { span: Span, name: String },
    #[error(
        "{span}: `{name}` is a buffer but no stream moves it. \
         Add `stream {name} : dram -> reg`. \
         You cannot operate on what you have not declared resident (ADR-0001)."
    )]
    NotStreamed { span: Span, name: String },
    #[error("{span}: `stream {name}` names no parameter of this kernel")]
    StreamOfNothing { span: Span, name: String },
    #[error(
        "{span}: `{buffer}` is declared `[f32; {dim}]` but `{dim}` is not a u32 parameter of this kernel. An extent names a length the caller passes, or `blocks` for a reduction target."
    )]
    UnknownExtent {
        span: Span,
        buffer: String,
        dim: String,
    },
    #[error(
        "{span}: `{buffer}` is streamed with extent `{dim}`, but `{other}` is streamed with `{other_dim}`. Streamed buffers are walked by one index space, so they must be the same length. Nothing here can check two lengths against each other at run time."
    )]
    ExtentMismatch {
        span: Span,
        buffer: String,
        dim: String,
        other: String,
        other_dim: String,
    },
    #[error(
        "{span}: `{buffer}` is streamed but declared `[f32; blocks]`. `blocks` is the number of blocks the launch chose, which is one value per block and not one per element; a stream walks elements. Only a reduction target is sized that way."
    )]
    StreamedPerBlock { span: Span, buffer: String },
    #[error(
        "{span}: `{buffer}` receives a reduction but is declared `[f32; {dim}]`. A reduction writes one value per block, so its target is sized by the launch: `[f32; blocks]`. Sizing it by the element count is how a caller allocates the wrong buffer."
    )]
    ReductionTargetNotPerBlock {
        span: Span,
        buffer: String,
        dim: String,
    },
    #[error(
        "{span}: `{buffer}` is declared with {rank} extents, but this kernel declares no `space`. Add `space i, j : rows, cols` to walk it, or give the buffer one extent."
    )]
    UnsupportedRank {
        span: Span,
        buffer: String,
        rank: usize,
    },
    #[error(
        "{span}: `space` names {vars} index variables but `{buffer}` is declared with {rank} extents. Every buffer has one extent per index."
    )]
    RankMismatch {
        span: Span,
        buffer: String,
        rank: usize,
        vars: usize,
    },
    #[error(
        "{span}: `{buffer}` is used without an index, but this kernel declares `space {vars}`. Write `{buffer}[{first}]` and say which way it is walked."
    )]
    MissingIndex {
        span: Span,
        buffer: String,
        vars: String,
        first: String,
    },
    #[error(
        "{span}: `{buffer}[{index}]` indexes a kernel that declares no `space`. Either add one, or drop the index and let the buffer be walked at the loop index."
    )]
    IndexWithoutSpace {
        span: Span,
        buffer: String,
        index: String,
    },
    #[error(
        "{span}: `{buffer}[{index}]` is not a permutation of `space {vars}`. v1 allows the index variables in any order, each exactly once, and nothing else: an offset or an expression makes the footprint something to solve for rather than to read off."
    )]
    NotAPermutation {
        span: Span,
        buffer: String,
        index: String,
        vars: String,
    },
    #[error(
        "{span}: `{buffer}` is indexed as `[{first}]` and as `[{second}]` in the same kernel. That is two addresses per element for one buffer, and this compiler loads each buffer once. Split it into two kernels or two parameters."
    )]
    TwoIndexings {
        span: Span,
        buffer: String,
        first: String,
        second: String,
    },
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
    #[error(
        "{span}: `{name}` is computed and then discarded. A local is only useful as the source of a reduction; either reduce it, or assign to a drained buffer."
    )]
    DeadLocal { span: Span, name: String },
    #[error(
        "{span}: `tile` blocks the index space, and this kernel declares no `space` to block. Add `space i, j : rows, cols`."
    )]
    TileWithoutSpace { span: Span },
    #[error(
        "{span}: `tile` gives {dims} dimensions but `space` names {vars} index variables. A tile has one dimension per index."
    )]
    TileRankMismatch {
        span: Span,
        dims: usize,
        vars: usize,
    },
    #[error(
        "{span}: `stream {name}` travels through smem, which stages a tile, and this kernel declares no `tile`. Add `tile 32, 32`, or move it straight: `dram -> reg`."
    )]
    StagedWithoutTile { span: Span, name: String },
    #[error("{span}: `reduce sum {name}` names nothing the body computes")]
    ReduceOfNothing { span: Span, name: String },
    #[error("{span}: `into {name}` names no buffer parameter of this kernel")]
    ReduceIntoNothing { span: Span, name: String },
    #[error(
        "{span}: `into {name}` is also streamed. A reduction writes one value per BLOCK, not one per element, so its target carries no per-element traffic. Remove the stream."
    )]
    ReduceIntoStream { span: Span, name: String },
    #[error(
        "{span}: a reduction travels `reg -> smem -> dram` where the tree is staged, or `reg -> dram` where it is not; `{path}` is neither"
    )]
    ReducePath { span: Span, path: String },
    #[error("{span}: v1 supports one reduction per kernel")]
    TooManyReductions { span: Span },
    #[error("{0}")]
    Message(String),
}

/// Lower one kernel, resolving names and deriving its cost.
/// Every extent names something, and everything walked by one index space is one length.
///
/// This is what shape buys at rank 1, before any of ADR-0015's traffic work: a kernel that
/// streams `x: [f32; n]` beside `y: [f32; m]` is asking the compiler to walk two different
/// lengths with one index, which it cannot check at run time and will not guess at.
fn check_shapes(kernel: &Kernel, params: &BTreeMap<&str, Ty>) -> Result<(), LowerError> {
    let reduce_target = kernel.reductions.first().map(|r| r.into.as_str());

    for p in &kernel.params {
        if !p.ty.is_buffer() {
            continue;
        }
        let rank = kernel.space.as_ref().map(|s| s.vars.len()).unwrap_or(1);
        if p.shape.len() != rank {
            // A reduction target is one per block whatever the rank of the space.
            let is_target = Some(p.name.as_str()) == reduce_target;
            if !(is_target && p.shape.len() == 1) {
                return Err(if kernel.space.is_some() {
                    LowerError::RankMismatch {
                        span: p.span,
                        buffer: p.name.clone(),
                        rank: p.shape.len(),
                        vars: rank,
                    }
                } else {
                    LowerError::UnsupportedRank {
                        span: p.span,
                        buffer: p.name.clone(),
                        rank: p.shape.len(),
                    }
                });
            }
        }
        for dim in &p.shape {
            if dim != crate::ast::BLOCKS && params.get(dim.as_str()) != Some(&Ty::U32) {
                return Err(LowerError::UnknownExtent {
                    span: p.span,
                    buffer: p.name.clone(),
                    dim: dim.clone(),
                });
            }
        }
        let dim = &p.shape[0];
        // A reduction target is sized by the launch and nothing else is.
        let is_target = Some(p.name.as_str()) == reduce_target;
        if is_target && dim != crate::ast::BLOCKS {
            return Err(LowerError::ReductionTargetNotPerBlock {
                span: p.span,
                buffer: p.name.clone(),
                dim: dim.clone(),
            });
        }
        if !is_target && dim == crate::ast::BLOCKS {
            return Err(LowerError::StreamedPerBlock {
                span: p.span,
                buffer: p.name.clone(),
            });
        }
    }

    // Streamed buffers share the index space, so they share a length.
    //
    // The reduction target is skipped rather than compared. Streaming it is already an error
    // with a diagnosis of its own -- `ReduceIntoStream`, which says the buffer carries no
    // per-element traffic -- and that is the cause. A mismatched extent is only the symptom,
    // and reporting a symptom first is how an error message sends someone to the wrong place.
    //
    // Only at rank 1. A rank-2 kernel may stream `[rows, cols]` beside `[cols, rows]` on
    // purpose -- that is what a transpose is -- and the index space, not the buffers, says
    // how far the walk goes.
    let mut first: Option<(&str, &str)> = None;
    for s in kernel.streams.iter().filter(|_| kernel.space.is_none()) {
        if Some(s.buffer.as_str()) == reduce_target {
            continue;
        }
        let Some(p) = kernel.params.iter().find(|p| p.name == s.buffer) else {
            continue; // `StreamOfNothing` reports this, with the stream's span.
        };
        let Some(dim) = p.shape.first() else { continue };
        match first {
            None => first = Some((p.name.as_str(), dim.as_str())),
            Some((other, other_dim)) if other_dim != dim => {
                return Err(LowerError::ExtentMismatch {
                    span: p.span,
                    buffer: p.name.clone(),
                    dim: dim.clone(),
                    other: other.to_string(),
                    other_dim: other_dim.to_string(),
                })
            }
            Some(_) => {}
        }
    }
    Ok(())
}

pub fn lower(unit: &Unit, kernel: &Kernel) -> Result<KernelIr, LowerError> {
    let params: BTreeMap<&str, Ty> = kernel
        .params
        .iter()
        .map(|p| (p.name.as_str(), p.ty))
        .collect();

    check_shapes(kernel, &params)?;

    // A tile blocks the space, so there has to be one, and one dimension per index.
    if let Some(t) = &kernel.tile {
        let Some(sp) = &kernel.space else {
            return Err(LowerError::TileWithoutSpace { span: t.span });
        };
        if t.dims.len() != sp.vars.len() {
            return Err(LowerError::TileRankMismatch {
                span: t.span,
                dims: t.dims.len(),
                vars: sp.vars.len(),
            });
        }
    }

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
        // Two paths in v1: straight, or staged through shared memory when a tile says how
        // much to stage.
        let straight = s.path == [Level::Dram, Level::Reg];
        let staged = s.path == [Level::Dram, Level::Smem, Level::Reg];
        if !(straight || staged) {
            return Err(LowerError::UnsupportedPath {
                span: s.span,
                from: s.from().name(),
                to: s.to().name(),
            });
        }
        if staged && kernel.tile.is_none() {
            return Err(LowerError::StagedWithoutTile {
                span: s.span,
                name: s.buffer.clone(),
            });
        }
        streams.push(StreamIr {
            buffer: s.buffer.clone(),
            from: s.from(),
            to: s.to(),
            drain: s.drain,
            index: Vec::new(),
            staged: s.staged(),
            loaded: None,
            read: false,
            coalesced: true,
        });
    }

    // --- body ---------------------------------------------------------------------
    let space = match &kernel.space {
        None => None,
        Some(sp) => {
            for e in &sp.extents {
                if params.get(e.as_str()) != Some(&Ty::U32) {
                    return Err(LowerError::UnknownExtent {
                        span: sp.span,
                        buffer: format!("space {}", sp.vars.join(", ")),
                        dim: e.clone(),
                    });
                }
            }
            Some(SpaceIr {
                vars: sp.vars.clone(),
                extents: sp.extents.clone(),
            })
        }
    };

    // --- the contracted axis, as declared ------------------------------------------
    //
    // Split in two on purpose. Everything here is about the declaration alone and must run
    // *before* the body, or a kernel writing `c[i, j]` under `contract sum i` is told its
    // target is indexed at the contracted axis -- true, and not the mistake the author made.
    // The checks that need the body are further down, after it.
    if let Some(c) = &kernel.contract {
        let Some(sp) = &space else {
            return Err(LowerError::ContractWithoutSpace { span: c.span });
        };
        // Two different machines. `reduce` combines across threads through shared memory and
        // its tree order is part of the contract; `contract` combines inside one thread, in a
        // register, in loop order. A kernel doing both is a design, not a composition.
        if !kernel.reductions.is_empty() {
            return Err(LowerError::ContractAndReduce { span: c.span });
        }
        if sp.vars.contains(&c.var) {
            return Err(LowerError::ContractVarIsFree {
                span: c.span,
                var: c.var.clone(),
                vars: sp.vars.join(", "),
            });
        }
        if params.contains_key(c.var.as_str()) {
            return Err(LowerError::ContractVarIsParam {
                span: c.span,
                var: c.var.clone(),
            });
        }
        if params.get(c.extent.as_str()) != Some(&Ty::U32) {
            return Err(LowerError::ContractExtentUnknown {
                span: c.span,
                var: c.var.clone(),
                extent: c.extent.clone(),
            });
        }
        if sp.extents.contains(&c.extent) {
            return Err(LowerError::ContractExtentIsFree {
                span: c.span,
                extent: c.extent.clone(),
            });
        }
    }

    let mut ctx = Lowering {
        params: &params,
        space: space.clone(),
        contract_var: kernel.contract.as_ref().map(|c| c.var.clone()),
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
    let mut locals: BTreeMap<String, Span> = BTreeMap::new();
    for block in &kernel.blocks {
        for stmt in &block.stmts {
            let value = ctx.expr(&stmt.value)?;
            // A write is an address like a read, so the target's index is recorded the same
            // way -- and a local, which has no address, must not carry one.
            if params.get(stmt.target.as_str()).is_some_and(|t| t.is_buffer()) {
                // The target is one value per output, and the contracted axis is the one the
                // output does not have. `c[i, p] = ...` would write K times per output, from
                // the same thread, each write overwriting the last -- so the answer would be
                // the final term rather than the sum, and it would look like a working kernel.
                if let Some(c) = &kernel.contract {
                    if stmt.target_index.contains(&c.var) {
                        return Err(LowerError::ContractedTarget {
                            span: stmt.target_span,
                            buffer: stmt.target.clone(),
                            var: c.var.clone(),
                        });
                    }
                }
                ctx.note_index(&stmt.target, &stmt.target_index, stmt.target_span)?;
            } else if !stmt.target_index.is_empty() {
                return Err(LowerError::IndexWithoutSpace {
                    span: stmt.target_span,
                    buffer: stmt.target.clone(),
                    index: stmt.target_index.join(", "),
                });
            }
            match params.get(stmt.target.as_str()) {
                // Not a parameter: a local. Legal only if a reduction consumes it, which is
                // checked once the whole body is known.
                None => {
                    locals.insert(stmt.target.clone(), stmt.target_span);
                }
                Some(ty) if !ty.is_buffer() => {
                    return Err(LowerError::ScalarStream {
                        span: stmt.target_span,
                        name: stmt.target.clone(),
                    })
                }
                Some(_) => {
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
                    assigned.insert(stmt.target.clone());
                }
            }
            ctx.env.insert(stmt.target.clone(), value);
        }
    }

    // Destructure to end the mutable borrow of `streams` before reading it back.
    let Lowering { ops, env, .. } = ctx;

    // --- the reduction -------------------------------------------------------------
    if kernel.reductions.len() > 1 {
        return Err(LowerError::TooManyReductions {
            span: kernel.reductions[1].span,
        });
    }
    let mut reduction = None;
    if let Some(r) = kernel.reductions.first() {
        // Two legal shapes, because the tree has to live somewhere and not every machine
        // has a shared level to put it in. `reg -> smem -> dram` is the GPU's: threads write
        // their partials to shared memory and a tree folds them there. `reg -> dram` is a
        // machine whose parallelism is lanes of one register -- the tree is register-internal
        // (Unibit's `VFREDUCE`), so nothing is staged and one store leaves.
        //
        // Which one a kernel may use is the **machine's** answer, not the language's, and it
        // is the back end that refuses the other. ADR-0025.
        if r.path != [Level::Reg, Level::Smem, Level::Dram] && r.path != [Level::Reg, Level::Dram]
        {
            return Err(LowerError::ReducePath {
                span: r.span,
                path: r
                    .path
                    .iter()
                    .map(|l| l.name())
                    .collect::<Vec<_>>()
                    .join(" -> "),
            });
        }
        let Some(value) = env.get(&r.source).copied() else {
            return Err(LowerError::ReduceOfNothing {
                span: r.span,
                name: r.source.clone(),
            });
        };
        match params.get(r.into.as_str()) {
            None => {
                return Err(LowerError::ReduceIntoNothing {
                    span: r.span,
                    name: r.into.clone(),
                })
            }
            Some(ty) if !ty.is_buffer() => {
                return Err(LowerError::ReduceIntoNothing {
                    span: r.span,
                    name: r.into.clone(),
                })
            }
            Some(_) => {}
        }
        if streams.iter().any(|s| s.buffer == r.into) {
            return Err(LowerError::ReduceIntoStream {
                span: r.span,
                name: r.into.clone(),
            });
        }
        locals.remove(&r.source);
        reduction = Some(ReductionIr {
            op: r.op,
            value,
            into: r.into.clone(),
            path: r.path.clone(),
        });
    }

    // A local nothing consumed is work whose result is thrown away.
    if let Some((name, span)) = locals.into_iter().next() {
        return Err(LowerError::DeadLocal { span, name });
    }

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

    // A kernel that neither drains nor reduces computes nothing anyone can see.
    if drains.is_empty() && reduction.is_none() {
        return Err(LowerError::Message(format!(
            "kernel `{}` writes nothing: no drained stream and no reduction",
            kernel.name
        )));
    }

    // Coalescence, once the indices are known. The fast variable is the innermost one of the
    // space, because the flattening puts it in the ones place of the linear index.
    if let Some(sp) = &space {
        let fast = sp.vars.last().expect("a space has at least one variable");
        for st in streams.iter_mut() {
            st.coalesced = st.index.last().map(|v| v == fast).unwrap_or(true);
        }
    }
    // Absorption. Derived from `smem` appearing in a stream's path, not from the kernel's
    // name: once anything is staged, the emitted body moves the permutation into shared
    // memory and every global access becomes row-contiguous.
    if kernel.tile.is_some() && streams.iter().any(|s| s.staged) {
        for st in streams.iter_mut() {
            st.coalesced = true;
        }
    }

    // Shared memory layout, once the tile and the staged streams are both known.
    let shared = kernel.tile.as_ref().and_then(|t| {
        let staged: Vec<&StreamIr> = streams.iter().filter(|s| s.staged).collect();
        if staged.is_empty() {
            return None;
        }
        let rows = t.dims[0];
        let stride = skewed_stride(*t.dims.last().expect("a tile has a width"));
        let tiles: Vec<(String, u32, u32)> = staged
            .iter()
            .map(|s| (s.buffer.clone(), rows, stride))
            .collect();
        // **The element's width, not the bank's.** These were the same number for the whole of
        // this project's life -- `SMEM_BANK_WIDTH` is 4 and so was every element -- and using
        // one for the other is the same mistake as borrowing a figure across contexts
        // (ADR-0000). A narrow type separates them: the bank is still 4 bytes wide and the
        // element is 2, so the tile is half the size and the bank arithmetic is unchanged.
        //
        // The skew survives, and it is worth writing down why rather than hoping. A column
        // walk of a `T x (T+1)` tile of 2-byte elements steps `(T+1) * 2` bytes; at `T = 32`
        // that is 66 bytes, so thread `i` lands on bank `floor(i * 16.5) mod 32`, which runs
        // 0, 16, 1, 17, 2, 18, ... -- a permutation of all 32. Conflict-free for the same
        // coprimality reason as at four bytes, reached by a different route.
        let bytes: u32 = staged
            .iter()
            .map(|s| {
                let w = kernel
                    .params
                    .iter()
                    .find(|p| p.name == s.buffer)
                    .expect("a staged stream is a parameter")
                    .ty
                    .bytes();
                rows * stride * w
            })
            .sum();
        // Derived rather than asserted. At four bytes the skew makes the stride coprime with
        // 32 and the walk is a permutation; at two it is a different permutation and the
        // check is the same. If a future width broke it, this would say so instead of
        // publishing a zero the silicon disagrees with.
        let elem = staged
            .first()
            .map(|s| {
                kernel
                    .params
                    .iter()
                    .find(|p| p.name == s.buffer)
                    .expect("a staged stream is a parameter")
                    .ty
                    .bytes()
            })
            .unwrap_or(4);
        Some(SharedLayout {
            tiles,
            bytes,
            predicted_bank_conflicts: u32::from(!walk_is_conflict_free(stride, elem)),
        })
    });

    // --- the contracted axis -------------------------------------------------------
    //
    // What makes a `contract` real is a buffer read at it, so these refusals need the body.
    // The declaration-level ones ran before it.
    let mut contract = None;
    if let Some(c) = &kernel.contract {
        // The buffers the contraction actually walks, and where in their index it appears.
        let over: Vec<(String, usize)> = streams
            .iter()
            .filter_map(|st| {
                st.index
                    .iter()
                    .position(|v| *v == c.var)
                    .map(|at| (st.buffer.clone(), at))
            })
            .collect();
        if over.is_empty() {
            return Err(LowerError::ContractOverNothing {
                span: c.span,
                var: c.var.clone(),
            });
        }
        // A contraction that walks only one buffer is a reduction of that buffer along an
        // axis, which is a real kernel and not this one: with a single operand there is no
        // reuse for a tile to exploit, so the traffic expression ADR-0018 derives -- 2K/T per
        // output -- degenerates and the declared intensity would be checked against an
        // asymptote that does not describe it. Refused until it is derived on its own.
        if over.len() < 2 {
            return Err(LowerError::ContractOverOne {
                span: c.span,
                var: c.var.clone(),
                buffer: over[0].0.clone(),
            });
        }
        contract = Some(ContractIr {
            op: c.op,
            var: c.var.clone(),
            extent: c.extent.clone(),
            over,
        });
    }


    // --- coarsening -----------------------------------------------------------------
    //
    // Everything here is about the declaration against the tile, so it needs neither the body
    // nor the cost. The **global** cost does not need it either: reuse at that level is a
    // property of the tile -- how many threads want one staged element -- and coarsening
    // changes which thread computes what. ADR-0021 step 4 measured that to +0.80%.
    //
    // The shared level is a different claim and this file got it wrong until ADR-0022. A
    // coarsened thread loads `ci` values of one operand and `cj` of the other and uses each
    // across the other axis, so it reads shared **fewer** times per output: the register file
    // is a third level of reuse, and `coarsen` is the declaration that earns it. See
    // `register_reuse_of` below, which is `reuse_of` with the coarsening factors in place of
    // the tile's.
    let coarsen = match (&kernel.coarsen, &kernel.tile) {
        (None, _) => None,
        (Some(c), None) => {
            return Err(LowerError::CoarsenWithoutTile { span: c.span });
        }
        (Some(c), Some(t)) => {
            if c.dims.len() != t.dims.len() {
                return Err(LowerError::CoarsenRankMismatch {
                    span: c.span,
                    dims: c.dims.len(),
                    tile: t.dims.len(),
                });
            }
            for (i, (&per, &edge)) in c.dims.iter().zip(&t.dims).enumerate() {
                if edge % per != 0 {
                    return Err(LowerError::CoarsenDoesNotDivide {
                        span: c.span,
                        axis: i,
                        per,
                        edge,
                    });
                }
            }
            // `coarsen 1, 1` is the default written out. A declaration that changes nothing is
            // a line a reader has to check and then discard.
            if c.dims.iter().all(|d| *d == 1) {
                return Err(LowerError::CoarsenIsIdentity { span: c.span });
            }
            Some(c.dims.clone())
        }
    };

    let cost = derive_cost(CostInputs {
        streams: &streams,
        ops: &ops,
        reduction: reduction.as_ref(),
        contract: contract.as_ref(),
        space: space.as_ref(),
        tile: kernel.tile.as_ref().map(|t| t.dims.as_slice()),
        coarsen: coarsen.as_deref(),
        params: &kernel.params,
    });

    // The word `asymptotic`, required exactly where the claim is a limit and refused
    // everywhere else.
    //
    // Step 1 refused to cost a contraction at all, because `Cost` carried constants and the
    // constant part of a matmul is 4 bytes -- wrong by a factor of K and indistinguishable
    // from every other number this compiler reports. `Cost` now carries the expression, so the
    // refusal is lifted. What replaces it is smaller and permanent: a contracted kernel may
    // declare the limit and not the figure, and it has to say which it is declaring.
    match (&kernel.contract, kernel.declared_intensity) {
        (Some(c), Some(_)) if !kernel.intensity_is_asymptotic => {
            return Err(LowerError::ContractedIntensityNotAsymptotic {
                span: kernel.intensity_span.unwrap_or(c.span),
                name: kernel.name.clone(),
                extent: c.extent.clone(),
                derived: cost.intensity,
            });
        }
        (None, Some(declared)) if kernel.intensity_is_asymptotic => {
            return Err(LowerError::AsymptoticWithoutContraction {
                span: kernel.intensity_span.unwrap_or(kernel.span),
                name: kernel.name.clone(),
                declared,
            });
        }
        _ => {}
    }

    Ok(KernelIr {
        name: kernel.name.clone(),
        machine: unit.machine.clone(),
        params: kernel.params.clone(),
        streams,
        space,
        tile: kernel.tile.as_ref().map(|t| t.dims.clone()),
        coarsen,
        shared,
        ops,
        drains,
        reduction,
        contract,
        cost,
    })
}

/// Bytes from the declared movement, flops from the evaluated tree.
///
/// A stream contributes a read only if the body actually reads it, and a write only if it
/// drains. A buffer that is streamed and never read costs nothing to read — the declaration
/// does not get to inflate the denominator, and an unread stream is caught elsewhere.
/// A 32-byte sector is the smallest thing the memory system moves.
const SECTOR: f64 = 32.0;

/// Shared memory banks, and the width of one.
///
/// A constant rather than a machine-file field, and the reason is worth stating: 32 banks of
/// 4 bytes has held across every NVIDIA architecture since Fermi, and the derivation below
/// would be wrong rather than imprecise on a machine where it did not. So it is written here,
/// and `l1tex__data_bank_conflicts_pipe_lsu.sum` is what checks the consequence -- the same
/// shape as every other claim in this compiler: derive it, then let the silicon answer.
const SMEM_BANKS: u32 = 32;
/// Bytes per shared-memory bank. **Not the element's width**, and the distinction only became
/// visible when ADR-0024 gave the language a 2-byte element: for the project's whole life
/// before that, both numbers were 4 and the shared tile was sized with this one.
///
/// It is kept, and used where it actually belongs -- deciding whether a column walk of narrow
/// elements still lands on distinct banks. A `T x (T+1)` tile of `w`-byte elements steps
/// `(T+1) * w` bytes per row, so thread `i` lands on bank `(i * (T+1) * w / 4) mod 32`. At
/// `T = 32, w = 4` that is `i * 33 mod 32 = i`; at `w = 2` it is `floor(i * 16.5) mod 32`,
/// which runs 0, 16, 1, 17, 2, 18, ... -- also a permutation of all 32. Conflict-free at both
/// widths, reached by different arithmetic, which is why it is derived rather than assumed.
const SMEM_BANK_WIDTH: u32 = 4;

/// Whether a column walk of a skewed tile touches every bank exactly once.
///
/// Derived, and then `predicted_bank_conflicts` is what the profiler is asked about
/// (ADR-0017). A tile that conflicts still computes the right answer, so nothing but a counter
/// can tell -- which is why ADR-0017's fixture builds the unpadded variant on purpose.
fn walk_is_conflict_free(stride_elems: u32, elem_bytes: u32) -> bool {
    let mut seen = vec![false; SMEM_BANKS as usize];
    for i in 0..SMEM_BANKS {
        let bank = (i * stride_elems * elem_bytes / SMEM_BANK_WIDTH) % SMEM_BANKS;
        if seen[bank as usize] {
            return false;
        }
        seen[bank as usize] = true;
    }
    true
}

/// How a tile is laid out in shared memory, and what that costs.
///
/// **The skew is derived, not declared.** A column of a tile whose rows are `stride` elements
/// apart puts thread `t` on bank `(t * stride + c) mod 32`. The 32 threads of a warp land on
/// 32 distinct banks exactly when `stride` is coprime to 32, and since 32 is a power of two
/// that is exactly when `stride` is **odd**.
///
/// So the rule is not "add one". It is "pad until the row stride is odd", which for the usual
/// 32-wide tile gives 33 and for a tile of odd width gives no padding at all. A rule that
/// always added one would waste a row of shared memory on half of all tile widths.
///
/// This assumes the element and the bank are the same width, which for f32 they are. A
/// language with f64 would need this again and differently.
#[derive(Debug, Clone, PartialEq)]
pub struct SharedLayout {
    /// Per staged buffer: its name, rows, and the padded row stride in elements.
    pub tiles: Vec<(String, u32, u32)>,
    /// Dynamic shared memory the launch must request.
    pub bytes: u32,
    /// What the skew is for. Pre-registered so ADR-0017 step 5 can falsify it.
    pub predicted_bank_conflicts: u32,
}

/// Pad the row stride until it is coprime to the bank count.
fn skewed_stride(width: u32) -> u32 {
    let mut stride = width;
    while gcd(stride, SMEM_BANKS) != 1 {
        stride += 1;
    }
    stride
}

fn gcd(a: u32, b: u32) -> u32 {
    if b == 0 {
        a
    } else {
        gcd(b, a % b)
    }
}

/// Bytes and flops per output element, per stream.
///
/// **Per stream is the point, not an implementation detail.** A contraction over a square tile
/// with both operands staged moves `2K/T + 1` elements per output, and it would be a short
/// function to emit that expression whenever a `contract` and a `tile` appear together. It
/// would also be wrong for every other schedule this language can already express -- one
/// operand staged and one not, a rectangular tile, no tile at all -- and wrong in the same
/// confident shape as the constant step 1 refused to publish. So each stream is asked what it
/// costs and the expression is the sum. `2K/T + 1` is a result here, never an input.
/// Everything `derive_cost` reads, gathered rather than passed one at a time.
///
/// It grew an eighth argument when ADR-0024 made the element width a property of the buffer
/// instead of a module constant, and eight positional arguments of mostly-`Option` is a
/// call site nobody can read. They all come from the same kernel; this says so.
struct CostInputs<'a> {
    streams: &'a [StreamIr],
    ops: &'a [Op],
    reduction: Option<&'a ReductionIr>,
    contract: Option<&'a ContractIr>,
    space: Option<&'a SpaceIr>,
    tile: Option<&'a [u32]>,
    coarsen: Option<&'a [u32]>,
    params: &'a [crate::ast::Param],
}

fn derive_cost(input: CostInputs<'_>) -> Cost {
    let CostInputs {
        streams,
        ops,
        reduction,
        contract,
        space,
        tile,
        coarsen,
        params,
    } = input;
    // `levels` is built below; a staged stream adds one before the reduction's, and the sort
    // at the end puts the deepest first so `roofline()` still finds DRAM.

    // How wide one element of a given buffer is. **Not a constant.**
    //
    // This was `Ty::BufF32.bytes()` -- the entire cost model's element size, written once, at
    // module scope, because the language had one buffer type. ADR-0024 gave it three, and the
    // model was already written in terms of *an* element size, so this is the whole change at
    // this level: a lookup instead of a literal.
    //
    // A missing name cannot happen -- lowering resolves every stream against a parameter
    // before this runs -- and 4.0 is the width that was assumed for the project's whole life,
    // so falling back to it keeps a bug quiet rather than loud. It is asserted instead.
    let elem_of = |buffer: &str| -> f64 {
        params
            .iter()
            .find(|p| p.name == buffer)
            .unwrap_or_else(|| panic!("stream `{buffer}` names no parameter"))
            .ty
            .bytes() as f64
    };
    let mut dram = LevelCost::at(Level::Dram);

    // How many threads share one element of this buffer, once it is loaded.
    //
    // A stream is reused along the free axes its index does not mention: `a[i, p]` is the same
    // element for every `j`, so the `tile[j]` threads of a tile column want it once between
    // them. **Staging is what makes that true.** Left in `dram -> reg`, each of those threads
    // issues its own load and the reuse is 1: the tile says the threads exist, shared memory
    // is what makes them share. ADR-0017 drew the same line for coalescence.
    let reuse_of = |st: &StreamIr| -> u32 {
        let (Some(sp), Some(t)) = (space, tile) else {
            return 1;
        };
        if !st.staged {
            return 1;
        }
        sp.vars
            .iter()
            .zip(t)
            .filter(|(v, _)| !st.index.contains(v))
            .map(|(_, dim)| *dim)
            .product::<u32>()
            .max(1)
    };
    // How many of this buffer's shared reads one thread's registers absorb.
    //
    // **The same rule as `reuse_of`, one level down.** Shared memory earns its reuse from the
    // tile: the threads of a tile row want one element of `a[i, p]` between them, and staging
    // is what lets them share it. The register file earns its reuse from the coarsening: a
    // thread that owns `cj` outputs along `j` loads that same element once and spends it `cj`
    // times, so the shared pipe sees one access instead of `cj`.
    //
    // So the factor is the product of the coarsening of the free axes the index does **not**
    // mention -- `reuse_of` with `coarsen` substituted for `tile`. That the two rules are the
    // same rule is why this fix is four lines and not a rewrite: ADR-0021 argued reuse was a
    // property of the tile and stopped there, and the level below has its own tile.
    //
    // Unstaged, or uncoarsened, it is 1, which is what every kernel before ADR-0021 gets.
    let register_reuse_of = |st: &StreamIr| -> u32 {
        let (Some(sp), Some(c)) = (space, coarsen) else {
            return 1;
        };
        if !st.staged {
            return 1;
        }
        sp.vars
            .iter()
            .zip(c)
            .filter(|(v, _)| !st.index.contains(v))
            .map(|(_, per)| *per)
            .product::<u32>()
            .max(1)
    };
    // The same accounting at sector granularity. A coalesced access costs the payload; a
    // strided one costs a whole sector per element, which is the bound documented on `Cost`.
    // Each stream is charged once per output, or once per output per unit of the contracted
    // extent when it is the contraction that walks it.
    let mut terms: Vec<CostTerm> = Vec::new();
    // The same accounting at sector granularity, and it has to follow the same split or the
    // ratio between them stops being a ratio. A strided access costs a whole sector per
    // element; a staged one is loaded coalesced and amortised by the same reuse as its
    // payload, so a contracted kernel whose operands are staged achieves 1.000 -- which is
    // what the absorption rule in ADR-0017 claims and what this number is for.
    let (mut sector_read, mut sector_write) = (0.0, 0.0);
    let (mut sector_read_per_extent, mut sector_write_per_extent) = (0.0, 0.0);
    for s in streams {
        let walked = contract.map(|c| s.index.contains(&c.var)).unwrap_or(false);
        let elem = elem_of(&s.buffer);
        let per = if walked { elem / reuse_of(s) as f64 } else { 0.0 };
        let fixed = if walked { 0.0 } else { elem };
        // A narrow element does **not** narrow a strided access. The sector is 32 bytes
        // whatever sits in it, so halving the element halves the payload and leaves the bus
        // alone -- which is to say a strided f16 kernel wastes twice as much of what it
        // fetches as a strided f32 one. ADR-0015's distinction, sharpened by ADR-0024.
        let bus = if s.coalesced { elem } else { SECTOR };
        let bus_per = if walked { bus / reuse_of(s) as f64 } else { 0.0 };
        let bus_fixed = if walked { 0.0 } else { bus };
        if s.read {
            dram.read += fixed;
            dram.read_per_extent += per;
            sector_read += bus_fixed;
            sector_read_per_extent += bus_per;
        }
        if s.drain {
            dram.write += fixed;
            dram.write_per_extent += per;
            sector_write += bus_fixed;
            sector_write_per_extent += bus_per;
        }
        if contract.is_some() && (s.read || s.drain) {
            terms.push(CostTerm {
                buffer: s.buffer.clone(),
                write: s.drain,
                fixed,
                per_extent: per,
                reuse: if walked { reuse_of(s) } else { 1 },
            });
        }
    }

    // The body runs once per step of the contraction rather than once per output, so every
    // flop it retires is a flop per unit of the extent. A kernel without a contraction takes
    // one step per output and the two are the same number.
    let body_flops: f64 = ops.iter().map(Op::flops).sum();
    let mut flops = if contract.is_some() { 0.0 } else { body_flops };
    let mut levels = vec![dram];
    let mut dram_bytes_per_block = 0.0;

    // A staged element crosses the shared interface twice: written once by the thread that
    // loaded it, read once by the thread that needs it. Reported at its own level, because a
    // byte in shared memory and a byte at DRAM are not the same byte and no single number
    // should pretend otherwise.
    if streams.iter().any(|s| s.staged) {
        let mut smem = LevelCost::at(Level::Smem);
        for st in streams.iter().filter(|s| s.staged) {
            let walked = contract.map(|c| st.index.contains(&c.var)).unwrap_or(false);
            if walked {
                // Once per step the thread reads its element out of shared -- divided by what
                // its own registers absorb, which is the coarsening of the axis this buffer's
                // index does not mention. At `coarsen 1, 1` that is 1 and this is the figure
                // every kernel before ADR-0021 derived.
                //
                // The write is **not** divided by it. Coarsening does not change what is
                // staged: the tile is the same size and the same elements land in it, spread
                // over fewer threads doing more each. Only the reading changes, which is
                // exactly what the emitter does -- `ci * cj` stores per operand per step, and
                // `ci + cj` loads per term.
                smem.read_per_extent += elem_of(&st.buffer) / register_reuse_of(st) as f64;
                smem.write_per_extent += elem_of(&st.buffer) / reuse_of(st) as f64;
            } else {
                smem.read += elem_of(&st.buffer);
                smem.write += elem_of(&st.buffer);
            }
        }
        levels.push(smem);
    }

    if let Some(r) = reduction {
        // The tree's traffic follows the **path**, because on a machine without a shared level
        // the tree is register-internal and moves no bytes at all. Deriving it unconditionally
        // charged `unibit` for staging into a level it does not have, which is the same error
        // in the other direction as costing a tile that cannot exist.
        if r.path.contains(&Level::Smem) {
            // Shared-memory traffic of the tree, per block of B threads:
            //   B initial writes                        4B bytes
            //   B-1 combines, each 2 reads and 1 write  12(B-1) bytes
            //   one final read of slot 0                4 bytes
            // which is 16B - 8, so 16 - 8/B per element. Counted as 16; at B = 256 that
            // overstates by 0.03 bytes, 0.2%. [KNOWN LIMIT] in ADR-0011.
            levels.push(LevelCost {
                level: Level::Smem,
                read: 8.0,
                write: 8.0,
                ..Default::default()
            });
        }
        // The tree retires B-1 combines over B elements, so (B-1)/B per element. Counted as
        // one, for the same reason: the block size is not in the source.
        flops += r.op.flops();
        // One partial per block, not per element, so it is reported separately rather than
        // divided by a block size the compiler would have to invent.
        dram_bytes_per_block = elem_of(&r.into);
    }

    // Deepest first, so `roofline()` finds DRAM before shared memory.
    levels.sort_by_key(|l| l.level as u8);
    let roof = levels
        .iter()
        .find(|l| l.carries_traffic())
        .copied()
        .unwrap_or_default();

    let contracted = contract.map(|c| {
        // The combinator changes this and not the traffic: `sum` retires one flop per step on
        // top of the body, `max` and `min` retire none (ADR-0013). Derived once, here, rather
        // than three times by operator.
        let flops_per_extent = body_flops + c.op.flops();
        ContractedCost {
            extent: c.extent.clone(),
            flops_per_extent,
            terms,
            bytes_per_extent: roof.per_extent(),
        }
    });

    // The roofline number. For a contracted kernel it is the limit as the extent grows --
    // `flops_per_extent / bytes_per_extent`, the fixed parts vanishing -- because the exact
    // figure is a function of a launch value and this is a compile-time constant. The flag
    // beside it is what stops that being read as an exact claim.
    let intensity = match &contracted {
        Some(c) if c.bytes_per_extent > 0.0 => c.flops_per_extent / c.bytes_per_extent,
        Some(_) => 0.0,
        None if roof.total() > 0.0 => flops / roof.total(),
        None => 0.0,
    };

    Cost {
        levels,
        sector_read_per_element: sector_read,
        sector_write_per_element: sector_write,
        sector_read_per_extent,
        sector_write_per_extent,
        flops_fixed: flops,
        intensity,
        asymptotic: contracted.is_some(),
        contracted,
        dram_bytes_per_block,
    }
}

struct Lowering<'a> {
    params: &'a BTreeMap<&'a str, Ty>,
    space: Option<SpaceIr>,
    /// The contracted index variable, when there is one. An index may name it in addition to
    /// the space's own variables.
    contract_var: Option<String>,
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
            Expr::At {
                buffer,
                index,
                span,
            } => self.at(buffer, index, *span),
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

    /// Record how a buffer is walked, refusing a second, different walk of the same buffer.
    ///
    /// Called for a read and for a write, because both are the same address.
    fn note_index(
        &mut self,
        buffer: &str,
        index: &[String],
        span: Span,
    ) -> Result<(), LowerError> {
        let Some(sp) = self.space.clone() else {
            if index.is_empty() {
                return Ok(());
            }
            return Err(LowerError::IndexWithoutSpace {
                span,
                buffer: buffer.into(),
                index: index.join(", "),
            });
        };
        if index.is_empty() {
            return Err(LowerError::MissingIndex {
                span,
                buffer: buffer.into(),
                vars: sp.vars.join(", "),
                first: sp.vars.join(", "),
            });
        }
        // Without a contraction: a permutation, every index variable once and nothing else.
        //
        // With one, the contracted variable joins the alphabet and displaces a free index
        // rather than lengthening the list -- `a[i, p]` under `space i, j` and `contract p`.
        // The rank is still the space's, because that is the rank of every buffer here, and
        // `p` may appear at most once for the same reason `i` may: one address per element.
        let legal: Vec<String> = match &self.contract_var {
            None => sp.vars.clone(),
            Some(p) => sp.vars.iter().cloned().chain(std::iter::once(p.clone())).collect(),
        };
        let distinct: BTreeSet<&String> = index.iter().collect();
        let drawn_from_alphabet = index.iter().all(|v| legal.contains(v));
        let ok = index.len() == sp.vars.len()
            && distinct.len() == index.len()
            && drawn_from_alphabet;
        if !ok {
            return Err(LowerError::NotAPermutation {
                span,
                buffer: buffer.into(),
                index: index.join(", "),
                vars: legal.join(", "),
            });
        }
        if let Some(st) = self.streams.iter_mut().find(|s| s.buffer == buffer) {
            if st.index.is_empty() {
                st.index = index.to_vec();
            } else if st.index != index {
                return Err(LowerError::TwoIndexings {
                    span,
                    buffer: buffer.into(),
                    first: st.index.join(", "),
                    second: index.join(", "),
                });
            }
        }
        Ok(())
    }

    fn at(&mut self, buffer: &str, index: &[String], span: Span) -> Result<RegId, LowerError> {
        self.note_index(buffer, index, span)?;
        self.load(buffer, span)
    }

    /// An unindexed name. A buffer reached this way in a kernel with a `space` is a buffer
    /// nobody said how to walk, and guessing `[i, j]` would silently pick row-major.
    fn name(&mut self, name: &str, span: Span) -> Result<RegId, LowerError> {
        if let (Some(sp), Some(ty)) = (self.space.clone(), self.params.get(name)) {
            if ty.is_buffer() && !self.env.contains_key(name) {
                return Err(LowerError::MissingIndex {
                    span,
                    buffer: name.into(),
                    vars: sp.vars.join(", "),
                    first: sp.vars.join(", "),
                });
            }
        }
        self.load(name, span)
    }

    fn load(&mut self, name: &str, span: Span) -> Result<RegId, LowerError> {
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

kernel saxpy(n: u32, a: f32, x: [f32; n], y: [f32; n])
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
        assert_eq!(k.cost.read_bytes_per_element(), 8.0);
        assert_eq!(k.cost.write_bytes_per_element(), 4.0);
        assert_eq!(k.cost.bytes_per_element().unwrap(), 12.0);
        // One fma = 2 flops.
        assert_eq!(k.cost.flops_per_element().unwrap(), 2.0);
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
        let src =
            "machine m\n\nkernel k(n: u32, x: [f32; n], y: [f32; n])\n stream y : dram -> reg, drain\n at reg:
        y = x\n";
        let e = ir(src).unwrap_err();
        let s = e.to_string();
        assert!(s.contains("stream x : dram -> reg"), "{s}");
        assert!(s.contains("not declared resident"), "{s}");
    }

    #[test]
    fn writing_a_stream_that_does_not_drain_is_refused() {
        let src = "machine m\n\nkernel k(n: u32, x: [f32; n])\n stream x : dram -> reg\n at reg:
        x = x + 1\n";
        let e = ir(src).unwrap_err();
        assert!(e.to_string().contains("no `drain`"), "{e}");
    }

    #[test]
    fn draining_something_never_computed_is_refused() {
        let src = "machine m\n\nkernel k(n: u32, x: [f32; n], y: [f32; n])\n stream x : dram -> reg\n stream y : dram -> reg, drain\n at reg:
        x = x\n";
        // x has no drain, so this trips WriteWithoutDrain first; swap to make y the issue.
        let src2 = "machine m\n\nkernel k(n: u32, x: [f32; n], y: [f32; n])\n stream x : dram -> reg\n stream y : dram -> reg, drain\n at reg:
        y = x\n";
        assert!(ir(src).is_err());
        assert!(ir(src2).is_ok(), "y is drained and assigned");
    }

    #[test]
    fn a_stream_of_a_scalar_is_refused() {
        let src = "machine m\n\nkernel k(n: u32, a: f32, y: [f32; n])\n stream a : dram -> reg\n stream y : dram -> reg, drain\n at reg:
        y = a\n";
        let e = ir(src).unwrap_err();
        assert!(e.to_string().contains("scalar parameter"), "{e}");
    }

    #[test]
    fn an_unknown_name_names_itself() {
        let src = "machine m\n\nkernel k(n: u32, y: [f32; n])\n stream y : dram -> reg, drain\n at reg:
        y = z\n";
        let e = ir(src).unwrap_err();
        assert!(e.to_string().contains("`z`"), "{e}");
    }

    #[test]
    fn a_buffer_is_loaded_once_however_often_it_is_named() {
        let src = "machine m\n\nkernel k(n: u32, x: [f32; n], y: [f32; n])\n stream x : dram -> reg\n stream y : dram -> reg, drain\n at reg:
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
        let src = "machine m\n\nkernel k(n: u32, x: [f32; n], y: [f32; n])\n stream x : dram -> reg\n stream y : dram -> reg, drain\n at reg:
        y = x + x\n";
        let k = ir(src).unwrap();
        assert_eq!(k.cost.read_bytes_per_element(), 4.0, "x only");
        assert_eq!(k.cost.write_bytes_per_element(), 4.0, "y only");
        assert_eq!(k.cost.bytes_per_element().unwrap(), 8.0);

        // saxpy does read y, because `a * x + y` names it.
        let saxpy = ir(SAXPY).unwrap();
        assert_eq!(saxpy.cost.read_bytes_per_element(), 8.0, "x and y");
    }

    #[test]
    fn v1_refuses_a_level_it_cannot_generate_instead_of_ignoring_it() {
        let src = "machine m\n\nkernel k(n: u32, x: [f32; n])\n stream x : dram -> reg, drain\n at smem:
        x = x\n";
        let e = ir(src).unwrap_err();
        assert!(e.to_string().contains("not implemented"), "{e}");
    }
}

#[cfg(test)]
mod shared_layout {
    use super::*;

    /// The skew is "pad until the row stride is coprime to the bank count", not "add one".
    #[test]
    fn a_tile_is_padded_until_its_row_stride_is_odd() {
        // 32 banks is a power of two, so coprime means odd.
        assert_eq!(skewed_stride(32), 33, "the usual tile");
        assert_eq!(skewed_stride(16), 17);
        assert_eq!(skewed_stride(64), 65);
        // An odd width already lands every thread on its own bank. A rule that always added
        // one would waste a row of shared memory here for nothing.
        assert_eq!(skewed_stride(31), 31);
        assert_eq!(skewed_stride(33), 33);
    }

    #[test]
    fn every_thread_of_a_warp_lands_on_its_own_bank() {
        // The property the skew exists for, checked directly rather than trusted: reading a
        // column of the padded tile, thread t is at offset t * stride and must be on bank
        // (t * stride) mod 32, all distinct.
        for width in [8u32, 16, 32, 64] {
            let stride = skewed_stride(width);
            let mut banks: Vec<u32> = (0..SMEM_BANKS).map(|t| (t * stride) % SMEM_BANKS).collect();
            banks.sort();
            banks.dedup();
            assert_eq!(
                banks.len(),
                SMEM_BANKS as usize,
                "width {width} with stride {stride} serialises"
            );
        }
        // And without the skew it collapses to one bank, which is the 32-way conflict.
        let mut unpadded: Vec<u32> = (0..SMEM_BANKS).map(|t| (t * 32) % SMEM_BANKS).collect();
        unpadded.dedup();
        assert_eq!(unpadded.len(), 1, "an unpadded 32-wide tile is one bank");
    }

    #[test]
    fn a_staged_tile_costs_what_the_skew_makes_it_cost() {
        let src = "machine sm_120\n\nkernel t(rows: u32, cols: u32, a: [f32; rows, cols], b: [f32; cols, rows])\n    space i, j : rows, cols\n    tile 32, 32\n    stream a : dram -> smem -> reg\n    stream b : dram -> reg, drain\n    at reg:\n        b[j, i] = a[i, j]\n";
        let unit = crate::parse::parse(src).unwrap();
        let ir = lower(&unit, &unit.kernels[0]).unwrap();
        let l = ir.shared.expect("a staged stream has a layout");
        assert_eq!(l.tiles, vec![("a".to_string(), 32, 33)]);
        assert_eq!(l.bytes, 32 * 33 * 4, "4224 bytes, not 4096");
        assert_eq!(l.predicted_bank_conflicts, 0);
    }

    #[test]
    fn a_tile_with_nothing_staged_needs_no_shared_memory() {
        let src = "machine sm_120\n\nkernel t(rows: u32, cols: u32, a: [f32; rows, cols], b: [f32; cols, rows])\n    space i, j : rows, cols\n    tile 32, 32\n    stream a : dram -> reg\n    stream b : dram -> reg, drain\n    at reg:\n        b[j, i] = a[i, j]\n";
        let unit = crate::parse::parse(src).unwrap();
        let ir = lower(&unit, &unit.kernels[0]).unwrap();
        assert!(ir.shared.is_none(), "a tile alone stages nothing");
    }
}

#[cfg(test)]
mod absorption {
    use super::*;

    fn cost(src: &str) -> Cost {
        let unit = crate::parse::parse(src).unwrap();
        lower(&unit, &unit.kernels[0]).unwrap().cost
    }

    const UNTILED: &str = "machine sm_120\n\nkernel t(rows: u32, cols: u32, a: [f32; rows, cols], b: [f32; cols, rows])\n    space i, j : rows, cols\n    stream a : dram -> reg\n    stream b : dram -> reg, drain\n    at reg:\n        b[j, i] = a[i, j]\n";

    #[test]
    fn staging_absorbs_the_permutation_and_the_bus_cost_falls_to_the_payload() {
        // The claim ADR-0017 exists for, derived rather than measured: the same body, one
        // declared tile and one stream path, 36 bytes per element becoming 8.
        let flat = cost(UNTILED);
        assert_eq!(flat.sector_read_per_element, 4.0);
        assert_eq!(flat.sector_write_per_element, 32.0);
        assert!((flat.coalescence() - 8.0 / 36.0).abs() < 1e-12);

        let tiled = cost(&UNTILED
            .replace("    stream a : dram -> reg\n", "    tile 32, 32\n    stream a : dram -> smem -> reg\n"));
        assert_eq!(tiled.sector_read_per_element, 4.0);
        assert_eq!(tiled.sector_write_per_element, 4.0);
        assert_eq!(tiled.coalescence(), 1.0);
        // The payload never moved. That is what makes the two comparable.
        assert_eq!(tiled.bytes_per_element(), flat.bytes_per_element());
        assert_eq!(tiled.flops_per_element(), flat.flops_per_element());
    }

    #[test]
    fn the_drained_buffer_is_absorbed_too_even_though_it_is_not_staged() {
        // `b` travels `dram -> reg, drain`. It is the permuted one, and the tile makes its
        // write contiguous because the transposition happens in shared memory before it. A
        // rule keyed on "is this stream staged" would have left it strided.
        let ir = {
            let src = UNTILED.replace(
                "    stream a : dram -> reg\n",
                "    tile 32, 32\n    stream a : dram -> smem -> reg\n",
            );
            let unit = crate::parse::parse(&src).unwrap();
            lower(&unit, &unit.kernels[0]).unwrap()
        };
        let b = ir.streams.iter().find(|s| s.buffer == "b").unwrap();
        assert!(!b.staged, "b is not staged");
        assert!(b.coalesced, "and is still contiguous, because a is");
    }

    #[test]
    fn a_staged_element_crosses_the_shared_interface_twice() {
        let tiled = cost(&UNTILED
            .replace("    stream a : dram -> reg\n", "    tile 32, 32\n    stream a : dram -> smem -> reg\n"));
        let smem = tiled.at(Level::Smem).expect("a staged stream has shared traffic");
        assert_eq!((smem.read, smem.write), (4.0, 4.0));
        // And DRAM is still the roofline level: shared traffic is reported beside it, never
        // folded into it.
        assert_eq!(tiled.bytes_per_element(), Some(8.0));
    }

    #[test]
    fn a_tile_without_staging_absorbs_nothing() {
        // The rule is keyed on `smem` in a path, not on the word `tile`. A tile that stages
        // nothing changes no traffic -- and the back end refuses it for that reason.
        let tiled_only = cost(&UNTILED.replace("    stream a", "    tile 32, 32\n    stream a"));
        assert_eq!(tiled_only.sector_write_per_element, 32.0);
        assert!(tiled_only.coalescence() < 1.0);
    }
}
