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
    /// `at <level>:` blocks, in source order.
    pub blocks: Vec<Block>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Param {
    pub name: String,
    pub ty: Ty,
    pub span: Span,
}

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

/// `<target> = <expr>`. The only statement there is.
#[derive(Debug, Clone, PartialEq)]
pub struct Stmt {
    pub target: String,
    pub target_span: Span,
    pub value: Expr,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Expr {
    /// A buffer element at the current index, or a scalar parameter.
    Name(String, Span),
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
            Expr::Name(_, s) | Expr::Const(_, s) | Expr::Bin { span: s, .. } | Expr::Neg(_, s) => {
                *s
            }
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
