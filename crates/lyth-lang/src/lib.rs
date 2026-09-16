//! LYTH front end: source text to a checked, lowered kernel.
//!
//! `lex` -> `parse` -> `ir::lower` -> `check::check_intensity`. Nothing here knows about
//! PTX or about CUDA; the back end consumes `ir::KernelIr` and nothing else.

pub mod half;
pub mod inputs;
pub mod ast;
pub mod check;
pub mod eval;
pub mod ir;
pub mod lex;
pub mod parse;

pub use ast::{BinOp, Kernel, Level, Param, Ty, Unit};
pub use check::{check_intensity, IntensityMismatch, IntensityReport, Regime, Ridge};
pub use eval::{eval, EvalError, Inputs};
pub use ir::{lower, Cost, KernelIr, LowerError, Op, RegId, StreamIr};
pub use lex::{lex, LexError, Span};
pub use parse::{parse, ParseError};
