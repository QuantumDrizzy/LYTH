//! Tokens to AST.
//!
//! Recursive descent, one token of lookahead. Every refusal names the line, what was found and
//! what was expected — ADR-0001 puts error messages in the product, not in the polish.

use crate::ast::*;
use crate::lex::{lex, Span, Tok, Token};

#[derive(Debug, thiserror::Error)]
pub enum ParseError {
    #[error("{0}")]
    Lex(#[from] crate::lex::LexError),
    #[error("{span}: expected {expected}, found {found}")]
    Expected {
        span: Span,
        expected: String,
        found: String,
    },
    #[error("{span}: {msg}")]
    Message { span: Span, msg: String },
}

pub fn parse(src: &str) -> Result<Unit, ParseError> {
    let tokens = lex(src)?;
    Parser { toks: tokens, i: 0 }.unit()
}

struct Parser {
    toks: Vec<Token>,
    i: usize,
}

impl Parser {
    fn peek(&self) -> &Tok {
        &self.toks[self.i.min(self.toks.len() - 1)].tok
    }

    fn span(&self) -> Span {
        self.toks[self.i.min(self.toks.len() - 1)].span
    }

    fn bump(&mut self) -> Token {
        let t = self.toks[self.i.min(self.toks.len() - 1)].clone();
        if self.i < self.toks.len() - 1 {
            self.i += 1;
        }
        t
    }

    fn expected<T>(&self, what: impl Into<String>) -> Result<T, ParseError> {
        Err(ParseError::Expected {
            span: self.span(),
            expected: what.into(),
            found: self.peek().to_string(),
        })
    }

    fn msg<T>(&self, msg: impl Into<String>) -> Result<T, ParseError> {
        Err(ParseError::Message {
            span: self.span(),
            msg: msg.into(),
        })
    }

    fn eat(&mut self, tok: &Tok) -> bool {
        if self.peek() == tok {
            self.bump();
            true
        } else {
            false
        }
    }

    fn expect(&mut self, tok: Tok) -> Result<Span, ParseError> {
        if self.peek() == &tok {
            Ok(self.bump().span)
        } else {
            self.expected(tok.to_string())
        }
    }

    fn word(&mut self) -> Result<(String, Span), ParseError> {
        match self.peek().clone() {
            Tok::Word(w) => {
                let span = self.bump().span;
                Ok((w, span))
            }
            _ => self.expected("a name"),
        }
    }

    /// Consume a specific keyword, or leave the position untouched.
    fn eat_word(&mut self, kw: &str) -> bool {
        if matches!(self.peek(), Tok::Word(w) if w == kw) {
            self.bump();
            true
        } else {
            false
        }
    }

    fn at_word(&self, kw: &str) -> bool {
        matches!(self.peek(), Tok::Word(w) if w == kw)
    }

    fn skip_newlines(&mut self) {
        while self.eat(&Tok::Newline) {}
    }

    fn number(&mut self) -> Result<(f64, Span), ParseError> {
        match *self.peek() {
            Tok::Float(v) => Ok((v, self.bump().span)),
            Tok::Int(v) => Ok((v as f64, self.bump().span)),
            _ => self.expected("a number"),
        }
    }

    // --- grammar ----------------------------------------------------------------------

    fn unit(mut self) -> Result<Unit, ParseError> {
        self.skip_newlines();
        if !self.eat_word("machine") {
            return self.msg(
                "a source file starts with `machine <id>` — the machine is a value, not a flag \
                 (ADR-0001), so a kernel with no machine has no ridge to be checked against",
            );
        }
        let (machine, machine_span) = self.word()?;
        self.skip_newlines();

        let mut kernels = Vec::new();
        loop {
            self.skip_newlines();
            match self.peek() {
                Tok::Eof => break,
                Tok::Word(w) if w == "kernel" => kernels.push(self.kernel()?),
                _ => return self.expected("`kernel` or the end of the file"),
            }
        }
        if kernels.is_empty() {
            return Err(ParseError::Message {
                span: machine_span,
                msg: "this file declares a machine but no kernel".into(),
            });
        }
        Ok(Unit {
            machine,
            machine_span,
            kernels,
        })
    }

    fn kernel(&mut self) -> Result<Kernel, ParseError> {
        let span = self.span();
        self.expect(Tok::Word("kernel".into()))?;
        let (name, _) = self.word()?;
        self.expect(Tok::LParen)?;
        let mut params = Vec::new();
        if !self.eat(&Tok::RParen) {
            loop {
                params.push(self.param()?);
                if self.eat(&Tok::Comma) {
                    continue;
                }
                self.expect(Tok::RParen)?;
                break;
            }
        }
        self.skip_newlines();
        self.expect(Tok::Indent)?;

        let mut declared_intensity = None;
        let mut intensity_span = None;
        let mut streams = Vec::new();
        let mut blocks = Vec::new();

        loop {
            self.skip_newlines();
            if self.eat(&Tok::Dedent) || matches!(self.peek(), Tok::Eof) {
                break;
            }
            if self.at_word("intensity") {
                let s = self.span();
                self.bump();
                if declared_intensity.is_some() {
                    return self.msg("`intensity` declared twice");
                }
                let (v, _) = self.number()?;
                // `flop/byte` may be written out; the unit is fixed, so it is decoration.
                if self.at_word("flop") {
                    self.bump();
                    self.expect(Tok::Slash)?;
                    self.expect(Tok::Word("byte".into()))?;
                }
                declared_intensity = Some(v);
                intensity_span = Some(s);
            } else if self.at_word("stream") {
                streams.push(self.stream()?);
            } else if self.at_word("at") {
                blocks.push(self.block()?);
            } else {
                return self.expected("`intensity`, `stream` or `at`");
            }
        }

        Ok(Kernel {
            name,
            span,
            params,
            declared_intensity,
            intensity_span,
            streams,
            blocks,
        })
    }

    fn param(&mut self) -> Result<Param, ParseError> {
        let (name, span) = self.word()?;
        self.expect(Tok::Colon)?;
        let ty = if self.eat(&Tok::LBracket) {
            let (t, _) = self.word()?;
            self.expect(Tok::RBracket)?;
            match t.as_str() {
                "f32" => Ty::BufF32,
                other => {
                    return self.msg(format!(
                        "`[{other}]` is not a buffer type in v1; only [f32]"
                    ))
                }
            }
        } else {
            let (t, _) = self.word()?;
            match t.as_str() {
                "u32" => Ty::U32,
                "f32" => Ty::F32,
                other => {
                    return self.msg(format!("`{other}` is not a type in v1; u32, f32, [f32]"))
                }
            }
        };
        Ok(Param { name, ty, span })
    }

    fn stream(&mut self) -> Result<StreamDecl, ParseError> {
        let span = self.span();
        self.expect(Tok::Word("stream".into()))?;
        let (buffer, _) = self.word()?;
        self.expect(Tok::Colon)?;
        let (from_w, from_span) = self.word()?;
        let Some(from) = Level::parse(&from_w) else {
            return Err(ParseError::Message {
                span: from_span,
                msg: format!("`{from_w}` is not a level; dram, l2, smem, reg"),
            });
        };
        self.expect(Tok::Arrow)?;
        let (to_w, to_span) = self.word()?;
        let Some(to) = Level::parse(&to_w) else {
            return Err(ParseError::Message {
                span: to_span,
                msg: format!("`{to_w}` is not a level; dram, l2, smem, reg"),
            });
        };
        let mut drain = false;
        while self.eat(&Tok::Comma) {
            let (w, s) = self.word()?;
            match w.as_str() {
                "drain" => drain = true,
                other => {
                    return Err(ParseError::Message {
                        span: s,
                        msg: format!("`{other}` is not a stream attribute in v1; only `drain`"),
                    })
                }
            }
        }
        Ok(StreamDecl {
            buffer,
            from,
            to,
            drain,
            span,
        })
    }

    fn block(&mut self) -> Result<Block, ParseError> {
        let span = self.span();
        self.expect(Tok::Word("at".into()))?;
        let (lvl, lvl_span) = self.word()?;
        let Some(level) = Level::parse(&lvl) else {
            return Err(ParseError::Message {
                span: lvl_span,
                msg: format!("`{lvl}` is not a level; dram, l2, smem, reg"),
            });
        };
        self.expect(Tok::Colon)?;
        self.skip_newlines();
        self.expect(Tok::Indent)?;

        let mut stmts = Vec::new();
        loop {
            self.skip_newlines();
            if self.eat(&Tok::Dedent) || matches!(self.peek(), Tok::Eof) {
                break;
            }
            stmts.push(self.stmt()?);
        }
        if stmts.is_empty() {
            return Err(ParseError::Message {
                span,
                msg: "an `at` block with no statements declares work that does not happen".into(),
            });
        }
        Ok(Block { level, span, stmts })
    }

    fn stmt(&mut self) -> Result<Stmt, ParseError> {
        let (target, target_span) = self.word()?;
        self.expect(Tok::Equals)?;
        let value = self.expr(0)?;
        Ok(Stmt {
            target,
            target_span,
            value,
        })
    }

    /// Precedence climbing. `+ -` bind loosest, then `* /`, then unary minus.
    fn expr(&mut self, min_prec: u8) -> Result<Expr, ParseError> {
        let mut lhs = self.unary()?;
        loop {
            let (op, prec) = match self.peek() {
                Tok::Plus => (BinOp::Add, 1),
                Tok::Minus => (BinOp::Sub, 1),
                Tok::Star => (BinOp::Mul, 2),
                Tok::Slash => (BinOp::Div, 2),
                _ => break,
            };
            if prec < min_prec {
                break;
            }
            let span = self.bump().span;
            let rhs = self.expr(prec + 1)?;
            lhs = Expr::Bin {
                op,
                lhs: Box::new(lhs),
                rhs: Box::new(rhs),
                span,
            };
        }
        Ok(lhs)
    }

    fn unary(&mut self) -> Result<Expr, ParseError> {
        if self.peek() == &Tok::Minus {
            let span = self.bump().span;
            return Ok(Expr::Neg(Box::new(self.unary()?), span));
        }
        self.atom()
    }

    fn atom(&mut self) -> Result<Expr, ParseError> {
        match self.peek().clone() {
            Tok::Word(w) => {
                let span = self.bump().span;
                Ok(Expr::Name(w, span))
            }
            Tok::Float(v) => {
                let span = self.bump().span;
                Ok(Expr::Const(v, span))
            }
            Tok::Int(v) => {
                let span = self.bump().span;
                Ok(Expr::Const(v as f64, span))
            }
            Tok::LParen => {
                self.bump();
                let e = self.expr(0)?;
                self.expect(Tok::RParen)?;
                Ok(e)
            }
            _ => self.expected("a name, a number or `(`"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
    fn saxpy_parses() {
        let u = parse(SAXPY).expect("saxpy should parse");
        assert_eq!(u.machine, "sm_120");
        assert_eq!(u.kernels.len(), 1);
        let k = &u.kernels[0];
        assert_eq!(k.name, "saxpy");
        assert_eq!(k.params.len(), 4);
        assert_eq!(k.params[2].ty, Ty::BufF32);
        assert_eq!(k.declared_intensity, Some(0.1667));
        assert_eq!(k.streams.len(), 2);
        assert!(!k.streams[0].drain);
        assert!(k.streams[1].drain, "y is written back");
        assert_eq!(k.blocks.len(), 1);
        assert_eq!(k.blocks[0].stmts.len(), 1);
    }

    #[test]
    fn multiplication_binds_tighter_than_addition() {
        let u = parse(SAXPY).unwrap();
        // a * x + y must group as (a * x) + y, or the fma pattern and the flop count are both
        // wrong, and the second error hides behind the first.
        match &u.kernels[0].blocks[0].stmts[0].value {
            Expr::Bin { op, lhs, .. } => {
                assert_eq!(*op, BinOp::Add);
                assert!(matches!(**lhs, Expr::Bin { op: BinOp::Mul, .. }), "{lhs:?}");
            }
            other => panic!("expected a binary node, got {other:?}"),
        }
    }

    #[test]
    fn a_file_without_a_machine_is_refused_with_the_reason() {
        let e = parse(
            "kernel k(x: [f32])\n    at reg:
        x = 1\n",
        )
        .unwrap_err();
        assert!(e.to_string().contains("machine is a value"), "{e}");
    }

    #[test]
    fn an_empty_at_block_is_refused() {
        let src =
            "machine m\n\nkernel k(x: [f32])\n    stream x : dram -> reg\n    at reg:\n        \n";
        assert!(
            parse(src).is_err(),
            "a block that does no work must not compile"
        );
    }

    #[test]
    fn errors_name_the_line_and_what_was_expected() {
        let src = "machine m\n\nkernel k(x: [f32])\n    stream x : dram => reg\n";
        let e = parse(src).unwrap_err();
        let s = e.to_string();
        assert!(s.contains("4:"), "should point at line 4: {s}");
    }

    #[test]
    fn an_unknown_level_lists_the_ones_that_exist() {
        let src = "machine m\n\nkernel k(x: [f32])\n    stream x : hbm -> reg\n    at reg:
        x = 1\n";
        let e = parse(src).unwrap_err();
        assert!(e.to_string().contains("dram, l2, smem, reg"), "{e}");
    }

    #[test]
    fn parentheses_override_precedence() {
        let src = "machine m\n\nkernel k(a: f32, x: [f32])\n    stream x : dram -> reg, drain\n    at reg:
        x = a * (x + x)\n";
        let u = parse(src).unwrap();
        match &u.kernels[0].blocks[0].stmts[0].value {
            Expr::Bin {
                op: BinOp::Mul,
                rhs,
                ..
            } => {
                assert!(matches!(**rhs, Expr::Bin { op: BinOp::Add, .. }), "{rhs:?}");
            }
            other => panic!("expected a multiply at the root, got {other:?}"),
        }
    }
}
