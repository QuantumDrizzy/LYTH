//! Tokens to AST.
//!
//! Recursive descent, one token of lookahead. Every refusal names the line, what was found and
//! what was expected — ADR-0001 puts error messages in the product, not in the polish.

use crate::ast::*;
use crate::lex::{lex, Span, Tok, Token};

/// One message for both spellings of a rank-2 contraction, so the refusal is about the design
/// rather than about whichever symbol the parser tripped over first.
const RANK_2_CONTRACTION: &str =
    "v1 contracts one axis; `contract sum p, q : k, l` is a rank-2 contraction and its traffic      is not the expression this compiler derives";

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
        let mut intensity_is_asymptotic = false;
        let mut streams = Vec::new();
        let mut space = None;
        let mut tile = None;
        let mut coarsen = None;
        let mut reductions = Vec::new();
        let mut contract = None;
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
                intensity_is_asymptotic = self.eat_word("asymptotic");
                let (v, _) = self.number()?;
                // `flop/byte` may be written out; the unit is fixed, so it is decoration.
                if self.at_word("flop") {
                    self.bump();
                    self.expect(Tok::Slash)?;
                    self.expect(Tok::Word("byte".into()))?;
                }
                declared_intensity = Some(v);
                intensity_span = Some(s);
            } else if self.at_word("space") {
                if space.is_some() {
                    return self.msg("`space` declared twice");
                }
                space = Some(self.space()?);
            } else if self.at_word("tile") {
                if tile.is_some() {
                    return self.msg("`tile` declared twice");
                }
                tile = Some(self.tile()?);
            } else if self.at_word("coarsen") {
                if coarsen.is_some() {
                    return self.msg("`coarsen` declared twice");
                }
                coarsen = Some(self.coarsen()?);
            } else if self.at_word("stream") {
                streams.push(self.stream()?);
            } else if self.at_word("reduce") {
                reductions.push(self.reduce()?);
            } else if self.at_word("contract") {
                if contract.is_some() {
                    return self.msg("`contract` declared twice; v1 contracts one axis");
                }
                contract = Some(self.contract()?);
            } else if self.at_word("at") {
                blocks.push(self.block()?);
            } else {
                return self.expected(
                    "`intensity`, `space`, `tile`, `coarsen`, `stream`, `reduce`, `contract` or `at`",
                );
            }
        }

        Ok(Kernel {
            name,
            span,
            params,
            declared_intensity,
            intensity_span,
            intensity_is_asymptotic,
            streams,
            space,
            tile,
            coarsen,
            reductions,
            contract,
            blocks,
        })
    }

    /// `contract sum p : k`
    fn contract(&mut self) -> Result<ContractDecl, ParseError> {
        let span = self.span();
        self.expect(Tok::Word("contract".into()))?;
        let (op_word, op_span) = self.word()?;
        let Some(op) = ReduceOp::parse(&op_word) else {
            return Err(ParseError::Message {
                span: op_span,
                msg: format!("`{op_word}` is not a combining operator; sum, max or min"),
            });
        };
        let (var, _) = self.word()?;
        // One axis, and the refusal is here rather than left to `expect(Colon)` so that both
        // spellings of the mistake say the same thing. `contract sum p, q : k` would otherwise
        // report "expected `:`, found `,`", which names the symbol and not the design.
        if *self.peek() == Tok::Comma {
            return self.msg(RANK_2_CONTRACTION);
        }
        self.expect(Tok::Colon)?;
        let (extent, _) = self.word()?;
        if *self.peek() == Tok::Comma {
            return self.msg(RANK_2_CONTRACTION);
        }
        Ok(ContractDecl {
            op,
            var,
            extent,
            span,
        })
    }

    /// `reduce sum p : reg -> smem -> dram into partial`
    fn reduce(&mut self) -> Result<ReduceDecl, ParseError> {
        let span = self.span();
        self.expect(Tok::Word("reduce".into()))?;
        let (op_word, op_span) = self.word()?;
        let Some(op) = ReduceOp::parse(&op_word) else {
            return Err(ParseError::Message {
                span: op_span,
                msg: format!("`{op_word}` is not a reduction in v1; only `sum`"),
            });
        };
        let (source, _) = self.word()?;
        self.expect(Tok::Colon)?;

        // The path, written out: reg -> smem -> dram.
        let mut path = Vec::new();
        loop {
            let (w, s) = self.word()?;
            let Some(level) = Level::parse(&w) else {
                return Err(ParseError::Message {
                    span: s,
                    msg: format!("`{w}` is not a level; dram, l2, smem, reg"),
                });
            };
            path.push(level);
            if !self.eat(&Tok::Arrow) {
                break;
            }
        }
        if !self.eat_word("into") {
            return self.expected("`into <buffer>` naming where the block result lands");
        }
        let (into, _) = self.word()?;
        Ok(ReduceDecl {
            op,
            source,
            path,
            into,
            span,
        })
    }

    fn param(&mut self) -> Result<Param, ParseError> {
        let (name, span) = self.word()?;
        self.expect(Tok::Colon)?;
        let mut shape = Vec::new();
        let ty = if self.eat(&Tok::LBracket) {
            let (t, _) = self.word()?;
            // The element's width is a **storage** decision and nothing more (ADR-0024). A
            // `[f16; n]` buffer moves two bytes per element and is loaded into an f32
            // register; the arithmetic does not narrow, because accumulating a contraction in
            // half precision is a different function and a much worse one. Same rule ADR-0010
            // applied to fusing a multiply and an add.
            let elem = match t.as_str() {
                "f32" => Ty::BufF32,
                "f16" => Ty::BufF16,
                "bf16" => Ty::BufBF16,
                _ => {
                    return self.msg(format!(
                        "`[{t}; ..]` is not a buffer type; [f32], [f16] and [bf16]"
                    ))
                }
            };
            // The shape is not optional. A buffer whose length is not written down is a
            // buffer whose bounds the compiler has to guess, and guessing is the one thing
            // this language does not do.
            if !self.eat(&Tok::Semi) {
                return self.msg(
                    "a buffer needs its extent: write `[f32; n]`, naming a u32 parameter,                      or `[f32; blocks]` for a reduction target"
                        .to_string(),
                );
            }
            loop {
                let (d, _) = self.word()?;
                shape.push(d);
                if !self.eat(&Tok::Comma) {
                    break;
                }
            }
            self.expect(Tok::RBracket)?;
            elem
        } else {
            let (t, _) = self.word()?;
            match t.as_str() {
                "u32" => Ty::U32,
                "f32" => Ty::F32,
                other => {
                    return self
                        .msg(format!(
                            "`{other}` is not a type; u32, f32, [f32; n], [f16; n], [bf16; n]"
                        ))
                }
            }
        };
        Ok(Param {
            name,
            ty,
            shape,
            span,
        })
    }

    /// `tile 32, 32`
    fn tile(&mut self) -> Result<TileDecl, ParseError> {
        let span = self.span();
        self.expect(Tok::Word("tile".into()))?;
        let mut dims = Vec::new();
        loop {
            let (v, s) = self.number()?;
            let d = v as u32;
            if d as f64 != v || d == 0 {
                return Err(ParseError::Message {
                    span: s,
                    msg: format!("`{v}` is not a tile dimension; give a positive whole number"),
                });
            }
            // A power of two, so a thread's place in the tile is a shift and a mask. Anything
            // else needs a division per thread, which on this hardware is a sequence and not
            // an instruction -- refused rather than paid for silently.
            if !d.is_power_of_two() {
                return Err(ParseError::Message {
                    span: s,
                    msg: format!(
                        "a tile dimension must be a power of two, got {d}. A thread's position inside the tile is a shift and a mask; any other width needs a division per thread."
                    ),
                });
            }
            dims.push(d);
            if !self.eat(&Tok::Comma) {
                break;
            }
        }
        Ok(TileDecl { dims, span })
    }

    /// `coarsen 2, 2`
    fn coarsen(&mut self) -> Result<CoarsenDecl, ParseError> {
        let span = self.span();
        self.expect(Tok::Word("coarsen".into()))?;
        let mut dims = Vec::new();
        loop {
            let (v, s) = self.number()?;
            let d = v as u32;
            if d as f64 != v || d == 0 {
                return Err(ParseError::Message {
                    span: s,
                    msg: format!("`{v}` is not a coarsening factor; give a positive whole number"),
                });
            }
            // A power of two for the same reason a tile dimension is one: the block's width is
            // the tile's divided by this, and a thread's place in the tile stays a shift and a
            // mask only if every one of those is a power of two.
            if !d.is_power_of_two() {
                return Err(ParseError::Message {
                    span: s,
                    msg: format!(
                        "a coarsening factor must be a power of two, got {d}. The block's width is the tile's divided by it, and a thread's position inside the tile is a shift and a mask."
                    ),
                });
            }
            dims.push(d);
            if !self.eat(&Tok::Comma) {
                break;
            }
        }
        Ok(CoarsenDecl { dims, span })
    }

    /// `space i, j : rows, cols`
    fn space(&mut self) -> Result<SpaceDecl, ParseError> {
        let span = self.span();
        self.expect(Tok::Word("space".into()))?;
        let mut vars = Vec::new();
        loop {
            let (v, _) = self.word()?;
            vars.push(v);
            if !self.eat(&Tok::Comma) {
                break;
            }
        }
        self.expect(Tok::Colon)?;
        let mut extents = Vec::new();
        loop {
            let (e, _) = self.word()?;
            extents.push(e);
            if !self.eat(&Tok::Comma) {
                break;
            }
        }
        if vars.len() != extents.len() {
            return self.msg(format!(
                "`space` names {} index variables and {} extents; they pair up one to one",
                vars.len(),
                extents.len()
            ));
        }
        Ok(SpaceDecl {
            vars,
            extents,
            span,
        })
    }

    fn stream(&mut self) -> Result<StreamDecl, ParseError> {
        let span = self.span();
        self.expect(Tok::Word("stream".into()))?;
        let (buffer, _) = self.word()?;
        self.expect(Tok::Colon)?;
        // A path, not a pair: `dram -> reg`, or `dram -> smem -> reg` when the tile stages it.
        let mut path = Vec::new();
        loop {
            let (w, w_span) = self.word()?;
            let Some(level) = Level::parse(&w) else {
                return Err(ParseError::Message {
                    span: w_span,
                    msg: format!("`{w}` is not a level; dram, l2, smem, reg"),
                });
            };
            path.push(level);
            if !self.eat(&Tok::Arrow) {
                break;
            }
        }
        if path.len() < 2 {
            return self.msg("a stream moves between levels: write `dram -> reg`".to_string());
        }
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
            path,
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
        let target_index = self.index_list()?;
        self.expect(Tok::Equals)?;
        let value = self.expr(0)?;
        Ok(Stmt {
            target,
            target_index,
            target_span,
            value,
        })
    }

    /// `[i, j]` after a name, or nothing. Only index variables: an index expression that can
    /// be arithmetic is an index expression whose footprint needs solving for, and v1 does not.
    fn index_list(&mut self) -> Result<Vec<String>, ParseError> {
        let mut idx = Vec::new();
        if self.eat(&Tok::LBracket) {
            loop {
                let (v, _) = self.word()?;
                idx.push(v);
                if !self.eat(&Tok::Comma) {
                    break;
                }
            }
            self.expect(Tok::RBracket)?;
        }
        Ok(idx)
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
                let index = self.index_list()?;
                if index.is_empty() {
                    Ok(Expr::Name(w, span))
                } else {
                    Ok(Expr::At {
                        buffer: w,
                        index,
                        span,
                    })
                }
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

kernel saxpy(n: u32, a: f32, x: [f32; n], y: [f32; n])
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
            "kernel k(x: [f32; n])\n    at reg:
        x = 1\n",
        )
        .unwrap_err();
        assert!(e.to_string().contains("machine is a value"), "{e}");
    }

    #[test]
    fn an_empty_at_block_is_refused() {
        let src =
            "machine m\n\nkernel k(x: [f32; n])\n    stream x : dram -> reg\n    at reg:\n        \n";
        assert!(
            parse(src).is_err(),
            "a block that does no work must not compile"
        );
    }

    #[test]
    fn errors_name_the_line_and_what_was_expected() {
        let src = "machine m\n\nkernel k(x: [f32; n])\n    stream x : dram => reg\n";
        let e = parse(src).unwrap_err();
        let s = e.to_string();
        assert!(s.contains("4:"), "should point at line 4: {s}");
    }

    #[test]
    fn an_unknown_level_lists_the_ones_that_exist() {
        let src = "machine m\n\nkernel k(x: [f32; n])\n    stream x : hbm -> reg\n    at reg:
        x = 1\n";
        let e = parse(src).unwrap_err();
        assert!(e.to_string().contains("dram, l2, smem, reg"), "{e}");
    }

    #[test]
    fn parentheses_override_precedence() {
        let src = "machine m\n\nkernel k(a: f32, x: [f32; n])\n    stream x : dram -> reg, drain\n    at reg:
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
