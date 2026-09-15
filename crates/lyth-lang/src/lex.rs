//! Tokens, with the position they came from.
//!
//! Every token carries a line and column because ADR-0001 says the error messages are the
//! product. A refusal that cannot point at the source is not one.
//!
//! Layout is significant: the body of `at reg:` is delimited by indentation rather than by
//! braces, so the lexer emits [`Tok::Indent`] and [`Tok::Dedent`] and the parser never has to
//! count spaces.

use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Span {
    pub line: u32,
    pub col: u32,
}

impl fmt::Display for Span {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{}", self.line, self.col)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum Tok {
    Word(String),
    Int(u64),
    Float(f64),
    LParen,
    RParen,
    LBracket,
    RBracket,
    Comma,
    Colon,
    /// Separates a buffer's element type from its shape: `[f32; n]`.
    Semi,
    Arrow,
    Equals,
    Plus,
    Minus,
    Star,
    Slash,
    Indent,
    Dedent,
    Newline,
    Eof,
}

impl fmt::Display for Tok {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Tok::Word(w) => write!(f, "`{w}`"),
            Tok::Int(v) => write!(f, "`{v}`"),
            Tok::Float(v) => write!(f, "`{v}`"),
            Tok::LParen => write!(f, "`(`"),
            Tok::RParen => write!(f, "`)`"),
            Tok::LBracket => write!(f, "`[`"),
            Tok::RBracket => write!(f, "`]`"),
            Tok::Comma => write!(f, "`,`"),
            Tok::Semi => write!(f, "`;`"),
            Tok::Colon => write!(f, "`:`"),
            Tok::Arrow => write!(f, "`->`"),
            Tok::Equals => write!(f, "`=`"),
            Tok::Plus => write!(f, "`+`"),
            Tok::Minus => write!(f, "`-`"),
            Tok::Star => write!(f, "`*`"),
            Tok::Slash => write!(f, "`/`"),
            Tok::Indent => write!(f, "an indented block"),
            Tok::Dedent => write!(f, "the end of an indented block"),
            Tok::Newline => write!(f, "a line break"),
            Tok::Eof => write!(f, "the end of the file"),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Token {
    pub tok: Tok,
    pub span: Span,
}

#[derive(Debug, thiserror::Error)]
#[error("{span}: {msg}")]
pub struct LexError {
    pub span: Span,
    pub msg: String,
}

pub fn lex(src: &str) -> Result<Vec<Token>, LexError> {
    Lexer::new(src).run()
}

/// Carriage return, named rather than written as a literal so that a script rewriting this
/// file cannot turn the escape into the character it stands for. That is how the CRLF bug
/// this constant exists for was found in the first place.
const CR: char = '\r';

struct Lexer {
    chars: Vec<char>,
    i: usize,
    line: u32,
    col: u32,
    /// Indentation column of each open block, outermost first. Never empty.
    levels: Vec<u32>,
    /// Open parentheses. Layout is suppressed while this is above zero.
    ///
    /// A kernel signature is the one construct here that outgrows a line -- a matmul takes
    /// three extents and three buffers -- so a language that indents its blocks has to say
    /// what a newline inside brackets means. It means nothing: no `Newline`, no `Indent`, no
    /// `Dedent`, and the next line's leading spaces are ordinary whitespace. The alternative
    /// is a continuation character, which is a second way to write one thing.
    paren_depth: u32,
    out: Vec<Token>,
}

impl Lexer {
    fn new(src: &str) -> Self {
        Self {
            // Carriage returns are dropped here rather than handled at every site that
            // looks at a character. A CR is not part of any token, and layout is measured in
            // columns, so removing it cannot move a span: it only ever appears immediately
            // before the LF it belongs to, past the last token on the line.
            //
            // This is not a nicety. This project is developed on Windows, where editors
            // write CRLF by default, and without it a .lyth file saved by Notepad fails with
            // "expected `kernel`, found an indented block" -- an error about the wrong thing
            // entirely. Found by accident when a script rewrote an example. ADR-0013.
            chars: src.chars().filter(|c| *c != CR).collect(),
            i: 0,
            line: 1,
            col: 1,
            levels: vec![0],
            paren_depth: 0,
            out: Vec::new(),
        }
    }

    fn span(&self) -> Span {
        Span {
            line: self.line,
            col: self.col,
        }
    }

    fn level(&self) -> u32 {
        *self.levels.last().expect("levels is never empty")
    }

    fn peek(&self) -> Option<char> {
        self.chars.get(self.i).copied()
    }

    fn peek_at(&self, ahead: usize) -> Option<char> {
        self.chars.get(self.i + ahead).copied()
    }

    fn bump(&mut self) -> Option<char> {
        let c = self.chars.get(self.i).copied()?;
        self.i += 1;
        if c == '\n' {
            self.line += 1;
            self.col = 1;
        } else {
            self.col += 1;
        }
        Some(c)
    }

    fn push(&mut self, tok: Tok, span: Span) {
        self.out.push(Token { tok, span });
    }

    fn err(&self, msg: impl Into<String>) -> LexError {
        LexError {
            span: self.span(),
            msg: msg.into(),
        }
    }

    fn run(mut self) -> Result<Vec<Token>, LexError> {
        while self.i < self.chars.len() {
            self.start_of_line()?;
            if self.i >= self.chars.len() {
                break;
            }
            self.rest_of_line()?;
        }
        let span = self.span();
        if self.paren_depth > 0 {
            // Without this the rest of the file is swallowed as one logical line and the error
            // surfaces somewhere with nothing to do with the missing bracket.
            return Err(self.err(format!(
                "{} unclosed `(` at end of file",
                self.paren_depth
            )));
        }
        while self.levels.len() > 1 {
            self.levels.pop();
            self.push(Tok::Dedent, span);
        }
        self.push(Tok::Eof, span);
        Ok(self.out)
    }

    /// Consume leading whitespace and emit any Indent/Dedent it implies.
    fn start_of_line(&mut self) -> Result<(), LexError> {
        let mut indent = 0u32;
        loop {
            match self.peek() {
                Some(' ') => {
                    self.bump();
                    indent += 1;
                }
                // A tab is worth an unknowable number of columns, and guessing one would make
                // a block's extent depend on the reader's editor.
                Some('\t') => return Err(self.err("tabs are not indentation here — use spaces")),
                _ => break,
            }
        }
        // Blank and comment-only lines carry no layout meaning.
        match self.peek() {
            None => return Ok(()),
            Some('\n') => {
                self.bump();
                return Ok(());
            }
            Some('#') => {
                self.skip_comment();
                return Ok(());
            }
            _ => {}
        }
        let span = self.span();
        if indent > self.level() {
            self.levels.push(indent);
            self.push(Tok::Indent, span);
        } else {
            while indent < self.level() {
                self.levels.pop();
                self.push(Tok::Dedent, span);
            }
            if indent != self.level() {
                return Err(self.err(format!(
                    "indentation of {indent} matches no open block (open levels: {:?})",
                    self.levels
                )));
            }
        }
        Ok(())
    }

    fn skip_comment(&mut self) {
        while let Some(c) = self.peek() {
            if c == '\n' {
                break;
            }
            self.bump();
        }
    }

    fn rest_of_line(&mut self) -> Result<(), LexError> {
        loop {
            let span = self.span();
            let Some(c) = self.peek() else { return Ok(()) };
            match c {
                '\n' => {
                    self.bump();
                    // Inside brackets a line break is whitespace, and so is the next line's
                    // indentation -- consumed by the space arm below rather than by
                    // `start_of_line`, which is not reached until the bracket closes.
                    if self.paren_depth > 0 {
                        continue;
                    }
                    self.push(Tok::Newline, span);
                    return Ok(());
                }
                ' ' | '\r' => {
                    self.bump();
                }
                '#' => self.skip_comment(),
                '(' => {
                    self.paren_depth += 1;
                    self.one(Tok::LParen, span)
                }
                ')' => {
                    // Saturating: an unmatched `)` is the parser's to report, with the token
                    // in hand, and the lexer must not turn it into a panic here.
                    self.paren_depth = self.paren_depth.saturating_sub(1);
                    self.one(Tok::RParen, span)
                }
                '[' => self.one(Tok::LBracket, span),
                ']' => self.one(Tok::RBracket, span),
                ',' => self.one(Tok::Comma, span),
                ';' => self.one(Tok::Semi, span),
                ':' => self.one(Tok::Colon, span),
                '=' => self.one(Tok::Equals, span),
                '+' => self.one(Tok::Plus, span),
                '*' => self.one(Tok::Star, span),
                '/' => self.one(Tok::Slash, span),
                '-' => {
                    self.bump();
                    if self.peek() == Some('>') {
                        self.bump();
                        self.push(Tok::Arrow, span);
                    } else {
                        self.push(Tok::Minus, span);
                    }
                }
                c if c.is_ascii_digit() => self.number(span)?,
                c if c.is_alphabetic() || c == '_' => self.word(span),
                other => return Err(self.err(format!("unexpected character `{other}`"))),
            }
        }
    }

    fn one(&mut self, tok: Tok, span: Span) {
        self.bump();
        self.push(tok, span);
    }

    fn word(&mut self, span: Span) {
        let mut s = String::new();
        while let Some(c) = self.peek() {
            if c.is_alphanumeric() || c == '_' {
                s.push(c);
                self.bump();
            } else {
                break;
            }
        }
        self.push(Tok::Word(s), span);
    }

    fn number(&mut self, span: Span) -> Result<(), LexError> {
        let mut s = String::new();
        let mut is_float = false;
        while let Some(c) = self.peek() {
            if c.is_ascii_digit() {
                s.push(c);
                self.bump();
            } else if c == '.' && !is_float {
                // `1.` before a non-digit is an integer sitting next to something else.
                if !matches!(self.peek_at(1), Some(d) if d.is_ascii_digit()) {
                    break;
                }
                is_float = true;
                s.push(c);
                self.bump();
            } else if (c == 'e' || c == 'E') && !s.is_empty() {
                is_float = true;
                s.push(c);
                self.bump();
                if let Some(sign @ ('-' | '+')) = self.peek() {
                    s.push(sign);
                    self.bump();
                }
            } else {
                break;
            }
        }
        if is_float {
            let v = s
                .parse::<f64>()
                .map_err(|e| self.err(format!("`{s}` is not a number: {e}")))?;
            self.push(Tok::Float(v), span);
        } else {
            let v = s
                .parse::<u64>()
                .map_err(|e| self.err(format!("`{s}` is not an integer: {e}")))?;
            self.push(Tok::Int(v), span);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn kinds(src: &str) -> Vec<Tok> {
        lex(src).unwrap().into_iter().map(|t| t.tok).collect()
    }

    #[test]
    fn indentation_opens_and_closes_a_block() {
        let t = kinds("at reg:\n y = x\nkernel\n");
        let i = t.iter().position(|x| *x == Tok::Indent).expect("an indent");
        let d = t.iter().position(|x| *x == Tok::Dedent).expect("a dedent");
        assert!(i < d, "indent must come before its dedent: {t:?}");
    }

    #[test]
    fn every_open_block_is_closed_at_end_of_file() {
        let t = kinds("kernel k()\n at reg:\n y = x\n");
        let opens = t.iter().filter(|x| **x == Tok::Indent).count();
        let closes = t.iter().filter(|x| **x == Tok::Dedent).count();
        assert_eq!(opens, closes, "unbalanced layout: {t:?}");
    }

    #[test]
    fn arrow_is_one_token_and_minus_is_not() {
        assert_eq!(kinds("a -> b")[1], Tok::Arrow);
        assert_eq!(kinds("a - b")[1], Tok::Minus);
    }

    #[test]
    fn comments_and_blank_lines_do_not_change_layout() {
        let t = kinds("kernel k()\n\n  # note\n  intensity 0.5\n");
        assert_eq!(t.iter().filter(|x| **x == Tok::Indent).count(), 1, "{t:?}");
    }

    #[test]
    fn a_tab_is_refused_rather_than_given_a_guessed_width() {
        let e = lex("kernel k()\n\ty = x\n").unwrap_err();
        assert!(e.to_string().contains("tabs"), "{e}");
    }

    #[test]
    fn a_dedent_to_no_open_level_is_refused() {
        let e = lex("kernel k()\n        a = 1\n    b = 2\n").unwrap_err();
        assert!(e.to_string().contains("matches no open block"), "{e}");
    }

    #[test]
    fn floats_and_ints_are_distinguished() {
        assert_eq!(kinds("0.1667")[0], Tok::Float(0.1667));
        assert_eq!(kinds("4")[0], Tok::Int(4));
        assert_eq!(kinds("1e-3")[0], Tok::Float(1e-3));
    }

    #[test]
    fn errors_carry_the_line_they_happened_on() {
        let e = lex("kernel k()\n  a = 1\n  b = $\n").unwrap_err();
        assert_eq!(e.span.line, 3, "{e}");
    }
}

#[cfg(test)]
mod crlf {
    use super::*;

    /// A .lyth file saved by a Windows editor must lex identically to one saved on Linux.
    #[test]
    fn a_file_saved_on_windows_lexes_the_same_as_one_saved_on_linux() {
        let unix = "machine sm_120\n\nkernel k(n: u32, x: [f32])\n    stream x : dram -> reg\n";
        let dos = unix.replace('\n', "\r\n");
        let a = lex(unix).expect("LF must lex");
        let b = lex(&dos).expect("CRLF must lex");
        // Spans included: a dropped CR must not shift a line or a column.
        assert_eq!(a, b, "line endings must not change the token stream");
    }

    #[test]
    fn a_stray_carriage_return_is_not_a_token() {
        assert_eq!(lex("machine sm_120\n").unwrap(), lex("machine\r sm_120\n").unwrap());
    }
}
