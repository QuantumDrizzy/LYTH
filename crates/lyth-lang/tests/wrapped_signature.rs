//! A kernel signature may span lines, and doing so must change nothing.
//!
//! This is the one construct in the language that outgrows a line: a matmul takes three
//! extents and three buffers, and ADR-0018 writes it wrapped. A layout-sensitive language has
//! to say what a newline inside brackets means, and here it means nothing at all -- the lexer
//! suppresses `Newline`, `Indent` and `Dedent` while a parenthesis is open, so the next line's
//! leading spaces are whitespace like any other.
//!
//! The alternative would be a continuation character, which is a second way to write one
//! thing. The test that matters is the first one: the two spellings produce the *same* IR, not
//! merely two IRs that both work.

use lyth_lang::ir::lower;
use lyth_lang::parse::parse;

const ONE_LINE: &str = "machine sm_120\n\n\
     kernel mm(m: u32, n: u32, k: u32, a: [f32; m, k], b: [f32; k, n], c: [f32; m, n])\n    \
     space i, j : m, n\n    \
     stream a : dram -> reg\n    \
     stream b : dram -> reg\n    \
     stream c : dram -> reg, drain\n    \
     at reg:\n        \
     c[i, j] = a[i, j] * b[i, j]\n";

const WRAPPED: &str = "machine sm_120\n\n\
     kernel mm(m: u32, n: u32, k: u32,\n\
     \x20             a: [f32; m, k],\n\
     \x20             b: [f32; k, n],\n\
     \x20             c: [f32; m, n])\n    \
     space i, j : m, n\n    \
     stream a : dram -> reg\n    \
     stream b : dram -> reg\n    \
     stream c : dram -> reg, drain\n    \
     at reg:\n        \
     c[i, j] = a[i, j] * b[i, j]\n";

#[test]
fn a_wrapped_signature_produces_the_same_kernel_as_a_flat_one() {
    let flat = parse(ONE_LINE).expect("the flat form parses");
    let wrapped = parse(WRAPPED).expect("the wrapped form parses");
    let mut a = lower(&flat, &flat.kernels[0]).expect("lowers");
    let mut b = lower(&wrapped, &wrapped.kernels[0]).expect("lowers");

    // Everything except where the text is. A parameter on line 4 carries `line: 4`, and it
    // should: an error about it must point at the line the reader is looking at. So the spans
    // are flattened and the rest compared exactly -- streams, indices, ops, drains, cost.
    for k in [&mut a, &mut b] {
        for p in &mut k.params {
            p.span = lyth_lang::lex::Span { line: 0, col: 0 };
        }
    }
    assert_eq!(a, b, "wrapping a signature must not change the kernel");
}

#[test]
fn the_body_after_a_wrapped_signature_is_still_a_block() {
    // The risk in suppressing layout is suppressing it a line too long: if the `Indent` for
    // the kernel body were swallowed, `space` and the streams would read as top level and the
    // parser would complain about `space` rather than about anything to do with brackets.
    let u = parse(WRAPPED).expect("parses");
    let k = &u.kernels[0];
    assert!(k.space.is_some(), "the space survived the wrap");
    assert_eq!(k.streams.len(), 3);
    assert_eq!(k.blocks.len(), 1);
    assert_eq!(k.params.len(), 6);
}

#[test]
fn an_unclosed_parenthesis_is_reported_as_one() {
    // Without the check this swallows the remainder of the file as a single logical line and
    // fails somewhere unrelated -- which is exactly how the CRLF bug used to present.
    let src = "machine sm_120\n\nkernel k(n: u32, x: [f32; n]\n    stream x : dram -> reg\n";
    let e = parse(src).expect_err("an unclosed bracket");
    assert!(e.to_string().contains("unclosed `(`"), "{e}");
}

#[test]
fn the_matmul_example_on_disk_is_the_wrapped_form() {
    // ADR-0018 writes the signature wrapped, so the checked-in example does too, and this is
    // what makes that example a test of the lexer rather than only of the grammar.
    let src = std::fs::read_to_string(
        std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../examples/matmul.lyth"),
    )
    .expect("examples/matmul.lyth");
    assert!(
        src.contains("kernel matmul(m: u32, n: u32, k: u32,\n"),
        "the example should wrap its signature"
    );
    let u = parse(&src).expect("the matmul example parses");
    assert_eq!(u.kernels[0].params.len(), 6);
    assert!(u.kernels[0].contract.is_some());
}
