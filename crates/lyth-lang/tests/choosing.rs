//! `max` and `min` in a body: the one form of choosing that costs the same either way.
//!
//! ADR-0027. A branch is refused because the byte count would depend on which side ran. A
//! compare-and-select does not have that property: one instruction, the same one whichever
//! operand wins, the same traffic. It passes ADR-0026's test — *can the compiler still say what
//! the kernel costs* — and it is the only shape of choice that does.
//!
//! What it unlocks is not abstract: `max(x, 0)` is a ReLU, and a clamp is the one piece a
//! branch-free Ising solver was missing.

use lyth_lang::ast::{BinOp, ReduceOp, Ty};
use lyth_lang::{eval, ir, parse};

fn lower(src: &str) -> ir::KernelIr {
    let unit = parse(src).expect("parses");
    ir::lower(&unit, &unit.kernels[0]).expect("lowers")
}

const RELU: &str = "machine sm_120\n\n\
     kernel relu(n: u32, x: [f32; n], out: [f32; n])\n    \
     intensity 0.0\n    \
     stream x   : dram -> reg\n    \
     stream out : dram -> reg, drain\n    \
     at reg:\n        \
     out = max(x, 0.0)\n";

#[test]
fn a_relu_is_a_memory_pass_and_the_compiler_says_so() {
    // 0 flop/byte is not a degenerate answer, it is the right one: an activation retires no
    // arithmetic and moves two words. Anyone who has profiled an activation layer knows this,
    // and the language now derives it rather than leaving it to be discovered.
    let k = lower(RELU);
    assert_eq!(k.cost.bytes_per_element(), Some(8.0));
    assert_eq!(k.cost.intensity, 0.0);
}

#[test]
fn the_two_spellings_of_max_cost_the_same() {
    // `reduce max` and `max(a, b)` are one operation reached by two syntaxes. If they charged
    // different flops, a kernel's declared intensity would depend on which the author wrote,
    // and the contract would be about the spelling instead of about the work.
    assert_eq!(BinOp::Max.flops(), ReduceOp::Max.flops());
    assert_eq!(BinOp::Min.flops(), ReduceOp::Min.flops());
    assert_eq!(BinOp::Max.flops(), 0.0, "compare-and-select is not arithmetic");
}

#[test]
fn the_elementwise_max_agrees_with_the_reduction_on_signed_zeros() {
    // The pair of inputs where this is hardest to notice, and where `f32::max` is *unspecified*:
    // it lowers to `llvm.maxnum`, which may return either operand when they compare equal, and
    // `-0.0 == 0.0`. On this machine it constant-folds to +0.0 and executes to -0.0.
    //
    // 135 of 65536 block partials once differed from the device over exactly this (ADR-0013).
    // `BinOp::apply` therefore delegates rather than calling `a.max(b)`, and this test fails the
    // moment someone simplifies it back.
    for (a, b) in [(-0.0f32, 0.0f32), (0.0, -0.0)] {
        assert_eq!(
            BinOp::Max.apply(a, b).to_bits(),
            ReduceOp::Max.combine(a, b).to_bits()
        );
        assert_eq!(
            BinOp::Min.apply(a, b).to_bits(),
            ReduceOp::Min.combine(a, b).to_bits()
        );
    }
    assert!(BinOp::Max.apply(-0.0, 0.0).is_sign_positive(), "PTX max.f32 gives +0.0");
    assert!(BinOp::Min.apply(-0.0, 0.0).is_sign_negative());
}

#[test]
fn the_oracle_evaluates_it() {
    let k = lower(RELU);
    let n = 256;
    let mut inputs = eval::Inputs::default();
    for p in &k.params {
        if p.ty.is_buffer() {
            inputs
                .buffers
                .insert(p.name.clone(), lyth_lang::inputs::buffer(&p.name, p.ty, n));
        }
    }
    let x = inputs.buffers["x"].clone();
    let out = eval::eval(&k, n as usize, &inputs).expect("evaluates");

    // Asserted as a rate as well as a value: if the generator happened to produce only
    // positives, a ReLU that did nothing at all would pass elementwise.
    let clamped = x.iter().filter(|v| **v < 0.0).count();
    assert!(clamped > n as usize / 8, "only {clamped} of {n} inputs are negative");
    for (i, v) in x.iter().enumerate().take(n as usize) {
        assert_eq!(out.buffers["out"][i], v.max(0.0), "element {i}");
    }
}

#[test]
fn a_clamp_is_two_of_them_and_still_costs_nothing() {
    // `min(max(x, lo), hi)` -- the piece a branch-free Ising solver was missing. Nested calls
    // parse, and two compare-and-selects are still zero flops and still one pass over memory.
    let src = "machine sm_120\n\n\
         kernel clamp(n: u32, lo: f32, hi: f32, x: [f32; n], out: [f32; n])\n    \
         intensity 0.0\n    \
         stream x   : dram -> reg\n    \
         stream out : dram -> reg, drain\n    \
         at reg:\n        \
         out = min(max(x, lo), hi)\n";
    let k = lower(src);
    assert_eq!(k.cost.intensity, 0.0);
    assert_eq!(k.cost.bytes_per_element(), Some(8.0));
}

#[test]
fn max_is_not_a_reserved_word() {
    // The lookahead earns its keep. `max` is only an operator when a `(` follows it, so a
    // buffer called `max` is still a buffer -- which somebody will write, because it is the
    // obvious name for the output of a reduction.
    let src = "machine sm_120\n\n\
         kernel k(n: u32, max: [f32; n], out: [f32; n])\n    \
         intensity 0.0833\n    \
         stream max : dram -> reg\n    \
         stream out : dram -> reg, drain\n    \
         at reg:\n        \
         out = max + out\n";
    let k = lower(src);
    assert!(k.params.iter().any(|p| p.name == "max" && p.ty == Ty::BufF32));
}

#[test]
fn a_call_to_anything_else_is_not_a_call() {
    // There are no user functions, so `(` after any other name has to be an error rather than
    // a call to something that does not exist. The message should say a name was expected,
    // not that `sqrt` is undefined -- the language has no concept of a defined function to be
    // absent from.
    let src = "machine sm_120\n\n\
         kernel k(n: u32, x: [f32; n], out: [f32; n])\n    \
         intensity 0.0833\n    \
         stream x   : dram -> reg\n    \
         stream out : dram -> reg, drain\n    \
         at reg:\n        \
         out = sqrt(x) + out\n";
    assert!(parse(src).is_err(), "`sqrt(x)` must not parse");
}
