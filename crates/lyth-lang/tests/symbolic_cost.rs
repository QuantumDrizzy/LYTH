//! ADR-0018 step 2: the cost stops being a number.
//!
//! The thing being tested is not that a matmul derives `0.25 * k + 4`. It is that **nothing
//! derives that by recognising a matmul**. Each stream is asked what it costs and the
//! expression is the sum, so a schedule that is one staging short, or has a rectangular tile,
//! or no tile at all, produces a different expression without anybody writing one down. If
//! `2K/T + 1` were reached by pattern-matching `contract` against `tile`, every test here
//! except the first would fail -- which is the point of every test here except the first.

use lyth_lang::ir::{lower, Cost, KernelIr};
use lyth_lang::parse::parse;

/// A matmul whose declarations a test can vary.
fn matmul(decls: &str) -> String {
    format!(
        "machine sm_120\n\n\
         kernel mm(m: u32, n: u32, k: u32,\n              \
         a: [f32; m, k], b: [f32; k, n], c: [f32; m, n])\n    \
         space i, j : m, n\n    \
         contract sum p : k\n\
         {decls}    \
         at reg:\n        \
         c[i, j] = a[i, p] * b[p, j]\n"
    )
}

/// Both operands staged under a square tile: the schedule ADR-0018 derives.
const TILED: &str = "    tile 32, 32\n    \
                     stream a : dram -> smem -> reg\n    \
                     stream b : dram -> smem -> reg\n    \
                     stream c : dram -> reg, drain\n";

fn ir(src: &str) -> KernelIr {
    let u = parse(src).expect("should parse");
    lower(&u, &u.kernels[0]).expect("should lower")
}

fn cost(decls: &str) -> Cost {
    ir(&matmul(decls)).cost
}

#[test]
fn a_tiled_matmul_derives_the_expression_adr_0018_wrote_down() {
    let c = cost(TILED);
    // 2K/T + 1 elements per output, in bytes: 4 * (2k/32) + 4 = 0.25k + 4.
    assert_eq!(c.bytes_expr().as_deref(), Some("0.25 * k + 4"));
    assert_eq!(c.flops_expr().as_deref(), Some("2 * k"));
    // The limit, which is T/4 and is what a source may declare.
    assert!((c.intensity - 8.0).abs() < 1e-12, "{}", c.intensity);
    assert!(c.asymptotic);
    // And the exact figure at a concrete extent, which is what a launch can be held to.
    assert_eq!(c.bytes_at(4096), 1028.0);
    assert_eq!(c.flops_at(4096), 8192.0);
    assert!((c.intensity_at(4096) - 7.9689).abs() < 1e-4);
}

#[test]
fn the_constant_form_is_unavailable_when_the_cost_is_not_constant() {
    // The step-1 refusal, now enforced by the type rather than by a check. The constant part
    // of a matmul is 4 bytes and 0 flops: two numbers that would print without complaint.
    let c = cost(TILED);
    assert_eq!(c.bytes_per_element(), None);
    assert_eq!(c.flops_per_element(), None);
    assert_eq!(c.bytes_fixed(), 4.0, "the write of c is the whole fixed part");
}

#[test]
fn staging_one_operand_and_not_the_other_gives_a_different_expression() {
    // The test this file exists for. `b` is left in `dram -> reg` inside the same tile, so
    // every thread of the tile issues its own load of `b[p, j]` and there is no reuse: k
    // elements per output instead of k/32.
    //
    // A deriver that emitted `2K/T + 1` on seeing `contract` and `tile` together would report
    // 0.25k + 4 here, and be wrong by a factor of 16 in the traffic and 16 in the intensity.
    let c = cost(
        "    tile 32, 32\n    \
         stream a : dram -> smem -> reg\n    \
         stream b : dram -> reg\n    \
         stream c : dram -> reg, drain\n",
    );
    // 4/32 for a, 4 for b: 4.125 per unit of k.
    assert_eq!(c.bytes_expr().as_deref(), Some("4.125 * k + 4"));
    assert!((c.intensity - 2.0 / 4.125).abs() < 1e-12, "{}", c.intensity);

    let terms = &c.contracted.as_ref().unwrap().terms;
    let a = terms.iter().find(|t| t.buffer == "a").unwrap();
    let b = terms.iter().find(|t| t.buffer == "b").unwrap();
    assert_eq!(a.reuse, 32, "a is staged, so a tile column shares it");
    assert_eq!(b.reuse, 1, "b is not staged, so nothing shares it");
}

#[test]
fn reuse_is_earned_by_staging_and_not_by_tiling() {
    // The same tile, neither operand staged. The threads exist; nothing makes them share.
    let c = cost(
        "    tile 32, 32\n    \
         stream a : dram -> reg\n    \
         stream b : dram -> reg\n    \
         stream c : dram -> reg, drain\n",
    );
    assert_eq!(c.bytes_expr().as_deref(), Some("8 * k + 4"));
    for t in &c.contracted.as_ref().unwrap().terms {
        assert_eq!(t.reuse, 1, "{} earned reuse it was not given", t.buffer);
    }
}

#[test]
fn a_rectangular_tile_gives_each_operand_its_own_reuse() {
    // `a[i, p]` is invariant along `j`, so it is shared by `tile[j]` threads; `b[p, j]` is
    // invariant along `i` and shared by `tile[i]`. A square tile makes those the same number,
    // which is exactly how a wrong rule survives a first test.
    let c = cost(
        "    tile 16, 64\n    \
         stream a : dram -> smem -> reg\n    \
         stream b : dram -> smem -> reg\n    \
         stream c : dram -> reg, drain\n",
    );
    let terms = &c.contracted.as_ref().unwrap().terms;
    let a = terms.iter().find(|t| t.buffer == "a").unwrap();
    let b = terms.iter().find(|t| t.buffer == "b").unwrap();
    assert_eq!(a.reuse, 64, "a is invariant along j, whose tile is 64");
    assert_eq!(b.reuse, 16, "b is invariant along i, whose tile is 16");
    // 4/64 + 4/16 = 0.0625 + 0.25
    assert_eq!(c.bytes_expr().as_deref(), Some("0.3125 * k + 4"));
}

#[test]
fn a_contraction_with_no_tile_is_the_untiled_control_and_not_a_special_case() {
    // T = 1. The same expression gives 2k + ... per output, which is a real schedule: one
    // thread per output, reading a whole row and a whole column. It falls out rather than
    // being written down, which is what makes it a control.
    let c = cost(
        "    stream a : dram -> reg\n    \
         stream b : dram -> reg\n    \
         stream c : dram -> reg, drain\n",
    );
    assert_eq!(c.bytes_expr().as_deref(), Some("8 * k + 4"));
    assert_eq!(c.bytes_at(4096), 32772.0);
    assert!((c.intensity - 0.25).abs() < 1e-12, "{}", c.intensity);
}

#[test]
fn the_combinator_changes_the_flops_and_not_the_traffic() {
    // `max` retires no flops -- a compare-and-select is not arithmetic (ADR-0013) -- so a
    // max-contraction does one flop per step, the multiply, and its limit is T/8. The bytes
    // are identical, because the same elements move whatever is done to them.
    let sum = cost(TILED);
    let src = matmul(TILED).replace("contract sum p", "contract max p");
    let u = parse(&src).expect("parses");
    let max = lower(&u, &u.kernels[0]).expect("lowers").cost;

    assert_eq!(max.bytes_expr(), sum.bytes_expr());
    assert_eq!(max.flops_expr().as_deref(), Some("1 * k"));
    assert!((max.intensity - 4.0).abs() < 1e-12, "{}", max.intensity);
}

#[test]
fn a_kernel_without_a_contraction_is_bit_for_bit_what_it_was() {
    // Every per-extent coefficient is zero, so `bytes_at` at any k is the constant, and the
    // expression accessors stay silent. This is what makes the change invisible to the other
    // fourteen examples.
    let src = "machine sm_120\n\n\
               kernel saxpy(n: u32, a: f32, x: [f32; n], y: [f32; n])\n    \
               stream x : dram -> reg\n    \
               stream y : dram -> reg, drain\n    \
               at reg:\n        \
               y = a * x + y\n";
    let c = ir(src).cost;
    assert_eq!(c.bytes_per_element(), Some(12.0));
    assert_eq!(c.flops_per_element(), Some(2.0));
    assert_eq!(c.bytes_expr(), None);
    assert!(!c.asymptotic);
    assert!(c.contracted.is_none());
    for k in [0u32, 1, 4096, u32::MAX] {
        assert_eq!(c.bytes_at(k), 12.0, "k must not enter a constant cost");
        assert_eq!(c.flops_at(k), 2.0);
    }
}

#[test]
fn a_contracted_kernel_must_declare_a_limit_and_say_so() {
    // `intensity 8.0` on a matmul would be the exact claim an elementwise kernel makes,
    // written by a source that cannot make it: the exact figure is 7.9689 at k = 4096 and
    // 7.8431 at k = 2048. The word is what keeps the weaker claim from travelling as the
    // stronger one.
    let src = matmul(&format!("    intensity 8.0\n{TILED}"));
    let u = parse(&src).expect("parses");
    let e = lower(&u, &u.kernels[0]).expect_err("a bare intensity on a contraction");
    let e = e.to_string();
    assert!(e.contains("intensity asymptotic 8.0000"), "{e}");
    assert!(e.contains("function of a launch extent"), "{e}");

    let ok = matmul(&format!("    intensity asymptotic 8.0\n{TILED}"));
    let u = parse(&ok).expect("parses");
    lower(&u, &u.kernels[0]).expect("the limit is declarable");
}

#[test]
fn a_constant_kernel_may_not_declare_a_limit() {
    // The same rule in the other direction. Saxpy's intensity is exactly 0.1667 and calling
    // it a limit is the same number wearing a weaker claim, which is a claim nobody derived.
    let src = "machine sm_120\n\n\
               kernel saxpy(n: u32, a: f32, x: [f32; n], y: [f32; n])\n    \
               intensity asymptotic 0.1667\n    \
               stream x : dram -> reg\n    \
               stream y : dram -> reg, drain\n    \
               at reg:\n        \
               y = a * x + y\n";
    let u = parse(src).expect("parses");
    let e = lower(&u, &u.kernels[0]).expect_err("asymptotic without a contraction");
    assert!(e.to_string().contains("does not contract"), "{e}");
}

#[test]
fn coalescence_compares_like_with_like() {
    // Both halves of the payload against both halves of the sector figure. A ratio that took
    // a per-output numerator over a per-step denominator gave 0.333 for a matmul whose every
    // access is absorbed into shared memory and coalesced -- a number that looks like a
    // finding and is an accounting mistake.
    let c = cost(TILED);
    assert!((c.coalescence() - 1.0).abs() < 1e-12, "{}", c.coalescence());
}
