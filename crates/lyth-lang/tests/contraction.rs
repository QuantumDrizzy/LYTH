//! ADR-0018 step 1: `contract` parses and resolves, and every way of writing one that the
//! derivation does not cover is refused at the declaration rather than at the numbers.
//!
//! What is deliberately *not* here is a cost. `lower` ends in `ContractCostNotDerived` for
//! every kernel in this file, and one test pins that. A contraction moves `2K/T + 1` elements
//! per output, which is an expression in a launch extent rather than a constant, and a
//! compiler whose entire claim is that the number it publishes is the number the silicon moves
//! must not publish the constant part of it in the meantime.

use lyth_lang::ast::ReduceOp;
use lyth_lang::ir::{lower, KernelIr, LowerError};
use lyth_lang::parse::parse;

/// A matmul, with the pieces a test wants to vary spliced in.
fn matmul(decls: &str, body: &str) -> String {
    format!(
        "machine sm_120\n\n\
         kernel mm(m: u32, n: u32, k: u32, a: [f32; m, k], b: [f32; k, n], c: [f32; m, n])\n    \
         space i, j : m, n\n\
         {decls}    \
         stream a : dram -> reg\n    \
         stream b : dram -> reg\n    \
         stream c : dram -> reg, drain\n    \
         at reg:\n\
         {body}"
    )
}

const CONTRACT: &str = "    contract sum p : k\n";
const BODY: &str = "        c[i, j] = a[i, p] * b[p, j]\n";

fn lower_src(src: &str) -> Result<KernelIr, LowerError> {
    let u = parse(src).expect("should parse");
    lower(&u, &u.kernels[0])
}

fn err(src: &str) -> String {
    match lower_src(src) {
        Err(e) => e.to_string(),
        Ok(_) => panic!("expected a refusal, got a kernel"),
    }
}

#[test]
fn a_matmul_parses() {
    // The first half of step 1. Nothing in this language could express a contraction before,
    // so this is the grammar accepting one.
    let u = parse(&matmul(CONTRACT, BODY)).expect("a matmul should parse");
    let c = u.kernels[0].contract.as_ref().expect("a contract");
    assert_eq!(c.op, ReduceOp::Sum);
    assert_eq!(c.var, "p");
    assert_eq!(c.extent, "k");
}

#[test]
fn it_resolves_before_it_refuses_to_cost_itself() {
    // The refusal must be the *last* thing lowering does, or none of the checks below are
    // reachable and this file would be testing one error message fifteen times.
    let e = err(&matmul(CONTRACT, BODY));
    assert!(e.contains("cost is not derived yet"), "{e}");
    assert!(e.contains("2K/T + 1"), "{e}");
    assert!(e.contains("ADR-0018"), "{e}");
}

#[test]
fn all_three_operators_are_accepted() {
    // `combine` and `identity` are already defined for each, and a contraction is sequential
    // in a register: no tree, so not even the ordering question a `reduce` has. Refusing max
    // and min here would be a claim nobody derived.
    for op in ["sum", "max", "min"] {
        let src = matmul(&format!("    contract {op} p : k\n"), BODY);
        let u = parse(&src).expect("should parse");
        assert_eq!(
            u.kernels[0].contract.as_ref().unwrap().op,
            ReduceOp::parse(op).unwrap()
        );
        // And it reaches the cost, which is where every contraction currently ends.
        assert!(err(&src).contains("cost is not derived yet"));
    }
}

#[test]
fn a_word_that_is_not_an_operator_is_refused_where_it_is_written() {
    let src = matmul("    contract product p : k\n", BODY);
    let e = parse(&src).expect_err("product is not an operator");
    assert!(e.to_string().contains("sum, max or min"), "{e}");
}

#[test]
fn two_contracted_axes_are_refused_rather_than_half_read() {
    // `contract sum p, q : k` would otherwise parse as `p : k` with `, q` left over, and the
    // second axis would silently not exist.
    let src = matmul("    contract sum p, q : k\n", BODY);
    let e = parse(&src).expect_err("rank-2 contraction");
    assert!(e.to_string().contains("rank-2 contraction"), "{e}");
}

#[test]
fn a_second_contract_is_refused() {
    let src = matmul("    contract sum p : k\n    contract sum q : k\n", BODY);
    let e = parse(&src).expect_err("two contracts");
    assert!(e.to_string().contains("declared twice"), "{e}");
}

#[test]
fn the_contracted_variable_cannot_be_one_the_space_iterates() {
    // `contract sum i : k` under `space i, j`. Free and contracted are the two things an axis
    // can be, and this asks for both.
    let src = matmul(
        "    contract sum i : k\n",
        "        c[i, j] = a[i, j] * b[i, j]\n",
    );
    let e = err(&src);
    assert!(e.contains("the space already iterates"), "{e}");
}

#[test]
fn the_contracted_extent_cannot_be_one_the_space_runs_over() {
    let src = matmul("    contract sum p : m\n", BODY);
    let e = err(&src);
    assert!(e.contains("the space already runs over"), "{e}");
}

#[test]
fn the_contracted_extent_must_be_a_parameter() {
    let src = matmul("    contract sum p : depth\n", BODY);
    let e = err(&src);
    assert!(e.contains("not a u32 parameter"), "{e}");
}

#[test]
fn the_contracted_variable_cannot_shadow_a_parameter() {
    let src = matmul(
        "    contract sum n : k\n",
        "        c[i, j] = a[i, n] * b[n, j]\n",
    );
    let e = err(&src);
    assert!(e.contains("is a parameter of this kernel"), "{e}");
}

#[test]
fn a_contraction_nothing_is_indexed_at_is_refused() {
    // The declaration would multiply the derived traffic by K while the body reads each
    // element once. Nothing downstream could tell.
    let src = matmul(CONTRACT, "        c[i, j] = a[i, j] * b[i, j]\n");
    let e = err(&src);
    assert!(e.contains("no buffer is indexed at `p`"), "{e}");
}

#[test]
fn a_contraction_over_a_single_buffer_is_refused_and_says_why() {
    // `c[i,j] = a[i,p] + b[i,j]` reduces `a` along an axis. It is a real kernel; it is not the
    // one ADR-0018 derives, because with one operand there is no reuse and `2K/T` is not its
    // traffic. Refused rather than costed by a formula that does not describe it.
    let src = matmul(CONTRACT, "        c[i, j] = a[i, p] + b[i, j]\n");
    let e = err(&src);
    assert!(e.contains("walks only `a`"), "{e}");
    assert!(e.contains("no reuse"), "{e}");
}

#[test]
fn writing_the_target_at_the_contracted_axis_is_refused() {
    // Each of the K terms would be stored in turn and the last would win: a kernel that runs,
    // returns numbers, and computes something nobody asked for.
    let src = "machine sm_120\n\n\
               kernel mm(m: u32, n: u32, k: u32, a: [f32; m, k], b: [f32; k, n], c: [f32; m, k])\n    \
               space i, j : m, n\n    \
               contract sum p : k\n    \
               stream a : dram -> reg\n    \
               stream b : dram -> reg\n    \
               stream c : dram -> reg, drain\n    \
               at reg:\n        \
               c[i, p] = a[i, p] * b[p, j]\n";
    let e = err(src);
    assert!(e.contains("is written at `p`"), "{e}");
    assert!(e.contains("keep the last"), "{e}");
}

#[test]
fn a_contraction_needs_a_space() {
    let src = "machine sm_120\n\n\
               kernel r(n: u32, k: u32, x: [f32; n], y: [f32; n])\n    \
               contract sum p : k\n    \
               stream x : dram -> reg\n    \
               stream y : dram -> reg, drain\n    \
               at reg:\n        \
               y = x\n";
    let e = err(src);
    assert!(e.contains("`contract` needs a `space`"), "{e}");
}

#[test]
fn a_kernel_cannot_both_reduce_and_contract() {
    let src = "machine sm_120\n\n\
               kernel rc(m: u32, n: u32, k: u32, a: [f32; m, k], b: [f32; k, n], partial: [f32; blocks])\n    \
               space i, j : m, n\n    \
               contract sum p : k\n    \
               stream a : dram -> reg\n    \
               stream b : dram -> reg\n    \
               reduce sum v : reg -> smem -> dram into partial\n    \
               at reg:\n        \
               v = a[i, p] * b[p, j]\n";
    let e = err(src);
    assert!(e.contains("cannot both `reduce` and `contract`"), "{e}");
    assert!(e.contains("different machines"), "{e}");
}

#[test]
fn an_index_may_name_the_contracted_variable_at_most_once() {
    // `a[p, p]` addresses a diagonal, which this index form cannot express, and which would
    // otherwise pass a check that only asked whether each name was in the alphabet.
    let src = matmul(CONTRACT, "        c[i, j] = a[p, p] * b[p, j]\n");
    let e = err(&src);
    assert!(e.contains("p, p"), "{e}");
}

#[test]
fn the_permutation_rule_is_unchanged_for_a_kernel_without_a_contraction() {
    // The alphabet grows only when a `contract` grows it. A transpose must still refuse an
    // index naming anything but its own space variables.
    let src = "machine sm_120\n\n\
               kernel t(rows: u32, cols: u32, a: [f32; rows, cols], b: [f32; cols, rows])\n    \
               space i, j : rows, cols\n    \
               stream a : dram -> reg\n    \
               stream b : dram -> reg, drain\n    \
               at reg:\n        \
               b[j, i] = a[i, p]\n";
    let e = err(src);
    assert!(e.contains("i, p"), "{e}");
    assert!(!e.contains("contract"), "no contraction was declared: {e}");
}
