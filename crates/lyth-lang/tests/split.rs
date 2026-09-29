//! ADR-0028 step 1: `split` in the parser and the IR, and its refusals.
//!
//! A view is a name, not a parameter: the launch signature keeps the base buffers only, the
//! body names views, and `buffer_param` resolves a view to its base.

use lyth_lang::{ir, parse};

fn lower(src: &str) -> Result<ir::KernelIr, String> {
    let u = parse(src).map_err(|e| e.to_string())?;
    ir::lower(&u, &u.kernels[0]).map_err(|e| e.to_string())
}

fn example() -> String {
    std::fs::read_to_string(std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../examples/hadamard_q.lyth"))
        .unwrap()
}

#[test]
fn hadamard_q_lowers_with_eight_views_and_four_base_parameters() {
    let k = lower(&example()).expect("lowers");
    assert_eq!(k.views.len(), 8);
    let buffers: Vec<&str> = k.params.iter().filter(|p| p.ty.is_buffer()).map(|p| p.name.as_str()).collect();
    assert_eq!(buffers, ["re", "im", "qr", "qi"], "the signature carries the bases, never the views");
    let v = k.view("p1r").unwrap();
    assert_eq!((v.base.as_str(), v.part, v.width.as_str()), ("re", 1, "w"));
    assert_eq!(k.buffer_param("q0i").unwrap().name, "qi");
    assert_eq!(k.buffer_param("re").unwrap().name, "re");
}

#[test]
fn the_split_does_not_move_the_flop_count_or_the_payload() {
    // ADR-0028: "the body is unchanged; hadamard_q derives the same 8 flop / 32 byte per pair as
    // hadamard, intensity 0.25". Steps 1-3 must not move a derived number.
    let root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let h = lower(&std::fs::read_to_string(root.join("examples/hadamard.lyth")).unwrap()).unwrap();
    let q = lower(&example()).unwrap();
    assert_eq!(format!("{:?}", h.cost), format!("{:?}", q.cost), "cost must be identical");
    assert_eq!(h.ops, q.ops, "the body lowers to the same ops");
}

fn with(split_lines: &str, streams: &str) -> String {
    format!(
        "machine sm_120\nkernel k(n: u32, w: u32, s: f32, re: [f32; n], qr: [f32; n], x: [f32; n])\n    intensity 0.25\n\n{split_lines}\n{streams}\n    at reg:\n        q0r = s * (p0r + p1r)\n        q1r = s * (p0r - p1r)\n"
    )
}

const OK_SPLITS: &str = "    split re into p0r, p1r : blocks w\n    split qr into q0r, q1r : blocks w";
const OK_STREAMS: &str = "    stream p0r : dram -> reg\n    stream p1r : dram -> reg\n    stream q0r : dram -> reg, drain\n    stream q1r : dram -> reg, drain";

#[test]
fn the_fixture_itself_lowers() {
    lower(&with(OK_SPLITS, OK_STREAMS)).expect("a two-split kernel lowers");
}

#[test]
fn refusals_name_what_is_wrong() {
    let cases: [(&str, &str, &str); 8] = [
        ("    split nothing into p0r, p1r : blocks w\n    split qr into q0r, q1r : blocks w", OK_STREAMS, "not a buffer parameter"),
        ("    split re into p0r, p1r : blocks s\n    split qr into q0r, q1r : blocks w", OK_STREAMS, "not a u32 parameter"),
        ("    split re into p0r, x : blocks w\n    split qr into q0r, q1r : blocks w", OK_STREAMS, "already taken"),
        ("    split re into p0r, p1r : blocks w\n    split re into a, b : blocks w\n    split qr into q0r, q1r : blocks w", OK_STREAMS, "split twice"),
        ("    split re into p0r, p1r : blocks w\n    split p0r into a, b : blocks w\n    split qr into q0r, q1r : blocks w", OK_STREAMS, "split twice"),
        (OK_SPLITS, "    stream re : dram -> reg\n    stream p0r : dram -> reg\n    stream p1r : dram -> reg\n    stream q0r : dram -> reg, drain\n    stream q1r : dram -> reg, drain", "never the base"),
        (OK_SPLITS, "    stream x : dram -> reg\n    stream p0r : dram -> reg\n    stream p1r : dram -> reg\n    stream q0r : dram -> reg, drain\n    stream q1r : dram -> reg, drain", "streams only views"),
        ("    split re into p0r, p1r : blocks w\n    split qr into q0r, q1r : blocks w\n    space i : n", OK_STREAMS, "rank 1 and elementwise"),
    ];
    for (splits, streams, want) in cases {
        let err = lower(&with(splits, streams)).expect_err(want);
        assert!(err.contains(want), "expected `{want}` in: {err}");
    }
}

#[test]
fn a_three_way_split_is_a_parse_error() {
    let src = with("    split re into p0r, p1r, p2r : blocks w\n    split qr into q0r, q1r : blocks w", OK_STREAMS);
    let err = parse(&src).expect_err("three views").to_string();
    assert!(err.contains("more than two views"), "{err}");
}

#[test]
fn an_in_place_gate_is_refused() {
    // Read p0r and p1r, drain p0r: two views of one base, one written, one read.
    let src = "machine sm_120\nkernel k(n: u32, w: u32, s: f32, re: [f32; n])\n    intensity 0.25\n\n    split re into p0r, p1r : blocks w\n    stream p0r : dram -> reg, drain\n    stream p1r : dram -> reg\n    at reg:\n        p0r = s * (p0r + p1r)\n";
    let err = lower(src).expect_err("in place");
    assert!(err.contains("in-place gate"), "{err}");
}

// ------------------------------------------------------------------ step 2: launch checks

use lyth_lang::ir::view_index;
use lyth_lang::program::{resolve, Program, ProgramError};

fn launch(n: u64, w: u64, tail: &str) -> Result<Program, ProgramError> {
    let src = format!("{}\nmain:\n    run hadamard_q(n = {n}, w = {w}, s = 0.5)\n{tail}", example());
    let unit = parse(&src).expect("parses");
    let ir = ir::lower(&unit, &unit.kernels[0]).expect("lowers");
    resolve(&unit, unit.main.as_ref().expect("has a main"), &ir)
}

#[test]
fn the_views_cover_the_base_exactly_when_2w_divides_n() {
    // "Checked, not trusted": enumerate the address rule instead of arguing about it.
    for n in 1..=64u32 {
        for w in 1..=40u32 {
            let mut hit = vec![0u8; n as usize + 2 * w as usize];
            for part in 0..2u8 {
                for k in 0..n / 2 {
                    hit[view_index(w, part, k) as usize] += 1;
                }
            }
            let covers = hit[..n as usize].iter().all(|&h| h == 1) && hit[n as usize..].iter().all(|&h| h == 0);
            let divides = n % (2 * w) == 0;
            // Divisible => an exact cover, no repeats, nothing outside. The converse is what the
            // launch check buys: a non-divisible launch is never an exact cover.
            if divides {
                assert!(covers, "n = {n}, w = {w}: 2w divides n, so the views must cover the base exactly");
            } else {
                assert!(!covers, "n = {n}, w = {w}: not divisible, yet an exact cover");
            }
        }
    }
}

#[test]
fn the_top_qubit_is_the_special_case_w_equals_n_over_2() {
    // ADR-0028: at w = n/2 the two views are the two contiguous halves, and hadamard_q IS hadamard.
    let n = 32u32;
    for k in 0..n / 2 {
        assert_eq!(view_index(n / 2, 0, k), u64::from(k));
        assert_eq!(view_index(n / 2, 1, k), u64::from(n / 2 + k));
    }
    // And w = 1 interleaves amplitude-by-amplitude: qubit 0.
    assert_eq!(view_index(1, 0, 5), 10);
    assert_eq!(view_index(1, 1, 5), 11);
}

#[test]
fn a_valid_launch_walks_pairs_and_its_buffers_stay_whole() {
    let p = launch(1024, 4, "    print qr[0:1024]").expect("resolves");
    assert_eq!(p.n, 512, "a split kernel walks pairs");
    assert_eq!(p.extents["n"], 1024);
    assert_eq!(p.extents["w"], 4);
    assert_eq!(p.prints[0].hi, 1024, "the printed buffer is the whole base, not half of it");
    // Printing past the base is still refused.
    assert!(matches!(launch(1024, 4, "    print qr[0:1025]"), Err(ProgramError::PrintOutOfRange { .. })));
}

#[test]
fn a_wrong_w_is_a_refusal_with_the_arithmetic_printed() {
    let cases: [(u64, u64, &str); 4] = [
        (12, 8, "2 * w = 16 must divide 12 but 12 mod 16 = 12"),
        (1000, 16, "2 * w = 32 must divide 1000 but 1000 mod 32 = 8"),
        (7, 1, "2 * w = 2 must divide 7 but 7 mod 2 = 1"),
        (4, 8, "hang past the end of the buffer"),
    ];
    for (n, w, want) in cases {
        let err = launch(n, w, "    print qr[0:1]").expect_err(want).to_string();
        assert!(err.contains(want), "n = {n}, w = {w}: expected `{want}` in: {err}");
        assert!(err.contains("ADR-0028"), "{err}");
    }
    // A leftover tail is named as such when the buffer is longer than one block.
    assert!(launch(1000, 16, "    print qr[0:1]").unwrap_err().to_string().contains("leave the last elements in neither view"));
}

#[test]
fn every_split_base_must_be_the_same_length() {
    let src = "machine sm_120\nkernel k(n: u32, m: u32, w: u32, s: f32, re: [f32; n], qr: [f32; m])\n    intensity 0.25\n\n    split re into p0r, p1r : blocks w\n    split qr into q0r, q1r : blocks w\n    stream p0r : dram -> reg\n    stream p1r : dram -> reg\n    stream q0r : dram -> reg, drain\n    stream q1r : dram -> reg, drain\n    at reg:\n        q0r = s * (p0r + p1r)\n        q1r = s * (p0r - p1r)\n";
    let err = lower(src).expect_err("n and m differ");
    assert!(err.contains("same length"), "{err}");
}
