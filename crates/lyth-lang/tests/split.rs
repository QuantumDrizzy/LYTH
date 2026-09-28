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
