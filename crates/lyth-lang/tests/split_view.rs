//! ADR-0029 step 1: a split of a view -- two-qubit gates -- in the IR, its launch check at every
//! level, and its refusals.
//!
//! The composed address is checked against **bit arithmetic**, not against `view_index`: leaf
//! `pXY` of a split at `(c, t)` must hold exactly the amplitudes whose bit `c` is `X` and bit `t`
//! is `Y`, in index order. That is the physics meaning of the declaration, and nothing in the
//! reference below divides or calls the implementation.

use std::collections::BTreeMap;

use lyth_lang::ir::{self, KernelIr};
use lyth_lang::parse;

fn root() -> std::path::PathBuf {
    std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn lower(src: &str) -> Result<KernelIr, String> {
    let u = parse(src).map_err(|e| e.to_string())?;
    ir::lower(&u, &u.kernels[0]).map_err(|e| e.to_string())
}

fn example(name: &str) -> KernelIr {
    lower(&std::fs::read_to_string(root().join(format!("examples/{name}.lyth"))).unwrap()).expect("lowers")
}

/// The widths a caller passes for control `c` and target `t` (ADR-0029): the inner width counts
/// elements of the parent view, which has lost bit `c`.
fn widths(c: u32, t: u32) -> (u32, u32) {
    (1 << c, if t < c { 1 << t } else { 1 << (t - 1) })
}

fn extents(n: u32, c: u32, t: u32) -> BTreeMap<String, u32> {
    let (wc, wt) = widths(c, t);
    BTreeMap::from([("n".into(), n), ("wc".into(), wc), ("wt".into(), wt)])
}

/// `k` with a zero inserted at bit `b`: the bits of `k` at and above `b` move up by one.
fn insert_zero(k: u64, b: u32) -> u64 {
    let low = k & ((1 << b) - 1);
    ((k >> b) << (b + 1)) | low
}

#[test]
fn cu_q_lowers_with_leaves_at_depth_two_and_four_base_parameters() {
    let k = example("cu_q");
    assert_eq!(k.views.len(), 24, "8 halves and 16 leaves");
    let buffers: Vec<&str> = k.params.iter().filter(|p| p.ty.is_buffer()).map(|p| p.name.as_str()).collect();
    assert_eq!(buffers, ["re", "im", "qr", "qi"], "the signature carries the buffers, never a view");
    assert_eq!(k.walk_depth(), 2);
    assert_eq!((k.depth("pr10"), k.depth("hr1"), k.depth("re")), (2, 1, 0));
    assert_eq!(k.root_of("pi01"), "im");
    assert_eq!(k.buffer_param("qr_11").unwrap().name, "qr");
    let v = k.view("pr10").unwrap();
    assert_eq!((v.base.as_str(), v.part, v.width.as_str()), ("hr1", 0, "wt"));
}

#[test]
fn the_nested_split_does_not_move_the_flop_count_or_the_payload() {
    // P1 in the model: cu_q derives what the contiguous control derives, per quad.
    let (q, c) = (example("cu_q"), example("cu"));
    assert_eq!(q.ops, c.ops, "the body lowers to the same ops");
    assert_eq!(q.cost.flops_per_element(), Some(28.0));
    assert_eq!(q.cost.bytes_per_element(), Some(64.0));
    assert_eq!(q.cost.flops_per_element(), c.cost.flops_per_element());
    assert_eq!(q.cost.bytes_per_element(), c.cost.bytes_per_element());
    assert_eq!(example("swap_q").cost.bytes_per_element(), Some(64.0));
    assert_eq!(example("swap_q").cost.flops_per_element(), Some(0.0));
}

#[test]
fn every_leaf_holds_exactly_the_amplitudes_its_two_bits_name() {
    // Enumerated for every ordered pair of qubits of a 7-qubit register, both nesting orders.
    let k = example("cu_q");
    let q = 7u32;
    let n = 1u32 << q;
    let mut pairs = 0;
    for c in 0..q {
        for t in 0..q {
            if c == t {
                continue;
            }
            let ex = extents(n, c, t);
            assert_eq!(k.split_pairs(&ex).unwrap(), Some(n / 4), "c = {c}, t = {t}: the walk is quads");
            let (lo, hi) = (c.min(t), c.max(t));
            let mut seen = vec![false; n as usize];
            for (x, y) in [(0u64, 0u64), (0, 1), (1, 0), (1, 1)] {
                let leaf = format!("pr{x}{y}");
                for kk in 0..u64::from(n / 4) {
                    let got = k.base_index(&leaf, &ex, kk).unwrap();
                    let want = insert_zero(insert_zero(kk, lo), hi) | (x << c) | (y << t);
                    assert_eq!(got, want, "c = {c}, t = {t}, leaf {leaf}, k = {kk}");
                    assert!(!seen[got as usize], "c = {c}, t = {t}: {got} is in two leaves");
                    seen[got as usize] = true;
                }
            }
            assert!(seen.iter().all(|s| *s), "c = {c}, t = {t}: the leaves do not cover the buffer");
            pairs += 1;
        }
    }
    assert_eq!(pairs, 42);
}

#[test]
fn depth_one_is_unchanged() {
    // ADR-0028's kernel walks pairs and its address is `view_index` itself.
    let k = example("hadamard_q");
    let ex = BTreeMap::from([("n".to_string(), 96u32), ("w".to_string(), 3u32)]);
    assert_eq!(k.walk_depth(), 1);
    assert_eq!(k.split_pairs(&ex).unwrap(), Some(48));
    for kk in 0..48u32 {
        assert_eq!(k.base_index("p1r", &ex, u64::from(kk)), Some(ir::view_index(3, 1, kk)));
    }
}

#[test]
fn a_level_that_does_not_cover_its_parent_is_refused_with_its_own_arithmetic() {
    let k = example("cu_q");
    // Level 1: 2 * wc = 16 does not divide n = 24.
    let ex = BTreeMap::from([("n".to_string(), 24u32), ("wc".to_string(), 8u32), ("wt".to_string(), 1u32)]);
    let err = k.split_pairs(&ex).unwrap_err().to_string();
    assert!(err.contains("2 * wc = 16 must divide 24"), "{err}");
    // Level 2: the halves have n / 2 = 12 elements, and 2 * wt = 8 does not divide 12.
    let ex = BTreeMap::from([("n".to_string(), 24u32), ("wc".to_string(), 3u32), ("wt".to_string(), 4u32)]);
    let err = k.split_pairs(&ex).unwrap_err().to_string();
    assert!(err.contains("n / 2 = 12") && err.contains("2 * wt = 8 must divide 12"), "{err}");
    // And a width of zero at the inner level.
    let ex = BTreeMap::from([("n".to_string(), 24u32), ("wc".to_string(), 3u32), ("wt".to_string(), 0u32)]);
    assert!(k.split_pairs(&ex).unwrap_err().to_string().contains("wt = 0"));
}

/// A two-buffer kernel with the given splits, streams and body.
fn kernel(splits: &str, streams: &str, body: &str) -> String {
    format!(
        "machine sm_120\nkernel k(n: u32, wc: u32, wt: u32, re: [f32; n], qr: [f32; n])\n    intensity 0.0\n\n{splits}\n{streams}\n    at reg:\n{body}"
    )
}

const TWO_LEVELS: &str = "    split re into h0, h1 : blocks wc\n    split h0 into p00, p01 : blocks wt\n    split h1 into p10, p11 : blocks wt\n    split qr into g0, g1 : blocks wc\n    split g0 into q00, q01 : blocks wt\n    split g1 into q10, q11 : blocks wt";

#[test]
fn the_rules_of_the_split_of_a_view_refuse_with_the_reason() {
    let quad_streams = "    stream p00 : dram -> reg\n    stream p11 : dram -> reg\n    stream q00 : dram -> reg, drain\n    stream q11 : dram -> reg, drain";
    let quad_body = "        q00 = p11\n        q11 = p00\n";
    lower(&kernel(TWO_LEVELS, quad_streams, quad_body)).expect("the fixture lowers");

    // Depth 3 is its own amendment.
    let deep = format!("{TWO_LEVELS}\n    split p00 into x, y : blocks wt");
    let err = lower(&kernel(&deep, quad_streams, quad_body)).unwrap_err();
    assert!(err.contains("depth 3"), "{err}");

    // A pair and a quad in one kernel: two walks.
    let mixed = "    stream h1 : dram -> reg\n    stream p00 : dram -> reg\n    stream q00 : dram -> reg, drain\n    stream q11 : dram -> reg, drain";
    let err = lower(&kernel(TWO_LEVELS.replace("    split h1 into p10, p11 : blocks wt\n", "").as_str(), mixed, "        q00 = h1\n        q11 = p00\n")).unwrap_err();
    assert!(err.contains("same depth"), "{err}");

    // The view that was split is not streamed: its leaves are.
    let halves = "    stream h0 : dram -> reg\n    stream p11 : dram -> reg\n    stream q00 : dram -> reg, drain\n    stream q11 : dram -> reg, drain";
    let err = lower(&kernel(TWO_LEVELS, halves, "        q00 = h0\n        q11 = p11\n")).unwrap_err();
    assert!(err.contains("never the base"), "{err}");

    // Splitting one view twice is still one name with two address rules.
    let twice = format!("{TWO_LEVELS}\n    split h0 into a, b : blocks wt");
    let err = lower(&kernel(&twice, quad_streams, quad_body)).unwrap_err();
    assert!(err.contains("split twice"), "{err}");
}

#[test]
fn in_place_is_refused_across_different_parents_of_one_buffer() {
    // p00 (under h0) is read and p10 (under h1) is drained: different parents, one buffer.
    let src = kernel(
        "    split re into h0, h1 : blocks wc\n    split h0 into p00, p01 : blocks wt\n    split h1 into p10, p11 : blocks wt",
        "    stream p00 : dram -> reg\n    stream p10 : dram -> reg, drain",
        "        p10 = p00\n",
    );
    let err = lower(&src).unwrap_err();
    assert!(err.contains("in-place gate") && err.contains("`re`"), "{err}");
}

// ------------------------------------------------------------------ step 2: the sector figure

#[test]
fn the_isolated_sector_figure_walks_the_composed_address() {
    // Per quad, each leaf alone. When both qubits are at 3 or above, every run of every leaf is
    // at least eight f32 -- one whole sector -- and the figure is the payload's. Below that a
    // warp's leaf straddles sectors, and the figure is printed as what it is: an upper bound,
    // the one ADR-0028 step 5 measured the bus to sit far below (1.066 against 2).
    let k = example("cu_q");
    let n = 1u32 << 12;
    let mut table = String::new();
    for c in 0..6u32 {
        for t in 0..6u32 {
            if c == t {
                continue;
            }
            let sec = k.split_sector(&extents(n, c, t)).unwrap().expect("a split kernel");
            assert_eq!(sec.pairs, n / 4, "the walk is quads");
            assert_eq!(sec.payload, 64.0, "P1: the payload does not move");
            let co = sec.coalescence();
            if c.min(t) >= 3 {
                assert_eq!(co, 1.0, "c = {c}, t = {t}");
            } else {
                assert!(co < 1.0, "c = {c}, t = {t}: {co}");
            }
            table += &format!("({c},{t}) {co:.3}  ");
        }
    }
    // The low corner, by hand: t = 0 fixes bit 0, so a warp of 32 quads spans 64 f32 -> 0.5; a
    // pair at bits 1 and 2 leaves runs of two in every eight -> 0.25.
    assert_eq!(k.split_sector(&extents(n, 4, 0)).unwrap().unwrap().coalescence(), 0.5);
    assert_eq!(k.split_sector(&extents(n, 2, 1)).unwrap().unwrap().coalescence(), 0.25);
    println!("cu_q isolated coalescence at n = 2^12: {table}");
}
