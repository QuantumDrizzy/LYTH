//! ADR-0021 step 3: a coarsened tile, emitted and checked against the host.
//!
//! One emitter, not two. `coarsen 1, 1` — the absent declaration — is this same code with both
//! factors set to one, so every test written before ADR-0021 keeps covering that path. The
//! first thing asserted here is therefore that the uncoarsened matmul is unchanged: a
//! generalisation that quietly broke the case it generalised would be a poor trade.
//!
//! Tests needing a device report that they skipped rather than passing quietly.

use std::path::PathBuf;
use std::process::Command;

fn repo() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn run(example: &str, extents: &[(&str, u32)], extra: &[&str]) -> (String, bool) {
    let repo = repo();
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_lyth"));
    cmd.args([
        "run",
        repo.join("examples").join(example).to_str().unwrap(),
        "--machine",
        repo.join("fixtures/machine/sm_120.json").to_str().unwrap(),
    ]);
    for (name, v) in extents {
        cmd.args(["--set", &format!("{name}={v}")]);
    }
    cmd.args(extra);
    let out = cmd.output().expect("the compiler should run");
    (
        format!(
            "{}{}",
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        ),
        out.status.success(),
    )
}

fn has_device(text: &str) -> bool {
    !text.contains("no CUDA device") && !text.contains("CUDA driver")
}

/// The shapes that divide nothing. 64 divides none of 97, 131 or 67, so the edge tile of `m`,
/// of `n` and of `k` are all partial at once — and with `coarsen 2, 2` each thread owns four
/// outputs, of which some are inside the matrix and some are not.
const RAGGED: &[(u32, u32, u32)] = &[
    (97, 131, 67),
    (65, 33, 129),
    (1, 1, 1),
    (7, 5, 3),
    (64, 64, 64),
    (256, 256, 31),
    (31, 31, 256),
];

#[test]
fn the_uncoarsened_matmul_is_unchanged() {
    // The generalisation is only worth having if it left this alone. Same shapes, same claim.
    for (m, n, k) in RAGGED {
        let (text, ok) = run("matmul.lyth", &[("m", *m), ("n", *n), ("k", *k)], &[]);
        if !has_device(&text) {
            eprintln!("skipped: no CUDA device");
            return;
        }
        assert!(ok, "m={m} n={n} k={k}:\n{text}");
        assert!(text.contains("BIT-EXACT"), "m={m} n={n} k={k}:\n{text}");
    }
}

#[test]
fn a_coarsened_matmul_is_bit_exact_where_nothing_divides() {
    for (m, n, k) in RAGGED {
        let (text, ok) = run("matmul-coarse.lyth", &[("m", *m), ("n", *n), ("k", *k)], &[]);
        if !has_device(&text) {
            eprintln!("skipped: no CUDA device");
            return;
        }
        assert!(ok, "m={m} n={n} k={k}:\n{text}");
        assert!(text.contains("BIT-EXACT"), "m={m} n={n} k={k}:\n{text}");
    }
}

#[test]
fn the_block_is_the_tile_divided_by_the_coarsening() {
    // 64 x 64 is 4096 elements at 2 x 2 each: 1024 threads, exactly the cap a tile of 64 could
    // not meet before. Asserted on what `lyth run` prints, because that is what it launches.
    let (text, ok) = run("matmul-coarse.lyth", &[("m", 97), ("n", 131), ("k", 67)], &[]);
    if !has_device(&text) {
        eprintln!("skipped: no CUDA device");
        return;
    }
    assert!(ok, "{text}");
    assert!(text.contains("blocks of 1024"), "{text}");
    // And the tile is what the traffic was derived from, not the block.
    assert!(text.contains("16.0000 flop/byte asymptotic"), "{text}");
    assert!(text.contains("0.125 * k + 4"), "{text}");
    assert!(text.contains("33280 B per block"), "{text}");
}

#[test]
fn the_coarsened_kernel_really_uses_its_shared_memory() {
    // ADR-0017's bypass control. A kernel that is correct without the shared memory it asked
    // for never staged, and its derived `k/T` reuse would be fiction -- which for a coarsened
    // tile is the entire claim, since the reuse is what the larger tile buys.
    let (text, ok) = run(
        "matmul-coarse.lyth",
        &[("m", 97), ("n", 131), ("k", 67)],
        &["--shared-bytes", "0"],
    );
    if !has_device(&text) {
        eprintln!("skipped: no CUDA device");
        return;
    }
    assert!(!ok, "{text}");
    assert!(
        text.contains("ILLEGAL_ADDRESS") || text.contains("DIFFER") || text.contains("FAILED"),
        "it failed, but not because the tile was gone:\n{text}"
    );
}

#[test]
fn the_emitted_ptx_holds_one_accumulator_per_output() {
    // Four outputs per thread means four accumulators alive across the `p` loop, four global
    // stores, and four operands feeding four multiply-adds where the uncoarsened body has two
    // feeding one. That ratio is where the intensity comes from, so it is asserted rather than
    // taken on trust from the cost model -- which does not see the coarsening at all.
    let repo = repo();
    let out = tempfile::Builder::new().suffix(".ptx").tempfile().unwrap();
    let status = Command::new(env!("CARGO_BIN_EXE_lyth"))
        .args([
            "build",
            repo.join("examples/matmul-coarse.lyth").to_str().unwrap(),
            "--machine",
            repo.join("fixtures/machine/sm_120.json").to_str().unwrap(),
            "-o",
            out.path().to_str().unwrap(),
        ])
        .output()
        .expect("the compiler should run");
    assert!(
        status.status.success(),
        "{}",
        String::from_utf8_lossy(&status.stderr)
    );
    let ptx = std::fs::read_to_string(out.path()).unwrap();

    // Two shared loads per operand, not one: `ci` values of A and `cj` of B per term.
    assert_eq!(ptx.matches("ld.shared.f32").count(), 4, "{ptx}");
    // Four multiplies and four adds in the term loop, from one pair of operand loads each way.
    assert_eq!(ptx.matches("mul.rn.f32").count(), 4, "{ptx}");
    assert_eq!(ptx.matches("add.rn.f32").count(), 4, "{ptx}");
    // Four outputs, stored once each after the loop.
    assert_eq!(ptx.matches("st.global.f32").count(), 4, "{ptx}");
    // Each thread stages four elements of each operand tile: 2 x 2 x 2.
    assert_eq!(ptx.matches("st.shared.f32").count(), 8, "{ptx}");
    // Still two barriers, and still outside every branch.
    assert_eq!(ptx.matches("bar.sync 0;").count(), 2, "{ptx}");
    // And still no fusion: one rounding against two is a different answer (ADR-0010).
    assert!(!ptx.contains("fma.rn.f32"), "{ptx}");
}

#[test]
fn coarsening_a_tile_that_does_not_contract_is_refused() {
    // A transpose's tiled body moves one element per thread and holds nothing across a loop,
    // so there is nothing for a thread to own several of. The cost model would derive the same
    // traffic either way, which is exactly why the emitter has to say so.
    let src = std::fs::read_to_string(repo().join("examples/transpose-tiled.lyth"))
        .unwrap()
        .replace("    tile 32, 32", "    tile 32, 32\n    coarsen 2, 2");
    let dir = tempfile::tempdir().unwrap();
    let f = dir.path().join("t.lyth");
    std::fs::write(&f, src).unwrap();
    let out = Command::new(env!("CARGO_BIN_EXE_lyth"))
        .args([
            "build",
            f.to_str().unwrap(),
            "--machine",
            repo().join("fixtures/machine/sm_120.json").to_str().unwrap(),
            "-o",
            if cfg!(windows) { "nul" } else { "/dev/null" },
        ])
        .output()
        .expect("the compiler should run");
    assert!(!out.status.success());
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(err.contains("coarsens a tile that does not contract"), "{err}");
    assert!(err.contains("nothing for a thread to own several of"), "{err}");
}

#[test]
fn an_asymmetric_coarsening_is_bit_exact_too() {
    // 2 x 2 is the case where `ci` and `cj` are interchangeable, so it cannot catch an emitter
    // that confused the two -- the thread's row stride with its column stride, the count of
    // staged A elements with the count of staged B. 2 x 4 can: the block is 32 x 16 = 512
    // threads, each holding eight accumulators, and every one of those quantities is different
    // from its transpose.
    let src = std::fs::read_to_string(repo().join("examples/matmul-coarse.lyth"))
        .unwrap()
        .replace("coarsen 2, 2", "coarsen 2, 4");
    let dir = tempfile::tempdir().unwrap();
    let f = dir.path().join("c24.lyth");
    std::fs::write(&f, src).unwrap();

    for (m, n, k) in RAGGED {
        let out = Command::new(env!("CARGO_BIN_EXE_lyth"))
            .args([
                "run",
                f.to_str().unwrap(),
                "--machine",
                repo().join("fixtures/machine/sm_120.json").to_str().unwrap(),
                "--set",
                &format!("m={m}"),
                "--set",
                &format!("n={n}"),
                "--set",
                &format!("k={k}"),
            ])
            .output()
            .expect("the compiler should run");
        let text = format!(
            "{}{}",
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        );
        if !has_device(&text) {
            eprintln!("skipped: no CUDA device");
            return;
        }
        assert!(out.status.success(), "m={m} n={n} k={k}:\n{text}");
        assert!(text.contains("BIT-EXACT"), "m={m} n={n} k={k}:\n{text}");
        assert!(text.contains("blocks of 512"), "m={m} n={n} k={k}:\n{text}");
    }
}
