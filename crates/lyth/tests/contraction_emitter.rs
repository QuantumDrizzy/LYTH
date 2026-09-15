//! ADR-0018 step 3: the tiled contraction, emitted.
//!
//! Every test that needs a device reports that it skipped rather than passing quietly — a test
//! that silently does nothing is worse than no test.
//!
//! The structural assertions are here rather than in `lyth-ptx` because what they are about is
//! the whole pipeline: a source file, its derived cost, and the PTX that has to earn it.

use std::path::PathBuf;
use std::process::Command;

fn repo() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

/// `lyth run` on the matmul example at the given extents. Returns (stdout+stderr, success).
fn run(extents: &[(&str, u32)], extra: &[&str]) -> (String, bool) {
    let repo = repo();
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_lyth"));
    cmd.args([
        "run",
        repo.join("examples/matmul.lyth").to_str().unwrap(),
        "--machine",
        repo.join("fixtures/machine/sm_120.json").to_str().unwrap(),
    ]);
    for (name, v) in extents {
        cmd.args(["--set", &format!("{name}={v}")]);
    }
    cmd.args(extra);
    let out = cmd.output().expect("the compiler should run");
    let text = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    let ok = out.status.success();
    (text, ok)
}

fn has_device(text: &str) -> bool {
    !text.contains("no CUDA device") && !text.contains("CUDA driver")
}

#[test]
fn a_matmul_is_bit_exact_at_extents_that_divide_nothing() {
    // The shapes are the test. 32 divides none of 97, 131 or 67, so every tile at the edge of
    // `m`, of `n` and along `k` is partial -- three ragged axes at once, which is the case a
    // guard that is wrong in the generous direction passes on every multiple of 32.
    let cases: &[(u32, u32, u32)] = &[
        (97, 131, 67),
        (7, 5, 3),
        (1, 1, 1),
        (33, 1, 97),
        (1, 33, 1),
        (31, 31, 256),
        (256, 256, 31),
        (32, 32, 32),
        (128, 96, 160),
    ];
    let mut ran = 0;
    for (m, n, k) in cases {
        let (text, ok) = run(&[("m", *m), ("n", *n), ("k", *k)], &[]);
        if !has_device(&text) {
            eprintln!("skipped: no CUDA device");
            return;
        }
        assert!(ok, "m={m} n={n} k={k} failed:\n{text}");
        assert!(
            text.contains("BIT-EXACT"),
            "m={m} n={n} k={k} is not bit-exact:\n{text}"
        );
        ran += 1;
    }
    assert_eq!(ran, cases.len());
}

#[test]
fn the_kernel_really_uses_shared_memory() {
    // ADR-0017's bypass control, which a contraction needs more than a transpose did: a
    // kernel that is correct without the shared memory it asked for is a kernel that never
    // staged, and its derived `k/T` reuse would be fiction. Launching with 0 bytes must not
    // quietly produce the right answer.
    let (text, ok) = run(&[("m", 97), ("n", 131), ("k", 67)], &["--shared-bytes", "0"]);
    if !has_device(&text) {
        eprintln!("skipped: no CUDA device");
        return;
    }
    assert!(!ok, "a contraction with no shared memory must not succeed:\n{text}");
    assert!(
        text.contains("ILLEGAL_ADDRESS") || text.contains("DIFFER") || text.contains("FAILED"),
        "it failed, but not because the tile was gone:\n{text}"
    );
}

#[test]
fn the_skew_is_not_what_makes_it_correct() {
    // `--no-skew` is the counterfactual, and for a contraction it is a counterfactual for a
    // *different* claim than it was for a transpose. There the skew makes a column read
    // conflict-free; here no access is a column read, so the unpadded variant must compute
    // the same bits. What it does to conflicts is measured in the ADR, not asserted here.
    let (text, ok) = run(&[("m", 97), ("n", 131), ("k", 67)], &["--no-skew"]);
    if !has_device(&text) {
        eprintln!("skipped: no CUDA device");
        return;
    }
    assert!(ok, "the unpadded variant must still be correct:\n{text}");
    assert!(text.contains("BIT-EXACT"), "{text}");
}

#[test]
fn the_emitted_ptx_has_the_structure_the_schedule_needs() {
    // Reading the PTX rather than trusting the generator: two barriers per step, two shared
    // tiles, a loop along the contracted axis, and an accumulator that is not fused.
    let repo = repo();
    let out = tempfile::Builder::new()
        .suffix(".ptx")
        .tempfile()
        .expect("a temporary file");
    let status = Command::new(env!("CARGO_BIN_EXE_lyth"))
        .args([
            "build",
            repo.join("examples/matmul.lyth").to_str().unwrap(),
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

    // Two barriers, and both outside every branch: one after the tiles are staged, one after
    // they are walked. A missing second barrier passes `racecheck` nowhere and is the failure
    // this asserts against.
    assert_eq!(ptx.matches("bar.sync 0;").count(), 2, "{ptx}");

    // Three loops: output tiles, steps along k, terms within a step.
    for label in ["$L_tile_matmul:", "$L_step_matmul:", "$L_term_matmul:"] {
        assert!(ptx.contains(label), "missing {label}:\n{ptx}");
    }

    // Two staged tiles: two global loads into shared, two shared loads back out.
    assert_eq!(ptx.matches("st.shared.f32").count(), 2, "{ptx}");
    assert_eq!(ptx.matches("ld.shared.f32").count(), 2, "{ptx}");

    // The accumulator is a multiply and an add, not an fma. One rounding against two is a
    // different answer, and the host oracle would have to change with it -- ADR-0010's
    // decision to make again rather than a free improvement to take here.
    assert!(!ptx.contains("fma.rn.f32"), "the accumulator must not fuse:\n{ptx}");
    assert!(ptx.contains("mul.rn.f32"), "{ptx}");
    assert!(ptx.contains("add.rn.f32"), "{ptx}");

    // The contract travels with the artifact, as an expression because it is one.
    assert!(
        ptx.contains("8.000000 flop/byte asymptotic = (2 * k) flop / (0.25 * k + 4) byte"),
        "{ptx}"
    );

    // The `k` tail is a loop bound, not a zero fill: `min` of what is left against the tile.
    assert!(ptx.contains("min.u32"), "the k tail must be a bound:\n{ptx}");
}

#[test]
fn the_shapes_the_emitter_cannot_honour_are_refused_with_the_reason() {
    // Each of these has a derived cost -- the model handles them -- and no schedule here. A
    // refusal that says which of the two is missing is the difference between a limit and a
    // bug.
    let cases: &[(&str, &str)] = &[
        // Rectangular tile: the operand tiles are T_i x T_p and T_p x T_j, one element per
        // thread of a T_i * T_j block, which closes only when the three are equal.
        (
            "    tile 16, 64\n    stream a : dram -> smem -> reg\n    stream b : dram -> smem -> reg\n    stream c : dram -> reg, drain\n",
            "closes only when the three are equal",
        ),
        // One operand staged and one not: half the reuse, and the cost model derives it.
        (
            "    tile 32, 32\n    stream a : dram -> smem -> reg\n    stream b : dram -> reg\n    stream c : dram -> reg, drain\n",
            "stages 1 buffers",
        ),
        // No tile: the untiled control, which the cost model derives and this does not emit.
        (
            "    stream a : dram -> reg\n    stream b : dram -> reg\n    stream c : dram -> reg, drain\n",
            "without a tile",
        ),
    ];
    for (decls, want) in cases {
        let src = format!(
            "machine sm_120\n\n\
             kernel mm(m: u32, n: u32, k: u32,\n              \
             a: [f32; m, k], b: [f32; k, n], c: [f32; m, n])\n    \
             space i, j : m, n\n    \
             contract sum p : k\n\
             {decls}    \
             at reg:\n        \
             c[i, j] = a[i, p] * b[p, j]\n"
        );
        let dir = tempfile::tempdir().expect("a temporary directory");
        let f = dir.path().join("mm.lyth");
        std::fs::write(&f, src).unwrap();
        let out = Command::new(env!("CARGO_BIN_EXE_lyth"))
            .args([
                "build",
                f.to_str().unwrap(),
                "--machine",
                repo()
                    .join("fixtures/machine/sm_120.json")
                    .to_str()
                    .unwrap(),
                "-o",
                if cfg!(windows) { "nul" } else { "/dev/null" },
            ])
            .output()
            .expect("the compiler should run");
        assert!(!out.status.success(), "{decls} should be refused");
        let err = String::from_utf8_lossy(&out.stderr);
        assert!(err.contains(want), "expected `{want}` in:\n{err}");
    }
}
