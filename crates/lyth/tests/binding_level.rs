//! ADR-0022 step 3: the ceiling belongs to the level that binds, and the others must not move.
//!
//! The regression this file exists for was named in the ADR before the code was written:
//!
//! > Every kernel in this repository has its regime printed by this code path, and nine of
//! > them are correctly DRAM-bound today. A change that makes the matmul right and a saxpy
//! > wrong has traded one error for another.
//!
//! So the first test is not about contractions at all. It is the list of kernels whose answer
//! must be exactly what it was.

use std::path::PathBuf;
use std::process::Command;

fn repo() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn check(example: &str, machine: &str) -> String {
    let repo = repo();
    let out = Command::new(env!("CARGO_BIN_EXE_lyth"))
        .args([
            "check",
            repo.join(example).to_str().unwrap(),
            "--machine",
            machine,
        ])
        .output()
        .expect("the compiler should run");
    assert!(
        out.status.success(),
        "{example}:\n{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8_lossy(&out.stdout).to_string()
}

fn sm_120() -> String {
    repo()
        .join("fixtures/machine/sm_120.json")
        .to_str()
        .unwrap()
        .to_string()
}

/// Every kernel that stages nothing, plus the two that stage without contracting.
///
/// `transpose-tiled` and `dot` are the interesting members: they **do** have shared traffic, so
/// they are the cases where a careless implementation would flip. A transpose moves 8 bytes per
/// element at DRAM and 2 shared accesses, and at this machine's rates that is 19,300 ns against
/// 1,307 per million -- DRAM by a factor of fifteen. The ceiling being per level does not mean
/// the shared level ever wins; it means it is asked.
const DRAM_BOUND: &[&str] = &[
    "examples/saxpy.lyth",
    "examples/axpby.lyth",
    "examples/lerp.lyth",
    "examples/horner.lyth",
    "examples/copy2d.lyth",
    "examples/transpose.lyth",
    "examples/transpose-tiled.lyth",
    "examples/dot.lyth",
    "examples/sum.lyth",
    "examples/max.lyth",
    "examples/min.lyth",
    "examples/split.lyth",
];

#[test]
fn every_kernel_that_was_dram_bound_still_is() {
    let m = sm_120();
    for ex in DRAM_BOUND {
        let text = check(ex, &m);
        assert!(
            text.contains("ceiling  dram binds"),
            "{ex} changed level:\n{text}"
        );
    }
}

#[test]
fn a_tiled_contraction_binds_at_shared_memory() {
    // The measurement this ADR exists for. ADR-0021 step 5: `tile 16` moves twice the derived
    // global traffic of `tile 32` and takes the same time, and DRAM never rose above 1.7% of
    // this device's bandwidth while the compiler printed "with bandwidth saturated" over it.
    let m = sm_120();
    for ex in ["examples/matmul.lyth", "examples/matmul-coarse.lyth"] {
        let text = check(ex, &m);
        assert!(
            text.contains("ceiling  smem binds"),
            "{ex} should bind at shared memory:\n{text}"
        );
        // Both candidates are printed, always. A ceiling that reported only the winner gives a
        // reader no way to tell whether it was close -- and here it is not: 2.24x.
        assert!(text.contains("dram "), "{ex}: the loser is not shown:\n{text}");
        assert!(text.contains("the two disagree by 2.24x"), "{ex}:\n{text}");
    }
}

#[test]
fn coarsening_doubles_the_ceiling_because_it_halves_the_shared_accesses() {
    // `ci + cj` loads feed `ci * cj` bodies: 2 for 1 uncoarsened, 4 for 4 at `coarsen 2, 2`.
    // The tile, and therefore the global traffic, is what changes at the *other* level -- and
    // it changes by the same factor of two, which is exactly why ADR-0021 could not tell the
    // two apart and needed `tile 16` to do it.
    let m = sm_120();
    let plain = check("examples/matmul.lyth", &m);
    let coarse = check("examples/matmul-coarse.lyth", &m);
    assert!(plain.contains("1.48 TFLOP/s"), "{plain}");
    assert!(coarse.contains("2.97 TFLOP/s"), "{coarse}");
    // 2.0625 against 1.0312 accesses per element per unit of k.
    assert!(plain.contains("2.0625 access/element/k"), "{plain}");
    assert!(coarse.contains("1.0312 access/element/k"), "{coarse}");
}

#[test]
fn a_machine_file_without_the_probe_offers_no_shared_ceiling() {
    // A missing rate must not become a rate of zero, which would divide by nothing and make
    // every staged kernel infinitely slow -- the same failure mode the block limits have, and
    // the reason those are `Option` too (ADR-0021 step 1).
    //
    // The old file is the honest test for it: every machine file written before ADR-0022's
    // probe looks exactly like this, and on one of those the matmul must fall back to the
    // answer the compiler gave for its whole life before today.
    let src = std::fs::read_to_string(repo().join("fixtures/machine/sm_120.json")).unwrap();
    let mut m: serde_json::Value = serde_json::from_str(&src).unwrap();
    for lv in m["levels"].as_array_mut().unwrap() {
        lv.as_object_mut().unwrap().remove("accesses_gps");
    }
    let dir = tempfile::tempdir().unwrap();
    let f = dir.path().join("old.json");
    std::fs::write(&f, serde_json::to_string_pretty(&m).unwrap()).unwrap();

    let text = check("examples/matmul-coarse.lyth", f.to_str().unwrap());
    assert!(text.contains("ceiling  dram binds"), "{text}");
    assert!(
        !text.contains("smem "),
        "a level with no measured rate must not be offered as a candidate:\n{text}"
    );
}
