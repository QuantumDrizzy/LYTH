//! The same kernel against two machines. ADR-0025 step 1.
//!
//! ADR-0025's argument for a second target was that a cost model checked on two ISAs with
//! nothing in common except the derivation is a much stronger claim than one checked on either.
//! This is the first instalment of that, and it paid immediately: **the compute ceiling was
//! missing and only the second machine could show it.**
//!
//! `sm_120`'s ridge is 36.9 flop/byte and the densest kernel this language can write is 16, so
//! every kernel was memory-bound and the memory answer was always the small one. `unibit`'s
//! ridge is **0.4999** — a matmul at 8 flop/byte sits sixteen times past it, and the ceiling
//! came out at **1600% of the machine's peak FLOPS**. A bound above peak is not a loose bound;
//! it is not a bound.

use std::path::PathBuf;
use std::process::Command;

fn repo() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

/// One example retargeted, since the source names its machine and the compiler refuses a file
/// that describes a different one -- which is itself the contract working.
fn check_on(example: &str, machine: &str) -> String {
    let src = std::fs::read_to_string(repo().join("examples").join(example))
        .unwrap()
        .replace("machine sm_120", &format!("machine {machine}"));
    let dir = tempfile::tempdir().unwrap();
    let f = dir.path().join(example);
    std::fs::write(&f, src).unwrap();
    let out = Command::new(env!("CARGO_BIN_EXE_lyth"))
        .args([
            "check",
            f.to_str().unwrap(),
            "--machine",
            repo()
                .join(format!("fixtures/machine/{machine}.json"))
                .to_str()
                .unwrap(),
            "--tol",
            "1.0",
        ])
        .output()
        .expect("the compiler should run");
    let text = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(out.status.success(), "{example} on {machine}:\n{text}");
    text
}

#[test]
fn a_machine_with_no_clock_reports_its_ceiling_in_cycles() {
    // `unibit` charges one cycle per instruction and names no frequency anywhere. Quoting its
    // ceiling in TFLOP/s would require inventing a megahertz, in a file whose first line says
    // EVERY NUMBER HERE IS MEASURED ON THIS MACHINE.
    let text = check_on("saxpy.lyth", "unibit");
    assert!(text.contains("flop/cycle"), "{text}");
    assert!(text.contains("cycles per 1e6"), "{text}");
    assert!(!text.contains("TFLOP/s"), "a machine with no clock has no TFLOP/s:\n{text}");

    // And the clocked machine is untouched.
    let gpu = check_on("saxpy.lyth", "sm_120");
    assert!(gpu.contains("TFLOP/s"), "{gpu}");
    assert!(gpu.contains("ns per 1e6"), "{gpu}");
}

#[test]
fn the_same_matmul_is_memory_bound_on_one_machine_and_compute_bound_on_the_other() {
    // The sentence ADR-0025 was written to be able to say. Same source, same derivation, and
    // the regime flips -- because the ridge is 36.9 on one machine and 0.4999 on the other.
    let gpu = check_on("matmul.lyth", "sm_120");
    assert!(gpu.contains("36.9 flop/byte — memory-bound"), "{gpu}");
    assert!(gpu.contains("ceiling  smem binds"), "{gpu}");

    let uni = check_on("matmul.lyth", "unibit");
    assert!(uni.contains("0.5 flop/byte — compute-bound"), "{uni}");
    assert!(uni.contains("ceiling  reg binds"), "{uni}");
}

#[test]
fn a_compute_bound_kernel_cannot_be_offered_more_than_the_machine_retires() {
    // The bug the second machine exposed. Before compute was a candidate, this printed
    // "227.407 flop/cycle, 1600.34% of peak FLOPS".
    //
    // A roofline has two sides and ADR-0022 had built only one of them, invisibly, because
    // nothing this language could write had ever crossed `sm_120`'s ridge.
    let uni = check_on("matmul.lyth", "unibit");
    assert!(
        uni.contains("100.00% of peak FLOPS"),
        "a compute-bound kernel's ceiling is exactly the machine's flop rate:\n{uni}"
    );
    for over in ["1600", "227.4"] {
        assert!(!uni.contains(over), "still offering more than peak:\n{uni}");
    }
}

#[test]
fn a_declared_time_unit_the_file_does_not_carry_is_refused() {
    // The machine file may say what it is, and then it has to be it -- the same rule the
    // language applies to `intensity`. A declaration nothing checks is a comment.
    let src = std::fs::read_to_string(repo().join("fixtures/machine/unibit.json")).unwrap();
    let mut m: serde_json::Value = serde_json::from_str(&src).unwrap();
    m.as_object_mut().unwrap().remove("flops_per_cycle");
    let dir = tempfile::tempdir().unwrap();
    let f = dir.path().join("lying.json");
    std::fs::write(&f, serde_json::to_string_pretty(&m).unwrap()).unwrap();

    let lyth_src = std::fs::read_to_string(repo().join("examples/saxpy.lyth"))
        .unwrap()
        .replace("machine sm_120", "machine unibit");
    let sf = dir.path().join("s.lyth");
    std::fs::write(&sf, lyth_src).unwrap();

    let out = Command::new(env!("CARGO_BIN_EXE_lyth"))
        .args([
            "check",
            sf.to_str().unwrap(),
            "--machine",
            f.to_str().unwrap(),
            "--tol",
            "1.0",
        ])
        .output()
        .expect("the compiler should run");
    let text = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(
        text.contains("declares time_unit"),
        "a file that says `cycle` and carries no per-cycle rate must be named:\n{text}"
    );
}
