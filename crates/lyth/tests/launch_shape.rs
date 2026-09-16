//! What `lyth run` actually launches.
//!
//! This is the third defect in this area and the first test of it, which is the wrong order.
//!
//! 1. `report_timing` assembled its own launch while `cmd_run` had already verified one, and
//!    reported 3.85 TB/s on a 358 GB/s device.
//! 2. The reduction grid was corrected in `cmd_run` and not in the manifest, so every generated
//!    binding published the shape that had just been measured at half the bandwidth.
//! 3. The rule was then moved into one place — correctly — and the caller fed it a map that
//!    does not contain `-n`. Rank 1 fell back to a product of 1 and **launched one block**:
//!    saxpy reported 9.2 GB/s where it reaches 398.
//!
//! Every one of those was a launch shape nobody asserted on, found by looking at a number that
//! seemed wrong. A shape is cheap to assert and a number is not, so: assert the shape.

use std::process::Command;
use std::path::PathBuf;

fn repo() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn run(example: &str, args: &[&str]) -> (String, bool) {
    let repo = repo();
    let out = Command::new(env!("CARGO_BIN_EXE_lyth"))
        .args([
            "run",
            repo.join("examples").join(example).to_str().unwrap(),
            "--machine",
            repo.join("fixtures/machine/sm_120.json").to_str().unwrap(),
        ])
        .args(args)
        .output()
        .expect("the compiler should run");
    (
        format!(
            "{}{}",
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        ),
        out.status.success(),
    )
}

/// The `grid N blocks of B` line, parsed.
fn shape(text: &str) -> Option<(u32, u32)> {
    let line = text.lines().find(|l| l.trim_start().starts_with("grid "))?;
    let mut words = line.split_whitespace();
    words.next()?; // grid
    let blocks: u32 = words.next()?.parse().ok()?;
    words.next()?; // blocks
    words.next()?; // of
    let block: u32 = words.next()?.parse().ok()?;
    Some((blocks, block))
}

fn skip_without_device(text: &str) -> bool {
    // Any `error[cuda]` before the kernel runs means there is no usable device. A GPU
    // that fell off the bus says `cuInit failed`, which the two-phrase version missed.
    let missing = text.contains("error[cuda]");
    if missing {
        eprintln!("skipped: no CUDA device");
    }
    missing
}

#[test]
fn a_rank_one_kernel_launches_a_grid_over_the_element_count_and_not_over_one() {
    // `-n` is its own flag and never appears in the `--set` map the grid rule is resolved
    // against. Falling back to 1 there is the whole of defect 3 above: the product of the
    // extents came out 1, the grid came out `ceil(1 / 256)`, and 67 million elements were
    // walked by 256 threads.
    let n: u32 = 1 << 20;
    let (text, ok) = run("saxpy.lyth", &["-n", &n.to_string()]);
    if skip_without_device(&text) {
        return;
    }
    assert!(ok, "{text}");
    let (blocks, block) = shape(&text).unwrap_or_else(|| panic!("no grid line in:\n{text}"));
    assert_eq!(block, 256);
    assert_eq!(
        blocks,
        n / block,
        "one element per thread is the elementwise default (ADR-0012):\n{text}"
    );
    // And the consequence, stated so a future regression is caught by meaning rather than by
    // arithmetic: every thread gets one element.
    assert!(text.contains("1 element(s) per thread"), "{text}");
}

#[test]
fn a_reduction_amortises_its_block_tree_at_rank_one_too() {
    // The fix from the dogfood, asserted where it is used rather than only where it is
    // defined. Eight elements per thread, because the block's shared-memory tree runs once per
    // thread and at one element per thread that is once per element -- 213 GB/s against 419.
    let n: u32 = 1 << 20;
    let (text, ok) = run("sum.lyth", &["-n", &n.to_string()]);
    if skip_without_device(&text) {
        return;
    }
    assert!(ok, "{text}");
    let (blocks, block) = shape(&text).unwrap_or_else(|| panic!("no grid line in:\n{text}"));
    assert_eq!(block, 256);
    assert_eq!(blocks, n / (block * 8), "{text}");
    assert!(text.contains("8 element(s) per thread"), "{text}");
}

#[test]
fn a_tiled_kernel_launches_one_block_per_tile() {
    // Not one thread per element: `ceil(rows/32) * ceil(cols/32)`. The extents are ragged on
    // purpose, so a rule that forgot the ceiling would be one tile short on each axis.
    let (text, ok) = run(
        "transpose-tiled.lyth",
        &["--set", "rows=100", "--set", "cols=70"],
    );
    if skip_without_device(&text) {
        return;
    }
    assert!(ok, "{text}");
    let (blocks, block) = shape(&text).unwrap_or_else(|| panic!("no grid line in:\n{text}"));
    assert_eq!(block, 1024, "a tile of 32x32 fixes the block");
    assert_eq!(blocks, 4 * 3, "ceil(100/32) * ceil(70/32):\n{text}");
}

#[test]
fn the_reported_bandwidth_is_within_sight_of_the_machine_file() {
    // Not a performance test -- it asserts nothing about how fast the kernel is. It asserts
    // that the *launch* is not pathological, which is what every defect in this file's
    // docstring produced: 3.85 TB/s on a 358 GB/s device, and 9.2 GB/s on a 414 GB/s one.
    //
    // The window is deliberately wide. A kernel that lands inside it may still be slow; a
    // kernel outside it is not slow, it is launched wrong.
    // 2^24 elements is 201 MB of traffic against a 34 MB L2, so the bytes really do cross the
    // memory controller. Larger would be no more conclusive and the host-side verification is
    // what costs the time.
    let (text, ok) = run("saxpy.lyth", &["-n", "16777216", "--time", "10"]);
    if skip_without_device(&text) {
        return;
    }
    assert!(ok, "{text}");
    let line = text
        .lines()
        .find(|l| l.trim_start().starts_with("achieved "))
        .unwrap_or_else(|| panic!("no achieved line in:\n{text}"));
    let gbs: f64 = line
        .split_whitespace()
        .nth(1)
        .and_then(|w| w.parse().ok())
        .unwrap_or_else(|| panic!("cannot parse:\n{line}"));
    assert!(
        (100.0..1000.0).contains(&gbs),
        "{gbs} GB/s is not a launch shape a streaming kernel has on this device:\n{text}"
    );
}
