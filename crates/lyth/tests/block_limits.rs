//! What one block may ask for, refused at compile time instead of by the driver.
//!
//! A tile puts one thread on each of its elements, so the block *is* the tile's area. Nothing
//! in the compiler knew the device's cap until now: `tile 64, 64` asked for 4096 threads
//! against 1024, compiled, wrote 3620 bytes of PTX, and failed at launch with
//! `CUDA_ERROR_INVALID_VALUE` — a driver error that names no argument and no reason.
//!
//! ADR-0018 had already argued that this cap is what stops the tile growing, and therefore what
//! caps a contraction's intensity at `T/4 = 8`. It argued it in prose, against a compiler that
//! did not know the number.
//!
//! None of these tests needs a GPU: the limits are data in the machine file, which is also how
//! the shared-memory refusal is reached. On *this* device the thread cap always bites first,
//! because the block is the tile's area — so that branch is exercised against a machine file
//! describing a smaller one, which is what machine files are for.

use std::process::Command;

/// Compile `src` against `machine`, returning (stderr+stdout, success).
fn build(src: &str, machine: &str) -> (String, bool) {
    let dir = tempfile::tempdir().expect("a temporary directory");
    let f = dir.path().join("k.lyth");
    std::fs::write(&f, src).unwrap();
    let m = dir.path().join("m.json");
    std::fs::write(&m, machine).unwrap();
    let out = Command::new(env!("CARGO_BIN_EXE_lyth"))
        .args([
            "build",
            f.to_str().unwrap(),
            "--machine",
            m.to_str().unwrap(),
            "-o",
            if cfg!(windows) { "nul" } else { "/dev/null" },
        ])
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

fn machine(threads: &str, shared: &str, optin: &str) -> String {
    format!(
        r#"{{"schema":"lyth-machine/0.1","id":"sm_120","peak_tflops":15.30,
            "max_threads_per_block":{threads},
            "max_shared_bytes_per_block":{shared},
            "max_shared_bytes_per_block_optin":{optin},
            "levels":[{{"name":"dram","bandwidth_gbs":414.51}}]}}"#
    )
}

fn matmul(tile: u32, asymptote: f64) -> String {
    format!(
        "machine sm_120\n\n\
         kernel mm(m: u32, n: u32, k: u32,\n              \
         a: [f32; m, k], b: [f32; k, n], c: [f32; m, n])\n    \
         space i, j : m, n\n    \
         contract sum p : k\n    \
         tile {tile}, {tile}\n    \
         intensity asymptotic {asymptote}\n    \
         stream a : dram -> smem -> reg\n    \
         stream b : dram -> smem -> reg\n    \
         stream c : dram -> reg, drain\n    \
         at reg:\n        \
         c[i, j] = a[i, p] * b[p, j]\n"
    )
}

#[test]
fn a_tile_wider_than_the_block_cap_is_refused_before_anything_is_emitted() {
    // 64 x 64 is 4096 threads. The traffic it would buy is real and the refusal says so --
    // the point is not that the tile is a bad idea, it is that no block can be that wide.
    let (text, ok) = build(&matmul(64, 16.0), &machine("1024", "49152", "101376"));
    assert!(!ok, "a 4096-thread block must not compile:\n{text}");
    assert!(text.contains("4096 threads per block"), "{text}");
    assert!(text.contains("allows 1024"), "{text}");
    assert!(text.contains("0.125 * k + 4"), "the traffic it would buy:\n{text}");
    assert!(text.contains("Thread coarsening"), "name the way out:\n{text}");
    // And nothing was written: the refusal is before emission, not after.
    assert!(!text.contains("wrote"), "{text}");
}

#[test]
fn the_tile_that_fits_still_compiles() {
    // 32 x 32 is exactly the cap. An off-by-one here would refuse every tiled kernel in the
    // repository, which is the failure mode a limit check has.
    let (text, ok) = build(&matmul(32, 8.0), &machine("1024", "49152", "101376"));
    assert!(ok, "32 x 32 is 1024 threads, exactly the cap:\n{text}");
    assert!(text.contains("8.0000 flop/byte asymptotic"), "{text}");
}

#[test]
fn a_machine_file_without_the_limits_checks_nothing() {
    // `None` must not become zero. Older machine files predate these fields and have to keep
    // working -- badly, but predictably, and the same way they did before.
    let old = r#"{"schema":"lyth-machine/0.1","id":"sm_120","peak_tflops":15.30,
                  "levels":[{"name":"dram","bandwidth_gbs":414.51}]}"#;
    let (text, ok) = build(&matmul(64, 16.0), old);
    assert!(ok, "an unknown limit is not a limit of zero:\n{text}");
}

#[test]
fn shared_memory_refuses_when_it_is_the_thing_in_the_way() {
    // On this device the thread cap always bites first, because the block is the tile's area:
    // any tile big enough to exhaust 101 KB of shared memory needs far more than 1024 threads.
    // So the branch is reached the way the compiler is meant to learn about hardware -- from a
    // machine file, here one describing a device with a smaller shared memory.
    //
    // `tile 32, 32` stages two 32 x 33 tiles: 8448 bytes.
    let small = machine("1024", "4096", "8192");
    let (text, ok) = build(&matmul(32, 8.0), &small);
    assert!(!ok, "8448 bytes against an 8192 maximum:\n{text}");
    assert!(text.contains("8448 bytes into shared memory"), "{text}");
    assert!(text.contains("8192"), "{text}");
}

#[test]
fn staging_above_the_default_but_within_the_opt_in_is_a_note_and_not_a_refusal() {
    // Between the default allowance and the opt-in maximum a kernel is legal and has to ask.
    // Saying so is the difference between a limit and a surprise.
    let mid = machine("1024", "4096", "16384");
    let (text, ok) = build(&matmul(32, 8.0), &mid);
    assert!(ok, "8448 is under the 16384 opt-in maximum:\n{text}");
    assert!(text.contains("above the 4096"), "{text}");
    assert!(text.contains("opts in"), "{text}");
}
