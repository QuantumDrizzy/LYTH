//! The generated binding, compiled and run (ADR-0016).
//!
//! `tests/generated/saxpy.rs` is checked in, produced by
//! `lyth build examples/saxpy.lyth --bind-rust`. Including it here means the test suite
//! **compiles** it, so a generator that emits code which no longer builds fails the build
//! rather than being discovered by whoever tried to use it. `the_checked_in_binding_matches_the_generator`
//! then re-runs the generator and refuses any drift between the two.
//!
//! The launch test needs a device. On a machine without one it reports that it skipped rather
//! than passing quietly, because a test that silently does nothing is worse than no test.

use std::path::PathBuf;
use std::process::Command;

mod saxpy {
    include!("generated/saxpy.rs");
}

fn repo() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

#[test]
fn the_contract_travels_in_the_generated_code() {
    // The point of ADR-0016: a caller can assert on the cost model without a profiler, and
    // without trusting a comment. These are the numbers `lyth run` prints for saxpy.
    assert_eq!(saxpy::MACHINE, "sm_120");
    assert_eq!(saxpy::FLOPS_PER_ELEMENT, 2.0);
    assert_eq!(saxpy::BYTES_PER_ELEMENT, 12.0);
    assert!((saxpy::DERIVED_INTENSITY - 2.0 / 12.0).abs() < 1e-12);
    assert_eq!(saxpy::DECLARED_INTENSITY, Some(0.1667));
    assert!(saxpy::PTX.contains(".visible .entry saxpy"));
    assert_eq!(saxpy::SHARED_BYTES, 0, "saxpy has no reduction");
}

#[test]
fn the_default_grid_is_one_element_per_thread() {
    assert_eq!(saxpy::grid(1), Some(1));
    assert_eq!(saxpy::grid(256), Some(1));
    assert_eq!(saxpy::grid(257), Some(2));
    assert_eq!(saxpy::grid(1 << 20), Some(4096));
}

#[test]
fn the_checked_in_binding_matches_the_generator() {
    // A fixture that drifts from its generator is a fixture that documents the past.
    let repo = repo();
    let out = tempfile::Builder::new()
        .suffix(".rs")
        .tempfile()
        .expect("a temporary file");
    let status = Command::new(env!("CARGO_BIN_EXE_lyth"))
        .args([
            "build",
            repo.join("examples/saxpy.lyth").to_str().unwrap(),
            "--machine",
            repo.join("fixtures/machine/sm_120.json").to_str().unwrap(),
            "-o",
            if cfg!(windows) { "nul" } else { "/dev/null" },
            "--bind-rust",
            out.path().to_str().unwrap(),
        ])
        .output()
        .expect("the compiler should run");
    assert!(status.status.success(), "{}", String::from_utf8_lossy(&status.stderr));

    let fresh = std::fs::read_to_string(out.path()).unwrap();
    let checked_in =
        std::fs::read_to_string(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/generated/saxpy.rs"))
            .unwrap();
    assert_eq!(
        fresh.replace("\r\n", "\n"),
        checked_in.replace("\r\n", "\n"),
        "tests/generated/saxpy.rs is stale; regenerate it with --bind-rust"
    );
}

#[test]
fn the_generated_binding_launches_and_computes_saxpy() {
    use lyth_cuda::Context;

    let ctx = match Context::new(0) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("skipped: no CUDA device ({e})");
            return;
        }
    };
    let module = saxpy::module(&ctx).expect("the embedded PTX should load");
    let k = saxpy::Saxpy::new(&module).expect("the entry point should resolve");

    let n = 4096usize;
    let a = 3.0f32;
    let host_x: Vec<f32> = (0..n).map(|i| i as f32 * 0.25 - 100.0).collect();
    let host_y: Vec<f32> = (0..n).map(|i| 7.0 - i as f32 * 0.5).collect();

    let x = ctx.upload(&host_x).expect("upload x");
    let mut y = ctx.upload(&host_y).expect("upload y");

    k.launch(n as u32, a, &x, &mut y).expect("launch");
    ctx.synchronize().expect("synchronize");

    let got = y.download().expect("download y");
    for i in 0..n {
        // The same `mul_add` the kernel's `fma.rn.f32` performs, so this is bit-exact and not
        // a tolerance: ADR-0010's claim, now made from generated code.
        let want = a.mul_add(host_x[i], host_y[i]);
        assert_eq!(
            got[i].to_bits(),
            want.to_bits(),
            "element {i}: got {}, want {want}",
            got[i]
        );
    }
}

/// The aliasing claim in `bind_rust.rs`, checked rather than asserted.
///
/// A written buffer takes `&mut`, so passing one buffer as both the input and the output of a
/// launch cannot borrow twice. `trybuild` would be the tidy way to test this; a throwaway crate
/// is the way that adds no dependency. It asserts on the error code, not merely on failure: a
/// build that breaks for an unrelated reason would otherwise look like a pass.
#[test]
fn aliasing_an_input_with_an_output_does_not_compile() {
    let dir = match tempfile::tempdir() {
        Ok(d) => d,
        Err(e) => {
            eprintln!("skipped: no temporary directory ({e})");
            return;
        }
    };
    let root = dir.path();
    std::fs::create_dir_all(root.join("src")).unwrap();

    let manifest = format!(
        concat!(
            "[package]\n",
            "name = \"alias_probe\"\n",
            "version = \"0.0.0\"\n",
            "edition = \"2021\"\n\n",
            "[dependencies]\n",
            "lyth-cuda = {{ path = {:?} }}\n\n",
            // Its own workspace, or cargo adopts the one above it and inherits its lockfile.
            "[workspace]\n",
        ),
        repo().join("crates/lyth-cuda")
    );
    std::fs::write(root.join("Cargo.toml"), manifest).unwrap();

    let binding = std::fs::read_to_string(
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/generated/saxpy.rs"),
    )
    .unwrap();
    let lib = format!(
        concat!(
            "#![allow(dead_code)]\n",
            "mod saxpy {{\n{}\n}}\n\n",
            "pub fn probe(k: &saxpy::Saxpy, b: &mut lyth_cuda::Buffer) {{\n",
            "    let _ = k.launch(4, 1.0, b, b);\n",
            "}}\n",
        ),
        binding
    );
    std::fs::write(root.join("src/lib.rs"), lib).unwrap();

    let out = match Command::new("cargo")
        .args(["build", "--quiet"])
        .current_dir(root)
        .output()
    {
        Ok(o) => o,
        Err(e) => {
            eprintln!("skipped: cargo not runnable ({e})");
            return;
        }
    };
    assert!(
        !out.status.success(),
        "aliasing one buffer as input and output must not compile"
    );
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(
        err.contains("E0502") || err.contains("cannot borrow"),
        "it failed, but not for the aliasing reason:\n{err}"
    );
}

/// The generated C and Python, pinned the same way the Rust binding is.
///
/// Neither can be compiled by `cargo test` without dragging in a C toolchain and cuda.h, so
/// what is held here is that the generator has not changed its output. Both were compiled and
/// run against the device by hand when they were written; ADR-0016 records how.
#[test]
fn the_checked_in_c_and_python_match_the_generator() {
    for (flag, name) in [("--bind-c", "saxpy.h"), ("--bind-py", "saxpy.py")] {
        let repo = repo();
        let out = tempfile::NamedTempFile::new().expect("a temporary file");
        let status = Command::new(env!("CARGO_BIN_EXE_lyth"))
            .args([
                "build",
                repo.join("examples/saxpy.lyth").to_str().unwrap(),
                "--machine",
                repo.join("fixtures/machine/sm_120.json").to_str().unwrap(),
                "-o",
                if cfg!(windows) { "nul" } else { "/dev/null" },
                flag,
                out.path().to_str().unwrap(),
            ])
            .output()
            .expect("the compiler should run");
        assert!(status.status.success(), "{}", String::from_utf8_lossy(&status.stderr));

        let fresh = std::fs::read_to_string(out.path()).unwrap();
        let checked_in = std::fs::read_to_string(
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("tests/generated")
                .join(name),
        )
        .unwrap();
        assert_eq!(
            fresh.replace("\r\n", "\n"),
            checked_in.replace("\r\n", "\n"),
            "tests/generated/{name} is stale; regenerate it"
        );
    }
}

#[test]
fn the_generated_python_is_valid_python() {
    // Cheap and toolchain-free: the interpreter parses it without importing it, so no driver
    // is needed and a generator that emits a syntax error fails the suite rather than the user.
    let file = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/generated/saxpy.py");
    let out = match Command::new("python")
        .args([
            "-c",
            "import sys; compile(open(sys.argv[1], encoding='utf-8').read(), sys.argv[1], 'exec')",
            file.to_str().unwrap(),
        ])
        .output()
    {
        Ok(o) => o,
        Err(e) => {
            eprintln!("skipped: no python on PATH ({e})");
            return;
        }
    };
    assert!(
        out.status.success(),
        "the generated Python does not parse:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
}
