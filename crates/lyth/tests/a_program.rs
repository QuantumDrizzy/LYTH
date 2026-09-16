//! `lyth build main.lyth -o main.ubo`, and then the machine runs it. ADR-0025 step 4.
//!
//! This is the test the ADR was written to be able to have. Every other end-to-end check in
//! this project compiles a kernel and hands it to a host in another language to launch; here
//! the output is a file the machine runs on its own, and the whole workflow is
//!
//! ```text
//! lyth build examples/main.lyth --machine fixtures/machine/unibit.json -o main.ubo
//! unibit run main.ubo
//! ```
//!
//! with no host language in either line. What makes it worth anything is the last step: the
//! numbers the program prints are compared **bit for bit** against `lyth_lang::eval`, the same
//! oracle the PTX back end is checked against. A program that runs and prints plausible floats
//! is not evidence of anything.
//!
//! Skips rather than passes when the emulator is not present.

use std::path::{Path, PathBuf};
use std::process::Command;

use lyth_lang::ast::Ty;
use lyth_lang::{eval, ir, parse};

fn repo() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

/// The emulator's own binary, built if it is not there yet.
///
/// Spelled out rather than reached with `Path::parent`, which is lexical and once sent this
/// lookup into `LYTH/crates/Unibit` -- where it found nothing, skipped, and reported seven
/// passes in a hundredth of a second.
fn unibit_binary() -> Option<PathBuf> {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../Unibit");
    if !dir.join("Cargo.toml").exists() {
        eprintln!("skipped: no Unibit at {}", dir.display());
        return None;
    }
    let built = Command::new("cargo")
        .args(["build", "--quiet"])
        .current_dir(&dir)
        .status()
        .ok()?;
    if !built.success() {
        eprintln!("skipped: Unibit does not build");
        return None;
    }
    let exe = dir
        .join("target/debug")
        .join(if cfg!(windows) { "unibit.exe" } else { "unibit" });
    exe.exists().then_some(exe)
}

fn lyth(args: &[&str]) -> (bool, String) {
    let out = Command::new(env!("CARGO_BIN_EXE_lyth"))
        .args(args)
        .output()
        .expect("the compiler should run");
    (
        out.status.success(),
        format!(
            "{}{}",
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        ),
    )
}

fn machine(id: &str) -> String {
    repo()
        .join(format!("fixtures/machine/{id}.json"))
        .to_string_lossy()
        .into_owned()
}

fn example(name: &str) -> String {
    repo().join("examples").join(name).to_string_lossy().into_owned()
}

/// What `examples/main.lyth` should print: `y = a * x + y` at n = 4096, a = 2.0, elements 0..8.
fn oracle(n: u32) -> Vec<f32> {
    let src = std::fs::read_to_string(repo().join("examples/main.lyth")).unwrap();
    let unit = parse(&src).expect("parses");
    let k = ir::lower(&unit, &unit.kernels[0]).expect("lowers");
    let mut inputs = eval::Inputs::default();
    for p in &k.params {
        if p.ty.is_buffer() {
            inputs
                .buffers
                .insert(p.name.clone(), lyth_lang::inputs::buffer(&p.name, p.ty, n));
        } else if p.ty == Ty::F32 {
            // From the source, not from a constant here: the point of `main` is that the file
            // says. A test that supplied its own 2.0 would agree with the program for a reason
            // the program did not give.
            let m = unit.main.as_ref().unwrap();
            let v = m
                .args
                .iter()
                .find(|(name, _, _)| name == &p.name)
                .expect("main gives every scalar")
                .1;
            inputs.scalars.insert(p.name.clone(), v as f32);
        }
    }
    let out = eval::eval(&k, n as usize, &inputs).expect("the oracle evaluates");
    out.buffers[&k.drains[0].0].clone()
}

#[test]
fn a_lyth_file_builds_to_an_object_the_machine_runs_on_its_own() {
    let Some(unibit) = unibit_binary() else {
        return;
    };
    let dir = tempfile::tempdir().unwrap();
    let obj = dir.path().join("main.ubo");

    let (ok, text) = lyth(&[
        "build",
        &example("main.lyth"),
        "--machine",
        &machine("unibit"),
        "-o",
        &obj.to_string_lossy(),
        "--assembler",
        &unibit.to_string_lossy(),
    ]);
    assert!(ok, "{text}");
    assert!(obj.exists(), "no object was written:\n{text}");

    // The contract is checked on the way past, as it is for every other target. A program that
    // builds without its declared intensity being confirmed is a program carrying a claim
    // nobody looked at.
    assert!(text.contains("declared 0.1667 — matches"), "{text}");

    let run = Command::new(&unibit)
        .args(["run", &obj.to_string_lossy()])
        .output()
        .expect("the emulator runs");
    let printed = String::from_utf8_lossy(&run.stdout).to_string();
    let got: Vec<f32> = printed
        .lines()
        .filter_map(|l| l.trim().parse::<f32>().ok())
        .collect();

    // `main.lyth` says `print y[0:8]`, so eight and not 4096. If the emitter ignored the range
    // this would be the wrong length rather than the wrong values.
    assert_eq!(got.len(), 8, "printed {} values:\n{printed}", got.len());
    assert!(printed.contains("y[0:8]"), "the output says what it is:\n{printed}");

    let want = oracle(4096);
    for i in 0..8 {
        assert_eq!(
            got[i].to_bits(),
            want[i].to_bits(),
            "element {i}: program {} , oracle {}\n{printed}",
            got[i],
            want[i]
        );
    }
}

#[test]
fn the_printed_text_is_not_a_rounded_view_of_the_answer() {
    // Why the program prints with `PRINT_F32` and not `PRINT_F64`, which formats to six
    // decimal places. The comparison above is bit-exact, and it can only be bit-exact if the
    // printing round-trips: at six places, `-1.2549801` and `-1.2549802` are the same eight
    // characters and the check would pass on a kernel that was one ulp wrong everywhere.
    let want = oracle(4096);
    for v in want.iter().take(8) {
        let rounded = format!("{v:.6}");
        let back: f32 = rounded.parse().unwrap();
        if back.to_bits() != v.to_bits() {
            return; // at least one value proves the point
        }
    }
    panic!("none of these eight values loses anything at six decimal places, so this test cannot tell the two formats apart");
}

#[test]
fn a_program_for_a_machine_with_no_entry_point_is_still_a_kernel() {
    // `main` is not a general feature of the language and should not become one. `sm_120` has
    // no entry point, so a `main` there would describe a launch that CUDA, not LYTH, performs
    // -- and `lyth build` for that machine still emits PTX, as it always has.
    let (ok, text) = lyth(&["build", &example("saxpy.lyth"), "--machine", &machine("sm_120")]);
    assert!(ok, "{text}");
    assert!(text.contains(".target sm_120"), "{text}");
    assert!(text.contains("visible .entry saxpy"), "{text}");
}

#[test]
fn building_a_program_from_a_file_with_no_main_is_refused() {
    // The refusal that makes `main` mean something. `saxpy.lyth` is a kernel and nothing else,
    // and on a machine that runs programs there is no host to supply the element count or to
    // decide what to print -- so the compiler cannot, either.
    let src = std::fs::read_to_string(repo().join("examples/saxpy.lyth"))
        .unwrap()
        .replace("machine sm_120", "machine unibit");
    let dir = tempfile::tempdir().unwrap();
    let f = dir.path().join("headless.lyth");
    std::fs::write(&f, src).unwrap();

    let (ok, text) = lyth(&[
        "build",
        &f.to_string_lossy(),
        "--machine",
        &machine("unibit"),
    ]);
    assert!(!ok, "a program with no entry point must be refused:\n{text}");
    assert!(text.contains("declares no `main`"), "{text}");
    assert!(text.contains("there is no host"), "{text}");
}

#[test]
fn without_the_assembler_the_assembly_is_still_written() {
    // A missing tool should not lose the compile. The `.uasm` is the compiler's output; the
    // object is the machine's assembler's, and that is the part that can be absent.
    let dir = tempfile::tempdir().unwrap();
    let obj = dir.path().join("main.ubo");
    let (ok, text) = lyth(&[
        "build",
        &example("main.lyth"),
        "--machine",
        &machine("unibit"),
        "-o",
        &obj.to_string_lossy(),
        "--assembler",
        &Path::new("no-such-assembler").to_string_lossy(),
    ]);
    assert!(ok, "{text}");
    assert!(obj.with_extension("uasm").exists(), "{text}");
    assert!(!obj.exists(), "no assembler, no object");
    assert!(text.contains("Finish with:  unibit build"), "{text}");
}
