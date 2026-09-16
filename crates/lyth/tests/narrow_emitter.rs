//! ADR-0024 step 2: narrow storage, wide arithmetic, checked against the host.
//!
//! The conversion itself is already tied to the silicon — `tests/half_rounding.rs` checks
//! `lyth_lang::half` against `cvt.rn.f16.f32` over 138 device-produced vectors, and that ran
//! before a single kernel was compiled. **So a bit-exactness failure from here can only be the
//! emitter**, which is where the ambiguity was deliberately spent.
//!
//! Tests needing a device report that they skipped rather than passing quietly.

use std::path::PathBuf;
use std::process::Command;

fn repo() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

/// One example with its buffer types rewritten, written to a temp file.
///
/// `--tol` is wide on purpose: the declared `intensity` is the f32 one and halving the element
/// doubles it, which the compiler correctly refuses. That refusal is the subject of
/// `lyth-lang/tests/element_width.rs`; here the subject is the emitted code, and conflating the
/// two would mean neither is tested.
fn run(example: &str, elem: &str, extra: &[&str]) -> (String, bool) {
    let src = std::fs::read_to_string(repo().join("examples").join(example))
        .unwrap()
        .replace("[f32;", &format!("[{elem};"));
    let dir = tempfile::tempdir().unwrap();
    let f = dir.path().join(example);
    std::fs::write(&f, src).unwrap();

    let mut cmd = Command::new(env!("CARGO_BIN_EXE_lyth"));
    cmd.args([
        "run",
        f.to_str().unwrap(),
        "--machine",
        repo().join("fixtures/machine/sm_120.json").to_str().unwrap(),
        "--tol",
        "1.0",
    ]);
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
    !text.contains("error[cuda]")
}

const WIDTHS: &[&str] = &["f32", "f16", "bf16"];

#[test]
fn elementwise_and_reductions_are_bit_exact_at_every_width() {
    // f32 is in the list because a generalisation that broke the case it generalised would be
    // a poor trade, and because these are the kernels ADR-0001 through ADR-0013 were measured
    // on. `sum` and `max` go through the reduction tree, which has its own store.
    for w in WIDTHS {
        for ex in ["saxpy.lyth", "axpby.lyth", "lerp.lyth", "sum.lyth", "max.lyth"] {
            let (text, ok) = run(ex, w, &["-n", "4096"]);
            if !has_device(&text) {
                eprintln!("skipped: no CUDA device");
                return;
            }
            assert!(ok, "{ex} at {w}:\n{text}");
            assert!(text.contains("BIT-EXACT"), "{ex} at {w}:\n{text}");
        }
    }
}

#[test]
fn a_staged_tile_is_bit_exact_at_every_width() {
    // The tiled path is a separate emitter with its own global load, shared store, shared load
    // and global store -- four interfaces a narrow element crosses, against the flat body's
    // two. It was still writing four bytes after the flat one was fixed, and the harness said
    // so: `FAILED -- 9216 of 12288 elements differ`.
    for w in WIDTHS {
        let (text, ok) = run(
            "transpose-tiled.lyth",
            w,
            &["--set", "rows=128", "--set", "cols=96"],
        );
        if !has_device(&text) {
            eprintln!("skipped: no CUDA device");
            return;
        }
        assert!(ok, "transpose-tiled at {w}:\n{text}");
        assert!(text.contains("BIT-EXACT"), "transpose-tiled at {w}:\n{text}");
    }
}

#[test]
fn a_kernel_may_read_narrow_and_write_wide() {
    // Nothing says the staged buffer and the drained one share a type, and assuming they do
    // would index one with the other's stride -- which reads real memory and returns plausible
    // numbers rather than crashing. The widths are looked up separately, and this is what says
    // so.
    let src = std::fs::read_to_string(repo().join("examples/transpose-tiled.lyth"))
        .unwrap()
        .replace("a: [f32;", "a: [f16;");
    let dir = tempfile::tempdir().unwrap();
    let f = dir.path().join("mixed.lyth");
    std::fs::write(&f, src).unwrap();
    let out = Command::new(env!("CARGO_BIN_EXE_lyth"))
        .args([
            "run",
            f.to_str().unwrap(),
            "--machine",
            repo().join("fixtures/machine/sm_120.json").to_str().unwrap(),
            "--set",
            "rows=128",
            "--set",
            "cols=96",
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
    if !has_device(&text) {
        eprintln!("skipped: no CUDA device");
        return;
    }
    assert!(out.status.success(), "{text}");
    assert!(text.contains("BIT-EXACT"), "f16 in, f32 out:\n{text}");
}

#[test]
fn the_emitted_ptx_loads_into_sixteen_bits_and_converts_from_there() {
    // The silent bug this shape avoids: loading a narrow element into a 32-bit register and
    // converting *that* converts sign- or zero-extended bits, which is a different number
    // arrived at without any error. Asserted on the emitted code because nothing downstream
    // would notice.
    let src = std::fs::read_to_string(repo().join("examples/saxpy.lyth"))
        .unwrap()
        .replace("[f32;", "[f16;");
    let dir = tempfile::tempdir().unwrap();
    let f = dir.path().join("s.lyth");
    std::fs::write(&f, src).unwrap();
    let out = tempfile::Builder::new().suffix(".ptx").tempfile().unwrap();
    let st = Command::new(env!("CARGO_BIN_EXE_lyth"))
        .args([
            "build",
            f.to_str().unwrap(),
            "--machine",
            repo().join("fixtures/machine/sm_120.json").to_str().unwrap(),
            "-o",
            out.path().to_str().unwrap(),
            "--tol",
            "1.0",
        ])
        .output()
        .expect("the compiler should run");
    assert!(st.status.success(), "{}", String::from_utf8_lossy(&st.stderr));
    let ptx = std::fs::read_to_string(out.path()).unwrap();

    assert!(ptx.contains(".reg .b16"), "no 16-bit register bank:\n{ptx}");
    assert_eq!(ptx.matches("ld.global.b16").count(), 2, "{ptx}");
    assert_eq!(ptx.matches("cvt.f32.f16").count(), 2, "{ptx}");
    assert_eq!(ptx.matches("cvt.rn.f16.f32").count(), 1, "{ptx}");
    assert_eq!(ptx.matches("st.global.b16").count(), 1, "{ptx}");
    assert_eq!(ptx.matches("ld.global.f32").count(), 0, "{ptx}");
    assert_eq!(ptx.matches("st.global.f32").count(), 0, "{ptx}");

    // The stride is the element's, and there is **one** of them: a uniform-width kernel still
    // computes its offset once, so the instruction counts ADR-0014 and ADR-0020 measured do
    // not move. A mixed-width kernel gets two, memoised by width rather than one per buffer.
    assert_eq!(ptx.matches(", 2;").count(), 1, "one 2-byte stride:\n{ptx}");
    assert_eq!(ptx.matches(", 4;").count(), 0, "no 4-byte stride left:\n{ptx}");

    // And the arithmetic never narrows (ADR-0024 decision 1): the values arrive as halves and
    // are multiplied and added in f32.
    //
    // `fma.rn.f32`, singular, and that is correct here. ADR-0010's refusal to fuse belongs to
    // the **contraction** emitter, where one rounding against two changes an accumulator over
    // thousands of terms; a flat body's `a * x + y` fuses, which is why ADR-0020's hand-written
    // comparison writes `fmaf` to match it. This test asserted the unfused pair and was wrong
    // about which emitter it was looking at.
    assert!(ptx.contains("fma.rn.f32"), "{ptx}");
    for narrow in ["add.rn.f16", "mul.rn.f16", "fma.rn.f16", "add.rn.bf16"] {
        assert!(
            !ptx.contains(narrow),
            "the arithmetic must not narrow, found {narrow}:\n{ptx}"
        );
    }
}
