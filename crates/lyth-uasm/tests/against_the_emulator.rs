//! A `.lyth` file becomes a program, and the program is right. ADR-0025 step 2.
//!
//! Every test here assembles the emitted text with the real `unibit` binary, runs it on the
//! real emulator, reads back the lanes it printed, and compares them against `lyth_lang::eval`
//! — the same host oracle the PTX back end is checked against.
//!
//! That is the whole point of the target. On `sm_120` the check is host-versus-device; here it
//! is host-versus-**program**, because there is no host: `unibit run prog.ubo` is the entire
//! invocation and no other language appears in it.
//!
//! Tests needing the emulator report that they skipped rather than passing quietly.

use std::path::{Path, PathBuf};
use std::process::Command;

use lyth_lang::ast::Ty;
use lyth_lang::{eval, ir, parse};

fn unibit() -> PathBuf {
    // Spelled out rather than `repo().parent()`, because `Path::parent` is **lexical**: on
    // `LYTH/crates/lyth-uasm/../..` it strips the last component and hands back
    // `LYTH/crates/lyth-uasm/..`, so the emulator was looked for in `LYTH/crates/Unibit`.
    //
    // The skip below then turned a wrong path into seven passing tests in 0.01 seconds, which
    // is the shape of defect this project hunts hardest: a check whose failure state is
    // unreachable. The skip now prints where it looked.
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../Unibit")
}

fn lower(src: &str) -> ir::KernelIr {
    let unit = parse(src).expect("parses");
    ir::lower(&unit, &unit.kernels[0]).expect("lowers")
}

/// Assemble and run, returning the f32 lanes the program printed in order.
///
/// `None` when the emulator is not present or will not build, which is a skip rather than a
/// failure — the same rule the CUDA tests follow.
fn run_on_emulator(asm: &str, dir: &Path) -> Option<Vec<f32>> {
    let uni = unibit();
    if !uni.join("Cargo.toml").exists() {
        eprintln!("skipped: no Unibit emulator at {}", uni.display());
        return None;
    }
    let src = dir.join("k.uasm");
    std::fs::write(&src, asm).unwrap();

    let out = Command::new("cargo")
        .args(["run", "--quiet", "--", "run", src.to_str().unwrap()])
        .current_dir(&uni)
        .output()
        .ok()?;
    let text = String::from_utf8_lossy(&out.stdout).to_string();
    if !text.contains("x5") {
        eprintln!(
            "the emulator printed no register:\n{text}\n{}",
            String::from_utf8_lossy(&out.stderr)
        );
        return None;
    }

    // `PRINT_REG256` writes `[x5  = [lane3 | lane2 | lane1 | lane0]]`, four 64-bit lanes, most
    // significant first, each holding two f32 with the low half the earlier element.
    //
    // **It emits no newline**, so eight prints arrive concatenated on one line. Scanning
    // `text.lines()` and taking one match each found the first of the eight and reported that
    // the program had computed 8 elements instead of 64 -- which looked like a loop that ran
    // once, and was a parser that stopped once.
    let mut vals = Vec::new();
    for chunk in text.split("x5  = [").skip(1) {
        let Some(inner) = chunk.split(']').next() else {
            continue;
        };
        let lanes: Vec<u64> = inner
            .split('|')
            .filter_map(|p| u64::from_str_radix(p.trim().trim_start_matches("0x"), 16).ok())
            .collect();
        if lanes.len() != 4 {
            continue;
        }
        for l in lanes.iter().rev() {
            vals.push(f32::from_bits(*l as u32));
            vals.push(f32::from_bits((*l >> 32) as u32));
        }
    }
    Some(vals)
}

/// What the host says the same kernel computes, on the same inputs.
fn oracle(k: &ir::KernelIr, n: u32) -> Vec<f32> {
    let mut inputs = eval::Inputs::default();
    for p in &k.params {
        if p.ty.is_buffer() {
            inputs
                .buffers
                .insert(p.name.clone(), lyth_lang::inputs::buffer(&p.name, p.ty, n));
        } else if p.ty == Ty::F32 {
            inputs
                .scalars
                .insert(p.name.clone(), lyth_uasm::DEFAULT_SCALAR);
        }
    }
    let out = eval::eval(k, n as usize, &inputs).expect("the oracle evaluates");
    let drained = &k.drains[0].0;
    out.buffers[drained].clone()
}

const SAXPY: &str = "machine unibit\n\n\
     kernel saxpy(n: u32, a: f32, x: [f32; n], y: [f32; n])\n    \
     intensity 0.1667\n    \
     stream x : dram -> reg\n    \
     stream y : dram -> reg, drain\n    \
     at reg:\n        \
     y = a * x + y\n";

#[test]
fn a_lyth_kernel_becomes_a_program_that_computes_the_right_thing() {
    let n = 64;
    let k = lower(SAXPY);
    let asm = lyth_uasm::emit_program(&k, n).expect("emits");
    let dir = tempfile::tempdir().unwrap();
    let Some(got) = run_on_emulator(&asm, dir.path()) else {
        eprintln!("skipped: no Unibit emulator");
        return;
    };
    let want = oracle(&k, n);

    assert_eq!(got.len(), n as usize, "the program printed {} lanes", got.len());
    for i in 0..n as usize {
        assert_eq!(
            got[i].to_bits(),
            want[i].to_bits(),
            "element {i}: program {} , oracle {}\n{asm}",
            got[i],
            want[i]
        );
    }
}

#[test]
fn a_kernel_with_no_fma_is_right_too() {
    // `y = (x + y) * a` has a multiply and an add that do not fuse, so it exercises `VFMUL`
    // and `VFADD` as separate instructions rather than the `VFMA` path.
    let src = SAXPY.replace("y = a * x + y", "y = (x + y) * a");
    let n = 32;
    let k = lower(&src);
    let asm = lyth_uasm::emit_program(&k, n).expect("emits");
    let dir = tempfile::tempdir().unwrap();
    let Some(got) = run_on_emulator(&asm, dir.path()) else {
        eprintln!("skipped: no Unibit emulator");
        return;
    };
    let want = oracle(&k, n);
    for i in 0..n as usize {
        assert_eq!(got[i].to_bits(), want[i].to_bits(), "element {i}\n{asm}");
    }
}

#[test]
fn the_two_input_buffers_really_differ() {
    // Without this the test above would pass under a kernel that read `y` twice. The generator
    // used to give `x` and `y` the same data in 99.6% of elements, which is exactly the bug
    // that would have hidden here.
    let x = lyth_lang::inputs::buffer("x", Ty::BufF32, 64);
    let y = lyth_lang::inputs::buffer("y", Ty::BufF32, 64);
    let same = x.iter().zip(&y).filter(|(a, b)| a == b).count();
    assert!(same < 2, "x and y agree on {same} of 64 elements");
}

#[test]
fn a_tile_is_refused_because_this_machine_has_nowhere_to_stage() {
    // Not "not yet". `unibit.json` names no `smem` level, so the ceiling ADR-0022 computes has
    // no shared candidate and a staged stream is asking for a level that does not exist.
    let src = "machine unibit\n\n\
        kernel t(rows: u32, cols: u32, a: [f32; rows, cols], b: [f32; cols, rows])\n    \
        space i, j : rows, cols\n    \
        tile 32, 32\n    \
        stream a : dram -> smem -> reg\n    \
        stream b : dram -> reg, drain\n    \
        at reg:\n        \
        b[j, i] = a[i, j]\n";
    let e = lyth_uasm::emit_program(&lower(src), 64).expect_err("must be refused");
    let msg = e.to_string();
    assert!(msg.contains("no shared memory to stage into"), "{msg}");
}

#[test]
fn a_ragged_n_is_refused_rather_than_rounded() {
    let e = lyth_uasm::emit_program(&lower(SAXPY), 65).expect_err("must be refused");
    let msg = e.to_string();
    assert!(msg.contains("not a multiple of 8"), "{msg}");
    assert!(msg.contains("no tail loop"), "{msg}");
}

#[test]
fn a_narrow_buffer_is_refused_because_the_machine_cannot_convert() {
    // Unibit's float unit is packed single precision only: there is no `cvt.f32.f16`. Emitting
    // it as f32 would run, and would move twice the bytes the cost model derived.
    let e = lyth_uasm::emit_program(&lower(&SAXPY.replace("[f32;", "[f16;")), 64)
        .expect_err("must be refused");
    assert!(e.to_string().contains("packed single precision only"), "{e}");
}

#[test]
fn the_default_scalar_matches_the_one_lyth_run_uses() {
    // Duplicated across two crates, so it is asserted rather than hoped. If `lyth run` ever
    // passes something else, the emitted program and the oracle would diverge on a kernel
    // with a scalar and the difference would be blamed on the emitter.
    assert_eq!(lyth_uasm::DEFAULT_SCALAR, 2.0);
}
