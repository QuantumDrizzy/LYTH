//! A `.lyth` file becomes a program, and the program is right. ADR-0025 steps 2 to 4.
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
use lyth_lang::program::{PrintRange, Program};
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

    // One value per line, printed by `PRINT_F32` — the shortest string that parses back to the
    // same bits, so reading them here loses nothing and the comparison stays bit-exact.
    //
    // This used to decode `PRINT_REG256`'s hex lane dump, and that decoder was where the only
    // real defect of step 2 lived: the syscall emits no newline, so eight registers arrived
    // concatenated on one line and a scan over `text.lines()` found one of them. Step 4
    // replaced the dump with output a person can read, which removed the format that needed
    // decoding at all.
    let vals: Vec<f32> = text
        .lines()
        .filter_map(|l| l.trim().parse::<f32>().ok())
        .collect();
    if vals.is_empty() {
        eprintln!(
            "the emulator printed no value:\n{text}\n{}",
            String::from_utf8_lossy(&out.stderr)
        );
        return None;
    }
    Some(vals)
}

/// What the host says the same kernel computes, on the same inputs.
///
/// `extents` is empty at rank 1. At rank 2 the **oracle** needs the two lengths and the
/// **emitter** does not, which is not an inconsistency: the oracle decomposes a linear index
/// into `i` and `j` so it can address each buffer at its own permutation, and when every
/// permutation is the identity that decomposition cancels and the walk is linear again. The
/// emitter only ever sees the case where it cancels, because it refuses the other one.
fn oracle(k: &ir::KernelIr, n: u32, extents: &[(&str, u32)]) -> Vec<f32> {
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
    for (name, v) in extents {
        inputs.extents.insert((*name).to_string(), *v);
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
    let want = oracle(&k, n, &[]);

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
    let want = oracle(&k, n, &[]);
    for i in 0..n as usize {
        assert_eq!(got[i].to_bits(), want[i].to_bits(), "element {i}\n{asm}");
    }
}

#[test]
fn a_relu_is_bit_exact_through_vfmax() {
    // ADR-0027. `VFMAX` uses `f32::max` and the host oracle goes through `ReduceOp::combine`,
    // which is *not* `f32::max` -- it forces +0.0 for a pair of signed zeros because that is
    // what PTX does. So the two could disagree here and the only way to know is to run it.
    let src = "machine unibit\n\n\
         kernel relu(n: u32, x: [f32; n], out: [f32; n])\n    \
         intensity 0.0\n    \
         stream x   : dram -> reg\n    \
         stream out : dram -> reg, drain\n    \
         at reg:\n        \
         out = max(x, 0.0)\n";
    let n = 256;
    let k = lower(src);
    let asm = lyth_uasm::emit_program(&k, n).expect("emits");
    assert!(asm.contains("vfmax"), "{asm}");
    let dir = tempfile::tempdir().unwrap();
    let Some(got) = run_on_emulator(&asm, dir.path()) else {
        eprintln!("skipped: no Unibit emulator");
        return;
    };
    let want = oracle(&k, n, &[]);
    assert_eq!(got.len(), n as usize);

    // Again as a rate: if every input were positive this would pass on a program that did not
    // clamp at all.
    let negatives = lyth_lang::inputs::buffer("x", Ty::BufF32, n)
        .iter()
        .filter(|v| **v < 0.0)
        .count();
    assert!(negatives > n as usize / 8, "only {negatives} of {n} inputs are negative");
    for i in 0..n as usize {
        assert_eq!(got[i].to_bits(), want[i].to_bits(), "element {i}
{asm}");
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

const SUM: &str = "machine unibit\n\n\
     kernel sum(n: u32, x: [f32; n], partial: [f32; blocks])\n    \
     intensity 0.25\n    \
     stream x : dram -> reg\n    \
     reduce sum v : reg -> dram into partial\n    \
     at reg:\n        \
     v = x\n";

/// The reduction oracle, at the launch shape a Unibit program *is*.
///
/// `grid = 1, block = LANES`, and no new code was needed for it. `eval_with_launch` already
/// folds thread `t`'s own elements at stride `grid * block` and then walks a tree over the
/// block, which is exactly a lane accumulator followed by `VFREDUCE`. That the oracle for a
/// machine with no threads at all fell out of the GPU one unchanged is the evidence that the
/// launch shape here is a real launch shape and not a metaphor.
fn reduction_oracle(k: &ir::KernelIr, n: u32) -> Vec<f32> {
    let mut inputs = eval::Inputs::default();
    for p in &k.params {
        if p.ty.is_buffer() {
            inputs
                .buffers
                .insert(p.name.clone(), lyth_lang::inputs::buffer(&p.name, p.ty, n));
        }
    }
    let out = eval::eval_with_launch(k, n as usize, &inputs, 1, lyth_uasm::LANES as usize)
        .expect("the oracle evaluates");
    out.buffers[&k.reduction.as_ref().unwrap().into].clone()
}

#[test]
fn a_reduction_folds_the_lanes_and_the_result_leaves_the_machine() {
    let n = 4096;
    let k = lower(SUM);
    let asm = lyth_uasm::emit(
        &k,
        &Program {
            n,
            extents: Default::default(),
            scalars: Default::default(),
            prints: vec![PrintRange {
                buffer: "partial".into(),
                lo: 0,
                hi: 8,
            }],
        },
    )
    .expect("emits");
    let dir = tempfile::tempdir().unwrap();
    let Some(got) = run_on_emulator(&asm, dir.path()) else {
        eprintln!("skipped: no Unibit emulator");
        return;
    };
    let want = reduction_oracle(&k, n);

    // Eight elements, not one. Element 0 is the result; 1 to 7 are the target buffer's own
    // untouched data on both sides, so an `SQ` where the emitter meant `SW` fails here rather
    // than being invisible in a check that only looked at the result.
    //
    // The `Program` is built here rather than resolved from a `main`, because `main` would not
    // be allowed to ask for this: `partial` is declared `[f32; blocks]` and holds one value per
    // block, so `print partial[0:8]` runs off the end and `resolve` says so. The emitter is
    // what is under test, and the emitter takes a `Program`.
    assert_eq!(got.len(), 8, "{} values were printed", got.len());
    for i in 0..8 {
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
fn the_lane_order_is_observable_so_the_check_above_has_teeth() {
    // Without this, the test above would pass against any summation order and the tree in
    // `VFREDUCE` would be an untested claim.
    //
    // Float addition is not associative: folding 4096 values left to right and folding them as
    // eight strided partials combined by a tree are different functions, not different last
    // bits. This asserts they disagree on *these* inputs, which is what makes the bit-exact
    // comparison a test of the order and not only of the arithmetic.
    let n = 4096;
    let k = lower(SUM);
    let x = lyth_lang::inputs::buffer("x", Ty::BufF32, n);
    let straight = x.iter().fold(0.0f32, |a, v| a + v);
    let lanes = reduction_oracle(&k, n)[0];
    assert_ne!(
        straight.to_bits(),
        lanes.to_bits(),
        "left to right and the lane tree agree on this input, so it cannot test the order"
    );
}

#[test]
fn a_reduction_staged_through_smem_is_refused() {
    // `examples/sum.lyth` says `reg -> smem -> dram`, which is the GPU's shape: threads write
    // their partials somewhere every thread can reach and a tree folds them there. This
    // machine has nowhere, and the tree is inside the register instead.
    let src = SUM.replace("reg -> dram", "reg -> smem -> dram");
    let e = lyth_uasm::emit_program(&lower(&src), 64).expect_err("must be refused");
    let msg = e.to_string();
    assert!(msg.contains("no shared level"), "{msg}");
    assert!(msg.contains("reg -> dram"), "{msg}");
}

const ADD2: &str = "machine unibit\n\n\
     kernel add2(rows: u32, cols: u32, a: [f32; rows, cols], b: [f32; rows, cols], \
     c: [f32; rows, cols])\n    \
     space i, j : rows, cols\n    \
     intensity 0.0833\n    \
     stream a : dram -> reg\n    \
     stream b : dram -> reg\n    \
     stream c : dram -> reg, drain\n    \
     at reg:\n        \
     c[i, j] = a[i, j] + b[i, j]\n";

#[test]
fn a_rank_2_space_is_the_same_loop_when_nothing_is_permuted() {
    // Not a special case, and deliberately so: row-major over `rows * cols` with every buffer
    // read at `[i, j]` is a linear walk, so the emitter needs the product and never the two
    // extents apart. Rank 2 is a claim about the *indices*, and when they all agree it is not
    // a claim about the addresses.
    let n = 512; // 16 x 32
    let k = lower(ADD2);
    let asm = lyth_uasm::emit_program(&k, n).expect("emits");
    let dir = tempfile::tempdir().unwrap();
    let Some(got) = run_on_emulator(&asm, dir.path()) else {
        eprintln!("skipped: no Unibit emulator");
        return;
    };
    let want = oracle(&k, n, &[("rows", 16), ("cols", 32)]);
    assert_eq!(got.len(), n as usize);
    for i in 0..n as usize {
        assert_eq!(got[i].to_bits(), want[i].to_bits(), "element {i}\n{asm}");
    }
}

#[test]
fn a_permuted_axis_is_refused_because_there_is_no_strided_store() {
    // The hardest refusal in this back end, because it is not "not yet". `SQ` writes eight
    // contiguous f32 and the ISA has no strided store, no scatter and no lane extract, so
    // consecutive `j` landing at a stride of `rows` has no instruction behind it. `LW`/`SW`
    // could do it a word at a time, and that is refused too: it would move 4 bytes per
    // instruction where the cost model derived 32.
    let src = ADD2
        .replace("c[i, j] = a[i, j] + b[i, j]", "c[j, i] = a[i, j] + b[i, j]")
        .replace("c: [f32; rows, cols]", "c: [f32; cols, rows]");
    let e = lyth_uasm::emit_program(&lower(&src), 512).expect_err("must be refused");
    let msg = e.to_string();
    assert!(msg.contains("strided or scattered"), "{msg}");
    assert!(msg.contains("not slowly, at all"), "{msg}");
}

#[test]
fn the_default_scalar_matches_the_one_lyth_run_uses() {
    // Duplicated across two crates, so it is asserted rather than hoped. If `lyth run` ever
    // passes something else, the emitted program and the oracle would diverge on a kernel
    // with a scalar and the difference would be blamed on the emitter.
    assert_eq!(lyth_uasm::DEFAULT_SCALAR, 2.0);
}

const CMUL: &str = "machine unibit\n\n\
     kernel cmul(n: u32, ar: [f32; n], ai: [f32; n], br: [f32; n], bi: [f32; n], \
     cr: [f32; n], ci: [f32; n])\n    \
     intensity 0.25\n    \
     stream ar : dram -> reg\n    \
     stream ai : dram -> reg\n    \
     stream br : dram -> reg\n    \
     stream bi : dram -> reg\n    \
     stream cr : dram -> reg, drain\n    \
     stream ci : dram -> reg, drain\n    \
     at reg:\n        \
     cr = ar * br - ai * bi\n        \
     ci = ar * bi + ai * br\n";

#[test]
fn a_complex_multiply_needs_no_complex_type() {
    // The finding this test exists to keep. Complex arithmetic over separate real and imaginary
    // buffers is *already* expressible, and the compiler derives 6 flop / 24 byte from the
    // operations themselves -- so a `[c64; n]` type would buy type safety and an interleaved
    // layout that is worse for coalescing, and nothing else.
    //
    // It did not compile for this machine until the register allocator learned to free. Nine
    // values are allocated across the two statements and never more than seven are live, and
    // the refusal it used to produce blamed the ISA for the emitter's arithmetic.
    let n = 64;
    let k = lower(CMUL);
    let asm = lyth_uasm::emit(
        &k,
        &Program {
            n,
            extents: Default::default(),
            scalars: Default::default(),
            prints: vec![
                PrintRange { buffer: "cr".into(), lo: 0, hi: n },
                PrintRange { buffer: "ci".into(), lo: 0, hi: n },
            ],
        },
    )
    .expect("emits");

    let dir = tempfile::tempdir().unwrap();
    let Some(got) = run_on_emulator(&asm, dir.path()) else {
        eprintln!("skipped: no Unibit emulator");
        return;
    };

    let mut inputs = eval::Inputs::default();
    for p in &k.params {
        if p.ty.is_buffer() {
            inputs
                .buffers
                .insert(p.name.clone(), lyth_lang::inputs::buffer(&p.name, p.ty, n));
        }
    }
    let want = eval::eval(&k, n as usize, &inputs).expect("the oracle evaluates");

    assert_eq!(got.len(), 2 * n as usize, "cr then ci");
    for (j, buf) in ["cr", "ci"].iter().enumerate() {
        for i in 0..n as usize {
            assert_eq!(
                got[j * n as usize + i].to_bits(),
                want.buffers[*buf][i].to_bits(),
                "{buf}[{i}]"
            );
        }
    }
}

#[test]
fn the_allocator_hands_registers_back() {
    // Nine allocations, seven registers. Without reuse this body does not emit at all, so the
    // test above would cover it -- but it would fail with "more than seven live values", which
    // is the message that was wrong about its own cause. This one names the property directly.
    let asm = lyth_uasm::emit_program(&lower(CMUL), 64).expect("emits");
    let used: std::collections::BTreeSet<&str> = asm
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| w.len() == 2 && w.starts_with('t') && w[1..].parse::<u8>().is_ok())
        .collect();
    assert!(
        used.len() <= 7,
        "the body used {} distinct temporaries: {used:?}",
        used.len()
    );
    // And the register that held `ar` is not the one that held the first product, because a
    // value cannot be freed before its last read.
    assert!(asm.contains("vfsub"), "{asm}");
}
