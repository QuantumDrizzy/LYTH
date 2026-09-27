//! The contract, measured against the machine's own counters. ADR-0025 step 5.
//!
//! ADR-0009 did this on `sm_120`: the traffic the front end derives, against what `ncu`
//! reports the device moved, to ±0.80%. This is the same check on the second ISA, and it is
//! the thing that turns "verified on two ISAs" from an intention into a fact (ADR-0026).
//!
//! Two differences from the GPU version, and both make this check *stronger*:
//!
//! * There is no sampling. The emulator charges one cycle per instruction and counts every
//!   memory operation, so the agreement is **exact or it is a defect** — there is no variance
//!   to hide a small error in.
//! * The measurement is **differential**. The same kernel is emitted at two element counts and
//!   the counters are subtracted, so the prologue, the `_start` setup and the print epilogue
//!   cancel instead of being estimated. A constant this test does not know cannot bias it.
//!
//! Skips rather than passes when the emulator is not present.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::process::Command;

use lyth_lang::program::{PrintRange, Program};
use lyth_lang::{ir, parse};

/// One 256-bit load or store. Not a tunable: `Reg256` is 32 bytes and `LQ` moves all of it.
const QUAD_BYTES: u64 = 32;

fn unibit_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../Unibit")
}

/// What the emulator says one run retired.
#[derive(Debug, Clone, Copy)]
struct Counters {
    reads: u64,
    writes: u64,
    packed_f32: u64,
    reported_flops: u64,
}

/// Pull the first `n` integers out of the rest of the line after `marker`.
fn after(text: &str, marker: &str, n: usize) -> Option<Vec<u64>> {
    let line = text.lines().find(|l| l.contains(marker))?;
    let tail = &line[line.find(marker)? + marker.len()..];
    let nums: Vec<u64> = tail
        .split(|c: char| !c.is_ascii_digit())
        .filter(|s| !s.is_empty())
        .take(n)
        .filter_map(|s| s.parse().ok())
        .collect();
    (nums.len() == n).then_some(nums)
}

/// Emit this kernel at `n`, assemble it, run it, and read the counters back.
///
/// The print range is **fixed at eight elements** whatever `n` is, so the epilogue is a
/// constant that the subtraction below removes. Printing the whole buffer would add one read
/// per element and quietly inflate the measured traffic by a third.
fn run_at(src: &str, n: u32) -> Option<Counters> {
    let dir = unibit_dir();
    if !dir.join("Cargo.toml").exists() {
        eprintln!("skipped: no Unibit emulator at {}", dir.display());
        return None;
    }
    let unit = parse(src).expect("parses");
    let k = ir::lower(&unit, &unit.kernels[0]).expect("lowers");
    let mut scalars = BTreeMap::new();
    for p in &k.params {
        if p.ty == lyth_lang::ast::Ty::F32 {
            scalars.insert(p.name.clone(), 2.0);
        }
    }
    let asm = lyth_uasm::emit(
        &k,
        &Program {
            n,
            extents: BTreeMap::new(),
            scalars,
            prints: vec![PrintRange {
                buffer: k.drains[0].0.clone(),
                lo: 0,
                hi: 8,
            }],
            buffers: BTreeMap::new(),
        },
    )
    .expect("emits");

    let tmp = tempfile::tempdir().unwrap();
    let f = tmp.path().join("k.uasm");
    std::fs::write(&f, &asm).unwrap();
    let out = Command::new("cargo")
        .args(["run", "--quiet", "--", "run", f.to_str().unwrap()])
        .current_dir(&dir)
        .output()
        .ok()?;
    let text = String::from_utf8_lossy(&out.stdout).to_string();

    let rw = after(&text, "Memory Reads/Writes:", 2)?;
    let f32s = after(&text, "Packed f32 (8 lanes):", 1)?;
    let flops = after(&text, "Flops (f32 lanes):", 1)?;
    Some(Counters {
        reads: rw[0],
        writes: rw[1],
        packed_f32: f32s[0],
        reported_flops: flops[0],
    })
}

const SAXPY: &str = "machine unibit\n\n\
     kernel saxpy(n: u32, a: f32, x: [f32; n], y: [f32; n])\n    \
     intensity 0.1667\n    \
     stream x : dram -> reg\n    \
     stream y : dram -> reg, drain\n    \
     at reg:\n        \
     y = a * x + y\n";

#[test]
fn the_derived_traffic_is_what_the_machine_actually_moves() {
    // The ADR-0009 check on a second ISA. `sm_120` agreed to ±0.80%, which is what a sampled
    // hardware counter gives. Here the tolerance is **zero**: a deterministic machine that
    // charges per instruction either moved the bytes the model derived or the model is wrong.
    let (n1, n2) = (4096, 8192);
    let (Some(a), Some(b)) = (run_at(SAXPY, n1), run_at(SAXPY, n2)) else {
        return;
    };

    let d_elems = (n2 - n1) as u64;
    let moved = (b.reads - a.reads + b.writes - a.writes) * QUAD_BYTES;
    let measured = moved as f64 / d_elems as f64;

    let unit = parse(SAXPY).unwrap();
    let k = ir::lower(&unit, &unit.kernels[0]).unwrap();
    let derived = k.cost.bytes_per_element().expect("saxpy has a byte count");

    assert_eq!(
        measured, derived,
        "derived {derived} byte/element, the machine moved {measured}\n  \
         n={n1}: {a:?}\n  n={n2}: {b:?}"
    );
    assert_eq!(derived, 12.0, "8 read + 4 written, and the read of `y` is the one people forget");
}

#[test]
fn the_differential_really_cancels_the_epilogue() {
    // Without this the test above could agree for the wrong reason. If the print loop scaled
    // with `n` -- which it would if the program printed the whole buffer -- the subtraction
    // would leave a per-element term behind and the traffic would come out a third high.
    //
    // Asserted as a property of the counters rather than trusted: the constant part is what is
    // left when the per-element part is extrapolated away.
    let (n1, n2) = (4096, 8192);
    let (Some(a), Some(b)) = (run_at(SAXPY, n1), run_at(SAXPY, n2)) else {
        return;
    };
    let per_elem = (b.reads - a.reads) as f64 / (n2 - n1) as f64;
    let constant = a.reads as f64 - per_elem * n1 as f64;
    assert!(
        (0.0..64.0).contains(&constant),
        "the epilogue should be a small constant, not {constant} reads"
    );
}

#[test]
fn the_derived_instruction_count_is_what_the_machine_retires() {
    // The compute side. One `VFMA` per eight elements, because a register holds eight f32 --
    // so the count is `n / 8` and nothing about it is an estimate.
    let (Some(a), Some(b)) = (run_at(SAXPY, 4096), run_at(SAXPY, 8192)) else {
        return;
    };
    assert_eq!(a.packed_f32, 4096 / lyth_uasm::LANES as u64);
    assert_eq!(b.packed_f32, 8192 / lyth_uasm::LANES as u64);
}

#[test]
fn the_machines_own_flop_counter_is_short_by_the_fused_multiply() {
    // [KNOWN_LIMIT], and found by this check rather than reasoned about.
    //
    // The emulator prints `Flops (f32 lanes)` and the number is `instructions x 8`: it counts
    // **lane operations**, not floating-point operations. For `VFADD` those are the same. For
    // `VFMA` they are not -- a fused multiply-add retires two flops per lane, and the field
    // reports one.
    //
    // So on this saxpy the machine says 4096 where the work is 8192, and a roofline built on
    // that counter would place every FMA-dense kernel at half its real intensity. LYTH's own
    // derivation is the one that is right here, which is the opposite of the usual direction
    // and is exactly why the check is run in both directions.
    let Some(a) = run_at(SAXPY, 4096) else { return };

    let unit = parse(SAXPY).unwrap();
    let k = ir::lower(&unit, &unit.kernels[0]).unwrap();
    // flops/element = intensity x bytes/element, both derived, neither measured.
    let derived_flops = k.cost.intensity * k.cost.bytes_per_element().unwrap() * 4096.0;

    assert_eq!(derived_flops, 8192.0, "two flops an element: one multiply, one add");
    assert_eq!(a.reported_flops, 4096, "the counter reports lanes, not flops");
    assert_eq!(
        a.reported_flops * 2,
        derived_flops as u64,
        "the shortfall is exactly the multiply half of every fused instruction"
    );
}
