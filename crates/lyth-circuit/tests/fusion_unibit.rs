//! ADR-0030 Q1 on the second machine: fused circuits on the Unibit emulator equal the unfused
//! circuit on the GPU, bit for bit, for every group whose loop nest the emulator can walk.
//!
//! Unibit walks a split as a loop nest read off `base_index` (step 4), and it does not spill: a
//! group whose values do not fit its register pool (19 for four buffers) is refused with the
//! reason. A refused group runs **unfused** on the emulator -- its gates one launch each, through
//! `gate_q`, `cu_q`, `swap_q` -- and since fusion changes no bit, the mixed run must still equal
//! the GPU's bit for bit. How many groups fused and how many fell back is printed: that count is
//! the known limit, stated, not hidden.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::process::Command;

use lyth_circuit::circuits::{ghz, qft, random, Rng};
use lyth_circuit::{fuse, run_unfused_gpu, Gate};
use lyth_cuda::Context;
use lyth_lang::program::{PrintRange, Program};

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

fn emulator() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../Labare/target/release/unibit.exe")
}

/// One program on the emulator: every float it printed.
fn emulate(asm: &str) -> Vec<f32> {
    let dir = tempfile::tempdir().unwrap();
    let src = dir.path().join("g.uasm");
    std::fs::write(&src, asm).unwrap();
    let out = Command::new(emulator()).args(["run", src.to_str().unwrap()]).output().expect("the Unibit emulator");
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    String::from_utf8_lossy(&out.stdout).lines().filter_map(|l| l.trim().parse::<f32>().ok()).collect()
}

fn program(n: usize, walk: usize, extents: BTreeMap<String, u32>, scalars: BTreeMap<String, f32>, re: &[f32], im: &[f32]) -> Program {
    Program {
        n: walk as u32,
        extents,
        scalars,
        prints: ["qr", "qi"].into_iter().map(|b| PrintRange { buffer: b.into(), lo: 0, hi: n as u32 }).collect(),
        buffers: BTreeMap::from([
            ("re".to_string(), re.to_vec()),
            ("im".to_string(), im.to_vec()),
            ("qr".to_string(), vec![1.0e30; n]),
            ("qi".to_string(), vec![1.0e30; n]),
        ]),
    }
}

fn example(name: &str) -> lyth_lang::ir::KernelIr {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(format!("../../examples/{name}.lyth"));
    let unit = lyth_lang::parse(&std::fs::read_to_string(path).unwrap()).unwrap();
    lyth_lang::ir::lower(&unit, &unit.kernels[0]).unwrap()
}

/// One gate, unfused, on the emulator.
fn unfused_gate(g: &Gate, n: usize, re: &[f32], im: &[f32]) -> Vec<f32> {
    let names = ["ar", "ai", "br", "bi", "cr", "ci", "dr", "di"];
    let sc = |m: &[(f64, f64); 4]| names.iter().map(|s| s.to_string()).zip(lyth_circuit::f32s(m)).collect::<BTreeMap<_, _>>();
    let (k, extents, scalars, walk) = match g {
        Gate::U(q, m) => (example("gate_q"), vec![("w", 1u32 << q)], sc(m), n / 2),
        Gate::CU(c, t, m) => {
            let (wc, wt) = lyth_circuit::widths_ct(*c, *t);
            (example("cu_q"), vec![("wc", wc), ("wt", wt)], sc(m), n / 4)
        }
        Gate::Swap(a, b) => {
            let (wc, wt) = lyth_circuit::widths_ct(*a, *b);
            (example("swap_q"), vec![("wc", wc), ("wt", wt)], BTreeMap::new(), n / 4)
        }
    };
    let mut ex = BTreeMap::from([("n".to_string(), n as u32)]);
    ex.extend(extents.into_iter().map(|(a, b)| (a.to_string(), b)));
    emulate(&lyth_uasm::emit(&k, &program(n, walk, ex, scalars, re, im)).expect("an unfused gate fits Unibit"))
}

/// The final state, (groups fused, groups that fell back), and the first refusal.
type Outcome = ((Vec<f32>, Vec<f32>), (usize, usize), Option<String>);

/// The circuit on the emulator: each group fused if Unibit can hold it, unfused otherwise.
/// Returns the state and (groups fused, groups that fell back).
fn run_on_unibit(qubits: u32, gates: &[Gate], k: usize) -> Outcome {
    let n = 1usize << qubits;
    let mut re = vec![0.0f32; n];
    re[0] = 1.0;
    let mut im = vec![0.0f32; n];
    let (mut fused, mut fell, mut why) = (0, 0, None);
    for (gi, g) in fuse(gates, k).iter().enumerate() {
        let kir = g.lower(&format!("fused{gi}")).unwrap();
        let mut ex = BTreeMap::from([("n".to_string(), n as u32)]);
        for (l, w) in g.widths().into_iter().enumerate() {
            ex.insert(format!("w{l}"), w);
        }
        let got = match lyth_uasm::emit(&kir, &program(n, n >> g.qubits.len(), ex, BTreeMap::new(), &re, &im)) {
            Ok(asm) => {
                fused += 1;
                emulate(&asm)
            }
            Err(e) => {
                let e = e.to_string();
                assert!(e.contains("registers for") || e.contains("loop counters"), "refused for an unexpected reason: {e}");
                why.get_or_insert(format!("qubits {:?}: {}", g.qubits, e.lines().next().unwrap_or("")));
                fell += 1;
                let mut st = [re.clone(), im.clone()].concat();
                for gate in &g.gates {
                    st = unfused_gate(gate, n, &st[..n], &st[n..]);
                }
                st
            }
        };
        assert_eq!(got.len(), 2 * n);
        re = got[..n].to_vec();
        im = got[n..].to_vec();
    }
    ((re, im), (fused, fell), why)
}

#[test]
fn fused_circuits_on_unibit_equal_the_unfused_gpu_run_bit_for_bit() {
    if !emulator().exists() {
        eprintln!("skipped: no Unibit emulator at {} (cargo build --release in Unibit)", emulator().display());
        return;
    }
    let ctx = match Context::new(0) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("skipped: no CUDA device ({e})");
            return;
        }
    };
    let mut rng = Rng(20_260_930);
    let circuits = vec![
        ("GHZ(10)", 10, ghz(10)),
        ("QFT(10)", 10, qft(10, &mut rng)),
        ("random(10, depth 4)", 10, random(10, 4, &mut rng)),
    ];
    for (name, q, gates) in circuits {
        let (ur, ui) = run_unfused_gpu(&ctx, q, &gates).unwrap();
        for k in 1..=5 {
            let ((fr, fi), (fused, fell), why) = run_on_unibit(q, &gates, k);
            assert_eq!((bits(&fr), bits(&fi)), (bits(&ur), bits(&ui)), "{name}, k = {k}: Unibit vs GPU unfused");
            assert!(k > 2 || fell == 0, "{name}, k = {k}: a group of at most two qubits must fit Unibit");
            println!(
                "Q1 Unibit {name}, k = {k}: bit-exact against the GPU; {fused} group(s) fused, {fell} fell back unfused{}",
                why.map(|w| format!(" -- first refusal: {w}")).unwrap_or_default()
            );
        }
    }
}
