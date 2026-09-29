//! ADR-0030 Q1 and Q5: a fused circuit is the unfused circuit, bit for bit, and still Qiskit's.

use std::path::PathBuf;
use std::process::Command;

use lyth_circuit::circuits::{ghz, qft, random, Rng};
use lyth_circuit::{fuse, run_fused_gpu, run_fused_host, run_unfused_gpu, Gate};
use lyth_cuda::Context;

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

fn suite() -> Vec<(&'static str, u32, Vec<Gate>)> {
    let mut rng = Rng(20_260_930);
    vec![
        ("GHZ(12)", 12, ghz(12)),
        ("QFT(10)", 10, qft(10, &mut rng)),
        ("random(10, depth 20)", 10, random(10, 20, &mut rng)),
        ("random(20, depth 10)", 20, random(20, 10, &mut Rng(20_260_931))),
    ]
}

#[test]
fn the_fuser_keeps_every_gate_in_order_and_respects_k() {
    for (name, _, gates) in suite() {
        for k in 1..=5 {
            let groups = fuse(&gates, k);
            let flat: Vec<Gate> = groups.iter().flat_map(|g| g.gates.clone()).collect();
            assert_eq!(flat, gates, "{name}, k = {k}: the gates, in order");
            for g in &groups {
                let need = g.gates.iter().map(|x| x.qubits().len()).max().unwrap();
                assert!(g.qubits.len() <= k.max(need), "{name}, k = {k}: {:?}", g.qubits);
                assert!(g.qubits.windows(2).all(|w| w[0] > w[1]), "descending");
            }
        }
    }
}

#[test]
fn every_generated_group_is_ordinary_lyth_and_costs_one_pass() {
    let (_, _, gates) = &suite()[2];
    for k in 1..=5 {
        for (i, g) in fuse(gates, k).iter().enumerate() {
            let kir = g.lower(&format!("g{i}")).unwrap_or_else(|e| panic!("k = {k}, group {i}: {e}"));
            let amps = 1u32 << g.qubits.len();
            // One pass: 16 bytes per amplitude (read re, im; write re, im), whatever the gate count.
            assert_eq!(kir.cost.bytes_per_element(), Some(16.0 * f64::from(amps)), "k = {k}, group {i}");
        }
    }
}

#[test]
fn a_fused_circuit_is_the_unfused_circuit_bit_for_bit_on_the_gpu() {
    let ctx = match Context::new(0) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("skipped: no CUDA device ({e})");
            return;
        }
    };
    for (name, q, gates) in suite() {
        let (ur, ui) = run_unfused_gpu(&ctx, q, &gates).unwrap();
        let mut passes = Vec::new();
        for k in 1..=5 {
            let (fr, fi, p) = run_fused_gpu(&ctx, q, &gates, k).unwrap_or_else(|e| panic!("{name}, k = {k}: {e}"));
            assert_eq!(bits(&fr), bits(&ur), "{name}, k = {k}: re");
            assert_eq!(bits(&fi), bits(&ui), "{name}, k = {k}: im");
            passes.push(p);
        }
        println!("Q1 {name}: {} gates; passes at k = 1..5: {passes:?}; fused == unfused bit for bit", gates.len());
    }
}

#[test]
fn the_host_oracle_agrees_bit_for_bit_with_both() {
    let ctx = match Context::new(0) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("skipped: no CUDA device ({e})");
            return;
        }
    };
    for (name, q, gates) in suite().into_iter().filter(|(_, q, _)| *q <= 12) {
        let (ur, ui) = run_unfused_gpu(&ctx, q, &gates).unwrap();
        for k in [1, 3, 5] {
            let (hr, hi) = run_fused_host(q, &gates, k).unwrap();
            assert_eq!((bits(&hr), bits(&hi)), (bits(&ur), bits(&ui)), "{name}, k = {k}: host fused vs GPU unfused");
        }
    }
}

// ------------------------------------------------------------------ Q5: against Qiskit

fn python() -> PathBuf {
    if let Some(p) = std::env::var_os("LYTH_QISKIT_PYTHON") {
        return p.into();
    }
    let local = PathBuf::from("C:/Users/Drizzy/AppData/Local/Programs/Python/Python311/python.exe");
    if local.exists() { local } else { "python".into() }
}

fn oracle(qubits: u32, gates: &[Gate]) -> Option<(Vec<f64>, Vec<f64>)> {
    let m = |m: &[(f64, f64); 4]| serde_json::json!(m.iter().map(|(r, i)| [r, i]).collect::<Vec<_>>());
    let gs: Vec<serde_json::Value> = gates
        .iter()
        .map(|g| match g {
            Gate::U(q, mm) => serde_json::json!({"kind": "u", "q": q, "m": m(mm)}),
            Gate::CU(c, t, mm) => serde_json::json!({"kind": "cu", "c": c, "t": t, "m": m(mm)}),
            Gate::Swap(a, b) => serde_json::json!({"kind": "swap", "a": a, "b": b}),
        })
        .collect();
    let dir = tempfile::tempdir().unwrap();
    let (src, dst) = (dir.path().join("c.json"), dir.path().join("s.json"));
    std::fs::write(&src, serde_json::json!({"qubits": qubits, "gates": gs}).to_string()).unwrap();
    let tool = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tools/qgpu_oracle.py");
    let out = match Command::new(python()).args([tool.as_os_str(), src.as_os_str(), dst.as_os_str()]).output() {
        Ok(o) => o,
        Err(e) => {
            eprintln!("skipped: no Python for the Qiskit oracle ({e})");
            return None;
        }
    };
    let err = String::from_utf8_lossy(&out.stderr);
    if !out.status.success() && err.contains("No module named") {
        eprintln!("skipped: no Qiskit");
        return None;
    }
    assert!(out.status.success(), "{err}");
    let v: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(dst).unwrap()).unwrap();
    let arr = |k: &str| v[k].as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect::<Vec<f64>>();
    Some((arr("re"), arr("im")))
}

#[test]
fn fused_circuits_are_still_qiskits_answer() {
    let ctx = match Context::new(0) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("skipped: no CUDA device ({e})");
            return;
        }
    };
    for (name, q, gates) in suite() {
        let Some((qr, qi)) = oracle(q, &gates) else { return };
        for k in [3, 5] {
            let (lr, li, passes) = run_fused_gpu(&ctx, q, &gates, k).unwrap();
            let (mut max_abs, mut dr, mut di, mut nl, mut nq) = (0.0f64, 0.0, 0.0, 0.0, 0.0);
            for i in 0..qr.len() {
                let (a, b) = (f64::from(lr[i]), f64::from(li[i]));
                max_abs = max_abs.max(((a - qr[i]).powi(2) + (b - qi[i]).powi(2)).sqrt());
                dr += a * qr[i] + b * qi[i];
                di += a * qi[i] - b * qr[i];
                nl += a * a + b * b;
                nq += qr[i] * qr[i] + qi[i] * qi[i];
            }
            let infidelity = 1.0 - (dr * dr + di * di) / (nl * nq);
            println!("Q5 {name}, k = {k}: {passes} passes, max |diff| {max_abs:.2e}, infidelity {infidelity:.2e}");
            assert!(max_abs <= 1e-5 && infidelity <= 1e-6, "{name}, k = {k}");
        }
    }
}
