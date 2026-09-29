//! ADR-0029 P5: whole circuits on the GPU, against an outside oracle.
//!
//! A circuit runs as a sequence of launches through the **generated** bindings of `gate_q`,
//! `cu_q` and `swap_q` -- the code a caller in another repository would use -- ping-ponging two
//! state buffers. The same circuit, with the same matrices in f64, is built in Qiskit by
//! `tools/qgpu_oracle.py` and its `Statevector` is the reference. Nothing is shared between the
//! two but the list of gates.
//!
//! Pre-registered pass (ADR-0029): max |amplitude difference| <= 1e-5 and
//! 1 - |<psi_lyth|psi_qiskit>|^2 <= 1e-6 on all three circuits. Expected: 1e-7 .. 1e-6.
//!
//! The oracle needs a Python with Qiskit: `LYTH_QISKIT_PYTHON`, else the interpreter this
//! machine's other oracles use. Without one the test says it skipped rather than passing quietly.

use std::path::PathBuf;
use std::process::Command;

use lyth_cuda::{Buffer, Context};

mod gate_q {
    #![allow(clippy::manual_clamp, clippy::too_many_arguments, dead_code)]
    include!("generated/gate_q.rs");
}
mod cu_q {
    #![allow(clippy::manual_clamp, clippy::too_many_arguments, dead_code)]
    include!("generated/cu_q.rs");
}
mod swap_q {
    #![allow(clippy::manual_clamp, clippy::too_many_arguments, dead_code)]
    include!("generated/swap_q.rs");
}

/// `[a, b, c, d]` of `[[a, b], [c, d]]`, complex, in f64.
type M = [(f64, f64); 4];

#[derive(Clone, Debug)]
enum Gate {
    U(u32, M),
    CU(u32, u32, M),
    Swap(u32, u32),
}

const S: f64 = std::f64::consts::FRAC_1_SQRT_2;
const H: M = [(S, 0.0), (S, 0.0), (S, 0.0), (-S, 0.0)];
const X: M = [(0.0, 0.0), (1.0, 0.0), (1.0, 0.0), (0.0, 0.0)];
const Z: M = [(1.0, 0.0), (0.0, 0.0), (0.0, 0.0), (-1.0, 0.0)];

fn phase(phi: f64) -> M {
    [(1.0, 0.0), (0.0, 0.0), (0.0, 0.0), (phi.cos(), phi.sin())]
}

fn zyz(theta: f64, phi: f64, lambda: f64) -> M {
    let (c, s) = ((theta / 2.0).cos(), (theta / 2.0).sin());
    let e = |x: f64| (x.cos(), x.sin());
    let sc = |k: f64, z: (f64, f64)| (k * z.0, k * z.1);
    [(c, 0.0), sc(-s, e(lambda)), sc(s, e(phi)), sc(c, e(phi + lambda))]
}

/// SplitMix64: a fixed, documented generator, so the circuits are the same on every run.
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn angle(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64 * std::f64::consts::TAU
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
}

fn ghz(n: u32) -> Vec<Gate> {
    let mut g = vec![Gate::U(0, H)];
    g.extend((1..n).map(|t| Gate::CU(0, t, X)));
    g
}

/// The textbook QFT with its final swaps, on a random product state.
fn qft(n: u32, rng: &mut Rng) -> Vec<Gate> {
    let mut g: Vec<Gate> = (0..n).map(|q| Gate::U(q, zyz(rng.angle(), rng.angle(), rng.angle()))).collect();
    for j in (0..n).rev() {
        g.push(Gate::U(j, H));
        for k in (0..j).rev() {
            g.push(Gate::CU(k, j, phase(std::f64::consts::PI / f64::from(1u32 << (j - k)))));
        }
    }
    g.extend((0..n / 2).map(|i| Gate::Swap(i, n - 1 - i)));
    g
}

/// Layers of a ZYZ on every qubit, then CNOT or CZ on a random matching in a random direction.
fn random_circuit(n: u32, depth: u32, rng: &mut Rng) -> Vec<Gate> {
    let mut g = Vec::new();
    for _ in 0..depth {
        g.extend((0..n).map(|q| Gate::U(q, zyz(rng.angle(), rng.angle(), rng.angle()))));
        let mut qs: Vec<u32> = (0..n).collect();
        for i in (1..qs.len()).rev() {
            qs.swap(i, rng.below(i as u64 + 1) as usize);
        }
        for p in qs.chunks(2).filter(|p| p.len() == 2) {
            let (c, t) = if rng.below(2) == 0 { (p[0], p[1]) } else { (p[1], p[0]) };
            g.push(Gate::CU(c, t, if rng.below(2) == 0 { X } else { Z }));
        }
    }
    g
}

/// The inner width counts elements of the parent view, which has lost bit `c` (ADR-0029).
fn widths(c: u32, t: u32) -> (u32, u32) {
    (1 << c, if t < c { 1 << t } else { 1 << (t - 1) })
}

fn f32s(m: &M) -> [f32; 8] {
    let mut out = [0.0; 8];
    for (i, (r, im)) in m.iter().enumerate() {
        out[2 * i] = *r as f32;
        out[2 * i + 1] = *im as f32;
    }
    out
}

/// Run a circuit from |0...0> through the generated bindings, returning the final state.
fn run_on_gpu(ctx: &Context, qubits: u32, gates: &[Gate]) -> (Vec<f32>, Vec<f32>) {
    let n = 1u32 << qubits;
    let (m1, m2, m3) = (gate_q::module(ctx).unwrap(), cu_q::module(ctx).unwrap(), swap_q::module(ctx).unwrap());
    let (one, cu, sw) = (gate_q::GateQ::new(&m1).unwrap(), cu_q::CuQ::new(&m2).unwrap(), swap_q::SwapQ::new(&m3).unwrap());
    let mut zero = vec![0.0f32; n as usize];
    let mut re0 = zero.clone();
    re0[0] = 1.0;
    let mut a: (Buffer, Buffer) = (ctx.upload(&re0).unwrap(), ctx.upload(&zero).unwrap());
    zero.fill(0.0);
    let mut b: (Buffer, Buffer) = (ctx.upload(&zero).unwrap(), ctx.upload(&zero).unwrap());
    for g in gates {
        match g {
            Gate::U(q, m) => {
                let s = f32s(m);
                one.launch(n, 1 << q, s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7], &a.0, &a.1, &mut b.0, &mut b.1)
            }
            Gate::CU(c, t, m) => {
                let s = f32s(m);
                let (wc, wt) = widths(*c, *t);
                cu.launch(n, wc, wt, s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7], &a.0, &a.1, &mut b.0, &mut b.1)
            }
            Gate::Swap(x, y) => {
                let (wc, wt) = widths(*x, *y);
                sw.launch(n, wc, wt, &a.0, &a.1, &mut b.0, &mut b.1)
            }
        }
        .unwrap_or_else(|e| panic!("{g:?}: {e}"));
        std::mem::swap(&mut a, &mut b);
    }
    ctx.synchronize().unwrap();
    (a.0.download().unwrap(), a.1.download().unwrap())
}

fn spec(qubits: u32, gates: &[Gate]) -> serde_json::Value {
    let m = |m: &M| serde_json::json!(m.iter().map(|(r, i)| [r, i]).collect::<Vec<_>>());
    let gs: Vec<serde_json::Value> = gates
        .iter()
        .map(|g| match g {
            Gate::U(q, mm) => serde_json::json!({"kind": "u", "q": q, "m": m(mm)}),
            Gate::CU(c, t, mm) => serde_json::json!({"kind": "cu", "c": c, "t": t, "m": m(mm)}),
            Gate::Swap(x, y) => serde_json::json!({"kind": "swap", "a": x, "b": y}),
        })
        .collect();
    serde_json::json!({"qubits": qubits, "gates": gs})
}

fn python() -> PathBuf {
    if let Some(p) = std::env::var_os("LYTH_QISKIT_PYTHON") {
        return p.into();
    }
    let local = PathBuf::from("C:/Users/Drizzy/AppData/Local/Programs/Python/Python311/python.exe");
    if local.exists() { local } else { "python".into() }
}

/// Qiskit's state for the circuit, or `None` (with the reason printed) when there is no oracle.
fn oracle(qubits: u32, gates: &[Gate]) -> Option<(Vec<f64>, Vec<f64>)> {
    let dir = tempfile::tempdir().unwrap();
    let (src, dst) = (dir.path().join("circuit.json"), dir.path().join("state.json"));
    std::fs::write(&src, spec(qubits, gates).to_string()).unwrap();
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
        eprintln!("skipped: the oracle's Python has no Qiskit ({})", err.lines().last().unwrap_or(""));
        return None;
    }
    assert!(out.status.success(), "the oracle failed: {err}");
    let v: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(dst).unwrap()).unwrap();
    let arr = |k: &str| v[k].as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect::<Vec<f64>>();
    Some((arr("re"), arr("im")))
}

struct Agreement {
    max_abs: f64,
    infidelity: f64,
    norm_drift: f64,
}

fn compare(l: &(Vec<f32>, Vec<f32>), q: &(Vec<f64>, Vec<f64>)) -> Agreement {
    let (mut max_abs, mut dot_r, mut dot_i, mut nl, mut nq) = (0.0f64, 0.0, 0.0, 0.0, 0.0);
    for i in 0..q.0.len() {
        let (lr, li) = (f64::from(l.0[i]), f64::from(l.1[i]));
        let (qr, qi) = (q.0[i], q.1[i]);
        max_abs = max_abs.max(((lr - qr).powi(2) + (li - qi).powi(2)).sqrt());
        // <l|q> = sum conj(l) q
        dot_r += lr * qr + li * qi;
        dot_i += lr * qi - li * qr;
        nl += lr * lr + li * li;
        nq += qr * qr + qi * qi;
    }
    Agreement { max_abs, infidelity: 1.0 - (dot_r * dot_r + dot_i * dot_i) / (nl * nq), norm_drift: (nl - 1.0).abs() }
}

#[test]
fn circuits_on_the_gpu_agree_with_qiskit() {
    let ctx = match Context::new(0) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("skipped: no CUDA device ({e})");
            return;
        }
    };
    let mut rng = Rng(20_260_930);
    let circuits: Vec<(&str, u32, Vec<Gate>)> = vec![
        ("GHZ(12)", 12, ghz(12)),
        ("QFT(10), random product state", 10, qft(10, &mut rng)),
        ("random(10 qubits, depth 20)", 10, random_circuit(10, 20, &mut rng)),
    ];
    for (name, qubits, gates) in &circuits {
        let Some(q) = oracle(*qubits, gates) else { return };
        let l = run_on_gpu(&ctx, *qubits, gates);
        let a = compare(&l, &q);
        println!(
            "P5 {name}: {} gates, max |diff| {:.2e}, infidelity {:.2e}, norm drift {:.2e}",
            gates.len(),
            a.max_abs,
            a.infidelity,
            a.norm_drift
        );
        assert!(a.max_abs <= 1e-5, "{name}: max |diff| {:e} > 1e-5", a.max_abs);
        assert!(a.infidelity <= 1e-6, "{name}: infidelity {:e} > 1e-6", a.infidelity);
    }
    // GHZ is exact enough to say what it is: half the weight on |0...0>, half on |1...1>.
    let l = run_on_gpu(&ctx, 12, &ghz(12));
    let (w0, w1) = (l.0[0].powi(2) + l.1[0].powi(2), l.0[4095].powi(2) + l.1[4095].powi(2));
    assert!((w0 - 0.5).abs() < 1e-6 && (w1 - 0.5).abs() < 1e-6, "GHZ weights {w0}, {w1}");
}

#[test]
fn a_wrong_gate_order_is_caught_by_the_oracle() {
    // The comparison has teeth: the random circuit with its two-qubit gates reversed in
    // direction is a different circuit, and the oracle says so.
    let ctx = match Context::new(0) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("skipped: no CUDA device ({e})");
            return;
        }
    };
    let gates = random_circuit(8, 6, &mut Rng(7));
    let flipped: Vec<Gate> = gates
        .iter()
        .map(|g| match g {
            Gate::CU(c, t, m) if m == &X => Gate::CU(*t, *c, *m),
            other => other.clone(),
        })
        .collect();
    let Some(q) = oracle(8, &gates) else { return };
    let a = compare(&run_on_gpu(&ctx, 8, &flipped), &q);
    assert!(a.infidelity > 1e-2, "a CNOT with control and target swapped went unnoticed: {:e}", a.infidelity);
}
