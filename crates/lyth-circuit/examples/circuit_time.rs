//! ADR-0030 Q3: time one circuit, fused at `k` or unfused, and print the samples as JSON.
//!
//!     circuit_time <unfused|K> <qubits> <depth> <seed> <reps> <out.json>
//!
//! The circuit is `circuits::random(qubits, depth)` from the given seed. Modules are loaded before
//! timing; one warm-up run is discarded; each timed run is first launch to last synchronize.

use lyth_circuit::circuits::{random, Rng};
use lyth_circuit::{execute, Mode};
use lyth_cuda::Context;

fn main() {
    let a: Vec<String> = std::env::args().collect();
    if a.len() != 7 {
        eprintln!("usage: circuit_time <unfused|K> <qubits> <depth> <seed> <reps> <out.json>");
        std::process::exit(2);
    }
    let mode = if a[1] == "unfused" { Mode::Unfused } else { Mode::Fused(a[1].parse().expect("k")) };
    let (q, depth, seed, reps): (u32, u32, u64, usize) =
        (a[2].parse().unwrap(), a[3].parse().unwrap(), a[4].parse().unwrap(), a[5].parse().unwrap());
    let gates = random(q, depth, &mut Rng(seed));
    let ctx = Context::new(0).expect("a CUDA device");
    let e = execute(&ctx, q, &gates, mode, reps).expect("the circuit runs");
    let mut sorted = e.ms.clone();
    sorted.sort_by(|x, y| x.partial_cmp(y).unwrap());
    let median = sorted[sorted.len() / 2];
    let json = format!(
        "{{\"mode\": \"{}\", \"qubits\": {q}, \"gates\": {}, \"passes\": {}, \"ms\": {:?}, \"ms_median\": {median}}}\n",
        a[1],
        gates.len(),
        e.passes,
        e.ms
    );
    std::fs::write(&a[6], json).expect("write the result");
}
