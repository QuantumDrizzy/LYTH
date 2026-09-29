//! ADR-0030 Q2: write one fused group of `g` gates on a qubit set as a `.lyth` file, and print the
//! `--set` arguments `lyth run` needs for it.
//!
//!     q2_group <q,q,...> <gates> <seed> <out.lyth>
//!
//! Gates: every third a controlled X or Z between two qubits of the set, the rest a ZYZ on one,
//! all from the seed. The group is taken whole: it is one pass, whatever `k` would have cut.

use lyth_circuit::circuits::{zyz, Rng, X, Z};
use lyth_circuit::{Gate, Group};

fn main() {
    let a: Vec<String> = std::env::args().collect();
    if a.len() != 5 {
        eprintln!("usage: q2_group <q,q,...> <gates> <seed> <out.lyth>");
        std::process::exit(2);
    }
    let mut qubits: Vec<u32> = a[1].split(',').map(|x| x.parse().unwrap()).collect();
    let (g, seed): (usize, u64) = (a[2].parse().unwrap(), a[3].parse().unwrap());
    let mut rng = Rng(seed);
    let pick = |rng: &mut Rng| qubits[rng.below(qubits.len() as u64) as usize];
    let mut gates = Vec::new();
    for i in 0..g {
        if i % 3 == 2 && qubits.len() > 1 {
            let c = pick(&mut rng);
            let mut t = pick(&mut rng);
            while t == c {
                t = pick(&mut rng);
            }
            gates.push(Gate::CU(c, t, if rng.below(2) == 0 { X } else { Z }));
        } else {
            let q = pick(&mut rng);
            gates.push(Gate::U(q, zyz(rng.angle(), rng.angle(), rng.angle())));
        }
    }
    qubits.sort_unstable_by(|x, y| y.cmp(x));
    let group = Group { qubits, gates };
    group.lower("q2").expect("the generated group is valid LYTH");
    std::fs::write(&a[4], group.source("q2")).expect("write the kernel");
    let sets: Vec<String> = group.widths().iter().enumerate().map(|(l, w)| format!("w{l}={w}")).collect();
    println!("{}", sets.join(" "));
}
