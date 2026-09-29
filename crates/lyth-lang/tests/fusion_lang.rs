//! ADR-0030 step 1: splits to depth 5, and locals that a later statement reads.
//!
//! Every leaf of a depth-`k` split over qubits `S` must hold exactly the amplitudes whose bits at
//! `S` spell the leaf's name, in index order -- checked against bit insertion, which shares no code
//! with `base_index`.

use std::collections::BTreeMap;

use lyth_lang::ir::{self, KernelIr};
use lyth_lang::{eval, parse};

fn lower(src: &str) -> Result<KernelIr, String> {
    let u = parse(src).map_err(|e| e.to_string())?;
    ir::lower(&u, &u.kernels[0]).map_err(|e| e.to_string())
}

/// A copy kernel `re -> qr` through a depth-`k` split: every leaf of `re` drained into the
/// matching leaf of `qr`. Widths `w0 .. w{k-1}`, outermost first.
fn deep_copy(k: usize) -> String {
    let ws: Vec<String> = (0..k).map(|l| format!("w{l}")).collect();
    let mut src = format!(
        "machine sm_120\nkernel deep(n: u32, {}, re: [f32; n], qr: [f32; n])\n    intensity 0.0\n\n",
        ws.iter().map(|w| format!("{w}: u32")).collect::<Vec<_>>().join(", ")
    );
    for (buf, pre) in [("re", "p"), ("qr", "q")] {
        // Level l splits every node of level l; a node's name is the prefix plus its bits.
        let mut level = vec![(buf.to_string(), String::new())];
        for (l, w) in ws.iter().enumerate() {
            let mut next = Vec::new();
            for (name, bits) in &level {
                let (a, b) = (format!("{pre}{bits}0"), format!("{pre}{bits}1"));
                let (a, b) = if l + 1 == k { (a, b) } else { (format!("{a}_"), format!("{b}_")) };
                src += &format!("    split {name} into {a}, {b} : blocks {w}\n");
                next.push((a, format!("{bits}0")));
                next.push((b, format!("{bits}1")));
            }
            level = next;
        }
    }
    let leaves: Vec<String> = (0..1usize << k).map(|i| format!("{i:0k$b}")).collect();
    for l in &leaves {
        src += &format!("    stream p{l} : dram -> reg\n");
    }
    for l in &leaves {
        src += &format!("    stream q{l} : dram -> reg, drain\n");
    }
    src += "\n    at reg:\n";
    for l in &leaves {
        src += &format!("        q{l} = p{l}\n");
    }
    src
}

/// `k` with zeros inserted at the (sorted) bit positions `at`.
fn insert_zeros(mut k: u64, at: &[u32]) -> u64 {
    for &b in at {
        let low = k & ((1 << b) - 1);
        k = ((k >> b) << (b + 1)) | low;
    }
    k
}

#[test]
fn leaves_at_depth_three_to_five_hold_exactly_the_amplitudes_their_bits_name() {
    let q = 8u32; // 256 amplitudes
    let n = 1u32 << q;
    let mut checked = 0;
    // Qubit sets, outermost level first; the widths follow ADR-0029's rule, one level at a time:
    // each inner width counts elements of a parent that has lost the bits of every level above.
    for set in [vec![7u32, 3, 0], vec![0, 1, 2], vec![2, 6, 4, 0], vec![5, 1, 7, 3, 6], vec![0, 1, 2, 3, 4]] {
        let k = deep_copy(set.len());
        let kir = lower(&k).expect("a depth-k copy lowers");
        assert_eq!(kir.walk_depth() as usize, set.len());
        let mut ex = BTreeMap::from([("n".to_string(), n)]);
        for (l, &b) in set.iter().enumerate() {
            let above = set[..l].iter().filter(|&&a| a < b).count() as u32;
            ex.insert(format!("w{l}"), 1 << (b - above));
        }
        assert_eq!(kir.split_pairs(&ex).unwrap(), Some(n >> set.len()));
        let mut sorted = set.clone();
        sorted.sort_unstable();
        let mut seen = vec![false; n as usize];
        for leaf in 0..1u64 << set.len() {
            let name = format!("p{leaf:0w$b}", w = set.len());
            // Leaf bit l (from the left) is the bit of qubit set[l].
            let fixed: u64 = set.iter().enumerate().map(|(l, &b)| ((leaf >> (set.len() - 1 - l)) & 1) << b).sum();
            for kk in 0..u64::from(n >> set.len()) {
                let got = kir.base_index(&name, &ex, kk).unwrap();
                assert_eq!(got, insert_zeros(kk, &sorted) | fixed, "{set:?} leaf {name} k = {kk}");
                assert!(!seen[got as usize]);
                seen[got as usize] = true;
            }
        }
        assert!(seen.iter().all(|s| *s), "{set:?}: the leaves cover the buffer");
        checked += 1;
    }
    assert_eq!(checked, 5);
}

#[test]
fn a_depth_five_copy_runs_on_the_host_oracle() {
    let kir = lower(&deep_copy(5)).unwrap();
    let n = 256usize;
    let mut inputs = eval::Inputs::default();
    inputs.extents.insert("n".into(), n as u32);
    for (l, w) in [16u32, 8, 4, 2, 1].into_iter().enumerate() {
        inputs.extents.insert(format!("w{l}"), w);
    }
    let x: Vec<f32> = (0..n).map(|i| i as f32).collect();
    inputs.buffers.insert("re".into(), x.clone());
    inputs.buffers.insert("qr".into(), vec![f32::NAN; n]);
    let out = eval::eval(&kir, n / 32, &inputs).unwrap();
    assert_eq!(out.buffers["qr"], x, "a copy through 32 leaves is the identity");
}

const CHAIN: &str = "machine sm_120\nkernel chain(n: u32, a: f32, x: [f32; n], y: [f32; n])\n    intensity 0.25\n    stream x : dram -> reg\n    stream y : dram -> reg, drain\n    at reg:\n";

#[test]
fn a_local_a_later_statement_reads_is_legal_and_computes() {
    let k = lower(&format!("{CHAIN}        t = a * x + x\n        u = t * t\n        y = u + t\n")).expect("chained locals lower");
    let mut inputs = eval::Inputs::default();
    inputs.scalars.insert("a".into(), 3.0);
    inputs.buffers.insert("x".into(), vec![1.0, 2.0, -0.5]);
    inputs.buffers.insert("y".into(), vec![0.0; 3]);
    let out = eval::eval(&k, 3, &inputs).unwrap();
    let want: Vec<f32> = [1.0f32, 2.0, -0.5].iter().map(|&x| { let t = 3.0f32.mul_add(x, x); t * t + t }).collect();
    assert_eq!(out.buffers["y"], want);
}

#[test]
fn a_local_nothing_reads_is_still_refused() {
    let err = lower(&format!("{CHAIN}        t = a * x\n        y = x\n")).unwrap_err();
    assert!(err.contains("`t` is computed and then discarded"), "{err}");
    // A local read only by itself before it is redefined is consumed; one redefined after its last
    // read and never read again is dead.
    let err = lower(&format!("{CHAIN}        t = a * x\n        y = t\n        t = x * x\n")).unwrap_err();
    assert!(err.contains("`t` is computed"), "{err}");
}
