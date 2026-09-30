//! QuBLAR ghost bits, one ZIPPER2 step at a time, on both machines.
//!
//! The pytest `test_ghost_bits_gate.py` writes the fixture and sets
//! `QUBLAR_GHOST_FIXTURE`. Without that variable this test does not invent a
//! tensor. Each chain is the `.lyth` kernel: host eval, PTX, and the emulator
//! instruction. The accumulator starts at the boundary matrix [[1, 0], [0, 0]].

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::process::Command;

use lyth_cuda::{Arg, Context};
use lyth_lang::{eval, ir, parse};

fn repo() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn words_from_lanes(lanes: [u64; 4]) -> Vec<f32> {
    let mut words = Vec::with_capacity(8);
    for lane in lanes {
        words.push(f32::from_bits(lane as u32));
        words.push(f32::from_bits((lane >> 32) as u32));
    }
    words
}

fn lanes_from_words(words: &[f32]) -> [u64; 4] {
    let mut lanes = [0u64; 4];
    for (index, word) in words.iter().enumerate() {
        let lane = index >> 1;
        let half = index & 1;
        lanes[lane] |= u64::from(word.to_bits()) << (half * 32);
    }
    lanes
}

fn kernel() -> ir::KernelIr {
    let src = std::fs::read_to_string(repo().join("examples/zipper2.lyth")).unwrap();
    let unit = parse(&src).unwrap();
    ir::lower(&unit, &unit.kernels[0]).unwrap()
}

fn parse_u64(text: &str) -> u64 {
    let text = text.trim();
    let digits = text
        .strip_prefix("0x")
        .or_else(|| text.strip_prefix("0X"))
        .unwrap_or(text);
    u64::from_str_radix(digits, 16).unwrap_or_else(|err| panic!("{text}: {err}"))
}

fn load_fixture(path: &str) -> Vec<Vec<([u64; 4], [u64; 4])>> {
    let text = std::fs::read_to_string(path).unwrap();
    let mut lines = text.lines().filter(|line| !line.trim().is_empty());
    let nchains: usize = lines.next().unwrap().trim().parse().unwrap();
    let mut chains = Vec::with_capacity(nchains);
    for _ in 0..nchains {
        let nsteps: usize = lines.next().unwrap().trim().parse().unwrap();
        let mut steps = Vec::with_capacity(nsteps);
        for _ in 0..nsteps {
            let ket: Vec<u64> = lines.next().unwrap().split_whitespace().map(parse_u64).collect();
            let bra: Vec<u64> = lines.next().unwrap().split_whitespace().map(parse_u64).collect();
            assert_eq!(ket.len(), 4);
            assert_eq!(bra.len(), 4);
            steps.push((
                [ket[0], ket[1], ket[2], ket[3]],
                [bra[0], bra[1], bra[2], bra[3]],
            ));
        }
        chains.push(steps);
    }
    assert!(lines.next().is_none(), "fixture has trailing lines");
    chains
}

fn host_chains(k: &ir::KernelIr, chains: &[Vec<([u64; 4], [u64; 4])>]) -> Vec<Vec<f32>> {
    chains
        .iter()
        .map(|chain| {
            let mut acc = words_from_lanes([0x3f800000, 0, 0, 0]);
            for (ket, bra) in chain {
                let mut buffers = BTreeMap::new();
                buffers.insert("acc".into(), acc);
                buffers.insert("ket".into(), words_from_lanes(*ket));
                buffers.insert("bra".into(), words_from_lanes(*bra));
                buffers.insert("out".into(), vec![0.0; 8]);
                let mut inputs = eval::Inputs::default();
                inputs.extents.insert("n".into(), 8);
                inputs.buffers = buffers;
                acc = eval::eval(k, 8, &inputs).expect("host").buffers["out"].clone();
            }
            acc
        })
        .collect()
}

fn ptx_chains(k: &ir::KernelIr, chains: &[Vec<([u64; 4], [u64; 4])>]) -> Vec<Vec<f32>> {
    let module = lyth_ptx::emit(k, "sm_120").expect("ptx");
    let ctx = Context::new(0).expect("cuda");
    let loaded = ctx.load_ptx(&module.ptx).expect("load");
    let func = loaded.function(&module.entry).expect("entry");
    let mut outs = Vec::with_capacity(chains.len());
    for chain in chains {
        let mut acc = words_from_lanes([0x3f800000, 0, 0, 0]);
        for (ket, bra) in chain {
            let host = [
                ("acc".to_string(), acc),
                ("ket".to_string(), words_from_lanes(*ket)),
                ("bra".to_string(), words_from_lanes(*bra)),
                ("out".to_string(), vec![0.0; 8]),
            ];
            let device: Vec<_> = host
                .iter()
                .map(|(name, words)| (name.clone(), ctx.upload(words).unwrap()))
                .collect();
            let args: Vec<Arg> = k
                .params
                .iter()
                .map(|p| match p.ty {
                    lyth_lang::ast::Ty::U32 => Arg::U32(8),
                    _ => Arg::Buf(&device.iter().find(|(n, _)| n == &p.name).unwrap().1),
                })
                .collect();
            func.launch(1, 32, &args).unwrap();
            acc = device
                .iter()
                .find(|(n, _)| n == "out")
                .unwrap()
                .1
                .download()
                .unwrap();
        }
        outs.push(acc);
    }
    outs
}

fn emulator_chains(chains: &[Vec<([u64; 4], [u64; 4])>]) -> Vec<[u64; 4]> {
    let mut pool: Vec<[u64; 4]> = Vec::new();
    let mut refs: Vec<Vec<(usize, usize)>> = Vec::new();
    for chain in chains {
        let mut steps = Vec::new();
        for (ket, bra) in chain {
            let kid = intern(&mut pool, *ket);
            let bid = intern(&mut pool, *bra);
            steps.push((kid, bid));
        }
        refs.push(steps);
    }
    let mut lines = vec![
        "        .data".to_string(),
        "ebnd:   .dword 0x000000003F800000, 0, 0, 0".to_string(),
        "nl:     .asciiz \"\\n\"".to_string(),
    ];
    for (index, lanes) in pool.iter().enumerate() {
        let words = lanes
            .iter()
            .map(|lane| format!("0x{lane:016X}"))
            .collect::<Vec<_>>()
            .join(", ");
        lines.push(format!("q{index}:    .dword {words}"));
    }
    lines.push("        .text".into());
    lines.push("        .global _start".into());
    lines.push("_start:".into());
    for chain in &refs {
        lines.push("        la      t0, ebnd".into());
        lines.push("        lq      s0, 0(t0)".into());
        for (ket, bra) in chain {
            lines.push(format!("        la      t0, q{ket}"));
            lines.push("        lq      s1, 0(t0)".into());
            lines.push(format!("        la      t0, q{bra}"));
            lines.push("        lq      s2, 0(t0)".into());
            lines.push("        zipper2 s0, s1, s2".into());
        }
        // s0 is x8. The syscall takes the register index, not the value.
        lines.push("        li      a0, 8".into());
        lines.push("        li      a7, 20".into());
        lines.push("        ecall".into());
        lines.push("        la      a0, nl".into());
        lines.push("        li      a7, 6".into());
        lines.push("        ecall".into());
    }
    lines.push("        halt".into());
    let dir = tempfile::tempdir().unwrap();
    let src = dir.path().join("ghost_gate.uasm");
    std::fs::write(&src, lines.join("\n") + "\n").unwrap();
    let emu = Command::new("cargo")
        .args(["run", "--quiet", "--", "run", src.to_str().unwrap()])
        .current_dir(repo().join("../Labare"))
        .output()
        .expect("emulator");
    let stdout = String::from_utf8_lossy(&emu.stdout);
    assert!(
        emu.status.success(),
        "{}\n{}",
        String::from_utf8_lossy(&emu.stderr),
        stdout
    );
    let parsed: Vec<[u64; 4]> = stdout
        .lines()
        .filter(|line| line.contains('|'))
        .map(|line| {
            let hex: Vec<u64> = line
                .split_whitespace()
                .filter_map(|tok| {
                    let tok = tok.trim_matches(|c: char| !c.is_ascii_hexdigit() && c != 'x');
                    tok.strip_prefix("0x")
                        .and_then(|digits| u64::from_str_radix(digits, 16).ok())
                })
                .collect();
            assert_eq!(hex.len(), 4, "{line}");
            // Debug order is lane 3, lane 2, lane 1, lane 0.
            [hex[3], hex[2], hex[1], hex[0]]
        })
        .collect();
    assert_eq!(parsed.len(), chains.len(), "{stdout}");
    parsed
}

fn intern(pool: &mut Vec<[u64; 4]>, key: [u64; 4]) -> usize {
    if let Some(index) = pool.iter().position(|found| *found == key) {
        return index;
    }
    pool.push(key);
    pool.len() - 1
}

#[test]
fn ghost_bits_match_on_the_host_the_emulator_and_ptx() {
    let Ok(path) = std::env::var("QUBLAR_GHOST_FIXTURE") else {
        eprintln!("ghost gate: QUBLAR_GHOST_FIXTURE is unset, the pytest drives this test");
        return;
    };
    let chains = load_fixture(&path);
    assert!(!chains.is_empty());
    let k = kernel();
    let host = host_chains(&k, &chains);
    let ptx = ptx_chains(&k, &chains);
    for (index, (left, right)) in host.iter().zip(ptx.iter()).enumerate() {
        for lane in 0..8 {
            assert_eq!(
                left[lane].to_bits(),
                right[lane].to_bits(),
                "ptx chain {index} word {lane}"
            );
        }
    }
    let emu = emulator_chains(&chains);
    for (index, (words, lanes)) in host.iter().zip(emu.iter()).enumerate() {
        assert_eq!(lanes_from_words(words), *lanes, "emulator chain {index}");
    }
    for words in &host {
        let line = words
            .iter()
            .map(|word| format!("{:08x}", word.to_bits()))
            .collect::<Vec<_>>()
            .join(" ");
        println!("w {line}");
    }
}
