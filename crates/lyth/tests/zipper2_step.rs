//! One ZIPPER2 step, host against the emulator and against PTX.
//!
//! The buffers are eight f32 lanes: one 256-bit register. The accumulator is
//! the boundary matrix [[1, 0], [0, 0]]. Ket and bra are the same packed core.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::process::Command;

use lyth_cuda::{Arg, Context};
use lyth_lang::program::{PrintRange, Program};
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

fn buffers() -> BTreeMap<String, Vec<f32>> {
    let mut b = BTreeMap::new();
    b.insert("acc".into(), words_from_lanes([0x3f800000, 0, 0, 0]));
    let core = words_from_lanes([
        0x007F00000000007F,
        0,
        0x3C0102043C010204,
        0,
    ]);
    b.insert("ket".into(), core.clone());
    b.insert("bra".into(), core);
    b.insert("out".into(), vec![0.0; 8]);
    b
}

fn kernel() -> ir::KernelIr {
    let src = std::fs::read_to_string(repo().join("examples/zipper2.lyth")).unwrap();
    let unit = parse(&src).unwrap();
    ir::lower(&unit, &unit.kernels[0]).unwrap()
}

#[test]
fn zipper2_matches_on_the_host_the_emulator_and_ptx() {
    let k = kernel();
    let buffers = buffers();
    let mut inputs = eval::Inputs::default();
    inputs.extents.insert("n".into(), 8);
    inputs.buffers.clone_from(&buffers);
    let host = eval::eval(&k, 8, &inputs).expect("host");
    let expect = host.buffers["out"].clone();

    let asm = lyth_uasm::emit(
        &k,
        &Program {
            n: 8,
            extents: BTreeMap::new(),
            scalars: BTreeMap::new(),
            prints: vec![PrintRange {
                buffer: "out".into(),
                lo: 0,
                hi: 8,
            }],
            buffers: buffers.clone(),
        },
    )
    .expect("uasm");
    let dir = tempfile::tempdir().unwrap();
    let src = dir.path().join("zipper2.uasm");
    std::fs::write(&src, asm).unwrap();
    let emu = Command::new("cargo")
        .args(["run", "--quiet", "--", "run", src.to_str().unwrap()])
        .current_dir(repo().join("../Unibit"))
        .output()
        .expect("emulator");
    assert!(emu.status.success(), "{}", String::from_utf8_lossy(&emu.stderr));
    let got: Vec<f32> = String::from_utf8_lossy(&emu.stdout)
        .lines()
        .filter_map(|line| line.trim().parse::<f32>().ok())
        .collect();
    assert_eq!(got.len(), 8, "{}", String::from_utf8_lossy(&emu.stdout));
    for i in 0..8 {
        assert_eq!(got[i].to_bits(), expect[i].to_bits(), "emulator [{i}]");
    }

    let module = lyth_ptx::emit(&k, "sm_120").expect("ptx");
    let ctx = Context::new(0).expect("cuda");
    let loaded = ctx.load_ptx(&module.ptx).expect("load");
    let func = loaded.function(&module.entry).expect("entry");
    let mut device = Vec::new();
    for p in &k.params {
        if p.ty.is_buffer() {
            device.push((p.name.clone(), ctx.upload(&buffers[&p.name]).unwrap()));
        }
    }
    let args: Vec<Arg> = k
        .params
        .iter()
        .map(|p| match p.ty {
            lyth_lang::ast::Ty::U32 => Arg::U32(8),
            _ => Arg::Buf(&device.iter().find(|(n, _)| n == &p.name).unwrap().1),
        })
        .collect();
    func.launch(1, 32, &args).unwrap();
    let out = device
        .iter()
        .find(|(n, _)| n == "out")
        .unwrap()
        .1
        .download()
        .unwrap();
    for i in 0..8 {
        assert_eq!(out[i].to_bits(), expect[i].to_bits(), "ptx [{i}]");
    }
}

/// GHZ-4 cores from `blaze.compress(..., max_rank=2)`, packed as ZIPPER2.
/// Four sites, same kernel each time, the output register fed back as `acc`.
const GHZ4: [[u64; 4]; 4] = [
    [0x007F00000000007F, 0, 0x3C0102043C010204, 0],
    [0x000000000000007F, 0x007F000000000000, 0x3C0102043C010204, 0],
    [0x000000000000007F, 0x0081000000000000, 0x3C0102043C010204, 0],
    [0x000000000000007F, 0x0000008100000000, 0x3F8000003BB671D7, 0],
];

fn host_step(k: &ir::KernelIr, acc: &[f32], core: &[f32]) -> Vec<f32> {
    let mut buffers = BTreeMap::new();
    buffers.insert("acc".into(), acc.to_vec());
    buffers.insert("ket".into(), core.to_vec());
    buffers.insert("bra".into(), core.to_vec());
    buffers.insert("out".into(), vec![0.0; 8]);
    let mut inputs = eval::Inputs::default();
    inputs.extents.insert("n".into(), 8);
    inputs.buffers = buffers;
    eval::eval(k, 8, &inputs).expect("host").buffers["out"].clone()
}

fn ptx_chain(k: &ir::KernelIr, cores: &[Vec<f32>]) -> Vec<f32> {
    let module = lyth_ptx::emit(k, "sm_120").expect("ptx");
    let ctx = Context::new(0).expect("cuda");
    let loaded = ctx.load_ptx(&module.ptx).expect("load");
    let func = loaded.function(&module.entry).expect("entry");
    let mut acc = words_from_lanes([0x3f800000, 0, 0, 0]);
    for core in cores {
        let host = [
            ("acc", acc.clone()),
            ("ket", core.clone()),
            ("bra", core.clone()),
            ("out", vec![0.0; 8]),
        ];
        let device: Vec<_> = host
            .iter()
            .map(|(name, words)| (name.to_string(), ctx.upload(words).unwrap()))
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
    acc
}

fn emulator_chain(cores: &[[u64; 4]]) -> Vec<f32> {
    let mut lines = vec![
        "        .data".to_string(),
        "ebnd:   .dword 0x000000003F800000, 0, 0, 0".to_string(),
        "nl:     .asciiz \"\\n\"".to_string(),
    ];
    for (index, lanes) in cores.iter().enumerate() {
        let words = lanes
            .iter()
            .map(|lane| format!("0x{lane:016X}"))
            .collect::<Vec<_>>()
            .join(", ");
        lines.push(format!("c{index}:    .dword {words}"));
    }
    lines.push("        .text".into());
    lines.push("        .global _start".into());
    lines.push("_start:".into());
    lines.push("        la      t0, ebnd".into());
    lines.push("        lq      s0, 0(t0)".into());
    for index in 0..cores.len() {
        lines.push(format!("        la      t0, c{index}"));
        lines.push("        lq      s1, 0(t0)".into());
        lines.push("        zipper2 s0, s1, s1".into());
    }
    lines.extend([
        "        mv      t0, s0".to_string(),
        "        li      t1, 0xFFFFFFFF".to_string(),
        "        and     s5, t0, t1".to_string(),
        "        srli    s6, t0, 32".to_string(),
        "        and     s6, s6, t1".to_string(),
        "        mv      a0, s5".to_string(),
        "        li      a7, 7".to_string(),
        "        ecall".to_string(),
        "        la      a0, nl".to_string(),
        "        li      a7, 6".to_string(),
        "        ecall".to_string(),
        "        mv      a0, s6".to_string(),
        "        li      a7, 7".to_string(),
        "        ecall".to_string(),
        "        halt".to_string(),
    ]);
    let dir = tempfile::tempdir().unwrap();
    let src = dir.path().join("ghz_chain.uasm");
    std::fs::write(&src, lines.join("\n") + "\n").unwrap();
    let emu = Command::new("cargo")
        .args(["run", "--quiet", "--", "run", src.to_str().unwrap()])
        .current_dir(repo().join("../Unibit"))
        .output()
        .expect("emulator");
    assert!(
        emu.status.success(),
        "{}",
        String::from_utf8_lossy(&emu.stderr)
    );
    String::from_utf8_lossy(&emu.stdout)
        .lines()
        .filter_map(|line| line.trim().parse::<f32>().ok())
        .collect()
}

#[test]
fn ghz4_chain_matches_on_the_host_the_emulator_and_ptx() {
    let k = kernel();
    let cores: Vec<Vec<f32>> = GHZ4.iter().copied().map(words_from_lanes).collect();
    let mut acc = words_from_lanes([0x3f800000, 0, 0, 0]);
    for core in &cores {
        acc = host_step(&k, &acc, core);
    }
    // Overlap of the packed GHZ: E[0][0] = 1 - 2^-24, imag 0. Not a tolerance.
    assert_eq!(acc[0].to_bits(), 0x3F7FFFFF);
    assert_eq!(acc[1].to_bits(), 0);

    let ptx = ptx_chain(&k, &cores);
    for i in 0..8 {
        assert_eq!(ptx[i].to_bits(), acc[i].to_bits(), "ptx [{i}]");
    }

    let emu = emulator_chain(&GHZ4);
    assert!(emu.len() >= 2, "{emu:?}");
    let n = emu.len();
    assert_eq!(emu[n - 2].to_bits(), acc[0].to_bits(), "emulator re");
    assert_eq!(emu[n - 1].to_bits(), acc[1].to_bits(), "emulator im");
}
