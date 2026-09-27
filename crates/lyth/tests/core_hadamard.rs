//! Hadamard on one TT core, on both machines.
//!
//! The core is shape (2, 2, 4): two physical slices, eight f32 each, which is one
//! Unibit register. `hadamard.lyth` is that gate. The host oracle, the MTLB emulator
//! and PTX on sm_120 must agree bit for bit, real and imaginary.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::process::Command;

use lyth_cuda::{Arg, Context};
use lyth_lang::program::{PrintRange, Program};
use lyth_lang::{eval, ir, parse};

const N: u32 = 8;

fn repo() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn core_slices() -> BTreeMap<String, Vec<f32>> {
    let mut buffers = BTreeMap::new();
    buffers.insert(
        "p0r".into(),
        vec![0.50, 0.25, -0.50, 0.00, 0.125, -0.25, 0.75, 0.00],
    );
    buffers.insert(
        "p0i".into(),
        vec![0.00, 0.00, 0.00, 0.50, 0.00, 0.25, 0.00, -0.125],
    );
    buffers.insert(
        "p1r".into(),
        vec![0.50, -0.25, 0.50, 0.00, -0.125, 0.25, 0.25, 0.00],
    );
    buffers.insert(
        "p1i".into(),
        vec![0.00, 0.00, 0.00, -0.50, 0.00, -0.25, 0.00, 0.125],
    );
    for name in ["q0r", "q0i", "q1r", "q1i"] {
        buffers.insert(name.into(), vec![0.0; N as usize]);
    }
    buffers
}

fn kernel() -> ir::KernelIr {
    let src = std::fs::read_to_string(repo().join("examples/hadamard.lyth")).unwrap();
    let unit = parse(&src).unwrap();
    ir::lower(&unit, &unit.kernels[0]).unwrap()
}

fn host(k: &ir::KernelIr, buffers: &BTreeMap<String, Vec<f32>>) -> BTreeMap<String, Vec<f32>> {
    let mut inputs = eval::Inputs::default();
    inputs.scalars.insert("s".into(), std::f32::consts::FRAC_1_SQRT_2);
    inputs.extents.insert("n".into(), N);
    inputs.buffers.clone_from(buffers);
    let out = eval::eval(k, N as usize, &inputs).expect("host oracle");
    out.buffers
}

fn emulator(k: &ir::KernelIr, buffers: &BTreeMap<String, Vec<f32>>) -> Vec<f32> {
    let mut scalars = BTreeMap::new();
    scalars.insert("s".into(), std::f32::consts::FRAC_1_SQRT_2);
    let asm = lyth_uasm::emit(
        k,
        &Program {
            n: N,
            extents: BTreeMap::new(),
            scalars,
            prints: ["q0r", "q0i", "q1r", "q1i"]
                .into_iter()
                .map(|name| PrintRange {
                    buffer: name.into(),
                    lo: 0,
                    hi: N,
                })
                .collect(),
            buffers: buffers.clone(),
        },
    )
    .expect("emits");
    let uni = repo().join("../Unibit");
    let dir = tempfile::tempdir().unwrap();
    let src = dir.path().join("core.uasm");
    std::fs::write(&src, asm).unwrap();
    let out = Command::new("cargo")
        .args(["run", "--quiet", "--", "run", src.to_str().unwrap()])
        .current_dir(&uni)
        .output()
        .expect("emulator");
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    String::from_utf8_lossy(&out.stdout)
        .lines()
        .filter_map(|line| line.trim().parse::<f32>().ok())
        .collect()
}

fn ptx(k: &ir::KernelIr, buffers: &BTreeMap<String, Vec<f32>>) -> BTreeMap<String, Vec<f32>> {
    let module = lyth_ptx::emit(k, "sm_120").expect("ptx");
    let ctx = Context::new(0).expect("cuda device");
    let loaded = ctx.load_ptx(&module.ptx).expect("load");
    let func = loaded.function(&module.entry).expect("function");
    let mut device = Vec::new();
    for p in &k.params {
        if p.ty.is_buffer() {
            let host = &buffers[&p.name];
            let uploaded = ctx.upload(host).expect("upload");
            device.push((p.name.clone(), uploaded));
        }
    }
    let args: Vec<Arg> = k
        .params
        .iter()
        .map(|p| match p.ty {
            lyth_lang::ast::Ty::U32 => Arg::U32(N),
            lyth_lang::ast::Ty::F32 => Arg::F32(std::f32::consts::FRAC_1_SQRT_2),
            _ => Arg::Buf(
                &device
                    .iter()
                    .find(|(name, _)| name == &p.name)
                    .expect("uploaded")
                    .1,
            ),
        })
        .collect();
    func.launch(1, 32, &args).expect("launch");
    let mut got = BTreeMap::new();
    for (name, buf) in &device {
        if name.starts_with('q') {
            got.insert(name.clone(), buf.download().expect("download"));
        }
    }
    got
}

#[test]
fn a_core_hadamard_matches_on_the_host_the_emulator_and_ptx() {
    let k = kernel();
    let buffers = core_slices();
    let oracle = host(&k, &buffers);
    let emu = emulator(&k, &buffers);
    let names = ["q0r", "q0i", "q1r", "q1i"];
    assert_eq!(emu.len(), names.len() * N as usize);
    for (j, name) in names.iter().enumerate() {
        for i in 0..N as usize {
            assert_eq!(
                emu[j * N as usize + i].to_bits(),
                oracle.buffers_bits(name, i),
                "emulator {name}[{i}]"
            );
        }
    }
    let device = ptx(&k, &buffers);
    for name in names {
        for i in 0..N as usize {
            assert_eq!(
                device[name][i].to_bits(),
                oracle.buffers_bits(name, i),
                "ptx {name}[{i}]"
            );
        }
    }
}

trait Bits {
    fn buffers_bits(&self, name: &str, i: usize) -> u32;
}

impl Bits for BTreeMap<String, Vec<f32>> {
    fn buffers_bits(&self, name: &str, i: usize) -> u32 {
        self[name][i].to_bits()
    }
}
