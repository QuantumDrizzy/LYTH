//! Hadamard on TT cores, one `.lyth` source, both machines.
//!
//! The pytest `test_walsh_machines.py` writes `WALSH_FIXTURE`. Each physical
//! slice is padded to eight f32 lanes. The kernel is elementwise, so every core
//! sits in the same launch. Without the variable this test does not invent cores.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::process::Command;

use lyth_cuda::{Arg, Context};
use lyth_lang::program::{PrintRange, Program};
use lyth_lang::{eval, ir, parse};

fn repo() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn kernel() -> ir::KernelIr {
    let src = std::fs::read_to_string(repo().join("examples/hadamard.lyth")).unwrap();
    let unit = parse(&src).unwrap();
    ir::lower(&unit, &unit.kernels[0]).unwrap()
}

fn parse_u32(text: &str) -> u32 {
    let digits = text
        .trim()
        .strip_prefix("0x")
        .or_else(|| text.trim().strip_prefix("0X"))
        .unwrap_or(text.trim());
    u32::from_str_radix(digits, 16).unwrap_or_else(|err| panic!("{text}: {err}"))
}

fn load_fixture(path: &str) -> BTreeMap<String, Vec<f32>> {
    let text = std::fs::read_to_string(path).unwrap();
    let mut lines = text.lines().filter(|line| !line.trim().is_empty());
    let n: usize = lines.next().unwrap().trim().parse().unwrap();
    assert!(n.is_multiple_of(8) && n > 0, "n = {n}");
    let mut buffers = BTreeMap::new();
    for name in ["p0r", "p0i", "p1r", "p1i"] {
        let words: Vec<f32> = lines
            .next()
            .unwrap()
            .split_whitespace()
            .map(|tok| f32::from_bits(parse_u32(tok)))
            .collect();
        assert_eq!(words.len(), n, "{name}");
        buffers.insert(name.to_string(), words);
    }
    assert!(lines.next().is_none(), "fixture has trailing lines");
    for name in ["q0r", "q0i", "q1r", "q1i"] {
        buffers.insert(name.to_string(), vec![0.0; n]);
    }
    buffers
}

fn host(k: &ir::KernelIr, buffers: &BTreeMap<String, Vec<f32>>, n: usize) -> BTreeMap<String, Vec<f32>> {
    let mut inputs = eval::Inputs::default();
    inputs.scalars.insert("s".into(), std::f32::consts::FRAC_1_SQRT_2);
    inputs.extents.insert("n".into(), u32::try_from(n).unwrap());
    inputs.buffers.clone_from(buffers);
    eval::eval(k, n, &inputs).expect("host").buffers
}

fn ptx(k: &ir::KernelIr, buffers: &BTreeMap<String, Vec<f32>>, n: u32) -> BTreeMap<String, Vec<f32>> {
    let module = lyth_ptx::emit(k, "sm_120").expect("ptx");
    let ctx = Context::new(0).expect("cuda");
    let loaded = ctx.load_ptx(&module.ptx).expect("load");
    let func = loaded.function(&module.entry).expect("entry");
    let device: Vec<_> = k
        .params
        .iter()
        .filter(|p| p.ty.is_buffer())
        .map(|p| (p.name.clone(), ctx.upload(&buffers[&p.name]).unwrap()))
        .collect();
    let args: Vec<Arg> = k
        .params
        .iter()
        .map(|p| match p.ty {
            lyth_lang::ast::Ty::U32 => Arg::U32(n),
            lyth_lang::ast::Ty::F32 => Arg::F32(std::f32::consts::FRAC_1_SQRT_2),
            _ => Arg::Buf(&device.iter().find(|(name, _)| name == &p.name).unwrap().1),
        })
        .collect();
    func.launch(1, 32, &args).unwrap();
    let mut got = BTreeMap::new();
    for (name, buf) in &device {
        if name.starts_with('q') {
            got.insert(name.clone(), buf.download().unwrap());
        }
    }
    got
}

fn emulator(k: &ir::KernelIr, buffers: &BTreeMap<String, Vec<f32>>, n: u32) -> Vec<f32> {
    let mut scalars = BTreeMap::new();
    scalars.insert("s".into(), std::f32::consts::FRAC_1_SQRT_2);
    let asm = lyth_uasm::emit(
        k,
        &Program {
            n,
            extents: BTreeMap::new(),
            scalars,
            prints: ["q0r", "q0i", "q1r", "q1i"]
                .into_iter()
                .map(|name| PrintRange {
                    buffer: name.into(),
                    lo: 0,
                    hi: n,
                })
                .collect(),
            buffers: buffers.clone(),
        },
    )
    .expect("uasm");
    let dir = tempfile::tempdir().unwrap();
    let src = dir.path().join("walsh.uasm");
    std::fs::write(&src, asm).unwrap();
    let emu = Command::new("cargo")
        .args(["run", "--quiet", "--", "run", src.to_str().unwrap()])
        .current_dir(repo().join("../Unibit"))
        .output()
        .expect("emulator");
    let stdout = String::from_utf8_lossy(&emu.stdout);
    assert!(
        emu.status.success(),
        "{}\n{stdout}",
        String::from_utf8_lossy(&emu.stderr)
    );
    stdout
        .lines()
        .filter_map(|line| line.trim().parse::<f32>().ok())
        .collect()
}

#[test]
fn walsh_cores_match_on_the_host_the_emulator_and_ptx() {
    let Ok(path) = std::env::var("WALSH_FIXTURE") else {
        eprintln!("walsh gate: WALSH_FIXTURE is unset, the pytest drives this test");
        return;
    };
    let buffers = load_fixture(&path);
    let n = buffers["p0r"].len();
    let k = kernel();
    let host = host(&k, &buffers, n);
    let device = ptx(&k, &buffers, n as u32);
    let emu = emulator(&k, &buffers, n as u32);
    let names = ["q0r", "q0i", "q1r", "q1i"];
    assert_eq!(emu.len(), names.len() * n, "emulator printed {}", emu.len());
    for (block, name) in names.iter().copied().enumerate() {
        for i in 0..n {
            assert_eq!(
                device[name][i].to_bits(),
                host[name][i].to_bits(),
                "ptx {name}[{i}]"
            );
            assert_eq!(
                emu[block * n + i].to_bits(),
                host[name][i].to_bits(),
                "emulator {name}[{i}]"
            );
        }
        let line = host[name]
            .iter()
            .map(|word| format!("{:08x}", word.to_bits()))
            .collect::<Vec<_>>()
            .join(" ");
        println!("w {name} {line}");
    }
}
