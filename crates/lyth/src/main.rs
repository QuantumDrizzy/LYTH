//! `lyth` — compile and run a `.lyth` source file.
//!
//!   lyth check <file>   parse, lower, derive the cost, check the declaration. No GPU.
//!   lyth build <file>   emit PTX.
//!   lyth run   <file>   compile, launch, and verify against the IR evaluated on the host.
//!
//! Exit codes match `lyth-probe`: 0 it holds, 1 the compiler refuses it, 2 the input could not
//! be read.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use clap::{Parser, Subcommand};
use serde::Deserialize;

use lyth_cuda::{grid_for, Arg, Context};
use lyth_lang::ast::Ty;
use lyth_lang::eval::{eval, Inputs};
use lyth_lang::{check_intensity, ir, parse, KernelIr, Ridge};
use lyth_ptx::emit;

const EXIT_REFUSED: u8 = 1;
const EXIT_UNUSABLE: u8 = 2;
/// Threads per block. 256 is the conventional starting point; v1 does not tune occupancy and
/// makes no performance claim, so this is a constant rather than a flag pretending otherwise.
const BLOCK: u32 = 256;

#[derive(Parser, Debug)]
#[command(
    name = "lyth",
    about = "Compile and run a .lyth kernel",
    long_about = "Memory-first: movement is declared, arithmetic happens at the stops. The \
                  intensity written in the source is checked against the one derived from the \
                  body — a mismatch does not compile."
)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand, Debug)]
enum Cmd {
    /// Parse, lower and check. Needs no GPU.
    Check {
        file: PathBuf,
        /// Machine file (lyth-machine/0.1). Without it there is no ridge to report.
        #[arg(long)]
        machine: Option<PathBuf>,
        #[arg(long, default_value_t = 0.05)]
        tol: f64,
    },
    /// Emit PTX.
    Build {
        file: PathBuf,
        #[arg(long)]
        machine: Option<PathBuf>,
        /// Write here instead of standard output.
        #[arg(short, long)]
        out: Option<PathBuf>,
        /// Also write the derived accounting as a `lyth-intensity/0.1` case, so
        /// `lyth-probe intensity-check --ncu` can check it against measured traffic.
        #[arg(long, value_name = "JSON")]
        evidence: Option<PathBuf>,
        /// Elements the profiled launch will process, recorded in the evidence.
        #[arg(long, default_value_t = 1 << 20)]
        elements: u32,
        #[arg(long, default_value_t = 0.05)]
        tol: f64,
    },
    /// Compile, launch on the GPU, and check the result against the host reference.
    Run {
        file: PathBuf,
        #[arg(long)]
        machine: Option<PathBuf>,
        /// Elements to run.
        #[arg(short, long, default_value_t = 1 << 20)]
        n: u32,
        /// Value for each `f32` parameter, as `name=value`. Repeatable.
        #[arg(long = "set", value_name = "NAME=VALUE")]
        sets: Vec<String>,
        #[arg(long, default_value_t = 0.05)]
        tol: f64,
    },
}

/// Just enough of `lyth-machine/0.1` to get the target and the ridge. The full schema lives in
/// `lyth-probe`; depending on that crate here would drag evidence bundles into the compiler.
#[derive(Debug, Deserialize)]
struct Machine {
    id: String,
    levels: Vec<MachineLevel>,
    #[serde(default)]
    peak_tflops: Option<f64>,
}

#[derive(Debug, Deserialize)]
struct MachineLevel {
    name: String,
    bandwidth_gbs: f64,
}

impl Machine {
    fn ridge(&self) -> Option<Ridge> {
        let dram = self.levels.iter().find(|l| l.name == "dram")?;
        Some(Ridge {
            peak_tflops: self.peak_tflops?,
            bandwidth_gbs: dram.bandwidth_gbs,
        })
    }
}

fn main() -> ExitCode {
    match Cli::parse().cmd {
        Cmd::Check { file, machine, tol } => cmd_check(&file, machine.as_deref(), tol),
        Cmd::Build {
            file,
            machine,
            out,
            evidence,
            elements,
            tol,
        } => cmd_build(
            &file,
            machine.as_deref(),
            out.as_deref(),
            evidence.as_deref(),
            elements,
            tol,
        ),
        Cmd::Run {
            file,
            machine,
            n,
            sets,
            tol,
        } => cmd_run(&file, machine.as_deref(), n, &sets, tol),
    }
}

/// Everything the front end produces for one file.
struct Front {
    ir: KernelIr,
    machine: Option<Machine>,
}

fn front(file: &Path, machine_path: Option<&Path>, tol: f64) -> Result<Front, ExitCode> {
    let src = std::fs::read_to_string(file).map_err(|e| {
        eprintln!("error: read {}: {e}", file.display());
        ExitCode::from(EXIT_UNUSABLE)
    })?;
    let unit = parse(&src).map_err(|e| {
        eprintln!("{}:{e}", file.display());
        ExitCode::from(EXIT_REFUSED)
    })?;
    if unit.kernels.len() > 1 {
        eprintln!(
            "error: v1 compiles one kernel per file; this one has {}",
            unit.kernels.len()
        );
        return Err(ExitCode::from(EXIT_UNUSABLE));
    }
    let kernel = &unit.kernels[0];
    let ir = ir::lower(&unit, kernel).map_err(|e| {
        eprintln!("{}:{e}", file.display());
        ExitCode::from(EXIT_REFUSED)
    })?;

    let machine: Option<Machine> = match machine_path {
        None => None,
        Some(p) => {
            let raw = std::fs::read_to_string(p).map_err(|e| {
                eprintln!("error: read {}: {e}", p.display());
                ExitCode::from(EXIT_UNUSABLE)
            })?;
            let m: Machine = serde_json::from_str(&raw).map_err(|e| {
                eprintln!("error: {} is not a machine file: {e}", p.display());
                ExitCode::from(EXIT_UNUSABLE)
            })?;
            // The source names a machine; a file describing a different one would silently
            // check the kernel against the wrong ridge.
            if m.id != ir.machine {
                eprintln!(
                    "error: the source declares `machine {}` but {} describes `{}`",
                    ir.machine,
                    p.display(),
                    m.id
                );
                return Err(ExitCode::from(EXIT_UNUSABLE));
            }
            Some(m)
        }
    };

    let report = check_intensity(
        &ir,
        kernel.declared_intensity,
        machine.as_ref().and_then(Machine::ridge),
        tol,
    )
    .map_err(|e| {
        eprintln!(
            "error[intensity]: {}:{}: {}",
            file.display(),
            kernel.span,
            e
        );
        ExitCode::from(EXIT_REFUSED)
    })?;
    print_cost(&ir, &report);
    Ok(Front { ir, machine })
}

fn print_cost(ir: &KernelIr, report: &lyth_lang::IntensityReport) {
    println!("kernel {} on machine {}", ir.name, ir.machine);
    println!(
        "  derived  {:.4} flop/byte  ({} flop / {} byte per element)",
        report.derived, report.flops, report.bytes
    );
    println!(
        "  traffic  {} read + {} written, at {}",
        ir.cost.read_bytes_per_element,
        ir.cost.write_bytes_per_element,
        ir.cost.level.name()
    );
    match report.declared {
        Some(d) => println!("  declared {d} — matches"),
        None => println!("  declared nothing — the source is silent about its cost"),
    }
    if let (Some(r), Some(regime)) = (report.ridge, report.regime) {
        println!("  ridge    {r:.1} flop/byte — {}", regime.name());
        if let Some(f) = report.peak_flops_fraction {
            println!(
                "           at this intensity the ceiling is {:.2}% of peak FLOPS, \
                 with bandwidth saturated",
                f * 100.0
            );
        }
    }
}

fn cmd_check(file: &Path, machine: Option<&Path>, tol: f64) -> ExitCode {
    match front(file, machine, tol) {
        Ok(_) => {
            println!("ok");
            ExitCode::SUCCESS
        }
        Err(code) => code,
    }
}

fn arch_of(f: &Front) -> String {
    f.machine
        .as_ref()
        .map(|m| m.id.clone())
        .unwrap_or_else(|| f.ir.machine.clone())
}

fn cmd_build(
    file: &Path,
    machine: Option<&Path>,
    out: Option<&Path>,
    evidence: Option<&Path>,
    elements: u32,
    tol: f64,
) -> ExitCode {
    let f = match front(file, machine, tol) {
        Ok(f) => f,
        Err(code) => return code,
    };
    let module = match emit(&f.ir, &arch_of(&f)) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("error[codegen]: {e}");
            return ExitCode::from(EXIT_REFUSED);
        }
    };
    match out {
        None => print!("{}", module.ptx),
        Some(p) => {
            if let Err(e) = std::fs::write(p, &module.ptx) {
                eprintln!("error: write {}: {e}", p.display());
                return ExitCode::from(EXIT_UNUSABLE);
            }
            println!("  wrote    {} ({} bytes)", p.display(), module.ptx.len());
        }
    }
    if let Some(p) = evidence {
        let json = evidence_json(&f.ir, elements);
        if let Err(e) = std::fs::write(p, json) {
            eprintln!("error: write {}: {e}", p.display());
            return ExitCode::from(EXIT_UNUSABLE);
        }
        println!("  wrote    {} (lyth-intensity/0.1, derived)", p.display());
    }
    ExitCode::SUCCESS
}

/// The accounting the compiler derived, in the schema `lyth-probe` checks against `ncu`.
///
/// This is the loop closing. `lyth-probe intensity-check --ncu` previously compared a
/// hand-written accounting against measurement, which left the accounting itself unverified
/// prose. Here it comes out of the compiler that generated the kernel, so **no human writes
/// the byte model and no human can fudge it**.
fn evidence_json(ir: &KernelIr, elements: u32) -> String {
    let mut moves = Vec::new();
    for s in &ir.streams {
        if s.read {
            moves.push(serde_json::json!({
                "name": format!("{}_read", s.buffer),
                "level": ir.cost.level.name(),
                "bytes": 4.0,
                "dir": "r"
            }));
        }
        if s.drain {
            moves.push(serde_json::json!({
                "name": format!("{}_write", s.buffer),
                "level": ir.cost.level.name(),
                "bytes": 4.0,
                "dir": "w"
            }));
        }
    }
    let case = serde_json::json!({
        "schema": "lyth-intensity/0.1",
        "kernel": ir.name,
        "declared_intensity": ir.cost.intensity,
        "unit": "flops_per_byte",
        "machine_id": ir.machine,
        "elements": elements,
        "body": {
            "moves": moves,
            "flops": ir.cost.flops_per_element,
            "flop_note": "Derived from the LYTH IR, not written by hand. An fma is two flops."
        },
        "notes": [
            "GENERATED BY THE LYTH COMPILER from the kernel it emitted. The accounting and the PTX come from the same IR, so a disagreement with measured traffic is a fact about the back end rather than about someone's reading of the source.",
            "One element per thread, bounds-checked, no grid-stride loop. Every thread performs exactly the listed moves once."
        ],
        "known_limits": [
            "[KNOWN LIMIT] The byte count is a lower bound: it counts the payload, not the              32-byte sector a scattered access pulls. v1 is elementwise and fully coalesced, so the two should agree here; that is the claim --ncu tests.",
            "[KNOWN LIMIT] Writes may not reach DRAM within a single launch if the working set fits in L2. Compare the read half with --ncu-dir read."
        ]
    });
    serde_json::to_string_pretty(&case).expect("a json object always serialises")
        + "
"
}

fn cmd_run(file: &Path, machine: Option<&Path>, n: u32, sets: &[String], tol: f64) -> ExitCode {
    let f = match front(file, machine, tol) {
        Ok(f) => f,
        Err(code) => return code,
    };
    let ir = &f.ir;

    let mut scalars: BTreeMap<String, f32> = BTreeMap::new();
    for s in sets {
        let Some((name, value)) = s.split_once('=') else {
            eprintln!("error: --set expects NAME=VALUE, got `{s}`");
            return ExitCode::from(EXIT_UNUSABLE);
        };
        match value.parse::<f32>() {
            Ok(v) => {
                scalars.insert(name.to_string(), v);
            }
            Err(e) => {
                eprintln!("error: --set {name}: `{value}` is not a number: {e}");
                return ExitCode::from(EXIT_UNUSABLE);
            }
        }
    }

    // Deterministic inputs. A fixed generator rather than random ones so a disagreement is
    // reproducible from the command line alone, and so the same bytes are compared every run.
    let mut inputs = Inputs::default();
    for p in &ir.params {
        match p.ty {
            Ty::F32 => {
                let v = *scalars.get(&p.name).unwrap_or(&2.0);
                inputs.scalars.insert(p.name.clone(), v);
            }
            Ty::BufF32 => {
                let seed = p
                    .name
                    .bytes()
                    .fold(1u32, |a, b| a.wrapping_mul(31).wrapping_add(b as u32));
                let data: Vec<f32> = (0..n)
                    .map(|i| {
                        let h = (i.wrapping_mul(2654435761).wrapping_add(seed)) >> 8;
                        (h % 2003) as f32 / 251.0 - 4.0
                    })
                    .collect();
                inputs.buffers.insert(p.name.clone(), data);
            }
            Ty::U32 => {}
        }
    }

    let expected = match eval(ir, n as usize, &inputs) {
        Ok(o) => o,
        Err(e) => {
            eprintln!("error[reference]: {e}");
            return ExitCode::from(EXIT_UNUSABLE);
        }
    };

    let module = match emit(ir, &arch_of(&f)) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("error[codegen]: {e}");
            return ExitCode::from(EXIT_REFUSED);
        }
    };

    let ctx = match Context::new(0) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("error[cuda]: {e}");
            return ExitCode::from(EXIT_UNUSABLE);
        }
    };
    println!("  device   {}", ctx.device_name);

    let loaded = match ctx.load_ptx(&module.ptx) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("error[ptx]: {e}");
            return ExitCode::from(EXIT_REFUSED);
        }
    };
    let func = match loaded.function(&module.entry) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("error[cuda]: {e}");
            return ExitCode::from(EXIT_UNUSABLE);
        }
    };

    // Upload in parameter order and keep the buffers alive until after the launch.
    let mut buffers = Vec::new();
    for p in &ir.params {
        if p.ty == Ty::BufF32 {
            let host = &inputs.buffers[&p.name];
            match ctx.upload(host) {
                Ok(b) => buffers.push((p.name.clone(), b)),
                Err(e) => {
                    eprintln!("error[cuda]: uploading {}: {e}", p.name);
                    return ExitCode::from(EXIT_UNUSABLE);
                }
            }
        }
    }
    let args: Vec<Arg> = ir
        .params
        .iter()
        .map(|p| match p.ty {
            Ty::U32 => Arg::U32(n),
            Ty::F32 => Arg::F32(inputs.scalars[&p.name]),
            Ty::BufF32 => Arg::Buf(
                &buffers
                    .iter()
                    .find(|(name, _)| *name == p.name)
                    .expect("every buffer parameter was uploaded")
                    .1,
            ),
        })
        .collect();

    let grid = grid_for(n, BLOCK);
    println!("  launch grid {grid} x block {BLOCK} over {n} elements");
    if let Err(e) = func.launch(grid, BLOCK, &args) {
        eprintln!("error[cuda]: {e}");
        return ExitCode::from(EXIT_UNUSABLE);
    }
    if let Err(e) = ctx.synchronize() {
        eprintln!("error[cuda]: the launch failed: {e}");
        return ExitCode::from(EXIT_UNUSABLE);
    }

    // Compare bit-exactly. A tolerance here would hide the code-generation bugs this check
    // exists to catch: every operation emitted is the IEEE one the host reference performs.
    let mut mismatches = 0usize;
    let mut first: Option<(String, usize, f32, f32)> = None;
    for (name, buf) in &buffers {
        if !ir.drains.iter().any(|(b, _)| b == name) {
            continue;
        }
        let got = match buf.download() {
            Ok(v) => v,
            Err(e) => {
                eprintln!("error[cuda]: downloading {name}: {e}");
                return ExitCode::from(EXIT_UNUSABLE);
            }
        };
        let want = &expected.buffers[name];
        for i in 0..(n as usize) {
            if got[i].to_bits() != want[i].to_bits() {
                mismatches += 1;
                if first.is_none() {
                    first = Some((name.clone(), i, want[i], got[i]));
                }
            }
        }
    }

    if mismatches == 0 {
        println!("  verify BIT-EXACT against the IR evaluated on the host, {n} elements");
        println!("ok");
        ExitCode::SUCCESS
    } else {
        eprintln!("  verify FAILED — {mismatches} of {n} elements differ");
        if let Some((name, i, want, got)) = first {
            eprintln!(
                "    first at {name}[{i}]: host {want:e} (0x{:08x}), device {got:e} (0x{:08x})",
                want.to_bits(),
                got.to_bits()
            );
        }
        eprintln!("  The back end and the IR disagree. The IR is the specification.");
        ExitCode::from(EXIT_REFUSED)
    }
}
