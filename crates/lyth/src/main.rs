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

mod bind_c;
mod bind_py;
mod bind_rust;
mod manifest;
use manifest::Manifest;
use serde::Deserialize;

use lyth_cuda::{grid_for, median_and_spread, time_launches, Arg, Context, Launch};
use lyth_lang::ast::{ReduceOp, Ty};
use lyth_lang::eval::{eval_with_launch, Inputs};
use lyth_lang::{check_intensity, ir, parse, KernelIr, Ridge};
use lyth_ptx::emit;

const EXIT_REFUSED: u8 = 1;
const EXIT_UNUSABLE: u8 = 2;
/// Threads per block. 256 is the conventional starting point and is not tuned here.
const BLOCK: u32 = 256;

/// Largest grid launched by default.
///
/// The default WAS the device's SM count times four waves, on the reasoning that a grid-stride
/// loop wants enough blocks to fill the machine and no more. **The sweep says that reasoning is
/// wrong for a memory-bound kernel** (`tools/grid_sweep.py`, ADR-0012): at 36 SMs x 4 waves
/// saxpy reaches 386.63 GB/s against 399.70 at one element per thread, 3.3% slower and well
/// outside the 0.4% spread of both points. More threads in flight is more outstanding loads.
///
/// So the default is one element per thread, and the loop engages only past this cap or when
/// `--grid` asks for it. The cap exists because a grid is a u32 and a large enough `n` would
/// need more blocks than one launch can carry.
const MAX_GRID: u32 = 1 << 20;

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
        /// Also write the signature, launch facts and cost contract as a
        /// `lyth-manifest/0.1`, which is what a binding generator reads to make this kernel
        /// callable from another language (ADR-0016).
        #[arg(long, value_name = "JSON")]
        manifest: Option<PathBuf>,
        /// Also write a Rust module that loads the embedded PTX and launches the kernel with
        /// a typed signature. Written buffers take `&mut`, so the borrow checker refuses a
        /// call that aliases an input with an output (ADR-0016).
        #[arg(long, value_name = "RS")]
        bind_rust: Option<PathBuf>,
        /// Also write a single-file C header: the embedded PTX, the launch facts and the cost
        /// contract, over the CUDA driver API and nothing else (ADR-0016).
        #[arg(long, value_name = "H")]
        bind_c: Option<PathBuf>,
        /// Also write a Python module: ctypes against the CUDA driver, no dependencies, the
        /// PTX embedded. Takes device pointers, so it composes with torch and cupy (ADR-0016).
        #[arg(long, value_name = "PY")]
        bind_py: Option<PathBuf>,
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
        /// Blocks to launch. Default: one element per thread, `ceil(n / block)`, capped at
        /// `MAX_GRID`. That default is the sweep's, not an obvious choice -- see ADR-0012.
        /// Exposed so the choice can be swept and measured rather than assumed (ADR-0012).
        #[arg(long)]
        grid: Option<u32>,
        /// Threads per block. Must be a power of two: the reduction tree halves its stride,
        /// so anything else leaves elements uncombined. Exposed for the same reason as
        /// `--grid` -- ADR-0014 needed it to show that the tree's cost is per block and not
        /// per thread, and could not vary it.
        #[arg(long)]
        block: Option<u32>,
        /// Override the dynamic shared memory the launch requests.
        ///
        /// This exists for one thing: ADR-0017's bypass control. A tiled kernel launched with
        /// 0 must produce a different answer, because a kernel that is right without shared
        /// memory is a kernel that never staged. Not a tuning knob -- the correct value is
        /// derived and passing anything else is asking for undefined behaviour on purpose.
        #[arg(long, value_name = "N")]
        shared_bytes: Option<u32>,
        /// Emit the tile without its derived skew.
        ///
        /// ADR-0017's counterfactual, and its only use. `bank_conflicts = 0` on the padded
        /// kernel shows the conflicts are absent, not that the skew removed them -- that
        /// number is also zero if the kernel never staged, or if the emitter dropped the
        /// padding. This builds the kernel that must conflict, and it computes the same bits.
        #[arg(long)]
        no_skew: bool,
        /// Time the kernel over this many runs after a warm-up, and report achieved
        /// bandwidth against the machine file's measured figure. 0 disables timing.
        #[arg(long, default_value_t = 0)]
        time: u32,
        /// Also write the timing as one JSON object, for a sweep to re-analyse. Evidence
        /// that has to be recovered by parsing prose depends on nobody editing a println.
        #[arg(long)]
        json: Option<PathBuf>,
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
            manifest,
            bind_rust,
            bind_c,
            bind_py,
            elements,
            tol,
        } => cmd_build(
            &file,
            machine.as_deref(),
            out.as_deref(),
            evidence.as_deref(),
            manifest.as_deref(),
            bind_rust.as_deref(),
            bind_c.as_deref(),
            bind_py.as_deref(),
            elements,
            tol,
        ),
        Cmd::Run {
            file,
            machine,
            n,
            grid,
            block,
            shared_bytes,
            no_skew,
            time,
            json,
            sets,
            tol,
        } => cmd_run(
            &file,
            machine.as_deref(),
            RunOpts {
                n,
                grid,
                block,
                shared_bytes,
                no_skew,
                reps: time,
                json: json.as_deref(),
                sets: &sets,
                tol,
            },
        ),
    }
}

/// Everything the front end produces for one file.
struct Front {
    ir: KernelIr,
    machine: Option<Machine>,
    /// What the source declared, if anything. The derived figure is in `ir.cost`; a manifest
    /// carries both, because "the author claimed X and the compiler computed X" and "the
    /// author claimed nothing" are different states and a caller may care which.
    declared: Option<f64>,
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
    Ok(Front {
        ir,
        machine,
        declared: kernel.declared_intensity,
    })
}

fn print_cost(ir: &KernelIr, report: &lyth_lang::IntensityReport) {
    println!("kernel {} on machine {}", ir.name, ir.machine);
    println!(
        "  derived  {:.4} flop/byte  ({} flop / {} byte per element)",
        report.derived, report.flops, report.bytes
    );
    println!(
        "  payload  {} read + {} written, at {}",
        ir.cost.read_bytes_per_element(),
        ir.cost.write_bytes_per_element(),
        ir.cost.level().name()
    );
    // What the bus carries, beside what the source asked for. Printed for every rank-2 kernel
    // rather than only when they differ: at rank 2 the two numbers are a function of the
    // declared schedule, and a kernel that achieves 1.000 is saying something. At rank 1 it is
    // always 1.000 and the line would be noise.
    let coalescence = ir.cost.coalescence();
    if ir.space.is_some() {
        println!(
            "  sectors  {} read + {} written at L1->L2  (coalescence {:.3})",
            ir.cost.sector_read_per_element, ir.cost.sector_write_per_element, coalescence
        );
    }
    if let Some(l) = &ir.shared {
        let (_, rows, stride) = &l.tiles[0];
        println!(
            "  shared   {} read + {} written per element, {} B per block",
            ir.cost.at(lyth_lang::ast::Level::Smem).map(|c| c.read).unwrap_or(0.0),
            ir.cost.at(lyth_lang::ast::Level::Smem).map(|c| c.write).unwrap_or(0.0),
            l.bytes
        );
        println!(
            "           tile {rows} x {} with a derived stride of {stride}, predicting {} bank conflicts",
            l.tiles[0].2 - (stride - ir.tile.as_ref().map(|t| t[1]).unwrap_or(*stride)),
            l.predicted_bank_conflicts
        );
        println!("           the permutation is absorbed here, so no global access is strided");
    }
    if coalescence < 1.0 {
        for st in ir.streams.iter().filter(|s| !s.coalesced) {
            println!(
                "           `{}` is indexed [{}] and its innermost index is not the fast one,",
                st.buffer,
                st.index.join(", ")
            );
            println!("           so neighbouring threads land in separate 32-byte sectors");
        }
    }
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

#[allow(clippy::too_many_arguments)]
fn cmd_build(
    file: &Path,
    machine: Option<&Path>,
    out: Option<&Path>,
    evidence: Option<&Path>,
    manifest: Option<&Path>,
    bind_rust: Option<&Path>,
    bind_c: Option<&Path>,
    bind_py: Option<&Path>,
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
    let m = Manifest::of(
        &f.ir,
        f.declared,
        BLOCK,
        MAX_GRID,
        known_limits().into_iter().map(str::to_string).collect(),
    );
    if let Some(p) = bind_rust {
        let name = file
            .file_name()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|| file.display().to_string());
        let code = bind_rust::generate(&m, &module.ptx, &name);
        if let Err(e) = std::fs::write(p, code) {
            eprintln!("error: write {}: {e}", p.display());
            return ExitCode::from(EXIT_UNUSABLE);
        }
        println!("  wrote    {} (Rust binding over lyth-cuda)", p.display());
    }
    if let Some(p) = bind_c {
        let name = file
            .file_name()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|| file.display().to_string());
        let code = bind_c::generate(&m, &module.ptx, &name);
        if let Err(e) = std::fs::write(p, code) {
            eprintln!("error: write {}: {e}", p.display());
            return ExitCode::from(EXIT_UNUSABLE);
        }
        println!("  wrote    {} (C header over the driver API)", p.display());
    }
    if let Some(p) = bind_py {
        let name = file
            .file_name()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|| file.display().to_string());
        let code = bind_py::generate(&m, &module.ptx, &name);
        if let Err(e) = std::fs::write(p, code) {
            eprintln!("error: write {}: {e}", p.display());
            return ExitCode::from(EXIT_UNUSABLE);
        }
        println!("  wrote    {} (Python over ctypes)", p.display());
    }
    if let Some(p) = manifest {
        let json = match serde_json::to_string_pretty(&m) {
            Ok(j) => j,
            Err(e) => {
                eprintln!("error: the manifest did not serialise: {e}");
                return ExitCode::from(EXIT_UNUSABLE);
            }
        };
        if let Err(e) = std::fs::write(p, json + "
") {
            eprintln!("error: write {}: {e}", p.display());
            return ExitCode::from(EXIT_UNUSABLE);
        }
        println!("  wrote    {} ({}, derived)", p.display(), manifest::SCHEMA);
    }
    ExitCode::SUCCESS
}

/// The limits the compiler attaches to every kernel it emits, in one place so the evidence
/// case and the manifest cannot drift into saying different things.
fn known_limits() -> [&'static str; 2] {
    [
        "[KNOWN LIMIT] `moves` is the payload. `bus` is what crosses the L1-to-L2 interface, where a strided access costs a whole 32-byte sector; measured exact on sm_120 (ADR-0015). Neither predicts DRAM: the L2 keeps writes when the working set fits, and re-reads partially written sectors when it does not.",
        "[KNOWN LIMIT] Writes may not reach DRAM within a single launch if the working set fits in L2. Compare the read half with --ncu-dir read.",
    ]
}

/// The launch shape the compiler derives, in one place.
///
/// This exists because of a bug, and the bug is the argument for it. `report_timing` took
/// `grid` and `shared` and supplied the block itself, so a tiled kernel verified at 1024
/// threads was timed at 256 — a quarter of every tile untouched, results nothing checked, and
/// a reported bandwidth ten times what the device has.
///
/// Grouping them into `Launch` fixed that call. It did not fix the class: any later path that
/// launches — a bench harness, an example, a second report — can invent a shape again. So the
/// shape is derived once, here, from the IR, and a path that wants to launch asks rather than
/// assembles.
///
/// What the IR determines is not negotiable: a tile fixes the block at one thread per element,
/// and staging or a reduction fixes the shared bytes. `--block` may only choose what the IR
/// leaves open.
fn launch_shape(ir: &KernelIr, block_arg: Option<u32>, grid: u32) -> Launch {
    let block = match &ir.tile {
        Some(t) => t.iter().product(),
        None => block_arg.unwrap_or(BLOCK),
    };
    let shared = match (&ir.shared, ir.reduction.is_some()) {
        (Some(l), _) => l.bytes,
        (None, true) => block * 4,
        (None, false) => 0,
    };
    Launch {
        grid,
        block,
        shared,
    }
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
                "level": ir.cost.level().name(),
                "bytes": 4.0,
                "dir": "r"
            }));
        }
        if s.drain {
            moves.push(serde_json::json!({
                "name": format!("{}_write", s.buffer),
                "level": ir.cost.level().name(),
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
        // Beside the payload, never instead of it. `moves` is what the source asks for and is
        // what `intensity` is checked against; this is what the bus carries once a strided
        // access is charged a whole 32-byte sector per element. A CI job can assert on the
        // ratio without a profiler, and `--ncu` is what decides whether the bound is right.
        "bus": {
            "sector_read": ir.cost.sector_read_per_element,
            "sector_write": ir.cost.sector_write_per_element,
            "coalescence": ir.cost.coalescence(),
            "bound": "upper: a strided access is charged 32 bytes per element, which assumes a row of 8 or more. The exact figure is min(32, 4 * row) and needs the launch extents.",
            "strided": ir.streams.iter().filter(|s| !s.coalesced).map(|s| s.buffer.clone()).collect::<Vec<_>>()
        },
        "notes": [
            "GENERATED BY THE LYTH COMPILER from the kernel it emitted. The accounting and the PTX come from the same IR, so a disagreement with measured traffic is a fact about the back end rather than about someone's reading of the source.",
            "One element per thread, bounds-checked, no grid-stride loop. Every thread performs exactly the listed moves once."
        ],
        "known_limits": [
            "[KNOWN LIMIT] `moves` is the payload. `bus` is what crosses the L1-to-L2 interface, where a strided access costs a whole 32-byte sector; measured exact on sm_120 (ADR-0015). Neither predicts DRAM: the L2 keeps writes when the working set fits, and re-reads partially written sectors when it does not.",
            "[KNOWN LIMIT] Writes may not reach DRAM within a single launch if the working set fits in L2. Compare the read half with --ncu-dir read."
        ]
    });
    serde_json::to_string_pretty(&case).expect("a json object always serialises")
        + "
"
}

/// Everything `run` needs beyond the source and the machine. Grouped because they are one
/// decision about how to execute, and eight loose parameters invite transposing two of them.
struct RunOpts<'a> {
    n: u32,
    grid: Option<u32>,
    block: Option<u32>,
    shared_bytes: Option<u32>,
    no_skew: bool,
    reps: u32,
    json: Option<&'a Path>,
    sets: &'a [String],
    tol: f64,
}

fn cmd_run(file: &Path, machine: Option<&Path>, o: RunOpts) -> ExitCode {
    let RunOpts {
        n,
        grid: grid_arg,
        block: block_arg,
        shared_bytes: shared_override,
        no_skew,
        reps,
        json,
        sets,
        tol,
    } = o;

    // A launch parameter, not a language one: the source says nothing about block size, so
    // nothing here can be checked against it. The power-of-two rule is the reduction tree's --
    // it halves its stride to 1, and an odd width would drop the elements above the halving
    // point without ever combining them.
    let block = block_arg.unwrap_or(BLOCK);
    if !block.is_power_of_two() || block > 1024 {
        eprintln!("error[launch]: --block must be a power of two no greater than 1024, got {block}");
        return ExitCode::from(EXIT_UNUSABLE);
    }
    let f = match front(file, machine, tol) {
        Ok(f) => f,
        Err(code) => return code,
    };
    let ir = &f.ir;

    // A tile fixes the block: one thread per element of the tile. A `--block` beside it would
    // be two answers to one question, and the launch would honour whichever was read last.
    if let (Some(t), Some(b)) = (&ir.tile, block_arg) {
        let want: u32 = t.iter().product();
        eprintln!(
            "error[launch]: this kernel declares `tile {}`, which fixes the block at {want}. Remove `--block {b}`, or the tile.",
            t.iter().map(u32::to_string).collect::<Vec<_>>().join(", ")
        );
        return ExitCode::from(EXIT_UNUSABLE);
    }
    let block = launch_shape(ir, block_arg, 1).block;

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

    let ctx = match Context::new(0) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("error[cuda]: {e}");
            return ExitCode::from(EXIT_UNUSABLE);
        }
    };
    println!(
        "  device   {} ({} SMs, {:.0} MB L2, from the driver)",
        ctx.device_name,
        ctx.sm_count,
        ctx.l2_bytes as f64 / 1e6
    );

    // One element per thread by default, because that is what measured fastest -- not because
    // it is obvious. `--grid` takes anything, and below `want` the loop starts doing real work.
    // One block per tile when tiled, one element per thread otherwise. The same rule the
    // manifest carries and the three generators emit.
    // A reduction's default grid is not the elementwise one.
    //
    // ADR-0012 swept `saxpy` and found one element per thread faster than filling the machine,
    // and that default was then applied to every kernel. For a reduction it is pathological:
    // the block's shared-memory tree runs once per *thread*, so at one element per thread it
    // runs once per element. ADR-0014 measured the instruction cost of exactly that -- 121 per
    // element against 10 -- and concluded it bought no time, which was true of the kernel that
    // had been swept and not of this one.
    //
    // Measured on sum.lyth at n = 2^26: 213.69 GB/s at one element per thread against 419.68
    // at grid 36864. Every grid from 144 to 36864 lands between 387 and 420, so the plateau is
    // wide and only the extreme collapses. Eight elements per thread is inside it with room on
    // both sides, and is a floor rather than a tuned peak.
    const REDUCTION_MIN_ELEMENTS_PER_THREAD: u32 = 8;
    let want = match (&ir.tile, &ir.space) {
        (Some(t), Some(sp)) => sp
            .extents
            .iter()
            .zip(t)
            .map(|(e, d)| scalars.get(e).map(|v| *v as u32).unwrap_or(1).div_ceil(*d))
            .product::<u32>()
            .max(1),
        _ if ir.reduction.is_some() => {
            let one_per_thread = grid_for(n, block);
            let amortised = grid_for(n, block.saturating_mul(REDUCTION_MIN_ELEMENTS_PER_THREAD));
            one_per_thread.min(amortised).max(1)
        }
        _ => grid_for(n, block),
    };
    let grid = grid_arg.unwrap_or_else(|| want.clamp(1, MAX_GRID));
    let per_thread = (n as f64 / (grid as f64 * block as f64)).ceil() as u64;
    println!(
        "  grid     {grid} blocks of {block} on {} SMs, {per_thread} element(s) per thread",
        ctx.sm_count
    );
    // A reduction writes one value per block, so its target is sized by the grid, not by the
    // element count. Sizing it by n would work and would hide a real constraint on the caller.
    let reduce_target = ir.reduction.as_ref().map(|r| r.into.clone());

    // Deterministic inputs. A fixed generator rather than random ones so a disagreement is
    // reproducible from the command line alone, and so the same bytes are compared every run.
    let mut inputs = Inputs::default();

    // A rank-2 kernel is walked by its space, so `-n` does not describe it: the extents do,
    // and their product is the element count. They come from `--set`, like any other value
    // the caller supplies, and a missing one is refused rather than guessed at -- guessing a
    // shape would silently transpose a different matrix than the caller meant.
    let n = if let Some(sp) = &ir.space {
        let mut total: u64 = 1;
        for e in &sp.extents {
            let Some(v) = scalars.get(e).map(|v| *v as u32) else {
                eprintln!(
                    "error[launch]: `space` walks `{e}`, but no value was given for it. Pass `--set {e}=<u32>`; a rank-2 kernel has no single element count."
                );
                return ExitCode::from(EXIT_UNUSABLE);
            };
            if v == 0 {
                eprintln!("error[launch]: extent `{e}` is 0, so the kernel walks nothing");
                return ExitCode::from(EXIT_UNUSABLE);
            }
            inputs.extents.insert(e.clone(), v);
            total *= v as u64;
        }
        // The flattened index is 32-bit. See the [KNOWN LIMIT] in `lyth-ptx`: 2^32 f32
        // elements is 17.2 GB, so this is unreachable on any device with a machine file here,
        // and checking once at launch is cheaper than 64-bit index arithmetic per element.
        if total > u32::MAX as u64 {
            eprintln!(
                "error[launch]: {} elements overflows the 32-bit flattened index. That is {:.1} GB of f32 and more than any device this compiler targets.",
                total,
                total as f64 * 4.0 / 1e9
            );
            return ExitCode::from(EXIT_UNUSABLE);
        }
        total as u32
    } else {
        n
    };
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

    let expected = match eval_with_launch(ir, n as usize, &inputs, grid as usize, block as usize) {
        Ok(o) => o,
        Err(e) => {
            eprintln!("error[reference]: {e}");
            return ExitCode::from(EXIT_UNUSABLE);
        }
    };

    if no_skew {
        println!("  [NO SKEW] the tile is emitted unpadded: ADR-0017's counterfactual.");
        println!("            It computes the same bits and must conflict on every shared read.");
    }
    let module = match lyth_ptx::emit_with_skew(ir, &arch_of(&f), !no_skew) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("error[codegen]: {e}");
            return ExitCode::from(EXIT_REFUSED);
        }
    };

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
            // Each u32 carries its own value. Passing `n` for all of them was right only
            // while a kernel had exactly one; a rank-2 kernel takes `rows` and `cols` and
            // would otherwise receive the element count twice.
            Ty::U32 => Arg::U32(inputs.extents.get(&p.name).copied().unwrap_or(n)),
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

    // The static sector figure is an upper bound because it assumes a row of eight elements
    // or more. With the extents in hand the exact number is available, and so is the question
    // that decides whether DRAM will show any of it.
    if ir.cost.coalescence() < 1.0 {
        if let Some(sp) = &ir.space {
            let mut payload = 0.0f64;
            let mut sectors = 0.0f64;
            for st in &ir.streams {
                let per_payload = 4.0;
                let per_sector = if st.coalesced {
                    4.0
                } else {
                    // The stride between neighbouring threads is this buffer's own row
                    // length; a sector is eight f32, so a narrow row wastes less than eight.
                    let row = ir
                        .params
                        .iter()
                        .find(|p| p.name == st.buffer)
                        .and_then(|p| p.shape.last())
                        .and_then(|d| inputs.extents.get(d))
                        .copied()
                        .unwrap_or(8) as f64;
                    (4.0 * row).min(32.0)
                };
                if st.read {
                    payload += per_payload;
                    sectors += per_sector;
                }
                if st.drain {
                    payload += per_payload;
                    sectors += per_sector;
                }
            }
            println!(
                "  exact    {sectors} byte per element at this shape (coalescence {:.3}), against the {} byte bound",
                payload / sectors,
                ir.cost.sector_read_per_element + ir.cost.sector_write_per_element
            );
            let buffers_touched = ir.params.iter().filter(|p| p.ty == Ty::BufF32).count() as u64;
            let working = sp
                .extents
                .iter()
                .filter_map(|e| inputs.extents.get(e))
                .map(|v| *v as u64)
                .product::<u64>()
                * 4
                * buffers_touched;
            if working <= ctx.l2_bytes {
                println!(
                    "  [LIMIT]  the {:.0} MB working set fits in {:.0} MB of L2, so DRAM will",
                    working as f64 / 1e6,
                    ctx.l2_bytes as f64 / 1e6
                );
                println!("           show less than this. Raise the extents to measure it.");
            }
        }
    }

    // One f32 slot per thread for the reduction tree; nothing without a reduction.
    // A tile's shared memory is sized by the TILE, not by the problem: the same block walks
    // however many tiles the grid-stride gives it, reusing one staging buffer.
    let derived = launch_shape(ir, block_arg, grid);
    let derived_shared = derived.shared;
    let shared = shared_override.unwrap_or(derived_shared);
    if shared != derived_shared {
        println!("  [BYPASS] launching with {shared} B instead of the derived {derived_shared} B.");
        println!("           A tiled kernel that is still correct here never staged anything.");
    }
    println!("  launch   grid {grid} x block {block} over {n} elements, {shared} B shared");
    // One shape, from the compiler, used by the verified launch and by the timed one. The
    // only thing a caller may override is the shared bytes, and only for ADR-0017's bypass
    // control, which announces itself above.
    let shape = Launch { shared, ..derived };
    if let Err(e) = func.launch_shared(shape.grid, shape.block, shape.shared, &args) {
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
        let is_drain = ir.drains.iter().any(|(b, _)| b == name);
        let is_partial = Some(name) == reduce_target.as_ref();
        if !is_drain && !is_partial {
            continue;
        }
        // Only the first `grid` entries of a reduction target are written.
        let count = if is_partial {
            grid as usize
        } else {
            n as usize
        };
        let got = match buf.download() {
            Ok(v) => v,
            Err(e) => {
                eprintln!("error[cuda]: downloading {name}: {e}");
                return ExitCode::from(EXIT_UNUSABLE);
            }
        };
        let want = &expected.buffers[name];
        for i in 0..count {
            if got[i].to_bits() != want[i].to_bits() {
                mismatches += 1;
                if first.is_none() {
                    first = Some((name.clone(), i, want[i], got[i]));
                }
            }
        }
    }

    if mismatches == 0 && reps > 0 {
        if let Err(e) = report_timing(&ctx, &func, shape, &args, reps, n, ir, &f.machine, json) {
            eprintln!("error[timing]: {e}");
            return ExitCode::from(EXIT_UNUSABLE);
        }
    }

    if mismatches == 0 {
        println!("  verify   BIT-EXACT against the IR evaluated on the host, {n} elements");
        if let Some(r) = &ir.reduction {
            // Finish the reduction the way the caller has to, **with the kernel's own
            // operator**. Printed because a bit-exact match between two buffers of zeros is
            // not evidence of anything, and this is the number the kernel was written to
            // produce.
            let name = &r.into;
            let partials = &expected.buffers[name];
            let slice = &partials[..grid as usize];
            let op = r.op.name();
            let total: f64 = match r.op {
                ReduceOp::Sum => slice.iter().map(|v| *v as f64).sum(),
                ReduceOp::Max => slice.iter().fold(f64::NEG_INFINITY, |a, v| a.max(*v as f64)),
                ReduceOp::Min => slice.iter().fold(f64::INFINITY, |a, v| a.min(*v as f64)),
            };
            // Whether the order of this last step can change the answer is a property of the
            // operator, not a detail: `sum` in a different order is a different bit pattern,
            // `max` in any order is the same one. ADR-0013.
            let ordering = if r.op.is_reorderable() {
                "in any order, because the operator is associative on the bits"
            } else {
                "in index order, which is one of several answers the caller could get"
            };
            println!("           {grid} block partials in `{name}`, combined with `{op}` {ordering}");
            println!("  reduced  {total:.6e}  (the caller combines the partials; ADR-0011)");
        }
        println!("ok");
        ExitCode::SUCCESS
    } else {
        eprintln!("  verify FAILED — {mismatches} of {n} elements differ");
        if let Some((name, i, want, got)) = first {
            eprintln!(
                " first at {name}[{i}]: host {want:e} (0x{:08x}), device {got:e} (0x{:08x})",
                want.to_bits(),
                got.to_bits()
            );
        }
        eprintln!("  The back end and the IR disagree. The IR is the specification.");
        ExitCode::from(EXIT_REFUSED)
    }
}

/// Time the kernel and report achieved bandwidth beside the baseline it is measured against.
///
/// The rules are ADR-0012's, fixed before the first number was taken: N runs after a warm-up,
/// median and full spread, the baseline named, and the byte count stated as derived rather than
/// measured. Exercise 01 of rse-hpc-lab overstates its own bandwidth by 12.9% by dividing an
/// analytic working set by a time; a compiler that derives its own byte count is *more* liable
/// to that, not less, so the output says where the number came from.
#[allow(clippy::too_many_arguments)]
fn report_timing(
    ctx: &Context,
    func: &lyth_cuda::Function,
    // The whole shape, not grid and shared with the block filled in here. `Launch` exists
    // because "grid, block and shared size are one decision", and this function took two of
    // the three and supplied the constant for the other -- so a tiled kernel verified at 1024
    // threads was timed at 256, ran a quarter of each tile, produced wrong results nothing
    // checked, and reported a bandwidth above what the device has.
    shape: Launch,
    args: &[lyth_cuda::Arg],
    reps: u32,
    n: u32,
    ir: &KernelIr,
    machine: &Option<Machine>,
    json: Option<&Path>,
) -> Result<(), Box<dyn std::error::Error>> {
    // One warm-up per five runs, at least one: the first launch pays for module residency and
    // clock ramp, neither of which is the kernel's cost.
    let warmup = (reps / 5).max(1);

    let samples = time_launches(ctx, func, shape, args, warmup, reps)?;
    let (median_ms, spread) =
        median_and_spread(&samples).ok_or("a timing with no runs has no median")?;

    let bytes =
        ir.cost.bytes_per_element() * n as f64 + ir.cost.dram_bytes_per_block * shape.grid as f64;
    let gbs = bytes / (median_ms * 1e-3) / 1e9;

    println!("  time     {median_ms:.4} ms median of n={reps} (warm-up {warmup} discarded), spread {:.1}%", spread * 100.0);
    println!(
        "  moved    {:.3} MB by the compiler's derived byte model, NOT measured",
        bytes / 1e6
    );
    println!("  achieved {gbs:.2} GB/s");

    match machine.as_ref().and_then(|m| {
        m.levels
            .iter()
            .find(|l| l.name == "dram")
            .map(|l| (m.id.clone(), l.bandwidth_gbs))
    }) {
        Some((id, peak)) => {
            let pct = gbs / peak * 100.0;
            println!(
                " vs {pct:.1}% of {peak:.2} GB/s, the measured DRAM bandwidth in the {id} machine file"
            );
            // A tool that prints "above peak" and says nothing is the tool that produced
            // exercise 01's 385.71 GB/s. Above the baseline means one of three things and the
            // reader is told which to check, in the order they are worth checking.
            if pct > 100.0 {
                let working_set = ir.cost.bytes_per_element() * n as f64;
                println!("  ABOVE    this is over the baseline, which is a claim about the");
                println!("           baseline, not a result. Four things to check, in order:");
                println!("           1. the working set is {:.0} MB. If that fits in L2 the bytes", working_set / 1e6);
                println!("              never crossed the memory controller and this is an L2");
                println!("              figure wearing a DRAM label. Raise -n until it does not.");
                println!("           2. the byte count is derived, not measured. Check it with");
                println!(
                    " `lyth build --evidence` and `lyth-probe --ncu` (ADR-0009)."
                );
                println!("           3. the baseline may not describe this access pattern. The");
                println!("              {id} figure came from a torch.sum reduction, and the");
                println!("              machine file says a streaming probe would be better.");
                println!("           4. only then, that the kernel is genuinely faster.");
                // Past three times the baseline nothing else is a plausible explanation: a
                // kernel is not three times its own memory system. Say so rather than leaving
                // a reader to weigh four equal-looking possibilities.
                if pct > 300.0 {
                    println!("           At {pct:.0}% the first is not one possibility among four.");
                    println!("           A kernel cannot outrun its own memory controller.");
                }
            }
        }
        None => println!("  vs       no machine file given, so this number has no baseline"),
    }
    println!("  [LIMIT]  kernel only: allocation and the host copy are outside the events.");
    println!("           Buffers stay resident across runs; L2 state is not controlled.");
    println!("           Clocks not locked, as the baseline's were not.");

    if let Some(path) = json {
        // Every sample, not just the summary: a median whose raw runs were discarded cannot
        // be re-analysed, and a sweep that can only read summaries cannot tell a slow drift
        // from a real difference.
        let record = serde_json::json!({
            "kernel": ir.name,
            "machine": ir.machine,
            "elements": n,
            "grid": shape.grid,
            "block": shape.block,
            "reps": reps,
            "warmup": warmup,
            "ms_samples": samples,
            "ms_median": median_ms,
            "ms_spread": spread,
            "bytes_source": "derived by the compiler, not measured",
            "bytes": bytes,
            "achieved_gbs": gbs,
            "baseline_gbs": machine.as_ref().and_then(|m| {
                m.levels.iter().find(|l| l.name == "dram").map(|l| l.bandwidth_gbs)
            }),
            "clock_state": "unlocked",
            "cache_state": "buffers resident across runs; L2 not controlled",
        });
        std::fs::write(
            path,
            serde_json::to_string_pretty(&record)?
                + "
",
        )?;
    }
    Ok(())
}
