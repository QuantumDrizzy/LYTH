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
use std::process::{Command, ExitCode};

use clap::{Parser, Subcommand};

mod bind_c;
mod bind_py;
mod bind_rust;
mod manifest;
use manifest::Manifest;
use serde::Deserialize;

use lyth_cuda::{median_and_spread, time_launches, Arg, Context, Launch};
use lyth_lang::ast::{ReduceOp, Ty};
use lyth_lang::eval::{eval_with_launch, Inputs};
use lyth_lang::ir::num;
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
        /// The machine's own assembler, used when the target is not a GPU and `-o` names a
        /// `.ubo`. Defaults to `unibit` on the path. Not reimplemented here: an object format
        /// with two implementations is a format with two opinions.
        #[arg(long, value_name = "PATH")]
        assembler: Option<PathBuf>,
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

/// The conversion a narrow buffer needs on the way to the device, or `None` for f32.
///
/// `lyth_lang::half` is the single implementation and it is the one checked against
/// `cvt.rn.f16.f32` over 138 vectors (ADR-0024 decision 3). Writing a second conversion here
/// would be a second place the rounding could differ, and the difference would show up as a
/// bit-exactness failure blamed on the emitter.
fn narrow_bits(ty: Ty) -> Option<fn(f32) -> u16> {
    match ty {
        Ty::BufF16 => Some(lyth_lang::half::f32_to_f16_bits),
        Ty::BufBF16 => Some(lyth_lang::half::f32_to_bf16_bits),
        _ => None,
    }
}

/// The same conversion on the way back.
fn wide_from(ty: Ty) -> Option<fn(u16) -> f32> {
    match ty {
        Ty::BufF16 => Some(lyth_lang::half::f16_bits_to_f32),
        Ty::BufBF16 => Some(lyth_lang::half::bf16_bits_to_f32),
        _ => None,
    }
}

/// Just enough of `lyth-machine/0.1` to get the target and the ridge. The full schema lives in
/// `lyth-probe`; depending on that crate here would drag evidence bundles into the compiler.
#[derive(Debug, Deserialize)]
struct Machine {
    id: String,
    /// Which back end compiles for this machine. "ptx" when absent, which is every file
    /// written before there was a second one -- a default that names the only thing that
    /// existed is a default that cannot surprise anyone.
    #[serde(default)]
    isa: Option<String>,
    levels: Vec<MachineLevel>,
    #[serde(default)]
    peak_tflops: Option<f64>,
    /// FLOPs retired per cycle, on a machine that has no clock.
    ///
    /// `unibit` charges one cycle per instruction and names no frequency, so `peak_tflops` has
    /// nothing to be per. A machine file declares one or the other and `ridge()` refuses to
    /// guess which -- see ADR-0025.
    #[serde(default)]
    flops_per_cycle: Option<f64>,
    /// `"s"` or `"cycle"`. Defaults to seconds, which is what every machine file written
    /// before ADR-0025 means.
    #[serde(default)]
    time_unit: Option<String>,
    /// What one block may ask for. `None` when the machine file predates these fields, and
    /// the check is then skipped -- a missing limit must not become a limit of zero.
    #[serde(default)]
    max_threads_per_block: Option<u32>,
    #[serde(default)]
    max_shared_bytes_per_block: Option<u32>,
    #[serde(default)]
    max_shared_bytes_per_block_optin: Option<u32>,
}

#[derive(Debug, Deserialize)]
struct MachineLevel {
    name: String,
    bandwidth_gbs: f64,
    /// Bytes this level moves per cycle, on a machine measured in cycles. See `flops_per_cycle`.
    #[serde(default)]
    bytes_per_cycle: Option<f64>,
    /// Accesses per second, in billions, where this level is priced in accesses rather than
    /// bytes. `None` on every level that is not `smem`, and on an `smem` written before
    /// ADR-0022's probe -- in which case no shared ceiling is offered at all, because a
    /// missing rate becoming a rate of zero would make every staged kernel infinitely slow.
    #[serde(default)]
    accesses_gps: Option<f64>,
}

impl Machine {
    /// The ridge, in whatever unit this machine keeps time in.
    ///
    /// Two shapes of machine file, and the choice is the file's rather than a default:
    ///
    /// * **clocked** -- `peak_tflops` and a `dram` level with `bandwidth_gbs`. Everything
    ///   before ADR-0025.
    /// * **cycle-counted** -- `flops_per_cycle` and a `mem` level with `bytes_per_cycle`.
    ///   `unibit` has no frequency anywhere in it, so a rate per second would have to be
    ///   invented, and an invented number in a file whose first line says EVERY NUMBER HERE IS
    ///   MEASURED is worse than no number.
    ///
    /// The ratio is a flop/byte either way, so the ridge, the regime and every ceiling are
    /// unit-agnostic and only the label changes.
    fn ridge(&self) -> Option<Ridge> {
        // The deepest level a machine names: `dram` on a GPU, `mem` on a machine with no
        // cache hierarchy to have a bottom of.
        let deep = self
            .levels
            .iter()
            .find(|l| l.name == "dram" || l.name == "mem")?;
        let smem = self
            .levels
            .iter()
            .find(|l| l.name == "smem")
            .and_then(|l| l.accesses_gps);

        // A declared unit that the data contradicts is refused rather than ignored. This is
        // the same rule the language applies to `intensity`: a file may say what it is, and
        // then it has to be it. `time_unit: "cycle"` with no `bytes_per_cycle` anywhere is a
        // machine file whose author changed their mind halfway.
        if let Some(u) = self.time_unit.as_deref() {
            let cycles = self.flops_per_cycle.is_some() && deep.bytes_per_cycle.is_some();
            if (u == "cycle") != cycles {
                eprintln!(
                    "error[machine]: `{}` declares time_unit \"{u}\" but {} the per-cycle                      rates that go with it",
                    self.id,
                    if cycles { "carries" } else { "does not carry" }
                );
                return None;
            }
        }

        match (self.peak_tflops, self.flops_per_cycle, deep.bytes_per_cycle) {
            // Cycle-counted. Checked first, because a file that declares both is a file whose
            // author has not decided, and the cycle figures are the measured ones there.
            (_, Some(flops), Some(bytes)) => Some(Ridge {
                flops_per_unit: flops,
                bytes_per_unit: bytes,
                unit: "cycle",
                smem_accesses_per_unit: smem,
            }),
            (Some(tflops), _, _) => Some(Ridge {
                flops_per_unit: tflops * 1e12,
                bytes_per_unit: deep.bandwidth_gbs * 1e9,
                unit: "s",
                smem_accesses_per_unit: smem.map(|g| g * 1e9),
            }),
            _ => None,
        }
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
            assembler,
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
            assembler.as_deref(),
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
    /// Kept whole, because `main` is a property of the file rather than of the kernel: it
    /// names which kernel runs, which is a question only the file can answer.
    unit: lyth_lang::ast::Unit,
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

    // What a block may ask for, checked before anything is emitted.
    //
    // A tile fixes the block at one thread per element, so `tile 64, 64` asks for 4096 threads
    // against a cap of 1024. Until this check existed that compiled, wrote 3620 bytes of PTX,
    // and failed at launch with `CUDA_ERROR_INVALID_VALUE` -- a driver error that names no
    // argument and no reason. ADR-0018 had already argued that this cap is what stops the tile
    // growing; it argued it in prose, against a compiler that did not know the number.
    if let (Some(m), Some(tile)) = (machine.as_ref(), ir.tile.as_ref()) {
        let threads = ir.block_threads(BLOCK);
        if let Some(cap) = m.max_threads_per_block {
            if threads > cap {
                let dims = tile
                    .iter()
                    .map(|d| d.to_string())
                    .collect::<Vec<_>>()
                    .join(", ");
                let traffic = ir
                    .cost
                    .bytes_expr()
                    .unwrap_or_else(|| ir.cost.bytes_fixed().to_string());
                eprintln!(
                    "error[launch]: {}:{}: `tile {dims}` is {threads} threads per block and {} allows {cap}.",
                    file.display(),
                    kernel.span,
                    m.id,
                );
                eprintln!("  A tile puts one thread on each of its elements, so the block is the tile's area.");
                eprintln!("  The traffic it would buy is derived and real -- {traffic} bytes per element --");
                eprintln!("  but no block is that wide.");
                eprintln!("  Thread coarsening, one thread computing several outputs, is the way to a");
                eprintln!("  larger tile. It is not implemented.");
                return Err(ExitCode::from(EXIT_REFUSED));
            }
        }
        // Shared memory is the *next* limit, and saying which one is in the way is the point:
        // ADR-0018 claimed the thread count binds first, and on this machine it does, but only
        // up to a tile of 64. Past that the memory is what refuses.
        if let Some(l) = &ir.shared {
            let optin = m.max_shared_bytes_per_block_optin.unwrap_or(0);
            if optin > 0 && l.bytes > optin {
                eprintln!(
                    "error[launch]: {}:{}: this tile stages {} bytes into shared memory,",
                    file.display(),
                    kernel.span,
                    l.bytes,
                );
                eprintln!(
                    "  and {} allows {optin} even when a kernel opts in to the maximum.",
                    m.id
                );
                return Err(ExitCode::from(EXIT_REFUSED));
            }
            if let Some(default) = m.max_shared_bytes_per_block {
                if l.bytes > default && l.bytes <= optin {
                    eprintln!(
                        "note: {} bytes of shared memory is above the {default} a block gets by default on {}; the launch opts in to the {optin} maximum.",
                        l.bytes, m.id
                    );
                }
            }
        }
    }

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
    let declared = kernel.declared_intensity;
    Ok(Front {
        ir,
        unit,
        machine,
        declared,
    })
}

fn print_cost(ir: &KernelIr, report: &lyth_lang::IntensityReport) {
    println!("kernel {} on machine {}", ir.name, ir.machine);
    match (&report.flops_expr, &report.bytes_expr) {
        // A contracted kernel is never told its cost as a constant, here least of all: this
        // line is what a reader quotes. The limit is labelled as one and the exact figure is
        // the expression beside it.
        (Some(f), Some(b)) => {
            println!(
                "  derived  {:.4} flop/byte asymptotic  ({f} flop / {b} byte per element)",
                report.derived
            );
        }
        _ => println!(
            "  derived  {:.4} flop/byte  ({} flop / {} byte per element)",
            report.derived,
            report.flops.unwrap_or(0.0),
            report.bytes.unwrap_or(0.0)
        ),
    }
    let (dram_r, dram_w) = ir.cost.traffic_words(ir.cost.level());
    println!(
        "  payload  {dram_r} read + {dram_w} written, at {}",
        ir.cost.level().name()
    );
    // What the bus carries, beside what the source asked for. Printed for every rank-2 kernel
    // rather than only when they differ: at rank 2 the two numbers are a function of the
    // declared schedule, and a kernel that achieves 1.000 is saying something. At rank 1 it is
    // always 1.000 and the line would be noise.
    let coalescence = ir.cost.coalescence();
    if ir.space.is_some() || !ir.views.is_empty() {
        let word = |fixed: f64, per: f64| match &ir.cost.contracted {
            Some(c) if per > 0.0 && fixed > 0.0 => format!("{} * {} + {}", num(per), c.extent, num(fixed)),
            Some(c) if per > 0.0 => format!("{} * {}", num(per), c.extent),
            _ => num(fixed),
        };
        println!(
            "  sectors  {} read + {} written at L1->L2  (coalescence {:.3})",
            word(
                ir.cost.sector_read_per_element,
                ir.cost.sector_read_per_extent
            ),
            word(
                ir.cost.sector_write_per_element,
                ir.cost.sector_write_per_extent
            ),
            coalescence
        );
    }
    if let Some(l) = &ir.shared {
        let (_, rows, stride) = &l.tiles[0];
        let (smem_r, smem_w) = ir.cost.traffic_words(lyth_lang::ast::Level::Smem);
        println!("  shared   {smem_r} read + {smem_w} written per element, {} B per block", l.bytes);
        println!(
            "           tile {rows} x {} with a derived stride of {stride}, predicting {} bank conflicts",
            l.tiles[0].2 - (stride - ir.tile.as_ref().map(|t| t[1]).unwrap_or(*stride)),
            l.predicted_bank_conflicts
        );
        println!("           the permutation is absorbed here, so no global access is strided");
    }
    if coalescence < 1.0 {
        // One line per split, not per view: `re` into `p0r, p1r` is one declaration.
        let mut split_bases: Vec<&str> = Vec::new();
        for v in &ir.views {
            if !split_bases.contains(&v.base.as_str()) {
                split_bases.push(v.base.as_str());
                let names: Vec<&str> = ir.views.iter().filter(|w| w.base == v.base).map(|w| w.name.as_str()).collect();
                println!(
                    "           `{}` is split into {} at width `{}`: runs of {} elements with a gap of {},",
                    v.base,
                    names.join(", "),
                    v.width,
                    v.width,
                    v.width
                );
            }
        }
        if !ir.views.is_empty() {
            println!("           so this is the bound (a sector per element); the exact figure needs the launch width");
        }
        for st in ir.streams.iter().filter(|s| !s.coalesced && ir.view(&s.buffer).is_none()) {
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
        print_ceiling(report);
    }
}

/// Which level this kernel runs out of first, and what that allows (ADR-0022).
///
/// This used to be one line — the DRAM intensity as a fraction of peak FLOPS, "with bandwidth
/// saturated". ADR-0021 step 5 measured that sentence false for a tiled contraction: DRAM was
/// at **1.7%** of this device's bandwidth, and a tile moving twice the derived traffic took
/// exactly the same time. Nothing was saturated, and the level that was is not in the roofline
/// at all.
///
/// So the ceiling is now computed per level and the slowest one is named. For every kernel
/// that stages nothing there is one candidate and the answer is what it always was.
fn print_ceiling(report: &lyth_lang::check::IntensityReport) {
    let Some(binding) = report.binding else {
        // No machine, or no level whose throughput this file names.
        if let Some(f) = report.peak_flops_fraction {
            println!(
                "           at this intensity the ceiling is {:.2}% of peak FLOPS",
                f * 100.0
            );
        }
        return;
    };

    // What one unit of this level's candidate is counted in. Three answers, because a
    // roofline has three kinds of limit and calling them all bytes is how ADR-0022 ended up
    // with a shared level priced in the wrong currency.
    let unit = |c: &lyth_lang::check::LevelCeiling| match c.level {
        lyth_lang::ast::Level::Smem => "access",
        lyth_lang::ast::Level::Reg => "flop",
        _ => "byte",
    };
    match report.ceiling_flops {
        Some(flops) => {
            let of_peak = report
                .ceiling_peak_fraction
                .map(|f| format!(", {:.2}% of peak FLOPS", f * 100.0))
                .unwrap_or_default();
            println!(
                "  ceiling  {} binds — {}{of_peak}",
                binding.level.name(),
                // A machine with no clock has no TFLOP/s. `unibit` counts cycles, so its
                // ceiling is flop/cycle and saying otherwise would invent a frequency
                // (ADR-0025).
                if report.unit == "cycle" {
                    format!("{flops:.3} flop/cycle")
                } else {
                    format!("{:.2} TFLOP/s", flops / 1e12)
                },
            );
        }
        None => println!(
            "  ceiling  {} binds — the kernel retires no flops, so there is no FLOP/s to report",
            binding.level.name(),
        ),
    }

    // Every candidate, always, including the one that lost. A ceiling that reported only the
    // binder would be a number with no way to tell whether it was close.
    //
    // `per` carries the contracted extent into the unit, because for a matmul these are bytes
    // per element *per unit of k* and a line that dropped the `k` would be wrong by a factor
    // of two thousand at the sizes this is measured at.
    let per = match &report.per_extent {
        Some(k) => format!("/element/{k}"),
        None => "/element".to_string(),
    };
    for c in &report.ceilings {
        let mark = if c.level == binding.level { "*" } else { " " };
        println!(
            "         {mark} {:<5} {:>9.4} {}{per}, {}",
            c.level.name(),
            c.units,
            unit(c),
            if report.unit == "cycle" {
                format!("{:.3} cycles per 1e6 of them", c.seconds * 1e6)
            } else {
                format!("{:.3} ns per 1e6 of them", c.seconds * 1e6 * 1e9)
            },
        );
    }
    if report.ceilings.len() > 1 {
        // The binder against the **runner-up**, not against the most generous candidate.
        //
        // With two levels those were the same number. Adding compute as a third (ADR-0025)
        // made "slowest over fastest" answer a question nobody asked: on a memory-bound
        // matmul the compute candidate is the fastest of the three by a mile, and quoting
        // 10.31x says nothing about whether naming `smem` rather than `dram` mattered. The
        // runner-up is what says that: close means the choice barely matters, far means it
        // decides the answer.
        let mut times: Vec<f64> = report.ceilings.iter().map(|c| c.seconds).collect();
        times.sort_by(|a, b| a.partial_cmp(b).expect("finite"));
        let runner_up = times[times.len() - 2];
        println!(
            "           the next candidate is {:.2}x faster, so which one binds is not a detail",
            binding.seconds / runner_up
        );
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

/// `lyth build main.lyth -o main.ubo` — a program, not a kernel.
///
/// This is what ADR-0025 exists to be able to write. Everything above this line compiles a
/// kernel that something else has to launch; below it, the output is a file the machine runs
/// on its own, and the whole workflow is two commands in one language.
///
/// The assembler is invoked rather than reimplemented. Unibit's two-pass assembler is that
/// machine's toolchain, the way `ptxas` is the GPU's, and a second implementation of an object
/// format is a second thing to disagree about. When `-o` does not end in `.ubo`, or the
/// assembler is not on the path, the `.uasm` is written and the remaining command is printed —
/// a missing tool should not lose the compile.
fn build_unibit(
    file: &Path,
    f: &Front,
    out: Option<&Path>,
    assembler: Option<&Path>,
) -> ExitCode {
    let Some(main) = f.unit.main.as_ref() else {
        eprintln!(
            "error: `unibit` builds a program, and {} declares no `main`.\n  A kernel compiled \
             for a GPU is launched by a host that supplies the element count and collects the \
             result;\n  here there is no host, so the file has to say. Add:\n\n\
             main:\n      run {}(n = 4096, ...)\n      print <buffer>[0:8]",
            file.display(),
            f.ir.name
        );
        return ExitCode::from(EXIT_REFUSED);
    };
    let program = match lyth_lang::program::resolve(&f.unit, main, &f.ir) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("{}:{e}", file.display());
            return ExitCode::from(EXIT_REFUSED);
        }
    };
    let asm = match lyth_uasm::emit(&f.ir, &program) {
        Ok(a) => a,
        Err(e) => {
            eprintln!("error[codegen]: {e}");
            return ExitCode::from(EXIT_REFUSED);
        }
    };

    let Some(out) = out else {
        print!("{asm}");
        return ExitCode::SUCCESS;
    };
    let wants_object = out.extension().is_some_and(|e| e == "ubo");
    let asm_path = if wants_object {
        out.with_extension("uasm")
    } else {
        out.to_path_buf()
    };
    if let Err(e) = std::fs::write(&asm_path, &asm) {
        eprintln!("error: write {}: {e}", asm_path.display());
        return ExitCode::from(EXIT_UNUSABLE);
    }
    println!("  wrote    {} ({} bytes)", asm_path.display(), asm.len());
    if !wants_object {
        return ExitCode::SUCCESS;
    }

    let tool = assembler.map(Path::to_path_buf).unwrap_or_else(|| "unibit".into());
    let run = Command::new(&tool)
        .args(["build", &asm_path.to_string_lossy(), "-o"])
        .arg(out)
        .output();
    match run {
        Ok(o) if o.status.success() => {
            let bytes = std::fs::metadata(out).map(|m| m.len()).unwrap_or(0);
            println!("  wrote    {} ({bytes} bytes)", out.display());
            println!("  run      unibit run {}", out.display());
            ExitCode::SUCCESS
        }
        Ok(o) => {
            eprintln!(
                "error: {} could not assemble {}:\n{}{}",
                tool.display(),
                asm_path.display(),
                String::from_utf8_lossy(&o.stdout),
                String::from_utf8_lossy(&o.stderr)
            );
            ExitCode::from(EXIT_UNUSABLE)
        }
        Err(e) => {
            // Not a failure of the compile. The assembly is on disk and correct; what is
            // missing is the machine's own toolchain, so say which command finishes the job.
            println!(
                "  note     `{}` is not on the path ({e}), so no object was written.\n  \
                 Finish with:  unibit build {} -o {}",
                tool.display(),
                asm_path.display(),
                out.display()
            );
            ExitCode::SUCCESS
        }
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
    assembler: Option<&Path>,
    tol: f64,
) -> ExitCode {
    let f = match front(file, machine, tol) {
        Ok(f) => f,
        Err(code) => return code,
    };
    // Which back end runs is the **machine file's** answer, not a flag. It is the same rule
    // ADR-0001 applied to the ridge: the machine is a value the source names, so everything
    // that follows from it follows from the file rather than from the invocation.
    if f.machine.as_ref().and_then(|m| m.isa.as_deref()) == Some("unibit") {
        return build_unibit(file, &f, out, assembler);
    }
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
    // C and Python do not yet emit the pair-walking grid or the `2 * w | n` launch check
    // (ADR-0028 step 4). Emitting without them would launch twice the blocks and never refuse a
    // width that does not divide, which writes past the buffer -- so they refuse to generate.
    if !m.splits.is_empty() && (bind_c.is_some() || bind_py.is_some()) {
        eprintln!(
            "error[bind]: kernel `{}` splits, and only the Rust binding emits the launch check for a split (ADR-0028).",
            f.ir.name
        );
        eprintln!("  The C and Python generators would launch twice the blocks and never refuse a width that does");
        eprintln!("  not divide the buffer, which writes past its end. Use --bind-rust for this kernel.");
        return ExitCode::from(EXIT_REFUSED);
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
    let block = ir.block_threads(block_arg.unwrap_or(BLOCK));
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
            "flops": ir.cost.flops_per_element().expect("evidence is emitted for a compiled kernel"),
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

    // A kernel that splits walks PAIRS (ADR-0028): `-n` is the length of its buffers, the block
    // width comes from `--set`, and the launch is refused with the arithmetic unless `2 * w`
    // divides every base. Refused here, not launched: a width that does not divide reaches past
    // the end of the buffer.
    let mut split_extents: BTreeMap<String, u32> = BTreeMap::new();
    let split_walked: Option<u32> = if ir.views.is_empty() {
        None
    } else {
        for e in crate::manifest::extent_params(ir) {
            split_extents.insert(e, n);
        }
        for v in &ir.views {
            let Some(w) = scalars.get(&v.width).copied() else {
                eprintln!(
                    "error[launch]: `split` at width `{0}`, but no value was given for it. Pass `--set {0}=<u32>`.",
                    v.width
                );
                return ExitCode::from(EXIT_UNUSABLE);
            };
            if w < 0.0 || w.fract() != 0.0 || w > u32::MAX as f32 {
                eprintln!("error[launch]: width `{}` must be a whole number of elements, got {w}", v.width);
                return ExitCode::from(EXIT_UNUSABLE);
            }
            split_extents.insert(v.width.clone(), w as u32);
        }
        match ir.split_pairs(&split_extents) {
            Ok(pairs) => pairs,
            Err(e) => {
                eprintln!("error[launch]: {e}");
                return ExitCode::from(EXIT_REFUSED);
            }
        }
    };

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
    // The rule lives in `manifest`, evaluated by the same code the manifest publishes, so the
    // generated bindings cannot drift from what `lyth run` does. They already had once: the
    // reduction grid was fixed here and the manifest went on emitting the elementwise rule,
    // which a caller in another repository then launched at half the achievable bandwidth.
    let rule = crate::manifest::GridRule::of(ir, crate::manifest::extent_params(ir));
    // The fallback is the element count, not 1.
    //
    // At rank 2 every extent arrives through `--set` and this map has it. At rank 1 the single
    // extent is `-n`, which is a different flag and is not in `scalars` -- so a fallback of 1
    // made `blocks()` see a product of 1 and launch **one block**. Every rank-1 timing between
    // that refactor and this line ran at grid 1 with 262,144 elements per thread: saxpy
    // reported 9.2 GB/s where it reaches 400.
    //
    // Same shape as the defect the dogfood found this morning, one layer up. The rule was
    // moved into one place correctly; the caller then fed it something the old code never fed
    // it, and no test looked at what `lyth run` actually launched.
    let want = rule.blocks(
        &|e| scalars.get(e).map(|v| *v as u32).unwrap_or(n),
        block,
        MAX_GRID,
    );
    let grid = grid_arg.unwrap_or_else(|| want.clamp(1, MAX_GRID));
    let per_thread = (split_walked.unwrap_or(n) as f64 / (grid as f64 * block as f64)).ceil() as u64;
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
    inputs.extents.extend(split_extents.clone());

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
    // What the kernel walks: pairs for a split (rank 1, so `n` above is still `-n`), else `n`.
    let walked = split_walked.unwrap_or(n);
    // The contracted extent is not one the space walks, so the loop above never sees it. It is
    // still a length the caller has to supply: `k` decides how much work each output costs and
    // there is nothing in the buffers to infer it from.
    if let Some(c) = &ir.contract {
        let Some(v) = scalars.get(&c.extent).map(|v| *v as u32) else {
            eprintln!(
                "error[launch]: `contract` runs over `{}`, but no value was given for it. Pass `--set {}=<u32>`.",
                c.extent, c.extent
            );
            return ExitCode::from(EXIT_UNUSABLE);
        };
        if v == 0 {
            eprintln!(
                "error[launch]: contracted extent `{}` is 0, so every output is the operator's identity",
                c.extent
            );
            return ExitCode::from(EXIT_UNUSABLE);
        }
        inputs.extents.insert(c.extent.clone(), v);
    }
    for p in &ir.params {
        match p.ty {
            Ty::F32 => {
                let v = *scalars.get(&p.name).unwrap_or(&2.0);
                inputs.scalars.insert(p.name.clone(), v);
            }
            // Input generation is the same for every width: the host holds f32, and a narrow
            // buffer's values are rounded to its width before upload so that host and device
            // start from the same numbers (ADR-0024 step 2). Generating here in f32 and
            // rounding at the boundary keeps one generator rather than three.
            Ty::BufF32 | Ty::BufF16 | Ty::BufBF16 => {
                // Sized by the buffer's **own** shape, not by the element count. They
                // coincide for every kernel written before a contraction: `a` is `m x k` and
                // `c` is `m x n`, and sizing both by `m * n` is over-allocation when `k < n`
                // and a buffer overrun when it is not.
                let len = if p.shape.is_empty() || p.shape.iter().any(|d| d == lyth_lang::ast::BLOCKS)
                {
                    n
                } else {
                    p.shape
                        .iter()
                        .map(|d| inputs.extents.get(d).copied().unwrap_or(n) as u64)
                        .product::<u64>()
                        .min(u32::MAX as u64) as u32
                };
                // `lyth_lang::inputs` is the one definition, because ADR-0025's second
                // backend bakes the same numbers into a `.uasm` `.data` section and the two
                // must agree. A generator copied into both is a generator that will drift,
                // which is ADR-0019's lesson with the launch shape.
                let data = lyth_lang::inputs::buffer(&p.name, p.ty, len);
                inputs.buffers.insert(p.name.clone(), data);
            }
            Ty::U32 => {}
        }
    }

    let expected = match eval_with_launch(ir, walked as usize, &inputs, grid as usize, block as usize) {
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
    //
    // A narrow buffer is converted **here**, on the way to the device, and the host's copy is
    // rounded to match (ADR-0024). Both halves matter. Uploading f32 bytes into a `[f16; n]`
    // allocation would have the kernel read pairs of halves out of single floats -- which runs,
    // and returns numbers. And leaving the host's inputs unrounded would make every comparison
    // a comparison against values the device never saw.
    let mut buffers = Vec::new();
    for p in &ir.params {
        if p.ty.is_buffer() {
            let host = &inputs.buffers[&p.name];
            let narrow = narrow_bits(p.ty);
            let uploaded = match narrow {
                Some(to_bits) => {
                    let bits: Vec<u16> = host.iter().map(|&v| to_bits(v)).collect();
                    ctx.upload_u16(&bits)
                }
                None => ctx.upload(host),
            };
            match uploaded {
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
            Ty::BufF32 | Ty::BufF16 | Ty::BufBF16 => Arg::Buf(
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
            let buffers_touched = ir.params.iter().filter(|p| p.ty.is_buffer()).count() as u64;
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

    // The same question for a split: the static figure is the bound, a sector per element, and
    // the exact one is derived by walking the address rule at this width (ADR-0028 step 3).
    if !ir.views.is_empty() {
        if let Ok(Some(sec)) = ir.split_sector(&split_extents) {
            println!(
                "  exact    {} byte per pair at this width (coalescence {:.3}), against the {} byte bound",
                sec.sectors,
                sec.coalescence(),
                ir.cost.sector_read_per_element + ir.cost.sector_write_per_element
            );
            println!("           each view taken alone; the partner view and the drain share these sectors,");
            println!("           and whether the L1 serves them is what a measurement decides, not this line.");
        }
        let buffers_touched = ir.params.iter().filter(|p| p.ty.is_buffer()).count() as u64;
        let working = u64::from(n) * 4 * buffers_touched;
        if working <= ctx.l2_bytes {
            println!(
                "  [LIMIT]  the {:.0} MB working set fits in {:.0} MB of L2, so DRAM will",
                working as f64 / 1e6,
                ctx.l2_bytes as f64 / 1e6
            );
            println!("           show less than this. Raise -n to measure it.");
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
    if ir.views.is_empty() {
        println!("  launch   grid {grid} x block {block} over {n} elements, {shared} B shared");
    } else {
        println!("  launch   grid {grid} x block {block} over {walked} pairs of {n} elements, {shared} B shared");
    }
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
        let is_drain = ir
            .drains
            .iter()
            .any(|(b, _)| b == name || ir.view(b).is_some_and(|v| v.base == *name));
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
        let ty = ir
            .params
            .iter()
            .find(|p| p.name == *name)
            .expect("a verified buffer is a parameter")
            .ty;
        let got = match wide_from(ty) {
            Some(from_bits) => match buf.download_u16() {
                Ok(v) => v.into_iter().map(from_bits).collect::<Vec<f32>>(),
                Err(e) => {
                    eprintln!("error[cuda]: downloading {name}: {e}");
                    return ExitCode::from(EXIT_UNUSABLE);
                }
            },
            None => match buf.download() {
                Ok(v) => v,
                Err(e) => {
                    eprintln!("error[cuda]: downloading {name}: {e}");
                    return ExitCode::from(EXIT_UNUSABLE);
                }
            },
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
        if let Err(e) = report_timing(&ctx, &func, shape, &args, reps, walked, ir, &f.machine, json) {
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
        ir.cost.bytes_fixed() * n as f64 + ir.cost.dram_bytes_per_block * shape.grid as f64;
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
                " vs {pct:.1}% of {peak:.2} GB/s, the fastest streaming rate measured on this device ({id})"
            );
            // **The baseline is a reference kernel, not a hardware bound**, and the wording
            // above says so now. `tools/peak_probe.py` times the best torch streaming kernel
            // it can, which is a number a tight specialised kernel can beat by a little. The
            // device's theoretical ceiling is a different and higher figure.
            //
            // That distinction is why this guard has two levels. It used to fire its whole
            // four-item list at anything over 100%, which was right when the baseline was
            // 10% too low and 112% meant the probe was broken. With the probe fixed, `sum`
            // comes out at 100.4% because it is 0.4% faster than `torch.sum`, and printing
            // four diagnostic lines at that is how a warning gets trained out of a reader.
            //
            // Nothing is hidden: a small excess still prints a line saying what it is.
            const REFERENCE_HEADROOM: f64 = 110.0;
            if pct > 100.0 && pct <= REFERENCE_HEADROOM {
                println!("  ABOVE    over the reference by {:.1}%. The baseline is the fastest", pct - 100.0);
                println!("           streaming kernel the probe could time, not a bound, so a");
                println!("           few percent means this kernel beat it. Past {REFERENCE_HEADROOM:.0}% it does");
                println!("           not mean that, and this line says so instead.");
            }
            // A tool that prints "above peak" and says nothing is the tool that produced
            // exercise 01's 385.71 GB/s. Well above the baseline means one of four things and
            // the reader is told which to check, in the order they are worth checking.
            if pct > REFERENCE_HEADROOM {
                let working_set = ir.cost.bytes_fixed() * n as f64;
                println!("  ABOVE    this is well over the baseline, which is a claim about the");
                println!("           baseline, not a result. Four things to check, in order:");
                println!("           1. the working set is {:.0} MB. If that fits in L2 the bytes", working_set / 1e6);
                println!("              never crossed the memory controller and this is an L2");
                println!("              figure wearing a DRAM label. Raise -n until it does not.");
                println!("           2. the byte count is derived, not measured. Check it with");
                println!(
                    " `lyth build --evidence` and `lyth-probe --ncu` (ADR-0009)."
                );
                println!("           3. the baseline may be measuring something else. It once");
                println!("              timed a host round trip as memory traffic and read 10%");
                println!("              low, which is how a kernel came to report 112%.");
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
        let mut record = serde_json::json!({
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
        if !ir.views.is_empty() {
            // `elements` is what the kernel walked: pairs, for a split (ADR-0028).
            record["walks"] = serde_json::json!("pairs");
        }
        std::fs::write(
            path,
            serde_json::to_string_pretty(&record)?
                + "
",
        )?;
    }
    Ok(())
}
