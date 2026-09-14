//! lyth-probe — Fase 0 CLI. No parser. No kernels.
//!
//! Exit codes are part of the interface:
//!   0  the claim holds
//!   1  the evidence refuses it — a result, not a crash
//!   2  the input could not be read or compared; nothing was decided

mod cli;

use std::path::PathBuf;
use std::process::ExitCode;

use clap::{Parser, Subcommand};

use cli::checks::TrafficOpts;

#[derive(Parser, Debug)]
#[command(
    name = "lyth-probe",
    about = "Evidence that refuses to lie — Fase 0 of LYTH (no language yet)",
    long_about = "A claim without a baseline exits 1. Clock/cache unknown without a known_limit exits 1. verified=true with open limits exits 1. See docs/ADR-0001-thesis.md."
)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

/// Default relative tolerance for every declared-vs-computed comparison.
const DEFAULT_TOL: f64 = 0.05;

#[derive(Subcommand, Debug)]
enum Cmd {
    /// Write a scaffold bundle (verified=false, open clock/cache limits).
    New {
        out: PathBuf,
        #[arg(long, default_value = "TODO: one sentence, with its baseline")]
        claim: String,
    },
    /// Validate a lyth-evidence/0.1 bundle. Exit 1 on any violation.
    Validate { path: PathBuf },
    /// Content-address a JSON file (prep for lyth probe anchor).
    Hash { path: PathBuf },
    /// Report what a foreign evidence file is missing for lyth-evidence/0.1.
    Gap { path: PathBuf },
    /// Rank-order Unibit oracle vs silicon scores (Fase 0.5). Exit 0/1/2.
    OracleCheck { path: PathBuf },
    /// Compare a machine file to a live measurement. Exit 0/1/2.
    MachineCheck {
        machine: PathBuf,
        measurement: PathBuf,
        #[arg(long, default_value_t = DEFAULT_TOL)]
        tol: f64,
    },
    /// Declared intensity vs body FLOPs/byte. Optional --machine names the ridge.
    IntensityCheck {
        path: PathBuf,
        #[arg(long)]
        machine: Option<PathBuf>,
        #[arg(long, default_value_t = DEFAULT_TOL)]
        tol: f64,
        /// `ncu --csv --metrics dram__bytes.sum` output. Checks the hand-written byte
        /// accounting against measured traffic instead of only against itself.
        #[arg(long, value_name = "CSV")]
        ncu: Option<PathBuf>,
        /// Which kernel in the report (substring). Omit only for a single-kernel report.
        #[arg(long, value_name = "SUBSTR")]
        ncu_kernel: Option<String>,
        /// Elements the profiled launch processed. Overrides `elements` in the case file.
        #[arg(long)]
        elements: Option<f64>,
        /// Check the accounting at this level instead of the one its moves declare.
        /// `dram` = dram__bytes.sum, `l2` = lts__t_bytes.sum. Use it to ask whether a
        /// byte model that misses at DRAM is right one level up.
        #[arg(long, value_name = "dram|l2")]
        ncu_level: Option<String>,
        /// Take the element count from this metric in the same ncu report instead of a
        /// hand-typed number. For a data-dependent kernel this is the only honest source:
        /// its element count changes with its input and cannot be recovered afterwards.
        #[arg(long, value_name = "METRIC")]
        elements_from: Option<String>,
        /// Compare only one half of the traffic: `all` (default), `read` or `write`.
        /// A write-back cache can retire every store into L2 and evict none before the
        /// kernel ends, so `read` is the honest comparison for a single launch whose
        /// working set fits in cache.
        #[arg(long, value_name = "all|read|write", default_value = "all")]
        ncu_dir: String,
        /// Tolerance for measured/analytic. Separate from --tol: that one bounds
        /// arithmetic error, this one bounds how far a model of silicon may sit from
        /// the silicon.
        #[arg(long, default_value_t = DEFAULT_TOL)]
        ncu_tol: f64,
    },
    /// Memory-first kernel IR: streams + ops + capability refuse.
    KernelCheck {
        path: PathBuf,
        #[arg(long)]
        machine: PathBuf,
        #[arg(long, default_value_t = DEFAULT_TOL)]
        tol: f64,
    },
    /// Constant-time taint: refuse secret-dependent addr/branch.
    CtCheck { path: PathBuf },
    /// Layout × machine polymorphism — cost re-checked per instance.
    PolyCheck {
        path: PathBuf,
        #[arg(long)]
        machine: PathBuf,
        #[arg(long, default_value_t = DEFAULT_TOL)]
        tol: f64,
    },
    /// Run a lyth-suite/0.1 checklist (gates that must not be forgotten).
    Suite { path: PathBuf },
    /// Print schema id and mandatory field list.
    Schema,
}

fn main() -> ExitCode {
    match Cli::parse().cmd {
        Cmd::New { out, claim } => cli::evidence::new(&out, &claim),
        Cmd::Validate { path } => cli::evidence::validate_bundle(&path),
        Cmd::Hash { path } => cli::evidence::hash(&path),
        Cmd::Gap { path } => cli::evidence::gap(&path),
        Cmd::Schema => cli::evidence::schema(),

        Cmd::OracleCheck { path } => cli::checks::oracle(&path),
        Cmd::MachineCheck {
            machine,
            measurement,
            tol,
        } => cli::checks::machine(&machine, &measurement, tol),
        Cmd::IntensityCheck {
            path,
            machine,
            tol,
            ncu,
            ncu_kernel,
            elements,
            elements_from,
            ncu_level,
            ncu_dir,
            ncu_tol,
        } => match lyth_probe::Direction::parse(&ncu_dir) {
            None => {
                eprintln!("error: --ncu-dir must be all, read or write; got `{ncu_dir}`");
                ExitCode::from(2)
            }
            Some(dir) => cli::checks::intensity(
                &path,
                machine.as_deref(),
                tol,
                &TrafficOpts {
                    report: ncu.as_deref(),
                    kernel: ncu_kernel.as_deref(),
                    elements,
                    elements_from: elements_from.as_deref(),
                    level: ncu_level.as_deref(),
                    dir,
                    tol: ncu_tol,
                },
            ),
        },
        Cmd::KernelCheck { path, machine, tol } => cli::checks::kernel(&path, &machine, tol),
        Cmd::CtCheck { path } => cli::checks::constant_time(&path),
        Cmd::PolyCheck { path, machine, tol } => cli::checks::poly(&path, &machine, tol),

        Cmd::Suite { path } => cli::suite_run::run(&path),
    }
}
