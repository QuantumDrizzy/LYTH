//! lith-probe — Fase 0 CLI. No parser. No kernels.

use std::fs;
use std::path::PathBuf;
use std::process::ExitCode;

use clap::{Parser, Subcommand};
use lith_probe::{
    content_hash, ct_check, format_ct_verdict, format_intensity_verdict, format_kernel_verdict,
    format_machine_verdict, format_oracle_verdict, format_poly_verdict, format_report, gap_report,
    intensity_check_with_machine, kernel_check, machine_check, match_expect, oracle_check,
    poly_check, suite_check_schema, validate, Bundle, CtCase, CtVerdict, Expect, IntensityCase,
    IntensityVerdict, KernelIr, KernelVerdict, Machine, MachineMeasurement, MachineVerdict,
    OracleCase, OracleVerdict, PolyCase, PolyVerdict, Step, StepOutcome, Suite, SCHEMA_ID,
};

#[derive(Parser, Debug)]
#[command(
    name = "lith-probe",
    about = "Evidence that refuses to lie — Fase 0 of LITH (no language yet)",
    long_about = "A claim without a baseline exits 1. Clock/cache unknown without a known_limit exits 1. verified=true with open limits exits 1. See docs/ADR-0001-thesis.md."
)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand, Debug)]
enum Cmd {
    /// Write a scaffold bundle (verified=false, open clock/cache limits).
    New {
        #[arg(long, default_value = "evidence/bundle.json")]
        out: PathBuf,
        #[arg(long, default_value = "REPLACE_ME — one falsifiable sentence")]
        claim: String,
    },
    /// Validate a lith-evidence/0.1 bundle. Exit 1 on any violation.
    Validate { path: PathBuf },
    /// Content-address a JSON file (prep for lith probe anchor).
    Hash { path: PathBuf },
    /// Report what a foreign evidence file is missing for lith-evidence/0.1.
    Gap { path: PathBuf },
    /// Rank-order Unibit oracle vs silicon scores (Fase 0.5). Exit 0/1/2.
    OracleCheck { path: PathBuf },
    /// Compare a machine file to a live measurement. Exit 0/1/2.
    MachineCheck {
        #[arg(long)]
        machine: PathBuf,
        #[arg(long)]
        measurement: PathBuf,
        #[arg(long, default_value_t = 0.05)]
        tol: f64,
    },
    /// Declared intensity vs body FLOPs/byte. Optional --machine names the ridge.
    IntensityCheck {
        path: PathBuf,
        #[arg(long)]
        machine: Option<PathBuf>,
        #[arg(long, default_value_t = 0.05)]
        tol: f64,
    },
    /// Memory-first kernel IR: streams + ops + capability refuse.
    KernelCheck {
        path: PathBuf,
        #[arg(long)]
        machine: PathBuf,
        #[arg(long, default_value_t = 0.05)]
        tol: f64,
    },
    /// Constant-time taint: refuse secret-dependent addr/branch.
    CtCheck { path: PathBuf },
    /// Layout × machine polymorphism — cost re-checked per instance.
    PolyCheck {
        path: PathBuf,
        #[arg(long)]
        machine: PathBuf,
        #[arg(long, default_value_t = 0.05)]
        tol: f64,
    },
    /// Run a lith-suite/0.1 checklist (gates that must not be forgotten).
    Suite { path: PathBuf },
    /// Print schema id and mandatory field list.
    Schema,
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    match cli.cmd {
        Cmd::New { out, claim } => cmd_new(&out, &claim),
        Cmd::Validate { path } => cmd_validate(&path),
        Cmd::Hash { path } => cmd_hash(&path),
        Cmd::Gap { path } => cmd_gap(&path),
        Cmd::OracleCheck { path } => cmd_oracle(&path),
        Cmd::MachineCheck {
            machine,
            measurement,
            tol,
        } => cmd_machine(&machine, &measurement, tol),
        Cmd::IntensityCheck {
            path,
            machine,
            tol,
        } => cmd_intensity(&path, machine.as_ref(), tol),
        Cmd::KernelCheck {
            path,
            machine,
            tol,
        } => cmd_kernel(&path, &machine, tol),
        Cmd::CtCheck { path } => cmd_ct(&path),
        Cmd::PolyCheck {
            path,
            machine,
            tol,
        } => cmd_poly(&path, &machine, tol),
        Cmd::Suite { path } => cmd_suite(&path),
        Cmd::Schema => {
            println!("schema: {SCHEMA_ID}");
            println!("mandatory:");
            for f in [
                "claim",
                "value",
                "unit",
                "baseline{{name,value,unit}}",
                "n_reps>=1",
                "arch",
                "compile_flags[]",
                "clock_state",
                "cache_state",
                "known_limits[] (required if clock/cache unknown)",
                "verified",
            ] {
                println!("  - {f}");
            }
            ExitCode::SUCCESS
        }
    }
}

fn cmd_new(out: &PathBuf, claim: &str) -> ExitCode {
    if let Some(parent) = out.parent() {
        if !parent.as_os_str().is_empty() {
            if let Err(e) = fs::create_dir_all(parent) {
                eprintln!("error: create {}: {e}", parent.display());
                return ExitCode::from(2);
            }
        }
    }
    let bundle = Bundle::template(claim);
    let text = match serde_json::to_string_pretty(&bundle) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("error: serialize: {e}");
            return ExitCode::from(2);
        }
    };
    if let Err(e) = fs::write(out, text + "\n") {
        eprintln!("error: write {}: {e}", out.display());
        return ExitCode::from(2);
    }
    println!("wrote {} (verified=false — fill and validate)", out.display());
    ExitCode::SUCCESS
}

fn cmd_validate(path: &PathBuf) -> ExitCode {
    let raw = match fs::read_to_string(path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("error: read {}: {e}", path.display());
            return ExitCode::from(2);
        }
    };
    let bundle: Bundle = match serde_json::from_str(&raw) {
        Ok(b) => b,
        Err(e) => {
            eprintln!(
                "error: {} is not lith-evidence/0.1 JSON: {e}",
                path.display()
            );
            eprintln!(
                "  hint: run `lith-probe gap {}` if this is a foreign bundle",
                path.display()
            );
            return ExitCode::from(1);
        }
    };
    match validate(&bundle) {
        Ok(()) => {
            let status = if bundle.verified {
                "verified"
            } else {
                "valid (not verified — open honesty is fine)"
            };
            println!("ok: {} — {status}", path.display());
            ExitCode::SUCCESS
        }
        Err(vs) => {
            eprintln!("error: {} failed {} check(s)", path.display(), vs.len());
            for (i, v) in vs.iter().enumerate() {
                eprintln!("\n[{}] {v}", i + 1);
            }
            ExitCode::from(1)
        }
    }
}

fn cmd_hash(path: &PathBuf) -> ExitCode {
    let raw = match fs::read_to_string(path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("error: read {}: {e}", path.display());
            return ExitCode::from(2);
        }
    };
    let value: serde_json::Value = match serde_json::from_str(&raw) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("error: json: {e}");
            return ExitCode::from(1);
        }
    };
    println!("sha256:{}", content_hash(&value));
    ExitCode::SUCCESS
}

fn cmd_gap(path: &PathBuf) -> ExitCode {
    let raw = match fs::read_to_string(path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("error: read {}: {e}", path.display());
            return ExitCode::from(2);
        }
    };
    let value: serde_json::Value = match serde_json::from_str(&raw) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("error: json: {e}");
            return ExitCode::from(1);
        }
    };
    let items = gap_report(&value);
    print!("{}", format_report(&path.display().to_string(), &items));
    let missing = items
        .iter()
        .filter(|i| i.status == lith_probe::GapStatus::Missing)
        .count();
    if missing > 0 {
        ExitCode::from(1)
    } else {
        ExitCode::SUCCESS
    }
}

fn cmd_oracle(path: &PathBuf) -> ExitCode {
    let raw = match fs::read_to_string(path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("error: read {}: {e}", path.display());
            return ExitCode::from(2);
        }
    };
    let case: OracleCase = match serde_json::from_str(&raw) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("error: not lith-oracle-case/0.1: {e}");
            return ExitCode::from(2);
        }
    };
    match oracle_check(&case) {
        Ok(v) => {
            print!("{}", format_oracle_verdict(&case, &v));
            match v {
                OracleVerdict::Pass { .. } => ExitCode::SUCCESS,
                OracleVerdict::Fail { .. } => ExitCode::from(1),
                OracleVerdict::Inconclusive { .. } => ExitCode::from(2),
            }
        }
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::from(2)
        }
    }
}

fn cmd_machine(machine_path: &PathBuf, meas_path: &PathBuf, tol: f64) -> ExitCode {
    let machine_raw = match fs::read_to_string(machine_path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("error: read {}: {e}", machine_path.display());
            return ExitCode::from(2);
        }
    };
    let meas_raw = match fs::read_to_string(meas_path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("error: read {}: {e}", meas_path.display());
            return ExitCode::from(2);
        }
    };
    let machine: Machine = match serde_json::from_str(&machine_raw) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("error: not lith-machine/0.1: {e}");
            return ExitCode::from(2);
        }
    };
    let meas: MachineMeasurement = match serde_json::from_str(&meas_raw) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("error: not lith-machine-measurement/0.1: {e}");
            return ExitCode::from(2);
        }
    };
    match machine_check(&machine, &meas, tol) {
        Ok(v) => {
            print!("{}", format_machine_verdict(&machine, &v));
            match v {
                MachineVerdict::Pass { .. } => ExitCode::SUCCESS,
                MachineVerdict::Fail { .. } => ExitCode::from(1),
            }
        }
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::from(2)
        }
    }
}

fn cmd_intensity(path: &PathBuf, machine_path: Option<&PathBuf>, tol: f64) -> ExitCode {
    let raw = match fs::read_to_string(path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("error: read {}: {e}", path.display());
            return ExitCode::from(2);
        }
    };
    let case: IntensityCase = match serde_json::from_str(&raw) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("error: not lith-intensity/0.1: {e}");
            return ExitCode::from(2);
        }
    };
    let machine_owned: Option<Machine> = match machine_path {
        None => None,
        Some(mp) => {
            let mraw = match fs::read_to_string(mp) {
                Ok(s) => s,
                Err(e) => {
                    eprintln!("error: read {}: {e}", mp.display());
                    return ExitCode::from(2);
                }
            };
            match serde_json::from_str(&mraw) {
                Ok(m) => Some(m),
                Err(e) => {
                    eprintln!("error: not lith-machine/0.1: {e}");
                    return ExitCode::from(2);
                }
            }
        }
    };
    match intensity_check_with_machine(&case, machine_owned.as_ref(), tol) {
        Ok(v) => {
            print!("{}", format_intensity_verdict(&case, &v));
            match v {
                IntensityVerdict::Pass { .. } => ExitCode::SUCCESS,
                IntensityVerdict::Fail { .. } => ExitCode::from(1),
            }
        }
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::from(2)
        }
    }
}

fn cmd_kernel(path: &PathBuf, machine_path: &PathBuf, tol: f64) -> ExitCode {
    let raw = match fs::read_to_string(path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("error: read {}: {e}", path.display());
            return ExitCode::from(2);
        }
    };
    let ir: KernelIr = match serde_json::from_str(&raw) {
        Ok(k) => k,
        Err(e) => {
            eprintln!("error: not lith-kernel-ir/0.1: {e}");
            return ExitCode::from(2);
        }
    };
    let mraw = match fs::read_to_string(machine_path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("error: read {}: {e}", machine_path.display());
            return ExitCode::from(2);
        }
    };
    let machine: Machine = match serde_json::from_str(&mraw) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("error: not lith-machine/0.1: {e}");
            return ExitCode::from(2);
        }
    };
    match kernel_check(&ir, &machine, tol) {
        Ok(v) => {
            print!("{}", format_kernel_verdict(&ir, &v));
            match v {
                KernelVerdict::Pass { .. } => ExitCode::SUCCESS,
                KernelVerdict::FailIntensity { .. } | KernelVerdict::FailCapability { .. } => {
                    ExitCode::from(1)
                }
            }
        }
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::from(2)
        }
    }
}

fn cmd_ct(path: &PathBuf) -> ExitCode {
    let raw = match fs::read_to_string(path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("error: read {}: {e}", path.display());
            return ExitCode::from(2);
        }
    };
    let case: CtCase = match serde_json::from_str(&raw) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("error: not lith-ct/0.1: {e}");
            return ExitCode::from(2);
        }
    };
    match ct_check(&case) {
        Ok(v) => {
            print!("{}", format_ct_verdict(&case, &v));
            match v {
                CtVerdict::Pass { .. } => ExitCode::SUCCESS,
                CtVerdict::Fail { .. } => ExitCode::from(1),
            }
        }
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::from(2)
        }
    }
}

fn cmd_poly(path: &PathBuf, machine_path: &PathBuf, tol: f64) -> ExitCode {
    let raw = match fs::read_to_string(path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("error: read {}: {e}", path.display());
            return ExitCode::from(2);
        }
    };
    let poly: PolyCase = match serde_json::from_str(&raw) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("error: not lith-poly/0.1: {e}");
            return ExitCode::from(2);
        }
    };
    let mraw = match fs::read_to_string(machine_path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("error: read {}: {e}", machine_path.display());
            return ExitCode::from(2);
        }
    };
    let machine: Machine = match serde_json::from_str(&mraw) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("error: not lith-machine/0.1: {e}");
            return ExitCode::from(2);
        }
    };
    match poly_check(&poly, &machine, tol) {
        Ok(v) => {
            print!("{}", format_poly_verdict(&poly, &v));
            match v {
                PolyVerdict::Pass { .. } => ExitCode::SUCCESS,
                PolyVerdict::Fail { .. } => ExitCode::from(1),
            }
        }
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::from(2)
        }
    }
}

fn cmd_suite(path: &PathBuf) -> ExitCode {
    let raw = match fs::read_to_string(path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("error: read {}: {e}", path.display());
            return ExitCode::from(2);
        }
    };
    let suite: Suite = match serde_json::from_str(&raw) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("error: not lith-suite/0.1: {e}");
            return ExitCode::from(2);
        }
    };
    if let Err(e) = suite_check_schema(&suite) {
        eprintln!("error: {e}");
        return ExitCode::from(2);
    }

    println!("suite: {}", suite.name);
    let mut failed = 0usize;
    for (i, step) in suite.steps.iter().enumerate() {
        let (label, expect, _outcome_pass, detail) = run_suite_step(step);
        let matched = match_expect(expect, label == "pass", label);
        match matched {
            StepOutcome::Ok => {
                println!(
                    "  [{}/{}] ok  expect={expect:?} got={label} -- {detail}",
                    i + 1,
                    suite.steps.len()
                );
            }
            StepOutcome::Unexpected { got } => {
                failed += 1;
                eprintln!(
                    "  [{}/{}] FAIL expect={expect:?} got={got} -- {detail}",
                    i + 1,
                    suite.steps.len()
                );
            }
        }
    }
    if failed == 0 {
        println!("suite verdict: PASS ({} steps)", suite.steps.len());
        ExitCode::SUCCESS
    } else {
        eprintln!("suite verdict: FAIL ({failed} unexpected)");
        ExitCode::from(1)
    }
}

fn run_suite_step(step: &Step) -> (&'static str, Expect, bool, String) {
    match step {
        Step::Validate { path, expect } => {
            let raw = match fs::read_to_string(path) {
                Ok(s) => s,
                Err(e) => return ("error", *expect, false, format!("read {path}: {e}")),
            };
            let bundle: Bundle = match serde_json::from_str(&raw) {
                Ok(b) => b,
                Err(e) => return ("error", *expect, false, format!("json {path}: {e}")),
            };
            match validate(&bundle) {
                Ok(()) => ("pass", *expect, true, path.clone()),
                Err(vs) => (
                    "fail",
                    *expect,
                    false,
                    format!("{path}: {} violations", vs.len()),
                ),
            }
        }
        Step::OracleCheck { path, expect } => {
            let raw = match fs::read_to_string(path) {
                Ok(s) => s,
                Err(e) => return ("error", *expect, false, format!("read {path}: {e}")),
            };
            let case: OracleCase = match serde_json::from_str(&raw) {
                Ok(c) => c,
                Err(e) => return ("error", *expect, false, format!("json {path}: {e}")),
            };
            match oracle_check(&case) {
                Ok(OracleVerdict::Pass { .. }) => ("pass", *expect, true, path.clone()),
                Ok(OracleVerdict::Fail { .. }) => ("fail", *expect, false, path.clone()),
                Ok(OracleVerdict::Inconclusive { .. }) => {
                    ("inconclusive", *expect, false, path.clone())
                }
                Err(e) => ("error", *expect, false, e.to_string()),
            }
        }
        Step::MachineCheck {
            machine,
            measurement,
            expect,
            tol,
        } => {
            let mr = match fs::read_to_string(machine) {
                Ok(s) => s,
                Err(e) => return ("error", *expect, false, format!("read {machine}: {e}")),
            };
            let msr = match fs::read_to_string(measurement) {
                Ok(s) => s,
                Err(e) => return ("error", *expect, false, format!("read {measurement}: {e}")),
            };
            let m: Machine = match serde_json::from_str(&mr) {
                Ok(m) => m,
                Err(e) => return ("error", *expect, false, format!("parse machine: {e}")),
            };
            let meas: MachineMeasurement = match serde_json::from_str(&msr) {
                Ok(m) => m,
                Err(e) => return ("error", *expect, false, format!("parse measurement: {e}")),
            };
            match machine_check(&m, &meas, *tol) {
                Ok(MachineVerdict::Pass { .. }) => {
                    ("pass", *expect, true, format!("{machine} vs {measurement}"))
                }
                Ok(MachineVerdict::Fail { .. }) => {
                    ("fail", *expect, false, format!("{machine} vs {measurement}"))
                }
                Err(e) => ("error", *expect, false, e.to_string()),
            }
        }
        Step::IntensityCheck {
            path,
            machine,
            expect,
            tol,
        } => {
            let raw = match fs::read_to_string(path) {
                Ok(s) => s,
                Err(e) => return ("error", *expect, false, format!("read {path}: {e}")),
            };
            let case: IntensityCase = match serde_json::from_str(&raw) {
                Ok(c) => c,
                Err(e) => return ("error", *expect, false, format!("json {path}: {e}")),
            };
            let mach: Option<Machine> = match machine {
                None => None,
                Some(mp) => {
                    let s = match fs::read_to_string(mp) {
                        Ok(s) => s,
                        Err(e) => return ("error", *expect, false, format!("read {mp}: {e}")),
                    };
                    match serde_json::from_str(&s) {
                        Ok(m) => Some(m),
                        Err(e) => return ("error", *expect, false, format!("parse {mp}: {e}")),
                    }
                }
            };
            match intensity_check_with_machine(&case, mach.as_ref(), *tol) {
                Ok(IntensityVerdict::Pass { .. }) => ("pass", *expect, true, path.clone()),
                Ok(IntensityVerdict::Fail { .. }) => ("fail", *expect, false, path.clone()),
                Err(e) => ("error", *expect, false, e.to_string()),
            }
        }
        Step::KernelCheck {
            path,
            machine,
            expect,
            tol,
        } => {
            let raw = match fs::read_to_string(path) {
                Ok(s) => s,
                Err(e) => return ("error", *expect, false, format!("read {path}: {e}")),
            };
            let ir: KernelIr = match serde_json::from_str(&raw) {
                Ok(k) => k,
                Err(e) => return ("error", *expect, false, format!("json {path}: {e}")),
            };
            let mraw = match fs::read_to_string(machine) {
                Ok(s) => s,
                Err(e) => return ("error", *expect, false, format!("read {machine}: {e}")),
            };
            let mach: Machine = match serde_json::from_str(&mraw) {
                Ok(m) => m,
                Err(e) => return ("error", *expect, false, format!("parse {machine}: {e}")),
            };
            match kernel_check(&ir, &mach, *tol) {
                Ok(KernelVerdict::Pass { .. }) => ("pass", *expect, true, path.clone()),
                Ok(KernelVerdict::FailIntensity { .. })
                | Ok(KernelVerdict::FailCapability { .. }) => {
                    ("fail", *expect, false, path.clone())
                }
                Err(e) => ("error", *expect, false, e.to_string()),
            }
        }
        Step::CtCheck { path, expect } => {
            let raw = match fs::read_to_string(path) {
                Ok(s) => s,
                Err(e) => return ("error", *expect, false, format!("read {path}: {e}")),
            };
            let case: CtCase = match serde_json::from_str(&raw) {
                Ok(c) => c,
                Err(e) => return ("error", *expect, false, format!("json {path}: {e}")),
            };
            match ct_check(&case) {
                Ok(CtVerdict::Pass { .. }) => ("pass", *expect, true, path.clone()),
                Ok(CtVerdict::Fail { .. }) => ("fail", *expect, false, path.clone()),
                Err(e) => ("error", *expect, false, e.to_string()),
            }
        }
        Step::PolyCheck {
            path,
            machine,
            expect,
            tol,
        } => {
            let raw = match fs::read_to_string(path) {
                Ok(s) => s,
                Err(e) => return ("error", *expect, false, format!("read {path}: {e}")),
            };
            let poly: PolyCase = match serde_json::from_str(&raw) {
                Ok(p) => p,
                Err(e) => return ("error", *expect, false, format!("json {path}: {e}")),
            };
            let mraw = match fs::read_to_string(machine) {
                Ok(s) => s,
                Err(e) => return ("error", *expect, false, format!("read {machine}: {e}")),
            };
            let mach: Machine = match serde_json::from_str(&mraw) {
                Ok(m) => m,
                Err(e) => return ("error", *expect, false, format!("parse {machine}: {e}")),
            };
            match poly_check(&poly, &mach, *tol) {
                Ok(PolyVerdict::Pass { .. }) => ("pass", *expect, true, path.clone()),
                Ok(PolyVerdict::Fail { .. }) => ("fail", *expect, false, path.clone()),
                Err(e) => ("error", *expect, false, e.to_string()),
            }
        }
    }
}
