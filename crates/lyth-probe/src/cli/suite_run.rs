//! The suite runner: a checklist of gates that must not be quietly forgotten.
//!
//! Each step reads one or two documents and runs one check. Written out per variant that
//! was seven copies of the same read-parse-dispatch; here the reading is one generic
//! function and `?` carries a failure out as the step's detail line.

use std::fs;
use std::process::ExitCode;

use lyth_probe::document::Document;
use lyth_probe::{
    ct_check, intensity_check_with_machine, kernel_check, machine_check, match_expect,
    oracle_check, poly_check, suite_check_schema, validate, Bundle, CtCase, CtVerdict,
    IntensityCase, IntensityVerdict, KernelIr, KernelVerdict, Machine, MachineMeasurement,
    MachineVerdict, OracleCase, OracleVerdict, Outcome, PolyCase, PolyVerdict, Step, StepOutcome,
    Suite,
};

use super::io;

pub fn run(path: &std::path::Path) -> ExitCode {
    let suite: Suite = match io::load(path) {
        Ok(s) => s,
        Err(code) => return code,
    };
    if let Err(e) = suite_check_schema(&suite) {
        return io::failed(e);
    }

    println!("suite: {}", suite.name);
    let total = suite.steps.len();
    let mut unexpected = 0usize;
    for (i, step) in suite.steps.iter().enumerate() {
        let expect = step.expect();
        let (outcome, detail) = run_step(step);
        let n = i + 1;
        match match_expect(expect, outcome) {
            StepOutcome::Ok => {
                println!(
                    "  [{n}/{total}] ok  expect={expect:?} got={} -- {detail}",
                    outcome.label()
                );
            }
            StepOutcome::Unexpected { got } => {
                unexpected += 1;
                eprintln!("  [{n}/{total}] FAIL expect={expect:?} got={got} -- {detail}");
            }
        }
    }

    if unexpected == 0 {
        println!("suite verdict: PASS ({total} steps)");
        io::ok()
    } else {
        eprintln!("suite verdict: FAIL ({unexpected} unexpected)");
        io::refused()
    }
}

fn run_step(step: &Step) -> (Outcome, String) {
    match dispatch(step) {
        Ok(result) => result,
        // A step that could not run is `error`, never `fail`: nothing was decided.
        Err(detail) => (Outcome::Error, detail),
    }
}

/// Read and parse one operand of a step, reporting failure as the step's detail line.
fn load<T: Document>(path: &str) -> Result<T, String> {
    let raw = fs::read_to_string(path).map_err(|e| format!("read {path}: {e}"))?;
    serde_json::from_str(&raw).map_err(|e| format!("{path} is not {}: {e}", T::SCHEMA))
}

fn dispatch(step: &Step) -> Result<(Outcome, String), String> {
    Ok(match step {
        Step::Validate { path, .. } => {
            let bundle: Bundle = load(path)?;
            match validate(&bundle) {
                Ok(()) => (Outcome::Pass, path.clone()),
                Err(violations) => (
                    Outcome::Fail,
                    format!("{path}: {} violations", violations.len()),
                ),
            }
        }

        Step::OracleCheck { path, .. } => {
            let case: OracleCase = load(path)?;
            let outcome = match oracle_check(&case).map_err(|e| e.to_string())? {
                OracleVerdict::Pass { .. } => Outcome::Pass,
                OracleVerdict::Fail { .. } => Outcome::Fail,
                OracleVerdict::Inconclusive { .. } => Outcome::Inconclusive,
            };
            (outcome, path.clone())
        }

        Step::MachineCheck {
            machine,
            measurement,
            tol,
            ..
        } => {
            let m: Machine = load(machine)?;
            let measured: MachineMeasurement = load(measurement)?;
            let v = machine_check(&m, &measured, *tol).map_err(|e| e.to_string())?;
            (
                Outcome::of(matches!(v, MachineVerdict::Pass { .. })),
                format!("{machine} vs {measurement}"),
            )
        }

        Step::IntensityCheck {
            path, machine, tol, ..
        } => {
            let case: IntensityCase = load(path)?;
            let m: Option<Machine> = match machine {
                None => None,
                Some(p) => Some(load(p)?),
            };
            let v =
                intensity_check_with_machine(&case, m.as_ref(), *tol).map_err(|e| e.to_string())?;
            (
                Outcome::of(matches!(v, IntensityVerdict::Pass { .. })),
                path.clone(),
            )
        }

        Step::KernelCheck {
            path, machine, tol, ..
        } => {
            let ir: KernelIr = load(path)?;
            let m: Machine = load(machine)?;
            let v = kernel_check(&ir, &m, *tol).map_err(|e| e.to_string())?;
            (
                Outcome::of(matches!(v, KernelVerdict::Pass { .. })),
                path.clone(),
            )
        }

        Step::CtCheck { path, .. } => {
            let case: CtCase = load(path)?;
            let v = ct_check(&case).map_err(|e| e.to_string())?;
            (
                Outcome::of(matches!(v, CtVerdict::Pass { .. })),
                path.clone(),
            )
        }

        Step::PolyCheck {
            path, machine, tol, ..
        } => {
            let case: PolyCase = load(path)?;
            let m: Machine = load(machine)?;
            let v = poly_check(&case, &m, *tol).map_err(|e| e.to_string())?;
            (
                Outcome::of(matches!(v, PolyVerdict::Pass { .. })),
                path.clone(),
            )
        }
    })
}
