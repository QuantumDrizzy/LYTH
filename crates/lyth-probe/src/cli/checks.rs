//! Commands that compare a declaration against something else: a machine file against
//! measured silicon, an accounting against its own arithmetic, an accounting against
//! measured traffic, an IR against a machine's capabilities.

use std::path::Path;
use std::process::ExitCode;

use lyth_probe::{
    ct_check, format_ct_verdict, format_intensity_verdict, format_kernel_verdict,
    format_machine_verdict, format_ncu_verdict, format_oracle_verdict, format_poly_verdict,
    intensity_check_with_machine, kernel_check, machine_check, ncu, ncu_compare, ncu_parse,
    oracle_check, poly_check, CtCase, CtVerdict, IntensityCase, IntensityVerdict, KernelIr,
    KernelVerdict, Machine, MachineMeasurement, MachineVerdict, OracleCase, OracleVerdict,
    PolyCase, PolyVerdict, TrafficVerdict,
};

use super::io;

pub fn oracle(path: &Path) -> ExitCode {
    let case: OracleCase = match io::load(path) {
        Ok(c) => c,
        Err(code) => return code,
    };
    match oracle_check(&case) {
        Ok(v) => {
            print!("{}", format_oracle_verdict(&case, &v));
            match v {
                OracleVerdict::Pass { .. } => io::ok(),
                OracleVerdict::Fail { .. } => io::refused(),
                // Not a verdict: the case did not carry enough to decide either way.
                OracleVerdict::Inconclusive { .. } => io::unusable(),
            }
        }
        Err(e) => io::failed(e),
    }
}

pub fn machine(machine_path: &Path, measurement_path: &Path, tol: f64) -> ExitCode {
    let machine: Machine = match io::load(machine_path) {
        Ok(m) => m,
        Err(code) => return code,
    };
    let measured: MachineMeasurement = match io::load(measurement_path) {
        Ok(m) => m,
        Err(code) => return code,
    };
    match machine_check(&machine, &measured, tol) {
        Ok(v) => {
            print!("{}", format_machine_verdict(&machine, &v));
            io::verdict(matches!(v, MachineVerdict::Pass { .. }))
        }
        Err(e) => io::failed(e),
    }
}

/// Everything `--ncu` needs, kept together so the signature stays readable.
pub struct TrafficOpts<'a> {
    pub report: Option<&'a Path>,
    /// Substring selecting one kernel from a multi-kernel report.
    pub kernel: Option<&'a str>,
    /// Overrides the case file's `elements`.
    pub elements: Option<f64>,
    /// Overrides the level the accounting's moves declare.
    pub level: Option<&'a str>,
    pub tol: f64,
}

pub fn intensity(
    path: &Path,
    machine_path: Option<&Path>,
    tol: f64,
    traffic: &TrafficOpts,
) -> ExitCode {
    let case: IntensityCase = match io::load(path) {
        Ok(c) => c,
        Err(code) => return code,
    };
    let machine: Option<Machine> = match machine_path {
        None => None,
        Some(p) => match io::load(p) {
            Ok(m) => Some(m),
            Err(code) => return code,
        },
    };

    let arithmetic = match intensity_check_with_machine(&case, machine.as_ref(), tol) {
        Ok(v) => {
            print!("{}", format_intensity_verdict(&case, &v));
            matches!(v, IntensityVerdict::Pass { .. })
        }
        Err(e) => return io::failed(e),
    };

    let measured = match traffic.report {
        None => true,
        Some(report) => match check_traffic(&case, report, traffic) {
            Some(holds) => holds,
            // A problem with the report is not a verdict on the kernel.
            None => return io::unusable(),
        },
    };
    io::verdict(arithmetic && measured)
}

/// The measured half of the intensity check.
///
/// `Some(true)` the byte model is confirmed against silicon; `Some(false)` it is not, and
/// the reasons are printed; `None` the report or the problem size is unusable, which is an
/// input error rather than a verdict.
fn check_traffic(case: &IntensityCase, report: &Path, opts: &TrafficOpts) -> Option<bool> {
    let csv = io::read_text(report).ok()?;

    // The accounting names its own level. Check against that one, not against whichever
    // metric happens to be in the report: a correct L2 accounting compared against DRAM
    // reads as a failure, and that mistake is indistinguishable from a real one.
    let level = match opts.level {
        Some(l) => l.to_string(),
        None => ncu::deepest_level(case.body.moves.iter().map(|m| m.level.as_str())),
    };
    let measured = match ncu_parse(&csv, opts.kernel, Some(&level)) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("error: ncu report: {e}");
            return None;
        }
    };

    // The accounting is per element; ncu reports a total. Refuse to invent the factor that
    // relates them — a guessed element count makes any ratio come out at 1.0.
    let elements = match opts.elements.or(case.elements) {
        Some(n) => n,
        None => {
            eprintln!("error: --ncu needs an element count.");
            eprintln!(
                "  The accounting in {} is per element and the report is a total;",
                case.kernel
            );
            eprintln!("  without the problem size they are not comparable.");
            eprintln!("  Pass --elements N, or add \"elements\" to the case file.");
            return None;
        }
    };

    let bytes_per_element: f64 = case.body.moves.iter().map(|m| m.bytes).sum();
    match ncu_compare(bytes_per_element, elements, &measured, opts.tol) {
        Ok(v) => {
            print!("{}", format_ncu_verdict(&v, &measured, elements));
            Some(matches!(v, TrafficVerdict::Confirmed { .. }))
        }
        Err(e) => {
            eprintln!("error: {e}");
            None
        }
    }
}

pub fn kernel(path: &Path, machine_path: &Path, tol: f64) -> ExitCode {
    let ir: KernelIr = match io::load(path) {
        Ok(i) => i,
        Err(code) => return code,
    };
    let machine: Machine = match io::load(machine_path) {
        Ok(m) => m,
        Err(code) => return code,
    };
    match kernel_check(&ir, &machine, tol) {
        Ok(v) => {
            print!("{}", format_kernel_verdict(&ir, &v));
            io::verdict(matches!(v, KernelVerdict::Pass { .. }))
        }
        Err(e) => io::failed(e),
    }
}

pub fn constant_time(path: &Path) -> ExitCode {
    let case: CtCase = match io::load(path) {
        Ok(c) => c,
        Err(code) => return code,
    };
    match ct_check(&case) {
        Ok(v) => {
            print!("{}", format_ct_verdict(&case, &v));
            io::verdict(matches!(v, CtVerdict::Pass { .. }))
        }
        Err(e) => io::failed(e),
    }
}

pub fn poly(path: &Path, machine_path: &Path, tol: f64) -> ExitCode {
    let case: PolyCase = match io::load(path) {
        Ok(p) => p,
        Err(code) => return code,
    };
    let machine: Machine = match io::load(machine_path) {
        Ok(m) => m,
        Err(code) => return code,
    };
    match poly_check(&case, &machine, tol) {
        Ok(v) => {
            print!("{}", format_poly_verdict(&case, &v));
            io::verdict(matches!(v, PolyVerdict::Pass { .. }))
        }
        Err(e) => io::failed(e),
    }
}
