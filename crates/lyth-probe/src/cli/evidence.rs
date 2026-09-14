//! Commands over evidence bundles: scaffold, validate, content-address, adopt.

use std::fs;
use std::path::Path;
use std::process::ExitCode;

use lyth_probe::{content_hash, format_report, gap_report, validate, Bundle, GapStatus, SCHEMA_ID};

use super::io;

/// Fields a bundle must carry before `validate` will pass it.
const MANDATORY_FIELDS: &[&str] = &[
    "claim",
    "value",
    "unit",
    "baseline{name,value,unit}",
    "n_reps>=1",
    "arch",
    "compile_flags[]",
    "clock_state",
    "cache_state",
    "known_limits[] (required if clock/cache unknown)",
    "verified",
];

pub fn new(out: &Path, claim: &str) -> ExitCode {
    if let Some(parent) = out.parent() {
        if !parent.as_os_str().is_empty() {
            if let Err(e) = fs::create_dir_all(parent) {
                eprintln!("error: create {}: {e}", parent.display());
                return io::unusable();
            }
        }
    }
    let text = match serde_json::to_string_pretty(&Bundle::template(claim)) {
        Ok(t) => t,
        Err(e) => return io::failed(e),
    };
    if let Err(e) = fs::write(out, text + "\n") {
        eprintln!("error: write {}: {e}", out.display());
        return io::unusable();
    }
    println!(
        "wrote {} (verified=false — fill and validate)",
        out.display()
    );
    io::ok()
}

pub fn validate_bundle(path: &Path) -> ExitCode {
    let bundle: Bundle = match io::load(path) {
        Ok(b) => b,
        Err(code) => {
            eprintln!(
                "  hint: run `lyth-probe gap {}` if this is a foreign bundle",
                path.display()
            );
            return code;
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
            io::ok()
        }
        Err(violations) => {
            eprintln!(
                "error: {} failed {} check(s)",
                path.display(),
                violations.len()
            );
            for (i, v) in violations.iter().enumerate() {
                eprintln!("\n[{}] {v}", i + 1);
            }
            io::refused()
        }
    }
}

pub fn hash(path: &Path) -> ExitCode {
    match io::load_value(path) {
        Ok(v) => {
            println!("sha256:{}", content_hash(&v));
            io::ok()
        }
        Err(code) => code,
    }
}

pub fn gap(path: &Path) -> ExitCode {
    let value = match io::load_value(path) {
        Ok(v) => v,
        Err(code) => return code,
    };
    let items = gap_report(&value);
    print!("{}", format_report(&path.display().to_string(), &items));
    let missing = items
        .iter()
        .filter(|i| i.status == GapStatus::Missing)
        .count();
    io::verdict(missing == 0)
}

pub fn schema() -> ExitCode {
    println!("schema: {SCHEMA_ID}");
    println!("mandatory:");
    for field in MANDATORY_FIELDS {
        println!("  - {field}");
    }
    io::ok()
}
