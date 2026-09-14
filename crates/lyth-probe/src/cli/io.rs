//! Reading the documents a check operates on, and the exit codes it answers with.
//!
//! Every command opened the same way: read a file, parse JSON, exit on either failure.
//! Spelled out by hand that was twenty-six near-identical `match` blocks whose only
//! varying content was the schema name in the error message. It is two functions here.

use std::fs;
use std::path::Path;
use std::process::ExitCode;

use lyth_probe::document::Document;

/// The claim holds, or nothing was claimed.
pub const OK: u8 = 0;
/// The evidence refuses the claim. **This is a result, not a crash** — a refusal is the
/// product, so it gets its own code and a caller can tell it from a broken invocation.
pub const REFUSED: u8 = 1;
/// The input could not be read, parsed, or compared. Nothing was decided either way, and
/// reporting a refusal here would be claiming a verdict that was never reached.
pub const UNUSABLE: u8 = 2;

pub fn ok() -> ExitCode {
    ExitCode::from(OK)
}

pub fn refused() -> ExitCode {
    ExitCode::from(REFUSED)
}

pub fn unusable() -> ExitCode {
    ExitCode::from(UNUSABLE)
}

/// `true` is 0, `false` is [`REFUSED`]. Keeps the verdict-to-exit mapping in one place
/// instead of at the bottom of every command.
pub fn verdict(holds: bool) -> ExitCode {
    if holds {
        ok()
    } else {
        refused()
    }
}

/// Report an error from a check and exit [`UNUSABLE`].
pub fn failed(e: impl std::fmt::Display) -> ExitCode {
    eprintln!("error: {e}");
    unusable()
}

pub fn read_text(path: &Path) -> Result<String, ExitCode> {
    fs::read_to_string(path).map_err(|e| {
        eprintln!("error: read {}: {e}", path.display());
        unusable()
    })
}

/// Parse a document that knows which schema it must carry, so the diagnostic cannot name
/// a different schema from the one the parser required.
pub fn load<T: Document>(path: &Path) -> Result<T, ExitCode> {
    let raw = read_text(path)?;
    serde_json::from_str(&raw).map_err(|e| {
        eprintln!("error: {} is not {}: {e}", path.display(), T::SCHEMA);
        unusable()
    })
}

/// Parse a file whose shape is not known in advance — a foreign bundle under `gap`, or
/// anything being content-addressed by `hash`.
pub fn load_value(path: &Path) -> Result<serde_json::Value, ExitCode> {
    let raw = read_text(path)?;
    serde_json::from_str(&raw).map_err(|e| {
        eprintln!("error: {} is not JSON: {e}", path.display());
        unusable()
    })
}
