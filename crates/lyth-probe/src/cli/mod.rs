//! The command layer: read arguments, load documents, call the domain, print, exit.
//!
//! No check lives here. Everything under `lyth_probe::` decides; this layer only moves
//! bytes between a shell and that decision, and maps a verdict onto an exit code.

pub mod checks;
pub mod evidence;
pub mod io;
pub mod suite_run;
