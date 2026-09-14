//! LITH evidence bundles — Fase 0.
//!
//! A claim without a baseline does not ship. Clock/cache unknown without a
//! `[KNOWN_LIMIT]` does not ship. `verified: true` with open limits does not ship.
//!
//! No parser. No kernels. This crate only refuses incomplete evidence.

pub mod adopt;
pub mod ct;
mod hash;
pub mod intensity;
pub mod kernel;
pub mod machine;
pub mod oracle;
pub mod poly;
mod schema;
pub mod suite;
mod validate;

pub use adopt::{format_report, gap_report, GapItem, GapStatus};
pub use ct::{
    check as ct_check, format_verdict as format_ct_verdict, CtCase, CtVerdict, CT_SCHEMA,
};
pub use hash::content_hash;
pub use intensity::{
    check as intensity_check, check_with_machine as intensity_check_with_machine,
    format_verdict as format_intensity_verdict, ridge_of, BodyAccounting, IntensityCase,
    IntensityVerdict, Move, INTENSITY_SCHEMA,
};
pub use kernel::{
    check as kernel_check, format_verdict as format_kernel_verdict, lower as lower_kernel,
    KernelIr, KernelVerdict, KERNEL_IR_SCHEMA,
};
pub use machine::{
    check as machine_check, format_verdict as format_machine_verdict, Machine,
    MachineMeasurement, MachineVerdict, MACHINE_SCHEMA, MEASUREMENT_SCHEMA,
};
pub use oracle::{check as oracle_check, format_verdict as format_oracle_verdict, OracleCase, OracleVerdict};
pub use poly::{
    check as poly_check, format_verdict as format_poly_verdict, instantiate as poly_instantiate,
    PolyCase, PolyVerdict, POLY_SCHEMA,
};
pub use schema::{
    Baseline, Bundle, CacheState, ClockState, KnownLimit, LimitStatus, SCHEMA_ID,
};
pub use suite::{
    check_schema as suite_check_schema, match_expect, Expect, Step, StepOutcome, Suite,
    SUITE_SCHEMA,
};
pub use validate::{validate, Violation};
