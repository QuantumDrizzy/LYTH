//! `lyth-evidence/0.1` — machine-readable claim with a baseline or nothing.

use serde::{Deserialize, Serialize};
use serde_json::Value;

pub const SCHEMA_ID: &str = "lyth-evidence/0.1";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Bundle {
    pub schema: String,
    /// One falsifiable sentence. Not a slogan.
    pub claim: String,
    /// The number (or string) being claimed.
    pub value: Value,
    pub unit: String,
    /// What this is compared against. Required — ADR-0001.
    pub baseline: Baseline,
    /// Sample count. Must be ≥ 1.
    pub n_reps: u32,
    /// Live device arch string, e.g. `sm_120`. Never a hardcoded banner alone.
    pub arch: String,
    /// Compiler / nvcc / rustc flags used for the binary under test.
    pub compile_flags: Vec<String>,
    pub clock_state: ClockState,
    pub cache_state: CacheState,
    /// Open limits. Syntax-level honesty. Empty only when there truly are none.
    #[serde(default)]
    pub known_limits: Vec<KnownLimit>,
    /// May be true only when every limit is `closed` and clock/cache are known.
    pub verified: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub device: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_repo: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub notes: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Baseline {
    pub name: String,
    pub value: Value,
    pub unit: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ClockState {
    /// Measured under a stated boost / locked policy.
    Measured { detail: String },
    /// Not recorded. Must appear as a known_limit id containing `clock`.
    Unknown,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum CacheState {
    Flushed { detail: String },
    Warm { detail: String },
    Unknown,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KnownLimit {
    pub id: String,
    pub text: String,
    pub status: LimitStatus,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LimitStatus {
    Open,
    Closed,
}

impl Bundle {
    /// Minimal honest scaffold — verified false, clock/cache unknown with limits.
    pub fn template(claim: &str) -> Self {
        Self {
            schema: SCHEMA_ID.to_string(),
            claim: claim.to_string(),
            value: Value::Null,
            unit: "REPLACE_ME".into(),
            baseline: Baseline {
                name: "REPLACE_ME".into(),
                value: Value::Null,
                unit: "REPLACE_ME".into(),
            },
            n_reps: 0,
            arch: "REPLACE_ME".into(),
            compile_flags: vec![],
            clock_state: ClockState::Unknown,
            cache_state: CacheState::Unknown,
            known_limits: vec![
                KnownLimit {
                    id: "clock-unmeasured".into(),
                    text: "Boost / locked clock not recorded for this run.".into(),
                    status: LimitStatus::Open,
                },
                KnownLimit {
                    id: "cache-unmeasured".into(),
                    text: "Cache flush / warm policy not recorded for this run.".into(),
                    status: LimitStatus::Open,
                },
            ],
            verified: false,
            device: None,
            source_repo: None,
            notes: vec![
                "Fill value, unit, baseline, n_reps, arch, compile_flags.".into(),
                "Set clock_state / cache_state to measured|flushed|warm and close the limits."
                    .into(),
                "verified may become true only when every known_limit is closed.".into(),
            ],
        }
    }
}

impl crate::document::Document for Bundle {
    const SCHEMA: &'static str = SCHEMA_ID;
}
