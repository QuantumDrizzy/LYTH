//! Validation — exit code 1 is the product.

use crate::schema::{Bundle, CacheState, ClockState, LimitStatus, SCHEMA_ID};
use thiserror::Error;

#[derive(Debug, Clone, PartialEq, Eq, Error)]
#[error("{0}")]
pub struct Violation(pub String);

pub fn validate(b: &Bundle) -> Result<(), Vec<Violation>> {
    let mut v = Vec::new();

    if b.schema != SCHEMA_ID {
        v.push(Violation(format!(
            "schema must be `{SCHEMA_ID}`, got `{}`",
            b.schema
        )));
    }
    if b.claim.trim().is_empty() {
        v.push(Violation("claim is empty".into()));
    }
    if b.claim.contains("REPLACE_ME") {
        v.push(Violation("claim still contains REPLACE_ME".into()));
    }
    if b.value.is_null() {
        v.push(Violation("value is null — a claim without a number does not ship".into()));
    }
    if b.unit.trim().is_empty() || b.unit == "REPLACE_ME" {
        v.push(Violation("unit missing or REPLACE_ME".into()));
    }
    if b.baseline.name.trim().is_empty() || b.baseline.name == "REPLACE_ME" {
        v.push(Violation(
            "baseline.name missing — never a number without its baseline".into(),
        ));
    }
    if b.baseline.value.is_null() {
        v.push(Violation("baseline.value is null".into()));
    }
    if b.baseline.unit.trim().is_empty() || b.baseline.unit == "REPLACE_ME" {
        v.push(Violation("baseline.unit missing or REPLACE_ME".into()));
    }
    if b.n_reps < 1 {
        v.push(Violation(format!(
            "n_reps must be ≥ 1, got {} (medians of 2 samples are how lies start)",
            b.n_reps
        )));
    }
    if b.arch.trim().is_empty() || b.arch == "REPLACE_ME" {
        v.push(Violation(
            "arch missing — must come from the live device / cubin, not a banner".into(),
        ));
    }

    match &b.clock_state {
        ClockState::Unknown => {
            if !has_limit_id(&b.known_limits, "clock") {
                v.push(Violation(
                    "clock_state is unknown but no known_limit id contains `clock`\n  \
                     options: (1) measure/lock boost and set clock_state.measured\n  \
                              (2) add known_limit { id: \"clock-…\", status: open }"
                        .into(),
                ));
            }
        }
        ClockState::Measured { detail } if detail.trim().is_empty() => {
            v.push(Violation("clock_state.measured.detail is empty".into()));
        }
        ClockState::Measured { .. } => {}
    }

    match &b.cache_state {
        CacheState::Unknown => {
            if !has_limit_id(&b.known_limits, "cache") {
                v.push(Violation(
                    "cache_state is unknown but no known_limit id contains `cache`\n  \
                     options: (1) flush or document warm and set cache_state\n  \
                              (2) add known_limit { id: \"cache-…\", status: open }"
                        .into(),
                ));
            }
        }
        CacheState::Flushed { detail } | CacheState::Warm { detail }
            if detail.trim().is_empty() =>
        {
            v.push(Violation("cache_state detail is empty".into()));
        }
        CacheState::Flushed { .. } | CacheState::Warm { .. } => {}
    }

    for lim in &b.known_limits {
        if lim.id.trim().is_empty() || lim.text.trim().is_empty() {
            v.push(Violation(
                "known_limit with empty id or text — [KNOWN_LIMIT] is syntax, not a shrug"
                    .into(),
            ));
        }
    }

    if b.verified {
        let open: Vec<&str> = b
            .known_limits
            .iter()
            .filter(|l| l.status == LimitStatus::Open)
            .map(|l| l.id.as_str())
            .collect();
        if !open.is_empty() {
            v.push(Violation(format!(
                "verified=true with open known_limits: {}\n  \
                 close them or set verified=false",
                open.join(", ")
            )));
        }
        if matches!(b.clock_state, ClockState::Unknown) {
            v.push(Violation(
                "verified=true but clock_state is unknown".into(),
            ));
        }
        if matches!(b.cache_state, CacheState::Unknown) {
            v.push(Violation(
                "verified=true but cache_state is unknown".into(),
            ));
        }
    }

    if v.is_empty() {
        Ok(())
    } else {
        Err(v)
    }
}

fn has_limit_id(limits: &[crate::schema::KnownLimit], needle: &str) -> bool {
    limits
        .iter()
        .any(|l| l.id.to_ascii_lowercase().contains(needle))
}
