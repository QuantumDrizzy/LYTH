//! Gap analysis against foreign evidence (rse-audit, neuromod, ad-hoc).
//!
//! Does not pretend foreign bundles are lith bundles. Lists what is missing
//! so adopting a repo is a checklist, not a rewrite.

use serde_json::Value;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GapItem {
    pub field: &'static str,
    pub status: GapStatus,
    pub note: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GapStatus {
    Present,
    Missing,
    Partial,
}

pub fn gap_report(raw: &Value) -> Vec<GapItem> {
    let mut items = Vec::new();

    let schema = raw.get("schema").and_then(|v| v.as_str()).unwrap_or("");
    items.push(GapItem {
        field: "schema",
        status: if schema == crate::SCHEMA_ID {
            GapStatus::Present
        } else if schema.is_empty() {
            GapStatus::Missing
        } else {
            GapStatus::Partial
        },
        note: if schema.is_empty() {
            "no schema field".into()
        } else {
            format!("foreign schema `{schema}` — adopt into lith-evidence/0.1")
        },
    });

    let has_claim = raw
        .get("claim")
        .and_then(|v| v.as_str())
        .is_some_and(|s| !s.is_empty());
    items.push(GapItem {
        field: "claim",
        status: if has_claim {
            GapStatus::Present
        } else if raw.get("exercise").is_some() {
            GapStatus::Partial
        } else {
            GapStatus::Missing
        },
        note: if has_claim {
            "ok".into()
        } else if raw.get("exercise").is_some() {
            "has exercise id — promote to an explicit claim sentence".into()
        } else {
            "no claim".into()
        },
    });

    let has_value = raw.get("value").map(|v| !v.is_null()).unwrap_or(false)
        || raw.pointer("/measurements/0").is_some();
    push(
        &mut items,
        "value",
        has_value,
        if has_value {
            "value or measurements[] present".into()
        } else {
            "no value / measurements".into()
        },
    );

    let has_baseline = raw.get("baseline").is_some()
        || raw.get("baselines").and_then(|v| v.as_array()).is_some_and(|a| !a.is_empty())
        || raw
            .pointer("/probes/0/evidence/baselines")
            .and_then(|v| v.as_array())
            .is_some_and(|a| !a.is_empty());
    items.push(GapItem {
        field: "baseline",
        status: if has_baseline {
            GapStatus::Present
        } else {
            GapStatus::Missing
        },
        note: if has_baseline {
            "baseline(s) found".into()
        } else {
            "MISSING — lith will refuse without a named baseline".into()
        },
    });

    let n = raw
        .get("n_reps")
        .and_then(|v| v.as_u64())
        .or_else(|| {
            raw.pointer("/measurements/0/samples_us")
                .and_then(|v| v.as_array())
                .map(|a| a.len() as u64)
        });
    items.push(GapItem {
        field: "n_reps",
        status: match n {
            Some(x) if x >= 1 => GapStatus::Present,
            Some(_) => GapStatus::Partial,
            None => GapStatus::Missing,
        },
        note: match n {
            Some(x) => format!("n={x}"),
            None => "no n_reps / samples".into(),
        },
    });

    let arch = raw
        .get("arch")
        .or_else(|| raw.pointer("/measurements/0/sm_major"))
        .or_else(|| raw.pointer("/calibration/sm_major"));
    push(
        &mut items,
        "arch",
        arch.is_some(),
        if arch.is_some() {
            "arch / sm_* present".into()
        } else {
            "MISSING live arch".into()
        },
    );

    push(
        &mut items,
        "compile_flags",
        raw.get("compile_flags").is_some()
            || raw.get("nvcc_host_flags").is_some()
            || raw.pointer("/measurements/0/nvcc_host_flags").is_some(),
        "compile_flags or nvcc_host_flags".into(),
    );

    push(
        &mut items,
        "clock_state",
        raw.get("clock_state").is_some(),
        if raw.get("clock_state").is_some() {
            "ok".into()
        } else {
            "MISSING — ncu taught you this one".into()
        },
    );

    push(
        &mut items,
        "cache_state",
        raw.get("cache_state").is_some(),
        if raw.get("cache_state").is_some() {
            "ok".into()
        } else {
            "MISSING".into()
        },
    );

    push(
        &mut items,
        "known_limits",
        raw.get("known_limits").is_some() || raw.get("notes").is_some(),
        if raw.get("known_limits").is_some() {
            "ok".into()
        } else if raw.get("notes").is_some() {
            "notes present — promote open limits to known_limits[]".into()
        } else {
            "MISSING".into()
        },
    );

    items
}

fn push(items: &mut Vec<GapItem>, field: &'static str, ok: bool, note: String) {
    items.push(GapItem {
        field,
        status: if ok {
            GapStatus::Present
        } else if note.contains("promote") || note.contains("PARTIAL") {
            GapStatus::Partial
        } else {
            GapStatus::Missing
        },
        note,
    });
}

pub fn format_report(path: &str, items: &[GapItem]) -> String {
    let mut out = format!("gap: {path}\n");
    for it in items {
        let mark = match it.status {
            GapStatus::Present => "ok ",
            GapStatus::Partial => "~  ",
            GapStatus::Missing => "NO ",
        };
        out.push_str(&format!("  [{mark}] {:<14} {}\n", it.field, it.note));
    }
    let missing = items
        .iter()
        .filter(|i| i.status == GapStatus::Missing)
        .count();
    out.push_str(&format!(
        "summary: {missing} mandatory field(s) missing for lith-evidence/0.1\n"
    ));
    out
}
