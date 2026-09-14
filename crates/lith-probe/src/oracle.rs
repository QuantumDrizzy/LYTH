//! Oracle rank-order gate — Fase 0.5.

use serde::Deserialize;
use thiserror::Error;

pub const ORACLE_CASE_SCHEMA: &str = "lith-oracle-case/0.1";

#[derive(Debug, Clone, Deserialize)]
pub struct OracleCase {
    pub schema: String,
    pub metric_oracle: String,
    pub metric_silicon: String,
    #[serde(default)]
    pub metric_note: String,
    pub kernels: Vec<OracleKernel>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct OracleKernel {
    pub id: String,
    pub oracle_score: f64,
    pub silicon_score: Option<f64>,
    #[serde(default)]
    pub silicon_note: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OracleVerdict {
    /// Same descending order.
    Pass { order: Vec<String> },
    /// Orders differ.
    Fail {
        oracle_order: Vec<String>,
        silicon_order: Vec<String>,
    },
    /// Missing silicon scores — do not pretend.
    Inconclusive { missing: Vec<String> },
}

#[derive(Debug, Error)]
pub enum OracleError {
    #[error("schema must be `{ORACLE_CASE_SCHEMA}`, got `{0}`")]
    BadSchema(String),
    #[error("need at least 2 kernels, got {0}")]
    TooFew(usize),
    #[error("{0}")]
    Message(String),
}

pub fn check(case: &OracleCase) -> Result<OracleVerdict, OracleError> {
    if case.schema != ORACLE_CASE_SCHEMA {
        return Err(OracleError::BadSchema(case.schema.clone()));
    }
    if case.kernels.len() < 2 {
        return Err(OracleError::TooFew(case.kernels.len()));
    }

    let missing: Vec<String> = case
        .kernels
        .iter()
        .filter(|k| k.silicon_score.is_none())
        .map(|k| k.id.clone())
        .collect();
    if !missing.is_empty() {
        return Ok(OracleVerdict::Inconclusive { missing });
    }

    let oracle_order = rank_desc(
        case.kernels
            .iter()
            .map(|k| (k.id.as_str(), k.oracle_score)),
    );
    let silicon_order = rank_desc(case.kernels.iter().map(|k| {
        (
            k.id.as_str(),
            k.silicon_score.expect("checked above"),
        )
    }));

    if oracle_order == silicon_order {
        Ok(OracleVerdict::Pass {
            order: oracle_order,
        })
    } else {
        Ok(OracleVerdict::Fail {
            oracle_order,
            silicon_order,
        })
    }
}

fn rank_desc<'a>(items: impl Iterator<Item = (&'a str, f64)>) -> Vec<String> {
    let mut v: Vec<(&str, f64)> = items.collect();
    v.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.0.cmp(b.0))
    });
    v.into_iter().map(|(id, _)| id.to_string()).collect()
}

pub fn format_verdict(case: &OracleCase, v: &OracleVerdict) -> String {
    let mut out = String::new();
    out.push_str(&format!(
        "oracle-check\n  oracle metric:  {}\n  silicon metric: {}\n",
        case.metric_oracle, case.metric_silicon
    ));
    if !case.metric_note.is_empty() {
        out.push_str(&format!("  note: {}\n", case.metric_note));
    }
    match v {
        OracleVerdict::Pass { order } => {
            out.push_str(&format!(
                "verdict: PASS — same order\n  {}\n",
                order.join(" > ")
            ));
        }
        OracleVerdict::Fail {
            oracle_order,
            silicon_order,
        } => {
            out.push_str(&format!(
                "verdict: FAIL — orders diverge\n  oracle:  {}\n  silicon: {}\n  \
                 @oracle does not rank like this silicon metric. Publish or change the metric.\n",
                oracle_order.join(" > "),
                silicon_order.join(" > ")
            ));
        }
        OracleVerdict::Inconclusive { missing } => {
            out.push_str(&format!(
                "verdict: INCONCLUSIVE — missing silicon_score for: {}\n  \
                 fill B and C (ising, mps) before claiming @oracle is validated.\n",
                missing.join(", ")
            ));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn incomplete_is_inconclusive() {
        let case: OracleCase = serde_json::from_str(include_str!(
            "../../../fixtures/oracle/case-incomplete.json"
        ))
        .unwrap();
        match check(&case).unwrap() {
            OracleVerdict::Inconclusive { missing } => {
                assert!(missing.contains(&"mps_chain".into()));
                assert!(missing.contains(&"ising_energy".into()));
            }
            other => panic!("expected inconclusive, got {other:?}"),
        }
    }

    #[test]
    fn density_vs_ceiling_diverges() {
        let case: OracleCase = serde_json::from_str(include_str!(
            "../../../fixtures/oracle/case-selfcheck-diverge.json"
        ))
        .unwrap();
        match check(&case).unwrap() {
            OracleVerdict::Fail {
                oracle_order,
                silicon_order,
            } => {
                assert_eq!(
                    oracle_order,
                    vec![
                        "mps_chain".to_string(),
                        "llm_matvec".to_string(),
                        "ising_energy".to_string()
                    ]
                );
                assert_eq!(
                    silicon_order,
                    vec![
                        "llm_matvec".to_string(),
                        "mps_chain".to_string(),
                        "ising_energy".to_string()
                    ]
                );
            }
            other => panic!("expected fail, got {other:?}"),
        }
    }
}
