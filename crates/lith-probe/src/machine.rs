//! Machine-as-value — Fase 1.
//!
//! A 2036 GPU is a new file, not a new backend. `machine-check` compares claimed
//! bandwidths against a measurement JSON and refuses if the file lies.

use serde::Deserialize;
use thiserror::Error;

pub const MACHINE_SCHEMA: &str = "lith-machine/0.1";
pub const MEASUREMENT_SCHEMA: &str = "lith-machine-measurement/0.1";

#[derive(Debug, Clone, Deserialize)]
pub struct Machine {
    pub schema: String,
    pub id: String,
    pub levels: Vec<Level>,
    /// Peak FP32 (or stated) TFLOPS for ridge = peak*1e3/dram_gbs. Optional until measured.
    #[serde(default)]
    pub peak_tflops: Option<f64>,
    /// Achieved FP16/TC-class GEMM peak when measured separately.
    #[serde(default)]
    pub peak_tflops_fp16: Option<f64>,
    #[serde(default)]
    pub ops: Vec<OpDecl>,
    #[serde(default)]
    pub notes: Vec<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Level {
    pub name: String,
    #[serde(default)]
    pub capacity: Option<String>,
    /// Peak or measured bandwidth in GB/s (decimal 1e9).
    pub bandwidth_gbs: f64,
    #[serde(default)]
    pub latency_cyc: Option<u64>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct OpDecl {
    pub name: String,
    /// `present` | `absent`
    pub status: String,
    #[serde(default)]
    pub at: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct MachineMeasurement {
    pub schema: String,
    pub machine_id: String,
    pub levels: Vec<MeasuredLevel>,
    #[serde(default)]
    pub source: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct MeasuredLevel {
    pub name: String,
    pub bandwidth_gbs: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub enum MachineVerdict {
    Pass { checked: Vec<String> },
    Fail { mismatches: Vec<String> },
}

#[derive(Debug, Error)]
pub enum MachineError {
    #[error("machine schema must be `{MACHINE_SCHEMA}`, got `{0}`")]
    BadMachineSchema(String),
    #[error("measurement schema must be `{MEASUREMENT_SCHEMA}`, got `{0}`")]
    BadMeasSchema(String),
    #[error("machine id `{0}` != measurement machine_id `{1}`")]
    IdMismatch(String, String),
    #[error("{0}")]
    Message(String),
}

/// Relative tolerance: measured must be within `tol` of claimed (fraction).
pub fn check(machine: &Machine, meas: &MachineMeasurement, tol: f64) -> Result<MachineVerdict, MachineError> {
    if machine.schema != MACHINE_SCHEMA {
        return Err(MachineError::BadMachineSchema(machine.schema.clone()));
    }
    if meas.schema != MEASUREMENT_SCHEMA {
        return Err(MachineError::BadMeasSchema(meas.schema.clone()));
    }
    if machine.id != meas.machine_id {
        return Err(MachineError::IdMismatch(
            machine.id.clone(),
            meas.machine_id.clone(),
        ));
    }
    if !(0.0..1.0).contains(&tol) {
        return Err(MachineError::Message(format!("tol must be in [0,1), got {tol}")));
    }

    let mut checked = Vec::new();
    let mut mismatches = Vec::new();

    for m in &meas.levels {
        let Some(level) = machine.levels.iter().find(|l| l.name == m.name) else {
            mismatches.push(format!(
                "measurement has level `{}` not in machine file",
                m.name
            ));
            continue;
        };
        let claimed = level.bandwidth_gbs;
        let got = m.bandwidth_gbs;
        let rel = if claimed == 0.0 {
            f64::INFINITY
        } else {
            (got - claimed).abs() / claimed
        };
        if rel <= tol {
            checked.push(format!(
                "{}: claimed {claimed:.2} GB/s, measured {got:.2} GB/s (rel {rel:.3} ≤ {tol})",
                m.name
            ));
        } else {
            mismatches.push(format!(
                "{name}: machine file claims {claimed:.2} GB/s, measured {got:.2} GB/s \
                 (rel {rel:.3} > tol {tol})\n  \
                 options: (1) update machine file to the measured value\n  \
                          (2) re-measure under the same clock/cache policy as the claim\n  \
                          (3) raise --tol only with a known_limit stating why",
                name = m.name
            ));
        }
    }

    if mismatches.is_empty() {
        Ok(MachineVerdict::Pass { checked })
    } else {
        Ok(MachineVerdict::Fail { mismatches })
    }
}

pub fn format_verdict(machine: &Machine, v: &MachineVerdict) -> String {
    let mut out = format!("machine-check: {}\n", machine.id);
    match v {
        MachineVerdict::Pass { checked } => {
            out.push_str("verdict: PASS\n");
            for c in checked {
                out.push_str(&format!("  ok  {c}\n"));
            }
        }
        MachineVerdict::Fail { mismatches } => {
            out.push_str("verdict: FAIL — machine file lies (or measurement disagrees)\n");
            for (i, m) in mismatches.iter().enumerate() {
                out.push_str(&format!("\n[{}] {m}\n", i + 1));
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn within_tol_passes() {
        let machine: Machine = serde_json::from_str(
            r#"{
              "schema":"lith-machine/0.1","id":"sm_120",
              "levels":[{"name":"dram","bandwidth_gbs":400.0}],
              "ops":[]
            }"#,
        )
        .unwrap();
        let meas: MachineMeasurement = serde_json::from_str(
            r#"{
              "schema":"lith-machine-measurement/0.1","machine_id":"sm_120",
              "levels":[{"name":"dram","bandwidth_gbs":398.4}]
            }"#,
        )
        .unwrap();
        match check(&machine, &meas, 0.05).unwrap() {
            MachineVerdict::Pass { .. } => {}
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn lie_fails() {
        let machine: Machine = serde_json::from_str(
            r#"{
              "schema":"lith-machine/0.1","id":"sm_120",
              "levels":[{"name":"dram","bandwidth_gbs":500.0}],
              "ops":[]
            }"#,
        )
        .unwrap();
        let meas: MachineMeasurement = serde_json::from_str(
            r#"{
              "schema":"lith-machine-measurement/0.1","machine_id":"sm_120",
              "levels":[{"name":"dram","bandwidth_gbs":398.4}]
            }"#,
        )
        .unwrap();
        match check(&machine, &meas, 0.05).unwrap() {
            MachineVerdict::Fail { mismatches } => assert!(!mismatches.is_empty()),
            other => panic!("{other:?}"),
        }
    }
}
