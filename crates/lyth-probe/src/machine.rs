//! Machine-as-value — Fase 1.
//!
//! A 2036 GPU is a new file, not a new backend. `machine-check` compares claimed
//! bandwidths against a measurement JSON and refuses if the file lies.

use serde::Deserialize;
use thiserror::Error;

pub const MACHINE_SCHEMA: &str = "lyth-machine/0.1";
pub const MEASUREMENT_SCHEMA: &str = "lyth-machine-measurement/0.1";

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
    /// The ridge, precomputed in the file for a human to read.
    ///
    /// A derived number stored beside its own inputs drifts the moment one of them is edited
    /// and the other is not. `check` recomputes it and refuses a file whose stated ridge
    /// disagrees with its own peak and bandwidth.
    #[serde(default)]
    pub ridge_flop_per_byte: Option<RidgeDecl>,
    #[serde(default)]
    pub notes: Vec<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct RidgeDecl {
    #[serde(default)]
    pub fp32: Option<f64>,
    #[serde(default)]
    pub fp16_tensor: Option<f64>,
    #[serde(default)]
    pub note: String,
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
    /// How the number was produced — the tool, not a repo name. A measurement whose
    /// provenance is another file is a copy, and a copy cannot be re-run.
    #[serde(default)]
    pub source: String,
    /// Achieved FLOPS, for the same reason bandwidth is here: the ridge has two terms and
    /// leaving one unchecked lets half of it drift.
    #[serde(default)]
    pub peak_tflops: Option<f64>,
    /// `locked` | `unlocked` | `base`. Bandwidth moves with clock state, so a measurement
    /// that does not say which one it ran under cannot be compared to one that does.
    #[serde(default)]
    pub clock_state: Option<String>,
    #[serde(default)]
    pub known_limits: Vec<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct MeasuredLevel {
    pub name: String,
    /// The value being claimed. When `runs` is present this must be their median, and
    /// `check` refuses the pair if it is not.
    pub bandwidth_gbs: f64,
    /// The individual runs behind the number.
    ///
    /// The project's own standard is N >= 5 with the median and the spread declared. Until
    /// this field existed a measurement document could not express any of it, so the one
    /// rule the tool enforces everywhere else was unenforceable exactly where it decides
    /// whether a machine file is honest.
    #[serde(default)]
    pub runs: Vec<f64>,
}

impl MeasuredLevel {
    /// Median of `runs`, or `None` when no runs were recorded.
    pub fn median(&self) -> Option<f64> {
        if self.runs.is_empty() {
            return None;
        }
        let mut v = self.runs.clone();
        v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let n = v.len();
        Some(if n % 2 == 1 {
            v[n / 2]
        } else {
            (v[n / 2 - 1] + v[n / 2]) / 2.0
        })
    }

    /// `(max - min) / median` as a fraction, or `None` without runs.
    pub fn spread(&self) -> Option<f64> {
        let median = self.median()?;
        let lo = self.runs.iter().cloned().fold(f64::INFINITY, f64::min);
        let hi = self.runs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        if median == 0.0 {
            None
        } else {
            Some((hi - lo) / median)
        }
    }
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
pub fn check(
    machine: &Machine,
    meas: &MachineMeasurement,
    tol: f64,
) -> Result<MachineVerdict, MachineError> {
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
        return Err(MachineError::Message(format!(
            "tol must be in [0,1), got {tol}"
        )));
    }

    let mut checked = Vec::new();
    let mut mismatches = Vec::new();

    for m in &meas.levels {
        // Judge the measurement before letting it judge the machine file. A headline that
        // is not the median of its own runs is the same defect this command exists to
        // catch, one document earlier.
        if let Some(median) = m.median() {
            let drift = (m.bandwidth_gbs - median).abs() / median;
            if drift > 0.005 {
                mismatches.push(format!(
                    "{name}: measurement claims {claimed:.2} GB/s but the median of its                      {n} runs is {median:.2} GB/s
                       the measurement is not self-consistent; fix it before it is used to judge the machine file",
                    name = m.name,
                    claimed = m.bandwidth_gbs,
                    n = m.runs.len(),
                ));
                continue;
            }
        }
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
            let provenance = match (m.runs.len(), m.spread()) {
                (0, _) => " [no runs recorded]".to_string(),
                (n, Some(spread)) => format!(" (median of n={n}, spread {:.1}%)", spread * 100.0),
                (n, None) => format!(" (median of n={n})"),
            };
            checked.push(format!(
                "{}: claimed {claimed:.2} GB/s, measured {got:.2} GB/s (rel {rel:.3} ≤ {tol}){provenance}",
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

    // The ridge has two terms. Checking bandwidth and leaving peak unchecked lets the other
    // half of every roofline claim drift unobserved.
    if let (Some(claimed), Some(measured)) = (machine.peak_tflops, meas.peak_tflops) {
        let rel = (measured - claimed).abs() / claimed.max(f64::MIN_POSITIVE);
        if rel <= tol {
            checked.push(format!(
                "peak: claimed {claimed:.2} TFLOP/s, measured {measured:.2} TFLOP/s (rel {rel:.3} ≤ {tol})"
            ));
        } else {
            mismatches.push(format!(
                "peak_tflops: machine file claims {claimed:.2}, measured {measured:.2} (rel {rel:.3} > tol {tol})"
            ));
        }
    }

    // A precomputed ridge is a derived number living next to its own inputs. Recompute it.
    if let Some(decl) = &machine.ridge_flop_per_byte {
        if let (Some(stated), Some(derived)) = (decl.fp32, ridge_fp32(machine)) {
            let rel = (stated - derived).abs() / derived.max(f64::MIN_POSITIVE);
            // Tighter than `tol`: this is arithmetic on two numbers in the same file, not a
            // measurement, so anything past rounding is a stale edit.
            if rel <= 0.005 {
                checked.push(format!(
                    "ridge fp32: stated {stated:.2}, derived {derived:.2} flop/byte from this file"
                ));
            } else {
                mismatches.push(format!(
                    "ridge_flop_per_byte.fp32 states {stated:.2} but {peak:.2} TFLOP/s over {bw:.2} GB/s derives {derived:.2}
  a stored ridge drifts when one of its inputs is edited and it is not;
  recompute it, or remove it and let the tool derive it",
                    peak = machine.peak_tflops.unwrap_or(0.0),
                    bw = machine
                        .levels
                        .iter()
                        .find(|l| l.name == "dram")
                        .map(|l| l.bandwidth_gbs)
                        .unwrap_or(0.0),
                ));
            }
        }
    }

    if mismatches.is_empty() {
        Ok(MachineVerdict::Pass { checked })
    } else {
        Ok(MachineVerdict::Fail { mismatches })
    }
}

/// FP32 ridge derived from this file: achieved TFLOP/s over achieved GB/s.
///
/// Both terms must come from the same source. Mixing a datasheet peak with a measured
/// bandwidth (or the reverse) produces a number that describes no machine.
pub fn ridge_fp32(machine: &Machine) -> Option<f64> {
    let peak = machine.peak_tflops?;
    let dram = machine.levels.iter().find(|l| l.name == "dram")?;
    if dram.bandwidth_gbs <= 0.0 {
        return None;
    }
    Some(peak * 1e3 / dram.bandwidth_gbs)
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

impl crate::document::Document for Machine {
    const SCHEMA: &'static str = MACHINE_SCHEMA;
}

impl crate::document::Document for MachineMeasurement {
    const SCHEMA: &'static str = MEASUREMENT_SCHEMA;
}

#[cfg(test)]
mod tests {
    use super::*;

    fn meas_with_runs(headline: f64, runs: Vec<f64>) -> MachineMeasurement {
        MachineMeasurement {
            schema: MEASUREMENT_SCHEMA.into(),
            machine_id: "sm_120".into(),
            source: "test".into(),
            peak_tflops: None,
            clock_state: Some("unlocked".into()),
            known_limits: vec![],
            levels: vec![MeasuredLevel {
                name: "dram".into(),
                bandwidth_gbs: headline,
                runs,
            }],
        }
    }

    #[test]
    fn the_stored_ridge_is_recomputed_from_the_file_that_stores_it() {
        // The defect this catches is one that shipped: an input was corrected and the number
        // derived from it was not. Both live in the same file, so nothing external can notice.
        let machine: Machine = serde_json::from_str(
            r#"{
              "schema":"lyth-machine/0.1","id":"sm_120","peak_tflops":15.37,
              "levels":[{"name":"dram","bandwidth_gbs":358.43}],
              "ridge_flop_per_byte":{"fp32":37.7},
              "ops":[]
            }"#,
        )
        .unwrap();
        let meas = meas_with_runs(358.43, vec![358.43]);
        match check(&machine, &meas, 0.05).unwrap() {
            MachineVerdict::Fail { mismatches } => {
                assert!(mismatches[0].contains("42.88"), "{}", mismatches[0]);
                assert!(mismatches[0].contains("drifts"), "{}", mismatches[0]);
            }
            other => panic!("a stale stored ridge must be refused, got {other:?}"),
        }
    }

    #[test]
    fn the_ridge_is_measured_over_measured_and_the_terms_are_not_mixed() {
        let machine: Machine = serde_json::from_str(
            r#"{
              "schema":"lyth-machine/0.1","id":"sm_120","peak_tflops":15.37,
              "levels":[{"name":"dram","bandwidth_gbs":358.43}],
              "ops":[]
            }"#,
        )
        .unwrap();
        // Achieved SGEMM over achieved bandwidth.
        assert!((ridge_fp32(&machine).unwrap() - 42.88).abs() < 0.05);

        // The datasheet pair for this card is 23.7 TFLOP/s over 448 GB/s, a ridge of 52.9.
        // It is a different, also-valid number and it is not what this file describes. What
        // must never happen is one term from each source.
        assert!((23.7e3_f64 / 448.0 - 52.90).abs() < 0.05);
    }

    #[test]
    fn peak_tflops_is_checked_against_measurement_like_bandwidth_is() {
        let machine: Machine = serde_json::from_str(
            r#"{
              "schema":"lyth-machine/0.1","id":"sm_120","peak_tflops":23.7,
              "levels":[{"name":"dram","bandwidth_gbs":358.43}],
              "ops":[]
            }"#,
        )
        .unwrap();
        // A datasheet peak against a measured one: 23.7 claimed, 15.37 achieved.
        let mut meas = meas_with_runs(358.43, vec![358.43]);
        meas.peak_tflops = Some(15.37);
        match check(&machine, &meas, 0.05).unwrap() {
            MachineVerdict::Fail { mismatches } => {
                assert!(
                    mismatches.iter().any(|m| m.contains("peak_tflops")),
                    "{mismatches:?}"
                );
            }
            other => panic!("a datasheet peak must not pass as a measured one: {other:?}"),
        }
    }

    #[test]
    fn median_and_spread_come_from_the_runs() {
        let m = meas_with_runs(358.43, vec![350.84, 352.26, 354.38, 362.48, 368.32, 368.42]);
        let level = &m.levels[0];
        assert!((level.median().unwrap() - 358.43).abs() < 0.01);
        assert!((level.spread().unwrap() - 0.0491).abs() < 0.001);
    }

    #[test]
    fn a_headline_that_is_not_the_median_of_its_own_runs_is_refused() {
        // The exact defect that shipped: the file kept 398.39 while every run was lower.
        let m = meas_with_runs(398.39, vec![350.84, 352.26, 354.38, 362.48, 368.32, 368.42]);
        let machine: Machine = serde_json::from_str(
            r#"{
              "schema":"lyth-machine/0.1","id":"sm_120",
              "levels":[{"name":"dram","bandwidth_gbs":358.43}],
              "ops":[]
            }"#,
        )
        .unwrap();
        let v = check(&machine, &m, 0.05).unwrap();
        match v {
            MachineVerdict::Fail { mismatches } => {
                assert!(
                    mismatches[0].contains("not self-consistent"),
                    "must blame the measurement, not the machine file: {}",
                    mismatches[0]
                );
            }
            other => panic!("expected the measurement to be refused, got {other:?}"),
        }
    }

    #[test]
    fn within_tol_passes() {
        let machine: Machine = serde_json::from_str(
            r#"{
              "schema":"lyth-machine/0.1","id":"sm_120",
              "levels":[{"name":"dram","bandwidth_gbs":400.0}],
              "ops":[]
            }"#,
        )
        .unwrap();
        let meas: MachineMeasurement = serde_json::from_str(
            r#"{
              "schema":"lyth-machine-measurement/0.1","machine_id":"sm_120",
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
              "schema":"lyth-machine/0.1","id":"sm_120",
              "levels":[{"name":"dram","bandwidth_gbs":500.0}],
              "ops":[]
            }"#,
        )
        .unwrap();
        let meas: MachineMeasurement = serde_json::from_str(
            r#"{
              "schema":"lyth-machine-measurement/0.1","machine_id":"sm_120",
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
