//! Arithmetic intensity as a checked value — pre-parser teeth.
//!
//! Declared FLOPs/byte must match body (Σflops / Σbytes). Mismatch is FAIL with
//! the machine ridge named. No `.lith` parser yet: the accounting JSON *is* the
//! contract the future frontend must emit.

use serde::Deserialize;
use thiserror::Error;

use crate::machine::Machine;

pub const INTENSITY_SCHEMA: &str = "lith-intensity/0.1";

#[derive(Debug, Clone, Deserialize)]
pub struct IntensityCase {
    pub schema: String,
    pub kernel: String,
    /// Declared arithmetic intensity (FLOPs / byte moved through the hierarchy).
    pub declared_intensity: f64,
    pub unit: String,
    pub body: BodyAccounting,
    /// How many of the accounted-for elements one launch processes.
    ///
    /// The accounting above is PER ELEMENT — per neuron, per output row, per tile. `ncu`
    /// reports a TOTAL. Without this field the two are not comparable, so `--ncu` refuses
    /// rather than assuming a problem size. Only `--ncu` reads it; the arithmetic check does
    /// not, because intensity is a ratio and the element count cancels.
    #[serde(default)]
    pub elements: Option<f64>,
    #[serde(default)]
    pub machine_id: Option<String>,
    #[serde(default)]
    pub notes: Vec<String>,
    #[serde(default)]
    pub known_limits: Vec<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct BodyAccounting {
    /// Explicit byte traffic (memory-first). Summed for the intensity denominator.
    pub moves: Vec<Move>,
    /// Floating-point ops attributed to the arithmetic at the stops.
    pub flops: f64,
    #[serde(default)]
    pub flop_note: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Move {
    pub name: String,
    /// Hierarchy level touched (dram, l2, smem, reg).
    pub level: String,
    pub bytes: f64,
    #[serde(default = "default_dir")]
    pub dir: String,
}

fn default_dir() -> String {
    "rw".into()
}

#[derive(Debug, Clone, PartialEq)]
pub enum IntensityVerdict {
    Pass {
        computed: f64,
        declared: f64,
        bytes: f64,
        flops: f64,
        ridge: Option<RidgeInfo>,
        regime: Regime,
    },
    Fail {
        computed: f64,
        declared: f64,
        bytes: f64,
        flops: f64,
        rel_err: f64,
        ridge: Option<RidgeInfo>,
        regime: Regime,
        hints: Vec<String>,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub struct RidgeInfo {
    pub machine_id: String,
    pub peak_tflops: f64,
    pub bandwidth_gbs: f64,
    /// FLOPs/byte at the ridge (peak_tflops * 1e3 / bandwidth_gbs).
    pub ridge_flops_per_byte: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Regime {
    MemoryBound,
    ComputeBound,
    NearRidge,
    Unknown,
}

#[derive(Debug, Error)]
pub enum IntensityError {
    #[error("schema must be `{INTENSITY_SCHEMA}`, got `{0}`")]
    BadSchema(String),
    #[error("unit must be `flops_per_byte`, got `{0}`")]
    BadUnit(String),
    #[error("{0}")]
    Message(String),
}

/// Relative tolerance for declare vs body (default 5%).
pub fn check(case: &IntensityCase, tol: f64) -> Result<IntensityVerdict, IntensityError> {
    check_with_machine(case, None, tol)
}

pub fn check_with_machine(
    case: &IntensityCase,
    machine: Option<&Machine>,
    tol: f64,
) -> Result<IntensityVerdict, IntensityError> {
    if case.schema != INTENSITY_SCHEMA {
        return Err(IntensityError::BadSchema(case.schema.clone()));
    }
    if case.unit != "flops_per_byte" {
        return Err(IntensityError::BadUnit(case.unit.clone()));
    }
    if !(0.0..1.0).contains(&tol) {
        return Err(IntensityError::Message(format!("tol must be in [0,1), got {tol}")));
    }
    if case.body.flops < 0.0 {
        return Err(IntensityError::Message("body.flops must be ≥ 0".into()));
    }
    if case.declared_intensity < 0.0 {
        return Err(IntensityError::Message("declared_intensity must be ≥ 0".into()));
    }

    let bytes: f64 = case.body.moves.iter().map(|m| m.bytes).sum();
    if bytes <= 0.0 {
        return Err(IntensityError::Message(
            "body.moves must sum to > 0 bytes — intensity without traffic is undefined".into(),
        ));
    }

    let computed = case.body.flops / bytes;
    let declared = case.declared_intensity;
    let rel_err = if declared == 0.0 {
        if computed == 0.0 {
            0.0
        } else {
            f64::INFINITY
        }
    } else {
        (computed - declared).abs() / declared
    };

    let ridge = machine.and_then(ridge_of);
    if let (Some(m), Some(id)) = (machine, case.machine_id.as_ref()) {
        if &m.id != id {
            return Err(IntensityError::Message(format!(
                "case.machine_id `{id}` != machine.id `{}`",
                m.id
            )));
        }
    }

    let regime = classify(computed, ridge.as_ref());

    if rel_err <= tol {
        Ok(IntensityVerdict::Pass {
            computed,
            declared,
            bytes,
            flops: case.body.flops,
            ridge,
            regime,
        })
    } else {
        let mut hints = vec![
            format!(
                "body computes {computed:.4} flop/byte from {flops} FLOPs / {bytes} bytes",
                flops = case.body.flops
            ),
            format!("declaration claims {declared:.4} flop/byte (rel err {rel_err:.3} > tol {tol})"),
            "options: (1) fix the declaration to match the body traffic\n           (2) fix the body.moves / body.flops accounting\n           (3) raise --tol only with a known_limit stating why".into(),
        ];
        if let Some(r) = &ridge {
            hints.push(format!(
                "machine.{} ridge ≈ {:.1} flop/byte (peak {:.1} TFLOPS / {:.1} GB/s)",
                r.machine_id, r.ridge_flops_per_byte, r.peak_tflops, r.bandwidth_gbs
            ));
            let frac = computed / r.ridge_flops_per_byte;
            hints.push(format!(
                "at computed intensity you ask for ~{:.1}% of peak FLOPS and are {:?} vs ridge",
                frac * 100.0,
                regime
            ));
        }
        Ok(IntensityVerdict::Fail {
            computed,
            declared,
            bytes,
            flops: case.body.flops,
            rel_err,
            ridge,
            regime,
            hints,
        })
    }
}

pub fn ridge_of(machine: &Machine) -> Option<RidgeInfo> {
    let peak = machine.peak_tflops?;
    let dram = machine.levels.iter().find(|l| l.name == "dram")?;
    if dram.bandwidth_gbs <= 0.0 || peak <= 0.0 {
        return None;
    }
    // TFLOPS / (GB/s) = 1e12 / 1e9 = 1e3 FLOPs/byte
    let ridge_flops_per_byte = peak * 1e3 / dram.bandwidth_gbs;
    Some(RidgeInfo {
        machine_id: machine.id.clone(),
        peak_tflops: peak,
        bandwidth_gbs: dram.bandwidth_gbs,
        ridge_flops_per_byte,
    })
}

fn classify(intensity: f64, ridge: Option<&RidgeInfo>) -> Regime {
    let Some(r) = ridge else {
        return Regime::Unknown;
    };
    let ratio = intensity / r.ridge_flops_per_byte;
    if ratio < 0.5 {
        Regime::MemoryBound
    } else if ratio > 2.0 {
        Regime::ComputeBound
    } else {
        Regime::NearRidge
    }
}

pub fn format_verdict(case: &IntensityCase, v: &IntensityVerdict) -> String {
    let mut out = format!("intensity-check: {}\n", case.kernel);
    match v {
        IntensityVerdict::Pass {
            computed,
            declared,
            bytes,
            flops,
            ridge,
            regime,
        } => {
            out.push_str("verdict: PASS\n");
            out.push_str(&format!(
                "  declared {declared:.4} ≈ computed {computed:.4} flop/byte\n"
            ));
            out.push_str(&format!("  body: {flops} FLOPs / {bytes} bytes\n"));
            out.push_str(&format!("  regime: {regime:?}\n"));
            if let Some(r) = ridge {
                out.push_str(&format!(
                    "  ridge({}): {:.1} flop/byte\n",
                    r.machine_id, r.ridge_flops_per_byte
                ));
            }
        }
        IntensityVerdict::Fail {
            computed,
            declared,
            ridge,
            regime,
            hints,
            ..
        } => {
            out.push_str("verdict: FAIL — declared intensity does not match body\n");
            out.push_str(&format!(
                "  declared {declared:.4} vs computed {computed:.4} flop/byte\n"
            ));
            out.push_str(&format!("  regime: {regime:?}\n"));
            if let Some(r) = ridge {
                out.push_str(&format!(
                    "  ridge({}): {:.1} flop/byte\n",
                    r.machine_id, r.ridge_flops_per_byte
                ));
            }
            for (i, h) in hints.iter().enumerate() {
                out.push_str(&format!("\n[{}] {h}\n", i + 1));
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gemv_lie() -> IntensityCase {
        serde_json::from_str(
            r#"{
              "schema":"lith-intensity/0.1",
              "kernel":"gemv_lie",
              "declared_intensity":2.0,
              "unit":"flops_per_byte",
              "body":{
                "moves":[{"name":"W","level":"dram","bytes":4.0,"dir":"r"},
                         {"name":"x","level":"dram","bytes":4.0,"dir":"r"},
                         {"name":"y","level":"dram","bytes":4.0,"dir":"w"}],
                "flops":2.0
              }
            }"#,
        )
        .unwrap()
    }

    #[test]
    fn lie_fails() {
        // 2 FLOPs / 12 bytes = 0.1667 ≠ 2.0
        match check(&gemv_lie(), 0.05).unwrap() {
            IntensityVerdict::Fail { .. } => {}
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn honest_passes() {
        let case: IntensityCase = serde_json::from_str(
            r#"{
              "schema":"lith-intensity/0.1",
              "kernel":"ok",
              "declared_intensity":0.5,
              "unit":"flops_per_byte",
              "body":{
                "moves":[{"name":"a","level":"dram","bytes":4.0}],
                "flops":2.0
              }
            }"#,
        )
        .unwrap();
        match check(&case, 0.05).unwrap() {
            IntensityVerdict::Pass { .. } => {}
            other => panic!("{other:?}"),
        }
    }
}
