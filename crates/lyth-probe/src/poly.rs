//! Layout × machine polymorphism — cost re-checked per instantiation.
//!
//! Same kernel body, different layout ⇒ different traffic ⇒ different intensity.
//! NIBBLE proof: holding the kernel constant and changing only INT4→NF4 costs
//! 19–31% wall time (gate_proj ≈ 27%).

use serde::Deserialize;
use thiserror::Error;

use crate::kernel::{
    check as kernel_check, ArithOp, CapabilityReq, KernelIr, KernelVerdict, Stream,
};
use crate::machine::Machine;

pub const POLY_SCHEMA: &str = "lyth-poly/0.1";

#[derive(Debug, Clone, Deserialize)]
pub struct PolyCase {
    pub schema: String,
    pub name: String,
    pub machine_id: String,
    /// Instantiations — one per layout (same logical kernel).
    pub instances: Vec<PolyInstance>,
    /// Optional falsifiable claim about format cost across layouts.
    #[serde(default)]
    pub claim: Option<FormatCostClaim>,
    #[serde(default)]
    pub notes: Vec<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct PolyInstance {
    pub layout: String,
    pub declared_intensity: f64,
    pub streams: Vec<Stream>,
    pub ops: Vec<ArithOp>,
    /// Elements one launch of this instance processes. A layout change can change it, so it
    /// is per instance rather than per case.
    #[serde(default)]
    pub elements: Option<f64>,
    #[serde(default)]
    pub requires: Vec<CapabilityReq>,
    #[serde(default)]
    pub silicon: Option<SiliconRow>,
    #[serde(default)]
    pub known_limits: Vec<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct SiliconRow {
    pub kernel_us: f64,
    #[serde(default)]
    pub gbs: Option<f64>,
    #[serde(default)]
    pub bytes: Option<f64>,
    #[serde(default)]
    pub pct_dram: Option<f64>,
    #[serde(default)]
    pub source: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct FormatCostClaim {
    pub baseline_layout: String,
    pub compare_layout: String,
    /// `(compare - baseline) / baseline` on `kernel_us`.
    pub expected_min: f64,
    pub expected_max: f64,
    #[serde(default)]
    pub note: String,
}

#[derive(Debug, Clone, PartialEq)]
pub enum PolyVerdict {
    Pass {
        layouts: Vec<String>,
        intensities: Vec<(String, f64)>,
        format_cost: Option<f64>,
    },
    Fail {
        reasons: Vec<String>,
    },
}

#[derive(Debug, Error)]
pub enum PolyError {
    #[error("schema must be `{POLY_SCHEMA}`, got `{0}`")]
    BadSchema(String),
    #[error("poly.machine_id `{0}` != machine.id `{1}`")]
    MachineMismatch(String, String),
    #[error("{0}")]
    Message(String),
}

pub fn instantiate(poly: &PolyCase, inst: &PolyInstance) -> KernelIr {
    KernelIr {
        schema: crate::kernel::KERNEL_IR_SCHEMA.to_string(),
        name: format!("{}[{}]", poly.name, inst.layout),
        machine_id: poly.machine_id.clone(),
        declared_intensity: inst.declared_intensity,
        streams: inst.streams.clone(),
        ops: inst.ops.clone(),
        elements: inst.elements,
        requires: inst.requires.clone(),
        notes: vec![format!("poly instance layout={}", inst.layout)],
        known_limits: inst.known_limits.clone(),
    }
}

pub fn check(poly: &PolyCase, machine: &Machine, tol: f64) -> Result<PolyVerdict, PolyError> {
    if poly.schema != POLY_SCHEMA {
        return Err(PolyError::BadSchema(poly.schema.clone()));
    }
    if poly.machine_id != machine.id {
        return Err(PolyError::MachineMismatch(
            poly.machine_id.clone(),
            machine.id.clone(),
        ));
    }
    if poly.instances.len() < 2 {
        return Err(PolyError::Message(
            "poly needs ≥2 layout instances — otherwise it is not polymorphism".into(),
        ));
    }

    let mut reasons = Vec::new();
    let mut layouts = Vec::new();
    let mut intensities = Vec::new();
    let mut bytes_by_layout: Vec<(String, f64)> = Vec::new();

    for inst in &poly.instances {
        let ir = instantiate(poly, inst);
        match kernel_check(&ir, machine, tol) {
            Ok(KernelVerdict::Pass { intensity, .. }) => {
                let computed = match &intensity {
                    crate::intensity::IntensityVerdict::Pass { computed, .. } => *computed,
                    crate::intensity::IntensityVerdict::Fail { computed, .. } => *computed,
                };
                layouts.push(inst.layout.clone());
                intensities.push((inst.layout.clone(), computed));
                let bytes: f64 = inst.streams.iter().map(|s| s.bytes).sum();
                bytes_by_layout.push((inst.layout.clone(), bytes));
            }
            Ok(KernelVerdict::FailIntensity { intensity }) => {
                reasons.push(format!(
                    "layout `{}`: intensity FAIL — {:?}",
                    inst.layout, intensity
                ));
            }
            Ok(KernelVerdict::FailCapability { missing }) => {
                reasons.push(format!(
                    "layout `{}`: capability refuse — {}",
                    inst.layout,
                    missing.join("; ")
                ));
            }
            Err(e) => reasons.push(format!("layout `{}`: {e}", inst.layout)),
        }
    }

    // Polymorphism must move the needle: identical byte totals across layouts is a smell.
    if bytes_by_layout.len() >= 2 {
        let b0 = bytes_by_layout[0].1;
        if bytes_by_layout.iter().all(|(_, b)| (*b - b0).abs() < 1e-12) {
            reasons.push(
                "all layouts declare identical Σ stream bytes — format polymorphism is fake; \
                 INT4 vs NF4 must differ in scale/weight traffic"
                    .into(),
            );
        }
    }

    let mut format_cost = None;
    if let Some(claim) = &poly.claim {
        let base = poly
            .instances
            .iter()
            .find(|i| i.layout == claim.baseline_layout);
        let cmp = poly
            .instances
            .iter()
            .find(|i| i.layout == claim.compare_layout);
        match (base, cmp) {
            (Some(b), Some(c)) => match (&b.silicon, &c.silicon) {
                (Some(bs), Some(cs)) => {
                    if bs.kernel_us <= 0.0 {
                        reasons.push("baseline silicon.kernel_us must be > 0".into());
                    } else {
                        let cost = (cs.kernel_us - bs.kernel_us) / bs.kernel_us;
                        format_cost = Some(cost);
                        if cost < claim.expected_min || cost > claim.expected_max {
                            reasons.push(format!(
                                "format cost {cost:.3} outside claimed [{:.3}, {:.3}] \
                                 (baseline {}={:.3}µs, compare {}={:.3}µs)\n  \
                                 options: (1) fix silicon rows to the published bench\n  \
                                          (2) widen the claim band with a known_limit\n  \
                                          (3) drop the claim until re-measured",
                                claim.expected_min,
                                claim.expected_max,
                                claim.baseline_layout,
                                bs.kernel_us,
                                claim.compare_layout,
                                cs.kernel_us
                            ));
                        }
                    }
                }
                _ => reasons.push(format!(
                    "claim needs silicon.kernel_us on both `{}` and `{}`",
                    claim.baseline_layout, claim.compare_layout
                )),
            },
            _ => reasons.push(format!(
                "claim layouts `{}` / `{}` not found in instances",
                claim.baseline_layout, claim.compare_layout
            )),
        }
    }

    if reasons.is_empty() {
        Ok(PolyVerdict::Pass {
            layouts,
            intensities,
            format_cost,
        })
    } else {
        Ok(PolyVerdict::Fail { reasons })
    }
}

pub fn format_verdict(poly: &PolyCase, v: &PolyVerdict) -> String {
    let mut out = format!("poly-check: {} on {}\n", poly.name, poly.machine_id);
    match v {
        PolyVerdict::Pass {
            layouts,
            intensities,
            format_cost,
        } => {
            out.push_str("verdict: PASS — cost re-checked per layout×machine\n");
            out.push_str(&format!("  layouts: {}\n", layouts.join(", ")));
            for (l, i) in intensities {
                out.push_str(&format!("  intensity[{l}] = {i:.4} flop/byte\n"));
            }
            if let Some(c) = format_cost {
                out.push_str(&format!(
                    "  format_cost (wall) = {:.1}% (compare vs baseline)\n",
                    c * 100.0
                ));
            }
        }
        PolyVerdict::Fail { reasons } => {
            out.push_str("verdict: FAIL — layout×machine poly\n");
            for (i, r) in reasons.iter().enumerate() {
                out.push_str(&format!("\n[{}] {r}\n", i + 1));
            }
        }
    }
    out
}

impl crate::document::Document for PolyCase {
    const SCHEMA: &'static str = POLY_SCHEMA;
}

#[cfg(test)]
mod tests {
    use super::*;

    fn machine() -> Machine {
        serde_json::from_str(
            r#"{
              "schema":"lyth-machine/0.1","id":"sm_120","peak_tflops":15.03,
              "levels":[{"name":"dram","bandwidth_gbs":398.39}],
              "ops":[{"name":"fma.f16","status":"present","at":"reg"},
                     {"name":"nvfp4","status":"present","at":"sm"}]
            }"#,
        )
        .unwrap()
    }

    #[test]
    fn int4_vs_nf4_gate_proj() {
        let poly: PolyCase = serde_json::from_str(include_str!(
            "../../../fixtures/poly/gate-proj-int4-nf4.json"
        ))
        .unwrap();
        match check(&poly, &machine(), 0.05).unwrap() {
            PolyVerdict::Pass {
                format_cost: Some(c),
                ..
            } => {
                assert!(c > 0.19 && c < 0.31, "cost={c}");
            }
            other => panic!("{other:?}"),
        }
    }
}
