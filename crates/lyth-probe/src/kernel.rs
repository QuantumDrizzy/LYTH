//! Kernel IR — memory-first AST the future `.lyth` frontend must emit.
//!
//! Streams declare traffic. Ops declare FLOPs at a level. `kernel-check`
//! lowers to intensity accounting and refuses missing machine ops.

use serde::Deserialize;
use thiserror::Error;

use crate::intensity::{check_with_machine, BodyAccounting, IntensityCase, IntensityVerdict, Move};
use crate::machine::Machine;

pub const KERNEL_IR_SCHEMA: &str = "lyth-kernel-ir/0.1";

#[derive(Debug, Clone, Deserialize)]
pub struct KernelIr {
    pub schema: String,
    pub name: String,
    pub machine_id: String,
    pub declared_intensity: f64,
    pub streams: Vec<Stream>,
    pub ops: Vec<ArithOp>,
    /// Elements one launch processes, carried through lowering so `--ncu` can scale the
    /// per-element stream accounting to the total `ncu` reports.
    #[serde(default)]
    pub elements: Option<f64>,
    #[serde(default)]
    pub requires: Vec<CapabilityReq>,
    #[serde(default)]
    pub notes: Vec<String>,
    #[serde(default)]
    pub known_limits: Vec<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Stream {
    pub name: String,
    /// Source level (dram, l2, smem, reg).
    pub from: String,
    /// Destination level.
    pub to: String,
    pub bytes: f64,
    #[serde(default = "default_dir")]
    pub dir: String,
    #[serde(default)]
    pub via: Option<String>,
    #[serde(default)]
    pub once: bool,
}

fn default_dir() -> String {
    "r".into()
}

#[derive(Debug, Clone, Deserialize)]
pub struct ArithOp {
    pub at: String,
    pub flops: f64,
    #[serde(default)]
    pub note: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct CapabilityReq {
    pub op: String,
    #[serde(default)]
    pub at: Option<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum KernelVerdict {
    Pass {
        intensity: IntensityVerdict,
        checked_caps: Vec<String>,
    },
    FailIntensity {
        intensity: IntensityVerdict,
    },
    FailCapability {
        missing: Vec<String>,
    },
}

#[derive(Debug, Error)]
pub enum KernelError {
    #[error("schema must be `{KERNEL_IR_SCHEMA}`, got `{0}`")]
    BadSchema(String),
    #[error("kernel.machine_id `{0}` != machine.id `{1}`")]
    MachineMismatch(String, String),
    #[error("{0}")]
    Message(String),
}

/// Lower streams + ops into the intensity body the checker already understands.
pub fn lower(ir: &KernelIr) -> Result<IntensityCase, KernelError> {
    if ir.schema != KERNEL_IR_SCHEMA {
        return Err(KernelError::BadSchema(ir.schema.clone()));
    }
    if ir.streams.is_empty() {
        return Err(KernelError::Message(
            "kernel has no streams — memory-first IR cannot be empty".into(),
        ));
    }
    for s in &ir.streams {
        if s.bytes <= 0.0 {
            return Err(KernelError::Message(format!(
                "stream `{}` bytes must be > 0",
                s.name
            )));
        }
        if s.from == s.to && s.dir != "resident" {
            // resident same-level is ok tagged; otherwise nonsense
        }
    }
    let flops: f64 = ir.ops.iter().map(|o| o.flops).sum();
    if flops < 0.0 {
        return Err(KernelError::Message("ops flops sum must be ≥ 0".into()));
    }
    let flop_note = ir
        .ops
        .iter()
        .map(|o| {
            if o.note.is_empty() {
                format!("{}@{}", o.flops, o.at)
            } else {
                format!("{}@{} ({})", o.flops, o.at, o.note)
            }
        })
        .collect::<Vec<_>>()
        .join("; ");

    let moves: Vec<Move> = ir
        .streams
        .iter()
        .map(|s| Move {
            // Charge traffic to the deeper (farther from reg) endpoint for intensity.
            name: if s.via.is_some() {
                format!("{} via {}", s.name, s.via.as_deref().unwrap_or(""))
            } else {
                s.name.clone()
            },
            level: deeper_level(&s.from, &s.to).to_string(),
            bytes: s.bytes,
            dir: s.dir.clone(),
        })
        .collect();

    Ok(IntensityCase {
        schema: crate::intensity::INTENSITY_SCHEMA.to_string(),
        kernel: ir.name.clone(),
        declared_intensity: ir.declared_intensity,
        unit: "flops_per_byte".into(),
        body: BodyAccounting {
            moves,
            flops,
            flop_note,
        },
        elements: ir.elements,
        machine_id: Some(ir.machine_id.clone()),
        notes: ir.notes.clone(),
        known_limits: ir.known_limits.clone(),
    })
}

fn deeper_level<'a>(a: &'a str, b: &'a str) -> &'a str {
    const ORDER: &[&str] = &["dram", "l2", "smem", "reg"];
    let ia = ORDER.iter().position(|&x| x == a).unwrap_or(0);
    let ib = ORDER.iter().position(|&x| x == b).unwrap_or(0);
    if ia <= ib {
        a
    } else {
        b
    }
}

pub fn check_capabilities(ir: &KernelIr, machine: &Machine) -> Result<Vec<String>, Vec<String>> {
    let mut ok = Vec::new();
    let mut missing = Vec::new();
    for req in &ir.requires {
        let found = machine.ops.iter().any(|op| {
            op.name == req.op
                && op.status == "present"
                && req
                    .at
                    .as_ref()
                    .map(|at| op.at.as_ref().map(|a| a == at).unwrap_or(false))
                    .unwrap_or(true)
        });
        // Also allow ops that live at "sm" when req doesn't pin at.
        let found = found
            || machine
                .ops
                .iter()
                .any(|op| op.name == req.op && op.status == "present" && req.at.is_none());
        if found {
            ok.push(req.op.clone());
        } else {
            let at = req.at.as_deref().unwrap_or("any");
            missing.push(format!(
                "op `{}` at `{at}` required by kernel `{}` but machine `{}` lacks it (or status!=present)\n  \
                 options: (1) remove the require / rewrite the stream that needs it\n  \
                          (2) retarget a machine file where `{}` is present",
                req.op, ir.name, machine.id, req.op
            ));
        }
    }
    if missing.is_empty() {
        Ok(ok)
    } else {
        Err(missing)
    }
}

pub fn check(ir: &KernelIr, machine: &Machine, tol: f64) -> Result<KernelVerdict, KernelError> {
    if ir.machine_id != machine.id {
        return Err(KernelError::MachineMismatch(
            ir.machine_id.clone(),
            machine.id.clone(),
        ));
    }
    match check_capabilities(ir, machine) {
        Err(missing) => Ok(KernelVerdict::FailCapability { missing }),
        Ok(checked_caps) => {
            let intensity_case = lower(ir)?;
            let intensity = check_with_machine(&intensity_case, Some(machine), tol)
                .map_err(|e| KernelError::Message(e.to_string()))?;
            match &intensity {
                IntensityVerdict::Pass { .. } => Ok(KernelVerdict::Pass {
                    intensity,
                    checked_caps,
                }),
                IntensityVerdict::Fail { .. } => Ok(KernelVerdict::FailIntensity { intensity }),
            }
        }
    }
}

pub fn format_verdict(ir: &KernelIr, v: &KernelVerdict) -> String {
    let mut out = format!("kernel-check: {} on {}\n", ir.name, ir.machine_id);
    match v {
        KernelVerdict::Pass {
            intensity,
            checked_caps,
        } => {
            out.push_str("verdict: PASS\n");
            if !checked_caps.is_empty() {
                out.push_str(&format!("  caps: {}\n", checked_caps.join(", ")));
            }
            out.push_str(&format_intensity_brief(intensity));
        }
        KernelVerdict::FailIntensity { intensity } => {
            out.push_str("verdict: FAIL — intensity (streams/ops vs declaration)\n");
            out.push_str(&format_intensity_brief(intensity));
        }
        KernelVerdict::FailCapability { missing } => {
            out.push_str("verdict: FAIL — capability refuse\n");
            for (i, m) in missing.iter().enumerate() {
                out.push_str(&format!("\n[{}] {m}\n", i + 1));
            }
        }
    }
    out
}

fn format_intensity_brief(v: &IntensityVerdict) -> String {
    match v {
        IntensityVerdict::Pass {
            computed,
            declared,
            bytes,
            flops,
            ridge,
            regime,
        } => {
            let mut s = format!(
                "  intensity: declared {declared:.4} ≈ computed {computed:.4} flop/byte\n  \
                 body: {flops} FLOPs / {bytes} bytes  regime={regime:?}\n"
            );
            if let Some(r) = ridge {
                s.push_str(&format!(
                    "  ridge({}): {:.1} flop/byte\n",
                    r.machine_id, r.ridge_flops_per_byte
                ));
            }
            s
        }
        IntensityVerdict::Fail {
            computed,
            declared,
            ridge,
            regime,
            hints,
            ..
        } => {
            let mut s = format!(
                "  intensity: declared {declared:.4} vs computed {computed:.4}  regime={regime:?}\n"
            );
            if let Some(r) = ridge {
                s.push_str(&format!(
                    "  ridge({}): {:.1} flop/byte\n",
                    r.machine_id, r.ridge_flops_per_byte
                ));
            }
            for (i, h) in hints.iter().enumerate() {
                s.push_str(&format!("  hint[{}]: {h}\n", i + 1));
            }
            s
        }
    }
}

impl crate::document::Document for KernelIr {
    const SCHEMA: &'static str = KERNEL_IR_SCHEMA;
}

#[cfg(test)]
mod tests {
    use super::*;

    fn machine_sm120() -> Machine {
        serde_json::from_str(
            r#"{
              "schema":"lyth-machine/0.1","id":"sm_120","peak_tflops":15.03,
              "levels":[{"name":"dram","bandwidth_gbs":398.39}],
              "ops":[{"name":"fma.f32","status":"present","at":"reg"},
                     {"name":"tma","status":"absent","at":"sm"},
                     {"name":"wgmma","status":"absent","at":"sm"}]
            }"#,
        )
        .unwrap()
    }

    #[test]
    fn integrate_lowers_and_passes() {
        let ir: KernelIr = serde_json::from_str(
            r#"{
              "schema":"lyth-kernel-ir/0.1",
              "name":"k_integrate",
              "machine_id":"sm_120",
              "declared_intensity":0.2069,
              "streams":[
                {"name":"ring","from":"dram","to":"reg","bytes":8,"dir":"rw"},
                {"name":"v","from":"dram","to":"reg","bytes":8,"dir":"rw"},
                {"name":"adapt","from":"dram","to":"reg","bytes":8,"dir":"rw"},
                {"name":"refrac","from":"dram","to":"reg","bytes":4,"dir":"rw"},
                {"name":"is_stim","from":"dram","to":"reg","bytes":1,"dir":"r"}
              ],
              "ops":[{"at":"reg","flops":6.0}],
              "requires":[{"op":"fma.f32","at":"reg"}]
            }"#,
        )
        .unwrap();
        match check(&ir, &machine_sm120(), 0.05).unwrap() {
            KernelVerdict::Pass { .. } => {}
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn missing_tma_refuses() {
        let ir: KernelIr = serde_json::from_str(
            r#"{
              "schema":"lyth-kernel-ir/0.1",
              "name":"bad",
              "machine_id":"sm_120",
              "declared_intensity":1.0,
              "streams":[{"name":"W","from":"dram","to":"smem","bytes":128,"via":"tma"}],
              "ops":[{"at":"reg","flops":128.0}],
              "requires":[{"op":"tma"}]
            }"#,
        )
        .unwrap();
        match check(&ir, &machine_sm120(), 0.05).unwrap() {
            KernelVerdict::FailCapability { missing } => assert!(!missing.is_empty()),
            other => panic!("{other:?}"),
        }
    }
}
