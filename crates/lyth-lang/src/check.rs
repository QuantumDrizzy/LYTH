//! The declared intensity against the derived one.
//!
//! This is the refusal ADR-0001 promised and could not deliver until the body was code. The
//! error names the machine's ridge and says what fraction of peak the kernel is asking for,
//! because "declared 2.0, computed 0.5" tells you that you were wrong and nothing about what
//! to write instead.

use crate::ir::KernelIr;

/// What a machine file contributes to the check. Kept as a plain struct so `lyth-lang` does
/// not depend on `lyth-probe`: the front end has no business knowing how machines are stored.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Ridge {
    pub peak_tflops: f64,
    pub bandwidth_gbs: f64,
}

impl Ridge {
    /// FLOPs per byte at which compute and memory are balanced on this machine.
    pub fn flops_per_byte(&self) -> f64 {
        self.peak_tflops * 1e3 / self.bandwidth_gbs
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Regime {
    MemoryBound,
    NearRidge,
    ComputeBound,
}

impl Regime {
    pub fn of(intensity: f64, ridge: f64) -> Self {
        let r = intensity / ridge;
        if r < 0.5 {
            Regime::MemoryBound
        } else if r <= 2.0 {
            Regime::NearRidge
        } else {
            Regime::ComputeBound
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Regime::MemoryBound => "memory-bound",
            Regime::NearRidge => "near the ridge",
            Regime::ComputeBound => "compute-bound",
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct IntensityReport {
    /// For a contracted kernel this is the **limit** the intensity approaches, and
    /// `asymptotic` says so. See ADR-0018.
    pub derived: f64,
    pub declared: Option<f64>,
    /// `None` when the traffic is a function of a launch extent rather than a constant.
    pub bytes: Option<f64>,
    pub flops: Option<f64>,
    /// `Some("0.25 * k + 4")` when the traffic is an expression.
    pub bytes_expr: Option<String>,
    pub flops_expr: Option<String>,
    /// Whether `derived` is a limit rather than an exact figure.
    pub asymptotic: bool,
    pub ridge: Option<f64>,
    pub regime: Option<Regime>,
    /// Fraction of the machine's peak FLOPS and peak bandwidth this kernel asks for, when a
    /// machine is known.
    pub peak_flops_fraction: Option<f64>,
    pub peak_bandwidth_fraction: Option<f64>,
}

#[derive(Debug, thiserror::Error)]
#[error("{0}")]
pub struct IntensityMismatch(pub String);

/// Check a lowered kernel's declared intensity against the derived one.
///
/// `tol` is relative. A kernel that declares nothing is not refused — the derived number is
/// reported and the source is simply silent about it.
pub fn check_intensity(
    ir: &KernelIr,
    declared: Option<f64>,
    ridge: Option<Ridge>,
    tol: f64,
) -> Result<IntensityReport, IntensityMismatch> {
    let derived = ir.cost.intensity;
    let ridge_fpb = ridge.map(|r| r.flops_per_byte());
    let report = IntensityReport {
        derived,
        declared,
        bytes: ir.cost.bytes_per_element(),
        flops: ir.cost.flops_per_element(),
        bytes_expr: ir.cost.bytes_expr(),
        flops_expr: ir.cost.flops_expr(),
        asymptotic: ir.cost.asymptotic,
        ridge: ridge_fpb,
        regime: ridge_fpb.map(|r| Regime::of(derived, r)),
        peak_flops_fraction: ridge.map(|r| {
            // At this intensity the kernel is limited by bandwidth; the flops it can retire
            // per second is bandwidth x intensity, as a fraction of peak.
            (r.bandwidth_gbs * derived) / (r.peak_tflops * 1e3)
        }),
        // A memory-bound kernel saturates bandwidth by definition; report it as such rather
        // than pretending a number was measured.
        peak_bandwidth_fraction: ridge.map(|_| 1.0),
    };

    let Some(declared) = declared else {
        return Ok(report);
    };
    let rel = if declared == 0.0 {
        if derived == 0.0 {
            0.0
        } else {
            f64::INFINITY
        }
    } else {
        (derived - declared).abs() / declared
    };
    if rel <= tol {
        return Ok(report);
    }

    // A contracted kernel is told its traffic as the expression it is. Printing the constant
    // part -- 4 bytes for a matmul -- beside a mismatch would be advice that is wrong by a
    // factor of the contracted extent, in the one message the author is reading closely.
    let mut msg = match (ir.cost.bytes_expr(), ir.cost.flops_expr()) {
        (Some(b), Some(f)) => format!(
            "declares {declared} flop/byte asymptotic, body approaches {derived:.4}\n  \
             bytes moved: {b} per element\n  \
             flops:       {f} per element",
        ),
        _ => format!(
            "declares {declared} flop/byte, body computes {derived:.4}\n  \
             bytes moved: {bytes} per element ({read} read + {write} written)\n  \
             flops:       {flops} per element",
            bytes = ir.cost.bytes_fixed(),
            read = ir.cost.read_bytes_per_element(),
            write = ir.cost.write_bytes_per_element(),
            flops = ir.cost.flops_per_element().unwrap_or(0.0),
        ),
    };
    if let (Some(r), Some(regime)) = (ridge_fpb, report.regime) {
        msg.push_str(&format!(
            "\n  machine {machine} ridge is {r:.1} flop/byte, so {derived:.4} is {name}",
            machine = ir.machine,
            name = regime.name(),
        ));
        if let Some(f) = report.peak_flops_fraction {
            msg.push_str(&format!(
                "\n  at this intensity the kernel can reach {:.2}% of peak FLOPS \
                 while saturating bandwidth",
                f * 100.0
            ));
        }
    }
    msg.push_str(&format!("\n  did you mean to declare {derived:.4}?"));
    Err(IntensityMismatch(msg))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ir::lower, parse::parse};

    fn saxpy_ir() -> KernelIr {
        let src = "machine sm_120\n\nkernel saxpy(n: u32, a: f32, x: [f32; n], y: [f32; n])\n    stream x : dram -> reg\n    stream y : dram -> reg, drain\n    at reg:
        y = a * x + y\n";
        let u = parse(src).unwrap();
        lower(&u, &u.kernels[0]).unwrap()
    }

    // Re-measured 2026-09-15: 15.30 TFLOP/s achieved SGEMM over 414.51 GB/s, a ridge of 36.9.
    //
    // The bandwidth was 358.43 here until the probe that produced it was found to be timing a
    // host round trip as memory traffic -- `float(x.sum())` inside the timed loop, worth 10.4%
    // of the figure. Every `% of peak` in the project inherited it, which is how `saxpy` came
    // to report 112% of a peak it cannot exceed. See ADR-0004.
    const SM_120: Ridge = Ridge {
        peak_tflops: 15.30,
        bandwidth_gbs: 414.51,
    };

    #[test]
    fn an_honest_declaration_passes() {
        let r = check_intensity(&saxpy_ir(), Some(0.1667), Some(SM_120), 0.05).unwrap();
        assert!((r.derived - 0.16666).abs() < 1e-4);
        assert_eq!(r.regime, Some(Regime::MemoryBound));
    }

    #[test]
    fn a_lie_is_refused_and_the_message_says_what_to_write() {
        let e = check_intensity(&saxpy_ir(), Some(2.0), Some(SM_120), 0.05).unwrap_err();
        let s = e.to_string();
        assert!(s.contains("body computes 0.1667"), "{s}");
        assert!(s.contains("12 per element"), "{s}");
        assert!(s.contains("ridge"), "{s}");
        assert!(s.contains("memory-bound"), "{s}");
        assert!(s.contains("did you mean to declare 0.1667"), "{s}");
    }

    #[test]
    fn declaring_nothing_is_allowed_and_still_derives() {
        let r = check_intensity(&saxpy_ir(), None, Some(SM_120), 0.05).unwrap();
        assert_eq!(r.declared, None);
        assert!(r.derived > 0.0);
    }

    #[test]
    fn the_ridge_comes_out_of_the_machine_not_a_constant() {
        // 15.30e3 / 414.51. It was 42.88 until the bandwidth was re-measured; a ridge is a
        // quotient of two measurements and moves when either of them is corrected, which is
        // exactly why it is not a constant.
        assert!(
            (SM_120.flops_per_byte() - 36.91).abs() < 0.1,
            "{}",
            SM_120.flops_per_byte()
        );
    }
}
