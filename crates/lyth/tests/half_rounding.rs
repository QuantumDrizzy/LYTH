//! The oracle's rounding against the device's, bit for bit. ADR-0024 step 2, decision 3.
//!
//! > The oracle models what the device **will** do, not what it **should**.
//!
//! `tools/half_probe.py` runs `cvt.rn.f16.f32` and `cvt.rn.bf16.f32` -- the exact instructions
//! the emitter will write, not an intrinsic that ought to lower to them -- over 138 vectors
//! chosen to be the cases where implementations diverge, checks the device against numpy, and
//! writes what the device produced to `fixtures/half_vectors.json`.
//!
//! This checks `lyth_lang::half` against that file. Every comparison is on raw bits: a rounding
//! that is *nearly* right is a rounding that produces a mismatch nobody can attribute, in a
//! benchmark that then measures the disagreement instead of the kernel.
//!
//! The fixture is checked in, the same way `fixtures/machine/sm_120.json` is, and for the same
//! reason: it is a measurement, and the file is the evidence. Regenerate it with the probe if
//! the device or the toolchain changes.

use std::path::PathBuf;

use lyth_lang::half::{f32_to_bf16_bits, f32_to_f16_bits};

fn fixture() -> serde_json::Value {
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/half_vectors.json");
    let text = std::fs::read_to_string(&p)
        .unwrap_or_else(|e| panic!("{}: {e}\nRun `python tools/half_probe.py`.", p.display()));
    serde_json::from_str(&text).expect("the probe writes valid JSON")
}

/// A NaN's payload does not survive 23 mantissa bits shrinking to 10, and nothing in this
/// language depends on which quiet NaN comes back -- only that it is one. Every other value is
/// compared exactly.
fn nan_class(bits: u16, mask_exp: u16, mask_mant: u16) -> bool {
    bits & mask_exp == mask_exp && bits & mask_mant != 0
}

#[test]
fn the_oracle_rounds_to_binary16_exactly_as_the_device_does() {
    let f = fixture();
    let vectors = f["vectors"].as_array().expect("vectors");
    assert!(vectors.len() > 100, "the probe should write a real set");

    let mut checked = 0;
    for v in vectors {
        let x = f32::from_bits(v["f32"].as_u64().unwrap() as u32);
        let want = v["f16"].as_u64().unwrap() as u16;
        let got = f32_to_f16_bits(x);
        if x.is_nan() {
            assert!(
                nan_class(got, 0x7c00, 0x03ff) && nan_class(want, 0x7c00, 0x03ff),
                "{x} should stay a NaN: oracle 0x{got:04x}, device 0x{want:04x}"
            );
        } else {
            assert_eq!(
                got, want,
                "f32 {x:?} (0x{:08x}): oracle gave 0x{got:04x}, device gave 0x{want:04x}",
                x.to_bits()
            );
        }
        checked += 1;
    }
    eprintln!("  {checked} vectors, oracle == device, bit for bit");
}

#[test]
fn the_oracle_rounds_to_bfloat16_exactly_as_the_device_does() {
    // numpy has no bfloat16, so this column has no third opinion: the device is the reference
    // and that is stated rather than quietly assumed. It is also the column where the range
    // differs -- 70000 and 3e38 are finite here and infinite in binary16.
    let f = fixture();
    for v in f["vectors"].as_array().unwrap() {
        let x = f32::from_bits(v["f32"].as_u64().unwrap() as u32);
        let want = v["bf16"].as_u64().unwrap() as u16;
        let got = f32_to_bf16_bits(x);
        if x.is_nan() {
            assert!(
                nan_class(got, 0x7f80, 0x007f) && nan_class(want, 0x7f80, 0x007f),
                "{x} should stay a NaN: oracle 0x{got:04x}, device 0x{want:04x}"
            );
        } else {
            assert_eq!(
                got, want,
                "f32 {x:?} (0x{:08x}): oracle gave 0x{got:04x}, device gave 0x{want:04x}",
                x.to_bits()
            );
        }
    }
}

#[test]
fn the_fixture_actually_contains_the_hard_cases() {
    // A rounding test that passed against a truncating implementation would be worse than no
    // test, and a fixture of ordinary floats is exactly that: a uniform sample essentially
    // never lands on a tie. So the fixture is checked for the cases that separate
    // implementations, rather than trusted to contain them.
    let f = fixture();
    let xs: Vec<f32> = f["vectors"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| f32::from_bits(v["f32"].as_u64().unwrap() as u32))
        .collect();

    let tie = xs.iter().any(|&x| x == 2049.0 || x == 2051.0);
    let overflow = xs.iter().any(|&x| x.is_finite() && x.abs() > 65504.0);
    let subnormal = xs.iter().any(|&x| x != 0.0 && x.abs() < 2.0f32.powi(-14));
    let nan = xs.iter().any(|x| x.is_nan());
    let neg_zero = xs.iter().any(|&x| x == 0.0 && x.is_sign_negative());

    assert!(tie, "no binary16 tie: round-half-to-even is untested");
    assert!(overflow, "nothing above 65504: the overflow-to-infinity path is untested");
    assert!(subnormal, "no subnormal: flush-to-zero would pass");
    assert!(nan, "no NaN");
    assert!(neg_zero, "no negative zero (ADR-0013 cares)");
}
