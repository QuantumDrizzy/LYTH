//! IEEE 754-2008 binary16 and bfloat16, round-half-to-even, written here.
//!
//! ADR-0024 decision 3. Rust's `f16` is unstable on this toolchain — checked rather than
//! recalled: `error[E0658]: the type f16 is unstable`, rustc 1.91.1 — and `lyth-lang` has
//! exactly one dependency. But the dependency count is the smaller reason.
//!
//! **The oracle models what the device will do, not what it should.** These functions exist to
//! reproduce `cvt.rn.f16.f32` and `cvt.rn.bf16.f32` bit for bit, and a crate that is *correct*
//! is not automatically a crate that *agrees with this GPU*. `tests/half_rounding.rs` checks
//! them against the device over the cases that separate implementations — ties, subnormals, the
//! overflow at 65504, and the values either side of each — and it runs before any kernel does.
//! An oracle that rounds a hair differently produces a mismatch that is nobody's bug.
//!
//! Both types are two bytes and the traffic model cannot tell them apart (ADR-0024). They round
//! very differently, which is the whole point of having both:
//!
//! | | exponent | mantissa | largest finite |
//! |---|---|---|---|
//! | `f32` | 8 bits | 23 bits | 3.4e38 |
//! | `binary16` | 5 bits | 10 bits | **65504** |
//! | `bfloat16` | 8 bits | 7 bits | 3.39e38 |

/// `f32` to binary16 bits, round-to-nearest-even. What `cvt.rn.f16.f32` does.
///
/// Written as integer bit manipulation rather than as arithmetic, because the subnormal and
/// tie cases are where a float-based version quietly differs and they are exactly the cases
/// the device test covers.
pub fn f32_to_f16_bits(x: f32) -> u16 {
    let b = x.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let exp = ((b >> 23) & 0xff) as i32;
    let mant = b & 0x007f_ffff;

    // NaN and infinity keep their class. A signalling NaN is quieted, which is what the
    // hardware conversion does; the payload's low bits do not survive 23 -> 10 anyway.
    if exp == 0xff {
        return if mant != 0 {
            sign | 0x7e00
        } else {
            sign | 0x7c00
        };
    }

    // Rebias: 127 for binary32, 15 for binary16.
    let e = exp - 127 + 15;

    if e >= 0x1f {
        // Overflow to infinity. 65504 is the largest finite binary16 and anything that rounds
        // above it goes to infinity rather than saturating -- a difference that matters to a
        // reduction and is one of the cases the device test pins.
        return sign | 0x7c00;
    }

    if e <= 0 {
        // Subnormal, or under it. The implicit leading 1 becomes explicit and the whole
        // significand is shifted right by `1 - e`; past 24 bits of shift nothing survives.
        if e < -10 {
            return sign;
        }
        let m = mant | 0x0080_0000;
        let shift = (14 - e) as u32; // 23 - 10 + (1 - e)
        let half = 1u32 << (shift - 1);
        let low = m & ((1u32 << shift) - 1);
        let mut out = m >> shift;
        // Round half to even: round up on more than half, or on exactly half with an odd
        // result. The `out & 1` is the "even" in round-half-to-even and is the single line a
        // truncating implementation is missing.
        if low > half || (low == half && (out & 1) == 1) {
            out += 1;
        }
        return sign | out as u16;
    }

    // Normal. Drop 13 bits of mantissa with the same rule.
    let half = 0x0000_1000;
    let low = mant & 0x0000_1fff;
    let mut out = ((e as u32) << 10) | (mant >> 13);
    if low > half || (low == half && (out & 1) == 1) {
        // Carrying into the exponent is correct and needs no special case: the mantissa
        // overflowing into bit 10 is exactly an exponent increment, and if that reaches 0x1f
        // the result is infinity, which is also what should happen.
        out += 1;
    }
    sign | out as u16
}

/// binary16 bits back to `f32`. Exact — every binary16 is representable in `f32`.
pub fn f16_bits_to_f32(h: u16) -> f32 {
    let sign = ((h & 0x8000) as u32) << 16;
    let exp = ((h >> 10) & 0x1f) as u32;
    let mant = (h & 0x03ff) as u32;

    if exp == 0 {
        if mant == 0 {
            return f32::from_bits(sign);
        }
        // Subnormal in binary16 is **normal** in f32, so it has to be renormalised rather than
        // shifted by a fixed amount.
        //
        // The value is `mant * 2^-24`. Writing `mant = 2^k * 1.f` with `k` the position of its
        // leading one, that is `2^(k-24) * 1.f`, so the biased f32 exponent is `k - 24 + 127`
        // and the significand is `mant` left-aligned with its leading one dropped.
        //
        // The first version of this shifted by `leading_zeros - 21` and produced `2^-25` for
        // the smallest subnormal -- a value binary16 cannot represent at all. The subnormal
        // test is what caught it, which is the reason it exists.
        let k = 31 - mant.leading_zeros();
        let e = k + 103;
        let m = (mant << (23 - k)) & 0x007f_ffff;
        return f32::from_bits(sign | (e << 23) | m);
    }
    if exp == 0x1f {
        return f32::from_bits(sign | 0x7f80_0000 | (mant << 13));
    }
    f32::from_bits(sign | ((exp + 127 - 15) << 23) | (mant << 13))
}

/// `f32` through binary16 and back: the value the device will actually hold.
pub fn round_f16(x: f32) -> f32 {
    f16_bits_to_f32(f32_to_f16_bits(x))
}

/// `f32` to bfloat16 bits, round-to-nearest-even. What `cvt.rn.bf16.f32` does.
///
/// bfloat16 is `f32` with 16 mantissa bits deleted, so this is a rounding shift and nothing
/// else: same exponent field, same bias, same infinities. **That is the reason it exists** --
/// no overflow at 65504, no subnormal cliff, and 8 fewer mantissa bits to pay for it.
pub fn f32_to_bf16_bits(x: f32) -> u16 {
    let b = x.to_bits();
    if (b & 0x7f80_0000) == 0x7f80_0000 && (b & 0x007f_ffff) != 0 {
        // NaN. Quieting explicitly rather than letting the rounding carry produce an infinity
        // out of a NaN whose low bits happen to be set.
        return ((b >> 16) as u16) | 0x0040;
    }
    let low = b & 0xffff;
    let mut out = b >> 16;
    if low > 0x8000 || (low == 0x8000 && (out & 1) == 1) {
        out += 1;
    }
    out as u16
}

/// bfloat16 bits back to `f32`. Exact, and a plain shift.
pub fn bf16_bits_to_f32(h: u16) -> f32 {
    f32::from_bits((h as u32) << 16)
}

/// `f32` through bfloat16 and back.
pub fn round_bf16(x: f32) -> f32 {
    bf16_bits_to_f32(f32_to_bf16_bits(x))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_values_survive_a_round_trip() {
        for v in [0.0f32, 1.0, -1.0, 0.5, 2.0, -0.25, 1024.0, 65504.0] {
            assert_eq!(round_f16(v), v, "{v} is representable in binary16");
            assert_eq!(round_bf16(f32::from_bits(v.to_bits() & 0xffff_0000)),
                       f32::from_bits(v.to_bits() & 0xffff_0000));
        }
    }

    #[test]
    fn ties_round_to_even_not_up() {
        // The values that separate round-half-to-even from round-half-up. Between 2048 and
        // 4096 binary16 has a step of 2, so 2049 is exactly a tie.
        assert_eq!(round_f16(2049.0), 2048.0, "tie down to the even neighbour");
        assert_eq!(round_f16(2051.0), 2052.0, "tie up to the even neighbour");
        assert_eq!(round_f16(2050.0), 2050.0, "not a tie at all");
    }

    #[test]
    fn overflow_goes_to_infinity_rather_than_saturating() {
        // 65504 is the largest finite binary16 and 65520 is the midpoint to the next
        // representable value, which does not exist -- so it rounds to infinity.
        assert_eq!(round_f16(65504.0), 65504.0);
        assert!(round_f16(65520.0).is_infinite());
        assert!(round_f16(70000.0).is_infinite());
        // bf16 keeps f32's range, so the same value is ordinary there. This is the difference
        // the traffic model cannot see (ADR-0024).
        assert!(round_bf16(70000.0).is_finite());
        assert!(round_bf16(3.0e38).is_finite());
    }

    #[test]
    fn subnormals_are_not_flushed_to_zero() {
        // The smallest binary16 subnormal is 2^-24. Flushing to zero here would be a different
        // function and a common shortcut.
        let tiny = 2.0f32.powi(-24);
        assert_eq!(round_f16(tiny), tiny);
        assert_eq!(round_f16(tiny * 0.4), 0.0, "below half the smallest step");
        assert_eq!(round_f16(tiny * 0.6), tiny, "above it");
    }

    #[test]
    fn sign_survives_zero() {
        assert!(round_f16(-0.0).is_sign_negative(), "ADR-0013 cares about this");
        assert!(round_bf16(-0.0).is_sign_negative());
    }
}
