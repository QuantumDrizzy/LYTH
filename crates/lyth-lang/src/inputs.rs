//! The deterministic inputs every backend has to agree about.
//!
//! This was inline in `lyth run` and nowhere else, which was fine while there was one backend.
//! ADR-0025 adds a second, and the moment a `.uasm` program bakes its own data into `.data`
//! the generator has to be **one definition**: the host oracle and the emitted program must
//! start from the same numbers or every comparison between them is a comparison of two
//! different functions.
//!
//! ADR-0019 is why this is a module rather than a copied loop. A rule that lived in two places
//! there — the launch shape, stated by `cmd_run` and again by the manifest — drifted the moment
//! one was corrected, and every generated binding published a figure that had already been
//! measured at half the achievable bandwidth. The fix belongs where the rule lives.

use crate::ast::Ty;

/// A buffer's seed, from its name. Two buffers in one kernel get different data, and the same
/// buffer gets the same data in every run and on every backend.
pub fn seed_of(name: &str) -> u32 {
    name.bytes()
        .fold(1u32, |a, b| a.wrapping_mul(31).wrapping_add(b as u32))
}

/// Element `i` of the buffer called `name`, at the width `ty` stores.
///
/// Knuth's multiplicative hash, shifted to drop the low bits that cycle, taken modulo a prime
/// and mapped to roughly `[-4, 4)`. The range matters: it keeps a `sum` over a large `n` away
/// from `f16`'s overflow at 65504 while still exercising the sign, and it is small enough that
/// a product does not immediately saturate.
///
/// **Rounded to the buffer's own width here**, once, so the host's copy and the device's are
/// the same numbers (ADR-0024). An oracle evaluating unrounded inputs would be computing on
/// values the kernel never saw.
pub fn element(name: &str, ty: Ty, i: u32) -> f32 {
    // The seed is **mixed**, not added, and that is a correction rather than a flourish.
    //
    // This generator used to be `(i * C + seed) >> 8`. `seed_of` gives adjacent integers to
    // short names -- 151 for `x`, 152 for `y`, 128 and 129 for a matmul's `a` and `b` -- and
    // the shift discards the low eight bits, so seeds a few apart were **indistinguishable**:
    // `x` and `y` agreed on 4080 of 4096 elements, 99.6%.
    //
    // Every bit-exactness check in this project therefore ran with its two input buffers
    // holding almost the same numbers, and a kernel that read the wrong pointer, or swapped
    // two operands, would have passed them. Multiplying the seed by a large odd constant
    // spreads it into the bits the shift keeps.
    let h = (i.wrapping_mul(2654435761) ^ seed_of(name).wrapping_mul(0x85EB_CA6B)) >> 8;
    let v = (h % 2003) as f32 / 251.0 - 4.0;
    match ty {
        Ty::BufF16 => crate::half::round_f16(v),
        Ty::BufBF16 => crate::half::round_bf16(v),
        _ => v,
    }
}

/// `len` elements of that buffer.
pub fn buffer(name: &str, ty: Ty, len: u32) -> Vec<f32> {
    (0..len).map(|i| element(name, ty, i)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn two_buffers_in_one_kernel_get_different_data() {
        // The test that found the bug. `x` and `y` used to agree on 99.6% of elements, because
        // their seeds differ by one and the generator shifted the difference away -- so a
        // kernel that read the wrong pointer would have passed every bit-exactness check in
        // the project.
        //
        // Asserted as a *rate*, not as `!=`: two buffers that differ in one element out of
        // 4096 are two buffers that will hide the same bug.
        for (a, b) in [("x", "y"), ("a", "b"), ("b", "c")] {
            let xa = buffer(a, Ty::BufF32, 4096);
            let xb = buffer(b, Ty::BufF32, 4096);
            let same = xa.iter().zip(&xb).filter(|(p, q)| p == q).count();
            assert!(
                same < 4096 / 100,
                "`{a}` and `{b}` agree on {same} of 4096 elements;                  a kernel reading the wrong one would pass"
            );
        }
    }

    #[test]
    fn the_same_buffer_is_the_same_everywhere() {
        assert_eq!(buffer("x", Ty::BufF32, 8), buffer("x", Ty::BufF32, 8));
        assert_eq!(element("x", Ty::BufF32, 5), buffer("x", Ty::BufF32, 8)[5]);
    }

    #[test]
    fn a_narrow_buffer_is_rounded_at_the_source() {
        // Not afterwards. If the generator handed out values the width cannot hold, the host
        // would evaluate one function and the device another, and the difference would be
        // blamed on the emitter.
        for i in 0..64 {
            let v = element("x", Ty::BufF16, i);
            assert_eq!(v, crate::half::round_f16(v), "element {i} is not f16-exact");
        }
    }

    #[test]
    fn the_range_stays_inside_what_a_narrow_type_can_hold() {
        // f16 overflows at 65504 and a sum over many of these must not reach it by accident.
        for i in 0..1024 {
            let v = element("x", Ty::BufF32, i);
            assert!((-4.0..4.0).contains(&v), "element {i} is {v}");
        }
    }
}
