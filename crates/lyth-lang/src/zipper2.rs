//! One `ZIPPER2` step on three 256-bit words, each held as eight `f32`.
//!
//! The layout is the instruction's: words 0..3 are the int8 codes, words 4 and 5
//! are the per-bond scales. The arithmetic is the same order as
//! `TensorNetworkUnit::zipper2_step`, in `f32`, so the host oracle and the
//! emulator can be compared bit for bit.

/// 256 flops in the instruction, eight `f32` lanes in the register.
pub const FLOPS_PER_ELEMENT: f64 = 32.0;

pub fn zipper2_words(acc: &[f32; 8], ket: &[f32; 8], bra: &[f32; 8]) -> [f32; 8] {
    let b = unpack(ket);
    let a = unpack(bra);
    let mut partial = [[[(0.0f32, 0.0f32); 2]; 2]; 2];
    for al in 0..2 {
        for phys in 0..2 {
            for br in 0..2 {
                let mut re = 0.0f32;
                let mut im = 0.0f32;
                for bl in 0..2 {
                    let (e_re, e_im) = e_at(acc, al, bl);
                    let (b_re, b_im) = b[bl][phys][br];
                    re += e_re * b_re - e_im * b_im;
                    im += e_re * b_im + e_im * b_re;
                }
                partial[al][phys][br] = (re, im);
            }
        }
    }
    let mut out = [0.0f32; 8];
    for ar in 0..2 {
        for br in 0..2 {
            let mut re = 0.0f32;
            let mut im = 0.0f32;
            for al in 0..2 {
                for phys in 0..2 {
                    let (a_re, a_im) = a[al][phys][ar];
                    let (t_re, t_im) = partial[al][phys][br];
                    re += a_re * t_re + a_im * t_im;
                    im += a_re * t_im - a_im * t_re;
                }
            }
            let w = 2 * (2 * ar + br);
            out[w] = re;
            out[w + 1] = im;
        }
    }
    out
}

fn e_at(acc: &[f32; 8], al: usize, bl: usize) -> (f32, f32) {
    let w = 2 * (2 * (al & 1) + (bl & 1));
    (acc[w], acc[w + 1])
}

fn i8_at(words: &[f32; 8], idx: usize) -> i8 {
    let bits = words[idx / 4].to_bits();
    let shift = (idx % 4) * 8;
    ((bits >> shift) & 0xFF) as u8 as i8
}

fn unpack(words: &[f32; 8]) -> [[[(f32, f32); 2]; 2]; 2] {
    let scale = [words[4], words[5]];
    let mut out = [[[(0.0f32, 0.0f32); 2]; 2]; 2];
    for left in 0..2 {
        for phys in 0..2 {
            for right in 0..2 {
                let idx = 2 * ((2 * left + phys) * 2 + right);
                out[left][phys][right] = (
                    i8_at(words, idx) as f32 * scale[right],
                    i8_at(words, idx + 1) as f32 * scale[right],
                );
            }
        }
    }
    out
}
