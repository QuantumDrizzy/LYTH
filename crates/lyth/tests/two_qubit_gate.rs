//! ADR-0029 step 3 (P4): `cu_q` and `swap_q`, two-qubit gates on any ordered pair of qubits,
//! bit-exact on the host oracle, PTX and the Unibit emulator; through the generated binding; and
//! the physics bit-exactness cannot see.
//!
//! **Two references, cross-checked.** `quads_by_blocks` builds the four leaves by walking blocks
//! -- `wc`-runs of the buffer, then `wt`-runs of each half -- with no division and no
//! `view_index`, so it serves every width, including those that are not powers of two.
//! `quads_by_bits` builds them from the physics: leaf `XY` holds the amplitudes whose bit `c` is
//! `X` and bit `t` is `Y`. For qubit widths the two must agree, and a test says they do.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::process::Command;

use lyth_cuda::{Arg, Context};
use lyth_lang::program::{PrintRange, Program};
use lyth_lang::{eval, ir, parse};

const SENTINEL: u32 = 0x7FC0_DEAD;

mod cu_q_binding {
    #![allow(clippy::manual_clamp, clippy::too_many_arguments, dead_code)]
    include!("generated/cu_q.rs");
}
mod swap_q_binding {
    #![allow(clippy::manual_clamp, clippy::too_many_arguments, dead_code)]
    include!("generated/swap_q.rs");
}

#[derive(Clone, Copy, Debug)]
struct U {
    a: (f32, f32),
    b: (f32, f32),
    c: (f32, f32),
    d: (f32, f32),
}

impl U {
    fn scalars(&self) -> [(&'static str, f32); 8] {
        [
            ("ar", self.a.0), ("ai", self.a.1), ("br", self.b.0), ("bi", self.b.1),
            ("cr", self.c.0), ("ci", self.c.1), ("dr", self.d.0), ("di", self.d.1),
        ]
    }
    fn dagger(&self) -> U {
        let cj = |z: (f32, f32)| (z.0, -z.1);
        U { a: cj(self.a), b: cj(self.c), c: cj(self.b), d: cj(self.d) }
    }
    fn zyz(theta: f64, phi: f64, lambda: f64) -> U {
        let (c, s) = ((theta / 2.0).cos(), (theta / 2.0).sin());
        let e = |x: f64| (x.cos() as f32, x.sin() as f32);
        let sc = |k: f64, z: (f32, f32)| ((k * z.0 as f64) as f32, (k * z.1 as f64) as f32);
        U { a: (c as f32, 0.0), b: sc(-s, e(lambda)), c: sc(s, e(phi)), d: sc(c, e(phi + lambda)) }
    }
}

fn gates() -> Vec<(&'static str, U)> {
    let (z, o) = ((0.0, 0.0), (1.0, 0.0));
    let ph = std::f64::consts::FRAC_PI_3;
    vec![
        ("CNOT", U { a: z, b: o, c: o, d: z }),
        ("CZ", U { a: o, b: z, c: z, d: (-1.0, 0.0) }),
        ("CPhase(pi/3)", U { a: o, b: z, c: z, d: (ph.cos() as f32, ph.sin() as f32) }),
        ("C-zyz", U::zyz(0.7, 1.9, -2.3)),
    ]
}

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn kernel(name: &str) -> ir::KernelIr {
    let unit = parse(&std::fs::read_to_string(root().join(format!("examples/{name}.lyth"))).unwrap()).unwrap();
    ir::lower(&unit, &unit.kernels[0]).unwrap()
}

/// The widths a caller passes for control `c`, target `t`: the inner one counts elements of the
/// parent view, which has lost bit `c`.
fn widths(c: u32, t: u32) -> (usize, usize) {
    (1 << c, if t < c { 1 << t } else { 1 << (t - 1) })
}

fn amplitudes(n: usize, seed: u32) -> Vec<f32> {
    let mut x = seed;
    (0..n)
        .map(|_| {
            x = x.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            ((x >> 8) as f32 / (1u32 << 24) as f32) - 0.5
        })
        .collect()
}

/// The four leaves, as lists of buffer indices, by walking blocks. `[00, 01, 10, 11]`.
fn quads_by_blocks(n: usize, wc: usize, wt: usize) -> [Vec<usize>; 4] {
    let halves: [Vec<usize>; 2] = [0, 1].map(|x| {
        let mut h = Vec::new();
        let mut base = 0;
        while base < n {
            h.extend(base + x * wc..base + x * wc + wc);
            base += 2 * wc;
        }
        h
    });
    let leaf = |h: &Vec<usize>, y: usize| {
        let mut l = Vec::new();
        let mut base = 0;
        while base < h.len() {
            l.extend_from_slice(&h[base + y * wt..base + y * wt + wt]);
            base += 2 * wt;
        }
        l
    };
    [leaf(&halves[0], 0), leaf(&halves[0], 1), leaf(&halves[1], 0), leaf(&halves[1], 1)]
}

/// The same leaves from the physics: bit `c` and bit `t` of the index.
fn quads_by_bits(n: usize, c: u32, t: u32) -> [Vec<usize>; 4] {
    let bit = |i: usize, b: u32| (i >> b) & 1;
    [(0, 0), (0, 1), (1, 0), (1, 1)].map(|(x, y)| (0..n).filter(|&i| bit(i, c) == x && bit(i, t) == y).collect())
}

/// `qr, qi` from `re, im` over the leaves: the gate's formula, in the kernel's fma order.
fn reference(kind: Option<&U>, re: &[f32], im: &[f32], q: &[Vec<usize>; 4]) -> (Vec<f32>, Vec<f32>) {
    let n = re.len();
    let (mut qr, mut qi) = (vec![f32::from_bits(SENTINEL); n], vec![f32::from_bits(SENTINEL); n]);
    for (((&i00, &i01), &i10), &i11) in q[0].iter().zip(&q[1]).zip(&q[2]).zip(&q[3]) {
        match kind {
            Some(u) => {
                let U { a: (ar, ai), b: (br, bi), c: (cr, ci), d: (dr, di) } = *u;
                let (p0r, p0i, p1r, p1i) = (re[i10], im[i10], re[i11], im[i11]);
                qr[i10] = ar.mul_add(p0r, (-ai).mul_add(p0i, br.mul_add(p1r, -bi * p1i)));
                qi[i10] = ar.mul_add(p0i, ai.mul_add(p0r, br.mul_add(p1i, bi * p1r)));
                qr[i11] = cr.mul_add(p0r, (-ci).mul_add(p0i, dr.mul_add(p1r, -di * p1i)));
                qi[i11] = cr.mul_add(p0i, ci.mul_add(p0r, dr.mul_add(p1i, di * p1r)));
                for i in [i00, i01] {
                    (qr[i], qi[i]) = (re[i], im[i]);
                }
            }
            None => {
                // swap: |01> <-> |10>
                for (to, from) in [(i00, i00), (i01, i10), (i10, i01), (i11, i11)] {
                    (qr[to], qi[to]) = (re[from], im[from]);
                }
            }
        }
    }
    (qr, qi)
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

fn scalars_of(kind: Option<&U>) -> BTreeMap<String, f32> {
    kind.map(|u| u.scalars().into_iter().map(|(a, b)| (a.to_string(), b)).collect()).unwrap_or_default()
}

fn host(k: &ir::KernelIr, kind: Option<&U>, re: &[f32], im: &[f32], wc: usize, wt: usize) -> (Vec<f32>, Vec<f32>) {
    let n = re.len();
    let mut inputs = eval::Inputs { scalars: scalars_of(kind), ..Default::default() };
    for (e, v) in [("n", n), ("wc", wc), ("wt", wt)] {
        inputs.extents.insert(e.into(), v as u32);
    }
    inputs.buffers.insert("re".into(), re.to_vec());
    inputs.buffers.insert("im".into(), im.to_vec());
    inputs.buffers.insert("qr".into(), vec![f32::from_bits(SENTINEL); n]);
    inputs.buffers.insert("qi".into(), vec![f32::from_bits(SENTINEL); n]);
    let out = eval::eval(k, n / 4, &inputs).expect("the host oracle evaluates a split of a view");
    (out.buffers["qr"].clone(), out.buffers["qi"].clone())
}

#[allow(clippy::too_many_arguments)]
fn device(ctx: &Context, f: &lyth_cuda::Function, k: &ir::KernelIr, kind: Option<&U>, re: &[f32], im: &[f32], wc: usize, wt: usize) -> (Vec<f32>, Vec<f32>) {
    let n = re.len();
    let sentinel = vec![f32::from_bits(SENTINEL); n];
    let (d_re, d_im) = (ctx.upload(re).unwrap(), ctx.upload(im).unwrap());
    let (d_qr, d_qi) = (ctx.upload(&sentinel).unwrap(), ctx.upload(&sentinel).unwrap());
    let sc = scalars_of(kind);
    let args: Vec<Arg> = k
        .params
        .iter()
        .map(|p| match p.name.as_str() {
            "n" => Arg::U32(n as u32),
            "wc" => Arg::U32(wc as u32),
            "wt" => Arg::U32(wt as u32),
            "re" => Arg::Buf(&d_re),
            "im" => Arg::Buf(&d_im),
            "qr" => Arg::Buf(&d_qr),
            "qi" => Arg::Buf(&d_qi),
            s => Arg::F32(sc[s]),
        })
        .collect();
    f.launch(((n / 4) as u32).div_ceil(256).max(1), 256, &args).expect("launch");
    (d_qr.download().unwrap(), d_qi.download().unwrap())
}

#[test]
fn the_two_references_agree_on_every_qubit_pair() {
    for q in [3u32, 6] {
        let n = 1usize << q;
        for c in 0..q {
            for t in (0..q).filter(|&t| t != c) {
                let (wc, wt) = widths(c, t);
                assert_eq!(quads_by_blocks(n, wc, wt), quads_by_bits(n, c, t), "{q} qubits, c = {c}, t = {t}");
            }
        }
    }
}

/// (n, wc, wt, label): every ordered pair of a 10-qubit register, and widths that are not qubit
/// widths (each level still covers its parent: 2wc | n, 2wt | n/2).
fn shapes() -> Vec<(usize, usize, usize, String)> {
    let mut v = Vec::new();
    for c in 0..10u32 {
        for t in (0..10).filter(|&t| t != c) {
            let (wc, wt) = widths(c, t);
            v.push((1024, wc, wt, format!("c = {c}, t = {t}")));
        }
    }
    for (n, wc, wt) in [(360, 5, 9), (360, 3, 2), (360, 12, 15), (48, 1, 1), (4, 1, 1)] {
        v.push((n, wc, wt, format!("n = {n}, wc = {wc}, wt = {wt}")));
    }
    v
}

#[test]
fn two_qubit_gates_are_bit_exact_on_the_host_oracle_and_on_ptx() {
    let ctx = match Context::new(0) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("skipped: no CUDA device ({e})");
            return;
        }
    };
    let mut launched = 0;
    for (name, kind) in [("cu_q", true), ("swap_q", false)] {
        let k = kernel(name);
        let module = lyth_ptx::emit(&k, "sm_120").expect("ptx emits a split of a view");
        let loaded = ctx.load_ptx(&module.ptx).expect("the driver accepts the PTX");
        let f = loaded.function(&module.entry).expect("entry");
        let gs: Vec<(&str, Option<U>)> = if kind { gates().into_iter().map(|(g, u)| (g, Some(u))).collect() } else { vec![("SWAP", None)] };
        for (g, u) in &gs {
            for (n, wc, wt, label) in shapes() {
                let (re, im) = (amplitudes(n, 3 + wc as u32), amplitudes(n, 70 + wt as u32));
                let (want_r, want_i) = reference(u.as_ref(), &re, &im, &quads_by_blocks(n, wc, wt));
                assert!(!bits(&want_r).contains(&SENTINEL), "the reference covers every element");
                let (hr, hi) = host(&k, u.as_ref(), &re, &im, wc, wt);
                assert_eq!(bits(&hr), bits(&want_r), "{g}: host qr, {label}");
                assert_eq!(bits(&hi), bits(&want_i), "{g}: host qi, {label}");
                let (dr, di) = device(&ctx, &f, &k, u.as_ref(), &re, &im, wc, wt);
                assert_eq!(bits(&dr), bits(&want_r), "{g}: ptx qr, {label}");
                assert_eq!(bits(&di), bits(&want_i), "{g}: ptx qi, {label}");
                launched += 1;
            }
        }
    }
    println!("two-qubit gates: {launched} launches bit-exact on the host oracle and PTX (5 gates x 95 shapes)");
}

#[test]
fn cnot_and_swap_are_their_own_inverses_bit_for_bit_and_a_controlled_unitary_undoes() {
    let ctx = match Context::new(0) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("skipped: no CUDA device ({e})");
            return;
        }
    };
    let (cu, sw) = (kernel("cu_q"), kernel("swap_q"));
    let mc = ctx.load_ptx(&lyth_ptx::emit(&cu, "sm_120").unwrap().ptx).unwrap();
    let ms = ctx.load_ptx(&lyth_ptx::emit(&sw, "sm_120").unwrap().ptx).unwrap();
    let (fc, fs) = (mc.function("cu_q").unwrap(), ms.function("swap_q").unwrap());
    let cnot = gates()[0].1;
    let czyz = gates()[3].1;
    let n = 1usize << 10;
    let mut worst = 0.0f32;
    for c in 0..10u32 {
        for t in (0..10).filter(|&t| t != c) {
            let (wc, wt) = widths(c, t);
            let (re, im) = (amplitudes(n, 5 + c), amplitudes(n, 9 + t));
            // CNOT's fma chains multiply by exact 0 and 1, so two of them are the identity on the
            // bits; a swap is a copy.
            let (r1, i1) = device(&ctx, &fc, &cu, Some(&cnot), &re, &im, wc, wt);
            let (r2, i2) = device(&ctx, &fc, &cu, Some(&cnot), &r1, &i1, wc, wt);
            assert_eq!((bits(&r2), bits(&i2)), (bits(&re), bits(&im)), "CNOT twice, c = {c}, t = {t}");
            assert_ne!(bits(&r1), bits(&re), "CNOT moved something, c = {c}, t = {t}");
            let (r1, i1) = device(&ctx, &fs, &sw, None, &re, &im, wc, wt);
            let (r2, i2) = device(&ctx, &fs, &sw, None, &r1, &i1, wc, wt);
            assert_eq!((bits(&r2), bits(&i2)), (bits(&re), bits(&im)), "SWAP twice, c = {c}, t = {t}");
            // C-U then C-U^dagger, to f32 precision.
            let (r1, i1) = device(&ctx, &fc, &cu, Some(&czyz), &re, &im, wc, wt);
            let (r2, i2) = device(&ctx, &fc, &cu, Some(&czyz.dagger()), &r1, &i1, wc, wt);
            let back = re.iter().zip(&r2).chain(im.iter().zip(&i2)).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
            assert!(back < 1e-6, "C-U then C-U^dagger, c = {c}, t = {t}: {back:e}");
            worst = worst.max(back);
        }
    }
    println!("two-qubit gates: CNOT and SWAP involutions bit-exact on 90 pairs; C-U^dagger C-U psi = psi to {worst:.1e}");
}

#[test]
fn the_generated_binding_launches_quads_and_refuses_an_inner_width_that_runs_off_the_view() {
    use cu_q_binding as g;
    let ctx = match Context::new(0) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("skipped: no CUDA device ({e})");
            return;
        }
    };
    let module = g::module(&ctx).unwrap();
    let kernel = g::CuQ::new(&module).unwrap();
    let u = gates()[3].1;
    let s = u.scalars().map(|(_, v)| v);
    for (c, t) in [(0u32, 1u32), (1, 0), (9, 2), (2, 9), (5, 4)] {
        let n = 1usize << 10;
        let (wc, wt) = widths(c, t);
        let (re, im) = (amplitudes(n, 1), amplitudes(n, 2));
        let (want_r, want_i) = reference(Some(&u), &re, &im, &quads_by_blocks(n, wc, wt));
        let sentinel = vec![f32::from_bits(SENTINEL); n];
        let (d_re, d_im) = (ctx.upload(&re).unwrap(), ctx.upload(&im).unwrap());
        let (mut d_qr, mut d_qi) = (ctx.upload(&sentinel).unwrap(), ctx.upload(&sentinel).unwrap());
        kernel
            .launch(n as u32, wc as u32, wt as u32, s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7], &d_re, &d_im, &mut d_qr, &mut d_qi)
            .expect("launch");
        ctx.synchronize().unwrap();
        assert_eq!(bits(&d_qr.download().unwrap()), bits(&want_r), "binding qr, c = {c}, t = {t}");
        assert_eq!(bits(&d_qi.download().unwrap()), bits(&want_i), "binding qi, c = {c}, t = {t}");
    }
    assert_eq!(g::grid(1 << 12), Some(4), "1024 quads at 256 per block");

    // n = 24: wc = 3 covers it (6 | 24), but wt = 4 does not cover the halves (8 does not divide
    // 12) although it divides n. A check against the whole extent would let this through.
    let sentinel = vec![f32::from_bits(SENTINEL); 24];
    let (d_re, d_im) = (ctx.upload(&amplitudes(24, 1)).unwrap(), ctx.upload(&amplitudes(24, 2)).unwrap());
    let (mut d_qr, mut d_qi) = (ctx.upload(&sentinel).unwrap(), ctx.upload(&sentinel).unwrap());
    for (wc, wt, want) in [
        (3u32, 4u32, "2 * wt = 8 must divide n / 2 = 12, but 12 mod 8 = 4"),
        (8, 1, "2 * wc = 16 must divide n = 24"),
        (3, 0, "`wt = 0`"),
    ] {
        let err = kernel
            .launch(24, wc, wt, s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7], &d_re, &d_im, &mut d_qr, &mut d_qi)
            .expect_err("refused")
            .to_string();
        assert!(err.contains(want), "wc = {wc}, wt = {wt}: {err}");
    }
    ctx.synchronize().unwrap();
    assert_eq!(bits(&d_qr.download().unwrap()), bits(&sentinel), "a refused launch wrote nothing");

    // swap through its own binding, one pair, so its generated launcher is exercised too.
    let sm = swap_q_binding::module(&ctx).unwrap();
    let swap = swap_q_binding::SwapQ::new(&sm).unwrap();
    let (n, (wc, wt)) = (1usize << 10, widths(7, 3));
    let (re, im) = (amplitudes(n, 3), amplitudes(n, 4));
    let (want_r, _) = reference(None, &re, &im, &quads_by_bits(n, 7, 3));
    let (d_re, d_im) = (ctx.upload(&re).unwrap(), ctx.upload(&im).unwrap());
    let sentinel = vec![f32::from_bits(SENTINEL); n];
    let (mut d_qr, mut d_qi) = (ctx.upload(&sentinel).unwrap(), ctx.upload(&sentinel).unwrap());
    swap.launch(n as u32, wc as u32, wt as u32, &d_re, &d_im, &mut d_qr, &mut d_qi).unwrap();
    ctx.synchronize().unwrap();
    assert_eq!(bits(&d_qr.download().unwrap()), bits(&want_r), "swap binding");
}

#[test]
fn the_checked_in_two_qubit_bindings_match_the_generator() {
    for name in ["cu_q", "swap_q", "gate_q"] {
        let out = tempfile::Builder::new().suffix(".rs").tempfile().unwrap();
        let run = Command::new(env!("CARGO_BIN_EXE_lyth"))
            .args([
                "build",
                root().join(format!("examples/{name}.lyth")).to_str().unwrap(),
                "--machine",
                root().join("fixtures/machine/sm_120.json").to_str().unwrap(),
                "-o",
                if cfg!(windows) { "nul" } else { "/dev/null" },
                "--bind-rust",
                out.path().to_str().unwrap(),
            ])
            .output()
            .unwrap();
        assert!(run.status.success(), "{}", String::from_utf8_lossy(&run.stderr));
        let fresh = std::fs::read_to_string(out.path()).unwrap();
        let checked_in = std::fs::read_to_string(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(format!("tests/generated/{name}.rs"))).unwrap();
        assert_eq!(fresh.replace("\r\n", "\n"), checked_in.replace("\r\n", "\n"), "tests/generated/{name}.rs is stale");
    }
}

#[test]
fn two_qubit_gates_are_bit_exact_on_the_unibit_emulator_too() {
    const UNWRITTEN: f32 = 1.0e30;
    let uni = root().join("../Labare");
    let u = gates()[3].1;
    let (mut vector, mut scalar) = (0, 0);
    let mut checked = 0;
    for (name, kind) in [("cu_q", Some(u)), ("swap_q", None)] {
        let k = kernel(name);
        // Both loop-nest cases (inner run shorter and longer than the outer), and runs that are
        // whole registers (>= 8, multiple of 8) and single f32s.
        for (c, t) in [(0u32, 1u32), (1, 0), (3, 0), (0, 3), (5, 2), (2, 5), (8, 4), (4, 8), (9, 8), (8, 9)] {
            let n = 1usize << 10;
            let (wc, wt) = widths(c, t);
            let (re, im) = (amplitudes(n, 3 + c), amplitudes(n, 70 + t));
            let (want_r, want_i) = reference(kind.as_ref(), &re, &im, &quads_by_bits(n, c, t));
            let asm = lyth_uasm::emit(
                &k,
                &Program {
                    n: (n / 4) as u32,
                    extents: BTreeMap::from([("n".to_string(), n as u32), ("wc".to_string(), wc as u32), ("wt".to_string(), wt as u32)]),
                    scalars: scalars_of(kind.as_ref()),
                    prints: ["qr", "qi"].into_iter().map(|b| PrintRange { buffer: b.into(), lo: 0, hi: n as u32 }).collect(),
                    buffers: BTreeMap::from([
                        ("re".to_string(), re.clone()),
                        ("im".to_string(), im.clone()),
                        ("qr".to_string(), vec![UNWRITTEN; n]),
                        ("qi".to_string(), vec![UNWRITTEN; n]),
                    ]),
                },
            )
            .unwrap_or_else(|e| panic!("{name} on Unibit, c = {c}, t = {t}: {e}"));
            if asm.lines().any(|l| l.trim_start().starts_with("lq") && l.contains("(s0)")) {
                vector += 1;
            } else {
                scalar += 1;
            }
            let dir = tempfile::tempdir().unwrap();
            let src = dir.path().join("q.uasm");
            std::fs::write(&src, &asm).unwrap();
            let out = Command::new("cargo")
                .args(["run", "--quiet", "--release", "--", "run", src.to_str().unwrap()])
                .current_dir(&uni)
                .output()
                .expect("the Unibit emulator");
            assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
            let got: Vec<f32> = String::from_utf8_lossy(&out.stdout).lines().filter_map(|l| l.trim().parse::<f32>().ok()).collect();
            assert_eq!(got.len(), 2 * n, "{name}, c = {c}, t = {t}");
            assert_eq!(bits(&got[..n]), bits(&want_r), "{name} emulator qr, c = {c}, t = {t}");
            assert_eq!(bits(&got[n..]), bits(&want_i), "{name} emulator qi, c = {c}, t = {t}");
            checked += 1;
        }
    }
    assert!(vector > 0 && scalar > 0, "both paths exercised: {vector} whole-register, {scalar} single-f32");
    println!("two-qubit gates: {checked} launches bit-exact on the Unibit emulator ({vector} whole-register, {scalar} one-f32-at-a-time)");
}

#[test]
fn unibit_refuses_widths_whose_runs_do_not_tile_rather_than_divide_per_element() {
    // 2wc | n and 2wt | n/2, so the host and PTX run it; but wt = 2 runs inside wc = 3 blocks
    // only if 2 * 2 divides 3, and it does not.
    let k = kernel("swap_q");
    let n = 24usize;
    let err = lyth_uasm::emit(
        &k,
        &Program {
            n: (n / 4) as u32,
            extents: BTreeMap::from([("n".to_string(), 24), ("wc".to_string(), 3), ("wt".to_string(), 2)]),
            scalars: BTreeMap::new(),
            prints: vec![],
            buffers: BTreeMap::from([
                ("re".to_string(), vec![0.0; n]),
                ("im".to_string(), vec![0.0; n]),
                ("qr".to_string(), vec![0.0; n]),
                ("qi".to_string(), vec![0.0; n]),
            ]),
        },
    )
    .expect_err("refused");
    assert!(err.to_string().contains("is not a loop nest at these widths"), "{err}");
}
