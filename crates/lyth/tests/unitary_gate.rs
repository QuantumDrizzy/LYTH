//! `gate_q`: any single-qubit unitary on any qubit, through the split of ADR-0028.
//!
//! Two kinds of check, because they fail for different reasons:
//!
//! * **Bit-exact** against a dense reference on the host oracle, PTX and the Unibit emulator.
//!   The reference walks blocks of `2w` (no division, no `view_index`), so it shares no
//!   addressing with the implementation. It does share the arithmetic order -- it has to, to be
//!   bit-exact -- so this check cannot catch a gate written with the wrong formula.
//! * **Physics**, which can: applying `U` and then `U^dagger` returns the state, and a unitary
//!   keeps the norm. Both within an f32 tolerance, because the second launch rounds again. A
//!   kernel with a sign error in one term, or `b` and `c` swapped, keeps its bit-exactness and
//!   fails here.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::process::Command;

use lyth_cuda::{Arg, Context};
use lyth_lang::program::{PrintRange, Program};
use lyth_lang::{eval, ir, parse};

const SENTINEL: u32 = 0x7FC0_DEAD;

/// `[[a, b], [c, d]]`, complex, as the eight scalars the kernel takes.
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
    /// The conjugate transpose.
    fn dagger(&self) -> U {
        let cj = |z: (f32, f32)| (z.0, -z.1);
        U { a: cj(self.a), b: cj(self.c), c: cj(self.b), d: cj(self.d) }
    }
    /// `Rz(phi) Ry(theta) Rz(lambda)` times a global phase, from f64 and rounded once: unitary
    /// to f32 precision, which is what the physics check's tolerance is for.
    fn zyz(theta: f64, phi: f64, lambda: f64) -> U {
        let (c, s) = ((theta / 2.0).cos(), (theta / 2.0).sin());
        let e = |x: f64| (x.cos() as f32, x.sin() as f32);
        let sc = |k: f64, z: (f32, f32)| ((k * z.0 as f64) as f32, (k * z.1 as f64) as f32);
        U {
            a: (c as f32, 0.0),
            b: sc(-s, e(lambda)),
            c: sc(s, e(phi)),
            d: sc(c, e(phi + lambda)),
        }
    }
}

fn unitaries() -> Vec<(&'static str, U)> {
    let s = std::f32::consts::FRAC_1_SQRT_2;
    vec![
        ("H", U { a: (s, 0.0), b: (s, 0.0), c: (s, 0.0), d: (-s, 0.0) }),
        ("X", U { a: (0.0, 0.0), b: (1.0, 0.0), c: (1.0, 0.0), d: (0.0, 0.0) }),
        ("Y", U { a: (0.0, 0.0), b: (0.0, -1.0), c: (0.0, 1.0), d: (0.0, 0.0) }),
        ("S", U { a: (1.0, 0.0), b: (0.0, 0.0), c: (0.0, 0.0), d: (0.0, 1.0) }),
        ("T", U { a: (1.0, 0.0), b: (0.0, 0.0), c: (0.0, 0.0), d: (s, s) }),
        ("zyz(0.7,1.9,-2.3)", U::zyz(0.7, 1.9, -2.3)),
        ("zyz(2.9,-0.4,3.1)", U::zyz(2.9, -0.4, 3.1)),
    ]
}

fn kernel() -> ir::KernelIr {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../examples/gate_q.lyth");
    let unit = parse(&std::fs::read_to_string(path).unwrap()).unwrap();
    ir::lower(&unit, &unit.kernels[0]).unwrap()
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

/// The dense reference, the same fma chains as the kernel, in the same order.
fn reference(u: &U, re: &[f32], im: &[f32], w: usize) -> (Vec<f32>, Vec<f32>) {
    let n = re.len();
    let (mut qr, mut qi) = (vec![f32::from_bits(SENTINEL); n], vec![f32::from_bits(SENTINEL); n]);
    let U { a: (ar, ai), b: (br, bi), c: (cr, ci), d: (dr, di) } = *u;
    let mut base = 0;
    while base < n {
        for p in 0..w {
            let (i, j) = (base + p, base + p + w);
            let (p0r, p0i, p1r, p1i) = (re[i], im[i], re[j], im[j]);
            qr[i] = ar.mul_add(p0r, (-ai).mul_add(p0i, br.mul_add(p1r, -bi * p1i)));
            qi[i] = ar.mul_add(p0i, ai.mul_add(p0r, br.mul_add(p1i, bi * p1r)));
            qr[j] = cr.mul_add(p0r, (-ci).mul_add(p0i, dr.mul_add(p1r, -di * p1i)));
            qi[j] = cr.mul_add(p0i, ci.mul_add(p0r, dr.mul_add(p1i, di * p1r)));
        }
        base += 2 * w;
    }
    (qr, qi)
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

fn host(k: &ir::KernelIr, u: &U, re: &[f32], im: &[f32], w: usize) -> (Vec<f32>, Vec<f32>) {
    let n = re.len();
    let mut inputs = eval::Inputs::default();
    for (name, v) in u.scalars() {
        inputs.scalars.insert(name.into(), v);
    }
    inputs.extents.insert("n".into(), n as u32);
    inputs.extents.insert("w".into(), w as u32);
    inputs.buffers.insert("re".into(), re.to_vec());
    inputs.buffers.insert("im".into(), im.to_vec());
    inputs.buffers.insert("qr".into(), vec![f32::from_bits(SENTINEL); n]);
    inputs.buffers.insert("qi".into(), vec![f32::from_bits(SENTINEL); n]);
    let out = eval::eval(k, n / 2, &inputs).expect("the host oracle evaluates gate_q");
    (out.buffers["qr"].clone(), out.buffers["qi"].clone())
}

struct Gpu<'c> {
    ctx: &'c Context,
    func: lyth_cuda::Function<'c>,
}

fn device(g: &Gpu, k: &ir::KernelIr, u: &U, re: &[f32], im: &[f32], w: usize) -> (Vec<f32>, Vec<f32>) {
    let n = re.len();
    let sentinel = vec![f32::from_bits(SENTINEL); n];
    let (d_re, d_im) = (g.ctx.upload(re).unwrap(), g.ctx.upload(im).unwrap());
    let (d_qr, d_qi) = (g.ctx.upload(&sentinel).unwrap(), g.ctx.upload(&sentinel).unwrap());
    let scalars: BTreeMap<&str, f32> = u.scalars().into_iter().collect();
    let args: Vec<Arg> = k
        .params
        .iter()
        .map(|p| match p.name.as_str() {
            "n" => Arg::U32(n as u32),
            "w" => Arg::U32(w as u32),
            "re" => Arg::Buf(&d_re),
            "im" => Arg::Buf(&d_im),
            "qr" => Arg::Buf(&d_qr),
            "qi" => Arg::Buf(&d_qi),
            s => Arg::F32(*scalars.get(s).unwrap_or_else(|| panic!("unexpected parameter `{s}`"))),
        })
        .collect();
    g.func.launch(((n / 2) as u32).div_ceil(256).max(1), 256, &args).expect("launch");
    (d_qr.download().unwrap(), d_qi.download().unwrap())
}

fn cases() -> Vec<(usize, usize)> {
    let mut v: Vec<(usize, usize)> = (0..12).map(|q| (4096, 1usize << q)).collect();
    v.extend([(384, 3), (384, 48), (1000, 5), (2, 1)]);
    v
}

#[test]
fn any_single_qubit_unitary_is_bit_exact_on_the_host_oracle_and_on_ptx() {
    let k = kernel();
    let ctx = match Context::new(0) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("skipped: no CUDA device ({e})");
            return;
        }
    };
    let module = lyth_ptx::emit(&k, "sm_120").expect("ptx emits gate_q");
    let loaded = ctx.load_ptx(&module.ptx).expect("the driver accepts the PTX");
    let g = Gpu { ctx: &ctx, func: loaded.function(&module.entry).expect("entry") };
    let mut checked = 0;
    for (name, u) in unitaries() {
        for (n, w) in cases() {
            let (re, im) = (amplitudes(n, 3 + w as u32), amplitudes(n, 900 + n as u32));
            let (want_r, want_i) = reference(&u, &re, &im, w);
            let (hr, hi) = host(&k, &u, &re, &im, w);
            assert_eq!(bits(&hr), bits(&want_r), "{name}: host qr, n = {n}, w = {w}");
            assert_eq!(bits(&hi), bits(&want_i), "{name}: host qi, n = {n}, w = {w}");
            let (dr, di) = device(&g, &k, &u, &re, &im, w);
            assert_eq!(bits(&dr), bits(&want_r), "{name}: ptx qr, n = {n}, w = {w}");
            assert_eq!(bits(&di), bits(&want_i), "{name}: ptx qi, n = {n}, w = {w}");
            checked += 1;
        }
    }
    println!("gate_q: {checked} launches bit-exact on the host oracle and PTX (7 unitaries x 16 shapes)");
}

#[test]
fn a_unitary_then_its_inverse_returns_the_state_and_keeps_the_norm() {
    let k = kernel();
    let ctx = match Context::new(0) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("skipped: no CUDA device ({e})");
            return;
        }
    };
    let module = lyth_ptx::emit(&k, "sm_120").unwrap();
    let loaded = ctx.load_ptx(&module.ptx).unwrap();
    let g = Gpu { ctx: &ctx, func: loaded.function(&module.entry).unwrap() };
    let norm = |r: &[f32], i: &[f32]| r.iter().zip(i).map(|(a, b)| (*a as f64).powi(2) + (*b as f64).powi(2)).sum::<f64>();
    let (mut worst_back, mut worst_norm) = (0.0f64, 0.0f64);
    for (name, u) in unitaries() {
        for q in 0..12 {
            let (n, w) = (4096usize, 1usize << q);
            let (re, im) = (amplitudes(n, 11 + q), amplitudes(n, 77 + q));
            let (r1, i1) = device(&g, &k, &u, &re, &im, w);
            let (r2, i2) = device(&g, &k, &u.dagger(), &r1, &i1, w);
            // Largest element error of U^dagger U psi against psi; amplitudes are below 0.5.
            let back = re.iter().zip(&r2).chain(im.iter().zip(&i2)).map(|(a, b)| (a - b).abs() as f64).fold(0.0, f64::max);
            let drift = (norm(&r1, &i1) / norm(&re, &im) - 1.0).abs();
            assert!(back < 1e-6, "{name} on qubit {q}: U^dagger U psi is off by {back:e}");
            assert!(drift < 1e-6, "{name} on qubit {q}: the norm moved by {drift:e}");
            worst_back = worst_back.max(back);
            worst_norm = worst_norm.max(drift);
        }
    }
    println!("gate_q: U^dagger U psi = psi to {worst_back:.1e}, norm kept to {worst_norm:.1e}, 7 unitaries x 12 qubits");
}

#[test]
fn a_wrong_gate_keeps_its_bit_exactness_and_fails_the_physics() {
    // The claim in the module comment, tested: swapping `b` and `c` of a non-symmetric unitary
    // is still a gate the kernel computes bit-exactly, and it is not the inverse's partner.
    let u = U::zyz(0.7, 1.9, -2.3);
    let wrong = U { b: u.c, c: u.b, ..u };
    let (re, im) = (amplitudes(64, 5), amplitudes(64, 6));
    let (r1, i1) = reference(&wrong, &re, &im, 4);
    let (r2, _) = reference(&u.dagger(), &r1, &i1, 4);
    let back = re.iter().zip(&r2).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
    assert!(back > 1e-2, "the physics check would not see a swapped gate: {back:e}");
}

#[test]
fn any_single_qubit_unitary_is_bit_exact_on_the_unibit_emulator_too() {
    let k = kernel();
    const UNWRITTEN: f32 = 1.0e30;
    let uni = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../Unibit");
    let mut checked = 0;
    // One general unitary over the widths that exercise both MTLB paths: whole registers
    // (w % 8 == 0) and one f32 at a time (w = 1, 2, 4, 3, 5).
    let u = U::zyz(0.7, 1.9, -2.3);
    for (n, w) in [(4096, 1), (4096, 2), (4096, 4), (4096, 8), (4096, 64), (4096, 2048), (384, 3), (1000, 5)] {
        let (re, im) = (amplitudes(n, 3 + w as u32), amplitudes(n, 900 + n as u32));
        let (want_r, want_i) = reference(&u, &re, &im, w);
        let scalars: BTreeMap<String, f32> = u.scalars().into_iter().map(|(a, b)| (a.to_string(), b)).collect();
        let asm = lyth_uasm::emit(
            &k,
            &Program {
                n: (n / 2) as u32,
                extents: BTreeMap::from([("n".to_string(), n as u32), ("w".to_string(), w as u32)]),
                scalars,
                prints: ["qr", "qi"].into_iter().map(|b| PrintRange { buffer: b.into(), lo: 0, hi: n as u32 }).collect(),
                buffers: BTreeMap::from([
                    ("re".to_string(), re.clone()),
                    ("im".to_string(), im.clone()),
                    ("qr".to_string(), vec![UNWRITTEN; n]),
                    ("qi".to_string(), vec![UNWRITTEN; n]),
                ]),
            },
        )
        .expect("the MTLB back end emits gate_q");
        let dir = tempfile::tempdir().unwrap();
        let src = dir.path().join("gate_q.uasm");
        std::fs::write(&src, &asm).unwrap();
        let out = Command::new("cargo")
            .args(["run", "--quiet", "--release", "--", "run", src.to_str().unwrap()])
            .current_dir(&uni)
            .output()
            .expect("the Unibit emulator");
        assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
        let got: Vec<f32> = String::from_utf8_lossy(&out.stdout).lines().filter_map(|l| l.trim().parse::<f32>().ok()).collect();
        assert_eq!(got.len(), 2 * n, "n = {n}, w = {w}");
        assert_eq!(bits(&got[..n]), bits(&want_r), "emulator qr, n = {n}, w = {w}");
        assert_eq!(bits(&got[n..]), bits(&want_i), "emulator qi, n = {n}, w = {w}");
        checked += 1;
    }
    println!("gate_q: {checked} launches bit-exact on the Unibit emulator");
}
