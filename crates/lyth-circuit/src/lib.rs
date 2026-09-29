//! Circuits as LYTH kernels (ADR-0030).
//!
//! A circuit is a list of gates on a state vector of `2^qubits` complex amplitudes, split re/im,
//! little-endian (bit `q` of an index is qubit `q`, as in Qiskit). It runs two ways:
//!
//! * **unfused** -- one launch per gate, through `gate_q`, `cu_q` and `swap_q` (ADR-0028/0029);
//! * **fused** -- [`fuse`] cuts the gate list, *in circuit order*, into groups of consecutive gates
//!   that touch at most `k` qubits, and each group becomes one generated kernel: a split to depth
//!   `|group|`, so a thread owns the `2^|group|` amplitudes the group mixes, and a body that applies
//!   the group's gates one after the other in registers, with **the same fma chains, in the same
//!   order, as the unfused kernels**. In f32 the values between two gates are the values the
//!   unfused circuit writes to memory and reads back, so the two runs agree bit for bit. That is
//!   the claim of ADR-0030 (Q1) and what the tests check.
//!
//! Nothing here reorders gates: commuting them would make bigger groups and change the arithmetic.

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::path::PathBuf;

use lyth_cuda::{Arg, Buffer, Context, CudaError};
use lyth_lang::{eval, ir, parse};

/// `[a, b, c, d]` of `[[a, b], [c, d]]`, complex `(re, im)`, in f64. Rounded to f32 once, where it
/// enters a kernel, the same way on both paths.
pub type M = [(f64, f64); 4];

#[derive(Clone, Debug, PartialEq)]
pub enum Gate {
    /// A single-qubit unitary on qubit `q`.
    U(u32, M),
    /// Controlled-`U`: control `c`, target `t`.
    CU(u32, u32, M),
    /// Swap two qubits.
    Swap(u32, u32),
}

impl Gate {
    pub fn qubits(&self) -> Vec<u32> {
        match self {
            Gate::U(q, _) => vec![*q],
            Gate::CU(c, t, _) => vec![*c, *t],
            Gate::Swap(a, b) => vec![*a, *b],
        }
    }
}

/// The f32 values a matrix enters a kernel as: `ar ai br bi cr ci dr di`.
pub fn f32s(m: &M) -> [f32; 8] {
    let mut out = [0.0; 8];
    for (i, (r, im)) in m.iter().enumerate() {
        out[2 * i] = *r as f32;
        out[2 * i + 1] = *im as f32;
    }
    out
}

/// A run of consecutive gates executed as one pass.
#[derive(Clone, Debug)]
pub struct Group {
    /// The qubits the group touches, **descending**: level 0 of the split is the highest qubit, so
    /// removing it shifts no lower bit and every level's width is the plain `2^q`.
    pub qubits: Vec<u32>,
    pub gates: Vec<Gate>,
}

/// Cut `gates` into groups of consecutive gates touching at most `k` qubits, in circuit order.
///
/// Greedy: a gate joins the open group if the union of qubits stays within `k`; otherwise the
/// group closes and the gate opens the next. A gate that alone needs more than `k` qubits (a
/// two-qubit gate at `k = 1`) is a group of its own.
pub fn fuse(gates: &[Gate], k: usize) -> Vec<Group> {
    let mut out: Vec<Group> = Vec::new();
    let mut cur: Option<Group> = None;
    for g in gates {
        match cur.as_mut() {
            Some(open) => {
                let mut union = open.qubits.clone();
                for q in g.qubits() {
                    if !union.contains(&q) {
                        union.push(q);
                    }
                }
                if union.len() <= k.max(1) {
                    open.qubits = union;
                    open.gates.push(g.clone());
                } else {
                    out.push(cur.take().unwrap());
                    cur = Some(Group { qubits: g.qubits(), gates: vec![g.clone()] });
                }
            }
            None => cur = Some(Group { qubits: g.qubits(), gates: vec![g.clone()] }),
        }
    }
    out.extend(cur);
    for grp in &mut out {
        grp.qubits.sort_unstable_by(|a, b| b.cmp(a));
    }
    out
}

/// A literal that parses back to exactly this f32: the f32's exact value printed as an f64
/// (every f32 is an f64), with the sign outside so the lexer sees a number. `-0.0` keeps its sign.
fn lit(x: f32) -> String {
    let v = f64::from(x);
    if v.is_sign_negative() {
        format!("(-{:?})", -v)
    } else {
        format!("{v:?}")
    }
}

impl Group {
    /// The widths the kernel takes, `w0 .. w{k-1}`, outermost level first.
    pub fn widths(&self) -> Vec<u32> {
        self.qubits.iter().map(|q| 1u32 << q).collect()
    }

    fn level_of(&self, q: u32) -> usize {
        self.qubits.iter().position(|x| *x == q).expect("a group's gate acts on the group's qubits")
    }

    /// The generated LYTH source of this group, as kernel `name`.
    pub fn source(&self, name: &str) -> String {
        let k = self.qubits.len();
        let size = 1usize << k;
        // Tuple element j: the bit of level l is bit (k - 1 - l) of j.
        let bits = |j: usize| -> String { (0..k).map(|l| if (j >> (k - 1 - l)) & 1 == 1 { '1' } else { '0' }).collect() };
        let mask = |l: usize| 1usize << (k - 1 - l);

        let mut s = String::new();
        let _ = writeln!(s, "machine sm_120\n");
        let _ = writeln!(s, "# GENERATED by lyth-circuit (ADR-0030): {} gate(s) on qubits {:?}, one pass.", self.gates.len(), self.qubits);
        let ws: Vec<String> = (0..k).map(|l| format!("w{l}: u32")).collect();
        let _ = writeln!(s, "kernel {name}(n: u32, {}, re: [f32; n], im: [f32; n], qr: [f32; n], qi: [f32; n])\n", ws.join(", "));
        for (buf, node, leaf) in [("re", "nre", "pr"), ("im", "nim", "pi"), ("qr", "nqr", "qr"), ("qi", "nqi", "qi")] {
            let mut level: Vec<(String, String)> = vec![(buf.to_string(), String::new())];
            for l in 0..k {
                let mut next = Vec::new();
                for (nm, b) in &level {
                    let child = |x: char| {
                        let bb = format!("{b}{x}");
                        if l + 1 == k { format!("{leaf}{bb}") } else { format!("{node}_{bb}") }
                    };
                    let (a, c) = (child('0'), child('1'));
                    let _ = writeln!(s, "    split {nm} into {a}, {c} : blocks w{l}");
                    next.push((a, format!("{b}0")));
                    next.push((c, format!("{b}1")));
                }
                level = next;
            }
        }
        let _ = writeln!(s);
        for j in 0..size {
            let _ = writeln!(s, "    stream pr{b} : dram -> reg\n    stream pi{b} : dram -> reg", b = bits(j));
        }
        for j in 0..size {
            let _ = writeln!(s, "    stream qr{b} : dram -> reg, drain\n    stream qi{b} : dram -> reg, drain", b = bits(j));
        }
        let _ = writeln!(s, "\n    at reg:");

        // The current name of each tuple element's (re, im).
        let mut cur: Vec<(String, String)> = (0..size).map(|j| (format!("pr{}", bits(j)), format!("pi{}", bits(j)))).collect();
        let mut fresh = 0usize;
        let mut body = String::new();
        // `gate_q`'s four chains on the pair (j0, j1), exactly as the unfused kernel writes them.
        let mut apply = |cur: &mut Vec<(String, String)>, m: &M, j0: usize, j1: usize, body: &mut String| {
            let [ar, ai, br, bi, cr, ci, dr, di] = f32s(m).map(lit);
            let ((p0r, p0i), (p1r, p1i)) = (cur[j0].clone(), cur[j1].clone());
            let names: [String; 4] = std::array::from_fn(|i| format!("x{}", fresh + i));
            fresh += 4;
            let _ = writeln!(body, "        {} = {ar} * {p0r} + ((-{ai}) * {p0i} + ({br} * {p1r} + (-{bi}) * {p1i}))", names[0]);
            let _ = writeln!(body, "        {} = {ar} * {p0i} + ({ai} * {p0r} + ({br} * {p1i} + {bi} * {p1r}))", names[1]);
            let _ = writeln!(body, "        {} = {cr} * {p0r} + ((-{ci}) * {p0i} + ({dr} * {p1r} + (-{di}) * {p1i}))", names[2]);
            let _ = writeln!(body, "        {} = {cr} * {p0i} + ({ci} * {p0r} + ({dr} * {p1i} + {di} * {p1r}))", names[3]);
            let [a, b, c, d] = names;
            cur[j0] = (a, b);
            cur[j1] = (c, d);
        };
        for g in &self.gates {
            match g {
                Gate::U(q, m) => {
                    let t = mask(self.level_of(*q));
                    for j0 in (0..size).filter(|j| j & t == 0) {
                        apply(&mut cur, m, j0, j0 | t, &mut body);
                    }
                }
                Gate::CU(c, q, m) => {
                    let (cm, t) = (mask(self.level_of(*c)), mask(self.level_of(*q)));
                    for j0 in (0..size).filter(|j| j & t == 0 && j & cm != 0) {
                        apply(&mut cur, m, j0, j0 | t, &mut body);
                    }
                }
                Gate::Swap(a, b) => {
                    // A swap moves values and computes nothing: a permutation of names.
                    let (ma, mb) = (mask(self.level_of(*a)), mask(self.level_of(*b)));
                    for j in (0..size).filter(|j| j & ma != 0 && j & mb == 0) {
                        cur.swap(j, j ^ ma ^ mb);
                    }
                }
            }
        }
        s += &body;
        for (j, (r, i)) in cur.iter().enumerate() {
            let _ = writeln!(s, "        qr{b} = {r}\n        qi{b} = {i}", b = bits(j));
        }
        s
    }

    /// Lower the generated source: the kernel is ordinary LYTH, checked and costed like any other.
    pub fn lower(&self, name: &str) -> Result<ir::KernelIr, String> {
        let src = self.source(name);
        let unit = parse(&src).map_err(|e| format!("{e}\n{src}"))?;
        ir::lower(&unit, &unit.kernels[0]).map_err(|e| format!("{e}\n{src}"))
    }
}

fn initial(qubits: u32) -> (Vec<f32>, Vec<f32>) {
    let n = 1usize << qubits;
    let mut re = vec![0.0f32; n];
    re[0] = 1.0;
    (re, vec![0.0; n])
}

/// The fused circuit on the host oracle, from |0...0>.
pub fn run_fused_host(qubits: u32, gates: &[Gate], k: usize) -> Result<(Vec<f32>, Vec<f32>), String> {
    let n = 1usize << qubits;
    let (mut re, mut im) = initial(qubits);
    for (gi, g) in fuse(gates, k).iter().enumerate() {
        let kir = g.lower(&format!("fused{gi}"))?;
        let mut inputs = eval::Inputs::default();
        inputs.extents.insert("n".into(), n as u32);
        for (l, w) in g.widths().into_iter().enumerate() {
            inputs.extents.insert(format!("w{l}"), w);
        }
        inputs.buffers.insert("re".into(), re);
        inputs.buffers.insert("im".into(), im);
        inputs.buffers.insert("qr".into(), vec![f32::NAN; n]);
        inputs.buffers.insert("qi".into(), vec![f32::NAN; n]);
        let mut out = eval::eval(&kir, n >> g.qubits.len(), &inputs).map_err(|e| e.to_string())?;
        re = out.buffers.remove("qr").unwrap();
        im = out.buffers.remove("qi").unwrap();
    }
    Ok((re, im))
}

fn launch_grid(walk: usize) -> u32 {
    (walk as u32).div_ceil(256).max(1)
}

/// The fused circuit on the GPU, from |0...0>, ping-ponging two state buffers. Returns the state
/// and the number of passes.
pub fn run_fused_gpu(ctx: &Context, qubits: u32, gates: &[Gate], k: usize) -> Result<(Vec<f32>, Vec<f32>, usize), CudaError> {
    let n = 1usize << qubits;
    let groups = fuse(gates, k);
    let msg = |e: String| CudaError::Message(e);
    let modules: Vec<_> = groups
        .iter()
        .enumerate()
        .map(|(gi, g)| {
            let kir = g.lower(&format!("fused{gi}")).map_err(msg)?;
            let m = lyth_ptx::emit(&kir, "sm_120").map_err(|e| msg(e.to_string()))?;
            ctx.load_ptx(&m.ptx).map(|module| (module, m.entry))
        })
        .collect::<Result<_, _>>()?;
    let funcs: Vec<_> = modules.iter().map(|(m, e)| m.function(e)).collect::<Result<_, _>>()?;
    let (re0, im0) = initial(qubits);
    let mut a: (Buffer, Buffer) = (ctx.upload(&re0)?, ctx.upload(&im0)?);
    let mut b: (Buffer, Buffer) = (ctx.alloc(n)?, ctx.alloc(n)?);
    for (g, f) in groups.iter().zip(&funcs) {
        let mut args = vec![Arg::U32(n as u32)];
        args.extend(g.widths().into_iter().map(Arg::U32));
        args.extend([Arg::Buf(&a.0), Arg::Buf(&a.1), Arg::Buf(&b.0), Arg::Buf(&b.1)]);
        f.launch(launch_grid(n >> g.qubits.len()), 256, &args)?;
        drop(args);
        std::mem::swap(&mut a, &mut b);
    }
    ctx.synchronize()?;
    Ok((a.0.download()?, a.1.download()?, groups.len()))
}

fn example(name: &str) -> Result<ir::KernelIr, CudaError> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(format!("../../examples/{name}.lyth"));
    let src = std::fs::read_to_string(&path).map_err(|e| CudaError::Message(format!("{}: {e}", path.display())))?;
    let unit = parse(&src).map_err(|e| CudaError::Message(e.to_string()))?;
    ir::lower(&unit, &unit.kernels[0]).map_err(|e| CudaError::Message(e.to_string()))
}

/// The inner width counts elements of the parent view, which has lost bit `c` (ADR-0029).
pub fn widths_ct(c: u32, t: u32) -> (u32, u32) {
    (1 << c, if t < c { 1 << t } else { 1 << (t - 1) })
}

/// The unfused circuit on the GPU: one launch of `gate_q`, `cu_q` or `swap_q` per gate.
pub fn run_unfused_gpu(ctx: &Context, qubits: u32, gates: &[Gate]) -> Result<(Vec<f32>, Vec<f32>), CudaError> {
    let n = 1usize << qubits;
    let kernels: Vec<ir::KernelIr> = ["gate_q", "cu_q", "swap_q"].into_iter().map(example).collect::<Result<_, _>>()?;
    let modules: Vec<_> = kernels
        .iter()
        .map(|k| {
            let m = lyth_ptx::emit(k, "sm_120").map_err(|e| CudaError::Message(e.to_string()))?;
            ctx.load_ptx(&m.ptx).map(|module| (module, m.entry))
        })
        .collect::<Result<_, _>>()?;
    let funcs: Vec<_> = modules.iter().map(|(m, e)| m.function(e)).collect::<Result<_, _>>()?;
    let (re0, im0) = initial(qubits);
    let mut a: (Buffer, Buffer) = (ctx.upload(&re0)?, ctx.upload(&im0)?);
    let mut b: (Buffer, Buffer) = (ctx.alloc(n)?, ctx.alloc(n)?);
    for g in gates {
        let (which, extents, m): (usize, Vec<(&str, u32)>, Option<&M>) = match g {
            Gate::U(q, m) => (0, vec![("w", 1 << q)], Some(m)),
            Gate::CU(c, t, m) => {
                let (wc, wt) = widths_ct(*c, *t);
                (1, vec![("wc", wc), ("wt", wt)], Some(m))
            }
            Gate::Swap(x, y) => {
                let (wc, wt) = widths_ct(*x, *y);
                (2, vec![("wc", wc), ("wt", wt)], None)
            }
        };
        let s: BTreeMap<&str, f32> = m
            .map(|m| ["ar", "ai", "br", "bi", "cr", "ci", "dr", "di"].into_iter().zip(f32s(m)).collect())
            .unwrap_or_default();
        let args: Vec<Arg> = kernels[which]
            .params
            .iter()
            .map(|p| match p.name.as_str() {
                "n" => Arg::U32(n as u32),
                "re" => Arg::Buf(&a.0),
                "im" => Arg::Buf(&a.1),
                "qr" => Arg::Buf(&b.0),
                "qi" => Arg::Buf(&b.1),
                other => match extents.iter().find(|(e, _)| *e == other) {
                    Some((_, v)) => Arg::U32(*v),
                    None => Arg::F32(s[other]),
                },
            })
            .collect();
        let walk = n >> if which == 0 { 1 } else { 2 };
        funcs[which].launch(launch_grid(walk), 256, &args)?;
        drop(args);
        std::mem::swap(&mut a, &mut b);
    }
    ctx.synchronize()?;
    Ok((a.0.download()?, a.1.download()?))
}

/// Circuits the tests and the measurement share, with a fixed generator.
pub mod circuits {
    use super::{Gate, M};

    const S: f64 = std::f64::consts::FRAC_1_SQRT_2;
    pub const H: M = [(S, 0.0), (S, 0.0), (S, 0.0), (-S, 0.0)];
    pub const X: M = [(0.0, 0.0), (1.0, 0.0), (1.0, 0.0), (0.0, 0.0)];
    pub const Z: M = [(1.0, 0.0), (0.0, 0.0), (0.0, 0.0), (-1.0, 0.0)];

    pub fn phase(phi: f64) -> M {
        [(1.0, 0.0), (0.0, 0.0), (0.0, 0.0), (phi.cos(), phi.sin())]
    }

    pub fn zyz(theta: f64, phi: f64, lambda: f64) -> M {
        let (c, s) = ((theta / 2.0).cos(), (theta / 2.0).sin());
        let e = |x: f64| (x.cos(), x.sin());
        let sc = |k: f64, z: (f64, f64)| (k * z.0, k * z.1);
        [(c, 0.0), sc(-s, e(lambda)), sc(s, e(phi)), sc(c, e(phi + lambda))]
    }

    /// SplitMix64.
    pub struct Rng(pub u64);
    impl Rng {
        pub fn next_u64(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        }
        pub fn angle(&mut self) -> f64 {
            (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64 * std::f64::consts::TAU
        }
        pub fn below(&mut self, n: u64) -> u64 {
            self.next_u64() % n
        }
    }

    pub fn ghz(n: u32) -> Vec<Gate> {
        let mut g = vec![Gate::U(0, H)];
        g.extend((1..n).map(|t| Gate::CU(0, t, X)));
        g
    }

    pub fn qft(n: u32, rng: &mut Rng) -> Vec<Gate> {
        let mut g: Vec<Gate> = (0..n).map(|q| Gate::U(q, zyz(rng.angle(), rng.angle(), rng.angle()))).collect();
        for j in (0..n).rev() {
            g.push(Gate::U(j, H));
            for k in (0..j).rev() {
                g.push(Gate::CU(k, j, phase(std::f64::consts::PI / f64::from(1u32 << (j - k)))));
            }
        }
        g.extend((0..n / 2).map(|i| Gate::Swap(i, n - 1 - i)));
        g
    }

    pub fn random(n: u32, depth: u32, rng: &mut Rng) -> Vec<Gate> {
        let mut g = Vec::new();
        for _ in 0..depth {
            g.extend((0..n).map(|q| Gate::U(q, zyz(rng.angle(), rng.angle(), rng.angle()))));
            let mut qs: Vec<u32> = (0..n).collect();
            for i in (1..qs.len()).rev() {
                qs.swap(i, rng.below(i as u64 + 1) as usize);
            }
            for p in qs.chunks(2).filter(|p| p.len() == 2) {
                let (c, t) = if rng.below(2) == 0 { (p[0], p[1]) } else { (p[1], p[0]) };
                g.push(Gate::CU(c, t, if rng.below(2) == 0 { X } else { Z }));
            }
        }
        g
    }
}
