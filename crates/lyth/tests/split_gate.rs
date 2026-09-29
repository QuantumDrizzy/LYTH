//! ADR-0028 step 4: `hadamard_q`, a Hadamard gate on any qubit, on PTX and on the host oracle,
//! checked bit for bit against a dense reference that shares no code with the implementation.
//!
//! The reference walks blocks of `2w` elements and pairs element `p` of the first half with
//! element `p` of the second: for `w = 2^q` that is the definition of a single-qubit gate on
//! qubit `q` (amplitudes whose bit `q` is 0 against their partner with bit `q` set). It never
//! divides, takes a remainder, or calls `view_index`, so a wrong address rule cannot agree with
//! it by construction.

use std::collections::BTreeMap;
use std::path::PathBuf;

use lyth_cuda::{Arg, Context};
use lyth_lang::{eval, ir, parse};

const S: f32 = std::f32::consts::FRAC_1_SQRT_2;
/// A NaN with a payload nothing computes: an output element still holding it was never written.
const SENTINEL: u32 = 0x7FC0_DEAD;

fn kernel() -> ir::KernelIr {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../examples/hadamard_q.lyth");
    let unit = parse(&std::fs::read_to_string(path).unwrap()).unwrap();
    ir::lower(&unit, &unit.kernels[0]).unwrap()
}

/// Deterministic values of mixed sign and magnitude.
fn amplitudes(n: usize, seed: u32) -> Vec<f32> {
    let mut x = seed;
    (0..n)
        .map(|_| {
            x = x.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let m = ((x >> 8) as f32 / (1u32 << 24) as f32) - 0.5;
            m * f32::from_bits(0x3F80_0000 + ((x & 7) << 23)) // scaled by 1, 2, 4 .. 128
        })
        .collect()
}

/// The dense reference: (q0, q1) = (s (p0 + p1), s (p0 - p1)) over each pair, from first principles.
fn reference(re: &[f32], im: &[f32], w: usize) -> (Vec<f32>, Vec<f32>) {
    let n = re.len();
    let (mut qr, mut qi) = (vec![f32::from_bits(SENTINEL); n], vec![f32::from_bits(SENTINEL); n]);
    let mut base = 0;
    while base < n {
        for p in 0..w {
            let (i, j) = (base + p, base + p + w);
            qr[i] = S * (re[i] + re[j]);
            qi[i] = S * (im[i] + im[j]);
            qr[j] = S * (re[i] - re[j]);
            qi[j] = S * (im[i] - im[j]);
        }
        base += 2 * w;
    }
    (qr, qi)
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

fn host(k: &ir::KernelIr, re: &[f32], im: &[f32], n: usize, w: usize) -> (Vec<f32>, Vec<f32>) {
    let mut inputs = eval::Inputs::default();
    inputs.scalars.insert("s".into(), S);
    inputs.extents.insert("n".into(), n as u32);
    inputs.extents.insert("w".into(), w as u32);
    inputs.buffers.insert("re".into(), re.to_vec());
    inputs.buffers.insert("im".into(), im.to_vec());
    inputs.buffers.insert("qr".into(), vec![f32::from_bits(SENTINEL); n]);
    inputs.buffers.insert("qi".into(), vec![f32::from_bits(SENTINEL); n]);
    let out = eval::eval(k, n / 2, &inputs).expect("the host oracle evaluates a split kernel");
    (out.buffers["qr"].clone(), out.buffers["qi"].clone())
}

fn device(ctx: &Context, k: &ir::KernelIr, re: &[f32], im: &[f32], n: usize, w: usize) -> (Vec<f32>, Vec<f32>) {
    let module = lyth_ptx::emit(k, "sm_120").expect("ptx emits a split kernel");
    let loaded = ctx.load_ptx(&module.ptx).expect("the driver accepts the PTX");
    let func = loaded.function(&module.entry).expect("entry");
    let sentinel = vec![f32::from_bits(SENTINEL); n];
    let d_re = ctx.upload(re).unwrap();
    let d_im = ctx.upload(im).unwrap();
    let d_qr = ctx.upload(&sentinel).unwrap();
    let d_qi = ctx.upload(&sentinel).unwrap();
    // The signature carries the four bases, never the views: n, w, s, re, im, qr, qi.
    let args: Vec<Arg> = k
        .params
        .iter()
        .map(|p| match (p.name.as_str(), p.ty) {
            ("n", _) => Arg::U32(n as u32),
            ("w", _) => Arg::U32(w as u32),
            ("s", _) => Arg::F32(S),
            ("re", _) => Arg::Buf(&d_re),
            ("im", _) => Arg::Buf(&d_im),
            ("qr", _) => Arg::Buf(&d_qr),
            ("qi", _) => Arg::Buf(&d_qi),
            (other, _) => panic!("unexpected parameter `{other}`"),
        })
        .collect();
    let pairs = (n / 2) as u32;
    func.launch(pairs.div_ceil(256).max(1), 256, &args).expect("launch");
    (d_qr.download().unwrap(), d_qi.download().unwrap())
}

/// (n, w) pairs: every power-of-two qubit of a 4096-amplitude register, the top qubit as the
/// special case w = n/2, and widths that are not powers of two.
fn cases() -> Vec<(usize, usize)> {
    let mut v: Vec<(usize, usize)> = (0..12).map(|q| (4096, 1usize << q)).collect();
    v.extend([(384, 3), (384, 12), (384, 48), (384, 192), (1000, 5), (1000, 25), (2, 1), (64, 32)]);
    v
}

#[test]
fn hadamard_on_any_qubit_is_bit_exact_on_the_host_oracle_and_on_ptx() {
    let k = kernel();
    let ctx = Context::new(0).expect("cuda device");
    let mut checked = 0usize;
    for (n, w) in cases() {
        assert_eq!(n % (2 * w), 0, "the case list only holds valid launches");
        let (re, im) = (amplitudes(n, 7 + w as u32), amplitudes(n, 1000 + n as u32));
        let (want_r, want_i) = reference(&re, &im, w);
        assert!(!bits(&want_r).contains(&SENTINEL) && !bits(&want_i).contains(&SENTINEL), "the reference covers every element");

        let (hr, hi) = host(&k, &re, &im, n, w);
        assert_eq!(bits(&hr), bits(&want_r), "host oracle qr, n = {n}, w = {w}");
        assert_eq!(bits(&hi), bits(&want_i), "host oracle qi, n = {n}, w = {w}");

        let (dr, di) = device(&ctx, &k, &re, &im, n, w);
        assert_eq!(bits(&dr), bits(&want_r), "ptx qr, n = {n}, w = {w}");
        assert_eq!(bits(&di), bits(&want_i), "ptx qi, n = {n}, w = {w}");
        checked += 1;
    }
    println!("hadamard_q: {checked} launches bit-exact on the host oracle and PTX against the dense reference");
}

#[test]
fn a_launch_that_does_not_cover_the_buffer_is_refused_before_it_runs() {
    // ADR-0028: a wrong `w` is a refusal with the arithmetic printed, not a wrong answer.
    let k = kernel();
    let ex = BTreeMap::from([("n".to_string(), 1000u32), ("w".to_string(), 16u32)]);
    let err = k.split_pairs(&ex).unwrap_err().to_string();
    assert!(err.contains("2 * w = 32 must divide 1000"), "{err}");
}

// ------------------------------------------------------------------ the second machine (MTLB)

use lyth_lang::program::{PrintRange, Program};
use std::process::Command;

/// Runs an emitted program on the Unibit emulator and returns every float it printed.
fn emulator(asm: &str) -> Vec<f32> {
    let uni = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../Unibit");
    let dir = tempfile::tempdir().unwrap();
    let src = dir.path().join("split.uasm");
    std::fs::write(&src, asm).unwrap();
    let out = Command::new("cargo")
        .args(["run", "--quiet", "--release", "--", "run", src.to_str().unwrap()])
        .current_dir(&uni)
        .output()
        .expect("the Unibit emulator");
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    String::from_utf8_lossy(&out.stdout).lines().filter_map(|l| l.trim().parse::<f32>().ok()).collect()
}

#[test]
fn hadamard_on_any_qubit_is_bit_exact_on_the_unibit_emulator_too() {
    let k = kernel();
    const UNWRITTEN: f32 = 1.0e30;
    let mut checked = 0usize;
    let (mut vector, mut scalar) = (0usize, 0usize);
    for (n, w) in cases() {
        let (re, im) = (amplitudes(n, 7 + w as u32), amplitudes(n, 1000 + n as u32));
        let (want_r, want_i) = reference(&re, &im, w);
        let asm = lyth_uasm::emit(
            &k,
            &Program {
                n: (n / 2) as u32,
                extents: BTreeMap::from([("n".to_string(), n as u32), ("w".to_string(), w as u32)]),
                scalars: BTreeMap::from([("s".to_string(), S)]),
                prints: ["qr", "qi"]
                    .into_iter()
                    .map(|b| PrintRange { buffer: b.into(), lo: 0, hi: n as u32 })
                    .collect(),
                buffers: BTreeMap::from([
                    ("re".to_string(), re.clone()),
                    ("im".to_string(), im.clone()),
                    ("qr".to_string(), vec![UNWRITTEN; n]),
                    ("qi".to_string(), vec![UNWRITTEN; n]),
                ]),
            },
        )
        .expect("the MTLB back end emits a split kernel");
        // The body's loads read through the base pointers; the print epilogue has its own `lw`.
        let body_loads = |mn: &str| asm.lines().filter(|l| l.trim_start().starts_with(mn) && l.contains("(s0)")).count();
        if w % 8 == 0 {
            assert!(body_loads("lq") > 0 && body_loads("lw") == 0, "w = {w} should use whole registers");
            vector += 1;
        } else {
            assert!(body_loads("lw") > 0 && body_loads("lq") == 0, "w = {w} has no contiguous register load");
            scalar += 1;
        }
        let got = emulator(&asm);
        assert_eq!(got.len(), 2 * n, "n = {n}, w = {w}");
        assert_eq!(bits(&got[..n]), bits(&want_r), "emulator qr, n = {n}, w = {w}");
        assert_eq!(bits(&got[n..]), bits(&want_i), "emulator qi, n = {n}, w = {w}");
        checked += 1;
    }
    println!("hadamard_q: {checked} launches bit-exact on the Unibit emulator ({vector} whole-register, {scalar} one-f32-at-a-time)");
}

// ------------------------------------------------------------------ the generated binding

mod hadamard_q_binding {
    // Same reason as `binding.rs`: `grid` ends in `.max(1).min(MAX_GRID)` so that a malformed
    // machine file cannot make it panic; the lint fires only where the fixture is compiled here.
    // A generated launcher takes the kernel's own signature, which is nine arguments here.
    #![allow(clippy::manual_clamp, clippy::too_many_arguments)]
    include!("generated/hadamard_q.rs");
}

fn lyth_build(extra: &[&str]) -> std::process::Output {
    let repo = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    Command::new(env!("CARGO_BIN_EXE_lyth"))
        .args([
            "build",
            repo.join("examples/hadamard_q.lyth").to_str().unwrap(),
            "--machine",
            repo.join("fixtures/machine/sm_120.json").to_str().unwrap(),
            "-o",
            if cfg!(windows) { "nul" } else { "/dev/null" },
        ])
        .args(extra)
        .output()
        .expect("the compiler should run")
}

#[test]
fn the_generated_binding_launches_bit_exact_and_refuses_a_width_that_does_not_divide() {
    use hadamard_q_binding as g;
    let ctx = match Context::new(0) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("skipped: no CUDA device ({e})");
            return;
        }
    };
    let module = g::module(&ctx).expect("the embedded PTX should load");
    let kernel = g::HadamardQ::new(&module).expect("the entry point should resolve");

    let mut launched = 0usize;
    for (n, w) in cases() {
        let (re, im) = (amplitudes(n, 7 + w as u32), amplitudes(n, 1000 + n as u32));
        let (want_r, want_i) = reference(&re, &im, w);
        let sentinel = vec![f32::from_bits(SENTINEL); n];
        let (d_re, d_im) = (ctx.upload(&re).unwrap(), ctx.upload(&im).unwrap());
        let (mut d_qr, mut d_qi) = (ctx.upload(&sentinel).unwrap(), ctx.upload(&sentinel).unwrap());
        // The default grid: this is the pair-walking rule, not one thread per amplitude.
        kernel.launch(n as u32, w as u32, S, &d_re, &d_im, &mut d_qr, &mut d_qi).expect("launch");
        ctx.synchronize().unwrap();
        assert_eq!(bits(&d_qr.download().unwrap()), bits(&want_r), "generated qr, n = {n}, w = {w}");
        assert_eq!(bits(&d_qi.download().unwrap()), bits(&want_i), "generated qi, n = {n}, w = {w}");
        launched += 1;
    }

    // The grid is half the extent, one thread per pair.
    assert_eq!(g::grid(4096), Some(8), "2048 pairs at 256 per block");
    assert_eq!(g::grid(2), Some(1));

    // A width that does not divide is refused before the launch, with the arithmetic, and
    // nothing is written. n = 12, w = 8 is the case that would reach element 13 of a
    // 12-element buffer; the sentinel proves the device was never asked.
    let sentinel = vec![f32::from_bits(SENTINEL); 12];
    let (re, im) = (amplitudes(12, 1), amplitudes(12, 2));
    let (d_re, d_im) = (ctx.upload(&re).unwrap(), ctx.upload(&im).unwrap());
    let (mut d_qr, mut d_qi) = (ctx.upload(&sentinel).unwrap(), ctx.upload(&sentinel).unwrap());
    for (n, w, want) in [
        (12u32, 8u32, "2 * w = 16 must divide n = 12, but 12 mod 16 = 12"),
        (1000, 16, "2 * w = 32 must divide n = 1000, but 1000 mod 32 = 8"),
        (12, 0, "a block of no elements splits nothing"),
    ] {
        for launch in [
            kernel.launch(n, w, S, &d_re, &d_im, &mut d_qr, &mut d_qi),
            kernel.launch_with(1, n, w, S, &d_re, &d_im, &mut d_qr, &mut d_qi),
        ] {
            let err = launch.expect_err("a launch that does not cover the buffer must be refused").to_string();
            assert!(err.contains(want), "n = {n}, w = {w}: {err}");
        }
    }
    ctx.synchronize().unwrap();
    assert_eq!(bits(&d_qr.download().unwrap()), bits(&sentinel), "a refused launch wrote nothing");
    assert_eq!(bits(&d_qi.download().unwrap()), bits(&sentinel), "a refused launch wrote nothing");
    println!("hadamard_q binding: {launched} launches bit-exact at the default grid, 3 bad widths refused on both launchers");
}

#[test]
fn the_contract_travels_in_the_split_binding() {
    // Per pair, not per element: the kernel walks pairs, so 8 flops and 32 bytes are what one
    // thread moves for two amplitudes of each of four buffers (ADR-0028 P1: the split does not
    // change the payload, only which addresses carry it).
    use hadamard_q_binding as g;
    assert_eq!(g::MACHINE, "sm_120");
    assert_eq!(g::FLOPS_PER_ELEMENT, 8.0);
    assert_eq!(g::BYTES_PER_ELEMENT, 32.0);
    assert!((g::DERIVED_INTENSITY - 8.0 / 32.0).abs() < 1e-12);
    assert_eq!(g::DECLARED_INTENSITY, Some(0.25));
    assert!(g::PTX.contains(".visible .entry hadamard_q"));
}

#[test]
fn the_checked_in_split_binding_matches_the_generator() {
    let out = tempfile::Builder::new().suffix(".rs").tempfile().expect("a temporary file");
    let run = lyth_build(&["--bind-rust", out.path().to_str().unwrap()]);
    assert!(run.status.success(), "{}", String::from_utf8_lossy(&run.stderr));
    let fresh = std::fs::read_to_string(out.path()).unwrap();
    let checked_in =
        std::fs::read_to_string(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/generated/hadamard_q.rs")).unwrap();
    assert_eq!(
        fresh.replace("\r\n", "\n"),
        checked_in.replace("\r\n", "\n"),
        "tests/generated/hadamard_q.rs is stale; regenerate it with --bind-rust"
    );
}

#[test]
fn the_manifest_says_which_bases_are_written_and_that_the_grid_walks_pairs() {
    let out = tempfile::Builder::new().suffix(".json").tempfile().expect("a temporary file");
    let run = lyth_build(&["--manifest", out.path().to_str().unwrap()]);
    assert!(run.status.success(), "{}", String::from_utf8_lossy(&run.stderr));
    let m: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(out.path()).unwrap()).unwrap();

    // A split base is streamed only through its views; a manifest that looked the streams up by
    // the base name would call the outputs read-only and a generator would take them as `&`.
    let flag = |name: &str, field: &str| {
        m["params"].as_array().unwrap().iter().find(|p| p["name"] == name).unwrap()[field].as_bool().unwrap()
    };
    for o in ["qr", "qi"] {
        assert!(flag(o, "written") && !flag(o, "read"), "`{o}` is an output");
    }
    for i in ["re", "im"] {
        assert!(flag(i, "read") && !flag(i, "written"), "`{i}` is an input");
    }
    let splits = m["splits"].as_array().expect("a kernel that splits publishes its splits");
    assert_eq!(splits.len(), 4);
    assert!(splits.iter().all(|s| s["width"] == "w" && s["extent"] == "n"));
    assert_eq!(m["launch"]["grid"]["split_depth"], 1);
}

#[test]
fn a_kernel_that_does_not_split_publishes_no_trace_of_one() {
    // The new fields are absent, not empty or false, so every manifest written before ADR-0028
    // is byte-identical to one written now (and the checked-in saxpy bindings still match).
    let repo = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let out = tempfile::Builder::new().suffix(".json").tempfile().expect("a temporary file");
    let run = Command::new(env!("CARGO_BIN_EXE_lyth"))
        .args([
            "build",
            repo.join("examples/saxpy.lyth").to_str().unwrap(),
            "--machine",
            repo.join("fixtures/machine/sm_120.json").to_str().unwrap(),
            "-o",
            if cfg!(windows) { "nul" } else { "/dev/null" },
            "--manifest",
            out.path().to_str().unwrap(),
        ])
        .output()
        .expect("the compiler should run");
    assert!(run.status.success(), "{}", String::from_utf8_lossy(&run.stderr));
    let text = std::fs::read_to_string(out.path()).unwrap();
    assert!(!text.contains("\"splits\"") && !text.contains("\"split_depth\""), "{text}");
}

#[test]
fn the_c_and_python_generators_refuse_a_kernel_that_splits() {
    // They would launch twice the blocks and never refuse a width that does not divide, so a
    // caller would get out-of-bounds writes from generated code. Refused, with the reason.
    for flag in ["--bind-c", "--bind-py"] {
        let out = tempfile::Builder::new().tempfile().expect("a temporary file");
        let run = lyth_build(&[flag, out.path().to_str().unwrap()]);
        assert!(!run.status.success(), "{flag} must refuse a split kernel");
        let err = String::from_utf8_lossy(&run.stderr);
        assert!(err.contains("splits") && err.contains("--bind-rust"), "{flag}: {err}");
        assert_eq!(std::fs::metadata(out.path()).unwrap().len(), 0, "{flag} wrote nothing");
    }
}

// ------------------------------------------------------------------ `lyth run` on a split kernel

fn lyth_run(extra: &[&str]) -> std::process::Output {
    let repo = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    Command::new(env!("CARGO_BIN_EXE_lyth"))
        .args([
            "run",
            repo.join("examples/hadamard_q.lyth").to_str().unwrap(),
            "--machine",
            repo.join("fixtures/machine/sm_120.json").to_str().unwrap(),
        ])
        .args(extra)
        .output()
        .expect("the compiler should run")
}

#[test]
fn lyth_run_takes_the_width_from_set_walks_pairs_and_compares_the_views_base() {
    if Context::new(0).is_err() {
        eprintln!("skipped: no CUDA device");
        return;
    }
    let ok = lyth_run(&["-n", "4096", "--set", "w=16"]);
    let out = String::from_utf8_lossy(&ok.stdout);
    assert!(ok.status.success(), "{out}{}", String::from_utf8_lossy(&ok.stderr));
    // n is the length of the buffers and the kernel walks half of it, one thread per pair.
    assert!(out.contains("grid 8 x block 256 over 2048 pairs of 4096 elements"), "{out}");
    assert!(out.contains("BIT-EXACT against the IR evaluated on the host, 4096 elements"), "{out}");
    // The exact sector figure at this width, derived, and the line that says what it is not.
    assert!(out.contains("exact    32 byte per pair at this width (coalescence 1.000)"), "{out}");
    assert!(out.contains("whether the L1 serves them is what a measurement decides"), "{out}");
    // Below eight elements a warp straddles sectors: 2x, per view in isolation.
    let low = lyth_run(&["-n", "4096", "--set", "w=1"]);
    let low_out = String::from_utf8_lossy(&low.stdout);
    assert!(low.status.success(), "{low_out}");
    assert!(low_out.contains("exact    64 byte per pair at this width (coalescence 0.500)"), "{low_out}");
}

#[test]
fn lyth_run_refuses_a_split_without_a_width_or_with_one_that_does_not_divide() {
    // No device is needed to be refused: the launch check is arithmetic and runs first.
    for (args, want) in [
        (&["-n", "4096"][..], "no value was given for it. Pass `--set w=<u32>`"),
        (&["-n", "4000", "--set", "w=24"][..], "2 * w = 48 must divide 4000 but 4000 mod 48 = 16"),
        (&["-n", "4096", "--set", "w=2.5"][..], "must be a whole number of elements"),
    ] {
        let run = lyth_run(args);
        assert!(!run.status.success(), "{args:?} must be refused");
        let err = String::from_utf8_lossy(&run.stderr);
        assert!(err.contains(want), "{args:?}: {err}");
        assert!(!String::from_utf8_lossy(&run.stdout).contains("BIT-EXACT"), "{args:?}: nothing ran");
    }
}
