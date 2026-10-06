// A quantum circuit from another repository: GHZ(20) on the GPU, fused, against the exact state.
use lyth_circuit::{circuits, run_fused_gpu, run_unfused_gpu};

fn main() -> Result<(), lyth_cuda::CudaError> {
    let n = 20;
    let gates = circuits::ghz(n);
    let ctx = lyth_cuda::Context::new(0)?;
    let (re, im, passes) = run_fused_gpu(&ctx, n, &gates, 5)?;
    let (ure, uim) = run_unfused_gpu(&ctx, n, &gates)?;
    let s = std::f32::consts::FRAC_1_SQRT_2;
    let last = (1usize << n) - 1;
    let rest: f32 = re.iter().zip(&im).enumerate().filter(|(i, _)| *i != 0 && *i != last).map(|(_, (a, b))| a * a + b * b).sum();
    let same = re == ure && im == uim;
    println!("GHZ({n}): {} gates in {passes} fused passes; amp|0..0> {:.6} amp|1..1> {:.6} (1/sqrt2 = {s:.6}); \
              weight elsewhere {rest:e}; fused == unfused bit for bit: {same}", gates.len(), re[0], re[last]);
    std::process::exit(if same && (re[0] - s).abs() < 1e-6 && (re[last] - s).abs() < 1e-6 && rest == 0.0 { 0 } else { 1 });
}
