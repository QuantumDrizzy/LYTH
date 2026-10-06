// A Rust project that calls a LYTH kernel through the generated binding.
mod saxpy;

fn main() -> Result<(), lyth_cuda::CudaError> {
    let n = 1u32 << 20;
    let x: Vec<f32> = (0..n).map(|i| 0.001 * i as f32).collect();
    let y: Vec<f32> = (0..n).map(|i| 1.0 - 0.0005 * i as f32).collect();
    let ctx = lyth_cuda::Context::new(0)?;
    let module = saxpy::module(&ctx)?;
    let kernel = saxpy::Saxpy::new(&module)?;
    let dx = ctx.upload(&x)?;
    let mut dy = ctx.upload(&y)?;
    kernel.launch(n, 2.5, &dx, &mut dy)?;
    ctx.synchronize()?;
    let out = dy.download()?;
    let worst = out.iter().zip(x.iter().zip(&y)).map(|(o, (a, b))| (o - 2.5f32.mul_add(*a, *b)).abs()).fold(0f32, f32::max);
    println!("Rust: max |y - fma(2.5, x, y)| = {worst}, contract {:.4} flop/byte", saxpy::DERIVED_INTENSITY);
    std::process::exit(if worst == 0.0 { 0 } else { 1 });
}
