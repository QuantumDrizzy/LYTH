// A hand-written coarsened matmul, for ADR-0021 step 5.
//
// ADR-0020 measured LYTH emitting 17% more instructions than nvcc on a saxpy and losing nothing
// for it, and closed by naming the debt rather than paying it:
//
//   > It would decide a compute-bound kernel's time, and this language cannot write one. So 17%
//   > is a debt recorded rather than a debt paid, and the ADR that raises intensity is where it
//   > comes due.
//
// This is that kernel. At `tile 64, 64` + `coarsen 2, 2` the inner loop is four multiply-adds
// on four shared loads instead of one on two, which is the densest arithmetic this language can
// express -- 16 flop/byte, against 8 before and a ridge of 36.9.
//
// FAIRNESS, which is again the hard part:
//
//   * **Same schedule, instruction for instruction.** Grid-stride over output tiles, `tx`/`ty`
//     from the block's width, two skewed tiles back to back in dynamic shared memory, a step
//     loop over `p`, `C x C` staged elements per operand, two `__syncthreads()` in the same two
//     places, a term loop bounded by `min(T, k - p)` rather than a zero fill, `C x C`
//     accumulators in registers, `C x C` guarded stores. Every one of those is read off
//     `crates/lyth-ptx/src/contracted.rs`, not invented here.
//   * **No `__restrict__`**, for ADR-0020's reason: LYTH's IR has no aliasing annotation to
//     emit, so granting nvrtc one compares two languages rather than two compilers.
//   * **`extern __shared__`**, the same dynamic allocation LYTH uses, launched with the same
//     byte count -- so ptxas sees the same thing on both sides.
//   * **`--fmad=false`**, which is the one option that has to be passed. LYTH emits `mul.rn.f32`
//     then `add.rn.f32` deliberately (ADR-0010: one rounding is a different answer from two),
//     so an nvrtc left to contract these into `fma.rn.f32` would be computing something else.
//     The comparison that matters is the one where both kernels produce the same bits, and the
//     harness checks exactly that before it times anything.
//
//     The default-`fmad` build is compiled too and reported beside it. That number is not a
//     code-generation result: it is the price of ADR-0010's rounding rule, measured, and it
//     belongs to that decision rather than to this one.
//
// Global indices are computed in 32 bits and widened at the load, because that is what
// `mad.lo.u32` + `mul.wide.u32` does in the emitter. `(size_t)` here -- which is what
// `handwritten.cu` uses -- would hand LYTH a cheaper index than the kernel it is being
// compared to, in LYTH's favour.
//
// What is deliberately NOT constrained is unrolling. nvrtc may unroll the term loop and LYTH
// does not, and that is code generation rather than schedule -- no different bytes move and the
// accumulation order is unchanged, which is why the bit-exactness check still has to pass.

// The tile's shared stride: skewed by one so a column walk lands on 32 distinct banks, which is
// what LYTH derives rather than what a person would remember to do.
template <unsigned T, unsigned C>
__device__ __forceinline__ void matmul_tiled(unsigned int m, unsigned int n, unsigned int k,
                                             const float* a, const float* b, float* c)
{
    extern __shared__ float smem[];

    const unsigned int stride = T + 1;
    const unsigned int bw = T / C;              // the block's width, not the tile's
    float* as = smem;
    float* bs = smem + T * stride;

    const unsigned int tx = threadIdx.x & (bw - 1);
    const unsigned int ty = threadIdx.x / bw;

    const unsigned int tiles_n = (n + T - 1) / T;
    const unsigned int total = ((m + T - 1) / T) * tiles_n;
    const unsigned int steps = (k + T - 1) / T;

    for (unsigned int t = blockIdx.x; t < total; t += gridDim.x) {
        const unsigned int trow = t / tiles_n;
        const unsigned int tcol = t % tiles_n;

        float acc[C][C];
        for (unsigned int u = 0; u < C; ++u)
            for (unsigned int v = 0; v < C; ++v)
                acc[u][v] = 0.0f;

        for (unsigned int s = 0; s < steps; ++s) {
            const unsigned int pbase = s * T;

            // Phase 1: each thread stages C x C elements of each operand, at its own places in
            // the tile. Predicated rather than branched, so the block stays together.
            for (unsigned int u = 0; u < C; ++u) {
                for (unsigned int v = 0; v < C; ++v) {
                    const unsigned int ay = trow * T + ty + u * bw;
                    const unsigned int ax = pbase + tx + v * bw;
                    float va = 0.0f;
                    if (ay < m && ax < k) va = a[ay * k + ax];
                    as[(ty + u * bw) * stride + tx + v * bw] = va;

                    const unsigned int by = pbase + ty + u * bw;
                    const unsigned int bx = tcol * T + tx + v * bw;
                    float vb = 0.0f;
                    if (by < k && bx < n) vb = b[by * n + bx];
                    bs[(ty + u * bw) * stride + tx + v * bw] = vb;
                }
            }

            __syncthreads();

            // Phase 2: the terms of this step. `min(T, k - p)` and not a zero fill -- a zero
            // term cannot change a sum and does clamp a maximum, and the emitter has one rule
            // for both combinators.
            const unsigned int inner = (k - pbase) < T ? (k - pbase) : T;
            for (unsigned int tt = 0; tt < inner; ++tt) {
                float av[C], bv[C];
                for (unsigned int u = 0; u < C; ++u) av[u] = as[(ty + u * bw) * stride + tt];
                for (unsigned int v = 0; v < C; ++v) bv[v] = bs[tt * stride + tx + v * bw];
                for (unsigned int u = 0; u < C; ++u)
                    for (unsigned int v = 0; v < C; ++v)
                        acc[u][v] += av[u] * bv[v];
            }

            __syncthreads();
        }

        for (unsigned int u = 0; u < C; ++u) {
            for (unsigned int v = 0; v < C; ++v) {
                const unsigned int gi = trow * T + ty + u * bw;
                const unsigned int gj = tcol * T + tx + v * bw;
                if (gi < m && gj < n) c[gi * n + gj] = acc[u][v];
            }
        }
    }
}

// `tile 16, 16` -- half the tile, so twice the global traffic per output and *the same* number
// of shared-load instructions per output as `tile 32`. That asymmetry is what makes it the
// discriminator ADR-0021 step 5 needed: coarsening halves both quantities and cannot tell them
// apart, and this tells them apart.
extern "C" __global__ void matmul_t16(unsigned int m, unsigned int n, unsigned int k,
                                      const float* a, const float* b, float* c)
{
    matmul_tiled<16, 1>(m, n, k, a, b, c);
}

// `tile 32, 32` -- one thread per tile element, 1024 threads, 8 flop/byte. What LYTH could
// already emit before ADR-0021.
extern "C" __global__ void matmul_t32(unsigned int m, unsigned int n, unsigned int k,
                                      const float* a, const float* b, float* c)
{
    matmul_tiled<32, 1>(m, n, k, a, b, c);
}

// `tile 64, 64` + `coarsen 2, 2` -- the same 1024 threads over four times the tile, four
// accumulators each, 16 flop/byte. The kernel this ADR exists for.
extern "C" __global__ void matmul_t64c2(unsigned int m, unsigned int n, unsigned int k,
                                        const float* a, const float* b, float* c)
{
    matmul_tiled<64, 2>(m, n, k, a, b, c);
}
