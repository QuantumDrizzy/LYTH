// Hand-written CUDA C++, for the comparison `Nine Kernels, Measured` says it does not have.
//
// The point of this file is to be *fair*, which makes it harder to write than a fast one.
// A hand-written kernel that beats LYTH by using a different schedule has compared two
// schedules, not two compilers. So each kernel here is written to the schedule LYTH emits:
//
//   * `saxpy_gridstride`  - the grid-stride loop LYTH emits for an elementwise kernel, with
//                           the same `fma` contraction and the same bounds test.
//   * `saxpy_naive`       - one thread per element, no loop. Not LYTH's schedule; included
//                           because it is what a person writes first, and the gap between the
//                           two is a property of the schedule rather than of either compiler.
//   * `transpose_tiled`   - the four-phase tiled transpose, 32x32, with the +1 skew. This is
//                           the textbook kernel, and it is the one LYTH's `tile 32, 32` line
//                           is claiming to be equivalent to.
//   * `transpose_naive`   - the strided store LYTH derives 36 bytes per element for.
//
// Compiled to PTX with `nvcc -arch=sm_120 -ptx`, then loaded through the same driver-API path
// the generated LYTH bindings use, so the launch is identical down to `cuLaunchKernel`.
//
// `__restrict__` is deliberately absent. LYTH's IR has no aliasing annotation to emit, so
// granting one here would hand nvcc an optimisation the comparison is not about.

extern "C" __global__ void saxpy_gridstride(unsigned int n, float a,
                                            const float* x, float* y)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int stride = gridDim.x * blockDim.x;
    for (; i < n; i += stride) {
        y[i] = fmaf(a, x[i], y[i]);
    }
}

extern "C" __global__ void saxpy_naive(unsigned int n, float a,
                                       const float* x, float* y)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        y[i] = fmaf(a, x[i], y[i]);
    }
}

// 32x32 tile, one element per thread, 1024 threads per block -- the same launch shape LYTH
// derives from `tile 32, 32`. The `+ 1` is the skew: a column of the tile then walks a row
// stride coprime with the 32 banks, so 32 threads land on 32 distinct banks.
#define TILE 32

extern "C" __global__ void transpose_tiled(unsigned int rows, unsigned int cols,
                                           const float* a, float* b)
{
    __shared__ float tile[TILE][TILE + 1];

    unsigned int tiles_x = (cols + TILE - 1) / TILE;
    unsigned int total = ((rows + TILE - 1) / TILE) * tiles_x;
    unsigned int tx = threadIdx.x & (TILE - 1);
    unsigned int ty = threadIdx.x >> 5;

    for (unsigned int t = blockIdx.x; t < total; t += gridDim.x) {
        unsigned int trow = t / tiles_x;
        unsigned int tcol = t % tiles_x;

        unsigned int gy = trow * TILE + ty;
        unsigned int gx = tcol * TILE + tx;
        float v = 0.0f;
        if (gy < rows && gx < cols) v = a[(size_t)gy * cols + gx];
        tile[ty][tx] = v;

        __syncthreads();

        unsigned int oy = tcol * TILE + ty;
        unsigned int ox = trow * TILE + tx;
        if (oy < cols && ox < rows) b[(size_t)oy * rows + ox] = tile[tx][ty];

        __syncthreads();
    }
}

extern "C" __global__ void transpose_naive(unsigned int rows, unsigned int cols,
                                           const float* a, float* b)
{
    unsigned int k = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int stride = gridDim.x * blockDim.x;
    unsigned int total = rows * cols;
    for (; k < total; k += stride) {
        unsigned int i = k / cols;
        unsigned int j = k % cols;
        b[(size_t)j * rows + i] = a[k];
    }
}
