// A CUDA project that calls a LYTH kernel: runtime API for memory, the generated header to launch.
#define LYTH_SAXPY_IMPLEMENTATION
#include "saxpy.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <vector>
#include <cmath>

int main() {
    const unsigned n = 1u << 20;
    std::vector<float> x(n), y(n), out(n);
    for (unsigned i = 0; i < n; ++i) { x[i] = 0.001f * i; y[i] = 1.0f - 0.0005f * i; }
    float *dx, *dy;
    cudaMalloc(&dx, n * 4); cudaMalloc(&dy, n * 4);          // the runtime makes the primary context current
    cudaMemcpy(dx, x.data(), n * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(dy, y.data(), n * 4, cudaMemcpyHostToDevice);
    CUmodule mod; CUfunction fn;
    if (lyth_saxpy_load(&mod, &fn) != CUDA_SUCCESS) { std::puts("load failed"); return 1; }
    if (lyth_saxpy_launch(fn, n, 2.5f, (CUdeviceptr)dx, (CUdeviceptr)dy) != CUDA_SUCCESS) { std::puts("launch failed"); return 1; }
    cudaMemcpy(out.data(), dy, n * 4, cudaMemcpyDeviceToHost);
    double worst = 0;
    for (unsigned i = 0; i < n; ++i) worst = std::fmax(worst, std::fabs(out[i] - std::fmaf(2.5f, x[i], y[i])));
    std::printf("C++/CUDA: max |y - fma(2.5, x, y)| = %g, contract %.4f flop/byte\n", worst, LYTH_SAXPY_DERIVED_INTENSITY);
    return worst == 0 ? 0 : 1;
}
