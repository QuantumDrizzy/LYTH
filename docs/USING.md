# Using LYTH in your own project

LYTH compiles a `.lyth` kernel to PTX and hands you the same kernel as a Rust module, a C/C++ header
and a Python module, each with the cost contract embedded. This page is the whole path, from a
fresh clone to a kernel launched from your code. Every recipe here was run; where something has not
been run, it says so.

## 1. Install

```bash
git clone https://github.com/QuantumDrizzy/LYTH && cd LYTH
cargo install --path crates/lyth          # the `lyth` binary, or use `cargo run -p lyth --` from the clone
```

`check` and `build` need no GPU. Today the `lyth` binary links the CUDA driver at load time, so:

- **Windows**: builds with Rust stable alone (`raw-dylib`, no toolkit), but starts only where an NVIDIA
  driver is installed -- even for `check`.
- **Linux**: linking needs `libcuda.so`, which the CUDA toolkit provides. Not yet built or run there.

`[KNOWN_LIMIT]` Both go away when `lyth-cuda` loads the driver at run time; that is the next change.

## 2. Describe your GPU: the machine file

Every number LYTH prints -- the roofline, the ceiling that binds, `% of peak` -- is measured against a
**machine file**, and the repository ships one only for the GPU it was built on (`sm_120`).
To make one for yours, from measurement rather than a datasheet:

```bash
python tools/peak_probe.py --machine-out fixtures/machine/sm_86.json     # needs torch with CUDA
```

It times three streaming patterns and a cuBLAS SGEMM/HGEMM with CUDA events, reads the block limits
from the driver, and writes a `lyth-machine/0.1` file. What it does not measure (L2, shared-memory
and register rates, the `ops` table) is left empty on purpose: an empty field is a refusal, a guessed
one would be a datasheet by another name.

Then say which machine the kernel is for, in the source:

```
machine sm_86
```

The compiler refuses a source whose `machine` does not match the file. That is the contract, not an
inconvenience: the intensity and the ceilings belong to a device.

**Targets.** The PTX back end knows `sm_75`, `sm_80`, `sm_86`, `sm_89`, `sm_90`, `sm_100` and `sm_120`,
each with the PTX ISA version its `ptxas` accepts. Seven examples (saxpy, matmul, dot, a tiled
transpose, a split, and the `hadamard_q` and `cu_q` quantum gates) were compiled for sm_75, sm_80,
sm_86, sm_89, sm_90 and sm_120 and **accepted by `ptxas` for every target** (42/42, CUDA 13.0).
`[KNOWN_LIMIT]` They have only been **run** on sm_120. Anything else not yet run is a
bug report waiting to happen; see section 5.

## 3. Build a kernel and its bindings

```bash
lyth check examples/saxpy.lyth --machine fixtures/machine/sm_86.json                # the contract, no GPU
lyth build examples/saxpy.lyth --machine fixtures/machine/sm_86.json \
    -o saxpy.ptx --bind-rust saxpy.rs --bind-c saxpy.h --bind-py saxpy.py
lyth run   examples/saxpy.lyth --machine fixtures/machine/sm_86.json -n 1048576 --set a=2.5   # launch + bit-exact check
```

## 4. Call it

Working projects for each of these live in [`examples/use-from/`](../examples/use-from/).

### Rust

```toml
[dependencies]
lyth-cuda = { git = "https://github.com/QuantumDrizzy/LYTH" }
```

```rust
mod saxpy;                                     // the generated --bind-rust file
let ctx = lyth_cuda::Context::new(0)?;
let module = saxpy::module(&ctx)?;
let kernel = saxpy::Saxpy::new(&module)?;
let x = ctx.upload(&host_x)?;
let mut y = ctx.upload(&host_y)?;
kernel.launch(n, 2.5, &x, &mut y)?;            // y is written, so it is &mut: aliasing does not compile
```

### C, C++ and CUDA

The header is single-file (stb style). Define the implementation macro in **one** translation unit:

```c
#define LYTH_SAXPY_IMPLEMENTATION
#include "saxpy.h"              /* needs cuda.h; link -lcuda (cuda.lib on Windows) */

CUmodule mod; CUfunction fn;
lyth_saxpy_load(&mod, &fn);
lyth_saxpy_launch(fn, n, 2.5f, (CUdeviceptr)x, (CUdeviceptr)y);
```

It is `extern "C"`, so it drops into C++ unchanged. Inside a CUDA project that uses the runtime API
(`cudaMalloc`), the runtime has already made the primary context current, and the header launches
into it. `examples/use-from/cuda/main.cu` does exactly that.

### Python

```python
import torch, saxpy                            # the generated --bind-py file: ctypes, no dependencies
saxpy.Kernel().launch(n, 2.5, saxpy.from_torch(x, n), saxpy.from_torch(y, n))
```

It takes device pointers, so anything that hands you one works: torch, cupy, your own allocator.
`from_torch` refuses what the kernel cannot address (wrong dtype, a strided view, too short).

### Quantum circuits

`lyth-circuit` turns a gate list into state-vector kernels, fuses runs of gates into one pass over
memory, and keeps the unfused run as the bit-for-bit reference. It is a Rust API:

```toml
lyth-circuit = { git = "https://github.com/QuantumDrizzy/LYTH" }
lyth-cuda    = { git = "https://github.com/QuantumDrizzy/LYTH" }
```

```rust
use lyth_circuit::{circuits, run_fused_gpu, Gate};
let gates = circuits::ghz(20);                                   // or your own Vec<Gate>: U, CU, Swap
let (re, im, passes) = run_fused_gpu(&ctx, 20, &gates, 5)?;      // amplitudes, and how many passes it took
```

From Python or C, the individual gate kernels (`examples/hadamard_q.lyth`, `gate_q.lyth`, `cu_q.lyth`)
are ordinary LYTH kernels: build them with `--bind-py` / `--bind-c` like any other.
Circuits against Qiskit agree to ≤ 6.6e-8 (ADR-0029); fusing changes no bit (ADR-0030).

## 5. When something fails

Open an issue with the **bug report** template. It asks for the things that make a report reproducible:
the `.lyth` source, the exact command, the machine file, the GPU, the driver version and the full
output. A kernel that compiles and gives a wrong answer, a contract that disagrees with what `ncu`
measures, or a GPU where a target was only assumed to work are the most valuable reports there are.

If you generated a machine file for a GPU that is not in `fixtures/machine/`, the **machine file**
template is for sending it in.
