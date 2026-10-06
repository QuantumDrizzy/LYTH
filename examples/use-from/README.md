# Using LYTH from another project

Four small projects, each the shape of a repository that is not this one. Every one was built and run
on an sm_120 device and checks its own answer (exit 0 on success).

| folder | language | what it shows |
|---|---|---|
| `rust/` | Rust | the generated `--bind-rust` module over `lyth-cuda`; written buffers take `&mut` |
| `cuda/` | C++ / CUDA | the generated `--bind-c` header inside a `.cu` that uses the CUDA runtime for memory |
| `python/` | Python | the generated `--bind-py` module (ctypes, no dependencies) with torch tensors |
| `circuit/` | Rust, quantum | `lyth-circuit`: GHZ(20) fused into 5 passes, checked against the exact state and the unfused run |

The bindings here were generated for `machine sm_120`. For your GPU, regenerate them against your own
machine file -- see [`docs/USING.md`](../../docs/USING.md).

```bash
cd rust    && cargo run --release
cd circuit && cargo run --release
cd cuda    && nvcc -O2 -o main main.cu -lcuda && ./main
cd python  && python use_saxpy.py
```
