# ADR-0016 — A kernel you can call

**Status:** Accepted
**Date:** 2026-09-16
**Depends on:** ADR-0010 (executable LYTH), ADR-0006 (kernel IR)
**Takes priority over:** ADR-0015 (shape), which is correct and comes second

## The gap nobody wrote down

This repository has fifteen ADRs about whether the compiler's numbers are true and **none about
whether the compiler is usable**. That is not a small omission; it is the reason a `.lyth` file
cannot appear in a real project today.

`lyth run` compiles a kernel, JITs it, verifies it and prints a report — inside its own process,
after which everything it built is discarded. There is no artifact, no header, no symbol, no
foreign function interface. A Rust or C++ program cannot call a LYTH kernel. Neither can Python.
So the language cannot be used for work, however correct its cost model is.

The blocker is not shape, and not syntax. **It is that there is nothing to link against.**

## What LYTH is, stated so the scope stops drifting

LYTH is not a general-purpose language and will not become one. It has no I/O, no allocation, no
composition, and needs none. What it is:

> **The one place where a kernel, its data layout, and its cost contract live — compiled to an
> artifact that Rust, C++ and Python all call.**

Today the same kernel gets written more than once: CUDA for the C++ path, something else for the
Python path, plus hand-written FFI glue for each. The cost model, if it exists at all, lives in a
comment or a slide. LYTH replaces that with one source file, generated bindings, and a contract
that travels with the binary and can be re-checked on whatever machine it lands on.

**Prior art, honestly:** Halide compiles ahead of time and emits a C++ header; Futhark emits a C
library with a C API and has wrappers for other languages. The mechanism here is not novel and
this ADR is not claiming it is. What neither of them carries is the checked cost contract. That
is the part that is ours, and it only means anything once the artifact is callable.

## Decision

`lyth build` gains a manifest and binding generation. Three rules fix the shape of all of it.

**1. LYTH never owns memory.** The caller passes device pointers. No allocator, no runtime, no
hidden copies, no lifetime story. This keeps a LYTH kernel composable with `cudaMalloc`, with
`torch.Tensor.data_ptr()`, with `cupy`, and with whatever Rust wrapper is in use, and it keeps
the generated code small enough to read. It is also the only choice that does not drag a runtime
into a language whose whole argument is that it adds nothing you cannot account for.

**2. The launch configuration is generated, and overridable.** The grid heuristic, the block
width, the shared-memory bytes and the partial-buffer sizing for a reduction are all derived
facts the caller should not have to rediscover. They are emitted as part of the binding, with an
explicit override for the caller who is sweeping them.

**3. The contract is part of the artifact.** The generated binding carries the declared
intensity, the derived bytes and flops per element, and the machine id it was checked against,
as constants. A kernel that is bandwidth-bound on sm_120 says so in the code that calls it, and
a build on different hardware can assert against it instead of finding out from a profiler six
months later. This is the whole argument of the project, reduced to something a CI job can read.

## The manifest

`lyth build --manifest k.json` emits the kernel's signature, which the evidence schema does not
carry: name, parameters in declaration order with their types, which buffer is written, which is
a reduction target and therefore sized by the grid rather than by `n`, the required shared
memory, and the launch constraints. Bindings are generated from the manifest, never from the IR
directly, so a third-party generator for a language this repository does not ship is possible
without touching the compiler.

## Build sequence

| step | | done when |
|---|---|---|
| 1 | the manifest: signature, launch facts, contract | `lyth build --manifest` round-trips every example |
| 2 | Rust binding: embedded PTX, typed launch over `lyth-cuda` | a Rust test calls saxpy and matches `lyth run` |
| 3 | C header: `cuModuleLoadData` + `cuLaunchKernel`, no C++ required | a C file compiles and links against it |
| 4 | Python: `ctypes` against the driver, from the same manifest | it launches and the result matches |
| 5 | ADR-0015 (shape), which now regenerates typed bindings | transpose is callable with its shapes checked |

All five are done. Steps 3 and 4 were driven against the device by hand, since neither can be
built by `cargo test` without a C toolchain and `cuda.h`:

```
cl /W4 /I "%CUDA%\include" use_sum.c /link cuda.lib
  blocks=256  contract: 0.25 flop/byte on sm_120
  host=1067171840.0  device=1067171840.0  MATCH

python use_saxpy.py
  contract: 0.1667 flop/byte on sm_120
  grid=256  mismatches=0 of 65536  MATCH
```

The C header produced no warnings of its own at `/W4`. What the suite holds afterwards is that
the generator has not changed its output, plus a parse of the generated Python, which needs no
driver.

### Two things the other languages cannot have

The Rust binding takes a written buffer by `&mut`, so the borrow checker refuses a launch that
aliases an input with an output. C and Python have no way to say that. **The C header does not
pretend to**: `const CUdeviceptr` would look like the guarantee without being it, because
`CUdeviceptr` is an integer handle and the qualifier would apply to the handle rather than the
memory. Both bindings name the written buffers in a comment and leave the signature honest.

Python's binding has one place it could be silently, catastrophically wrong: the driver reads
the parameter buffer by offset, so an argument passed at the wrong width shifts every argument
after it. Each is built with an explicit `ctypes` type and a test asserts on which.

### The API that moved underneath

`cuCtxCreate` is `_v4` in CUDA 13 and takes a params pointer, and a caller that passes the old
three arguments gets a success code and a context the next call rejects as invalid. Neither
binding creates a context -- that is the caller's, by rule 1 -- but the worked examples use
`cuDevicePrimaryCtxRetain` instead, which is stable across versions and is what a caller sharing
a device with torch or cupy wants anyway.

Rust first because that is where the kernels this compiler exists for are being written.

## What this does not solve

Performance. The generated kernel is this compiler's PTX, which is naive, and calling it from
Rust does not make it faster. A caller who needs a fast kernel still writes CUDA. What the
caller gets here is a kernel whose cost is stated and checked, which is a different thing and
the only thing this project has ever claimed.
