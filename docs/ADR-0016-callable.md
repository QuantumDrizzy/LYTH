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
| 4 | Python: `cuda-python` or `cupy` loader from the same manifest | a numpy round trip matches the host reference |
| 5 | ADR-0015 (shape), which now regenerates typed bindings | transpose is callable with its shapes checked |

Rust first because that is where the kernels this compiler exists for are being written.

## What this does not solve

Performance. The generated kernel is this compiler's PTX, which is naive, and calling it from
Rust does not make it faster. A caller who needs a fast kernel still writes CUDA. What the
caller gets here is a kernel whose cost is stated and checked, which is a different thing and
the only thing this project has ever claimed.
