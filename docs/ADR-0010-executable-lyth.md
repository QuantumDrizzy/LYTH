# ADR-0010 — Making a `.lyth` file run

**Status:** Accepted
**Date:** 2026-09-14
**Depends on:** ADR-0001 (thesis), ADR-0005 (intensity), ADR-0006 (kernel IR), ADR-0009 (measured traffic)

## The gap this closes, stated honestly

ADR-0001's phase table reads *"Fase 2: parser + intensity typecheck"*, which implies the middle
of a compiler exists and only a front end is missing. It does not exist. What Fase 1c built is:

```rust
pub struct ArithOp { at: String, flops: f64, note: String }
pub struct Stream  { name, from, to, bytes: f64, dir, via, once }
```

That is a **cost model, not semantics**. No instruction can be generated from `flops: 6.0` and no
address from `bytes: 4.0`. **LYTH has never emitted a single instruction.** Four things are
missing, not one: a semantic IR, a front end, a back end, and a host loader.

This ADR builds all four, for the smallest language that is still the thesis.

## Why the back end first, and why this does not jump the ADR-0001 gate

The parser question — macros or a real frontend — is about *how the declaration is written*. The
back end question is *whether LYTH can produce anything at all*. The second can kill the project
and nobody has touched it. Building a parser first is a front door on a building with no rooms.

The ADR-0001 three-week gate stays open and untouched: this work adds a front end **and** a back
end, and the gate is about which *kind* of front end wins. The evidence in `docs/DOGFOOD.md`
keeps accumulating either way.

## What v1 of the language is, and what it deliberately is not

**In:** elementwise kernels over 1-D buffers. One element per thread. Scalar parameters, buffer
parameters, streams between `dram` and `reg`, arithmetic at `reg`, one store.

**Out of v1, on purpose:** reductions, shared memory, control flow inside the body, multiple
dimensions, atomics, anything sparse. Every one of them is a real language feature and none of
them is needed to answer the question this slice exists to answer.

That covers SAXPY, scale, vector add, and fused chains of them. It does not cover `k_integrate`
(branches, atomics, hashes) and is not meant to.

## The surface

```
machine sm_120

kernel saxpy(n: u32, a: f32, x: [f32], y: [f32])
    intensity 0.1667

    stream x : dram -> reg
    stream y : dram -> reg, drain

    at reg:
        y = a * x + y
```

Movement is declared first and arithmetic is subordinate to it — the inversion ADR-0001 asks
for. **You cannot operate on a buffer you have not streamed.** The compiler does not infer
movement; it checks that the arithmetic fits the movement declared.

## The part that makes this the thesis rather than a toy

Until now `intensity-check` compared a declared number against a hand-written accounting: two
declarations, checked against each other. The body was prose.

Here the body is **code**, so the compiler derives bytes and flops **from the AST**:

- every `stream` that is read contributes its element size
- every stream marked `drain` contributes its element size again, as a store
- every arithmetic node contributes its flops, with `a*x + y` recognised as one `fma` = 2 flops

`intensity` in the source is then checked against a number nobody typed. A mismatch is a
**compile error** naming the machine's ridge. That is "arithmetic intensity is a type, not a
comment" for the first time.

And ADR-0009 closes the loop the rest of the way: the compiler's own derived byte model is
checked against measured DRAM traffic **of the kernel it generated**. No human writes that
accounting, so no human can fudge it.

## Build order — each stage ships something testable

| stage | ships | done when |
|---|---|---|
| 1 | `lyth-lang`: lexer, parser, AST | `saxpy.lyth` parses; a syntax error names line and column |
| 2 | `lyth-lang`: semantic IR + derived cost | derived intensity = 0.1667; a lying `intensity` is refused, ridge named |
| 3 | `lyth-ptx`: IR → PTX | emitted PTX is accepted by `ptxas -arch=sm_120` |
| 4 | `lyth-cuda`: Driver API FFI | `cuModuleLoadData` + `cuLaunchKernel`; result is bit-exact against a CPU reference |
| 5 | `lyth` CLI: `build` / `run` / `check` | `lyth run saxpy.lyth` prints a verdict and a checked result |
| 6 | traffic | `ncu` on the generated kernel; derived model CONFIRMED against silicon |

Stage 6 is the point of the whole thing. Before it, this is a compiler. After it, it is a
compiler that cannot lie about what it produced.

## Crates

```
lyth-probe   unchanged — evidence, machine, intensity, ncu
lyth-lang    lexer, parser, AST, semantic IR, cost derivation
lyth-ptx     IR -> PTX text. No dependency on CUDA.
lyth-cuda    Driver API FFI. Unsafe confined here, behind a safe wrapper.
lyth         the binary: build, run, check
```

`lyth-ptx` not depending on CUDA matters: PTX emission stays testable on a machine with no GPU,
and `ptxas` validation is a separate, optional step.

## Backend choice

PTX text, loaded with the CUDA Driver API (`cuModuleLoadData`), as ADR-0001 chose. Not CUDA C
through `nvcc`: shelling out to a C compiler would mean LYTH never emits an instruction, and the
one question this slice exists to answer would stay unanswered.

## What would falsify this

- **If the emitted PTX cannot be made bit-exact against a CPU reference for SAXPY**, the back end
  is not trustworthy and no amount of front-end work matters. Stop and publish.
- **If the derived intensity cannot be made to agree with measured traffic on a kernel this
  compiler generated**, then deriving cost from an AST does not work, and the thesis —
  intensity as a checked type — is wrong at its root. That is the result worth publishing most.
- If v1's restrictions (no reductions, no control flow) turn out to make the language unable to
  express anything anyone would write, the scope was chosen wrong and the next ADR says so.

## [KNOWN LIMIT], written before the code

- One element per thread, bounds-checked. No grid-stride loop in v1, so occupancy tuning is not
  expressible and any performance number from it is uninteresting. **This slice is about
  correctness and honest cost, not speed.** No `X times faster` claim will come out of it.
- `fma` contraction is recognised structurally (`a * b + c`), which is a pattern match, not a
  numerical guarantee. Whether `ptxas` fuses the same way a CUDA compiler would is not claimed.
- f32 only.

---

## Result, 2026-09-14 — all six stages closed

A `.lyth` file compiles and runs. `examples/saxpy.lyth`:

```
machine sm_120

kernel saxpy(n: u32, a: f32, x: [f32], y: [f32])
    intensity 0.1667

    stream x : dram -> reg
    stream y : dram -> reg, drain

    at reg:
        y = a * x + y
```

```
$ lyth run examples/saxpy.lyth --machine fixtures/machine/sm_120.json -n 1048576 --set a=2.5
kernel saxpy on machine sm_120
  derived  0.1667 flop/byte  (2 flop / 12 byte per element)
  traffic  8 read + 4 written, at dram
  declared 0.1667 — matches
  ridge    42.9 flop/byte — memory-bound
           at this intensity the ceiling is 0.39% of peak FLOPS, with bandwidth saturated
  device   NVIDIA GeForce RTX 5060 Ti
  launch   grid 4096 x block 256 over 1048576 elements
  verify   BIT-EXACT against the IR evaluated on the host, 1048576 elements
ok
```

**Bit-exact, not within a tolerance.** The reference is the same IR evaluated on the host, and
every operation emitted is the IEEE one the host performs — `f32::mul_add` against `fma.rn.f32`.
A tolerance would have hidden exactly the code-generation bugs the check exists to catch.

### The refusal is real

```
$ lyth check examples/saxpy-lie.lyth --machine fixtures/machine/sm_120.json
error[intensity]: examples/saxpy-lie.lyth:5:1: declares 2 flop/byte, body computes 0.1667
  bytes moved: 12 per element (8 read + 4 written)
  flops:       2 per element
  machine sm_120 ridge is 42.9 flop/byte, so 0.1667 is memory-bound
  at this intensity the kernel can reach 0.39% of peak FLOPS while saturating bandwidth
  did you mean to declare 0.1667?
```

Nobody typed `0.1667` on the right-hand side of that comparison. It came from counting the
streams the program declares and the flops the expression tree retires. **Arithmetic intensity
is a type, not a comment**, for the first time in this project.

```
$ lyth check examples/forgot-stream.lyth
examples/forgot-stream.lyth:7:13: `x` is a buffer but no stream moves it.
Add `stream x : dram -> reg`. You cannot operate on what you have not declared resident.
```

### Stage 6: the compiler's own byte model against silicon

`lyth build --evidence` writes the derived accounting as a `lyth-intensity/0.1` case, so
`lyth-probe intensity-check --ncu` checks it against measured traffic. **The accounting and the
PTX come from the same IR, so no human writes the byte model and no human can fudge it.**

16,777,216 elements, one launch, profiled with ncu 2025.3.1:

| | measured | derived | ratio |
|---|---|---|---|
| **l2, whole model** | 12.0013 B/element | 12.0000 | **1.0001x** CONFIRMED |
| **dram, read half** | 8.0003 B/element | 8.0000 | **1.0000x** CONFIRMED |
| dram, write half | 2.8669 B/element | 4.0000 | 0.7167x |

The write half is the write-back absorption ADR-0009 already documented: 28% of the stores were
still dirty in L2 when the kernel ended.

### Found by running it

- **`.version 8.5` with `.target sm_120` is rejected outright.** sm_120 needs ISA 8.7. Found by
  generating exactly that and reading the JIT log, which is why `load_ptx` captures
  `CU_JIT_ERROR_LOG_BUFFER`: `INVALID_PTX` alone is useless, and the log named the line.
  The ISA version now follows the target from a table rather than being a constant.
- **CUDA 13's `cuda.lib` on Windows carries no `__imp_` entries**, so linking it as an import
  library leaves every driver symbol unresolved. `raw-dylib` against `nvcuda` fixes it and is
  strictly better: **no CUDA toolkit is needed to build**, only a driver to run.
- **`cuGetErrorString` does not link at all** in CUDA 13. The result codes are named in a table
  instead, which also lets each one carry a sentence about what to do.

## [KNOWN LIMIT], as written before the code and still true

- One element per thread, bounds-checked, no grid-stride loop. Occupancy is not expressible and
  **no performance claim comes out of v1.** This slice is about correctness and honest cost.
- `fma` contraction is a structural pattern match on `a * b + c`, not a numerical guarantee
  about what a CUDA compiler would have done.
- f32 only. No reductions, no shared memory, no control flow in the body, one dimension.
- A divide is counted as one flop, in line with every published flop count and understating what
  the hardware does.

## What this does not settle

The ADR-0001 parser gate is **still open** and this does not close it. The question is whether a
Rust macro could carry the same declaration, and everything in `docs/DOGFOOD.md` still says it
could. What this settles is the other half: LYTH can produce something, and what it produces
moves the bytes it claims.
