# ADR-0020 — Against a hand-written kernel: 17% more instructions, the same bandwidth

**Status:** Accepted
**Date:** 2026-09-16
**Depends on:** ADR-0000 (why), ADR-0014 (loop cost), ADR-0009 (measured traffic)

`Nine Kernels, Measured` listed four things it did not contain, and put this first:

> **LYTH against a hand-written CUDA C++ kernel.** No such kernel exists in the repository, so
> there is no measurement to plot. It would be a fair comparison to run; it has not been run.

It has now. It is the only comparison in this project that is about the **compiler** rather
than about the memory system, because both sides move the same bytes by construction.

## Fairness was the hard part, not speed

A hand-written kernel that wins by using a different schedule has compared two schedules.
`bench/cuda/handwritten.cu` is written to the schedule LYTH emits — a grid-stride loop, `fmaf`,
the same bounds test — and carries **no `__restrict__`**, because LYTH's IR has no aliasing
annotation to emit and granting nvcc one would hand it an optimisation this is not about.

Both PTX modules are loaded through the same `ctypes` path and launched with the same
`cuLaunchKernel`, at the same grid and block, in the same process, interleaved. The only thing
that differs is the instructions.

**The toolchain had to be worked around, and that is worth recording.** `nvcc -ptx` needs a host
C++ compiler even when it emits no host code, and the MSVC on this machine (14.51) refuses
CUDA 13.0 from inside its own STL headers — `error STL1002: Unexpected compiler version,
expected CUDA 13.2 or newer`. `-allow-unsupported-compiler` does not help, because the failure
is a `static_assert` in the STL and not nvcc's own version gate. NVRTC compiles CUDA C++ to PTX
with no host compiler at all, which removes a dependency the comparison never needed.

## What each compiler emitted

The PTX, read rather than modelled — the inner loop of `saxpy`:

| | loop body | branches |
|---|---|---|
| LYTH | 11 | 2 |
| nvcc (NVRTC) | 10 | 1 |

nvcc **rotates the loop**: it hoists the first bounds test out, puts the test at the bottom, and
computes the stride only if the loop runs at all. LYTH does not — it emits a guard at the top
and an unconditional branch at the bottom, so two branches per iteration instead of one.

## ptxas does not erase it

`smsp__thread_inst_executed.sum`, one launch, n = 2²⁴, one element per thread:

| | instructions per element | branch instructions |
|---|---|---|
| LYTH `saxpy` | **28.00** | 0.03 |
| nvcc `saxpy_gridstride` | **24.00** | 0.03 |

**LYTH emits 17% more instructions and the assembler keeps them all.** The 28.00 is the same
figure ADR-0014 measured independently, which is the cross-check that says both measurements
are of the thing they claim.

The branch counts are identical — 0.03 per element is one branch per warp — because at one
element per thread the loop body runs once and the branch nvcc saved is the one neither kernel
takes.

## And it makes no measurable difference

Nine rounds of forty launches, interleaved, same grid and block:

| | GB/s median | min | max |
|---|---|---|---|
| LYTH | 380.8 | 376.8 | 382.4 |
| nvcc | 379.9 | 377.4 | 380.5 |
| torch `add_` | 377.1 | 373.3 | 378.2 |

**LYTH / nvcc = 100.2%, range 99–101%.** The range crosses parity, so at this resolution the
two are not distinguishable.

This is the plainest confirmation ADR-0000 has:

> Of course it did: instructions are not the scarce resource. Movement is.

LYTH's code generation is **measurably worse** — 17% more instructions, and the difference
survives the assembler — and it costs nothing here because the kernel waits on memory. The
honest form of the result is both halves at once: the compiler is behind, and the quantity it
is behind on is not the one that decides this kernel's time.

**It would decide a compute-bound kernel's time**, and this language cannot write one. Its
highest intensity is 8 flop/byte against a ridge of 36.9. So 17% is a debt recorded rather than
a debt paid, and the ADR that raises intensity is where it comes due.

## Two faults found in the instrument, both mine

**A short run said there was a difference.** Three rounds of fifteen gave
`LYTH / nvcc = 97.6%, range 98–99%` — a range that does *not* cross parity, which reads as a
real 2.4% deficit. Nine rounds of forty gave 100.2% and a range that does. The first result was
underpowered and I nearly published it. The rule the reference spread already taught, applied
again: a point estimate is not a result until its band is drawn.

**The first run reported 3924 GB/s on a 448 GB/s card.** `cuLaunchKernel` takes an array of
pointers into the caller's argument objects, and the launcher returned without keeping those
objects alive, so Python freed them and the kernel launched with a garbage `n`.

The correctness check passed. It built a launcher and called it in one expression, so the
arguments were still on the stack for *that* call and dead for every timed one — **the check
covered a path the timing did not use**. The launchers are now built once and both checked and
timed, which is the only version of that check worth having.

Third impossible number this project has produced from a launch that did not do the work:
3.85 TB/s from a mismatched block, 1646 GB/s from a byte count multiplied twice, 3924 GB/s from
freed arguments. Each was caught by being physically impossible rather than by a test.

## Files

| | |
|---|---|
| `bench/cuda/handwritten.cu` | the hand-written kernels, written to LYTH's schedule |
| `bench/nvrtc.py` | CUDA C++ to PTX with no host compiler |
| `bench/vs_handwritten.py` | the interleaved comparison, correctness first |
| `docs/ADR-0014-loop-cost.md` | the 28.00 this reproduces, measured another way |
