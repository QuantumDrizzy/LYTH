# ADR-0026 — What LYTH is, what it refuses, and what it will never depend on

**Status:** Accepted
**Date:** 2026-09-17
**Depends on:** ADR-0000 (why), ADR-0001 (thesis), ADR-0022 (the level that binds), ADR-0025 (standalone)

## Why this exists, said plainly

In three days LYTH has been described as a language, a compiler, a DSL, a cost model and "a
solape". **A thing described five ways is not defined**, and an undefined thing accretes
features that pull in different directions until it collapses under its own inconsistency. Its
author put it exactly right:

> "que falle algo no es problema, AHORA, la dirección y la arquitectura, SÍ, porque sin eso,
> aunque el lenguaje sea el mejor del mundo, se come a sí mismo con el tiempo y colapsa"

A failing test is found in a minute. A wrong direction is found in a year. This ADR fixes the
direction before more code is written, and it is short on purpose: a definition nobody can
recite is a definition nobody applies.

## The sentence

> **LYTH is a language where the bottleneck is not a surprise.**
> You declare what your kernel must cost. If the code does not meet it, it does not compile.

Everything below is that sentence with its edges drawn.

## What it is

A compiled domain-specific language for kernels that move data. Movement is declared first and
arithmetic is subordinate to it (ADR-0001), which is the inversion the whole project rests on:

```
kernel saxpy(n: u32, a: f32, x: [f32; n], y: [f32; n])
    intensity 0.1667          # the claim
    stream x : dram -> reg    # the movement
    stream y : dram -> reg, drain
    at reg:
        y = a * x + y         # the arithmetic, subordinate
```

Four things happen that no other kernel language does all of:

1. The compiler **derives** the arithmetic intensity from the code — 2 flop / 12 byte = 0.1667.
2. It **refuses to compile** if the declaration disagrees with the derivation.
3. It computes a ceiling **per memory level** against a machine file whose every number is
   measured on that machine, and names which level binds (ADR-0022).
4. It checks the emitted code **bit for bit** against a host oracle, on every backend.

That is a type system whose checked property is cost. `intensity 0.1667` is a type.

## What it refuses, and why every refusal is load-bearing

| refused | because |
|---|---|
| data-dependent branching | a branch on data makes the byte count unknowable at compile time |
| dynamic allocation | a size the compiler cannot see is traffic it cannot count |
| recursion | an unbounded depth is an unbounded cost |
| a tile on a machine with no shared level | a declaration that changes nothing (ADR-0025) |
| a permuted axis with no strided store | the right answer at a cost the contract does not describe |
| a narrow buffer where the machine cannot convert | twice the bytes the model derived |

**The refusals are the product.** A language that could express anything could not check
anything, and the check is the entire value. Every restriction buys the guarantee, and any
future feature that would make a byte count unknowable is refused by this ADR in advance.

## What it is not

Stated here so it is never implied elsewhere.

* **Not general-purpose, and never will be.** "Can I write any program in it" is the wrong
  test. The right test is "does it express its domain and refuse what it cannot cost".
* **Not a replacement for CUDA, Rust or C++.** On a GPU it is a guest, like every GPU
  language: a `.cu` file is not a program either, because the device has no entry point.
* **Not a promise that you hit the ceiling.** The contract is about what the code *costs*, not
  about how close to peak it lands. Measured against a hand-written matmul, the guarded runs
  were **94.5%** and **95.9%** — good, and not 100%, and said so (ADR-0021).

## The boundary — zero dependency, in both directions

LYTH has two back ends: PTX for `sm_120`, and Unibit assembly. **Neither is the product, and
the second is not LYTH's story.**

* `lyth-uasm` emits text. It links against nothing in the Unibit repository.
* The assembler is **invoked**, the way `ptxas` is, and when it is absent the `.uasm` is still
  written. A missing tool must never lose a compile.
* The tests that need the emulator **skip and say where they looked** when it is not there.

And the rule this ADR exists to protect, which applies to every public project under the same
roof:

> ### Each project must be interesting with ZERO of the others.
>
> LYTH useful with no Unibit. Unibit interesting with no LYTH. The moment one needs the other
> to be interesting, both are a toy.

**ADR-0025 violated this in its framing and this ADR corrects it.** ADR-0025 presented the
Unibit back end as the answer to "LYTH is not a language". It was never the answer — the answer
is that LYTH does not need to be general-purpose to be a language. The code ADR-0025 produced
is kept, because it is good and it found a real defect (a ceiling at 1600% of peak, invisible on
one machine). Its **claim** is demoted: the second back end is internal verification, not a
feature anyone is sold.

There is also a reason of method, and it is the stronger one. "Verified on two ISAs" means
something only if the two ISAs are independent artifacts. Merged, it is one thing checking
itself, which is worth nothing.

## What may be claimed today, and what may not

| | state |
|---|---|
| `sm_120`, traffic | **measured** against `ncu`, ±0.80% (ADR-0009) |
| `sm_120`, machine file | **measured**: 414.51 GB/s read, 15.30 TFLOP/s SGEMM, ridge 36.9 |
| `unibit`, machine file | **measured** from the emulator's counters: 28.43 B/cycle, 14.21 flop/cycle, ridge 0.4999 |
| `unibit`, a kernel's derived cost | **not yet** checked against what an emitted program retires |

So the publishable sentence today is:

> Verified against NVIDIA silicon to ±0.80%; a second, independent ISA is in progress.

and **not** "verified on two ISAs". The verb changes the day ADR-0025 step 5 measures it, and
not before. This is exactly the class of claim the project's own method exists to catch — and
the one kind that `ncu` cannot catch, because it is caught by reading a repository rather than a
counter.

## The domains this exists for

LYTH is not a general tool looking for users. It exists because every domain its author works
in is one where **an unpredicted bottleneck is fatal rather than annoying**:

| | why cost has to be known before the run |
|---|---|
| **IGNIS / KARDASHEV** — rocket engines, satellites, robotics | hard real time; a missed deadline is not a slow frame |
| **DRIFT** — computronium | it measures the **Landauer floor** of a computation. The same question one layer up: what is the least this can cost? |
| **Blaze** — tensor-network compression | the cost *is* the algorithm: `O(nχ³)` against a dense `O(2ⁿ)` that does not fit in memory |
| **SUBSTRATE, QuBLAR, gate-fusion** — quantum simulation | state vectors where one careless copy is the whole budget |
| **orbital-hpc** | compute where you cannot add a node |

DRIFT is worth naming twice. It asks what the minimum **energy** a computation can cost is;
LYTH asks what the minimum **traffic** a kernel can cost is. The same question against two
different floors, one thermodynamic and one architectural, arrived at independently.

## What a good base means, concretely

LYTH is not finished and this ADR does not pretend otherwise. What it fixes is which additions
are growth and which are drift:

**May be added.** More machines. More back ends. More operations at a level a stream has
reached. More levels a machine can name. Anything whose cost is derivable from the source.

**May never be added.** Anything that makes the byte count unknowable at compile time — see the
refusal table. Anything that makes a project require another of the same author's to be worth
using. Any claim that outruns its measurement.

The test for a future feature is one question: **after this, can the compiler still say what
the kernel costs, and still refuse the code that lies about it?** If not, it is not LYTH,
whatever else it might be worth building.
