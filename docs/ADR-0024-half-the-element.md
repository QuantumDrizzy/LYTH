# ADR-0024 — Half the element, and the two predictions that disagree

**Status:** Proposed — step 1 built
**Date:** 2026-09-16
**Depends on:** ADR-0022 (the level that binds), ADR-0009 (measured traffic), ADR-0000 (why)

Every buffer in this language is `f32`. `Ty` has three variants, and `derive_cost` opens with

```rust
let elem = Ty::BufF32.bytes() as f64;
```

which is the entire cost model's element size, written once, as a constant.

This ADR halves it. Not because f16 is fashionable, but because **it is the sharpest test this
project can run on its own thesis**, and it is sharp precisely because ADR-0022 makes the model
predict two *different* answers to the same change.

## The prediction, in two halves that disagree

ADR-0000's claim is that movement decides. ADR-0022 refined it: movement decides **at whichever
level runs out first**, and for a tiled contraction that level is the shared pipe, which is
priced in **accesses** and not in bytes — measured, 1530 G accesses/s with a broadcast and a
coalesced read agreeing to 0.2% while their payloads differ by 32x.

Halving the element size does two things that are not the same thing:

| | f32 → f16 |
|---|---|
| bytes moved at DRAM | **halved** |
| shared-memory **accesses** | **unchanged** — one access per operand per term either way |

So:

> **`saxpy` should run about 2x faster. The matmul should not speed up at all.**

Both come out of one model, from different levels of it, and nothing else this project has
measured separates them. A result where both kernels double is a result where ADR-0022's
`smem binds` line is decoration. A result where neither moves is a result where the traffic
model was never about time.

Pre-registered before a line is written, using the figures already in the repository:

| kernel | binds at | f32 today | f16 predicted | why |
|---|---|---|---|---|
| `saxpy` | dram, 12 B/element | 0.07 TFLOP/s ceiling | **~2x** | 6 B/element |
| `sum`, `max` | dram, 8 B/element | — | **~2x** | 4 B/element |
| `transpose-tiled` | dram, 8 B/element | — | **~2x** | 4 B/element, and the skew must still work at 2 bytes |
| `matmul` `tile 32` | **smem**, 2.0625 access/element/k | 1.22 TFLOP/s measured | **~1x** | the access count does not depend on the element's width |
| `matmul` `coarsen 2,2` | **smem**, 1.0312 | 2.52 measured | **~1x** | same |

The matmul rows are the interesting ones and they are the ones that can embarrass this ADR.

## What has to change, and what deliberately does not

### The element size stops being a constant

`Ty` gains `BufF16` and `BufBF16`, `Ty::bytes()` answers 2 for them, and `derive_cost`'s `elem`
becomes a property of the stream's buffer rather than a module-level assumption. That one line
is the whole reason this is tractable: the cost model was written in terms of *an* element size
from the beginning, it simply only ever had one.

### Storage width and arithmetic width are not the same decision

This is the trap, and it is ADR-0010's trap one type over.

ADR-0010 refused to fuse a multiply and an add because one rounding is a different answer from
two, and that refusal is still measured every time `bench/vs_handwritten_matmul.py` finds LYTH
and nvcc **bit-identical** under `--fmad=false`. The same question arrives here as: *does a
`f16` buffer mean the arithmetic is `f16`?*

**No.** A half-precision buffer loaded into a register and accumulated in `f32` is the
overwhelmingly common hardware practice and it is a *different function* from accumulating in
`f16` — over `k = 2048` terms, catastrophically so. So:

* `[f16; n]` is a **storage** declaration. It says what crosses the bus.
* The accumulator's width is the **contraction's**, and it stays `f32` unless the source says
  otherwise.
* The host oracle does the same thing, or every bit-exactness check in the project becomes a
  tolerance check, which is the thing ADR-0003 was closed as a failure for allowing.

That decision is what makes the traffic prediction testable at all: if the arithmetic changed
too, a slower or faster result could be either the bytes or the rounding, and nothing would be
separated.

### `bf16` as well as `f16`, because they differ where it matters

Both are two bytes and move identically, so the **traffic model cannot tell them apart** — and
that is a claim worth checking rather than assuming, since it predicts the two will run at
identical speed and differ only in accuracy. `bf16` has f32's exponent range and 8 fewer
mantissa bits; `f16` has 10 mantissa bits and a range that overflows at 65504. A sum over 2048
terms will find that edge, and the oracle is what will say so.

## Two semantics step 1 decided, written down as decisions

Both of these came out of tests of mine that were wrong. A rule that surprises the person who
wrote the code is exactly what an ADR is for, and neither should be rediscovered in an argument
about whether some future refusal was fair.

### `drain` asserts the outgoing movement, not an incoming dependency

**A drained buffer is read only if the body reads it.** `y = a * x + y` reads `y`; `y = x` does
not, and the second moves 6 bytes per element at half width where the first moves 10. The
`drain` keyword says a value leaves; it says nothing about whether one arrived.

A test here asserted 10 for `y = x` on the assumption that `drain` implied both, and the cost
model was right. Written down now because tomorrow this decides whether a refusal is just.

### A sector does not know what is in it

ADR-0015 separated the payload from the bus. Narrow elements sharpen that separation into two
different behaviours in the same kernel:

| access | at `f32` | at `f16` |
|---|---|---|
| coalesced | 4 B — **the payload is the bus** | 2 B, halves with the element |
| strided | 32 B — a whole sector for one element | **32 B, unchanged** |

> **A strided `f16` kernel wastes twice the fraction of the bus that a strided `f32` one does.**

That is a reusable prediction rather than an observation about this type: **every narrow type
this language ever gains inherits it**, and any future `f8` would waste four times the fraction.
It also means coalescence is no longer a property of the access pattern alone — it is the
pattern *and* the width, and `coalescence()` already reports it that way.

A test summed the read and write sectors, got 36 against 34, and read that as the bus narrowing.
It did not. The halves have to be checked apart.

### And a note on method

The exhaustiveness check earned its own line. Converting `== Ty::BufF32` into `.is_buffer()`
first was necessary because equality checks are invisible to the compiler and would have gone
silently false. But the sharper lesson is the guard pattern that was written and then deleted:
`t if t.is_buffer()` compiles, reads better, and **switches exhaustiveness off**. The step where
a shortcut is most tempting is the step where the compiler enumerating what has not been thought
of is worth the most.

## Step 2's three decisions, made before any of it is written

Left to the implementation these get decided by accident, and then a measurement cannot tell
whether it validated the physics or the emitter.

### 1. Storage is narrow; arithmetic is not

```
ld.global.b16  →  cvt.f32.f16  →  every operation in f32  →  cvt.rn.f16.f32  →  st.global.b16
```

Two bytes cross the bus, four bytes sit in the register, and `.rn` is round-to-nearest-even on
the way back out. This is what the hardware's own mixed-precision path does and it is the only
choice under which the traffic prediction is testable at all: if the arithmetic narrowed too, a
changed time could be the bytes **or** the rounding and nothing would be separated.

### 2. A reduction's accumulator is f32, and that is not a detail

A sum of `f16` accumulated in `f16` rounds once per term; accumulated in `f32` it rounds once,
at the end. **Those are different functions with the same signature**, and over `k = 2048` the
difference is not subtle. The accumulator is f32. A future `reduce sum f16` that wants the other
behaviour will have to say so, and will be a different kernel rather than the same one compiled
differently.

### 3. The host oracle is the actual work of step 2

Rust's `f16` is **unstable** on this toolchain — checked, not assumed: `error[E0658]: the type
f16 is unstable`, rustc 1.91.1 — and `lyth-lang` has exactly one dependency. So the conversion
is written here, as IEEE 754-2008 binary16 with round-half-to-even, including subnormals and the
overflow at 65504.

**And it is verified against the device before anything else runs.** The rule is the one this
project has always had: the oracle models what the device *will* do, not what it *should*. An
oracle that rounds a hair differently from `cvt.rn.f16.f32` produces a mismatch that is nobody's
bug and eats an afternoon, so step 2 starts with a rounding test over the hard cases — ties,
subnormals, overflow, and the values either side of each — and only then compiles a kernel.

## Build sequence

| step | | testable on its own |
|---|---|---|
| **1** | **`Ty::BufF16` / `BufBF16` in the AST, parser and IR; `elem` per stream** | **done — 7 tests, no GPU; a `f16` saxpy derives 6 bytes and the compiler refuses the stale declaration** |
| 2 | the emitter: `ld.global.b16` + `cvt.f32.f16`, arithmetic still `f32`, `st.global.b16` | bit-exact against a host oracle that converts the same way |
| 3 | measurement: the two predictions above, guarded | `saxpy` ~2x, `matmul` ~1x, on the same instrument as ADR-0022 |
| 4 | `bf16` beside `f16` | identical time, different error — the traffic model's blind spot, confirmed as blind |

Step 3 is the one this ADR exists for. Steps 1 and 2 are the price of admission.

### Step 3's outcomes, pre-registered

| result | conclusion |
|---|---|
| `saxpy` ~2x in **time**, matmul unchanged | both levels confirmed, and ADR-0022 is tied down as it has not been |
| `saxpy` ~2x, **matmul speeds up too** | ADR-0022's level is wrong: the shared pipe is priced in bytes after all, and `smem binds` is decoration |
| `saxpy` does **not** halve its time | the DRAM traffic model is missing a term for narrow types |

Three precisions on how that is read, because each one is a way to get a false result:

* **The claim is 2x in time, not in GB/s.** The achievable bandwidth does not change; the bytes
  do. A kernel that halves its bytes and keeps its bandwidth halves its time, and quoting the
  unchanged GB/s as "no improvement" would be reading the wrong number.
* **The traffic claim lives at `lts__t_bytes`, not at DRAM.** ADR-0015's level discipline and
  ADR-0018's measurement both say so: write-back does not respect a launch boundary, and DRAM
  figures in this project have been 1.5x off the payload for that reason.
* **"Unchanged" means inside the band, not exactly zero** — and the band is **±5%**, not
  ADR-0022's ±0.80%. Those are different tolerances for different quantities: ±0.80% is how
  closely the *traffic* model matched `ncu`, while this row is a *timing* claim, and the
  measured run-to-run reproducibility of the ADR-0022 timing harness is **1.6%** across two
  guarded nine-round runs. ±5% covers that with room for thermal drift. Using a traffic
  tolerance on a timing prediction would manufacture a falsification out of ordinary noise.

## What this does not claim

It does not add tensor cores. `mma` / `wgmma` need a fragment layout this language has no way
to express, and the machine file records `wgmma` as **absent** on this device anyway. This is
about the width of a number in memory, not about a new instruction.

And it does not make the matmul faster. The ADR predicts that explicitly, and if it is right
the honest headline is **"half the bytes bought nothing on the kernel that matters"** — which
is a considerably more useful sentence than a speedup would have been.
