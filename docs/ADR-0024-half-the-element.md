# ADR-0024 — Half the element, and the two predictions that disagree

**Status:** Accepted — all four steps built and measured
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
| **2** | **the emitter, and the oracle that has to agree with it** | **done — 4 tests; bit-exact at f32/f16/bf16 on five elementwise and reduction kernels, a staged tile, and one kernel reading narrow and writing wide** |
| **3** | **measurement: the two predictions above, guarded** | **done — `saxpy` 1.96x, `matmul` 1.010x, and two rows that missed the band and sharpened the model** |
| **4** | **`bf16` beside `f16`** | **done — 0.999-1.009x across six kernels, against a predicted 1.0** |

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

## Step 3, as measured

`bench/narrow_vs_wide.py`. Every kernel verified bit-exact at the size it is timed at, sizes
chosen to exceed the 34 MB L2, guarded by ADR-0023's watch — **no display-driver reset during
the run**.

| kernel | binds at | f32 | f16 | f32/f16 | predicted | |
|---|---|---|---|---|---|---|
| `saxpy` | dram | 2.0178 ms | 1.0296 | **1.960** | 2.0 | ✅ |
| `axpby` | dram | 2.0168 | 1.0307 | **1.957** | 2.0 | ✅ |
| `sum` | dram | 0.6501 | 0.4829 | 1.346 | 2.0 | ❌ |
| `transpose-tiled` | dram | 0.4601 | 0.3736 | 1.232 | 2.0 | ❌ |
| **`matmul`** | **smem** | 1.7065 | 1.6901 | **1.010** | 1.0 | ✅ |
| **`matmul` + `coarsen 2,2`** | **smem** | 0.8377 | 0.8317 | **1.007** | 1.0 | ✅ |

### The row that could have killed ADR-0022 did not

**1.010 and 1.007, against a pre-registered band of ±5%.** Halving every byte the matmul moves
bought **nothing**, on the kernel this language works hardest on. That is the sentence ADR-0022
needed and had never been given: the shared pipe is priced in *accesses*, and the access count
does not know how wide an element is.

It survived a second way, too. Narrowing **adds** a `cvt.f32.f16` per shared load — registered
before the run as the interesting way for this to go wrong — and it cost nothing either. A
kernel with room for two extra instructions per term is a kernel that is not waiting on
instructions, which is the same thing ADR-0022 concluded from LYTH executing 35% more of them
than nvcc and finishing first.

### Two rows missed the band, and they are the result

`sum` and `transpose-tiled` are declared DRAM-bound by the compiler and did not halve. The
reason is measurable and it is not noise:

| | f32, % of the 414.51 GB/s this device was measured at | f16 | |
|---|---|---|---|
| `saxpy` | **96%** | 94% | halved: 1.96x |
| `axpby` | **96%** | 94% | halved: 1.96x |
| `sum` | **99.6%** | **67%** | 1.35x |
| `transpose-tiled` | **70%** | 43% | 1.23x |

> **`binds at dram` says which ceiling is lowest. It does not say the kernel reaches it.**

`saxpy` and `axpby` were at 96% of the achievable bandwidth, so halving their bytes halved their
time and they stayed at 94%. `sum` was at **99.6%** — as saturated as this device gets — and at
half the bytes it fell to 67%: the bytes halved and the time did not, because something else
became the constraint on the way down. `transpose-tiled` began at 70% and had headroom the
narrowing could not use.

So the honest statement of the result is not four hits and two misses. It is a **continuum the
model has no term for**:

> Halving the element halves the **bytes** and leaves the **work** alone. A kernel's time halves
> exactly to the extent that bytes were what it was waiting for — 1.96x at 96% of bandwidth,
> 1.35x for a kernel that was saturated at four bytes and was not at two, 1.23x for one that
> never was, and **1.01x for one that was never waiting on bytes at all**.

That last entry is the matmul, and it puts the two "failures" and the two successes on one axis
with it. ADR-0022 made the ceiling a per-level minimum; this says the ceiling is an upper bound
whose **achieved fraction is not a constant**, and nothing in the compiler models that fraction.
ADR-0022 already had the same gap in smaller print — its five kernels reached 79–86% of their
shared ceilings — and it was easy to read that as a fixed efficiency. It is not: 96%, 99.6%,
70%, and 43% are all in this one table.

### What the model would need

Not attempted here, and named so the next ADR does not have to rediscover it: the compiler
derives traffic and a ceiling, and **has no notion of how close a kernel gets**. That gap is
invisible while every kernel is f32, because the fraction is roughly stable across a set of
kernels that move the same kind of bytes. Halving the element is what pulled the fraction apart
— the same kernel, at 99.6% and then at 67%.

## Step 4, as measured — the blind spot is blind

`bf16` beside `f16`, same run, same instrument, guarded.

| kernel | f32 | f16 | bf16 | f32/f16 | **bf16/f16** |
|---|---|---|---|---|---|
| `saxpy` | 2.0236 ms | 1.0302 | 1.0392 | 1.964 | **1.009** |
| `sum` | 0.6534 | 0.4863 | 0.4871 | 1.344 | **1.002** |
| `axpby` | 2.0217 | 1.0352 | 1.0343 | 1.953 | **0.999** |
| `transpose-tiled` | 0.4456 | 0.3747 | 0.3750 | 1.189 | **1.001** |
| `matmul` | 1.7064 | 1.6903 | 1.6897 | 1.010 | **1.000** |
| `matmul` + `coarsen 2,2` | 0.8400 | 0.8389 | 0.8398 | 1.001 | **1.001** |

**0.999 to 1.009 across all six**, against a pre-registered 1.0 ± 5%.

The traffic model cannot tell `bf16` from `f16` — two bytes is two bytes, and `Ty::bytes()`
answers 2 for both — so it predicts they take the same time. They do, to within 1%. That is a
confirmation worth having precisely because it is boring: a model that says two things are
identical and is right about it has had a real opportunity to be wrong, and

> **boring equalities are where people stop pre-registering, and then cannot say afterwards
> whether the tie was theory or accident.**

What the model is blind to is everything that makes the two formats different. `bf16` keeps
f32's exponent range with 8 fewer mantissa bits; `f16` keeps 10 mantissa bits and overflows at
**65504** — the number `tools/half_probe.py` pins with 65520, which is the midpoint to a value
that does not exist and therefore becomes infinity rather than saturating. A sum over enough
terms finds that edge in one and not the other. **The compiler has nothing to say about it**,
and the right response is to write that down rather than to add an accuracy model it cannot
check against the silicon the way it checks traffic.

So the honest one-line summary of the pair: **same bytes, same time, different failure mode.**

### And a false alarm, recorded because it was mine

The run was invoked as `python bench/narrow_vs_wide.py | tail`, reported exit code 0, and two
rows said `MISSED`. That looked exactly like a check whose failure state is unreachable — the
defect this project hunts hardest. It was not: a shell pipeline reports the **last** command's
status, so the 0 was `tail`'s. The script's predicate was correct and would have returned 2.

Worth a line because the reflex was right and the conclusion was wrong: suspecting the check is
the correct instinct, and verifying before announcing is the rest of it.

## What this does not claim

It does not add tensor cores. `mma` / `wgmma` need a fragment layout this language has no way
to express, and the machine file records `wgmma` as **absent** on this device anyway. This is
about the width of a number in memory, not about a new instruction.

And it does not make the matmul faster. The ADR predicts that explicitly, and if it is right
the honest headline is **"half the bytes bought nothing on the kernel that matters"** — which
is a considerably more useful sentence than a speedup would have been.
