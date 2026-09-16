# ADR-0024 — Half the element, and the two predictions that disagree

**Status:** Proposed
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

## Build sequence

| step | | testable on its own |
|---|---|---|
| 1 | `Ty::BufF16` / `BufBF16` in the AST, parser and IR; `elem` per stream | the derived traffic of a `f16` saxpy is 6 bytes, not 12 — no GPU needed |
| 2 | the emitter: `ld.global.b16` + `cvt.f32.f16`, arithmetic still `f32`, `st.global.b16` | bit-exact against a host oracle that converts the same way |
| 3 | measurement: the two predictions above, guarded | `saxpy` ~2x, `matmul` ~1x, on the same instrument as ADR-0022 |
| 4 | `bf16` beside `f16` | identical time, different error — the traffic model's blind spot, confirmed as blind |

Step 3 is the one this ADR exists for. Steps 1 and 2 are the price of admission.

## What this does not claim

It does not add tensor cores. `mma` / `wgmma` need a fragment layout this language has no way
to express, and the machine file records `wgmma` as **absent** on this device anyway. This is
about the width of a number in memory, not about a new instruction.

And it does not make the matmul faster. The ADR predicts that explicitly, and if it is right
the honest headline is **"half the bytes bought nothing on the kernel that matters"** — which
is a considerably more useful sentence than a speedup would have been.
