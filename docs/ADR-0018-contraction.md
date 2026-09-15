# ADR-0018 — Contraction, where the cost model stops being a number

**Status:** Proposed
**Date:** 2026-09-16
**Depends on:** ADR-0000 (why), ADR-0017 (the tile), ADR-0015 (shape)

## The transpose was the mechanism; this is the theorem

ADR-0017 showed that a declared schedule changes a derived cost: the same body, one `tile` line,
36 bytes per element becoming 8, measured exact. That is the machinery working.

It is not yet the claim ADR-0000 makes. A transpose has no arithmetic, so its intensity is zero
under every schedule, and the 8 is recoverable by hand once you know the emitter absorbs the
permutation. Nothing about it needed a theorem.

A contraction does. `C[i,j] = sum_p A[i,p] * B[p,j]` over a `T`-by-`T` tile:

* each block loads `2T²` elements per step and takes `K/T` steps, so `2TK` elements per block;
* there are `MN/T²` blocks, so `2KMN/T` elements move for `MN` outputs;
* that is `2K/T` elements read per output, `+1` written, and `2K` flops.

```
intensity = 2K / (4 * (2K/T + 1))  ->  T/4 as K grows
```

**The intensity is a function of the tile.** Not of the body, not of the buffers, not of the
problem: of a number the author declared. Doubling `T` doubles it. That is the quantity
ADR-0000 says determines performance, derived from a schedule, and it is the first time in this
language that nobody could work it out by inspection.

It is also the Hong-Kung bound becoming operational rather than cited. The bound says
`Ω(n³/√M)` words must move for a fast memory of size `M`; the tiled schedule moves `2n³/T` and
`T` is bounded by what fits in `M`, so choosing the largest legal tile is what walks toward the
proof. The compiler derives the traffic and can say how far from the horizon a declared tile
sits.

## The decision that makes this different from every ADR before it

Every derived cost so far has been a **number**. `bytes_per_element` is 12 for saxpy, 8 for a
transpose, and the check compares a constant against a constant.

`2K/T + 1` is not a number. `K` is a launch extent, so a contraction's traffic per element is an
**expression in the extents**, and the cost model has to carry one.

This is not avoidable by picking a different formula — it is what reuse means. Traffic per
output falls as the contraction gets longer, because the tile is loaded once and used `T` times,
and any model that reports a constant is reporting a different kernel.

So `Cost` grows a symbolic form, and the existing split is the precedent: ADR-0015's sector
figure is already a static upper bound refined at launch when the extents are known. Same shape,
one level deeper.

* **Statically**, the compiler derives and reports the expression, and the asymptote `T/4`.
* **At launch**, with `K` in hand, it evaluates it exactly and checks the declaration against
  that.
* `intensity` in the source is checked against the asymptote within a tolerance, because a
  source constant cannot be a function of a runtime extent. The exact figure is a launch-time
  report, not a compile-time refusal.

That asymmetry is honest and it is a limit: **a contraction's declared intensity is a weaker
claim than an elementwise kernel's.** Written here rather than discovered.

## Syntax

```
kernel matmul(m: u32, n: u32, k: u32,
              a: [f32; m, k], b: [f32; k, n], c: [f32; m, n])
    space i, j : m, n
    contract sum p : k
    tile 32, 32
    intensity 8.0

    stream a : dram -> smem -> reg
    stream b : dram -> smem -> reg
    stream c : dram -> reg, drain

    at reg:
        c[i, j] = a[i, p] * b[p, j]
```

`contract sum p : k` declares an axis that is **walked and summed**, not one of the free indices
the space iterates. It reads like `reduce sum`, and deliberately: both say "this is combined",
and the word `sum` is there for the same reason it is there in a reduction — the operator is
written, not implied.

The two are different machines and the ADR should say so before someone conflates them.
`reduce` combines **across threads** through shared memory, and its tree order is part of the
contract because float addition is not associative. `contract` combines **within one thread**,
sequentially over `p`, in a register. The order is the loop order and there is nothing to
choose.

`c[i, j] = ...` assigns once per output, not once per `p`. The accumulation is what `contract`
declares; the body states the term being accumulated. An `+=` would put the schedule in the
body, which is the thing this language exists not to do.

## What the emitter has to grow

Four things, and the first three are new shapes rather than new ideas.

**Two staged buffers.** ADR-0017 refused more than one on purpose. `Plan::of` lifts to two and
the shared layout holds two skewed tiles — `2 * T * (T+1) * 4` bytes, 8448 at `T = 32`.

**A loop over the contracted axis, inside the tile loop.** Load both tiles, barrier, accumulate
`T` products into a register, barrier, advance `p`. Two barriers per step, for the reason
ADR-0017 measured: the next step's load races this step's read.

**A register accumulator that survives the loop**, initialised to the operator's identity, which
`ReduceOp::identity` already provides.

**Boundary guards on three axes**, not two. `m`, `n` and `k` may each be ragged, and a guard
that is wrong in the generous direction is correct on every multiple of 32. **The first shapes
run are non-divisible**, as in ADR-0017: `k` especially, since a partial final step reads tile
elements that were never loaded unless the staging zero-fills — which it does, and that is why
the identity matters.

## Pre-registered, per the rule ADR-0000 now carries

Written as what the derivation implies, not as what would be tidy.

**Claim 1 — intensity tracks the tile and nothing else.** At `m = n = k = 1024`, the derived
intensity `2K / (4(2K/T + 1))` is **3.97 at `tile 16, 16`** and **7.88 at `tile 32, 32`**: the
same body, the same buffers, the same extents, one number changed in a declaration. Each lands
within 2% of the asymptote `T/4`, below it because the write of `C` is in the denominator and
does not amortise.

**Claim 2 — the traffic is what the model says.** `lts__t_bytes.sum / (m*n)` at
`m = n = k = 1024`, `T = 32`, against a derived `4 * (2K/T + 1)` = **260 bytes per output**. The
tolerance is the one ADR-0017 earned rather than a round number: the untiled transpose overshot
its sector model by read-for-ownership, and a matmul writes `C` once and coalesced, so the same
mechanism should not appear. **If measured exceeds derived by more than 2%, the excess is the
result and gets its own investigation**, not a widened tolerance.

**Claim 3 — the language can say why it cannot reach the ridge, and the reason is the block.**
A tile of `T` is `T²` threads under one-thread-per-element, and this device reports
`MAX_THREADS_PER_BLOCK = 1024`, so `T ≤ 32` and the intensity ceiling is `8 flop/byte` against a
ridge of `42.9`. Reaching the ridge needs `T ≈ 172`, which is 29,584 threads per block — 28x the
cap. Shared memory would also refuse it (`238 KB` against `101,376 B` opt-in), but **the thread
cap binds first**, which is worth stating because the obvious guess is the memory.

So: a tiled matmul in this language is memory-bound on this machine and cannot be otherwise, and
the compiler should derive that from the machine file rather than the reader inferring it. The
way out is thread coarsening — one thread computing several outputs — and that is the next ADR,
not this one.

## What is deliberately not here

No thread coarsening, no double buffering, no vectorised loads, no tensor cores, no rank above
2, no contraction over more than one axis. This will not be fast. It is the first kernel in this
language whose cost is a theorem rather than a count, and that is the whole of what it claims.

## Build sequence

| step | | testable on its own |
|---|---|---|
| 1 | `contract` in the AST, parser and IR, with its refusals | a matmul source parses; every existing example unchanged |
| 2 | the symbolic cost, and the launch-time evaluation | derived intensity moves with `T` and nothing else |
| 3 | two staged tiles and the `p` loop in the emitter | bit-exact against the host at non-divisible `m`, `n`, `k` |
| 4 | `racecheck`, and the structural assertions | 0 hazards, four barriers, two skewed strides |
| 5 | `--ncu` | claims 1 and 2, separately |
