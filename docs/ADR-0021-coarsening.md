# ADR-0021 — Thread coarsening, and the limit that was argued but never checked

**Status:** Proposed — steps 1 and 2 built
**Date:** 2026-09-16
**Depends on:** ADR-0018 (contraction), ADR-0017 (the tile), ADR-0000 (why)

ADR-0018 closed with its own way out named:

> The way out is thread coarsening — one thread computing several outputs — and that is the next
> ADR, not this one.

This is that ADR. It starts somewhere unexpected, because the first thing the work found was
that the limit coarsening exists to escape **was never checked**.

## The limit was prose

ADR-0018's claim 3 reads:

> A tile of `T` is `T²` threads under one-thread-per-element, and this device reports
> `MAX_THREADS_PER_BLOCK = 1024`, so `T ≤ 32` and the intensity ceiling is `8 flop/byte`.

Every word of that is true and **the compiler did not know any of it.** `max_threads_per_block`
was in no machine file and no source file. So `tile 64, 64`:

* parsed;
* derived `0.125 * k + 4` bytes per element and an asymptote of 16 flop/byte, correctly;
* **emitted 3620 bytes of PTX**;
* and failed at launch with `CUDA_ERROR_INVALID_VALUE` — a driver error that names no argument
  and gives no reason.

A compiler whose whole claim is that it refuses programs that lie about their cost was happily
emitting a kernel that cannot run at all. The claim was in a document; the check was nowhere.

**Step 1 is therefore not coarsening. It is making the limit a fact the compiler holds**, which
is the precondition for coarsening being the answer to anything.

### What the device actually reports

Queried from the driver rather than recalled:

| attribute | value |
|---|---|
| `MAX_THREADS_PER_BLOCK` | 1,024 |
| `MAX_SHARED_MEMORY_PER_BLOCK` | 49,152 |
| `MAX_SHARED_MEMORY_PER_BLOCK_OPTIN` | 101,376 |
| `MAX_REGISTERS_PER_BLOCK` | 65,536 |
| `MAX_THREADS_PER_MULTIPROCESSOR` | 1,536 |

Three of these are now fields in the machine file and are checked before anything is emitted.
`None` is preserved as "this file predates the field, check nothing" — **a missing limit must
not silently become a limit of zero**, which is the failure mode a limit check has.

The refusal names which limit, by how much, what the tile would have bought, and what would be
needed:

```
error[launch]: k.lyth:5:1: `tile 64, 64` is 4096 threads per block and sm_120 allows 1024.
  A tile puts one thread on each of its elements, so the block is the tile's area.
  The traffic it would buy is derived and real -- 0.125 * k + 4 bytes per element --
  but no block is that wide.
  Thread coarsening, one thread computing several outputs, is the way to a
  larger tile. It is not implemented.
```

## The design

**`coarsen 2, 2`.** One new declaration, one meaning: how many outputs of the tile each thread
owns. The block is then `product(tile) / product(coarsen)`.

`tile` keeps meaning exactly what it meant — the patch of the output space a block owns, and the
thing the traffic derivation is written in terms of. `coarsen` says how the threads are spread
over it. Two lines, two facts, neither inferred.

The alternative considered and rejected was inferring the coarsening from the tile and the
thread cap: `tile 64, 64` with a 1024-thread cap implies 2×2. It is one fewer line and it is
the compiler choosing a schedule, which is the job this project says it does not do — see
ADR-0000's honest limit. A declared schedule that is checked is the whole shape of the thing.

### The cost model does not change

This is the property that makes the design right rather than merely workable. Traffic is derived
per stream from the **tile**, and coarsening does not touch the tile:

```
reuse = product of the tile dimensions of the free axes the buffer's index does not mention
```

`tile 64, 64` already derives `0.125 * k + 4` and an asymptote of 16, today, with no
coarsening implemented and no change to `derive_cost`. Verified: the figures in this ADR came
out of the compiler before any of it was written.

What changes is `Manifest::of`'s block (`product(tile) / product(coarsen)`), and the emitter.

### What the emitter has to grow

* **`C_i × C_j` accumulators per thread**, alive across the whole `p` loop.
* **A staging loop.** A block of 1024 threads stages a 64 × 64 tile, so each thread stages four
  elements instead of one.
* **`C_i + C_j` operand loads and `C_i × C_j` multiply-adds per step**, instead of two and one.
  This is where the arithmetic intensity actually comes from: the register file becomes a third
  level of reuse below shared memory.

## What coarsening buys, derived before it is built

Shared memory for a contraction tile `T` is two skewed tiles: `2 · T · (T+1) · 4` bytes.

| `T` | intensity `T/4` | threads at one per element | shared | what refuses it |
|---|---|---|---|---|
| 32 | 8 | 1,024 ✓ | 8,448 ✓ | nothing — this is today |
| 64 | 16 | 4,096 ✗ | 33,280 ✓ | **threads** → `coarsen 2, 2` |
| 96 | 24 | 9,216 ✗ | 74,496 (opt-in) | threads, then opt-in shared |
| 128 | 32 | 16,384 ✗ | 132,096 ✗ | **shared memory** |

**So coarsening moves the ceiling from 8 to 16, or to 24 with an opt-in shared allocation — and
the ridge at 36.9 stays out of reach.** Reaching it needs `T ≈ 148`, which is 176,416 bytes of
shared memory against a maximum of 101,376.

That is claim 3 of ADR-0018, rewritten by measurement:

> **The thread count binds first, and only up to a tile of 64. Past that it is the shared
> memory, and no tile this device can hold reaches the ridge.**

The honest consequence is that coarsening is worth doing for the factor of two it buys and for
the register-level reuse it introduces, and **not** because it makes this language
compute-bound. Nothing in the roofline changes category. A kernel at 16 flop/byte against a
ridge of 36.9 is still memory-bound, at 43% of peak FLOPS instead of 22%.

## Step 2, as built

`coarsen 2, 2` parses, resolves, and sets the block width. `KernelIr::block_threads` is the one
definition — the manifest publishes it and `lyth run` launches it, and when that rule lived in
two places they drifted once already (ADR-0019), so both now call it.

What the compiler says for each shape, on this machine:

| source | block | refused by |
|---|---|---|
| `tile 32, 32` | 1024 | nothing |
| `tile 64, 64` | 4096 | the thread cap |
| `tile 64, 64` + `coarsen 2, 2` | **1024** | no emitter yet |
| `tile 64, 64` + `coarsen 2, 4` | 512 | no emitter yet |
| `tile 128, 128` + `coarsen 4, 4` | 1024 | **shared memory, 132,096 bytes** |

**The last row is the prediction landing.** This ADR derived, before any of it was written, that
past a tile of 64 the binder stops being the thread count and becomes the shared memory. Step 1
implemented that refusal and could not reach it on this device — the block is the tile's area,
so the thread cap always bit first, and the branch had to be tested against a machine file
describing a smaller card. Step 2 made it reachable, and the number the compiler prints is the
number the ADR derived.

### Five refusals, and one of them is the identity

`coarsen` without a `tile`; a factor count that does not match the tile's rank; a factor that
does not divide its tile edge; a factor that is not a power of two (in the parser, like a tile
dimension, and for the same reason — the block's width is the tile's divided by it, and a
thread's position stays a shift and a mask only if every one of them is a power of two); and
`coarsen 1, 1`, which is what a tile does without the line.

Refusing the identity is a judgement and worth the sentence: a declaration that changes nothing
is a line a reader has to check and then discard, and this language has spent ADR-0018 refusing
exactly that shape — a `contract` nothing is indexed at, a `tile` that stages nothing.

### The cost model still does not appear

`crates/lyth-lang/tests/coarsen.rs` asserts it directly: `tile 64, 64` with and without
`coarsen 2, 2` derive the same `0.125 * k + 4`, the same asymptote of 16, and the same 33,280
bytes of shared memory. That absence is the design being right rather than an omission — reuse
is a property of the tile, and coarsening changes which thread computes what, not what is
staged.

### One test was testing the wrong rule

`a_factor_that_does_not_divide_its_tile_edge_is_refused` was written with `coarsen 2, 3`, and 3
is not a power of two — so the parser refused it first and the test never reached the rule it
named. It needs a factor that is a power of two *and* does not divide: 64 against a tile edge of
32. The original version asserted the parser twice and the divisibility check never.

## Build sequence

| step | | testable on its own |
|---|---|---|
| **1** | **the block limits as machine-file facts, checked before emission** | **done — 5 tests, no GPU needed** |
| **2** | **`coarsen` in the AST, parser and IR, with its refusals** | **done — 9 tests, no GPU needed** |
| 3 | the emitter: register accumulators, the staging loop, the term loop | bit-exact against the host at non-divisible `m`, `n`, `k` |
| 4 | `--ncu` | the derived `0.125 * k + 4` at `tile 64, 64`, against measurement |
| 5 | against a hand-written coarsened matmul | the ADR-0020 comparison, on a kernel where instructions might finally matter |

Step 5 is the interesting one. ADR-0020 measured LYTH emitting 17% more instructions than nvcc
and losing nothing for it, because every kernel this language can write waits on memory. A
coarsened matmul at 16 flop/byte is the first kernel where that debt could come due.

## Files

| | |
|---|---|
| `crates/lyth/src/main.rs` | the check, in `front`, before anything is emitted |
| `crates/lyth-probe/src/machine.rs` | the three limits, `Option` so an old file checks nothing |
| `fixtures/machine/sm_120.json` | the values, from the driver |
| `crates/lyth/tests/block_limits.rs` | including the shared-memory branch, reached through a machine file describing a smaller device |
