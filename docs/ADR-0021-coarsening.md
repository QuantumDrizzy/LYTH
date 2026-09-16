# ADR-0021 — Thread coarsening, and the limit that was argued but never checked

**Status:** Proposed — steps 1, 2 and 3 built
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

## Step 3, as built

**One emitter, not two.** `contracted_body` was generalised rather than forked, with the absent
declaration meaning `(1, 1)`:

```rust
let (ci, cj) = match &ir.coarsen { Some(c) => (c[0], c[1]), None => (1, 1) };
let bi = t / ci;
let bj = t / cj;
let log2_bj = bj.trailing_zeros();
```

That choice is the reason the uncoarsened path is still covered: every contraction test written
for ADR-0018 exercises this code with both factors at one. A second emitter would have left them
testing a path that no longer runs.

The one substantive edit to an existing line is the third of those: **a thread's position in the
tile now masks and shifts by the block's width `bj`, not the tile's `t`.** Those were the same
number until this ADR, which is exactly the kind of coincidence that hides in an emitter.

What the body became, per thread:

| | uncoarsened | `coarsen 2, 2` |
|---|---|---|
| elements staged per operand | 1 | `ci·cj` = 4 |
| accumulators alive across the whole `p` loop | 1 | 4 |
| shared loads per term | 2 | `ci + cj` = 4 |
| bodies run per term | 1 | `ci·cj` = 4 |
| guarded global stores | 1 | 4 |

**The fourth row over the third is the register-level reuse**, and it is what makes the larger
tile affordable: two shared loads feeding one multiply-add becomes four feeding four. The
DRAM-side traffic this ADR derives is a property of the tile and does not see any of it; what
coarsening buys at the shared-memory interface is a separate halving, and neither is claimed by
the other.

### Verified

* **The uncoarsened matmul is unchanged** — seven shapes, still bit-exact against the host.
* **The coarsened matmul is bit-exact** on those same seven, chosen so that 64 divides none of
  `m`, `n` or `k` (`97×131×67`, `65×33×129`, `256×256×31`, `31×31×256`, `7×5×3`, `1×1×1`, and
  `64³` as the one that does divide). A thread owning four outputs on a ragged edge holds some
  that are inside the matrix and some that are not, which is the case a divisible shape cannot
  reach.
* **`coarsen 2, 4` is bit-exact on the same seven**, at 512 threads per block. `2, 2` cannot
  catch an emitter that confused `ci` with `cj`; the asymmetric one can, and it is in the suite
  rather than in a shell history.
* **Structure asserted on the PTX**: 4 `ld.shared.f32`, 4 `mul.rn.f32`, 4 `add.rn.f32`, 4
  `st.global.f32`, 8 `st.shared.f32`, exactly 2 `bar.sync` and still no `fma.rn.f32` (ADR-0010).
  The cost model does not see the coarsening, so the ratio it is supposed to buy is asserted on
  the emitted code instead of taken on trust.
* **Three controls**: `racecheck` reports 0 hazards, `memcheck` 0 errors, and
  `--shared-bytes 0` fails with `ILLEGAL_ADDRESS` — ADR-0017's bypass check, which for a
  coarsened tile *is* the claim, since the staged tile is the whole reason the larger tile has
  any reuse to sell.
* **Refused**: `coarsen` on a tile that does not contract. A tiled transpose moves one element
  per thread and holds nothing across a loop, so there is nothing for a thread to own several
  of — and the cost model would derive identical traffic either way, which is precisely why the
  emitter has to be the one to say so.

Seven tests in `crates/lyth/tests/coarsen_emitter.rs`; full suite **240 tests green**, clippy
clean.

### And the traffic it was built for

`ncu --metrics lts__t_bytes.sum`, at `m = n = k = 2048`, same source both rows with two
declarations changed:

| source | block | L2 bytes | per output | derived | |
|---|---|---|---|---|---|
| `tile 32, 32` | 1024 | 2,176,532,096 | 518.93 | 516 | **+0.57%** |
| `tile 64, 64` + `coarsen 2, 2` | 1024 | 1,091,252,448 | 260.17 | 260 | **+0.07%** |

**1.99× less traffic for the same answer, same block width, from one extra line.** The derived
ratio is `516 / 260 = 1.985`; the measured one is 1.995.

The second row is the closest this cost model has come to a measurement at any point in
ADR-0018 or here. That is not a claim about coarsening — the derivation it is being checked
against is the tile's, and ADR-0021's whole design argument is that coarsening does not enter
it. It is the tile of 64 that the compiler could not previously launch, now launched and moving
what it was derived to move.

Both figures sit **above** the derivation, so this is not the L1 effect ADR-0018 isolated, which
is signed the other way (`lts__t_bytes` is what the L1 asks the L2 for, so an L1 hit shows up as
the measurement falling *below* what the kernel asked for). The `tile 32` excess continues a
sequence already in ADR-0018's table — +0.16% at 512, +0.35% at 1024, +0.57% here — and **has
no explanation yet**. It cannot get one from this run, which collected a single metric and not
the L1 hit rate the ADR-0018 adjustment needs. That is step 4's job, and this is a spot check,
not step 4.

It joins the honest open list next to the bank-conflict excess from ADR-0018: small, consistent,
inside the 2% tolerance, and not understood.

## Step 5, pre-registered before it was measured

The first run of `bench/vs_handwritten_matmul.py` produced a number that looks like the ADR
landing and is **not evidence for it**, and the reason is worth writing down before the
instrument is pointed at it.

Coarsening halves three different quantities at once, all by exactly two:

| per flop | `tile 32` | `tile 64` + `coarsen 2, 2` |
|---|---|---|
| global bytes (derived, and measured in step 4) | 2.16 GB / launch | 1.09 GB |
| **shared-memory reads** | 4.0 B/flop — 68.7 GB | 2.0 B/flop — 34.4 GB |
| **instructions** — `ci + cj` loads feeding `ci · cj` bodies | ~13 per 2 flop | ~27 per 8 flop |

A 2× speedup is therefore consistent with all three and discriminates between none of them. If
this ADR reports "traffic halved, time halved, the cost model works", it has fitted the one
mechanism it happens to model to a result three mechanisms predict.

Three things are pre-registered, before the counters are read:

1. **DRAM is not the constraint, and the printed ceiling is a category error at this size.**
   `A + B` at 2048 is **33.6 MB** against an L2 of **34 MB**, so the operands fit in cache and
   ADR-0018 already measured `DRAM/output = 8.02`, which is each matrix read exactly once. The
   derived figure is an **L1→L2** claim — step 4 checks it as one — but `lyth run` prints
   `payload ... at dram` and computes its "43.35% of peak FLOPS" ceiling by dividing that
   figure into a **DRAM** bandwidth. Predicted: DRAM moves ≈33.6 MB per launch for both tiles,
   which at 414.51 GB/s is 0.08 ms against a launch that takes several.

2. **Neither kernel reaches its printed roofline ceiling.** 8 flop/byte × 414.51 GB/s = 3.32
   TFLOP/s and 16 × 414.51 = 6.63. Predicted: both land near 40% of those, for the reason in
   (1) — the number they are a fraction of is not about this kernel.

3. **The binding constraint is instruction issue**, which is the quantity ADR-0020 measured
   LYTH to be **17% worse at** and explicitly deferred to this ADR. Counting warp-instructions
   from the emitted inner loop against 36 SMs × 4 schedulers × ~2.4 GHz:

   | | thread-inst | warp-inst | predicted | first measured |
   |---|---|---|---|---|
   | `tile 32` | ~112 G | 3.5 G | ~10.1 ms | 13.6 ms |
   | `tile 64` + `coarsen 2, 2` | ~58 G | 1.8 G | ~5.2 ms | 6.4 ms |

   74% and 82% of issue peak — close enough that issue is a candidate and neither of the other
   two is.

**And (3) predicts something the first run already contradicts.** If this kernel is issue-bound
and LYTH emits 17% more instructions, LYTH should lose by something near 17%. The first,
contaminated run had it at **98.2%** and **103.4%**. So at most one of these is true:

* the kernel is not issue-bound, or
* LYTH is not 17% behind **on this kernel** — the saxpy figure does not transfer.

`smsp__thread_inst_executed.sum` on both sides decides which, and it is one counter. Whichever
way it falls, the losing half of this paragraph stays in the document.

## Build sequence

| step | | testable on its own |
|---|---|---|
| **1** | **the block limits as machine-file facts, checked before emission** | **done — 5 tests, no GPU needed** |
| **2** | **`coarsen` in the AST, parser and IR, with its refusals** | **done — 9 tests, no GPU needed** |
| **3** | **the emitter: register accumulators, the staging loop, the term loop** | **done — 7 tests, bit-exact on 7 non-divisible shapes at `2,2` and `2,4`** |
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
| `crates/lyth-ptx/src/contracted.rs` | the emitter, generalised so `coarsen 1, 1` is the path ADR-0018 already tested |
| `examples/matmul-coarse.lyth` | `tile 64, 64` + `coarsen 2, 2`, the tile step 1 refused |
| `crates/lyth/tests/coarsen_emitter.rs` | bit-exactness, the PTX structure, the bypass control, the refusal |
