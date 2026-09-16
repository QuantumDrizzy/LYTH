# ADR-0022 — The ceiling belongs to the level that binds, and it is not always DRAM

**Status:** Proposed — steps 1-3 built, step 4 built and awaiting a working GPU
**Date:** 2026-09-16
**Depends on:** ADR-0021 (coarsening, and the measurement that forced this), ADR-0015 (payload vs
bus), ADR-0000 (why)

ADR-0021 step 5 ran a tiled matmul at three tiles and found two things that do not sit together:

> The cost model is exact about a quantity that does not decide the time — ±0.80% over a
> fourfold range — and wrong about the one that does.

`tile 16` moves **twice** the global traffic of `tile 32`, per output, measured. It takes the
**same time**: 1.27 against 1.26 TFLOP/s. Meanwhile DRAM never rose above **1.7%** of the
414.51 GB/s this device was measured at, and `lyth run` printed *"with bandwidth saturated"* over
all of it.

This ADR is the repair. It is not a bug fix with an ADR attached: the fix changes what the
compiler's central claim is *about*, and that deserves the argument written down.

## What ADR-0021 measured

Six kernels, two compilers, a factor of two in time:

| | LYTH t16 | nvrtc t16 | LYTH t32 | nvrtc t32 | LYTH t64c2 | nvrtc t64c2 | spread |
|---|---|---|---|---|---|---|---|
| **shared accesses /s** | 39.7 G | 43.4 G | 39.4 G | 40.3 G | 41.1 G | 40.0 G | **10%** |
| instructions /s | 237 G | 204 G | 204 G | 159 G | 230 G | 166 G | 49% |
| L2 GB/s | 318 | 348 | 159 | 163 | 167 | 163 | **118%** |
| DRAM GB/s | 3.9 | 4.3 | 3.9 | 4.0 | 7.0 | 6.8 | — |

One quantity is flat across a factor of two in runtime and two compilers whose instruction
counts differ by 35%. It is the number of shared-memory accesses issued per second.

## The claim

> **A kernel's ceiling is set by whichever level's throughput it exhausts first. The compiler
> already derives traffic at every level it declares; it should divide each one by that level's
> measured throughput and report the smallest answer, naming which level it came from.**

DRAM is not demoted to unimportant. It is demoted to *one candidate*. For every elementwise and
transposing kernel in this repository it will still win, because those kernels stage nothing and
their shared traffic is zero or trivial. A tiled contraction is the first shape where it does
not, which is why this ADR could not have been written before ADR-0018.

## The shared level's throughput is not a bandwidth

This is the part that has to be right, and it is the part a careless version gets wrong.

The `smem` level in `fixtures/machine/sm_120.json` has a `bandwidth_gbs` field, currently `0.0`,
and a note saying it stays zero until a probe names it. **Filling it in would be a mistake.**

In the matmul's term loop a warp issues two shared loads. One reads `A`, and every lane of the
warp reads *the same address* — a broadcast, four useful bytes. The other reads `B` along a row,
32 consecutive floats — 128 useful bytes. **Both cost one wavefront.** A model in bytes per
second would price the first at 4 and be wrong by 32x, or price it at 128 and be quoting a
number that is not bytes.

That distinction already exists in this project. ADR-0015 named it:

> `moves` is the payload. `bus` is what crosses the L1-to-L2 interface, where a strided access
> costs a whole 32-byte sector.

**This is the same distinction one level down.** The payload is what the threads asked for; the
bus is the wavefront. So the smem level gets a throughput in **accesses per second**, and
`bandwidth_gbs` stays `0.0` — not because nobody measured it, but because dividing by it would
be the wrong division.

## What is derivable, exactly

Per output element, a contraction's term loop issues `ci + cj` shared accesses to produce
`ci · cj` outputs' worth of work, for each of `k` terms:

```
shared accesses per element = k · (ci + cj) / (ci · cj)
```

Checked against the profiler at 2048³, before any of this was written:

| | derived | `smsp__inst_executed_op_shared_ld.sum` x 32 | |
|---|---|---|---|
| `tile 16` | `2k` → 68.7 G | 536,870,912 x 32 = 17.2 G accesses → 68.7 G thread-accesses | **exact** |
| `tile 32` | `2k` → 68.7 G | same | **exact** |
| `tile 64` + `coarsen 2, 2` | `k` → 34.4 G | 268,435,456 x 32 → 34.4 G | **exact** |

The compiler currently derives `8 * k` **bytes** read at all three — right at the first two,
wrong by exactly 2x at the third, which is the `[KNOWN_LIMIT]` pinned in
`crates/lyth-lang/tests/coarsen.rs`. The expression above is that figure with the coarsening in
it, and `8 * k` bytes is `2k` accesses at four bytes each, so the two agree wherever the old one
was right.

### The ceiling that follows

```
flops per shared access = 2k / [k · (ci + cj) / (ci · cj)] = 2 · ci · cj / (ci + cj)
ceiling_smem            = that, times the device's access rate
```

| coarsening | flop per access | block at `tile 64` |
|---|---|---|
| none (`1, 1`) | 1.00 | 4096 — refused |
| `2, 2` | 2.00 | 1024 |
| `2, 4` | 2.67 | 512 |
| `4, 4` | 4.00 | 256 |

**Coarsening is a register-level reuse mechanism, and this is the number it moves.** It is the
same quantity ADR-0021 described in prose — `ci + cj` operands feeding `ci · cj` bodies — finally
written where the cost model can see it.

## Pre-registered, before the probe is written

Taking the access rate implied by ADR-0021's runs, ~40.1 G wavefronts/s = 1.283 T thread-accesses
per second:

| | flop/access | predicted | ADR-0021 measured |
|---|---|---|---|
| `tile 16` | 1.00 | 1.28 TFLOP/s | **1.27** |
| `tile 32` | 1.00 | 1.28 | **1.26** |
| `tile 64` + `coarsen 2, 2` | 2.00 | 2.57 | **2.63** |
| `tile 64` + `coarsen 2, 4` | 2.67 | **3.42** | not measured |
| `tile 64` + `coarsen 4, 4` | 4.00 | **5.13** | not measured |

*(Step 1 replaced the circular rate with a measured 1530.0 G accesses/s, which moves these five
predictions to 1.53, 1.53, 3.06, **4.08** and **6.12**. The originals are left above as written.)*

The first three are **circular** and are not evidence: the rate was read off those very runs.
Step 1 exists to break that circle by measuring the rate with a kernel that does nothing else,
after which those three become a test rather than a definition.

The last two are not circular. They are schedules nothing has run, at 512 and 256 threads per
block, and **the block gets narrower as the coarsening grows** — so if occupancy or latency
hiding binds before the shared pipe does, these are where it shows. A model that predicts 5.13
and measures 3 has found its own limit, and that is the outcome worth having.

## Step 1, as measured — the unit is accesses, decisively

`tools/shared_probe.py`. One kernel, one instruction stream, a **runtime** `mask` as the only
difference between the two patterns: `31` gives each lane its own address (32 consecutive
floats, 128 useful bytes per wavefront) and `0` gives every lane the same address (a broadcast,
**4** useful bytes). Same instructions, same access count, 32x the payload.

| block | pattern | G thread-accesses/s | scaling control |
|---|---|---|---|
| 256 | coalesced | 1527.2 | 1.00x |
| 256 | broadcast | 1530.0 | 1.00x |
| 1024 | coalesced | 1530.0 | 1.00x |
| 1024 | broadcast | 1531.5 | 1.00x |

**1530.0 G thread-accesses/s = 47.8 G wavefronts/s, and the spread across all four
configurations is 0.3%.**

> **Broadcast and coalesced agree to 0.2% while their payloads differ by 32x.** At that one
> rate the coalesced pattern moves **6.1 TB/s** and the broadcast pattern moves **0.19 TB/s**.

That settles the design question rather than arguing it. A `bandwidth_gbs` for this level would
have been right for one pattern and wrong by a factor of 32 for the other, and the matmul this
ADR exists for uses **both, in the same instruction pair** — `A` broadcast within the warp, `B`
along a row. The field stays `0.0` and the level gets a rate.

### The controls

* **Scaling.** Every point was run at `iters` and `2 * iters`; all eight reported the same
  throughput to within 2%, so the time doubled with the work and the loop really ran. A hoisted
  load would have shown up as a rate that climbed with `iters`.
* **The PTX was counted**, not assumed: 8 `ld.shared.f32` per iteration in the emitted code,
  asserted before anything is timed. `#pragma unroll 1` holds the shape fixed so the two
  patterns cannot be unrolled differently.
* **The context bug.** The first run died in `cuModuleLoadData` with `invalid device context`:
  torch creates its CUDA context lazily and the module was being loaded before the device had
  been touched. Allocating first fixes it. Worth a line because it fails loudly, which is the
  good case — the bad case is a probe that runs in a context nobody checked.

### And the circle is broken

ADR-0021's six matmuls paced at 39.4–43.4 G wavefronts/s. Against an independently measured
47.8, that is **82–91%** — the same relationship a saxpy has with `peak_probe.py`'s DRAM figure.
The rate is now a property of the device, measured by a kernel that does nothing else, and the
matmul numbers are a test of it rather than its definition.

Which makes the predictions in the table above concrete, and one of them **larger**: at
1530.0 G accesses/s the ceilings are 1.53, 3.06, 4.08 and 6.12 TFLOP/s, and the three measured
kernels sit at 82%, 83% and 86% of theirs.

## Step 2, as built — the rule was already in the file

The fix is four lines, and that is the finding rather than a boast. `reuse_of` has read, since
ADR-0017:

> the product of the tile dimensions of the free axes the buffer's index does not mention

Shared memory earns its reuse from the **tile**: staging is what lets the threads of a tile row
share one element of `a[i, p]`. The register file earns its reuse from the **coarsening**, by the
identical rule: a thread owning `cj` outputs along `j` loads that element once and spends it
`cj` times, so the shared pipe sees one access instead of `cj`. `register_reuse_of` is therefore
`reuse_of` with `coarsen` substituted for `tile`.

Derived, against the profiler:

| | derived shared read | loads feeding outputs |
|---|---|---|
| none | `8 * k` | 2 feed 1 |
| `coarsen 2, 2` | `4 * k` | 4 feed 4 |
| `coarsen 2, 4` | `3 * k` | 6 feed 8 |
| `coarsen 4, 4` | `2 * k` | 8 feed 16 |

The **write** side is deliberately not divided by it, and the emitter is the check: coarsening
does not change what is staged — the same tile, the same elements, spread over fewer threads
doing more each — so it is `ci · cj` stores per operand per step against `ci + cj` loads per
term. The global figure does not move either, which is the ADR-0021 claim that survives.

This also corrected a comment that **was** the bug, written down in the source:

> Everything here is about the declaration against the tile, so it needs neither the body nor
> the cost — and the cost does not need it either.

True of the global level. False of the level below. The level below is the one that decides the
time.

## Step 3, as built

`Ridge` gains `smem_accesses_gps: Option<f64>`, the `smem` level of the machine file gains
`accesses_gps: 1530.0` beside a `bandwidth_gbs` that **stays 0.0**, and the ceiling is computed
per level:

```
seconds per element at dram = bytes    / bandwidth
seconds per element at smem = accesses / access rate      (accesses = bytes / 4, one f32 each)
```

The slowest binds. For a contracted kernel the figures are the per-extent coefficients, because
that is what survives to the limit — the same `asymptotic` distinction ADR-0018 already decided.

What `lyth check` now prints for the kernel this ADR exists for:

```
  ridge    36.9 flop/byte — memory-bound
  ceiling  smem binds — 2.97 TFLOP/s, 19.39% of peak FLOPS
           dram     0.1250 byte/element/k, 301.561 ns per 1e6 of them
         * smem     1.0312 access/element/k, 674.020 ns per 1e6 of them
           the two disagree by 2.24x, so which one is quoted is not a detail
```

Three decisions in that block worth defending:

* **Both candidates are always printed, including the loser.** A ceiling that named only the
  winner gives a reader no way to tell whether it was close. Here it is not close, and a reader
  who only ever sees `dram` on their own kernels learns what the other column looks like.
* **The extent is in the unit.** `0.1250 byte/element/k`, not `byte/element`. At the size these
  were measured, dropping the `k` is a factor of two thousand — and printing a contraction's
  cost as a constant is precisely what ADR-0018 spent itself refusing.
* **"with bandwidth saturated" is gone.** It was measured false: DRAM at 1.7% of this device's
  bandwidth. `peak_bandwidth_fraction` is still on the report, still `1.0`, and now carries a
  `[KNOWN_LIMIT]` saying so — the twelve kernels that stage nothing are still described by it.

### Against the regression this ADR named in advance

> A change that makes the matmul right and a saxpy wrong has traded one error for another. The
> test is that the nine do not move.

`crates/lyth/tests/binding_level.rs` lists **twelve** and asserts each still reports `dram
binds`. Two of them are the interesting ones: `transpose-tiled` and `dot` *do* have shared
traffic, so they are exactly where a careless implementation flips. A transpose is 8 bytes and 2
shared accesses per element, which at this machine's two rates is 19,300 ns against 1,307 per
million — DRAM by a factor of fifteen. **The ceiling being per level does not mean the shared
level ever wins; it means it is asked.**

And a fourth test strips `accesses_gps` back out of the machine file and checks the matmul falls
back to `dram`, with no `smem` row offered at all. A missing rate must not become a rate of
zero — the same failure mode the block limits have, and the reason those are `Option` too.

247 tests green, clippy clean.

## Step 4, half measured — the correctness half

The instrument is built and the two new schedules are in it. `bench/cuda/handwritten_matmul.cu`
is now templated on `<T, CI, CJ>` rather than a single `C`, because at `coarsen 2, 4` the
block's two widths differ and the emitter takes a thread's column from `bj` and its row from
what is left — a template with one `C` would have been a different schedule wearing the same
name.

**All five are BIT-IDENTICAL between LYTH and nvrtc `--fmad=false`**, over 4,194,304 outputs
each, and all five are bit-exact against the host evaluator at `97x131x67` where 64 divides
nothing. The two new ones launch at **512 and 256 threads per block**.

The compiler's predictions, printed by the compiler rather than written here:

| | shared read | access/element/k | ceiling | block |
|---|---|---|---|---|
| `coarsen 2, 2` | `4 * k` | 1.0312 | **2.97 TFLOP/s** | 1024 |
| `coarsen 2, 4` | `3 * k` | 0.7812 | **3.92** | 512 |
| `coarsen 4, 4` | `2 * k` | 0.5312 | **5.76** | 256 |

`coarsen 2, 2` was measured at 2.63, which is **89%** of its ceiling. If the other two land near
that, they are ~3.5 and ~5.1 TFLOP/s. If they fall short, the narrowing block is the reason and
the model has found its own limit, which is the outcome this step was written to allow.

### Not measured: the GPU fell off the bus

The timing run died partway through with `CUDA error: unknown error`, and `nvidia-smi` then
reported:

```
Unable to determine the device handle for GPU0: 0000:03:00.0: GPU is lost.
```

A driver-level fall-off-the-bus, needing a reboot. **No throughput number for `2, 4` or `4, 4`
is recorded here**, and none will be until the machine is back and the run repeats cleanly. The
correctness results above completed before it and stand on their own; the predictions stand as
written, unadjusted, which is the point of having committed them first.

### One thing it did find

Five tests **failed** rather than skipping, because the guard listed two phrasings of "no
device" and a lost GPU produces a third: `cuInit failed: INVALID_VALUE`. The guard now treats
any `error[cuda]` before the kernel runs as no usable device. A test that cannot tell "this is
broken" from "this was not measured here" is a test that will eventually claim the wrong one.

## Build sequence

| step | | testable on its own |
|---|---|---|
| **1** | **`tools/shared_probe.py` — the access rate, from a kernel that does nothing else** | **done — 1530.0 G accesses/s, 0.3% spread, two controls** |
| **2** | **the derivation, with the coarsening in it** | **done — the `[KNOWN_LIMIT]` inverted; four coarsenings asserted** |
| **3** | **the machine file gains the rate; a ceiling per level, and the binder named** | **done — 12 kernels still `dram`, the matmul `smem`, 4 tests** |
| 4 | the falsification: `coarsen 2, 4` and `4, 4` | **built and bit-identical; the timing run is blocked on a lost GPU** |

Step 3 has a regression risk worth naming in advance: **every kernel in this repository has its
regime printed by this code path**, and nine of them are correctly DRAM-bound today. A change
that makes the matmul right and a saxpy wrong has traded one error for another. The test is that
the nine do not move.

## What this does not claim

It does not claim the shared pipe is the last level. Register bandwidth, instruction issue and
occupancy are all below it, and ADR-0021 measured instruction issue **not** binding here — LYTH
ran 35% more instructions than nvcc and finished first. What this ADR adds is a second candidate
and a rule for choosing between candidates. The next one to bind will need the same treatment,
and the structure will be there for it.

It also does not make this language compute-bound. `coarsen 4, 4` predicts 5.13 TFLOP/s against
a measured cuBLAS SGEMM of 15.30. The ceiling moves; the category does not.
