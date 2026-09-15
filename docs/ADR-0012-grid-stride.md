# ADR-0012 — Grid-stride, and the first performance number

**Status:** Accepted
**Date:** 2026-09-14

> **[SCOPE, added 2026-09-16]** Everything below was measured on `saxpy`, and the default
> it established was applied to every kernel. For a **reduction** it is wrong by a factor of
> two: the block tree runs once per thread, so one element per thread runs it once per
> element. `sum` at n = 2^26 measures 213.69 GB/s at that default against 419.68 at grid
> 36864. The default is now per kernel kind. A conclusion carries the kernel it was measured
> on — see ADR-0000.

**Depends on:** ADR-0010 (executable LYTH), ADR-0011 (reductions), ADR-0004 (machine as value)

## Why this is not really about a loop

ADR-0010 carries a limit written before its code:

> One element per thread, bounds-checked. No grid-stride loop in v1, so occupancy tuning is not
> expressible and any performance number from it is uninteresting. **This slice is about
> correctness and honest cost, not speed.**

That is still true, and the loop is the thing that ends it. With one element per thread the
launch shape is a function of `n` and nothing about it can be varied, so there is no experiment
to run. A grid-stride loop decouples the grid from the problem size, and **that is what makes a
measurement possible at all.**

So this ADR does two things: the loop, and the first performance number this compiler is allowed
to produce. The second is the one that needs rules.

## The rules for the number, fixed before it is measured

The house standard, applied to a compiler's own output:

1. **A baseline beside it or it does not ship.** The baseline here is the machine file's own
   measured DRAM bandwidth — 358.43 GB/s, median of six runs — not a datasheet figure. Achieved
   bandwidth is reported as a percentage of that, and the file it came from is named.
2. **Bytes measured, not assumed.** Exercise 01 of rse-hpc-lab overstates its bandwidth by 12.9%
   by dividing an analytic working set by a time (`docs/DOGFOOD.md`, kernel 2). This compiler
   derives its own byte count, which makes it *more* liable to that mistake, not less. The
   reported figure uses the derived count and the report says so; `--ncu` remains how the count
   itself is checked.
3. **N ≥ 5, median and full spread, warm-up discarded, cache state stated.** A single timing is
   not a measurement.
4. **No comparison against CUDA in this ADR.** A fair comparison needs a hand-written kernel
   built to the same standard, and inventing one to lose to or beat is exactly how a benchmark
   flatters itself. What ships here is "this kernel reaches X% of this machine's measured
   bandwidth", which is a statement about the machine and the kernel, not a race.

**What would falsify the loop's value:** if grid-stride does not change achieved bandwidth
against one-element-per-thread on this device, the loop bought nothing for performance and its
justification is the *measurement* it enables, not speed. That result gets published either way.

## The loop

```
i      = ctaid * ntid + tid
stride = nctaid * ntid
loop:
    if i >= n: exit
    <body at i>
    i += stride
    goto loop
```

Element addresses move **inside** the loop, since each iteration touches a different element.

## What it does to a reduction

More than it looks like. With one element per thread, a thread contributed one value. Now it
accumulates over its own strided elements first, and only then does the block tree run:

```
acc = identity
i   = ctaid * ntid + tid
while i < n:
    acc = combine(acc, <body at i>)
    i  += stride
smem[tid] = acc
<tree>
```

Two consequences.

The **idle-thread branch from ADR-0011 disappears**: `acc` starts at the identity, so a thread
with no elements simply never enters the loop and its slot is already correct. That was the
subtlest part of the previous design and grid-stride deletes it.

The **order changes, and so must the host reference**. A thread now folds its elements
sequentially — `((e0 ⊕ e_S) ⊕ e_2S) ⊕ …` — before the tree combines threads. Float addition is
not associative, so the host has to walk that same two-stage order or the bit-exact check fails
on a correct kernel. ADR-0011 already established the principle; this changes the shape it
applies to, and the grid now has to be known to the reference as well as the block.

## Choosing the grid

The grid stops being determined by `n`, so something must choose it. Options rejected:

- **A constant.** A magic number tuned on one machine, which is the flag-shaped thing ADR-0001
  argues against.
- **Still `ceil(n / block)`.** Correct, and it makes the loop run exactly once per thread, which
  measures nothing new.

Taken: **derive it from the device** — `multiProcessorCount × waves`, capped at the blocks the
problem actually needs, with `--grid` to override so the choice can be swept and measured. The
SM count comes from `cuDeviceGetAttribute`, not from the machine file: it is a property the
driver reports about the silicon present, and a machine file that disagreed with it would be
describing a different card.

**[KNOWN LIMIT]** `waves` is a guess until swept. It is a starting point, not a tuned value, and
the sweep is the point of exposing `--grid`.

## Build order

| stage | ships | done when |
|---|---|---|
| 1 | grid-stride for elementwise kernels | bit-exact at a grid smaller than `n / block` |
| 2 | grid-stride reductions; host reference folds then trees | bit-exact for `dot` at several grids |
| 3 | `--grid`, device-derived default | SM count read from the driver |
| 4 | timing: N reps, warm-up, median, spread | `lyth run --time` reports all four |
| 5 | achieved bandwidth against the machine file | the baseline is named in the output |

## [KNOWN LIMIT], before the code

- Timing measures the kernel only, by CUDA events around the launch. Allocation and the
  host-to-device copy are excluded and that is stated in the output; a user timing a whole
  program will not see these numbers.
- Clocks are not locked. The machine file's bandwidth was measured unlocked too, so the ratio is
  between two numbers taken under the same policy — which is the only reason the percentage
  means anything.
- The reported byte count is the compiler's derived one. It is checked against `ncu` separately
  (ADR-0009) and that check is not automatic here.

---

## The falsification, tested — and it came back negative (2026-09-14)

The section above recorded that this ADR's own falsification had not been tested, and why:

> The one-element-per-thread path no longer exists to compare against, which was a scoping
> mistake. It can be recovered by forcing `--grid` to `ceil(n / block)`.

`tools/grid_sweep.py` does that. saxpy, n = 67,108,864 (256 MB per buffer, far past L2), five
**interleaved** passes of seven timed runs each. Interleaved because running every repetition of
one grid and then the next attributes a thermal drift to whichever grid was measured late; each
pass visits every grid once, so a drift becomes noise shared by all points instead of a result
at one of them. Raw records: `fixtures/sweeps/grid-sweep-saxpy-sm_120-2026-09-14.json`.

```
      grid      GB/s         ms   within   across   vs base
         1      9.41    85.5819    0.8%    1.5%      2.6%
         9     82.04     9.8154    0.3%    0.2%     22.9%
        18    150.24     5.3601    0.4%    5.3%     41.9%
        36    258.99     3.1094    0.4%    0.1%     72.3%     1 block/SM
        72    370.34     2.1745    1.6%    0.1%    103.3%     2 blocks/SM
       144    386.63     2.0829    1.6%    0.4%    107.9%     the old default
       288    383.34     2.1007   15.0%    0.3%    107.0%
       576    389.53     2.0674    1.4%    0.1%    108.7%
      1152    391.20     2.0586    1.7%    0.1%    109.1%
      4608    393.71     2.0454    1.7%    0.1%    109.8%
    262144    399.70     2.0148    1.9%    0.1%    111.5%     one element per thread
```

**The answer is negative, and it is the useful direction.** At the default grid, grid-stride was
**3.3% slower** than the shape it replaced — outside the 0.4% across-pass spread of both points,
so not noise. The best grid-stride point (4608) is still 1.5% slower.

The reason is the expected one for a memory-bound kernel: at grid 144 there are 36,864 threads
each folding ~1,820 elements; at 262,144 there are 67 million threads each doing one. **More
threads in flight is more outstanding loads**, and this kernel is waiting on memory, not on
arithmetic.

### Acted on

The default was SM count × 4 waves, justified by "a grid-stride loop wants enough blocks to fill
the machine and no more". The sweep says that reasoning is wrong here, so **the default is now
one element per thread**, capped at 2^20 blocks because a large enough `n` would need more than
one launch can carry. The loop engages past that cap or when `--grid` asks. `WAVES` is gone.

Measured after the change, same command as the section above: **399.64 GB/s, spread 1.9%** —
against 386.95 and 16.2% before it. The larger grid is both faster and considerably steadier.

### Two things the table says that the verdict does not

**The saturation curve is the more useful result.** One block per SM reaches 72% of the plateau,
two reach 93%, four reach 97%. For anyone sizing a launch on this device, that curve is worth
more than the 3.3%.

**Interleaving earned its keep.** Grid 288 shows a 15.0% `within` spread — one timing call caught
something else running — while its `across` spread is 0.3%. Had the passes not been interleaved,
that one bad call would have been the whole figure for that grid.

### [KNOWN LIMIT] What was *not* measured

The comparison is **small grid against large grid, both with the loop**. It is not loop against
no-loop: at 262,144 blocks each thread still executes the loop head twice, once to enter and once
to find itself past the end. The old no-loop codegen was deleted before the comparison was taken,
so the loop's own overhead remains unmeasured. It is bounded above by the numbers here — whatever
it costs, the full-grid shape pays it too and still wins.

One kernel, one size, one device, one block size. `--grid` and `tools/grid_sweep.py` exist so the
next person does not have to take this on faith.
