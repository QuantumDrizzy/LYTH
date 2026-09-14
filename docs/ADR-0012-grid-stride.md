# ADR-0012 — Grid-stride, and the first performance number

**Status:** Accepted
**Date:** 2026-09-14
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
