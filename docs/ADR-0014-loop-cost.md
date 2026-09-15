# ADR-0014 — What the loop costs, and why it is measured rather than derived

**Status:** Accepted
**Date:** 2026-09-15
**Depends on:** ADR-0012 (grid-stride), ADR-0009 (measured traffic), ADR-0013 (max and min)

## The question ADR-0012 answered and the one it did not

ADR-0012 pre-registered a falsification and the sweep returned it negative:

> If grid-stride does not change achieved bandwidth against one-element-per-thread, the loop
> bought nothing for performance and its justification is the measurement it enables.

At the default grid the two shapes cannot be told apart in time. That is a complete answer to
*is it faster* and no answer at all to *what does it cost*, because a memory-bound kernel hides
arithmetic behind memory latency. "No difference in time" and "no work" are not the same claim,
and only one of them was measured.

There is a second reason to ask. This compiler reports `flops_per_element` and
`bytes_per_element` as **derived**, and the loop's instructions appear in neither. ADR-0013
settled that they do not belong in the flop count — a branch is not arithmetic — which leaves
them in no count at all.

## The model, derived before it was measured

Each iteration of the loop handles exactly one element. So the loop's per-element cost **cannot**
depend on the grid: the same guard, increment and branch run once per element under every launch
shape. What the grid changes is how many elements a thread's one-time setup is spread over.

That gives an affine model with two unknowns:

```
thread_instructions = threads * S  +  n * L
```

`S` is everything a thread does once — computing its starting index and stride, and, for a
reduction, the whole shared-memory tree. `L` is everything done once per element: the body plus
the loop's bookkeeping.

The prediction worth testing is not the numbers but the **shape**. If instruction count is
affine in thread count at fixed `n`, two measurements determine the line and every other grid is
forced. `tools/inst_sweep.py` solves at grids 36 and 4096 and then checks at 144 and 576, chosen
far from both ends: a line fitted at the extremes that also lands on the middle is a model, and
one that does not is a story.

## What it measured

`smsp__thread_inst_executed.sum`, one launch, n = 2^20, block 256, RTX 5060 Ti (sm_120).
Every kernel's held-out grids were predicted **exactly** — to the instruction, not to a
tolerance.

| kernel | `S` per thread | `L` per element | one element per thread | default grid |
|---|---|---|---|---|
| `sum` | 115.02 | 6.0000 | 121.02 | 10.04 |
| `max` | 115.02 | 6.0000 | 121.02 | 10.04 |
| `saxpy` | 15.0000 | 13.0000 | 28.00 | 13.53 |
| `horner` | 17.0000 | 12.0000 | 29.00 | 12.60 |

The last two columns are instructions per element under the two launch shapes — the quantity
ADR-0012 could not see.

**Grid-stride halves the instruction count of an elementwise kernel and divides a reduction's by
twelve.** The reduction case is the one worth understanding: a block's tree is eight rounds of
load, load, combine, store and barrier, and it runs **once per thread**, not once per element.
With one element per thread that whole tree is paid for every single element. Grid-stride does
not make the tree cheaper; it makes a thread's elements share one.

So the loop's value is not that it removed work from the inner path. It removed the need to
create a thread per element, and a thread is expensive before it reaches its first element.

### `sum` and `max` execute the same instructions, exactly

Both columns are identical for the two operators, which measures a limit written into ADR-0013
the same day:

> [KNOWN LIMIT] Zero flops is not zero time. Each combine still issues.

That was a caution. It is now a number: `max.f32` and `add.rn.f32` cost one instruction each,
and the cost model reports 0 flops for one and 1 flop for the other. Both statements are correct
and they are about different things. Anyone reading `flops_per_element` as an instruction count
now has the measurement that shows why they should not.

### The loop's own share

`sum` is the leanest kernel this language can express: one load, one combine, no store per
element, and 6 instructions per element in total. How many of those six are the loop is a
separate question, and the tempting answer is an estimate. It can be measured instead, because
the loop is the only thing in these kernels that branches.

`smsp__sass_inst_executed_op_branch.sum`, under the same affine solve:

```
branches = warps * 10.0  +  elements * 1.0
```

Both coefficients whole, again. **Exactly one branch instruction per element** — the branch back
to the top of the loop; the guard is folded into it rather than costing its own. The 10 per
thread are the tree's eight rounds plus entering and leaving the loop.

So of `sum`'s six instructions per element, one is measurably the loop's branch. The index
increment and the bounds compare belong to the loop as well, and these counters do not separate
them from the body's address arithmetic, so the loop's share is **between one and about three of
the six** — bounded by measurement below and by reading the emitted code above, rather than
asserted in the middle.

It still costs no measurable time, because every kernel here is far below the ridge and bounded
by bandwidth.

That is the honest shape of the ADR-0012 result: not "the loop is free" but "the loop is
invisible at this intensity". A kernel near the ridge would see it, and this language cannot
currently write one — its highest intensity is 0.75 flop/byte against a ridge of 42.9.

## Why this is not a field in `Cost`

The obvious next move is to derive an instruction count the way bytes and flops are derived, and
check it with `--ncu` as ADR-0009 checks traffic. It is the wrong move, for a reason worth
writing down.

**Bytes survive ptxas and instructions do not.** A byte the source moves is a byte the memory
controller sees; the accounting is a statement about the program that remains true whatever the
assembler does. An instruction count is a statement about *code*, and between the PTX this
compiler emits and the SASS the machine runs sits ptxas, which reorders, folds, selects its own
instructions and unrolls. A derived per-element instruction count would be a number the compiler
asserts and cannot check, which is the exact failure ADR-0009 exists to prevent.

So the asymmetry is kept and made explicit: **traffic and arithmetic are derived and checkable;
instruction count is measured and not claimed.** `tools/inst_sweep.py` produces it, this ADR
records what it said on this device, and `Cost` stays silent about it rather than guessing.

## [KNOWN LIMIT] the two that are not integers

`S` for the reductions is 115.0195, not 115. Every other coefficient came out a whole number, as
a straight-line instruction count should. The fractional part survives the held-out check, so it
is not fitting noise; the likely cause is the ragged tail, where `n` is not a multiple of the
thread count and the last partial iteration is not identical across threads. It is not chased
here, and 115.0195 is reported rather than rounded to the number it probably is.

## Files

| | |
|---|---|
| `tools/inst_sweep.py` | the measurement, the affine solve, and the held-out check |
| `docs/ADR-0012-grid-stride.md` | the time result this completes |
| `docs/ADR-0013-max-min.md` | the known limit this turns into a number |
