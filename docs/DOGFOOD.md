# Dogfood log — the three weeks that decide whether LYTH gets a parser

ADR-0001 put a question in front of the parser and said it must stay open:

> **What can LYTH express that a Rust DSL with procedural macros cannot?**
>
> **Falsification:** if after three weeks of dogfooding `lyth probe` on hand-written CUDA it
> turns out `#[kernel]` + macros cover ~90%, **stop the parser, publish the negative result**,
> keep the probe CLI.

This file is the evidence for that decision. One row per kernel put through the tool, and for
each one the only question that matters: **would a macro have given me this?**

Nothing here is a plan. A kernel enters this table after it has been profiled.

---

## Kernels measured

| # | kernel | source | verdict | measured vs counted |
|---|---|---|---|---|
| 1 | `k_integrate` | rse-hpc-lab 08-connectome-lif | **CONFIRMED at dram, read half** | 15.048 vs 15.000 B/neuron — 1.003x |
| 2 | `rmsnorm_fused_block` | rse-hpc-lab 01-fused-rmsnorm | **ABSORBED at dram** | 7.09 vs 8.00 B/element — 0.886x |
| 3 | `k_propagate` | rse-hpc-lab 08-connectome-lif | **INFLATED at dram** | 31.50 vs 10.00 B/edge — 3.150x |

### 1 — `k_integrate`, 166,700 neurons

First read as "the byte model is confirmed at L2 and mislabelled `dram`". That was directionally
right and **imprecise**, and the precise version is better. Splitting the traffic by direction:

```
dram read   2,508,800 B = 15.048 B/neuron
dram write          0 B = exactly zero
```

The accounting's 29 bytes are 15 read (ring 4, refrac 2, adapt 4, v 4, is_stim 1) and 14 written.
**Measured DRAM read matches the read half to 0.32%.** The write half contributes *nothing*:
every store retired into L2 as a dirty line and not one was evicted before the kernel ended.

So the accounting is not half-wrong. It is exactly right, and **a single-launch DRAM measurement
can only ever see its read half** for a kernel whose working set fits in cache. Comparing the
full 29 bytes against a read-only measurement reads as a 48% failure that does not exist.

This is why `--ncu-dir` exists, and why `ABSORBED` now names write-back absorption as cause [1]
before it suggests anything is wrong with the accounting.

**Would a macro have caught it?** A macro could hold the same accounting and check the same
arithmetic. It could not have known that the writes never leave L2. **Neither can LYTH.** That
came from `ncu`; the tool's contribution was holding the `dir` of each move — which the
accounting already carried and the checker was throwing away — and comparing like with like.
→ *Argues for the probe, not for a parser.*

### 2 — `rmsnorm_fused_block`, N=2048 × D=8192

The kernel reads `x` twice: once for the sum of squares, once to normalise. The accounting
declared **one** DRAM read, claiming the second is served by L2. Measurement settles it:

```
l2    214,566,432 B  = 12.79 B/element   ~= x twice + y once   -> the claim holds
dram  118,885,888 B  =  7.09 B/element   =  0.886x the counted 8
```

**The finding is about the lab, not about LYTH.** Exercise 01 reports **385.71 GB/s** for this
kernel by dividing its 128 MiB analytic working set by 348.06 µs. Measured traffic gives
**341.57 GB/s**. The headline is overstated by **12.9%** because it counts bytes rather than
measuring them.

The tell was available without any profiler: 385.71 GB/s is **107.6% of this machine's own
measured reference bandwidth** (358.43 GB/s, `fixtures/machine/meas-sm_120-2026-09-14.json`).
A number above the machine's own measured ceiling should never have been published.

**[KNOWN LIMIT]** "Above peak" is *not* established. 358.43 GB/s is itself a lower bound — a
`torch.sum` reduction under unlocked clocks — and the card's spec figure is 448 GB/s, so
385.71 is not physically impossible. The established claim is narrower and sufficient: the
kernel moves 12.9% fewer bytes than the headline divides by.

**Would a macro have caught it?** Yes, in principle — this is a byte count against a measured
byte count, and nothing about it needs a new syntax.
→ *Argues for the probe. Still no evidence for a parser.*

### 3 — `k_propagate`, CSR edge walk with a scattered atomic

The hard case, and the one that broke an assumption in the tool.

**The element count is data-dependent.** An element is one touched out-edge, and that number
changes with which neurons fired. It cannot be typed in after the fact and it cannot be
recovered from the report's byte counts. The exercise prints `424 spikes / 15,858 edges` for its
frozen-step benchmark, but that launch could not be isolated by index: `--launch-skip` at 12,000
and at 42,000 both landed in the main loop, identified by `k_propagate` alternating with
`k_advance` rather than with `k_bump_step`.

The way out is to **measure the element count from the same report**. The `atomicAdd` discards
its result, so it compiles to a global reduction — one per edge:

```
--elements-from l1tex__t_sectors_pipe_lsu_mem_global_op_red.sum
```

That is a new flag, and it is the only honest source for a kernel like this one.

**Result: INFLATED 3.15x.** 10 B/edge counted as reads, 31.50 B/edge measured.

The cause is *measured in the same report*, not chosen from a list:

```
39,062 red sectors / 1,447 warp-level reduction instructions = 27.0 sectors per instruction,
                                                               out of a maximum of 32
```

Nearly every thread's 4-byte atomic lands in its own 32-byte sector. The decomposition closes:

```
39,062 sectors x 32 B      = 1,249,984 B
measured dram read         = 1,230,336 B   -> 98.4% of it
col_idx + weight, 8 B/edge =   312,496 B   -> never reaches dram; L2 serves it
l2 measured 1,643,424 vs ring+CSR 1,562,480 -> 1.05x
```

**The model that actually describes this kernel at DRAM is one 32-byte sector per edge and
nothing else.** The accounting's `4 bytes` for the ring atomic is the payload; the machine moves
a whole sector for it.

**[KNOWN LIMIT]** The element count used is red *sectors*, a **lower** bound on edges — two
threads sharing a sector count once — so 31.50 B/edge is an **upper** bound.

**[KNOWN LIMIT]** This is a main-loop launch (~39k edges), not the frozen-step launch (15,858
edges) the exercise reports. Nothing here is attributed to the frozen launch.

**[KNOWN LIMIT]** The exercise reports 18.7 GB/s for scatter from a ~12 B/edge count. If
31.5 B/edge holds at the frozen step, that is understated ~2.6x and the reported "11.4x worse
efficiency than cuSPARSE" is overstated by the same factor. **Not published as a revised ratio**
— the cuSPARSE side was not measured and the frozen launch was not isolated.

**Would a macro have caught it?** No — and neither would LYTH without `ncu`. But note what the
*declaration* bought: `dir` per move is what made the read/write comparison possible, and
`level` per move is what made "wrong level" a checkable hypothesis. Both are attributes. A macro
could carry them.
→ *Argues for the probe. Still no evidence for a parser.*

---

## The parser question, current standing

**Days elapsed of three weeks: 1. Kernels measured: 3. Evidence that a parser is needed: none
so far.**

All three findings came from comparing a hand-written accounting against a measurement. A
`#[kernel]` attribute macro carrying the same accounting — including the `dir` and `level` of
each move, which is what made findings 1 and 3 checkable — would have produced all three. The
thing that found them is the *comparison*, and the comparison is a CLI.

Kernel 3 is the closest thing to a counter-argument so far, and it argues the other way: what
the tool needed was not a richer language but a way to **measure** a quantity the declaration
could not know. That is `--elements-from`, and it is a flag.

What would count as evidence the other way, stated now so it cannot be adjusted later:

1. An accounting that is unreadable as attributes — enough streams, levels and conditions that
   the macro form stops being writable by a human.
2. A refusal that needs the compiler to see the **body**, not a declaration beside it: the
   arithmetic contradicting the declared movement in a way no attribute could state.
3. Lowering one source to PTX **and** SPIR-V **and** LLVM with compile-time refuse turning out
   to be materially harder inside macros than as a frontend.

None observed yet.

---

## Cross-cutting: what the lab is missing

`lyth-probe gap` over all 11 exercises returns the same four fields missing every time:
`arch`, `compile_flags`, `clock_state`, `cache_state`. Two exercises (00, 01) also have no
baseline.

`cache_state` is not bookkeeping. Finding 2 exists because a benchmark loop reruns over the
same 128 MiB buffers and part of `x` survives in L2 between iterations. An exercise that does
not declare its cache state cannot be compared to one that does, and the field was missing in
exactly the exercise whose headline turned out to be flattered by cache carry-over.

Both emitters are in one place each: `common/evidence.hpp` and `labkit/schema.py`.
