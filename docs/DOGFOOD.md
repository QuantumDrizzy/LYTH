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
| 1 | `k_integrate` | rse-hpc-lab 08-connectome-lif | **CONFIRMED at l2** | 28.41 vs 29.00 B/neuron — 0.980x |
| 2 | `rmsnorm_fused_block` | rse-hpc-lab 01-fused-rmsnorm | **ABSORBED at dram** | 7.09 vs 8.00 B/element — 0.886x |

### 1 — `k_integrate`, 166,700 neurons

The 29-byte model is confirmed against silicon to 2.1%. Its **label** was wrong: the moves
declare `dram`, and at one fly the 9.00 MB state is L2-resident, so the memory controller sees
15.05 B/neuron — 52% of the count. Full record in ADR-0009.

**Would a macro have caught it?** A macro could hold the same accounting and check the same
arithmetic. It could not have known which level the traffic actually crosses. **Neither can
LYTH.** That came from `ncu`, and the tool's contribution was putting the two numbers side by
side and printing the neighbouring level automatically.
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

---

## The parser question, current standing

**Days elapsed of three weeks: 1. Kernels measured: 2. Evidence that a parser is needed: none
so far.**

Both findings came from comparing a hand-written accounting against a measurement. A
`#[kernel]` attribute macro carrying the same accounting would have produced both. The thing
that found them is the *comparison*, and the comparison is a CLI.

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
