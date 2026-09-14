# ADR-0009 — The byte accounting is checked against silicon, not against itself

**Status:** Accepted
**Date:** 2026-09-14
**Depends on:** ADR-0004 (machine), ADR-0005 (intensity), ADR-0006 (kernel IR)

## The hole this closes

ADR-0005 makes `intensity-check` refuse `declared_intensity != Σflops / Σbytes`. That is
real — it catches a declaration that contradicts its own body. It is also *not* a check on
the kernel. Both sides of the comparison come from a JSON accounting a human wrote after
reading the source. Put 29 bytes on something that moves 40 and it still says PASS.

That is the same failure mode Unibit already published against itself: dividing instruction
counts by an assumed 3 GHz looks like a measurement and is arithmetic on an assumption.
ADR-0004 fixed it for the machine — the claimed bandwidth is refused when it diverges from
the measured one. This ADR applies the identical pattern one level down, to per-kernel traffic.

## Decision

`intensity-check` accepts `--ncu <report.csv>` and compares the accounting to measured
traffic from Nsight Compute.

```bash
ncu --csv --kernel-name k_integrate --launch-skip 2000 --launch-count 1 \
    --metrics dram__bytes.sum,lts__t_bytes.sum ./connectome_lif.exe --steps 20 --flies 1

lyth-probe intensity-check fixtures/intensity/k_integrate.json \
  --machine fixtures/machine/sm_120.json \
  --ncu fixtures/ncu/k_integrate-sm_120-2026-09-14.csv --elements 166700
```

### The ratio is the result, not a pass mark

The accounting is a **lower bound**: the bytes the algorithm must move. Measurement can land
on either side of it, and both sides are informative.

| verdict | meaning | what to look at |
|---|---|---|
| `CONFIRMED` | within tol — the byte model describes the kernel | nothing; the roofline position is now evidence |
| `INFLATED` | more traffic than counted | cache-line granularity, uncoalesced access, partial-line RMW, register spills, traffic the accounting omits |
| `ABSORBED` | less traffic than counted | it never left cache — the accounting names the wrong level, or the working set fits at this size |

Exit 0 on `CONFIRMED`, 1 on either divergence, 2 when the report or the problem size is
unusable. A divergence is a failure of the *claim*, not necessarily of the kernel — so the
output names the shortlist instead of just refusing.

### An element count is mandatory, and is never guessed

The accounting is per element (per neuron, per output row). `ncu` reports a total. Without
the problem size relating them, any ratio can be made to come out at 1.0, so `--ncu` exits 2
rather than assume one. Supply `--elements N`, or put `"elements"` in the case / kernel IR —
it is carried through `poly` instantiation and `kernel` lowering.

### The level is taken from the accounting, and can be overridden

Each `move` names a level. The deepest one selects the metric: `dram` → `dram__bytes.sum`,
`l2` → `lts__t_bytes.sum`. Comparing an L2-resident kernel against DRAM reads as a failure
that is indistinguishable from a real one, so the report also prints every *other* level it
measured, with its own ratio. The alternative hypothesis is confirmed or killed from the same
profiling run. `--ncu-level l2` re-checks the same accounting one level up.

## Result on the first kernel it was pointed at

`k_integrate`, the connectome LIF integrate step — 166,700 neurons, RTX 5060 Ti, one launch.

```
analytic:   4.834 MB  (29.0000 bytes/element)
measured:   4.736 MB  (28.4084 bytes/element)   lts__t_bytes.sum
ratio:    0.9796x                               CONFIRMED at l2
also dram   2.509 MB  (0.5190x analytic)
```

**The 29-byte model is confirmed against silicon to 2.1%.** The label on it was wrong: those
moves are declared `dram`, and at F = 1 the 9.00 MB state is L2-resident, so the memory
controller sees 15.05 B/neuron — 52% of the count. Run without `--ncu-level`, the tool
reports `ABSORBED` and names cache residency as cause [1]; the `also l2` line proves it in
the same output.

This is the shape of finding the layer exists for. The arithmetic check said PASS and was
right about the arithmetic. It could not have told anyone that half the traffic does not exist
on this machine at this size.

**[KNOWN LIMIT]** The instance count at which the working set exceeds L2 and the DRAM lower
bound is actually reached is **not measured**. Until it is, `29 B/neuron of DRAM traffic` is a
statement about the design, not about this device.

**[KNOWN LIMIT]** One launch, one problem size, clocks unlocked. Byte counters are
deterministic per launch so no repetition is claimed or needed; no timing is reported here.

## Found by running it

`ncu` formats through the host locale. On this machine it printed 2,509,056 as `2.509.056` —
dots grouping the digits. Read as a decimal that is 2.5 bytes and a ratio wrong by six orders
of magnitude, which is precisely the kind of number that looks like a result. `parse_value`
now resolves the separator by position and by unit rather than assuming a convention, and the
real report is checked in as a regression test. `--csv` implies `--print-units base`, so a
byte metric is an exact count and a lone separator in it must be grouping.

## Second kernel: the check pointed back at the lab

`rmsnorm_fused_block`, exercise 01, N=2048 × D=8192. The kernel reads `x` twice; the
accounting declared one DRAM read on the claim that L2 serves the second. Measurement agrees
at L2 (12.79 B/element ≈ x twice + y once) and comes in **11.4% under** the count at DRAM.

That gap is the finding. Exercise 01 publishes **385.71 GB/s** by dividing its analytic 128 MiB
working set by 348.06 µs; measured traffic gives **341.57 GB/s**. Overstated by 12.9%, and
385.71 is **107.6% of this machine's own measured reference bandwidth**.

Full record, including what is and is not established, in `docs/DOGFOOD.md`.

## What would falsify this

- If `--ncu` reports `CONFIRMED` on an accounting that is demonstrably wrong about a kernel,
  the check is theatre. Guard: the element count is refused rather than inferred, so the one
  free parameter that could be tuned to force agreement is not available.
- If `INFLATED` and `ABSORBED` fire on almost every real kernel, the tolerance is a fiction
  and the honest output is a ratio with no verdict at all.

## Two corrections the dogfood forced (2026-09-14)

### Direction, and why the first write-up of `k_integrate` was imprecise

The original record above reads "at F = 1 the 9.00 MB state is L2-resident, so the memory
controller sees 52% of the count". Directionally right, imprecise, and the precise version is
more useful. Splitting the same launch by direction:

```
dram read   2,508,800 B = 15.048 B/neuron
dram write          0 B = exactly zero
```

The accounting's 29 bytes are **15 read and 14 written**. Measured read matches the read half to
**0.32%**, and the write half contributes nothing at all: every store retired into L2 as a dirty
line and none was evicted inside the launch.

So the accounting is not half-wrong — it is exact, and a single-launch DRAM measurement can only
see its read half for a kernel whose working set fits in cache. Hence `--ncu-dir all|read|write`,
which restricts the accounting using the `dir` each move already carried and the checker was
discarding, and hence `ABSORBED` naming write-back absorption as cause **[1]**, before it
suggests the accounting is at fault.

**[CORRECTED]** The `ncu.rs` fallback constants were `dram__bytes_read.sum`. On sm_120 the
metric is `dram__bytes_op_read.sum`. The wrong name does not error — ncu prints the row with
`n/a` — so that path could never have fired. Checked against `ncu --query-metrics` on the device.

### `--elements-from`, for a kernel whose element count is not knowable in advance

`k_propagate` walks the out-edges of whichever neurons happened to fire. Its element count
changes with its input, cannot be typed in after the fact, and cannot be recovered from the byte
counts. The frozen-step launch the exercise reports could not be isolated by launch index.

The element count is therefore **measured from the same report**. The `atomicAdd` discards its
result, so it lowers to a global reduction, one per edge:

```
--elements-from l1tex__t_sectors_pipe_lsu_mem_global_op_red.sum
```

Result: **INFLATED 3.15x**, 10 B/edge counted against 31.50 measured — and the cause is measured
rather than picked off the shortlist. 39,062 red sectors over 1,447 warp-level reduction
instructions is **27.0 sectors per instruction out of a possible 32**: nearly every thread's
4-byte atomic in its own 32-byte sector. `39,062 x 32 B = 1,249,984 B` against a measured DRAM
read of `1,230,336 B` — 98.4%. The model that describes this kernel at DRAM is one sector per
edge and nothing else.

Full record and limits in `docs/DOGFOOD.md`.
