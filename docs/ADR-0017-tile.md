# ADR-0017 — The tile, and the first cost that depends on the schedule

**Status:** Accepted — built, measured, and depended on

*(Status corrected 2026-09-16. This said `Proposed` while `tile` was shipping in the emitter,
measured to +0.80% by ADR-0022 step 4, and depended on by ADR-0018, ADR-0021 and ADR-0022. A
status line that says a built thing is proposed is a document lying about the code.)*
**Date:** 2026-09-16
**Depends on:** ADR-0000 (why), ADR-0015 (shape), ADR-0016 (callable), ADR-0011 (reductions)

## Why this one is the point of the project

ADR-0015 made the traffic model able to be wrong and then measured it right: a transpose costs
36 bytes per element at the L1-to-L2 interface against a payload of 8, and the model predicted
that to 0.1%.

But look at what the source says. `b[j, i] = a[i, j]` with `space i, j` states an algorithm, and
the 36 is a consequence of a *schedule* the language never wrote down — one element per thread,
no staging, whatever order the flattening produced. The compiler derived a cost for a schedule
that was implicit, and a cost model over an implicit schedule cannot be *wrong* in the way that
matters: there was no alternative to compare it against.

A tile changes that and nothing else:

```
kernel transpose(rows: u32, cols: u32, a: [f32; rows, cols], b: [f32; cols, rows])
    space i, j : rows, cols
    tile 32, 32

    stream a : dram -> smem -> reg
    stream b : dram -> reg, drain

    at reg:
        b[j, i] = a[i, j]
```

**The body is untouched.** Same algorithm, same flops, same payload. What changed is two declared
lines, and the derived bus cost should fall from 36 bytes per element to 8 — the same as a
straight copy. That is the first time in this language that:

* two programs with identical semantics and identical payload have different derived costs;
* the difference comes from a declaration the author wrote rather than from the body;
* and the compiler can refuse a claim about it.

That is the thesis stopping being degenerate. Everything before this ADR derived a cost that was
a function of the source alone, which is the easy case and the one nobody gets wrong.

## Decision

### The tile blocks the space, and the body does not know

`tile 32, 32` says each block handles a 32 by 32 patch of the index space. Extents need not
divide it; the emitter guards the edges. The tile's dimensions must be **powers of two**, so a
thread's position inside it is a shift and a mask rather than a division — the same reasoning
that put `div.u32` in ADR-0015's cost and the same refusal shape as `--block`.

### Staging is a stream with a longer path

`stream a : dram -> smem -> reg` already parses: `reduce` declares `reg -> smem -> dram` and the
level vocabulary is shared. A staged stream loads a tile into shared memory, synchronises, and
the body reads from there. **This is the language's existing idea, not a new one:** movement is
declared, and the compiler checks the arithmetic fits it.

### The block stays one-dimensional

A 32 by 32 tile is 1024 threads, and `tx = tid & 31`, `ty = tid >> 5` recovers the position in
two instructions because the width is a power of two. The alternative — a 2-D `blockDim` — would
change `cuLaunchKernel` in all three bindings ADR-0016 just closed, the grid heuristic, and the
reduction tree's assumption that a block is a line of threads. It buys two instructions per
thread. It is not worth it.

### The padding is derived, not declared

Shared memory on sm_120 is 32 banks of 4 bytes. A `[32][32]` tile of f32 read down a column puts
all 32 threads of a warp on the same bank: a 32-way conflict, serialised. One element of skew per
row — `[32][33]` — rotates the bank by one per row and the conflict goes to zero.

That 33 is a function of the bank count in the machine file and the tile width. The compiler
derives it, the way it derives bytes and flops, and a caller who wanted to write it would be
doing arithmetic the compiler would then have to check — which means the compiler knows the
answer already.

**And bank conflicts are a cost, so this is a second falsifiable prediction.** The compiler
states zero; `l1tex__data_bank_conflicts_pipe_lsu.sum` decides.

The machine file grows `smem_banks` and `smem_bank_width_bytes`. They are architectural
constants rather than measurements, so they are recorded with that provenance — the machine file
already distinguishes a measured bandwidth from a stated capacity, and the conflict counter is
what checks the consequence.

### The grid becomes a grid of tiles

`ceil(rows / 32) * ceil(cols / 32)`, not `ceil(rows * cols / block)`. This is the one change that
reaches outside the compiler: the manifest's `grid_rule` is consumed by three generators that
currently hard-code the element rule. The rule moves into the manifest as data the generators
read rather than a sentence they ignore.

## The PTX, in four phases

1. **Position.** `%ctaid` gives the tile, `%tid` gives the position inside it by shift and mask,
   and the two combine into a global `(i, j)` with a bounds guard per phase, not one for both.
2. **Coalesced load into shared.** Threads read consecutive elements of a row of `a` and write
   them into the padded tile. Both sides contiguous.
3. **`bar.sync 0;`** — the whole block, before any thread reads what another wrote.
4. **Transposed read from shared, coalesced store.** Threads read the tile down its columns,
   which the skew makes conflict-free, and write consecutive elements of a row of `b`.

The transposition moves from DRAM, where it costs 32 bytes per element, into shared memory,
where the skew makes it cost nothing. That sentence is the whole optimisation and the cost model
should be able to say it.

## The falsification, pre-registered in full

Two claims, two runs, deliberately **not** in the same report. Reading them together means
negotiating one number against the other.

### Claim 1 — the traffic

> At 4096 x 4096, `lts__t_bytes.sum` divided by `rows * cols`: the untiled body measures
> **36.00 ± 0.2** and the tiled body measures **8.00 ± 0.2**. Same `ncu`, same machine file,
> same extents, same counter.

The tolerance admits the 36.05 already measured and refuses a 12, or a 36 wearing a tile. **At
L2 and not at DRAM**: that counter counts what crosses the interface whatever the cache does
with it, which is what makes the two kernels comparable. DRAM is a statement about capacity —
ADR-0015 measured 4.13 bytes per element at 1024² there, *half the payload*, because the L2 kept
the writes.

1024² is a smoke test and 8192² is a cache-capacity experiment. Neither is this claim.

`intensity` is not part of it and must not move: 0 flops over 8 payload bytes is 0 at both ends.
If the intensity check complains about this kernel, the bug is in the check.

#### What it measured

4096 x 4096, `lts__t_bytes.sum` over `rows * cols`, one `ncu` invocation per kernel, same
machine file, same extents. The direction split is `lts__t_sectors_op_{read,write}` at 32 bytes
a sector — `lts__t_bytes` has no per-direction suffix on this device, checked with
`--query-metrics` rather than recalled, which is the second time that lesson has paid.

| kernel | total | read | write | derived |
|---|---|---|---|---|
| `transpose` | 38.97 | 4.03 | 34.94 | 36 |
| `transpose-tiled` | **8.00** | **4.00** | **4.00** | **8** |
| `copy2d` | 8.00 | 4.00 | 4.00 | 8 |

**The tiled body measures 8.00, exactly, against a derived 8.00 and a pre-registered tolerance
of ± 0.2.** It matches the coalesced control to two decimals, and the split is symmetric, which
rules out the write-allocate contaminant: a store that provoked a line fill would show read
above 4 with write at 4, and a schedule that never reached the bus would show write near 32.

Same body. Same payload. Two declared lines.

#### The pre-registration was wrong about the control, and the measurement caught it

The control was pre-registered at **36.00 ± 0.2** and measured **38.97**, which is outside that
tolerance. The error is in the pre-registration, not the measurement, and it is a careless one:
**ADR-0015 had already measured 39.02 at this exact size and explained the excess** — partially
written sectors evicted before the other seven writes arrive, and fetched back. At 1024² and
2048², where the working set fits in L2, it measured 36.05 and 36.02.

Writing "36.00" here took the *model's* number and recorded it as the *expected measurement*,
against a published measurement of the same kernel at the same size in this repository. The
tolerance was right and the centre was wrong. Recorded rather than adjusted after the fact.

#### And the tiled kernel has no excess at all

38.97 overshoots its model by 2.97. **8.00 overshoots its model by nothing.**

That is not luck and it was not pre-registered. The read-for-ownership traffic in the untiled
kernel exists because a strided store touches one 4-byte word of a 32-byte sector and the sector
is evicted before the other seven arrive. In the tiled kernel every sector is filled completely
by one warp in one instruction, so there is no partial sector to evict and nothing to fetch
back. **Absorbing the permutation does not only reduce the sector count; it removes the mechanism
that made the sector count an underestimate.**

[NOTED, not fixed here] The manifest labels this level `dram` while the counter it names is
`lts__t_bytes.sum`, which is L2. The level is right and the name is wrong. Changing it in the
report that uses it would be editing the instrument during the experiment.

### Claim 2 — the skew, with its counterfactual

> The nominal binary (stride 33) measures `l1tex__data_bank_conflicts_pipe_lsu.sum` = **0**.
> The counterfactual (stride 32, `emit_with_skew(.., false)`) measures **conflicts far above
> zero** — a 32-way conflict on every transposed shared load, so orders of magnitude and not
> slightly positive.

Zero on its own shows the conflicts are absent, not that the skew removed them: that number is
also zero if the kernel never staged, if the emitter dropped the padding, or if the counter does
not cover the instruction. The counterfactual is what turns an absence into a cause.

**And the expectation is per instruction.** The shared *store* is `ty*33 + tx`: neighbouring
threads differ in `tx`, so it is conflict-free at stride 32 as well. Only the transposed *load*,
`tx*33 + ty`, collapses onto one bank without the skew. A counterfactual that reports conflicts
on loads and none on stores is the experiment working — written down here so that nobody
"fixes" a correct result.

If the counterfactual also reads zero, the instrument is not measuring what it is being asked to
measure, and that is a finding in itself.

#### What it measured

1024 x 1024. The nominal binary against `--no-skew`, which emits the same kernel with the
padding removed. Before measuring contention, the counterfactual was checked for **semantics**:
it comes out `BIT-EXACT` against the host, so it is the same computation and only the layout
differs. Conflicts without that check would be a number about some other kernel.

| variant | `mem_shared_op_ld` | `mem_shared_op_st` |
|---|---|---|
| nominal, stride 33 | **0** | 4172 |
| counterfactual, stride 32 | **1,017,813** | 4566 |

**The claim holds, and the magnitude is the mechanism.** At 1024² there are 1,048,576 / 32 =
32,768 warp-level shared loads, and a 32-way conflict costs 31 extra accesses each:
32,768 x 31 = **1,015,808**. Measured 1,017,813, which is 0.2% from full 32-way serialisation.
The skew is not merely correlated with the absence of conflicts; the counterfactual reproduces
the exact serialisation the derivation says it removes.

The counter names were queried with `--query-metrics` before the run rather than taken from this
document. Third time that has paid.

#### The second pre-registration was also wrong, and in the same way as the first

This ADR predicted `st = 0` in **both** variants, reasoning that the shared store is
`ty * stride + tx` and neighbouring threads differ in `tx`, sweeping all 32 banks at either
stride. The reasoning is right. The prediction was still too strong.

| size | `op_st` nominal | `op_ld` nominal |
|---|---|---|
| 256² | 0 | 0 |
| 512² | 0 | 0 |
| 1024² | 4380 | 0 |

Stores conflict zero times below 1024² and a few thousand times at 1024², on an access pattern
that is identical at all three sizes, with a figure that moves between runs — 4172, 4380, 4566.
**This is not explained here.** It is small, it is absent at smaller sizes, it does not scale
with anything the kernel does, and it is not what the skew is about.

> **[UPDATE, ADR-0018 step 3 — now caused]** The same phenomenon appears in the tiled
> contraction: 0 at 128 and below, ~5,200 at 256, ~67,700 at 512, skew irrelevant, occupancy
> ruled out. Split metrics there show the residue is entirely `mem_shared_op_st`, equals
> `type_arbitration`, and sits on top of a conflict-free address pattern (store wavefronts
> track the arbitration count; loads stay at 1 wavefront per instruction). Same reading fits
> this transpose's few-thousand `op_st` at `1024²`: not a skew miss, an L1TEX client-arbitration
> count the hardware bank-conflict counter includes. See ADR-0018 "Cause of the falsified zero".

What the claim should have said, and what the measurement supports, is narrower:

> The skew changes the **load** conflicts and leaves the store conflicts alone.

That holds at every size: 0 against 1,017,813 on loads, and a difference within its own
run-to-run noise on stores. Predicting an absolute zero where "unchanged" was the real claim is
the same error as writing 36.00 for the untiled control — reaching past what the reasoning
supported. Two for two in one ADR, both caught by measuring.

### Two ways this can fail that are worth naming now

The skew could be wrong for a tile width other than 32, in which case the conflict counter says
so and the derivation is wrong rather than the idea.

The L2 figure could stop short of 8 because the boundary guards break coalescing on the last
tile of a ragged matrix. That would show as a size-dependent excess, and the extents chosen
divide exactly so that it cannot hide in this measurement — with 4095 measured separately if it
appears.

## Build sequence

| step | | testable on its own |
|---|---|---|
| 1 | `tile` syntax, the staged stream path, and both in the IR | every existing example is unchanged; a tiled source parses and refuses what it must |
| 2 | derived shared layout and padding, in `Cost` and the manifest | `shared_bytes` is 32 x 33 x 4 and the three bindings pass it |
| 3 | the four-phase PTX | **done**, below |
| 4 | the bus model over a staged stream | **done**: derived 8 bytes per element, and the report shows both |
| 5a | `--ncu`, traffic | **done**, below: 8.00 measured against 8.00 derived |
| 5b | `--ncu`, the skew counterfactual | **done**, below: 0 against 1,017,813 |

## What bit-exactness does not prove here, and what does

The first draft of this ADR said step 3 was done when the tiled transpose came out bit-exact
against the host. **That sentence is true before the emitter is written.** A transpose has no
accumulator and no cross-element dependence, so its result is the same under every schedule:
untiled, tiled, tiled with the skew wrong, tiled with a barrier missing and no race that
particular run. A check whose failing state is unreachable is not a check, and this is the
second time that trap has come up in this project — the first was comparing infinity with
infinity and calling it verified.

The reflex was to make the host evaluator model shared memory so the check would have something
to catch. It would not have. The comparison is on the output buffer, and the output buffer is
the same either way; modelling shared memory in the evaluator adds a second place to make the
same index mistake and catches nothing the first place missed.

### Where the evaluator does model the schedule, and why

The rule this settles, which was implicit and is now written down:

> **The evaluator models the schedule exactly where the schedule changes the values.**

For a reduction it must: floating-point addition is not associative, there *is* no
order-independent specification of a float sum, and `tree_reduce` has reproduced the device's
tree step for step since ADR-0011. For a transpose it must not: nothing about staging changes a
value, so mirroring it is mirror-bug risk bought with no coverage.

That is why the shared-memory model does not arrive with this ADR. It arrives with the first
tiled **reduction**, where the tile geometry changes the rounding, derived from the tile in the
IR rather than copied from the emitter, and tested on properties that need no device: that the
skewed index function is injective over the tile, and that two elements the model puts in one
bank without the skew are in different banks with it.

### The three oracles, and what each can refuse

| claim | what can falsify it | what cannot |
|---|---|---|
| it computes a transpose | host against device on `b` | the schedule, the skew, a barrier |
| it moves 8 bytes per element, not 36 | `lts__t_bytes` at 4096², against the untiled kernel | the evaluator |
| the skew removed the conflicts | `l1tex__data_bank_conflicts_pipe_lsu.sum` | anything on the host |
| it synchronises | `compute-sanitizer --tool racecheck` | bit-exactness, except by luck |
| it staged at all | launching with `sharedMemBytes = 0` and getting a different answer | `cuFuncGetAttribute` |

The last row is worth explaining, because the obvious check does not work.
`cuFuncGetAttribute(CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES)` reports the **static** shared memory a
module declares. Measured on the existing reduction, which uses 1024 bytes of shared memory and
works: it reports **0**, because `.extern .shared` sizes the array at launch. The attribute would
refuse a correct kernel. What does work is the negative control: a kernel that produces the right
answer with no shared memory is a kernel that is not using shared memory.

[KNOWN LIMIT] Reading past the end of a zero-sized `.extern .shared` array is undefined, so
"the answer differs" is expected rather than guaranteed. The check is a signal, not a proof, and
is written down as one.

### `racecheck`, validated in both directions

A missing `bar.sync` is caught by bit-exactness only when the race happens to corrupt the output
on that run, which is worse than not catching it: it is a test that passes intermittently.
`compute-sanitizer --tool racecheck` detects the hazard itself.

The instrument was checked before being trusted, the way the machine file's ridge should have
been and was not for fifteen ADRs. On the existing reduction it reports **0 hazards**. With one
`bar.sync` deleted from the emitter it reports **130,816 hazards, 1 error** — and, that run,
bit-exactness also failed, on 137 of 65536 elements, which is exactly the coin-flip being
described. The barrier was restored and the tree is clean again.

An instrument that has only ever said "clean" has not been verified.

## Step 3 as it came out

```
kernel transpose_tiled(rows: u32, cols: u32, a: [f32; rows, cols], b: [f32; cols, rows])
    space i, j : rows, cols
    tile 32, 32
    stream a : dram -> smem -> reg
    stream b : dram -> reg, drain
    at reg:
        b[j, i] = a[i, j]
```

Byte for byte the body of `examples/transpose.lyth`. Two declared lines and a stream path are
the whole difference.

| oracle | result |
|---|---|
| bit-exact, 1024², 4095², 4097², 33x17, 31x129, 1024x33, 1x1 | all exact |
| `compute-sanitizer --tool racecheck` | 0 hazards |
| `compute-sanitizer --tool memcheck`, ragged 4095 x 257 | 0 errors |
| `bar.sync` in the emitted PTX | 2 |
| skewed stride in the emitted PTX | twice, transposed: `ty*33+tx` and `tx*33+ty` |
| predicated branches in the body | 1, the loop exit on the tile index |
| launched with `--shared-bytes 0` | `ILLEGAL_ADDRESS` |

The non-divisible extents were the first shapes run, not a regression pass afterwards. A guard
that is wrong in the generous direction gives correct results on every multiple of 32, and 1024²
would have said nothing.

The bypass control came back stronger than predicted. The ADR expected "a different answer",
with a [KNOWN LIMIT] that reading past a zero-length `.extern .shared` array is undefined and so
a match would prove nothing. On this device it faults outright, which is a harder signal than the
one that was pre-registered — recorded as what happened rather than upgraded into a guarantee,
because the undefined behaviour is still undefined.

### What the emitter had to get right, and how it is held

Every trap listed in advance turned out to be a real constraint on the shape of the code, and
three of them are now structural assertions that need no GPU.

**Two barriers.** The loop is grid-stride over tiles, so tile `T + gridDim`'s load races tile
`T`'s read. The PTX contains exactly two `bar.sync`, asserted.

**No branch around a barrier.** A thread that skipped one while its block reached it would
deadlock the block or abort the launch, so the loop bound is the **tile** index — uniform across
the block — and the boundary guards predicate the global accesses instead of branching. The test
asserts exactly one predicated branch in the body and that every global access is guarded.

**The skew in both phases.** A store on the padded stride and a load on the unpadded one
compiles, runs and produces correct bits while conflicting on every read. The test finds both
`mad` instructions carrying the derived 33 and asserts their operands are transposes of each
other.

**No second fma rule.** The mnemonic table moved into one function the flat and tiled bodies
share, rather than being copied.

## Two claims that needed a counterfactual, and one that needed normalising

Three readings of the table above found the same hole twice, and it is worth fixing before the
emitter exists rather than after.

### `bank_conflicts = 0` does not show the skew did anything

That number is also zero if the kernel never used shared memory, if the emitter ignored the skew
and got lucky with the access pattern, or if the counter does not cover the instruction. It says
the conflicts are absent; the claim is that **the skew removed them**, which is a claim about
cause.

So the fixture comes in two: the same body and the same tile, once with the derived skew and
once with the skew forced to zero.

| | predicted |
|---|---|
| skew derived (stride 33) | 0 conflicts |
| skew forced to 0 (stride 32) | **greater than zero** |

If both are zero the skew is a no-op and this ADR is wrong about why it works. That is the
falsifiable form.

### The bypass check needs a deterministic half

Launching with `sharedMemBytes = 0` and getting a different answer catches an emitter that
declares shared memory and does not use it. It cannot prove the converse: reading past a
zero-length `.extern .shared` array is undefined, so "the answer matched" means the undefined
behaviour was not observed, not that there is no staging.

The deterministic half is structural and needs no device: the emitted PTX contains
`.extern .shared`, and it contains exactly the number of `bar.sync` the IR says it should. The
PTX cannot lie about what it contains; the launch check catches the case where what it contains
is unused. Two layers, neither sufficient alone.

### The traffic comparison has to be like for like

`8 bytes per element against 36` compares two kernels. They must be the same `rows` and `cols`,
the same counter divided by the same element count, and measured the same way — otherwise it is
two different measurements with a ratio taken between them. Fixed here because it is the central
claim of the stage: **4096 x 4096, `lts__t_bytes.sum` divided by `rows * cols`, tiled against the
same body untiled.**

## Where the emitter will get it wrong, listed before it does

In order of how easy each is to write and not notice.

**Edge tiles.** Until now an index was a permutation of the space variables with no offsets. A
tile introduces offsets — `i = tile_row * 32 + ty` — and a guard, and a guard that is wrong in
the generous direction produces correct results whenever the extents are multiples of 32. **The
tests use 4095 and 4097**, not 4096, or a broken edge passes green. This is the one failure the
counters cannot see and bit-exactness can: the only place in this ADR where the host oracle is
the sharp instrument.

**One barrier where two are needed.** The loop is grid-stride over *tiles*, so a block handles
tile `T`, then `T + gridDim`. Phase 2 of the second iteration writes the shared tile that phase 4
of the first is still reading. That needs a barrier after the reads as well as after the writes:
**two `bar.sync` per tile iteration, not one.** The count is derivable from the IR and is
asserted without a GPU, and `racecheck` — already calibrated at 130,816 hazards for exactly this
shape of mistake — is what catches it if the derivation is also wrong.

**The skew applied to one phase only.** If the store uses the padded stride and the transposed
load does not, the kernel compiles, runs, and produces correct bits while conflicting on every
read. Nothing but the counterfactual fixture above separates that from a correct emitter.

**A second fma rule.** The contraction of `a * b + c` is tested and lives in one place. The tile
does not get its own.

## What is deliberately not here

No matmul. A tiled transpose stages one buffer and accumulates nothing; a matmul stages two and
carries a running sum across tiles, which needs a loop over the reduction dimension that this
language has no way to write. That is the next ADR and it should not be smuggled into this one.

No automatic tiling. The tile is declared, like every other movement in this language. A
compiler that chose it would be searching a schedule space, which is what Halide and TVM do well
and what this project has no claim to do at all.
