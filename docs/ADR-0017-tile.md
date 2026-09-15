# ADR-0017 — The tile, and the first cost that depends on the schedule

**Status:** Proposed
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

## The falsification, pre-registered

On a 4096 by 4096 transpose, against the untiled kernel measured in ADR-0015:

| | untiled, measured | tiled, predicted |
|---|---|---|
| `lts__t_bytes.sum` per element | 39.02 | **8.00 to 8.02**, the same as `copy2d` |
| `l1tex__data_bank_conflicts_pipe_lsu.sum` | n/a | **0** |

Both get published whichever way they come back. Two ways this can fail that are worth naming in
advance: the skew could be wrong for a tile width other than 32, in which case the conflict
counter says so and the derivation is wrong rather than the idea; and the L2 figure could stop
short of 8 because the edge guards break coalescing on the last tile of a ragged matrix, which
would show as a size-dependent excess and is testable by picking extents that divide.

## Build sequence

| step | | testable on its own |
|---|---|---|
| 1 | `tile` syntax, the staged stream path, and both in the IR | every existing example is unchanged; a tiled source parses and refuses what it must |
| 2 | derived shared layout and padding, in `Cost` and the manifest | `shared_bytes` is 32 x 33 x 4 and the three bindings pass it |
| 3 | the four-phase PTX | the transpose is bit-exact **and** clean under `racecheck`, and wrong when launched with no shared memory |
| 4 | the bus model over a staged stream | derived bus cost falls to 8 bytes per element |
| 5 | `--ncu` | the table above, either way |

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
