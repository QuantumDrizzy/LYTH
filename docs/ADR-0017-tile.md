# ADR-0017 — The tile, and the first cost that depends on the schedule

**Status:** Proposed
**Date:** 2026-09-16
**Depends on:** ADR-0015 (shape), ADR-0016 (callable), ADR-0011 (reductions)

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
| 3 | the four-phase PTX and the evaluator that matches it | the tiled transpose is bit-exact against the host |
| 4 | the bus model over a staged stream | derived bus cost falls to 8 bytes per element |
| 5 | `--ncu` | the table above, either way |

## What is deliberately not here

No matmul. A tiled transpose stages one buffer and accumulates nothing; a matmul stages two and
carries a running sum across tiles, which needs a loop over the reduction dimension that this
language has no way to write. That is the next ADR and it should not be smuggled into this one.

No automatic tiling. The tile is declared, like every other movement in this language. A
compiler that chose it would be searching a schedule space, which is what Halide and TVM do well
and what this project has no claim to do at all.
