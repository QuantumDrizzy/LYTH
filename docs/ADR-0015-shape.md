# ADR-0015 — Shape in the type, and the limit it exists to kill

**Status:** Proposed
**Date:** 2026-09-15
**Depends on:** ADR-0005 (intensity), ADR-0006 (kernel IR), ADR-0008 (layout × machine, scaffold)

## The thesis is currently exercised only where it is degenerate

ADR-0001 claims that arithmetic intensity is a type. Every kernel this compiler has ever
compiled is elementwise over one-dimensional `[f32]`, with exactly one `u32` bounding the index
space, so every kernel's traffic is one element in and one element out. Under those rules
`bytes_per_element` is a count of streams, and the derivation cannot be wrong.

That is the problem. **The intensity of anything interesting is a function of shape and
schedule, not of the source.** A naive matmul and a tiled one have identical semantics and
intensities separated by a factor of the tile width. A language whose types have no shapes can
only type the cases where intensity is constant — which are exactly the cases where nobody
computes it wrong by hand. The check is real and it has never had anything to catch.

Shape is not a missing convenience. It is the boundary of the claim.

## What this is really aimed at

`crates/lyth/src/main.rs` carries a limit written when the evidence schema was designed:

> [KNOWN LIMIT] The byte count is a lower bound: it counts the payload, not the 32-byte sector
> a scattered access pulls. v1 is elementwise and fully coalesced, so the two should agree
> here; that is the claim `--ncu` tests.

Read it again. The byte model is only correct **because the language cannot express an access
that would break it.** The limit is not being managed; it is being avoided by construction.

A transpose breaks it immediately. With both operands row-major, `b[i, j] = a[j, i]` reads `a`
down a column: consecutive threads touch addresses `cols * 4` bytes apart, so each 4-byte
element costs a 32-byte sector and the real traffic is **eight times the payload** on that side.
A compiler that reports the payload is not making a small error; it is reporting a number that
would place the kernel in the wrong regime and pass its own check while doing it.

So this ADR is not "add tensors". It is: **make the traffic model able to be wrong, then check
it against silicon.** That is the only way the refusal earns anything.

## Decision

Shape and layout enter the type. The iteration space is **declared, not inferred**, for the same
reason movement is declared: this language does not guess what a program does, it checks that
the program fits what was written.

```
kernel transpose(rows: u32, cols: u32, a: [f32; rows, cols], b: [f32; cols, rows])
    space i, j : rows, cols
    intensity 0.0

    stream a : dram -> reg
    stream b : dram -> reg, drain

    at reg:
        b[j, i] = a[i, j]
```

Four changes, in dependency order.

1. **Dimensions are `u32` parameters.** `[f32; n]` names its extent. The special rule that
   exactly one `u32` bounds the loop disappears: the bound comes from the shape, and a `u32`
   that no shape mentions is a scalar like any other. This alone changes no cost and is
   honestly cosmetic — it is the foundation the rest needs, not a result.

2. **Rank 2, with a declared iteration space.** `space i, j : rows, cols` names the loop
   variables outermost first and binds them to extents. The grid-stride loop flattens it; the
   flattening order is part of the contract, the way `tree_reduce`'s order already is.

3. **Affine indexing, restricted.** An index expression is a permutation of the space
   variables plus a constant offset: `a[i, j]`, `a[j, i]`, `a[i, j + 1]`. Nothing data-dependent,
   nothing multiplied by a variable. This is the largest set for which the footprint is exactly
   derivable without a polyhedral pass, and the restriction is stated so the next person knows
   where the wall is rather than discovering it.

4. **A sector-aware traffic model.** For each access, the derivation asks whether the fastest-
   varying index is the innermost space variable. If it is, the access is coalesced and costs
   the payload. If it is not, consecutive threads are `stride * 4` bytes apart and each element
   costs a full 32-byte sector. `LevelCost` already carries per-level read and write, so the
   shape of the model does not change — only the number going into it.

## What is deliberately not in this slice

No tiling, no shared-memory staging, no `tile` construct, no rank above 2, no layout other than
row-major-with-declared-extents, no f16, no data-dependent indexing. Matmul is still not
expressible and is not the target here: it needs staging, which needs a tile declaration, which
is its own ADR. **Transpose is the target** because it breaks the byte model with the smallest
possible language change.

## The falsification, pre-registered

The sector model says a strided read costs 32 bytes per element against 4 for a coalesced one,
so a transpose should measure **8x** the payload on its strided side and 1x on the other.

`--ncu` decides it, on `dram__bytes_op_read.sum`, against the derived count. Three outcomes and
all three get published:

* **Measured ≈ derived.** The model holds and the limit above is retired with evidence.
* **Measured below derived.** The sector model overstates, most likely because L2 absorbs
  partial sectors across warps. Then 32 bytes per element is a *ceiling*, not a count, and the
  cost model has to say which it is — a lower bound and an upper bound are different fields.
* **Measured above derived.** Something is pulling more than a sector per element and the model
  is missing a mechanism. That is the most interesting outcome and the least likely.

The number to beat is that this compiler currently reports `4` for both sides of a transpose and
is confident about it.

## Why this is the right next slice and not `matmul`

Matmul is the demonstration everyone wants and it needs three things this language does not
have: staging in shared memory, a tile declaration, and a footprint model that depends on the
tile. Building all three at once means the first measurement arrives after all of them work,
and there is no way to tell which one is wrong.

Transpose needs one thing — shape — and produces a number that is either 8x or is not. It is the
smallest change that can *fail*.

## Build sequence

| step | | testable on its own |
|---|---|---|
| 1 | `[f32; n]`, extents as `u32` params, one-dimensional | every existing example compiles unchanged in meaning |
| 2 | `space` declaration, rank 2, flattened grid-stride loop | a rank-2 copy is bit-exact against the host |
| 3 | indexed access with permutation and offset | transpose is bit-exact against the host |
| 4 | sector-aware traffic in `Cost` | transpose derives 8x on the strided side |
| 5 | `--ncu` on transpose | the falsification above, published either way |

Steps 1 to 3 change no cost and must not: if a derived intensity moves before step 4, something
was wrong before or is wrong now, and the existing per-example cost table in
`crates/lyth/tests/pipeline.rs` is what will say so.
