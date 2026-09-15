# ADR-0018 — Contraction, where the cost model stops being a number

**Status:** Proposed — steps 1, 2 and 3 built
**Date:** 2026-09-16
**Depends on:** ADR-0000 (why), ADR-0017 (the tile), ADR-0015 (shape)

## The transpose was the mechanism; this is the theorem

ADR-0017 showed that a declared schedule changes a derived cost: the same body, one `tile` line,
36 bytes per element becoming 8, measured exact. That is the machinery working.

It is not yet the claim ADR-0000 makes. A transpose has no arithmetic, so its intensity is zero
under every schedule, and the 8 is recoverable by hand once you know the emitter absorbs the
permutation. Nothing about it needed a theorem.

A contraction does. `C[i,j] = sum_p A[i,p] * B[p,j]` over a `T`-by-`T` tile:

* each block loads `2T²` elements per step and takes `K/T` steps, so `2TK` elements per block;
* there are `MN/T²` blocks, so `2KMN/T` elements move for `MN` outputs;
* that is `2K/T` elements read per output, `+1` written, and `2K` flops.

```
intensity = 2K / (4 * (2K/T + 1))  ->  T/4 as K grows
```

**The intensity is a function of the tile.** Not of the body, not of the buffers, not of the
problem: of a number the author declared. Doubling `T` doubles it. That is the quantity
ADR-0000 says determines performance, derived from a schedule, and it is the first time in this
language that nobody could work it out by inspection.

It is also the Hong-Kung bound becoming operational rather than cited. The bound says
`Ω(n³/√M)` words must move for a fast memory of size `M`; the tiled schedule moves `2n³/T` and
`T` is bounded by what fits in `M`, so choosing the largest legal tile is what walks toward the
proof. The compiler derives the traffic and can say how far from the horizon a declared tile
sits.

## The decision that makes this different from every ADR before it

Every derived cost so far has been a **number**. `bytes_per_element` is 12 for saxpy, 8 for a
transpose, and the check compares a constant against a constant.

`2K/T + 1` is not a number. `K` is a launch extent, so a contraction's traffic per element is an
**expression in the extents**, and the cost model has to carry one.

This is not avoidable by picking a different formula — it is what reuse means. Traffic per
output falls as the contraction gets longer, because the tile is loaded once and used `T` times,
and any model that reports a constant is reporting a different kernel.

So `Cost` grows a symbolic form, and the existing split is the precedent: ADR-0015's sector
figure is already a static upper bound refined at launch when the extents are known. Same shape,
one level deeper.

* **Statically**, the compiler derives and reports the expression, and the asymptote `T/4`.
* **At launch**, with `K` in hand, it evaluates it exactly and checks the declaration against
  that.
* `intensity` in the source is checked against the asymptote within a tolerance, because a
  source constant cannot be a function of a runtime extent. The exact figure is a launch-time
  report, not a compile-time refusal.

That asymmetry is honest and it is a limit: **a contraction's declared intensity is a weaker
claim than an elementwise kernel's.** Written here rather than discovered.

## Syntax

```
kernel matmul(m: u32, n: u32, k: u32,
              a: [f32; m, k], b: [f32; k, n], c: [f32; m, n])
    space i, j : m, n
    contract sum p : k
    tile 32, 32
    intensity 8.0

    stream a : dram -> smem -> reg
    stream b : dram -> smem -> reg
    stream c : dram -> reg, drain

    at reg:
        c[i, j] = a[i, p] * b[p, j]
```

`contract sum p : k` declares an axis that is **walked and summed**, not one of the free indices
the space iterates. It reads like `reduce sum`, and deliberately: both say "this is combined",
and the word `sum` is there for the same reason it is there in a reduction — the operator is
written, not implied.

The two are different machines and the ADR should say so before someone conflates them.
`reduce` combines **across threads** through shared memory, and its tree order is part of the
contract because float addition is not associative. `contract` combines **within one thread**,
sequentially over `p`, in a register. The order is the loop order and there is nothing to
choose.

`c[i, j] = ...` assigns once per output, not once per `p`. The accumulation is what `contract`
declares; the body states the term being accumulated. An `+=` would put the schedule in the
body, which is the thing this language exists not to do.

## What the emitter has to grow

Four things, and the first three are new shapes rather than new ideas.

**Two staged buffers.** ADR-0017 refused more than one on purpose. `Plan::of` lifts to two and
the shared layout holds two skewed tiles — `2 * T * (T+1) * 4` bytes, 8448 at `T = 32`.

**A loop over the contracted axis, inside the tile loop.** Load both tiles, barrier, accumulate
`T` products into a register, barrier, advance `p`. Two barriers per step, for the reason
ADR-0017 measured: the next step's load races this step's read.

**A register accumulator that survives the loop**, initialised to the operator's identity, which
`ReduceOp::identity` already provides.

**Boundary guards on three axes**, not two. `m`, `n` and `k` may each be ragged, and a guard
that is wrong in the generous direction is correct on every multiple of 32. **The first shapes
run are non-divisible**, as in ADR-0017: `k` especially, since a partial final step reads tile
elements that were never loaded unless the staging zero-fills — which it does, and that is why
the identity matters.

## Pre-registered, per the rule ADR-0000 now carries

Written as what the derivation implies, not as what would be tidy.

**Claim 1 — intensity tracks the tile and nothing else.** At `m = n = k = 1024`, the derived
intensity `2K / (4(2K/T + 1))` is **3.97 at `tile 16, 16`** and **7.88 at `tile 32, 32`**: the
same body, the same buffers, the same extents, one number changed in a declaration. Each lands
within 2% of the asymptote `T/4`, below it because the write of `C` is in the denominator and
does not amortise.

**Claim 2 — the traffic is what the model says.** `lts__t_bytes.sum / (m*n)` at
`m = n = k = 1024`, `T = 32`, against a derived `4 * (2K/T + 1)` = **260 bytes per output**. The
tolerance is the one ADR-0017 earned rather than a round number: the untiled transpose overshot
its sector model by read-for-ownership, and a matmul writes `C` once and coalesced, so the same
mechanism should not appear. **If measured exceeds derived by more than 2%, the excess is the
result and gets its own investigation**, not a widened tolerance.

**Claim 3 — the language can say why it cannot reach the ridge, and the reason is the block.**
A tile of `T` is `T²` threads under one-thread-per-element, and this device reports
`MAX_THREADS_PER_BLOCK = 1024`, so `T ≤ 32` and the intensity ceiling is `8 flop/byte` against a
ridge of `42.9`. Reaching the ridge needs `T ≈ 172`, which is 29,584 threads per block — 28x the
cap. Shared memory would also refuse it (`238 KB` against `101,376 B` opt-in), but **the thread
cap binds first**, which is worth stating because the obvious guess is the memory.

So: a tiled matmul in this language is memory-bound on this machine and cannot be otherwise, and
the compiler should derive that from the machine file rather than the reader inferring it. The
way out is thread coarsening — one thread computing several outputs — and that is the next ADR,
not this one.

## What is deliberately not here

No thread coarsening, no double buffering, no vectorised loads, no tensor cores, no rank above
2, no contraction over more than one axis. This will not be fast. It is the first kernel in this
language whose cost is a theorem rather than a count, and that is the whole of what it claims.


## Step 1, as built

`examples/matmul.lyth` is checked in, parses, resolves, and **does not compile**. The last
thing `lower` does for a contraction is refuse:

```
examples/matmul.lyth:28:5: `matmul` parses and resolves, but its cost is not derived yet.
A contraction moves `2K/T + 1` elements per output -- an expression in a launch extent, not a
constant -- and this compiler will not publish a constant in its place. ADR-0018 step 2.
```

That refusal is the point of the step boundary. `Cost` carries constants, and the constant part
of a matmul's traffic is 4 bytes per output — the write of `C`. Emitting it would be wrong by a
factor of `K` and would look exactly like every other number this compiler publishes. So the
front end accepts the syntax and declines to cost it, and a test pins the refusal so that the
checks below are reachable rather than shadowed by it.

### Nine refusals, and why each is a refusal rather than a warning

Every one of these is a source that would otherwise produce a kernel that runs.

| written | refused because |
|---|---|
| `contract` with no `space` | the contracted axis is the one the space does not iterate; with no free indices there is nothing to contract against |
| `contract` and `reduce` in one kernel | different machines — across threads through shared memory, against inside one thread in a register |
| `contract sum i : k` under `space i, j` | free and contracted are the two things an axis can be, and this asks for both |
| `contract sum p : m` under `space i, j : m, n` | one extent cannot be both walked-and-kept and walked-and-combined |
| `contract sum n : k` where `n` is a parameter | an index variable is not a value the caller passes |
| `contract sum p : depth`, no such parameter | the contracted extent is a length the caller passes |
| `contract sum p : k` with no buffer at `p` | the declaration multiplies derived traffic by `K` while the body reads each element once |
| `c[i,j] = a[i,p] + b[i,j]` | one operand: no reuse, so `2K/T` is not its traffic. Reducing a buffer along an axis is a different kernel |
| `c[i,p] = ...` | the target has one value per output; writing at `p` stores each term in turn and keeps the last |

The last one is the one worth reading twice. It compiles under any language that lets you write
`+=`, runs, returns numbers, and computes the final term of the sum.

### Two things decided here that the proposal did not settle

**All three operators are accepted, not only `sum`.** `ReduceOp::combine` and `identity` are
already defined for `max` and `min`, and a contraction is sequential in a register: there is no
tree, so not even the ordering question a `reduce` has — it is the *easier* case. Refusing them
would have been a claim nobody derived. What they still lack is a measurement, which step 3
owes them along with `sum`.

**The order of the checks is part of the design.** The declaration-level refusals run *before*
the body and the body-level ones after. Written the other way round, `contract sum i : k` under
`space i, j` reported "`c` is written at `i`, the contracted axis" — true, and not the mistake
the author made. Found by a test asserting the message, not the failure.

### One thing the language grew, because the matmul needed it

A kernel signature may now span lines. `matmul` takes three extents and three buffers and does
not fit on one, and a layout-sensitive language has to say what a newline inside brackets means.
It means nothing: no `Newline`, no `Indent`, no `Dedent` while a parenthesis is open, and the
next line's leading spaces are ordinary whitespace. The alternative was a continuation
character, which is a second way to write one thing.

`crates/lyth-lang/tests/wrapped_signature.rs` asserts the two spellings lower to the **same**
IR — not to two IRs that both work — with parameter spans flattened, because a parameter on
line 4 should carry line 4 and an error about it should point there. An unclosed `(` is now
reported as one; without that check the rest of the file is swallowed as a single logical line
and the error surfaces somewhere unrelated, which is how the CRLF bug used to present.


## Step 2, as built

`Cost` is no longer a number. `lyth build examples/matmul.lyth`:

```
kernel matmul on machine sm_120
  derived  8.0000 flop/byte asymptotic  (2 * k flop / 0.25 * k + 4 byte per element)
  payload  0.25 * k read + 4 written, at dram
  sectors  0.25 * k read + 4 written at L1->L2  (coalescence 1.000)
  shared   8 * k read + 0.25 * k written per element, 8448 B per block
  declared 8 — matches
  ridge    42.9 flop/byte — memory-bound
error[codegen]: kernel `matmul` contracts over `k` and there is no emitter for a contraction yet.
```

At `k = 4096`: 1028 bytes per output, intensity 7.9689, against a limit of `T/4 = 8`.

### The expression is a result, never an input

The one line worth defending. It would be short to emit `2K/T + 1` on seeing a `contract` and a
`tile` together, and it would be right for the schedule this ADR describes and wrong for every
other schedule the language can already express — in the same confident shape as the constant
step 1 refused to publish.

So each stream is asked what it costs and the expression is the sum. A stream whose index names
the contracted axis is read `K / reuse` times per output; everything else is read once.

**Reuse is earned by staging, not by tiling.** `a[i, p]` is the same element for every `j`, so
the `tile[j]` threads of a tile column want it once between them — but only if it is staged.
Left in `dram -> reg`, each of those threads issues its own load and the reuse is 1: the tile
says the threads exist, shared memory is what makes them share. ADR-0017 drew the same line for
coalescence.

Four schedules, one rule, no special cases:

| schedule | derived | |
|---|---|---|
| both staged, `tile 32, 32` | `0.25 * k + 4` | the intended one |
| `a` staged, `b` not | `4.125 * k + 4` | `b` has no reuse; wrong by 16x if pattern-matched |
| neither staged, same tile | `8 * k + 4` | the tile alone earns nothing |
| no tile at all | `8 * k + 4` | **the untiled control, which falls out rather than being written** |
| `tile 16, 64` | `0.3125 * k + 4` | `a` reuses 64, `b` reuses 16 |

The rectangular tile is there because a square one makes the two reuse factors the same number,
which is exactly how a wrong rule survives its first test.

### `intensity asymptotic`, and a tenth and eleventh refusal

A contracted kernel's exact intensity is a function of a launch extent and a source constant is
not one, so a contracted kernel may declare **the limit** and has to say that is what it is
declaring:

```
intensity asymptotic 8.0
```

A bare `intensity 8.0` on a contraction is refused, and `intensity asymptotic` on a kernel whose
cost is constant is refused too — the same number wearing a weaker claim is a claim nobody
derived. This is ADR-0000's pre-registration rule moved out of the ADRs and into the source
language.

### The constant form is unavailable, not merely discouraged

`flops_per_element()` and `bytes_per_element()` return `Option`, and both are `None` for a
contracted kernel. The constant part of a matmul is **0 flops and 4 bytes** — two numbers that
would print without complaint and are wrong by a factor of `k`. The type is what makes that
unreachable; the compiler named all fourteen call sites, which is a better review than reading
for them.

The same accounting mistake was sitting in the sector model and printed as
`coalescence 0.333` — a per-output numerator over a per-step denominator, for a kernel whose
every access is absorbed into shared memory. It reads like a finding. It was arithmetic. Now
1.000, which is what ADR-0017's absorption rule claims.

### Three deviations from what the reviewers proposed, and why

**Affine, not a general expression tree.** One reviewer asked for a small AST
(`Const | Extent | Tile | + * /`). The value is affine in one extent, and that is closed for
what this language expresses: tiling the contracted axis as well would add another term in `k`
and stay affine, and only a *second* contracted axis gives a product — which v1 refuses. The
danger that reviewer named is real and it does not live in the datatype; it lives in the
deriver, and the four-schedule table above is what holds it.

**No canonicalisation.** Another asked for a normal form so two spellings of one expression do
not mismatch. Nothing ever compares two expressions: a source constant is compared against the
asymptote, and both are `f64`. Building a normal form nobody needs is the general algebra system
the first reviewer correctly warned against. If a source is ever allowed to declare an
expression, that is an ADR and not a detail of this one.

**The manifest carries the form; the bindings do not project it yet.** A generated binding with
a `derived_intensity(k)` function and no kernel behind it is a contract with nothing to check
it against. `Manifest` carries `symbolic` — the expression, its coefficients, the asymptote, and
**per-stream provenance** so a reader can check `0.25` against `4/32` rather than take it — and
the three generators emit the contracted contract line. The evaluator functions land in step 3,
with the PTX.

### One thing measured rather than assumed

A reviewer proposed the gap "`tile` only over the contracted axis — that is not `2K/T+1`, so
refuse it". It is unwritable: `tile` takes one dimension per **space** variable and the
contracted axis is not one, so `tile 32, 32, 8` is refused for rank. Asserted with a test rather
than left to construction, because "impossible by construction" is what stops being true when
someone adds a dimension.

### Pre-registered for step 5, corrected before measuring

**The 1028 bytes per output is a claim about L2**, against `lts__t_bytes`. **DRAM will be lower,
not equal** — a block's `A` panel is shared by every block in its row of `C` and the L2 serves
it. Writing "17.25 GB of DRAM" would have been the tidy sentence and false in the direction
ADR-0015 already measured.

And the ratio against a naive schedule is **not a result**. Naive is `2K` elements per output;
tiled is `2K/T`; the ratio is `T`, by construction, minus the `+1`. At 4096 that is 549.8 GB
against 17.25 GB — 31.88x, which is 32 with the write of `C` in it. Reporting that as a measured
speedup would be reporting the definition of a tile. The falsifiable claim is the 1028.

(A reviewer put the naive figure at ~824 GB and the ratio at 48x. It is 549.8 GB and 31.88x, for
the reason above — which is why the baseline had to be defined before it was quoted.)


## Step 3, as built

A matmul runs. `lyth run examples/matmul.lyth --set m=97 --set n=131 --set k=67`:

```
verify   BIT-EXACT against the IR evaluated on the host, 12707 elements
```

Bit-exact at every shape tried: `97x131x67`, `7x5x3`, `1x1x1`, `33x1x97`, `1x33x1`,
`31x31x256`, `256x256x31`, `32x32x32`, `128x96x160`. The first is the one that matters — 32
divides none of 97, 131 or 67, so the edge tile of `m`, of `n` and of `k` are all partial at
once, which is the case a guard that is wrong in the generous direction passes on every
multiple of 32.

### Five decisions, each a correctness question the transpose never had to answer

**The tile is square.** `A` stages a `T_i x T_p` tile and `B` a `T_p x T_j` one, both at one
element per thread of a `T_i * T_j` block, and that closes only when the three are equal. The
cost model derives a rectangular tile correctly (`tile 16, 64` gives `0.3125 * k + 4`); there is
no schedule for it here, and the refusal says which of the two is missing.

**The `k` tail is a loop bound, not a zero fill.** Zero-filling the last partial tile is correct
for `sum`, where a zero term changes nothing, and **wrong for `max`**, where it clamps the
result to be at least zero. The inner loop runs `min(T, k - p)` times instead: exact for every
operator, and less work. This is the ADR-0017 lesson again — the schedule has to be modelled
exactly where it changes the values.

**Both barriers sit outside every branch**, and the step loop's bound is `ceil(k/T)`, which is
block-uniform.

**The accumulator is not fused.** `acc = acc + a * b` emits a multiply and an add, not
`fma.rn.f32`. An fma rounds once where two instructions round twice, so fusing changes the
answer — for the better, and only after the host oracle agrees to change with it. ADR-0010's
decision to make again, not a free improvement to take here.

**Out-of-range threads still take part.** A thread whose output is past `m` or `n` loads zeros,
accumulates nonsense and is discarded by the store guard. Branching out would leave its
neighbours waiting at a barrier.

### The oracle is a separate walk

`eval` assumes every buffer's index is a permutation of the space variables, so one linear
element index addresses them all. `a[i, p]` names an axis the space does not iterate and no
linear index reaches it. The contraction oracle walks `(i, j)` and then `p` ascending — the
order the device walks, because float addition is not associative and the check is bit-exact
rather than tolerant.

### Three controls, because "it ran" is not "it is right"

| control | result |
|---|---|
| `compute-sanitizer --tool racecheck` | **0 hazards** |
| the same, with barrier 2 deleted | **2 hazards**, and 11182 of 12707 elements differ |
| `compute-sanitizer --tool memcheck`, ragged shape | **0 errors** |
| `--shared-bytes 0` (ADR-0017's bypass) | **ILLEGAL_ADDRESS** |

The second row is the one that makes the first mean anything: the tool was shown to fire before
it was believed. The last is what makes the derived `k/T` reuse a claim about this kernel — a
contraction that is correct without the shared memory it asked for never staged, and its
traffic expression would be fiction.

### A pre-registration, falsified, and two eliminations

Written before measuring: *no access in this schedule is a column read — `A` is a broadcast
within a warp and `B` is row-contiguous — so the derived skew changes nothing and
`l1tex__data_bank_conflicts_pipe_lsu.sum` is **0 with and without it**.*

| `m = n = k` | skewed | unskewed |
|---|---|---|
| 64 | 0 | 0 |
| 128 | 0 | 0 |
| 256 | 5,218 | 4,786 |
| 512 | 67,684 | 68,221 |

**Half right, and the half that was wrong was the absolute.** The skew is not the mechanism —
that part held at every size. But the conflicts are not zero past 128, and "0" was the tidier
sentence rather than the one the derivation supported. ADR-0000's rule, broken a third time by
the same move.

The calibration matters here: the same metric on an unskewed transpose at 512x512 reports
**253,952**, which is 8,192 warps times 31 — exactly the conflict a column read produces. So the
counter works and the numbers above are small, not absent.

Two mechanisms eliminated by measurement rather than by argument:

* **Not the skew.** Skewed and unskewed agree to within the run-to-run noise (which is itself a
  few percent, so the counter is not deterministic) at every size.
* **Not occupancy.** The zeros end where the grid passes 36 blocks on 36 SMs, which is the
  obvious suspect. Holding the work fixed at `512^3` and launching `--grid 36` instead of 256 —
  one block per SM against seven — gives **67,667 against 67,684**. It does not move.

What is left is an excess that scales with shared traffic and appears only past a threshold:
0.024% of shared accesses at 512, 0 at 128. It is the **same shape** as the unexplained finding
in ADR-0017 — a few thousand `op_st` conflicts at `1024^2` only, absent at smaller sizes,
unaffected by the skew. Two kernels, one phenomenon, and it is recorded as one open question
rather than explained twice.

The consequence for the language is concrete and not deferred: **`predicted_bank_conflicts: 0`
is falsified for this kernel.** It is derived from the skew being coprime with the bank count,
which remains true and is not what decides the outcome here.

### What step 3 does not claim

It is not fast, and no timing appears above. Intensity 8 against a ridge of 42.9 makes it
memory-bound by construction (claim 3), and thread coarsening is the way out and the next ADR.
Step 5 is the `--ncu` traffic measurement: 1028 bytes per output at `k = 4096`, **as an L2
claim**, with DRAM expected lower.

## Build sequence

| step | | testable on its own |
|---|---|---|
| 1 | `contract` in the AST, parser and IR, with its refusals | a matmul source parses; every existing example unchanged |
| 2 | the symbolic cost, and the launch-time evaluation | derived intensity moves with `T` and nothing else |
| 3 | two staged tiles and the `p` loop in the emitter | bit-exact against the host at non-divisible `m`, `n`, `k` |
| 4 | `racecheck`, and the structural assertions | 0 hazards, four barriers, two skewed strides |
| 5 | `--ncu` | claims 1 and 2, separately |
