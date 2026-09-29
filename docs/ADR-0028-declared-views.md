# ADR-0028 — The declared split: the interior qubit, without the copy

**Status:** Accepted (2026-09-29), for the QGPU rung of MTLB (ADR-0002). Frozen as proposed at
`docs/prereg/ADR-0028.prereg-frozen.md` before any split code; P1–P4 stand unedited
**Date:** 2026-09-18
**Depends on:** ADR-0015 (shape, and the sector model), ADR-0016 (callable), ADR-0026 (what LYTH
is), and the known limit recorded on the Hadamard commit (2026-09-17)

## The gap, stated exactly

`examples/hadamard.lyth` expresses a Hadamard gate on the **most significant qubit** of a state
vector. It can do that with no language feature at all, because of a fact about the layout: the
two amplitude sets a single-qubit gate mixes — the amplitudes whose q-th bit is 0 and those
whose q-th bit is 1 — are, for the top qubit, **two contiguous halves**. The caller passes them
as eight ordinary buffers (`p0r p0i p1r p1i q0r q0i q1r q1i`) and the kernel is elementwise.

For every other qubit the two sets **interleave in blocks of 2^q**, and no caller can hand them
over as contiguous buffers. The workarounds available today are both worse than the problem:

* **Pack the two sets into contiguous buffers before the gate, unpack after.** Two copies of a
  state vector per gate. This project exists because *"one careless copy is the whole budget"*
  (ADR-0026, the SUBSTRATE / QuBLAR row) — the workaround is the failure mode.
* **Stride the kernel's addressing.** Expressible neither today nor wanted: it puts partner
  arithmetic into index expressions, and the footprint stops being readable off the source —
  exactly what ADR-0015's restricted indexing exists to prevent, and what ADR-0026 refuses in
  advance for any feature that makes the byte count unknowable.

So the honest statement of the gap is **not** "the language needs flexible indexing". It is:
the kernel's *body* is already right; what the language cannot say is that two of its buffers
are **interleaved windows onto one buffer**. That is a declaration about layout, and layout
declarations are what this language does.

## The claim

> One new declaration — the **split** — says that two kernel buffers are the even and odd
> blocks of one real buffer, at a declared block width. The body does not change. The cost
> stays derivable, because a split view's traffic is the row-cost machinery ADR-0015 already
> built, at run length `w` instead of `cols`.

```
kernel hadamard_q(n: u32, w: u32, s: f32,
                  re: [f32; n], im: [f32; n],
                  qr: [f32; n], qi: [f32; n])
    intensity 0.25

    split re into p0r, p1r : blocks w
    split im into p0i, p1i : blocks w
    split qr into q0r, q1r : blocks w
    split qi into q0i, q1i : blocks w

    stream p0r : dram -> reg
    ...
    at reg:
        q0r = s * (p0r + p1r)        # the hadamard.lyth body, byte for byte
        ...
```

The decomposition is the state vector's own: linear slot `b·2w + j·w + p` is amplitude
`b·2^(q+1) + j·2^q + p` with `w = 2^q` — so `split … : blocks w` is qubit `q`, and the top
qubit is the special case `w = n/2`, at which point `hadamard_q` **is** `hadamard`.

Four things the split is, and one it is not:

1. It is a **name**, not a copy. A split view moves nothing; its elements are the real
   buffer's, at addresses derived from the declaration. The launch signature carries four base
   pointers instead of eight, and the split manifest travels in the generated Rust, C and
   Python exactly the way ADR-0017's skew padding already does — the caller never computes an
   address.
2. It restores, rather than breaks, the **one address per name** rule. The rule's reason
   (LANGUAGE.md) is that two index patterns for one buffer are two addresses per element. A
   view is a new name with exactly one pattern; the base buffer is never indexed directly in
   the body. The invariant generalises from buffers to names and loses nothing.
3. Its cost is the **row cost the model already derives**. ADR-0015's rule — a buffer is
   coalesced when its innermost index is the space's innermost variable, at
   `min(32, 4 · run)` bytes of sector per element — applies with `run = w`. No new machine
   field, no new level, no new derivation: the same formula at a different run length. For
   `w ≥ 8` (32 bytes) a split kernel is indistinguishable at the bus from the contiguous one.
4. It is **checked, not trusted**: at launch, `2·w` must divide `n`, and the views of one
   buffer must cover it exactly and pairwise disjointly. A wrong `w` is a refusal with the
   arithmetic printed, not a wrong answer.

It is not a general view mechanism. No offsets, no arithmetic on the split point, no
sub-ranges, no rank-3 spaces, no re-splitting a view. It splits a buffer into **two** at
declared block width — the smallest thing that closes the gap, and deliberately nothing more
(ADR-0026: anything whose cost is derivable may come later; nothing that unstated may come
now).

## Why the split and not the alternatives

| considered | why not |
|---|---|
| rank-3 space, body indexes `re[b, j, p]` | the body then needs both `j` values per output — two patterns per buffer, the invariant dies, and every elementwise kernel in the repository becomes rank-3 for nothing |
| offsets in index expressions (`a[p + w]`) | ADR-0015 designed them in; the shipped language refused them (the halo problem). The split makes them unnecessary for this domain: the partner is another *name*, not an offset |
| a strided-stream qualifier on `stream` | same expressiveness, worse shape: layout information scattered across stream declarations, and the launch signature has no clean way to carry it |
| pack/unpack in the caller | not a language answer; two copies of the state vector per gate is the budget |

## What stays true, what gets checked

Per ADR-0026's test — *after this, can the compiler still say what the kernel costs, and still
refuse the code that lies about it?* — the split passes because everything it adds is declared
before the run and checked at it:

* **Extents and cover** — `2·w | n`, views pairwise disjoint, exact cover. Launch-time refusals.
* **The flop count does not move.** The body is unchanged; `hadamard_q` derives the same
  `8 flop / 32 byte` per pair as `hadamard`, intensity 0.25, at every `w`. A split changes
  where bytes live, never how many the body asks for — the dram payload is invariant.
* **The sector figure becomes a function of `w`**, derived and printed per launch the way the
  `exact` line already is. This is the part that must be measured before it is claimed.

## Pre-registered, before any of it runs

State vector of `N = 2^23` amplitudes (16 MB re + 16 MB im, past the 34 MB L2 so DRAM is
real), `n = 2^22` pairs, machine file `sm_120`, `ncu` at `lts__t_bytes.sum`:

* **P1 — payload invariance.** Derived and measured DRAM traffic stays `32 B/pair` at
  `w ∈ {1, 2, 4, 8, 16, …}`. A layout declaration must not move the payload.
* **P2 — the bus step.** L1→L2 sector traffic per pair equals the payload (32 B) for
  `w ≥ 8`, and exceeds it monotonically as `w` halves below that. Worked bound at `w = 1`:
  each 4-byte load lands alone in a 32-byte sector at worst 8x — but the partner load and the
  drain reuse the same sectors, and whether the L1 serves them is exactly what the measurement
  decides. The ADR-0015 refinement already named this: for streaming kernels L1 misses
  everything; a split kernel is the first one whose two loads *deliberately* collide. **All
  outcomes will be published.**
* **P3 — time.** At `w ≥ 8` the split kernel times equal to contiguous `hadamard` at the same
  size, within run-to-run spread. At `w = 1` it is slower by no more than the P2 bus ratio
  implies against the machine file.
* **P4 — correctness.** Bit-exact against the host oracle at every `w`, on **both** back ends —
  PTX on `sm_120` and Unibit on the emulator — per the two-ISA discipline. The split changes
  addressing; the arithmetic must not know.

Nothing here has run. Per the house rule these predictions stay in the ADR unedited, and the
measurement lands beside them whichever way it goes.

## Build sequence

| step | | testable on its own |
|---|---|---|
| 1 | `split` in the parser and IR: two names, one base, declared width; refusals for re-splits, offsets, three-way splits | existing examples compile unchanged; `hadamard` untouched |
| 2 | launch checks: cover, disjointness, divisibility | wrong-`w` refusals name the arithmetic |
| 3 | cost: sector figure at run length `w`; the 12 `binding_level` kernels and `hadamard` must not move | regression green, clippy clean |
| 4 | codegen on both back ends + host oracle; split manifest into the generated Rust / C / Python signatures (ADR-0016) | `hadamard_q` bit-exact at `w = n/2` and at small `w` |
| 5 | the falsification sweep of P1–P4 | published either way |

Steps 1–3 change no measured cost and must not: the only derived number allowed to move is the
one step 5 measures.

## Scope, said plainly

**What this closes:** every single-qubit gate on every qubit of a state vector, with the body
of the contiguous kernel and no copy.

**What it does not open.** Two-qubit gates (CNOT and friends) mix four amplitude sets at two
strides — that is a nested split, a different declaration with its own cover algebra, and its
own ADR. In-place gates (output views aliasing input views) are refused until the host oracle
can evaluate them honestly, which it cannot: aliasing makes the element's old value part of a
neighbour's input. `sqrt`, `exp`, `tanh` remain where ADR-0027 left them. And no performance
number in this ADR is a claim — P1–P4 are predictions with their falsification attached, which
is what makes them claims *later*.

One sentence for ADR-0026's ledger: the split adds a declaration whose cost is derivable, so
it is growth. The day a split needs an uncountable feature to be useful, this ADR is the thing
to amend — not the refusal table.

## Progress notes (2026-09-29; the frozen predictions above are unedited)

- **Steps 1–3 landed** (LYTH `6bdceaf`, `9dd046a` and the step-3 commit): the `split` declaration
  and its refusals, the launch checks (`view_index`, `split_pairs`), and the sector figure.
  `hadamard` and every existing kernel are unchanged, and the suite grew from 325 to 339 tests.
- **P1 holds in the model:** `hadamard_q` derives 8 flop / 32 byte per pair, identical to
  `hadamard`, at every `w`.
- **A refinement of P2's arithmetic, stated before the measurement.** P2's worked bound at
  `w = 1` (8×) is one sector per element, and that is now the *static bound*. Counting the
  sectors a warp of 32 consecutive elements touches, walking `view_index` itself, gives the
  figure at launch: **1.0× at `w ≥ 8` and 2× below it, per view in isolation**. Every sector a
  view touches is half the other view's. P2's shape (equal to the payload at `w ≥ 8`, exceeding
  it as `w` halves below that) stands; the magnitude below 8 is predicted at 2×, not 8×. The
  partner view and the drain share those sectors, so what the bus actually shows is for step 5
  to say, and it is published either way.

- **Step 4 landed.** `hadamard_q` (a Hadamard on any qubit, eight views over `re, im, qr, qi`)
  is bit-exact against a dense reference that shares no code with the implementation, on all
  three machines: the host oracle and PTX on `sm_120` (20 launches: every qubit of a
  4096-amplitude register, the top qubit `w = n/2`, and widths that are not powers of two), and
  the Unibit emulator (the same 20). **P4 holds.** Both new tests were mutation-checked: a wrong
  `2w` in the PTX and a wrong run skip in the MTLB emission each make them fail.
- **MTLB below eight elements is one f32 per instruction.** `LQ` reads eight contiguous
  floats, so a view is whole registers only when `w` is a multiple of 8; for the low qubits
  (`w = 1, 2, 4`) the emission is `LW`/`SW` through lane 0 with the same packed `VF*` ops. It is
  bit-exact and it is 8x the instructions the cost model counted. The derived traffic and
  intensity hold; the ceiling ADR-0022 computes for this machine does **not** at those widths.
  `[KNOWN_LIMIT]`, in the emitter and here.
- **Deviation from the step-4 row: only the Rust binding.** The manifest carries the splits
  (`splits`, and `grid.pairs`), and it now marks a split base written or read from its views;
  reading the streams by the base name called the outputs read-only. The **C and Python
  generators refuse** a kernel that splits, with the reason, rather than emit a grid twice as
  large and no divisibility check: without that check a bad width writes past the buffer. The
  Rust binding refuses it before the launch, with the arithmetic, and a refused launch is shown
  to write nothing. Manifests of kernels that do not split are byte-identical to before.
- **`lyth run` takes a split.** `-n` is the buffer length, `--set w=<u32>` the width, and the
  kernel walks `n/2` pairs; a missing, fractional or non-dividing width is refused before any
  device is touched. It compares the views base (the buffer that holds the answer), and a
  mutation of the PTX fails it, so it does not report bit-exact over nothing.
- **Exploratory, not step 5.** One smoke run each at `N = 2^23`, `w = 8` and `w = 1`, `--time 50`,
  to see the harness work: both about 388 GB/s (93.6% of the 414.5 GB/s baseline), spread 76-78%.
  That spread is too wide to say anything with, the clocks are unlocked, and it is a single
  run: it decides nothing about P3. Step 5 fixes its repetitions and its order first.

## Step 5 -- the sweep, as measured (2026-09-29)

Protocol and analyzer frozen before the first launch (`docs/prereg/ADR-0028.step5-protocol.md`,
`tools/split_sweep.py`, hashes in `ADR-0028.step5.PREREG_SHA256.txt`); run at commit `ed73546`; every
raw number in `docs/prereg/ADR-0028.step5-results.json`; the verdicts below are the output of
`tools/split_sweep.py analyze`, unedited. RTX 5060 Ti, `sm_120`, 2^22 pairs, 128 MB touched, 3 `ncu`
launches and 7 timing processes per configuration, all 23 widths against the contiguous `hadamard`.

| | DRAM B/pair | L2 B/pair | L2 / control | ms | time / control | spread |
|---|---|---|---|---|---|---|
| control (`hadamard`) | 29.96 | 32.01 | | 0.3492 | | 0.6% |
| w = 1 | 29.89 | 34.12 | 1.066 | 0.3456 | 0.990 | 0.4% |
| w = 2 | 29.97 | 34.14 | 1.067 | 0.3456 | 0.990 | 0.1% |
| w = 4 | 29.90 | 34.11 | 1.066 | 0.3455 | 0.989 | 0.1% |
| w = 8 | 30.00 | 32.04 | 1.001 | 0.3461 | 0.991 | 0.3% |
| w = 64 | 29.93 | 32.01 | 1.000 | 0.3452 | 0.989 | 0.2% |
| w = 1024 | 30.01 | 32.01 | 1.000 | 0.3458 | 0.990 | 0.1% |
| w = 65536 | 29.91 | 32.01 | 1.000 | 0.3456 | 0.990 | 0.3% |
| w = 2^22 (top qubit) | 29.90 | 32.04 | 1.001 | 0.3456 | 0.990 | 0.7% |

The other 15 widths sit in the same band: L2 ratio 1.000-1.002, time ratio 0.986-0.996, DRAM 29.9-31.1
B/pair (the highest, 31.13 at w = 2^20, is the L2 keeping fewer writes; still within 5%).

| prediction | verdict |
|---|---|
| **P1**, invariance: DRAM bytes per pair within 5% of the control at every `w` | **PASS** |
| **P1 as stated**: the control at 32 B/pair | **NOT confirmed as stated**: the control itself moves 29.96 B/pair; the L2 keeps about 2 B/pair of the writes, ADR-0015's known limit |
| **P2 shape**: equal to the payload at `w >= 8`, above it at `w < 8`, non-increasing in `w` | **PASS**, and narrowly: the step below 8 is 6.6%, over a 5% tolerance |
| **P2 magnitude, refined 2x** (the progress note above) | **FAIL**: 1.066, not 2 |
| **P2 bound, frozen 8x** | **PASS** (1.067 at worst) |
| **P3**, time equal to the control at `w >= 8`, within the bus ratio below | **PASS**: every width, `w = 1` included, is 0.986-0.996 of the control |
| **P4**, bit-exact on PTX at this size | **PASS**: every one of the 24 configurations, 10 launches each |

What this says, and what it does not:

* **The refinement I wrote before the measurement is falsified.** I derived 2x at `w < 8` by walking
  `view_index` and counting the sectors a warp touches **per view in isolation**, and said the partner
  view and the drain share those sectors and that the measurement would decide. It decided: the bus
  carries 1.066x the payload, not 2x, so the L1 absorbs most of the collision the isolated count
  charged. The isolated figure is a bound on what one view alone would ask of the L2, and it
  over-predicts the bus by about 15x in its excess (6.6% against 100%). The frozen P2 text stands as
  written: it exceeds the payload below 8, on a plateau (w = 1, 2, 4 all at 1.066) and not as a
  staircase that keeps rising, which the "monotonically" reading was written to allow.
* **The mechanism is not shown.** `lts__t_bytes` does not split reads from writes, so "the L1 serves
  the partner view" is the hypothesis this is consistent with, not something measured. A
  `lts__t_sectors_op_read` / `_write` split would say which half carries the 2 B/pair; that is a new
  measurement and outside this protocol.
* **The extra 2 B/pair at the bus costs no time here.** Time is flat across all 23 widths, so at this
  size the kernel is bound by DRAM, which did not move. P3's bus-ratio clause was never tested hard:
  it needed a bus that binds, and this one did not. A case where it would (a smaller working set in
  L2, or a wider kernel) is not covered by this result.
* **The control is 1% slower than every split width**, a little over its own 0.6% spread and inside
  the 5% floor the protocol fixed. Not explained; the control has eight buffers and the split has
  four, which is the obvious difference and not a demonstrated cause.
* **The printed per-launch figure is now known to be an upper bound on this device.** `lyth check` and
  `lyth run` print, for `w < 8`, 64 B/pair (coalescence 0.500) with the caveat that each view is taken
  alone. That caveat stands and this is the measurement it was pointing at. Whether the printed figure
  should carry a machine-file L1 factor (ADR-0023) is open and not decided here.
* Not measured here: the Unibit back end's speed (its `w < 8` path is one f32 per instruction), any
  `N` other than 2^23, any device but this one.

One erratum, from the protocol and not an edit of the frozen text: ADR-0028 says "16 MB re + 16 MB
im"; at 2^23 amplitudes each buffer is 32 MB. The 128 MB working set is what the argument needed.
