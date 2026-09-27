# ADR-0028 — The declared split: the interior qubit, without the copy

**Status:** Proposed
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
