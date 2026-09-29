# ADR-0029 — A split of a view: two-qubit gates, and circuits against an outside oracle

**Status:** Accepted (2026-09-30). Frozen as written at `docs/prereg/ADR-0029.prereg-frozen.md`
before any code for it; P1–P5 stand unedited from that moment
**Date:** 2026-09-30
**Depends on:** ADR-0028 (the split, `view_index`, its launch check and its measured sweep),
ADR-0015 (the sector model), ADR-0016 (callable), ADR-0026 (what LYTH is)

## The gap, stated exactly

ADR-0028 closed every **single**-qubit gate on every qubit, and `gate_q` (commit `d52a9dd`)
takes any 2x2 unitary. A state-vector simulator needs one more thing before it can run a circuit
worth the name: a gate that couples **two** qubits — CNOT, CZ, a controlled phase, a swap.

A two-qubit gate on qubits `(c, t)` mixes the **four** amplitudes of each quad: the indices that
agree everywhere except bits `c` and `t`. For the two top qubits the four sets are contiguous
quarters; for every other pair they interleave at **two** strides at once. ADR-0028 said so and
left it out: "that is a nested split, a different declaration with its own cover algebra, and its
own ADR". This is that ADR.

## The claim

> A two-qubit gate needs **no new address rule**. The four quad sets are what you get by
> splitting a *view*: split the state at the width of one qubit, then split each half at the
> width of the other. Element `k` of a leaf is `view_index` applied twice. Lifting ADR-0028's
> refusal of "a split of a view" — and nothing else — gives every two-qubit gate on every pair.

```
split re into h0, h1   : blocks wc      # bit c of the index: 0 -> h0, 1 -> h1
split h0 into p00, p01 : blocks wt      # of h0, bit t: 0 -> p00, 1 -> p01
split h1 into p10, p11 : blocks wt
```

**The inner width counts elements of its parent view, not of the base.** For `c > t` that is
`wc = 2^c`, `wt = 2^t`, the plain widths. For `c < t` the parent view has lost bit `c`, so the
inner width is `2^(t-1)`. The language does not know about qubits; it composes two maps, and the
caller (or the circuit runner) states the widths. Both orders are tested.

**One definition, extended.** Element `k` of a leaf at depth `d` is
`view_index(w_1, part_1, view_index(w_2, part_2, … view_index(w_d, part_d, k)))`, read from the
leaf upwards: `KernelIr::base_index`. The host oracle, both back ends, the launch check and the
sector figure all read it and nothing else.

**Cover.** A split of a view is an ADR-0028 split of a buffer of length `len / 2`: its launch check
is `2 * w_inner | len / 2`. A composition of bijections is a bijection, so the four leaves cover
the base exactly and disjointly when every level's check passes. No new algebra, one check per
level, each refused with its arithmetic.

## Rules (refusals are part of the design)

* Every **streamed** view is a leaf, and all leaves streamed by a kernel are at the **same depth**:
  the kernel walks `len / 2^depth` elements, one quad (or pair) per thread. Mixing depths would
  walk two different counts. Refused with both depths named.
* A view that is split is not itself streamed (its leaves are). Refused, like ADR-0028's
  "base streamed".
* Depth is **at most 2** in this ADR. The IR composes any depth, but only depth 2 is tested and
  claimed; a three-qubit gate is its own amendment with its own predictions.
* Everything ADR-0028 refuses stays refused: in-place gates (a circuit ping-pongs two state
  buffers), `space`/`tile`/`coarsen`/`reduce`/`contract` beside a split, three-way splits.
* **Unibit (MTLB)** walks a depth-2 split as a three-level loop nest. It requires, per level,
  that the smaller width divides the larger in the way a loop nest needs (`2 w_t | w_c` when the
  inner run is shorter, `w_c | w_t` when it is longer) — always true for qubit widths — and
  refuses anything else with the arithmetic, rather than computing a division per element.

## The kernels

* `cu_q.lyth` — a controlled-U: the control-0 half is copied (both of its leaves), `U` (eight
  scalars, `gate_q`'s body) acts on the control-1 half. CNOT is `U = X`, CZ `U = Z`, a controlled
  phase `U = diag(1, e^{iφ})`.
* `swap_q.lyth` — the swap of two qubits: `q01 = p10`, `q10 = p01`, the others copied.
* `cu.lyth` — the **control** for the measurement: the same body over sixteen contiguous buffers
  of `n/4`, no split. Nothing it does can depend on the split machinery.

## Pre-registered, before any of it runs

**Size and instruments** as ADR-0028 step 5: `N = 2^23` amplitudes (four buffers of 32 MB), 2^21
quads, RTX 5060 Ti `sm_120`, `ncu` at `lts__t_bytes.sum` and `dram__bytes_op_{read,write}.sum`,
3 launches per configuration (median), `lyth run --time 200` over 7 processes in shuffled rounds
(median of medians). Configurations: `cu_q` at `(c, t)` for `c, t ∈ {0, 1, 2, 3, 4, 8, 16, 22}`,
`c ≠ t` (56), against `cu.lyth`. `tol = 5%`, `eps = max(tol, control's run-to-run spread)`.
The same tooling rule: the tool and its protocol are hashed and committed before the first launch.

* **P1 — payload.** DRAM bytes per quad within `tol` of the control at every `(c, t)`. (Not "64
  B/quad" absolutely: ADR-0028 measured that the L2 keeps writes, 29.96 of 32 B/pair.)
* **P2 — the bus.** L2 bytes per quad over the control's: within `tol` of 1.0 whenever
  `min(c, t) ≥ 3` (every run of every leaf is at least one 32-byte sector), and at most **1.10**
  at every `(c, t)`. The prior is a measurement, stated as one: ADR-0028 saw 1.066 at a single
  narrow width, and two narrow widths might compound; 1.10 allows for a little of that and not
  for the isolated count. **And:** the derived per-view-isolated figure is at or above the
  measurement at every `(c, t)` — it is an upper bound, which is what ADR-0028 step 5 showed it to
  be and what `lyth run` will print it as.
* **P3 — time.** Within `eps` of the control at every `(c, t)`. This is predicted because the
  kernel stays DRAM-bound (ADR-0028: the bus excess cost nothing); if P2's excess ever binds, this
  is where it will show.
* **P4 — correctness, three machines.** `cu_q` (CNOT, CZ, a controlled phase, a controlled ZYZ
  unitary) and `swap_q` bit-exact against a dense quad-walking reference — no division, no
  `view_index` — on the host oracle and PTX at every tested `(n, c, t)` including both orders and
  widths that are not powers of two, and on the Unibit emulator at every qubit-width pair tested.
  And the physics check `gate_q` introduced: `U†U ψ = ψ`, norm kept, to `1e-6`.
* **P5 — circuits, against an outside oracle.** A circuit runs as a sequence of LYTH launches on
  the GPU (ping-pong buffers, `gate_q`, `cu_q`, `swap_q`), and is compared with Qiskit's
  `Statevector` (f64) on the same circuit:
  * GHZ on 12 qubits (H, then 11 CNOTs);
  * QFT on 10 qubits of a random product state (H, controlled phases, the final swaps);
  * a random circuit on 10 qubits, depth 20: a ZYZ gate on every qubit per layer, then CNOT or CZ
    on a random matching, seed fixed in the test.

  **Pass:** max `|amplitude difference| ≤ 1e-5` and `1 − |⟨ψ_lyth|ψ_qiskit⟩|² ≤ 1e-6` on all three.
  **Expected** (stated so it can be wrong): max error in the `1e-7 … 1e-6` range — f32 rounds at
  `6e-8` relative per operation and a few hundred gates walk that to around `1e-6` at worst. The
  measured error is published whatever it is. Little-endian qubit order, like Qiskit, so no
  reordering sits between the two.

Nothing here has run. Per the house rule these predictions stay unedited, and the measurements
land beside them whichever way they go.

## Build sequence

| step | | testable on its own |
|---|---|---|
| 1 | a split of a view in the IR: `base_index`, depth, the rules above; `split_pairs` becomes the walk at any depth, with one launch check per level | every ADR-0028 test unchanged; the new refusals name their arithmetic |
| 2 | cost: the sector figure walks `base_index` | ADR-0028's figures do not move |
| 3 | host oracle, PTX, Unibit, `lyth run`, manifest and Rust binding; `cu_q`, `swap_q`, `cu` | P4 |
| 4 | the circuit runner and the Qiskit oracle | P5 |
| 5 | the falsification sweep of P1–P3 | published either way |

## Scope, said plainly

**What this closes:** every two-qubit controlled-U and the swap, on every ordered pair of
qubits, with no copy of the state; and circuits built from them, checked against a simulator
nobody here wrote.

**What it does not.** A general 4x4 unitary (32 scalars: it would fit PTX and not the Unibit
register pool; it is the sum of what is here plus arithmetic, and not claimed); three-qubit gates
(depth 3); in-place gates; noise, measurement and mid-circuit feed-forward, which live in MTLB's
QPU rung and not in this compiler; and any claim of speed against cuStateVec, which ADR-0028's
research note already said is not the point — the kernels are DRAM-bound and so is everyone's.
