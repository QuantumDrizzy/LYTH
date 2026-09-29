# ADR-0030 — Fusion: many gates, one pass over memory, the same bits

**Status:** Accepted (2026-09-29). Frozen as written at `docs/prereg/ADR-0030.prereg-frozen.md`
before any code for it; Q1–Q5 stand unedited from that moment
**Date:** 2026-09-29
**Depends on:** ADR-0028 and ADR-0029 (the split, `base_index`, their measured sweeps and the P2
follow-up), ADR-0015 (the sector model), ADR-0022 (the level that binds)

## The gap, stated exactly

ADR-0028 and ADR-0029 measured it twice: a gate on a state vector is bound by DRAM. Every gate
reads the whole state and writes it back, and nothing else it does shows in the time. A circuit of
`G` gates is `G` passes over memory, and that is the whole cost.

The only way to make a circuit cheaper on this machine is to make **fewer passes**: apply several
gates while the amplitudes are in registers, and touch memory once. Every serious state-vector
simulator does this ("gate fusion"). They do it by multiplying the gates of a group into one dense
`2^k x 2^k` matrix, which changes the rounding: the fused circuit is close to the unfused one, not
equal to it.

The P2 follow-up of ADR-0029 found a second cost the model does not see: **the order of the
stores**. The same kernel with its drained leaves declared grouped by buffer moves 1.06x the
payload at the L2 instead of 1.84x. An author should not have to know that.

## The claim

> A fused pass is a LYTH kernel with a **deeper split** -- one level per qubit of the group, so each
> thread owns the `2^k` amplitudes the group mixes -- whose body is the gates' own fma chains,
> **one after the other, in registers**. No product matrix. In f32 the values between gates are the
> values the unfused circuit would have written to memory and read back, so **fusion changes the
> traffic and not one bit of the result**. That is the claim this ADR can be falsified on.

Nothing new is needed in the address rule: `base_index` already composes `view_index` to any depth.
What changes:

1. **Depth up to 5.** `MAX_SPLIT_DEPTH` goes from 2 to 5: a group of up to five qubits, 32 amplitudes
   per thread, 64 f32 in flight. Deeper is refused.
2. **Locals that a later statement reads are legal.** Today a local is legal only as a reduction's
   source, so that dead work is refused. A fused body chains gates through locals; a local that
   some later statement consumes is not dead work. A local nothing reads stays refused, with the
   same message (`dead-local.lyth` still fails).
3. **The emitter owns the store order.** Drained leaves are stored grouped by their buffer, in
   split-tree order, whatever order the source declares them in. Width-independent, so it holds for
   a kernel compiled once for every qubit.
4. **A fuser**, `crates/lyth-circuit`: a circuit (qubits; single-qubit `U`, controlled-`U`, swap),
   a greedy partition into groups of consecutive gates whose qubits number at most `k`, **in circuit
   order** (no commuting, so nothing reorders the arithmetic), and for each group a generated `.lyth`
   kernel with the matrices as literals. The generated source is ordinary LYTH: it is checked,
   costed and verified like any other kernel.
5. **Unibit** derives its loop nest from `base_index` numerically -- runs and hops read off the
   address sequence of the first leaf -- and **checks** the derived nest against `base_index` for
   every element before emitting. It has four loop counters, so a nest deeper than that is refused.

## Pre-registered, before any of it runs

Machine as before: RTX 5060 Ti `sm_120`, `fixtures/machine/sm_120.json`, Qiskit 2.2.3 as the outside
oracle.

* **Q1 — fusion changes no bit.** For the three ADR-0029 P5 circuits, and a random 20-qubit circuit
  of depth 10, the fused run at every `k ∈ {1, 2, 3, 4, 5}` equals the unfused run (one launch per
  gate, `gate_q`/`cu_q`/`swap_q`) **bit for bit**, on the GPU. On the host oracle for circuits of up
  to 12 qubits. On the Unibit emulator for fused groups whose nest it can walk, at 10 qubits.
* **Q2 — a fused pass costs one pass.** DRAM bytes per amplitude of a fused pass, measured with
  `ncu` at `2^23` amplitudes, within 5% of a single-gate pass (`gate_q`) at the same size, for groups
  of 1 to at least 20 gates. L2 bytes per amplitude at most 1.10x the single contiguous pass for
  every group whose lowest qubit is at least 3, **with the emitter-owned store order**.
* **Q3 — time follows passes.** A random 23-qubit circuit of depth 10 (a ZYZ on every qubit, then
  CNOT or CZ on a random matching, per layer; `G` gates). At each `k`, with `F` fused passes:
  `T_fused / T_unfused <= 1.15 * F / G`, times as in ADR-0028 step 5 (median of 7 processes). The
  15% is launch overhead and the arithmetic of the larger groups; stated so it can fail. Also
  published: `F / G` itself at each `k`, whatever it is.
* **Q4 — the store order is the emitter's.** `cu_q`, as declared, measures L2 bytes per quad within
  5% of `cu_q_grouped` at the twelve P2 follow-up pairs, once the emitter orders the stores.
* **Q5 — still Qiskit's answer.** The fused circuits of Q1, and the 20-qubit random circuit, agree
  with Qiskit's `Statevector` within ADR-0029 P5's thresholds (max |diff| <= 1e-5, infidelity <= 1e-6).

Nothing here has run. The measurements land beside these, unedited, whichever way they go.

## Build sequence

| step | | testable on its own |
|---|---|---|
| 1 | depth to 5; locals a later statement reads | bit-insertion enumeration at depth 3–5; `dead-local` still refused; every earlier test unchanged |
| 2 | emitter-owned store order, PTX and Unibit | Q4; every bit-exact test unchanged |
| 3 | `lyth-circuit`: the circuit, the greedy fuser, kernel generation, a GPU runner | Q1 on GPU and host; Q5 |
| 4 | Unibit nest derived from `base_index` and checked | ADR-0028/0029 emulator tests unchanged; Q1 on the emulator |
| 5 | the measurement of Q2 and Q3 | published either way |

## Scope, said plainly

**What this closes:** circuits at the cost of their fused passes, with no change to their bits, on
every qubit, through generated kernels the compiler costs like any other.

**What it does not.** Commuting gates to make bigger groups (it would reorder arithmetic; that is a
different claim, "equal to rounding", and not this one). A dense product matrix. In-place passes
(still ping-pong: 29 qubits on 16 GB, not 30). Groups above five qubits. Anything about speed against
cuStateVec, which does not run on this Windows machine and is not measured here.

## Progress notes (the frozen predictions above are unedited)

- **Step 1** (`953d1ae`): splits to depth 5, locals a later statement reads; every leaf of depth-3,
  4 and 5 splits checked against bit insertion; `dead-local` still refused.
- **Step 2** (`644d023`) and **Q4, measured** (tool frozen with that commit; results in
  `docs/prereg/ADR-0030.q4-results.json`): the emitter stores a split kernel's leaves grouped by
  buffer, in split-tree order. `cu_q` as declared now reaches the L2 at 64.9 - 68.6 B/quad at the
  twelve P2 follow-up pairs, down from 73.6 - 119.1; against `cu_q_grouped`'s recorded figures the
  ratio is 0.958 - 1.050. **Q4 PASS**, and at `(0, 8)` only just: 1.0499 against a 5% bound. The tree
  order (00, 01, 10, 11) is not the order `cu_q_grouped` measured with (00, 10, 01, 11); grouping by
  buffer is what mattered, which is what the pass shows and no more. Every bit-exact test unchanged.
