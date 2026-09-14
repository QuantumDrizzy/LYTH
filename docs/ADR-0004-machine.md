# ADR-0004 — Machine as value (Fase 1 start)

**Status:** Accepted (scaffold)  
**Date:** 2026-09-14  
**Depends on:** ADR-0001  

## Decision

A machine is a **JSON value** (`lyth-machine/0.1`), not a CLI flag.
`lyth-probe machine-check` compares claimed level bandwidths to a measurement
file and **FAIL**s if the file lies beyond `--tol` (default 5%).

```bash
lyth-probe machine-check \
  --machine fixtures/machine/sm_120.json \
  --measurement fixtures/machine/meas-sm_120-2026-09-14.json
```

## sm_120 snapshot (this box)

- `dram.bandwidth_gbs = 398.39` from `host_reference.py` torch-sum read
- Prior int4-gemv calibrated peak: 384.95 GB/s (~3.5% relative) — within tol

## Non-goals

Inventing L2/shared numbers without a probe. Levels with `bandwidth_gbs: 0`
are ignored until a measurement names them.

---

## Which figures the ridge is made of, and why (2026-09-14)

Asked of `sm_120.json`: the FP32 ridge reads **42.9 flop/byte**, but this card's datasheet gives
23.7 TFLOP/s over 448 GB/s, a ridge of **52.9**. Where does 42.9 come from?

**From measurement, both terms:**

```
peak_tflops     15.37 TFLOP/s   achieved cuBLAS SGEMM, n = 8192
dram bandwidth 358.43 GB/s      median of six runs, 4.9% spread
ridge          15.37e3 / 358.43 = 42.88 flop/byte
```

Measurement reaches **65%** of the datasheet FLOPS and **80%** of the datasheet bandwidth, which
is ordinary for both. `tools/peak_probe.py` produced both, on the same day, on this machine.

A coincidence worth naming, because it invites a wrong conclusion: `19.2e3 / 448 = 42.86`, and
19.2 TFLOP/s is the datasheet FP32 of the **non-Ti** RTX 5060. That pair lands within 0.05% of
the measured ridge by accident. No figure from that card appears in this file or anywhere in
this repository.

### The rule

**Both terms come from the same source, or the ratio describes no machine.**

- measured over measured → 42.9. Answers "is this kernel memory-bound *on this machine*".
- datasheet over datasheet → 52.9. Answers "is this kernel memory-bound on a card of this
  model, at its ceilings". Correct for a procurement decision; not for a kernel on this desk.
- one of each → a number about nothing. This is the failure the rule exists to prevent.

LYTH takes the measured pair, for the reason ADR-0001 gives: a machine is a value you can
re-derive, and a datasheet is a claim by someone who is not in the room. The conclusion is
unchanged either way — 0.1667 flop/byte is 0.39% of a 42.9 ridge and 0.32% of a 52.9 one, deeply
memory-bound against both — but "unchanged either way" is a thing you get to say only after
checking, not instead of it.

**[KNOWN LIMIT]** 15.37 TFLOP/s is what cuBLAS *achieved*, so it is a floor on what the silicon
can retire rather than a ceiling. A kernel that beat cuBLAS would sit above this ridge. None
here does, and none here claims to.

### What the question exposed

The answer above was already in the file's `measurement` block. The file's `notes` were not: they
still quoted 398.39 GB/s, 15.03 TFLOP/s and a ridge of 37.7 — the values from **before**
`f3d3d6a` corrected them. That commit updated the fields and left the prose contradicting them.

Worse, `ridge_flop_per_byte` was a **derived number stored beside its own inputs that nothing
checked**: the `Machine` struct did not even have the field, so serde discarded it. Edit the
bandwidth, forget the ridge, and the file is wrong in a way no tool could see. That is the same
defect class as the stale measurement fixture in `52cfe85`, in the same file, three commits later.

`machine-check` now:

1. recomputes `ridge_flop_per_byte.fp32` from this file's own peak and bandwidth, refusing a
   disagreement past 0.5% — arithmetic on two numbers in one file, so anything past rounding is
   a stale edit rather than measurement noise;
2. checks `peak_tflops` against the measurement, as it already did for bandwidth. Checking one
   term of the ridge and not the other left half of every roofline claim unobserved.

`fixtures/machine/sm_120-stale-ridge.json` is the defect frozen as a fixture, and the suite runs
it expecting FAIL.

**Not done, deliberately:** cross-checking the ridge against a table of datasheet specs per `sm`.
That would install vendor numbers as the authority, which is the arrangement this whole ADR
exists to replace. The useful half of that idea is the one implemented above — a file that cannot
contradict itself.

### `peak_tflops` is achieved, and the file is defended mechanically

Stated once, at the point where it matters: **`peak_tflops` is an achieved cuBLAS SGEMM at
n = 8192, not a theoretical peak.** For sm_120 that is 15.37 TFLOP/s against a datasheet 23.7.

Raised as a worry that someone reads the JSON in six months, recognises the card, and "corrects"
15.37 to 23.7 by the same intuition that prompted this section. Three things stand in the way,
in increasing order of usefulness:

1. The file's own `notes` say every figure is measured. A reader has to read them.
2. The `peak_tflops` field in `machine.rs` says so. Someone editing the *code* sees it. It did
   not say so before this question: it read `Peak FP32 (or stated) TFLOPS`, and "(or stated)" is
   a licence to put a datasheet number there. That wording is gone.
3. **`machine-check` refuses the file.** This is the only defence that does not depend on anyone
   reading anything:

```
$ machine-check <file with peak_tflops: 23.7> meas-sm_120-2026-09-14.json
verdict: FAIL — machine file lies (or measurement disagrees)
[1] peak_tflops: machine file claims 23.70, measured 15.37 (rel 0.351 > tol 0.05)
```

The scenario tested is the hard one: the hypothetical editor *also* recomputed the stored ridge,
so the file is internally consistent. The internal check passes and the comparison against
measurement is what catches it — which is the argument for checking both terms of the ridge
rather than one. Frozen as a test.

**[KNOWN LIMIT]** Editing the machine file *and* the measurement file together defeats all three.
Nothing in a static check can prevent falsified evidence; what it can do is make the falsification
have to be deliberate and leave two edits behind. The measurement file records the method
(`tools/peak_probe.py`) so the claim stays re-runnable by anyone who doubts it.
