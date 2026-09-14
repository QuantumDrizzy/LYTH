# ADR-0001 — What LYTH is, and what would kill it

**Status:** Accepted  
**Date:** 2026-09-14  
**Deciders:** Antonio / QuantumDrizzy  

---

## Thesis (one sentence)

> **A kernel that cannot say what it costs does not compile.**  
> Memory-first, not arithmetic-first. The machine is a value, not a flag.

## The question that can lose (must stay open for three weeks)

> **What can LYTH express that a Rust DSL with procedural macros cannot?**

Everything else in the plan — profiles, capability refuse, evidence bundles, cost
oracle — is *a Rust crate plus a CLI*. A new parser is the most expensive thing
you can build. It is justified only if:

1. Lowering one source to **PTX + SPIR-V + LLVM + uasm** with **compile-time refuse**
   is materially harder inside macros than as a real frontend, **and**
2. Memory-first syntax (movement declared, arithmetic subordinate) cannot be
   expressed as attributes without becoming unreadable.

**Falsification:** if after three weeks of dogfooding `lyth probe` on hand-written
CUDA it turns out `#[kernel]` + macros cover ~90%, **stop the parser, publish the
negative result**, keep the probe CLI. Same discipline as Unibit (256-bit width
bought nothing) and SWEEP (no kernel margin over cuStateVec).

---

## Inversion (why this is not “another LLVM”)

Arithmetic-first languages write `c = a + b` and leave movement *implicit*.

On a machine where moving bytes costs 100–1000× operating on them, that is backwards.

**In LYTH the program describes data movement through the memory hierarchy.
Arithmetic is what happens to bytes at the stops.**

You already write this way in your head (NIBBLE: “FP16 weights never materialise”,
“16-entry table lives in the warp register file”). LYTH removes the hand
translation into CUDA’s arithmetic syntax.

```
kernel gemv_int4 on machine.sm_120:
    stream  W : dram -> reg   once, coalesced 128B
    resident x : smem          for grid
    accum   y : reg -> dram    drain once
    at reg: y += unpack4(W) * x
```

You cannot operate on what you have not declared resident. The compiler does not
*infer* movement; it *checks* that the arithmetic fits the declared traffic.

**Arithmetic intensity is a type, not a comment.** Declared intensity must match
intensity computed from the body (bytes moved × flops). Mismatch is a compile error
that names the ridge point of the machine file.

**Baseline is part of the signature.** A kernel that cannot produce its comparison
does not compile.

**`[KNOWN_LIMIT]` is syntax**, propagated into the evidence bundle; a kernel with
undeclared open limits cannot be marked verified.

---

## Machine as a value (ten-year durability)

`-t sm_120` is a flag. Flags rot.

```
machine sm_120:
    level dram  capacity 16GiB  bandwidth 448GB/s  latency 350cyc
    level l2    capacity 32MiB  bandwidth 2.1TB/s
    level smem  capacity 100KiB per block
    level reg   capacity 64KiB  per block
    op   fma.f16  at reg
    op   tma      absent   # or present — capability refuse if required
```

A 2036 GPU is a **new file**, not a new backend. `lyth machine probe` measures live
silicon and **fails if the description lies**.

**Falsification (hardware):** if a relevant machine cannot be described as levels
(true processing-in-memory where operate-where-data-lives collapses the model),
publish that the model is wrong and stop.

---

## Polymorphism (the axis that matches measured reality)

Not `fn f<T>`. **Machine and layout:**

```
kernel gemv[L: layout, M: machine](...)
```

Monomorphised at compile time; **cost check re-run per instantiation** — same body,
different layout ⇒ different intensity (NIBBLE: format alone was 19–31% with kernel
held constant).

---

## Profiles for v1 — one primary

| Profile | Backend | Why |
|---------|---------|-----|
| `@gpu` | PTX → cubin (sm_120 first) | Measured silicon, Nsight, roofline already here |

`@oracle` (Unibit uasm) was gated in Fase 0.5 and **FAILED** rank-order vs silicon
GFLOP/s (GPU and CPU). See ADR-0003. Unibit stays a density/museum instrument —
not a v1 design profile.

`@fsw`, `@mcu`, SPIR-V/Android: later. Sound serious; least validatable today.

### Oracle gate — closed FAIL (2026-09-14)

Three kernels on RTX 5060 Ti: Unibit density order `mps > llm > ising` ≠ silicon
GPU GFLOP/s `llm > ising > mps` (CPU: `ising > llm > mps`). Museum; `@oracle`
dropped from v1.

---

## Inside / outside

| Inside | Outside |
|--------|---------|
| Kernels (hot path) | Orchestration, UI, HTTP, DB, consensus, p2p |
| Data movement by hierarchy | Runtime, GC, framework |
| PQC: NTT, Merkle, hashing (FIPS 203/204/205) | Chain primitives (rot in ~3 years) |
| `constant_time` verified by taint on addresses/branches | Smart contracts |

PQC: NTT/Merkle are first-class `@gpu` candidates. Do **not** assume memory-bound —
Kyber n=256 may be latency- or compute-bound until batched. LYTH answers which.
`constant_time` is the feature nobody ships: no secret-dependent address or branch.

---

## Adoption (must answer before CHIASMA dogfood)

How does a `.lyth` enter an existing CUDA tree?

| Path | Pros | Cons |
|------|------|------|
| **PTX blob + CUDA Driver API** (`cuModuleLoad`) | Drop-in tomorrow; no nvcc link | Separate load path |
| **Object / cubin that `nvcc` links** | Feels native | Toolchain coupling |

**Default for v1 dogfood: Driver API PTX blob.** C ABI for host glue. Document both;
pick one in ADR-0002 when Fase 2 starts.

Dogfood #1: CHIASMA `k_integrate` / `k_propagate` (163k LIF, known `(int)weights`
quantisation bite) — not a toy GEMV.

---

## Error messages are the product

Refuse must say **what is missing and what to do**:

```
error: gemv_int4 declares 2.0 flop/byte, body computes 0.5
  bytes moved: 4 per output × N (dram→reg, once)
  flops:       2 per output
  machine.sm_120 ridge ≈ 68 flop/byte: you are asking for ~0.7% peak FLOPS
  and ~100% peak bandwidth — correct for a GEMV. Did you mean to declare 0.5?
```

```
error: target machine.mali-g78 lacks op tma
  required by: stream W : dram -> smem via tma
  options: (1) remove `via tma` and use coalesced global load
           (2) retarget machine.sm_120 where tma is present
```

A bare `lacks TMA` without next steps is worse than a silent fallback.

---

## Phases (closed)

| Phase | Ship | Done when |
|-------|------|-----------|
| **0** | `lyth probe` — **no parser** | Evidence CLI over existing CUDA/Rust; useful in CHIASMA, NIBBLE, SWEEP, Blaze, QuBLAR |
| **0.5** | Oracle rank-order experiment | **DONE FAIL** — kill `@oracle` from v1 (ADR-0003) |
| **1** | `machine` as value + `lyth machine probe` | Live bandwidth vs file; fail if lie — scaffold in ADR-0004 |
| **1b** | Intensity as checked type + CHIASMA dogfood | `intensity-check` + suite — ADR-0005 |
| **1c** | Kernel IR (streams / ops / capability refuse) | `kernel-check` — ADR-0006; peak measured |
| **1d** | `constant_time` taint refuse | `ct-check` — ADR-0007 (PQC gate before NTT) |
| **2** | Parser + intensity typecheck | `@gpu` PTX only; dogfood CHIASMA LIF or NIBBLE GEMV |
| **3** | Polymorphism layout × machine | Cost re-checked per instance; NIBBLE becomes a language answer |
| **4** | `constant_time` + PQC NTT | Kyber path; KHAOS consumer |
| **5** | `lyth probe anchor` | Content-addressed hash of bundle; chain-agnostic; **outside** the compiler |

### Evidence bundle (mandatory fields — exit 1 if missing)

- Claimed number + **baseline**
- Compile flags
- Clock / boost state (ncu lesson)
- Cache state / flush policy
- Repetition count N
- Arch from live device (not a banner)
- Open `[KNOWN_LIMIT]` list

---

## What this does **not** promise

Faster than hand CUDA at 83.9% of DRAM. The ceiling is silicon.

What it promises:

1. Hard to write something *accidentally* slow — extra moves are visible.
2. Compile-time fraction of peak before the profiler.
3. You cannot lie about cost — declaration checked against body.

---

## Relation to existing Desktop work

| Repo | Role under this ADR |
|------|---------------------|
| Unibit | Density/museum instrument — **not** v1 `@oracle` (0.5 FAIL) |
| TRM `gpu-ir` | IR spine for `@gpu` (extend, do not fork) |
| rse-hpc-lab labkit | G1–G8 patterns absorbed into `lyth probe` |
| KARDASHEV | Civilisation gates stay TS; LYTH does not reimplement them |
| VENTUS | `@fsw` later; harness `source` field informs probe cases |
| CHIASMA | First `@gpu` dogfood (`k_integrate` / `k_propagate`) |
| KHAOS | PQC consumer for Fase 4 |
| int4-gemv / NIBBLE | Layout polymorphism proof case |

---

## Decision

Accept this thesis and phase order. **Do not start a parser before Fase 0 and 0.5
have either succeeded or published a kill.**
