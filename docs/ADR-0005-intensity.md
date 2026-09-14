# ADR-0005 — Intensity as a checked type (pre-parser)

**Status:** Accepted (scaffold)  
**Date:** 2026-09-14  
**Depends on:** ADR-0001, ADR-0004  

## Decision

Arithmetic intensity is **not a comment**. Before any `.lith` parser exists,
`lith-probe intensity-check` refuses a declaration that does not match
`body.flops / Σ body.moves.bytes` (default 5% relative tol).

Optional `--machine` attaches the ridge (`peak_tflops * 1e3 / dram.bandwidth_gbs`)
and classifies MemoryBound / NearRidge / ComputeBound.

```bash
lith-probe intensity-check fixtures/intensity/k_integrate.json \
  --machine fixtures/machine/sm_120.json
# PASS — ~0.21 flop/byte, MemoryBound vs ridge ~60

lith-probe intensity-check fixtures/intensity/gemv-declare-lie.json \
  --machine fixtures/machine/sm_120.json
# FAIL — declared 2.0 vs computed ~0.44; error names the ridge
```

## CHIASMA dogfood

`k_integrate` / `k_propagate` accounting lives under `fixtures/intensity/`.
The `(int)weights` plasticity bite is a `known_limit`, not an intensity failure.

## Non-goals

Parsing CUDA. Inferring moves from PTX. That is Fase 2.
