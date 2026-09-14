# ADR-0008 — Layout × machine polymorphism

**Status:** Accepted (scaffold)  
**Date:** 2026-09-14  
**Depends on:** ADR-0001, ADR-0005, ADR-0006  

## Decision

`kernel gemv[L: layout, M: machine]` is not sloganeering. `lith-poly/0.1` lists
**instances** (one per layout). Each instance lowers to `lith-kernel-ir` and must
`kernel-check` PASS on the machine. Identical Σ stream bytes across layouts is
**FAIL** (fake poly).

Optional `claim`: format wall-cost band from silicon rows.

## NIBBLE anchor (gate_proj, sm_120)

| layout | kernel_us | bytes (shape) | scale B/w |
|--------|-----------|---------------|-----------|
| `int4_g128` | 96.384 | 35 008 512 | 0.015625 |
| `nf4_b64` | 122.080 | 38 191 104 | 0.0625 |

Format cost: **(122.080 − 96.384) / 96.384 ≈ 26.7%** ∈ [19%, 31%] (README).

```bash
lith-probe poly-check fixtures/poly/gate-proj-int4-nf4.json \
  --machine fixtures/machine/sm_120.json
```

## Why this justifies (or kills) a parser later

If authoring many `instances[]` by hand stays readable, macros may suffice.
When `L × M` tables explode and people start copy-pasting streams wrong, syntax
earns its keep. Until then: **JSON poly is the dogfood.**
