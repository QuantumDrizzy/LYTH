# ADR-0006 — Kernel IR (pre-parser AST)

**Status:** Accepted (scaffold)  
**Date:** 2026-09-14  
**Depends on:** ADR-0001, ADR-0004, ADR-0005  

## Decision

Before a `.lyth` parser exists, the contract is **`lyth-kernel-ir/0.1` JSON**:

- `streams[]` — memory-first traffic (`from`/`to`/`bytes`/`via`)
- `ops[]` — FLOPs at a level
- `requires[]` — capability refuse against `machine.ops`
- `declared_intensity` — checked after lower → `lyth-intensity/0.1`

```bash
lyth-probe kernel-check fixtures/kernel/k_integrate.json \
  --machine fixtures/machine/sm_120.json
```

## Peak (live)

`tools/peak_probe.py` on this box (2026-09-14):

| metric | value |
|--------|-------|
| SGEMM 8192 | **15.03 TFLOP/s** |
| HGEMM 8192 | **44.08 TFLOP/s** |
| dram read (same script) | 357 GB/s (vs 398 host_reference — unlocked variance) |

`machine.sm_120.peak_tflops` = achieved FP32, not datasheet. Ridge ≈ **37.7** flop/byte.

## Non-goals

Parsing CUDA. Emitting PTX. That is Fase 2.
