# ADR-0004 — Machine as value (Fase 1 start)

**Status:** Accepted (scaffold)  
**Date:** 2026-09-14  
**Depends on:** ADR-0001  

## Decision

A machine is a **JSON value** (`lith-machine/0.1`), not a CLI flag.
`lith-probe machine-check` compares claimed level bandwidths to a measurement
file and **FAIL**s if the file lies beyond `--tol` (default 5%).

```bash
lith-probe machine-check \
  --machine fixtures/machine/sm_120.json \
  --measurement fixtures/machine/meas-sm_120-2026-09-14.json
```

## sm_120 snapshot (this box)

- `dram.bandwidth_gbs = 398.39` from `host_reference.py` torch-sum read
- Prior int4-gemv calibrated peak: 384.95 GB/s (~3.5% relative) — within tol

## Non-goals

Inventing L2/shared numbers without a probe. Levels with `bandwidth_gbs: 0`
are ignored until a measurement names them.
