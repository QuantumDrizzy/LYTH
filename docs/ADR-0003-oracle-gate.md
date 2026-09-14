# ADR-0003 — Oracle gate (Fase 0.5)

**Status:** Closed — **FAIL**  
**Date:** 2026-09-14  
**Depends on:** ADR-0001, ADR-0002  

## Question

> If `@oracle` (Unibit uasm) is used to *design* kernels, does it **rank-order**
> the same way as measured silicon?

## Verdict

**FAIL.** Live silicon on RTX 5060 Ti (`sm_120`), 2026-09-14
(`Unibit/tools/host_reference.py` → `Unibit/docs/host-reference.json`):

| id | Oracle FLOPs/inst | GPU GFLOP/s | CPU GFLOP/s |
|----|-------------------|-------------|-------------|
| `mps_chain` | 69.57 | ~0.0007 (launch-bound) | 0.024 |
| `llm_matvec` | 13.60 | **10.38** (fp16) | 1.94 |
| `ising_energy` | 10.48 | 0.67 | **2.09** |

- Oracle order (desc): **mps > llm > ising**
- GPU GFLOP/s order: **llm > ising > mps**
- CPU GFLOP/s order: **ising > llm > mps**

```bash
lyth-probe oracle-check fixtures/oracle/case-gpu-gflops-2026-09-14.json   # exit 1
lyth-probe oracle-check fixtures/oracle/case-cpu-gflops-2026-09-14.json   # exit 1
```

Evidence: `fixtures/adopted/oracle-gate-0.5-fail.json`.

## Consequence (ADR-0001 falsifier 2)

**Drop `@oracle` from v1.** Unibit remains a density / museum instrument.
Do not use uasm FLOPs/inst to choose CUDA kernel designs.

`[KNOWN_LIMIT]` on the GPU case: MPS row is launch-latency dominated — even
so, the CPU ranking (no launch tax) still diverges from oracle density.

## Non-goals (unchanged)

Predicting absolute CUDA wall-clock from Unibit MIPS.
