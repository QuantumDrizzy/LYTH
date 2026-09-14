# ADR-0007 — Constant-time taint (pre-parser)

**Status:** Accepted (scaffold)  
**Date:** 2026-09-14  
**Depends on:** ADR-0001  

## Decision

`lyth-ct/0.1` is a tiny op list with taint. Refuse:

1. **Load/store** whose address is secret-tainted  
2. **Branch** whose condition is secret-tainted  

Secret arithmetic is allowed (taint joins). `declassify` requires a non-empty note.

```bash
lyth-probe ct-check fixtures/ct/kyber-ntt-ok.json          # PASS
lyth-probe ct-check fixtures/ct/secret-gather-fail.json    # FAIL
```

## Why now

ADR-0001 named `constant_time` as the feature nobody ships. Shipping the
checker **before** the NTT kernel means Fase 4 dogfood has a gate on day one.

## Non-goals

Spectre/ISA fence proofs. Full LLVM TF. Parsing C. That stays later.
