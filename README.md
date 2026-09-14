# LITH

> A kernel that cannot say what it costs does not compile.

Memory-first kernel dialect. **Not a language yet** — `lith-probe` over CUDA you already have.

## Status

| Phase | What | State |
|-------|------|-------|
| 0 | evidence CLI | **done** |
| 0.5 | Unibit oracle vs silicon | **FAIL — `@oracle` out** |
| 1 | machine-as-value + peak | **live** |
| 1b | intensity-as-type | **live** |
| 1c | kernel IR | **live** (integrate / propagate / gate_proj) |
| 1d | constant-time taint | **live** |
| 1e | layout × machine poly | **live** (gate_proj INT4 vs NF4) |
| 2 | `.lith` parser | blocked |

## Commands

```bash
cargo test -p lith-probe
cargo run -p lith-probe -- suite fixtures/suite-fase1.json

cargo run -p lith-probe -- kernel-check fixtures/kernel/gate_proj_int4.json \
  --machine fixtures/machine/sm_120.json

cargo run -p lith-probe -- poly-check fixtures/poly/gate-proj-int4-nf4.json \
  --machine fixtures/machine/sm_120.json

cargo run -p lith-probe -- ct-check fixtures/ct/kyber-ntt-ok.json
cargo run -p lith-probe -- ct-check fixtures/ct/secret-gather-fail.json
```

## Docs

ADR-0001 … ADR-0008 under `docs/`.
