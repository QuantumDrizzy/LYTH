<p align="center">
  <img src="assets/lyth_banner.jpg" alt="LYTH" width="550"/>
</p>

# LYTH

> A kernel that cannot say what it costs does not compile.

Memory-first kernel dialect for bare-metal HPC & quantum computing. **Not a language yet** — `lyth-probe` over CUDA you already have.

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
| 2 | `.lyth` parser | blocked |

## Commands

```bash
cargo test -p lyth-probe
cargo run -p lyth-probe -- suite fixtures/suite-fase1.json

cargo run -p lyth-probe -- kernel-check fixtures/kernel/gate_proj_int4.json \
  --machine fixtures/machine/sm_120.json

cargo run -p lyth-probe -- poly-check fixtures/poly/gate-proj-int4-nf4.json \
  --machine fixtures/machine/sm_120.json

cargo run -p lyth-probe -- ct-check fixtures/ct/kyber-ntt-ok.json
cargo run -p lyth-probe -- ct-check fixtures/ct/secret-gather-fail.json

# The byte accounting against measured DRAM/L2 traffic, not against itself (ADR-0009).
# CONFIRMED at l2 — 28.41 measured B/neuron vs 29.00 counted, 0.9796x.
cargo run -p lyth-probe -- intensity-check fixtures/intensity/k_integrate.json   --machine fixtures/machine/sm_120.json   --ncu fixtures/ncu/k_integrate-sm_120-2026-09-14.csv --elements 166700 --ncu-level l2

# Same accounting at dram: ABSORBED, 0.5190x. Half the state is L2-resident at F=1.
cargo run -p lyth-probe -- intensity-check fixtures/intensity/k_integrate.json   --ncu fixtures/ncu/k_integrate-sm_120-2026-09-14.csv --elements 166700
```

## Docs

ADR-0001 … ADR-0009 under `docs/`.

`docs/DOGFOOD.md` is the running record for the ADR-0001 parser gate: one row per kernel
put through the tool, and for each one whether a Rust macro would have done the same job.
