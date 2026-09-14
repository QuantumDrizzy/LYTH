# ADR-0002 — `lyth probe` evidence schema (Fase 0)

**Status:** Accepted  
**Date:** 2026-09-14  
**Depends on:** ADR-0001  

## Decision

Ship `lyth-probe` as a Rust CLI **with no language frontend**. Schema id:
`lyth-evidence/0.1`.

Mandatory fields: `claim`, `value`, `unit`, `baseline{name,value,unit}`,
`n_reps≥1`, `arch`, `compile_flags[]`, `clock_state`, `cache_state`,
`known_limits[]` (required when clock/cache are `unknown`), `verified`.

`verified: true` is illegal while any `known_limit.status == open` or while
clock/cache remain `unknown`.

## Commands

| Command | Role |
|---------|------|
| `lyth-probe new` | Scaffold (always `verified: false`) |
| `lyth-probe validate` | Exit 1 on violations; messages include next steps |
| `lyth-probe gap` | Foreign bundles (rse / neuromod) → missing-field checklist |
| `lyth-probe hash` | Canonical SHA-256 (prep for Fase 5 anchor) |
| `lyth-probe schema` | Print mandatory list |

## Non-goals (this ADR)

Parser, PTX emit, machine files, Unibit oracle, chain anchoring.

## Dogfood

Run `lyth-probe gap` on existing Desktop evidence. Missing clock/cache/baseline
is expected — that **is** the Fase 0 deliverable: a shared refuse list.
