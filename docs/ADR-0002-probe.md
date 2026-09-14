# ADR-0002 — `lith probe` evidence schema (Fase 0)

**Status:** Accepted  
**Date:** 2026-09-14  
**Depends on:** ADR-0001  

## Decision

Ship `lith-probe` as a Rust CLI **with no language frontend**. Schema id:
`lith-evidence/0.1`.

Mandatory fields: `claim`, `value`, `unit`, `baseline{name,value,unit}`,
`n_reps≥1`, `arch`, `compile_flags[]`, `clock_state`, `cache_state`,
`known_limits[]` (required when clock/cache are `unknown`), `verified`.

`verified: true` is illegal while any `known_limit.status == open` or while
clock/cache remain `unknown`.

## Commands

| Command | Role |
|---------|------|
| `lith-probe new` | Scaffold (always `verified: false`) |
| `lith-probe validate` | Exit 1 on violations; messages include next steps |
| `lith-probe gap` | Foreign bundles (rse / neuromod) → missing-field checklist |
| `lith-probe hash` | Canonical SHA-256 (prep for Fase 5 anchor) |
| `lith-probe schema` | Print mandatory list |

## Non-goals (this ADR)

Parser, PTX emit, machine files, Unibit oracle, chain anchoring.

## Dogfood

Run `lith-probe gap` on existing Desktop evidence. Missing clock/cache/baseline
is expected — that **is** the Fase 0 deliverable: a shared refuse list.
