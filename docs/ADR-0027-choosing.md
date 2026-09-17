# ADR-0027 — `max` and `min`: the one form of choosing that costs the same either way

**Status:** Accepted — built and checked on both back ends
**Date:** 2026-09-17
**Depends on:** ADR-0013 (the signed zero), ADR-0005 (what a flop is), ADR-0026 (what LYTH is)

## The question this answers

LYTH refuses a data-dependent branch, and ADR-0026 refuses one in advance for every future
feature. The reason is exact: **the byte count would depend on which side ran**, and a count
that depends on the data is not a count the compiler can state.

`max(a, b)` does not have that property. It is one instruction, the same instruction whichever
operand wins, moving the same bytes either way. It chooses, and its cost does not.

So it passes ADR-0026's test — *can the compiler still say what the kernel costs, and still
refuse the code that lies about it?* — and it is, as far as this language goes, the **only**
shape of choice that does.

## What it unlocks, concretely

Both of these were unwritable in LYTH the day before this ADR:

| | |
|---|---|
| `out = max(x, 0.0)` | a **ReLU**. Every activation layer in every network. |
| `out = min(max(x, lo), hi)` | a **clamp** — and with it the one piece a branch-free Ising solver was missing. Simulated bifurcation (Goto 2019, the Toshiba SBM) integrates an ODE with no randomness and no condition in the update; its only non-arithmetic step is the boundary clamp. |

One operator, and the two domains this project's own repositories live in — AI and annealing —
stop being inexpressible.

## The decision that was already made, and was kept

`ReduceOp::flops` has said since ADR-0013 that **`max` and `min` retire no flops**: they are a
compare-and-select, no vendor's FLOP/s figure counts them, and charging them would place the
kernel against a compute ceiling measured with FMA — the one comparison ADR-0005 forbids.

`BinOp::Max` and `BinOp::Min` are charged the same, and a test asserts the two agree. They have
to: `reduce max` and `max(a, b)` are one operation reached by two syntaxes, and a kernel's
declared intensity must not depend on which the author wrote.

The consequence is worth stating rather than discovering. A ReLU derives:

```
derived  0.0000 flop/byte  (0 flop / 8 byte per element)
payload  4 read + 4 written, at dram
ceiling  dram binds — the kernel retires no flops, so there is no FLOP/s to report
```

**Zero is the right answer, not a degenerate one.** An activation retires no arithmetic and
moves two words; anyone who has profiled one knows it is a memory pass wearing a compute name.
The language now derives that instead of leaving it to be found.

`[KNOWN LIMIT]`, inherited from ADR-0013 and repeated here so it is not lost: zero flops is not
zero time. The instruction still issues.

## The trap, which was already mapped

`f32::max` is **not a deterministic function of its inputs when both are zero.** It lowers to
`llvm.maxnum`, which may return either operand when they compare equal, and `-0.0 == 0.0`.
Measured on this machine: constant-folded it yields `+0.0`, executed it yields `-0.0` — the same
source, two answers, decided by the optimiser. PTX `max.f32` has no such freedom and returns
`+0.0`, which is IEEE 754-2019 `maximumNumber`.

ADR-0013 found this the hard way: 135 of 65536 block partials differed from the device.

So `BinOp::apply` — the new one-definition entry point the evaluator now uses for **every**
operator — delegates `max` and `min` to `ReduceOp::combine` rather than calling `a.max(b)`. A
test fails the moment anyone simplifies it back, and a second test checks that the elementwise
and reduction spellings agree on exactly the pair of inputs where it is hardest to notice.

## Syntax, and why it is a closed set

`max(a, b)` and `min(a, b)`, written as calls because there is no infix spelling anyone would
read. This is **not** a call syntax: there are no user functions in this language, so `(` after
any other name is an error rather than a call to something undefined, and `sqrt(x)` does not
parse. The parser needed one token of lookahead to tell `max(` from a buffer named `max` — which
somebody will write, because it is the obvious name for the output of a reduction.

## What it cost to add

| file | |
|---|---|
| `ast.rs` | two variants, `flops`, `is_call`, and `apply` as the single definition |
| `parse.rs` | one lookahead, one arm in `atom` |
| `eval.rs` | **two match sites deleted**, both replaced by `apply` |
| `lyth-ptx` | `max.f32`, `min.f32` — no `.rn`, because a select does not round |
| `lyth-uasm` | `vfmax`, `vfmin` — step 0 had already put them in the ISA |
| tests | 7 in `lyth-lang`, 1 bit-exact on the emulator |

The evaluator got *smaller*: it had two copies of the operator table and now has none.

## What this does not open

Stated because the next request will be `sqrt`, `exp` or `tanh`, and the answer is not
automatic. Those are arithmetic — they retire flops, and how many is a question about a
particular unit's implementation rather than about the source. They are not refused by this
ADR, but neither are they implied by it: each would need its own flop count, justified, before
it could be costed. `max` is the easy case precisely because the honest answer is zero.
