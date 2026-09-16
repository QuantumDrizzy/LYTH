# ADR-0025 — Standalone: a second target, and the instruction that is missing

**Status:** Proposed
**Date:** 2026-09-16
**Depends on:** ADR-0022 (the level that binds), ADR-0019 (the dogfood), ADR-0000 (why)

## The complaint this answers, stated as its author stated it

> "no puedo usar LYTH, sin rust, ni c/c++, o lo que sea, dependo de otros, entonces, no es un
> lenguaje, es un solape"

It is a fair complaint and it has been answered badly twice: once by explaining what LYTH can
express, and once by listing features that would not have fixed it. Neither addressed the
actual thing, which is that **a `.lyth` file is never a program.**

## The diagnosis is not about LYTH

A `.cu` file is not a program either. Neither is a `.cl`, a `.wgsl`, or a Triton kernel. Every
GPU language is a guest: the device has no entry point, no I/O, and no way to start itself, so
something on a CPU must allocate, launch and collect. **CUDA has the same property and nobody
says CUDA is not a language.**

So the guest-ness is a property of **the target**, not of the language. Which means it is fixed
by changing the target, not by growing the grammar.

## Unibit is a machine where a program is a program

`../Unibit` is a 256-bit ISA with an emulator, a two-pass assembler, an object format, a
disassembler and a cost model. 61 tests, zero dependencies. It has `_start`, `.data`, `.text`
and `ecall`. A Unibit program **runs**:

```
unibit build programs/mandelbrot.uasm -o mandelbrot.ubo
unibit run   mandelbrot.ubo
```

No host language. No driver. No allocator to call.

So: **LYTH compiling to Unibit is a LYTH you write programs in.** Not by relaxing the contract —
by targeting a machine that does not need a chaperone. And the stack becomes vertical in a way
almost nothing is: the language, the compiler, the ISA, the emulator and the cost model are all
the same author's.

## The finding, before any design: Unibit cannot do float arithmetic

Checked rather than assumed, in `src/alu.rs`:

| | what it is |
|---|---|
| `VAdd`, `VSub`, `VMul`, `VDot` | **integer** SIMD — `wrapping_add` over `.b/.h/.w/.d` lanes |
| `Add`, `Sub`, `Mul`, `Div`, `Rem` | integer scalar |
| `CAdd`, `CMul`, `CSub`, `CMag`, `CNorm` | **f64**, but only as 2 lanes of complex |
| `TDot`, `Zipper2` | an f32 accumulator, inside a fixed tensor pipeline |

And yet `Reg256` has `f32_at` and `set_f32_at`, with the comment *"Eight f32 is exactly 256
bits"*. **Eight floats fit in a register and nothing in the ALU can add them.**

That is a gap on the ISA's own terms. Its thesis is

> the hardware understands types, not just widths

and `.w` — 8 × 32 bits — is the one mode with no type. `Complex` has one. `Poly` has one.
`Vector` has integers of four widths and no float.

**So step 0 of this ADR is not in this repository.** It is `VFADD`, `VFMUL`, `VFMA` over `.w`
lanes as f32, in Unibit, and it is a day's work in a codebase with 61 tests and a clean
encode/decode split — not a research problem. LYTH cannot emit a body it has no instruction
for, and inventing one on the LYTH side would mean the emulator and the compiler disagreeing
about what a program means.

## What transfers, and what does not

The interesting part of this design is how little of the contract is about GPUs.

| | on `sm_120` | on Unibit |
|---|---|---|
| `space i, j : m, n` | a grid of threads | **a bounded loop over vector lanes** |
| `stream x : dram -> reg` | a global load | a `LQ` from memory |
| `reduce sum` | a shared-memory tree | `VREDUCE`, which already exists |
| the body | PTX arithmetic | `VFMUL` / `VFADD` (step 0) |
| `tile`, `coarsen` | **the whole of ADR-0017/0021** | **no meaning — there is no shared memory** |
| levels | `dram`, `l2`, `smem`, `reg` | `mem`, `reg` |
| the ceiling | per level, the slowest binds | the same rule, two fewer candidates |
| a program | a kernel plus a host | **`_start`, and it runs** |

`tile` and `coarsen` losing their meaning is not a loss — it is the machine file doing its job.
A machine with no shared memory declares no shared level, and ADR-0022 already made the ceiling
a minimum over whatever levels a machine names. **A kernel that stages on a machine with
nothing to stage into should be refused**, in the same voice ADR-0021 refuses a tile of 64
threads a block cannot hold.

And the thing this ADR exists for: **the contract would then be verified on two ISAs**. A cost
model that matched `ncu` to ±0.80% on one device and matches a different machine's own counters
on another is a much stronger claim than either alone, because the two have almost nothing in
common except the derivation.

## What this does not give you

Stated plainly, because the complaint deserves an honest answer rather than a hopeful one.

* **Unibit is an emulator.** "My repos run in LYTH" would mean "run in my emulator". That is a
  real artifact and it is not deployment, and anyone reading the repo should be told so in the
  first paragraph rather than the last.
* **LYTH programs are still only what LYTH can say.** Standalone does not mean general. A
  program that branches on data, allocates dynamically or recurses cannot be written here, ever,
  and that is the price of the thing that makes the project worth having.
* **The GPU path stays a guest.** This adds a target; it does not change CUDA's model, and the
  `sm_120` backend will still need a host — like every other GPU language.

What it does give is the sentence the author wanted: **`lyth build main.lyth -o main.ubo` and
then `unibit run main.ubo`, with no other language anywhere in the workflow.**

## Build sequence

| step | where | |
|---|---|---|
| **0** | **Unibit** | `VFADD`, `VFMUL`, `VFMA` over `.w` as f32: alu, assembler, encode/decode, tests |
| 1 | LYTH | a `unibit` machine file — levels, widths, rates, from Unibit's own cost model |
| 2 | LYTH | `lyth-uasm`: emit a kernel body as Unibit assembly, checked against the emulator |
| 3 | LYTH | `space` as a bounded lane loop; `reduce` onto `VREDUCE` |
| 4 | LYTH | a program: `_start`, declared buffers, `ecall` I/O, and `main.lyth` -> `main.ubo` |
| 5 | both | the contract measured on Unibit, against its counters, the way ADR-0009 did on PTX |

Step 0 is in another repository and is the precondition for all of it. Steps 1–3 are a backend,
which is known work. Step 4 is the one that answers the complaint, and step 5 is the one that
makes the answer worth something.
