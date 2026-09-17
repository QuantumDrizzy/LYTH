# ADR-0025 — Standalone: a second target, and the instruction that is missing

**Status:** Accepted — all five steps built and measured
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
| `reduce sum` | a shared-memory tree | `VFREDUCE` — **`VREDUCE` turned out to be integer; see step 3** |
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

## Step 1, as measured — and the bug only a second machine could show

`tools/unibit_probe.py`, applying `sm_120.json`'s own first line to the second machine:

> EVERY NUMBER HERE IS MEASURED ON THIS MACHINE, NOT A DATASHEET

| | measured | this loop's limit | loop-free |
|---|---|---|---|
| memory, `LQ` | **28.43 bytes/cycle** | 28.44 | 32 |
| arithmetic, `VFMA` | **14.21 flops/cycle** | 14.22 | 16 |
| **ridge** | **0.4999 flop/byte** | | |

Both within 0.07% of the derivation, which is what a deterministic emulator gives: there is no
variance to average out, so the probe confirms the accounting rather than discovering a rate.

### The unit is a cycle, and pretending otherwise would have been a datasheet

Unibit has **no clock**. `src/cpu.rs` charges one cycle per instruction, three more for a
division, three for a mispredicted branch, and names no frequency anywhere. So `bandwidth_gbs`
is meaningless for it — the same shape of error ADR-0022 caught when the shared pipe turned out
to want accesses per second rather than bytes per second.

`Ridge` now carries `flops_per_unit`, `bytes_per_unit` and a `unit` label. The ratio is a
flop/byte either way, so the ridge, the regime and every ceiling are unit-agnostic and only the
printing changes: TFLOP/s and nanoseconds on a clocked device, flop/cycle and cycles on one
without. And a machine file that **declares** `time_unit: "cycle"` without carrying per-cycle
rates is refused, because a declaration nothing checks is a comment.

### The regime flips, which is what a second machine is for

| | `sm_120` | `unibit` |
|---|---|---|
| ridge | 36.9 flop/byte | **0.4999** |
| `saxpy`, 0.167 flop/byte | memory-bound | memory-bound |
| `sum`, 0.25 | memory-bound | **near the ridge** |
| `matmul`, 8.0 | **memory-bound**, `smem` binds | **compute-bound**, `reg` binds |

Same source, same derivation, opposite answer — because the machines are a factor of 74 apart.

### And it found a missing side of the roofline

The first run of the matmul on `unibit` printed:

```
ceiling  dram binds — 227.407 flop/cycle, 1600.34% of peak FLOPS
```

**A ceiling at sixteen times the machine's peak.** ADR-0022 built the ceiling as a minimum over
*memory* levels and never compared against compute, and on `sm_120` that was invisible: the
densest kernel this language can write is 16 flop/byte against a ridge of 36.9, so nothing had
ever crossed it and the memory answer was always the small one.

A roofline has two sides. The compute side is now a candidate like any other — you cannot retire
more flops per unit of time than the machine retires — and the matmul on `unibit` reports
**exactly 100% of peak**, which is what a compute-bound ceiling is.

One consequence worth recording: with three candidates, "slowest over fastest" stopped being a
useful spread. On a memory-bound matmul the compute candidate is the fastest of the three by a
mile, and quoting `10.31x` answers a question nobody asked. The line now compares the binder
against the **runner-up**, which restores the 2.24x that was meaningful and keeps meaning it.

## Step 2, built — a kernel body that the emulator agrees with

`crates/lyth-uasm` takes a `KernelIr` and `n` and returns a complete `.uasm` text: `.data`
holding the inputs, `.text` holding `_start`, a lane loop, and an `ecall` per drained buffer so
the result leaves the machine. `emit_program` is the whole surface.

The inputs are baked in from **`lyth_lang::inputs`**, the same generator the host oracle reads.
That is not tidiness. A program computing on different numbers from the oracle would make every
comparison between them a comparison of two functions, and this back end has no other way to be
checked — there is no `ncu` for Unibit and no second implementation to agree with.

| LYTH | Unibit |
|---|---|
| `stream x : dram -> reg` | `LQ t, 0(s_i)`, the base bumped by 32 each iteration |
| a scalar parameter | `LI` + `VSPLAT.w` once, **before** the loop |
| `a * x + y` (one `Op::Fma`) | `MV` the addend, then `VFMA` — the instruction accumulates into `rd` |
| `+`, `-`, `*` | `VFADD`, `VFSUB`, `VFMUL` — step 0's instructions |
| `-x` | `VFSUB` against a splatted zero, because there is no packed negate |
| `drain` | `SQ`, then a second loop that prints it |

Nine things are refused by name rather than approximated: a `tile` (no level to stage into), a
`coarsen`, a `reduce` (step 3), a rank-2 `space` (step 3), an `n` that is not a multiple of 8
(no tail loop — refused rather than rounded), a narrow buffer (the float unit is packed single
precision only, and emitting it as f32 would move twice the bytes the cost model derived), more
than eight buffers (one base pointer per saved register), a body needing more than seven live
values (no spilling), and a float divide (step 0 added add, sub, mul, fma, max and min).

`tests/against_the_emulator.rs` assembles each emitted program with the real `unibit` binary,
runs it, reads back the printed lanes and compares them **bit for bit** against `lyth_lang::eval`
— the same host oracle the PTX back end is checked against. Two kernels, one that fuses and one
that does not, 64 and 32 elements. They pass.

### The defect, and it was the check again

The first honest run said:

```
the program printed 8 lanes
  left: 8
 right: 64
```

Which reads as a loop that ran once. It was a **parser** that ran once: `PRINT_REG256` emits no
newline, so the eight registers arrive concatenated on a single line, and the test scanned
`text.lines()` taking one match each. The emitted program had been right from the first run.

It is the third time in this project that a test's own reading of the output was the thing that
was wrong, and the second in this file — the skip above it had been reporting seven passes in
0.01 seconds because `Path::parent` is lexical and the emulator was being looked for in
`LYTH/crates/Unibit`. Both have the same shape: **a check whose failure state was unreachable**,
one by skipping and one by parsing. The rule ADR-0000 already states covers it, and the fix in
both cases was to make the check say where it looked.

## Step 3, built — the level that has no level

Step 3 was meant to be plumbing: wire `space` at rank 2, wire `reduce` onto `VREDUCE`, done.
Both halves turned out to be about the same thing, which is that **a reduction needs somewhere
to put the tree**, and this machine's somewhere is not a memory level at all.

### `VREDUCE` sums integers, and the ADR above says otherwise

The table earlier in this file says

> `reduce sum` | `VREDUCE`, which already exists

and that was checked as far as the instruction's existence and no further. `alu.rs` reduces
`a.w(i) as u64`: over eight f32 lanes it adds their **bit patterns**. The row was wrong in
exactly the way step 0's finding was wrong — eight floats fit in a register and nothing could
fold them — and it is the second time this project has assumed an instruction did the float
thing because `.w` is 32 bits wide.

So `VFREDUCE rd, rs1` was added to Unibit: the eight lanes into lane 0, the rest zeroed.

### Its order had to be written down, and the integer one's never did

`VREDUCE` does not document whether it sums in a chain or a tree. It does not need to:
`wrapping_add` is associative, so **no vector of integers can tell the two apart**, and the
order was never a decision there. Float addition is not associative. On
`[1.0, 2^-24 x 7]` the chain gives `0x3F800000` and the tree gives `0x3F800003`, and an
instruction that will not say which one it is has not been specified.

It is a tree — stride 4, then 2, then 1 — and that is also the more honest hardware: three
layers of adders rather than a chain of eight.

### The launch shape of a Unibit program is grid 1, block 8

The choice of a tree was not free. It was chosen to match `eval_with_launch`, and the reason it
*could* be is the finding worth keeping:

| | on `sm_120` | on Unibit |
|---|---|---|
| a thread | a thread | **a lane** |
| threads per block | 256 | **8 — the register** |
| blocks | `n / 256` | **1** |
| what one thread folds | elements at stride `grid * block` | elements at stride 8, which is one `LQ` |
| how the block combines | a tree in shared memory | a tree in the register, `VFREDUCE` |

So the oracle for the reduction is `eval_with_launch(ir, n, inputs, 1, 8)` and **not one line of
new host code was written for it**. The function that models a GPU's two-stage fold — each
thread in sequence, then the block tree — describes a lane accumulator followed by a horizontal
sum exactly, because that is the same two stages. That the GPU oracle fell out unchanged for a
machine with no threads is the strongest evidence so far that the launch shape here is a real
launch shape and not a metaphor.

`tests/against_the_emulator.rs` checks the emitted program bit for bit against it at n = 4096,
and a second test asserts that a plain left-to-right fold of the same 4096 inputs gives a
*different* bit pattern — without which the first test would pass against any summation order
and the tree would be an untested claim.

### `reduce` had the GPU's shape written into the language

`ir.rs` said, in one line:

```rust
if r.path != [Level::Reg, Level::Smem, Level::Dram] {
```

One legal path, and it is a GPU's. There are now two — `reg -> smem -> dram` where the tree is
staged, `reg -> dram` where it is not — and **which one a kernel may use is the machine's
answer, not the language's**. The PTX back end refuses the short one (its tree needs shared
memory); `lyth-uasm` refuses the long one (there is no shared level to stage in). Both say so
by name.

The cost model followed. A reduction used to derive 8 bytes read and 8 written at `smem`
unconditionally, and on `unibit` that charged the kernel for a level it does not have — the
same error as costing a tile that cannot be staged, pointing the other way. The traffic now
follows the declared path, and the same `sum` reads:

| | `sm_120` | `unibit` |
|---|---|---|
| candidates | dram, smem, reg | **dram, reg** |
| binds | dram, 0.68% of peak | dram, **50.01% of peak** |
| runner-up | 3.69x faster | 2.00x faster |
| intensity | 0.25 | 0.25 — unchanged, because `smem` was never in the ratio |

A kernel at 0.25 flop/byte against a ridge of 0.4999 is sitting almost exactly on it, which is
the second time this machine has put a kernel somewhere `sm_120` never could.

### Rank 2, and a refusal that is not "not yet"

A rank-2 space whose buffers are all read at `[i, j]` is a linear walk over `rows * cols`, so it
is the rank-1 loop with a larger bound and the emitter never needs the two extents apart. Rank 2
is a claim about the *indices*; when they all agree it is not a claim about the addresses. The
oracle does need the extents, because it decomposes the linear index to address each buffer at
its own permutation — and when every permutation is the identity, that decomposition cancels.

A **permuted** one is refused, and permanently. `b[j, i] = a[i, j]` puts consecutive `j` at a
stride of `rows`, and `SQ` is the only vector store: 32 contiguous bytes, no strided store, no
scatter, no lane extract. `LW`/`SW` could copy it a word at a time and that is refused too,
which is the part worth stating: it would move **4 bytes per instruction where the cost model
derived 32**, so ADR-0022's ceiling would be eight times the machine. A back end that can
produce the right answer at a cost the contract does not describe has to decline.

### And a hole in the other repository's test table

The six packed-float instructions from step 0 were never added to `binary.rs`'s round-trip
list, so their encoding had no test at all. An opcode typo in `decode` would have surfaced as a
LYTH program failing to verify rather than as the table that exists to catch it. All seven are
in it now.

## Step 4, built — the sentence this ADR was written to be able to say

```
$ lyth build examples/main.lyth --machine fixtures/machine/unibit.json -o main.ubo
kernel saxpy on machine unibit
  derived  0.1667 flop/byte  (2 flop / 12 byte per element)
  declared 0.1667 - matches
  ceiling  dram binds - 4.738 flop/cycle, 33.34% of peak FLOPS
  wrote    main.uasm (99488 bytes)
  wrote    main.ubo (33336 bytes)
  run      unibit run main.ubo

$ unibit run main.ubo
y[0:8]
-1.2549801
2.1394424
-6.199203
...
```

No host language in either line. The complaint at the top of this file was that a `.lyth` file
is never a program; it is one now, for one machine, and the eight numbers above are checked
**bit for bit** against `lyth_lang::eval` in `crates/lyth/tests/a_program.rs`. A program that
runs and prints plausible floats is not evidence of anything.

### `main` is a declaration, not a function body

```
main:
    run saxpy(n = 4096, a = 2.0)
    print y[0:8]
```

Two statements and no third. No loop, no condition, no arithmetic — because a `main` that could
branch would take back at the top of the file the refusal every kernel body is held to, and the
refusals are the product. What it says is which kernel runs, the value of every non-buffer
parameter, and which elements leave.

Everything it does not say is a refusal rather than a default, and the one that matters most is
the scalar: a `main` that omits `a` and a compiler that supplies 2.0 produce a program whose
output no reader of the file can predict, and an oracle that agrees with it only by guessing the
same number. `lyth build` on a file with no `main` at all is refused too — on a machine with no
host, the element count and the choice of what to print have nowhere else to come from.

What `main` **cannot** say is where the data comes from, and that is a limit rather than an
omission: this language has no way to read any. Buffer contents are the deterministic generator
in `lyth_lang::inputs`, which is the same one the oracle uses, and the file says so in a comment
because a reader will ask.

### Which back end runs is the machine file's answer

`fixtures/machine/unibit.json` gained `"isa": "unibit"`; `sm_120.json` says `"ptx"`, and a file
that says nothing means `ptx`, which is every file written before there were two. `lyth build`
reads it. That is ADR-0001's rule applied one level further out — the machine is a value the
source names, so the choice of back end follows from the file rather than from the invocation.

The assembler is **invoked, not reimplemented**. Unibit's two-pass assembler is that machine's
toolchain the way `ptxas` is the GPU's, and a second implementation of an object format is a
second thing to disagree about. When it is not on the path the `.uasm` is still written and the
remaining command is printed: a missing tool should not lose the compile.

### A third gap of the same shape: the machine could not print a float

`PRINT_F64` takes f64 bits and there is no conversion instruction to make them from an f32, so
a program that computed a float could only dump the register as hex. That is the same finding as
step 0's and step 3's, for the third time: **`Width::B32` is where this ISA's types run out.**

Unibit gained `PRINT_F32`, and its *formatting* is part of the syscall. `PRINT_F64` beside it
prints to six decimal places, which cannot be read back — `-1.2549801` and `-1.2549802` are the
same eight characters there, so a check against a host oracle would pass on a kernel that was
one ulp wrong in every element. `PRINT_F32` prints the shortest string that parses back to the
same bits. A test asserts that at least one of the eight values really does lose something at six
places, so the distinction is not hypothetical.

That also retired `PRINT_REG256` from the emitted programs, and with it the hex lane decoder in
the test harness — which is where step 2's only real defect had lived.

### And the defect `main` exposed, which was older than any of this

`examples/main.lyth` parsed. The same file with its comments stripped did not:

```
10:5: expected `=`, found `:`
```

The lexer's `run` calls `start_of_line` and then `rest_of_line`, once each per pass. A blank line
consumed its own newline and returned, so the **following** line went straight to `rest_of_line`,
whose space arm ate that line's indentation as ordinary whitespace. No `Dedent` was emitted and
the block never closed. A comment line left its newline behind and so went round the loop again,
which is the entire reason `main.lyth` worked and its comment-free twin did not.

It had been there since the lexer was written and nothing could see it, because every file until
now **ended** with a kernel body and `run` pops the open levels at end of file anyway. `main:` at
column 0 was the first construct that needed a block to close before then.

Two tests now hold it: one that `main` arrives after the dedent, and one that a blank line and a
comment line close a block **the same way** — two spellings of "nothing on this line" must not
mean two things.

## Step 5, measured — the contract against the machine's own counters

ADR-0009 checked the derived traffic against `ncu` on `sm_120` and agreed to ±0.80%, which is
what a sampled hardware counter gives. The same check on this ISA agrees to **zero**:

| | derived | the machine moved |
|---|---|---|
| saxpy, bytes per element | **12.0** | **12.0** |
| packed-float instructions, n = 4096 | n/8 = **512** | **512** |
| packed-float instructions, n = 8192 | n/8 = **1024** | **1024** |

There is no tolerance because there is nothing to average. A machine that charges one cycle per
instruction and counts every memory operation either moved the bytes the model derived or the
model is wrong.

The measurement is **differential**: the same kernel is emitted at two element counts and the
counters are subtracted, so `_start`, the pointer setup and the print epilogue cancel rather
than being estimated. A constant the test does not know cannot bias it — and a second test
asserts the constant really is constant, because if the program printed the whole buffer the
epilogue would scale with `n` and the traffic would come out a third high while still agreeing
with itself.

### And the counter that was wrong was the machine's

```
Packed f32 (8 lanes):  512        Flops (f32 lanes): 4096
```

The work is **8192** flops: 512 instructions x 8 lanes x 2 flops, because `VFMA` is a fused
multiply-add and retires both. The field counts **lane operations** and calls them flops. For
`VFADD` the two coincide; for `VFMA` the counter is short by exactly the multiply.

A roofline built on that counter would place every FMA-dense kernel at half its real intensity.
**LYTH's derivation is the one that is right here**, which is the opposite of the usual
direction — the compiler is normally the thing being checked — and is why the comparison is run
in both directions rather than treating the hardware as the oracle. Left as a `[KNOWN_LIMIT]`
with a test pinning the present behaviour, because changing what `float_ops` weighs is a
decision about that ISA's metrics and not a thing to slip into a verification commit.

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
| **0** | **Unibit** | **done — six instructions, 5 tests, `saxpy_f32.uasm` runs on the emulator** |
| **1** | **LYTH** | **done — 28.43 B/cycle, 14.21 flop/cycle, ridge 0.4999; and it found the missing compute ceiling** |
| **2** | **LYTH** | **done — `lyth-uasm`, bit-exact against the emulator on two kernels, nine refusals by name** |
| **3** | **both** | **done — `VFREDUCE` in Unibit; rank 2 and `reduce` in `lyth-uasm`; the reduction path opened to two shapes** |
| **4** | **both** | **done — `main`, `PRINT_F32`, `lyth build -o main.ubo`; and a lexer defect older than the ADR** |
| **5** | **both** | **done — traffic exact to the byte, and it found the machine's own flop counter short by every fused multiply** |

Step 0 is in another repository and is the precondition for all of it. Steps 1–3 are a backend,
which is known work. Step 4 is the one that answers the complaint, and step 5 is the one that
makes the answer worth something.
