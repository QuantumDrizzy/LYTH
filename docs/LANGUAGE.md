# The LYTH language, as it stands

Kept current with the compiler, not ahead of it. **Everything in this file compiles and runs
today**; anything that does not is under "Not yet". Examples are the real files in `examples/`,
which the test suite compiles on every build.

## Shape of a file

```
machine sm_120

kernel <name>(<params>)
    intensity <number>          # optional; checked against the body

    <stream declarations>
    <reduce declaration>        # optional

    at reg:
        <statements>
```

Movement is declared before arithmetic. **You cannot operate on a buffer you have not
streamed** — the compiler does not infer movement, it checks that the arithmetic fits what was
declared. Indentation delimits blocks; tabs are refused rather than given a guessed width.

## Types

| in source | is | passed as |
|---|---|---|
| `u32` | a scalar count | launch parameter |
| `f32` | a scalar, the same for every element | launch parameter |
| `[f32; n]` | a buffer of f32, `n` elements long | device pointer |
| `[f32; blocks]` | a reduction target: one element **per block** | device pointer |

f32 only, one dimension. **A buffer's extent is not optional.** `n` names a `u32` parameter of
the same kernel, and that parameter is what bounds the loop — not whichever `u32` happens to
come first, so a kernel may take a count that is not a length without the compiler mistaking
one for the other.

Two things the compiler refuses that it could not see before extents existed:

**Two streamed buffers of different lengths.** `x: [f32; n]` beside `y: [f32; m]`, both
streamed, asks one index space to walk two lengths. A device pointer does not carry its size,
so there is nothing to check at run time; the declaration is the only place. See
`examples/extent-mismatch.lyth`.

**A reduction target sized by the element count.** `reduce` writes one value per block, so its
target is `[f32; blocks]` and the grid decides how many that is. Declaring it `[f32; n]` tells
a caller to allocate one per element, which is far too big until someone tightens it to what
the declaration says and the kernel writes past the end. See `examples/partial-by-n.lyth`.

## The index space

A kernel without a `space` is rank 1: it walks its buffers at the loop index and the body names
no indices. That is every kernel above.

```
kernel transpose(rows: u32, cols: u32, a: [f32; rows, cols], b: [f32; cols, rows])
    space i, j : rows, cols
    ...
    at reg:
        b[j, i] = a[i, j]
```

`space` names the index variables **outermost first** and the extent each runs over. With one
declared, every buffer access must be indexed: a buffer used bare is a buffer nobody said how to
walk, and assuming row-major would silently pick one of the two answers.

An index is a **permutation** of the space variables — each one once, in any order — and nothing
else. No offsets, no arithmetic. An offset brings the halo problem at the edges with it, and a
computed index makes the memory footprint something to solve for rather than to read off, which
is the property the whole cost model rests on.

A buffer is indexed one way per kernel. `a[i, j]` and `a[j, i]` in the same kernel would be two
addresses per element for one buffer, and this compiler loads each buffer once.

**The traversal order is part of the contract**, exactly as the reduction tree's is. The linear
index decomposes row-major, outermost first:

```
k = i * cols + j        i = k / cols        j = k % cols
```

The host reference walks that same order, so a reduction over a rank-2 space folds its operands
in the order the device folded them.

[KNOWN LIMIT] The flattened index is 32 bits, so `rows * cols` must fit in a `u32`. It is
checked once at launch rather than carried as 64-bit arithmetic on every iteration: 2^32 f32
elements is 17.2 GB, more than any device this compiler has a machine file for.

## Streams

```
stream x : dram -> reg          # read only
stream y : dram -> reg, drain   # read and written back
```

`drain` means the value in registers is stored at the end of the element. A stream without it
is read-only, and assigning to such a buffer does not compile.

A buffer that is **assigned but never read** costs a write and not a read — `split.lyth`
derives `4 read + 8 written`, not `12 + 8`. The declaration does not get to inflate the byte
count with traffic that does not happen.

## The body

One `at reg:` block. Statements are `<target> = <expression>`, evaluated in order; a later
statement sees the value an earlier one produced.

Expressions: names, numeric literals, `+ - * /`, unary minus, parentheses. `a * b + c` and
`c + a * b` contract into one `fma`.

A target is either a **drained buffer** or a **local**. A local is a name that is not a
parameter and never reaches memory; it is legal only as the source of a reduction, because a
local nothing consumes is work whose result is discarded.

## Reductions

```
reduce sum p : reg -> smem -> dram into partial
```

Combines the per-element value `p` across the block and writes **one value per block** into
`partial`, which must therefore hold at least `gridDim.x` elements. Combining those partials is
the caller's job — `lyth run` does it with the kernel's own operator and prints the result.

The path `reg -> smem -> dram` is written out rather than implied, for the same reason a stream
writes its own: this language does not infer movement.

One reduction per kernel. The target must **not** be streamed: it carries no per-element
traffic, and declaring some would inflate the byte count.

| operator | identity | flop/combine | reorderable |
|---|---|---|---|
| `sum` | `0.0` | 1 | no |
| `max` | `-inf` | 0 | yes |
| `min` | `+inf` | 0 | yes |

**`max` and `min` retire no flops.** A compare-and-select is not arithmetic: no vendor counts it
in a FLOP/s figure and no published flop count for a reduction counts its comparisons. So
`examples/max.lyth` derives `intensity 0.0`, and that is a statement with content — a max
reduction cannot be compute-bound at any size, on any machine. Zero flops is not zero time: the
combine still issues. See ADR-0013.

**Only `sum` cares about order.** Float addition is not associative, so the tree shape is part
of the contract and the host reference walks the identical tree. `examples/dot.lyth` at four
elements gives 2.0 by the tree and 1.0 by a left fold — the difference is not rounding noise.
`max` and `min` select an operand and never round, so every tree shape gives the same bits.

Threads past the end of the buffer **contribute the identity** rather than leaving, or the
shared slots they own hold whatever was there before. For `max` that identity is `-inf`, so a
block with no elements at all writes the max of an empty set, which is the right answer.

The awkward values were measured rather than assumed. `examples/not-a-number.lyth` puts NaN and
both infinities through a reduction and host and device agree bit for bit. `examples/signed-zero.lyth`
puts `+0.0` and `-0.0` through one and they did not: `f32::max` returns *either* operand when
both compare equal, giving `+0.0` folded and `-0.0` executed, so the evaluator stopped using it
for that case. An oracle cannot be built on unspecified behaviour.

## What the bus carries, beside what the source asks for

The byte count above is the **payload**: the bytes the kernel wants. The memory system moves
32-byte sectors, and whether those two agree depends on how the buffer is walked.

A 32-byte sector is eight f32. A warp is 32 threads. When neighbouring threads are on
neighbouring elements, a warp covers 128 contiguous bytes in four sectors and every byte fetched
is a byte wanted. When they are not, each thread lands in its own sector and 32 bytes move for
every 4 wanted.

The rule, at rank 2, is about the **fast index** and not about transposition:

> A buffer is coalesced when its **innermost** index is the space's **innermost** variable.

Under `space i, j`, the flattening puts `j` in the ones place of the linear index, so `j` is what
advances between neighbouring threads. `a[i, j]` is contiguous; `a[j, i]` is not. A rank-2 copy
that indexes both buffers `[i, j]` is contiguous on both sides, and swapping only the *space's*
variable order — changing no buffer — makes the same source strided.

```
  payload  4 read + 4 written, at dram
  sectors  4 read + 32 written  (coalescence 0.222, upper bound)
  exact    36 byte per element at this shape (coalescence 0.222), against the 36 byte bound
```

Three numbers because they answer three questions. The **payload** is what `intensity` is checked
against and what the source asked for. The **sectors** figure is a static upper bound: a strided
access is charged a whole sector, which assumes a row of eight elements or more. The **exact**
line needs the launch extents, because the real cost is `min(32, 4 * row)` — a matrix four
columns wide wastes four times, not eight.

`lyth run` also says when the working set fits in L2, read from the driver rather than assumed,
because a model of DRAM traffic cannot be compared against a device that never went to DRAM.

## Cost, derived not declared

The compiler counts bytes from the streams and flops from the expression tree, and checks
`intensity` against the result. **Nobody types the number on the right-hand side of that
comparison.** A mismatch names the machine's ridge and what to write instead.

Traffic is reported per level. A reduction moves bytes at `dram` and at `smem`; the roofline
number stays the DRAM one and the shared traffic sits beside it rather than being mixed in.

## The examples

| file | body | derived |
|---|---|---|
| `saxpy.lyth` | `y = a * x + y` | 2 flop / 12 B = 0.167 |
| `axpby.lyth` | `y = a * x + b * y` | 3 flop / 12 B = 0.25 |
| `horner.lyth` | `y = ((c3*x + c2)*x + c1)*x + c0` | 6 flop / 8 B = 0.75 |
| `lerp.lyth` | `z = x + t * (y - x)` | 3 flop / 12 B = 0.25 |
| `split.lyth` | `lo = a*x` ; `hi = x - a*x` | 3 flop / 12 B = 0.25 |
| `dot.lyth` | `p = x * y`, reduced | 2 flop / 8 B = 0.25 |
| `sum.lyth` | `v = x`, reduced | 1 flop / 4 B = 0.25 |

`horner.lyth` is the one to read: the same traffic as a copy and six times the arithmetic, so
the intensity climbs from 0.167 to 0.75 while the bytes stay put. That is the roofline position
moving because of what the body does, which is what the language exists to make visible.

Three files are meant **not** to compile, and the tests assert that they do not:
`saxpy-lie.lyth` (a declared intensity the body does not have), `forgot-stream.lyth` (a buffer
read with no stream), `dead-local.lyth` and `reduce-streamed.lyth`.

## Running and measuring

```
lyth run <file> --machine <machine.json> -n <elements> [--grid N] [--time REPS] [--set k=v]
```

Every thread loops over a grid-stride, so the grid is independent of `n`. The default is the
device's SM count times four, capped at the blocks the problem needs; `--grid` overrides it so
the choice can be swept rather than believed.

`--time REPS` reports the median of REPS runs after a warm-up, the full spread, and achieved
bandwidth against the machine file's measured figure. **If the result comes out above the
baseline the tool says so and says what to check first**, because a number over 100% of peak is
a claim about the baseline before it is a result.

## Not yet

Control flow in the body. Neighbours — no stencils, no convolution, nothing but the current
element. `max`, `min`, `prod`. More than one reduction per kernel. Types other than f32.
Atomics. Two dimensions. More than one kernel per file. A grid-stride loop, so **no performance
number comes out of this compiler yet** and none is claimed.
