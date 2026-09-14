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
| `u32` | the element count bounding the index space | launch parameter |
| `f32` | a scalar, the same for every element | launch parameter |
| `[f32]` | a buffer of f32 | device pointer |

f32 only. One dimension. Exactly one `u32` parameter, which bounds the loop.

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
`partial`, which must therefore hold at least `gridDim.x` elements. Summing those partials is
the caller's job — `lyth run` prints the total.

The path `reg -> smem -> dram` is written out rather than implied, for the same reason a stream
writes its own: this language does not infer movement.

`sum` only, one reduction per kernel. The target must **not** be streamed: it carries no
per-element traffic, and declaring some would inflate the byte count.

Two details that are not decoration. Threads past the end of the buffer **contribute the
identity** rather than leaving, or the shared slots they own hold whatever was there before.
And the tree order is part of the contract: float addition is not associative, so the host
reference walks the identical tree. `examples/dot.lyth` at four elements gives 2.0 by the tree
and 1.0 by a left fold — the difference is not rounding noise.

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

## Not yet

Control flow in the body. Neighbours — no stencils, no convolution, nothing but the current
element. `max`, `min`, `prod`. More than one reduction per kernel. Types other than f32.
Atomics. Two dimensions. More than one kernel per file. A grid-stride loop, so **no performance
number comes out of this compiler yet** and none is claimed.
