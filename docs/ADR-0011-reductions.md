# ADR-0011 — Reductions, and the cost model growing a second level

**Status:** Accepted
**Date:** 2026-09-14
**Depends on:** ADR-0010 (executable LYTH)

## What this breaks

v1 rests on an assumption stated in ADR-0010: elementwise, one element per thread, no
cross-element communication. That is exactly why one thread per element is a legal schedule.

A reduction breaks it. Threads must cooperate, which forces four things v1 does not have:

1. **A second level of the hierarchy.** The partial sums live in shared memory. The cost model
   has one byte count today; it needs one per level.
2. **Locals.** `p = x * y` names a value that is not a buffer and is never stored.
3. **Threads past the end must participate.** Today they branch out. A thread that skips leaves
   garbage in the shared array, so it has to contribute the operation's identity instead.
4. **A result that is not per element.** One value per block, not one per thread.

## Decision: block partials, not a single number

The kernel produces **one value per block**. Finishing the reduction — summing `gridDim.x`
partials — is the caller's business.

The alternative is `atomicAdd` into a single float, which is shorter to write and **disqualified
here**: float addition is not associative, so the result depends on the order blocks happen to
finish, and it changes between runs. ADR-0010's verification is bit-exact equality against the
same IR evaluated on the host. Giving that up to save the caller one line would trade the only
check that can catch a code-generation bug for a convenience.

Block partials are deterministic, verifiable bit-for-bit, and what a real implementation does
anyway before its second pass.

## Surface

```
machine sm_120

kernel dot(n: u32, x: [f32], y: [f32], partial: [f32])
    intensity 0.25

    stream x : dram -> reg
    stream y : dram -> reg
    reduce sum p : reg -> smem -> dram into partial

    at reg:
        p = x * y
```

The `reduce` clause sits with the `stream` declarations, before the body, because it is a
movement declaration: it says where the value goes and through which level. `reg -> smem -> dram`
is written out rather than implied by the word `reduce`, for the same reason a stream writes its
path — the language does not infer movement, it checks that the arithmetic fits what was declared.

`p` is a **local**: a name that is not a parameter and is never stored to memory. A local that no
reduction consumes is refused, because it is work whose result is discarded.

## The cost model grows a level

Today `Cost` carries one `bytes_per_element`. A reduction moves bytes at `dram` **and** at
`smem`, and collapsing them into one number would either hide the shared traffic or corrupt the
roofline position, which is about DRAM.

`Cost` becomes per-level. The reported intensity stays the DRAM one — that is the roofline
number — and shared traffic is reported beside it rather than mixed into it.

For `dot` at a block of B threads:

```
dram   x 4 read + y 4 read                          = 8 bytes/element
dram   partial: 4 bytes per BLOCK                   = 4/B bytes/element, 0.016 at B=256
smem   the tree: one write in, then ~2B reads and ~B writes per block
flops  1 multiply per element + the tree's B-1 adds = ~2 flop/element
```

**[KNOWN LIMIT]** The tree contributes exactly `(B-1)/B` adds per element, and the model counts
1.0. At B = 256 that overstates flops by 0.4%. Counted as 1.0 because the block size is not in
the source, and recorded here rather than absorbed silently.

## What the back end must do differently

```
v = 0                      // the operation's identity
if idx < n:                // predicated, not a branch out
    v = <body>
smem[tid] = v
barrier
for stride = ntid/2; stride > 0; stride >>= 1:
    if tid < stride: smem[tid] += smem[tid + stride]
    barrier
if tid == 0:
    partial[blockIdx.x] = smem[0]
```

Two things are load-bearing. **Out-of-range threads contribute the identity** rather than
branching to the exit, or the shared array holds garbage that the tree then sums. And the
**host reference must walk the identical tree**: float addition is not associative, so summing
the block's elements in a different order gives a different last bit, and the bit-exact check
would fail on a correct kernel.

Shared memory is **dynamic** (`.extern .shared`), sized at launch, so the block size is not
baked into the generated module.

## Build order

| stage | ships | done when |
|---|---|---|
| 1 | `reduce` in lexer, AST, parser; locals | `dot.lyth` parses; a local nothing consumes is refused |
| 2 | per-level `Cost`; reduction lowering | derived cost names dram and smem separately |
| 3 | host reference walks the same tree | `eval` returns block partials |
| 4 | PTX: predicated body, smem tree, block store | `ptxas` accepts it |
| 5 | runner: partial buffer, dynamic smem | bit-exact against the host tree |

## What would falsify this

- **If the host tree and the device tree cannot be made bit-identical**, the verification method
  from ADR-0010 does not survive contact with reductions, and either the method or the design is
  wrong. Publish which.
- If expressing a reduction turns out to need syntax that cannot be written as an attribute on a
  Rust function, that is the **first evidence in this project** for the ADR-0001 parser, and
  `docs/DOGFOOD.md` gets its first entry on that side of the ledger.

## [KNOWN LIMIT], before the code

- One reduction per kernel. `sum` only; `max`, `min` and `prod` are the same shape and are not
  implemented until something needs them.
- The caller sums the partials. No second pass, no atomics, no single-value output.
- Block size is a launch parameter and does not appear in the source, so a kernel cannot state a
  block-size requirement and the tree's flop count is approximated as above.

---

## Result, 2026-09-14 — all five stages closed

```
$ lyth run examples/dot.lyth --machine fixtures/machine/sm_120.json -n 1048576
kernel dot on machine sm_120
  derived  0.2500 flop/byte  (2 flop / 8 byte per element)
  traffic  8 read + 0 written, at dram
  declared 0.25 — matches
  ridge    42.9 flop/byte — memory-bound
  device   NVIDIA GeForce RTX 5060 Ti
  launch   grid 4096 x block 256 over 1048576 elements, 1024 B shared
  verify   BIT-EXACT against the IR evaluated on the host, 1048576 elements
           4096 block partials in `partial`, summed in the same tree order
  reduced  5.564727e6  (the caller adds the partials; ADR-0011)
ok
```

Bit-exact at 1,048,576 / 1,000,000 / 257 / 1 elements. The last two are the cases that break a
reduction: 257 is two blocks with one element in the second, and 1 is 255 idle threads that must
reach every barrier carrying the identity.

**The reduced total is printed** because a bit-exact match between two buffers of zeros is not
evidence of anything, and the sum is the number the kernel was written to produce.

### The cost model now names two levels

```
dram   8 bytes/element      x and y
smem  16 bytes/element      the tree
dram   4 bytes/block        the partial, reported separately
```

The roofline stays the DRAM number. Folding 16 bytes of shared traffic into it would move the
kernel's reported position on a chart that is about the memory controller.

### The order is the contract

`examples/dot.lyth` over `[1e8, 1.0, -1e8, 1.0]`:

```
tree       (x0 + x2) + (x1 + x3)  = 2.0
left fold  ((x0 + x1) + x2) + x3  = 1.0
```

The large pair cancels first in the tree, so both ones survive; the left fold loses one into 1e8
before the cancellation happens. **A factor of two, not a last-bit difference.** This is why the
host reference walks the identical tree rather than summing the block, and it is frozen as a test
that fails if the two ever diverge.

### Found by running it

- The first launch died with `ILLEGAL_ADDRESS`. Cause: `launch_shared` was written but the call
  site still read `func.launch(...)`, so the `.extern .shared` array was sized 0 and every thread
  indexed into nothing. The driver reports this as a fault at synchronise, not at launch.
- `eval` demanded `len >= n` of every buffer parameter, which would have forced a caller to
  allocate `n` slots for `n / block` results. The reduction target is now sized by the grid.

## [KNOWN LIMIT], reviewed against the implementation

- Still `sum` only, one reduction per kernel, caller sums the partials. All three as written.
- The tree's flop count is one per element; the true figure is `(B-1)/B`, so at B = 256 this
  overstates by 0.4%. Shared traffic is counted as 16 bytes/element against a true `16 - 8/B`.
  Both because the block size is a launch parameter and does not appear in the source.
- **Shared memory is sized `block * 4` by the runner.** A kernel cannot state a block-size
  requirement, so a caller launching with a larger block than the shared allocation would read
  past the array. `lyth run` is the only caller today and it sizes them together.

## The ADR-0001 parser gate

ADR-0011 named a falsification: if a reduction needed syntax that could not be an attribute on a
Rust function, that would be the first evidence for the parser.

**It did not.** `reduce sum p : reg -> smem -> dram into partial` is five values — an operation,
a source name, a path, a target — and an attribute macro could carry all five. The work was in
the cost model growing a level, in the host reference matching the tree, and in the back end;
none of it is front-end shape. `docs/DOGFOOD.md` gains no entry on the parser's side.
