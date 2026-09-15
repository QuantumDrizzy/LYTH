# ADR-0019 — The first use outside the compiler, and the two things it found

**Status:** Accepted
**Date:** 2026-09-15
**Depends on:** ADR-0016 (callable), ADR-0000 (why)

## What was done

A LYTH kernel was called from another repository, on memory this compiler did not allocate:
two kernels (`sumsq`, `scale`) L2-normalising a 16.7M-element `torch` CUDA tensor, checked
against `torch.nn.functional.normalize`. No LYTH dependency — the generated Python speaks to
the driver through `ctypes` and takes `data_ptr()` integers.

It worked on the first run: `max |diff| 1.164e-10`, `|y| = 0.99999994`. That is the result
this ADR is *not* about.

## The first defect: the CLI and the bindings disagreed about how to launch

The run printed `blocks 65,536` for a reduction over 2²⁴ elements — one element per thread.
Twenty minutes earlier that default had been measured at half the achievable bandwidth and
fixed (ADR-0000, "a conclusion has a domain"). The fix went into `cmd_run` and stopped there.
The manifest went on publishing the elementwise rule, and all three generators emitted it.

Measured from the binding, on the dogfood's own kernel and data:

| launch the binding published | grid | achieved |
|---|---|---|
| one element per thread | 65,536 | 205.4 GB/s |
| eight per thread | 8,192 | 398.3 GB/s |

**1.94×**, agreeing with the 213.69 / 419.68 measured through the CLI on `sum`.

So `lyth run kernel.lyth` and `from kernel import Kernel` launched the same PTX differently,
and only one of them was fast. Nothing could have caught it: the manifest tests asserted the
rule was *structured data* rather than a hard-coded sentence, which it was, and said nothing
about whether it was the same data `cmd_run` used.

The fix is not "update the manifest too". The rule now exists once —
`manifest::min_elements_per_thread` and `GridRule::{of, blocks}` — and `cmd_run` calls it
instead of holding its own copy. A test asserts the published rule evaluates to what the
compiler launches, and a second asserts all three generated languages divide by
`BLOCK * 8` for a reduction and by `BLOCK` for an elementwise kernel.

> **A fix belongs where the rule lives, not where the symptom appeared.**
>
> This is the third instance of one shape. `report_timing` assembled a launch that `cmd_run`
> had already verified; ADR-0012's saxpy conclusion was applied to reductions without being
> re-measured; and here a rule was corrected in one of two places that both claimed to state
> it. Every time, two pieces of code said the same thing and only one was right.

Found by running a binding in another repository. Not by a test, and not by review.

## The second defect: an integer is not a tensor

`launch` takes device pointers as integers, which is the property that makes the binding
dependency-free (ADR-0016, rule 1). It is also a hole. `torch.Tensor.data_ptr()` returns a
valid device pointer for a transposed view, a strided slice, or a `float64` tensor, and the
kernel addresses element `i` at `base + i*4`. Every one of those launches succeeds. The
arithmetic is performed on the wrong elements and the answer is plausible.

The generated Python now carries two guards:

**`check_device()`**, called by `Kernel()` before it loads the module. It compares the device's
compute capability against the PTX's target: refuse below (it cannot run), warn above (it runs,
JIT-compiled, but the contract — `0.5000 flop/byte`, `4 B/element` — was derived on another
machine), silent on a match. The warning is the point: ADR-0000 says performance is a quantity
the program determines times a price the machine sets, and a binding that carries the first
without naming the second is half a claim.

**`from_torch(t, count=None)`**, optional, the only function in the module that assumes
anything about its caller. It refuses a CPU tensor, a non-contiguous one, a dtype that is not
`float32`, and a buffer smaller than the launch will walk. It imports nothing: the checks are
attribute calls and a string compare, so the module still loads on a bare interpreter.

The dogfood exercises all four refusals and asserts each one fires, because a guard nobody has
seen fail is a guard nobody has tested.

## What this does not claim

One repository, two kernels, one device. It is evidence that the bindings are usable and that
using them finds things reading them did not; it is not evidence that they are correct in
general. The `from_torch` checks are about `float32` and contiguity because that is all this
language can currently express — a wider type system will need wider guards, and this file
should be read again then.

## Files

| | |
|---|---|
| `crates/lyth/src/manifest.rs` | `min_elements_per_thread`, `GridRule::{of, blocks}`, `extent_params` |
| `crates/lyth/src/main.rs` | `cmd_run` calls the rule instead of restating it |
| `crates/lyth/src/bind_py.rs` | `arch_of`, `check_device`, `from_torch` |
| `crates/lyth/src/bind_rust.rs`, `bind_c.rs` | the same grid rule, and the test that all three agree |
