# ADR-0000 — Why: information has to move, and the source code never says how much

**Status:** Accepted
**Date:** 2026-09-16
**Depended on by:** everything

Written seventeen ADRs late. The quantities this project derives were chosen correctly and for
reasons that were never written down, which is how a principle becomes a habit and then a
superstition. This is the principle.

## The premise

A computation has a **minimum amount of information that must move** between a fast memory and a
slow one. That minimum is a property of the algorithm and of the schedule that executes it,
together with the size of the fast memory — and it is not a property of the hardware at all. The
hardware decides only what it costs to move a bit.

This is not a metaphor. It is a theorem. Hong and Kung's red-blue pebble game (STOC 1981)
established lower bounds on data movement for a computation given a fast memory of size `M`; for
matrix multiplication the classical bound is `Ω(n³/√M)` words. Ballard, Demmel, Holtz and
Schwartz extended it across numerical linear algebra in 2011, and the communication-avoiding
line of work descends from it.

The observation that matters here is what the bound is *about*. It is not about how fast the
machine is. Two machines running the same schedule move the same information; they differ in
what that costs. **Performance is the product of a quantity the program determines and a price
the machine sets, and almost every source language in existence talks about neither.**

## What follows from it

Three things this project treats as fundamental turn out to be corollaries, which is the test of
whether a premise is load-bearing or ornamental.

**Arithmetic intensity is the right quantity, and not by convention.** `flops / bytes` is
operations per unit of unavoidable movement. The roofline is the practical shadow of an I/O lower
bound: the ridge is where the price of moving stops dominating the price of computing. ADR-0005
put intensity in the source; this is why it is intensity and not, say, a target runtime.

**A tile is not one optimisation among many.** Tiling is *the* mechanism by which the Hong-Kung
bound is approached: choosing a working set that fits the fast memory is exactly what the proof
is about. ADR-0017 is the practical side of a 1981 theorem, and the reason its claim is
"36 bytes per element becomes 8" rather than "it gets faster".

**Coalescence is an information efficiency, by definition.** `payload / sectors` is the bits
wanted over the bits moved. That is not an analogy borrowed to sound rigorous — it is what the
ratio is. Measured on this machine, an untiled transpose moves **4.5 bits for every bit it
needs** at the L1-to-L2 interface (ADR-0015). Saying it that way makes the number mean something
a percentage does not.

It also explains a result that looked like a curiosity. ADR-0014 found that grid-stride bought no
time while cutting instructions by a factor of two to twelve. Of course it did: instructions are
not the scarce resource. Movement is, and grid-stride moved exactly the same bytes.

## What this project is, in one sentence

**A compiler that makes the movement a program causes into a declared, checked property of the
program.**

Not a faster compiler. Not a scheduler. The claim is that the quantity which determines
performance is knowable from the source plus a declared schedule, and that a compiler which
knows it should refuse a program that lies about it — the way a type system refuses a program
that lies about its values.

## The honest limit, which has to be here or the rest is decoration

**LYTH does not compute lower bounds.** It derives the movement a declared schedule actually
causes, and checks that against silicon. The Hong-Kung bound is the horizon those numbers are
measured against; it is not something this compiler proves, and nothing in it is a proof of
optimality. A kernel that passes every check in this repository may still be far from the
minimum its problem admits.

Two further gaps, stated plainly:

The bounds are theorems about specific computations — matmul, FFT, stencils — and this language
cannot yet express any of them. The theory that justifies the quantities is, for now, ahead of
the language that uses them.

And a cost model derived from a *declared* schedule is not a cost model of the *best* schedule.
Choosing the schedule is what Halide and TVM do, by search. This project has no claim to do that
and should not acquire one by implication: it checks the schedule you wrote, which is a smaller
and different job.

## The rule that pre-registration needs, learned by breaking it twice

ADR-0017 pre-registered two claims and both were wrong. They were wrong in the same way, which
is what makes it a rule rather than two mistakes.

| claim as written | what the theory supported | what measured |
|---|---|---|
| the untiled control measures **36.00 ± 0.2** | 36 sectors, plus the read-for-ownership ADR-0015 had already published at that size | 38.97 |
| shared store conflicts are **0** in both variants | the skew changes the **loads**; the stores were never what it was about | 0 below 1024², a few thousand at 1024², unchanged by the skew |

Both times the reasoning was right and narrow, and the sentence written down was wider and
tidier. An absolute zero and an exact equality read better than "unchanged from nominal" and
"36 plus the excess we already measured", and neither was what the theory implied.

> **Pre-register what the theory implies, not what would look clean.**

A claim that is stronger than its derivation is not a bolder version of the same claim. It is a
different claim, one nobody derived, and when it fails it fails for reasons that have nothing to
do with the thing being tested — which is how a real result gets thrown out with a bad
prediction. Both of these were caught by measuring rather than by review, and review is where
they should have been caught, because the derivation was sitting in the same document.

## The other rule, learned four times

> **A conclusion has a domain, and the domain is the kernel it was measured on.**

ADR-0012 swept `saxpy` across launch shapes and found one element per thread faster than
filling the machine. That became *the* default, for every kernel. ADR-0014 then measured that
grid-stride cuts a reduction's instruction count by twelve and concluded it bought no time —
citing the saxpy sweep. Both conclusions were correct about the kernel in front of them. Applied
to `sum`, the default runs at **half the achievable bandwidth**, because a reduction's block
tree runs once per thread and at one element per thread that is once per element.

It is the same shape as the timing bug: `cmd_run` verified its launch, `report_timing` assumed
that verification and assembled its own shape, and nothing re-checked. Something measured in one
place was carried to another without being measured again.

So: **an ADR's conclusion carries the kernel it was measured on, and applying it elsewhere is a
new measurement, not an inference.** The fix for the grid default is written that way — one
element per thread for elementwise, grid-stride for reductions, each confirmed on both kernels
before either was changed — rather than as a special case for `sum`.

**Postscript, the same day.** That fix was applied to `cmd_run` and not to the manifest, so the
generated bindings went on publishing the launch shape that had just been measured at half the
bandwidth. `lyth run` and a caller of the binding launched the same PTX differently for twenty
minutes. It is the same shape a third time — two pieces of code stating one rule, one of them
right — and the correction is in ADR-0019: **a fix belongs where the rule lives, not where the
symptom appeared.** It was found by calling a binding from another repository, which is the
only place any of this has ever been caught.

**Postscript, a fourth time, and the sharpest version of it.** ADR-0024 pre-registers that a
half-precision matmul will not speed up, and a reviewer proposed borrowing ADR-0022's **±0.80%**
as the band for "will not". That figure is how closely the *traffic* model matched `ncu` —
derived bytes against counted bytes, two counters of one event. The matmul row is a **wall-clock
time**. The measured run-to-run reproducibility of that timing harness is **1.6%** across two
guarded nine-round runs, so the band is ±5%.

Importing a cheap quantity's uncertainty into an expensive measurement would have manufactured a
falsification out of ordinary noise — and since step 3 exists *in order to be able to* kill
ADR-0022, that was precisely the way to fail. So, generalised:

> **A pre-registered band for a claim about X is the measured reproducibility of measuring X,
> plus margin. Never a band borrowed from a claim about Y, even one in the same ADR.**

The value and its uncertainty travel together or neither travels. `fixtures/machine/sm_120.json`
already says the value half of this — *"mixing a datasheet peak with a measured bandwidth, or
the reverse, produces a number that describes nothing"* — and this is the uncertainty half.

**One thing this rule is not.** It is easy to file the grid-stride postscript above under the
same heading, and it does not belong there. Two pieces of code stating one rule and one of them
corrected is a **duplicated definition**, not a borrowed figure, and it has its own recurrence:
it happened again on 2026-09-16 in `rse-hpc-lab`, where `inst_matmul.py` kept a second copy of a
variant list and it drifted the moment two schedules were added. The two failures want different
fixes — *one definition* for that one, *one provenance* for this one — and merging them would
make both rules vaguer than either.

## Why this is written down late, and kept

Every ADR after this one derives a number and then measures it. The reason the numbers are bytes
and flops and sectors, rather than milliseconds, has been implicit in all of them. An implicit
reason is one that cannot be argued with, and the ones that cannot be argued with are the ones
that turn out to be wrong.
