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

## Why this is written down late, and kept

Every ADR after this one derives a number and then measures it. The reason the numbers are bytes
and flops and sectors, rather than milliseconds, has been implicit in all of them. An implicit
reason is one that cannot be argued with, and the ones that cannot be argued with are the ones
that turn out to be wrong.
