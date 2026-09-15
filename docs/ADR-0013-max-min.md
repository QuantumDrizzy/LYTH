# ADR-0013 — `max` and `min`, and whether a comparison is arithmetic

**Status:** Accepted
**Date:** 2026-09-15
**Depends on:** ADR-0011 (reductions), ADR-0005 (intensity)

## Why a second reduction operator is not just a second match arm

Mechanically it is. `ReduceOp` has four methods and four `match` sites, every one exhaustive, so
adding `Max` and `Min` is the kind of change the compiler walks you through.

What it is not is free of decisions. Two of them change a number this compiler claims to have
**derived**, and a derived number that is wrong is worse than a comment that is wrong, because
the language sells it as checked.

## Decision 1 — `max` and `min` retire zero FLOPs

`ReduceOp::flops` feeds exactly one thing:

```
intensity = flops_per_element / bytes_at_the_deepest_level
```

and `intensity` places the kernel against a roofline whose compute ceiling is a **measured FP32
peak**. That peak is an FMA number. `max.f32` is a compare-and-select: it is not counted in any
vendor's FLOP/s figure, and no published flop count for a reduction counts its comparisons.

This is the same rule ADR-0005 already applies to division, running the other way. The comment
on `BinOp::flops` says divide is charged as one *because that is what published counts do*, even
though the hardware spends several instructions on it. The rule is "match the published
convention", not "count instructions". Published convention does not count `max`.

So `sum` is worth 1.0 and `max` and `min` are worth 0.0, and a max-reduction over a plain stream
derives:

```
intensity 0.0000 flop/byte
```

That is not a degenerate case to be worked around. It is the result: **a max reduction cannot be
compute-bound at any problem size, on any machine.** No arithmetic ever enters it. The only
lever is bandwidth. A language whose thesis is that the cost model belongs in the type system
should be able to say that, and now it can.

### [KNOWN LIMIT] zero flops is not zero time

The combine still issues. Each of the B-1 tree steps occupies a slot on the same pipe an
`add.f32` would use, and `bar.sync` between rounds costs what it costs. `flops_per_element = 0`
means **no arithmetic**, not **no work**. Anyone reading the cost model as an instruction count
will be wrong here, and the field is not named `instructions_per_element` by accident.

The alternative was charging 1.0 and reporting `intensity 0.25`, identical to `sum`. It would
have changed no verdict — both are far below every ridge this machine has — which is precisely
why it was tempting and precisely why it is not worth doing. A number that changes no decision
still has to be true.

## Decision 2 — `max` and `min` are exactly associative, and `sum` is not

ADR-0011 records that a tree reduction's result depends on the shape of the tree, because
floating-point addition is not associative; the evaluator therefore reproduces the tree
**step for step** rather than summing a slice, and the two agree bit for bit only because of
that.

`max` and `min` select an operand. They do not round. For finite, non-NaN inputs the result of a
max-tree is the same bit pattern for every tree shape, every block size and every grid. The
evaluator would agree with the kernel even if it summed the slice in source order.

It reproduces the tree anyway. The evaluator is the oracle for what the GPU does, not a
convenient way to get the right answer, and an oracle that happens to agree for a reason the
operator supplies is one operator away from disagreeing silently. The `tree_reduce` shape is the
contract.

This is worth stating because it is the first place this language can make a claim that
distinguishes two operators by something other than their result: **one of these reductions is
reorderable and the other is not.** A compiler that knew this could legally fuse, split or
re-block a max-reduction and may not touch a sum-reduction. This one does not do that yet. The
property is recorded here so the future optimisation has a document to point at instead of an
assumption to make.

## Decision 3 — the identity is an infinity, and an empty block is correct

`Sum` contributes `0.0` from a thread with no element, which is what lets every thread reach
every barrier without an idle-thread branch (ADR-0011). `Max` contributes `-inf` and `Min`
contributes `+inf`, emitted by the existing `hex_f32` with no special case because it goes
through `f32::to_bits`.

A block whose threads all fall past `n` therefore writes `-inf` to its partial. That is the max
of an empty set and it is the right answer: a caller combining partials with `max` is unaffected
by it, which is the same property that makes `0.0` safe for `sum`.

## What the awkward values did, measured rather than assumed

This section was first written as an open question: PTX `max.f32` and Rust `f32::max` are
*believed* to agree on NaN, and `max(-0.0, +0.0)` is *believed* to be a place implementations
differ. Asserting either from memory is the move this project keeps catching in other people's
benchmarks, so neither was asserted. Both were then provoked, in the language itself, using the
bit-exact check as the instrument.

**NaN: they agree.** `x / (x - x)` is `+inf` or `-inf` by the sign of `x`, and NaN where `x` is
exactly zero, which the input generator produces 14 times in 65536 elements. Both infinities and
14 NaNs therefore meet inside one reduction. Host and device came out bit-identical, both
returning the non-NaN operand rather than propagating, and the reduction resolves to `+inf`.
Kept as `examples/not-a-number.lyth`.

**Signed zeros: 135 of 65536 partials differed**, and the cause is not the one the section
originally guessed at. `x * (x - x)` is `+0.0` where `x` is positive and `-0.0` where it is
negative, so both zeros meet in one reduction. The device returned `+0.0` for `max`; the host
returned `-0.0`.

The host was not "ordering signed zeros differently". It was not ordering them at all:

```
(-0.0f32).max(0.0)   folded at compile time  ->  +0.0
(-0.0f32).max(0.0)   executed at run time    ->  -0.0
```

`f32::max` lowers to `llvm.maxnum`, which is specified to return **either** operand when they
compare equal, and `-0.0 == 0.0` is true. The same expression has two answers and the optimiser
picks. A first check of this by hand got `+0.0` and nearly closed the investigation as a
non-finding, because it was written in a form the constant folder could see through.

PTX `max.f32` has no such freedom: `+0.0`, matching IEEE 754-2019 `maximumNumber`.

So `ReduceOp::combine` handles the both-zero case explicitly. The point is not that the hardware
outranks the host language. It is that **an oracle cannot be built on unspecified behaviour** —
an evaluator whose answer depends on how it was compiled is not a specification of anything.
Kept as `examples/signed-zero.lyth`.

### What this does to the bit-exactness claim

ADR-0010's claim is widened, with evidence rather than by assertion: `max` and `min` reductions
are verified bit-exact on finite inputs, on NaN, on both infinities and on both signed zeros.
The instrument was the language's own verification, which is the first time a LYTH program has
been used to find something out about the machine instead of about itself.

## An unrelated bug this work fell over, and what it was hiding

A script rewriting `examples/signed-zero.lyth` left it with CRLF line endings, and it stopped
parsing:

```
examples/signed-zero.lyth:19:5: expected `kernel` or the end of the file, found an indented block
```

The error is about the wrong thing entirely, which is the first problem. The second is larger.
`core.autocrlf` is `true` on this machine, so **git stores LF and hands out CRLF on checkout**.
The examples in the working tree are LF only because they were created here and never checked
out again. A fresh clone gets CRLF, and that was verified rather than reasoned about:

```
$ git clone <repo> && cd clone && cargo build --release -p lyth
$ ./target/release/lyth.exe run examples/sum.lyth --machine fixtures/machine/sm_120.json -n 4096
examples/sum.lyth:8:5: expected `kernel` or the end of the file, found an indented block
```

That is the compiler at `16c7003`, on its own example, one `git clone` old. **Every example in
the repository was unparseable for anyone who cloned it on Windows**, which is the platform it
is developed on, and none of the 96 tests caught it because they read files this machine wrote.

The fix is one line in `Lexer::new`: filter CR out of the character vector. A CR is not part of
any token and layout is measured in columns, so it cannot move a span -- it only ever appears
after the last token on a line. Two tests pin it, one comparing the token streams of the same
source in both line endings, spans included.

Two things worth keeping from this. A test suite built on fixtures the build machine wrote is
blind to everything the *checkout* does, and the way this surfaced -- a tool mangling a file --
is the second time a script of mine has damaged source in this project. The earlier one was a
regex that collapsed indentation inside Rust string literals. Both were found by tests, which is
the argument for the tests.

## Files

| | |
|---|---|
| `crates/lyth-lang/src/ast.rs` | the two variants and their four methods |
| `crates/lyth-lang/src/eval.rs` | the per-element fold and the tree combine |
| `crates/lyth-lang/src/lex.rs` | CR filtered out, and the CRLF bug above |
| `crates/lyth-ptx/src/lib.rs` | the same two, as `max.f32` / `min.f32` |
| `examples/max.lyth` | a reduction with no arithmetic in it at all |
| `examples/not-a-number.lyth` | the NaN probe, which came back clean |
| `examples/signed-zero.lyth` | the probe that did not, kept as its regression test |
| `crates/lyth/src/main.rs` | partials combined with the kernel's operator, not always `sum` |
| `docs/LANGUAGE.md` | the operator table, and the associativity note |
