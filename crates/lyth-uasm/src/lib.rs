//! LYTH to Unibit assembly. ADR-0025 steps 2 to 4.
//!
//! The other back end emits a kernel. **This one emits a program**: `.data`, `.text`,
//! `_start`, and a `halt`. That difference is the whole reason the target exists — a `.cu`
//! file is not a program and neither is a `.lyth` one, because a GPU has no entry point and
//! something on a CPU must launch it. Unibit does have one, so a `.lyth` file compiled for it
//! is something you run.
//!
//! ## The mapping
//!
//! | LYTH | Unibit |
//! |---|---|
//! | `space i : n` | a bounded loop, eight elements an iteration |
//! | `space i, j : rows, cols` | the same loop over `rows * cols`, when nothing is permuted |
//! | `stream x : dram -> reg` | `LQ` — one 256-bit load is eight f32 |
//! | a scalar `f32` parameter | `VSPLAT.w`, broadcast once before the loop |
//! | `y = a * x + y` | `VFMA`, or `VFMUL` and `VFADD` |
//! | `stream y : ..., drain` | `SQ` |
//! | `reduce sum : reg -> dram` | a partial per lane, then `VFREDUCE` and one `SW` |
//!
//! Eight lanes an instruction is not a tiling choice the compiler made: it is the register.
//! `Reg256` is 256 bits and an f32 is 32, so the loop steps by eight because that is what one
//! load holds.
//!
//! ## What it refuses, and why each is a refusal rather than a lowering
//!
//! Everything outside that table is refused **by name**, because a back end that quietly lowers
//! what it does not understand produces output nobody can reason about. `tile` and `coarsen`
//! are refused *permanently* on this machine rather than pending: Unibit has no shared memory,
//! `fixtures/machine/unibit.json` names no `smem` level, and ADR-0022 made the ceiling a
//! minimum over the levels a machine names. A kernel that stages where there is nothing to
//! stage into is asking for a level that does not exist.
//!
//! A **permuted** rank-2 access is refused for a different reason, and a harder one: `SQ` is
//! the only vector store and it writes 32 contiguous bytes. There is no strided store, no
//! scatter and no lane extract, so `b[j, i] = a[i, j]` cannot be emitted here slowly either.

use std::fmt::Write;

use std::collections::BTreeMap;

use lyth_lang::ast::{BinOp, Level, ReduceOp, Ty};
use lyth_lang::program::{PrintRange, Program};
use lyth_lang::ir::{KernelIr, Op, RegId};

/// Elements one 256-bit register holds at `f32`. Not a tunable.
pub const LANES: u32 = 8;

/// How a kernel that splits walks its buffers on this machine (ADR-0028).
///
/// A view is runs of `w` elements with a gap of `w`. One `LQ` reads eight **contiguous** f32, so
/// when `w` is a multiple of eight a run is whole registers and the loop is the ordinary one
/// with a jump over the other block at the end of each run. When it is not -- the low qubits,
/// `w = 1, 2, 4` -- no register load is contiguous, and the only correct lowering is one element
/// at a time through lane 0: `LW`, the same packed `VF*` instructions (whose other seven lanes
/// hold junk that nothing reads), `SW`. That is bit-exact and it is **eight times the
/// instructions** the cost model derived for bytes: `[KNOWN_LIMIT]`, stated rather than hidden.
/// The derived traffic and intensity hold; the ceiling ADR-0022 computes for this machine does
/// not, at those widths.
///
/// **A loop nest (ADR-0029).** At depth 1 a view is `run = w` contiguous elements, then a hop of
/// `w`, `pairs / w` times. At depth 2 -- outer width `A` in the buffer, inner width `B` in the
/// parent view -- it is a third level: when `B < A` (and `2B | A`) runs of `B`, hop `B`,
/// `A / 2B` times, then hop `A`; when `B >= A` (and `A | B`) runs of `A`, hop `A`, `B / A` times,
/// then hop `2B`. Every leaf walks the same nest from its own first element, `base_index(leaf,
/// 0)`. Anything else -- widths whose runs do not tile -- is refused with the arithmetic rather
/// than walked with a division per element.
struct SplitPlan {
    /// Contiguous elements before the first hop.
    run: u32,
    /// (iterations, hop in elements after them), innermost first.
    levels: Vec<(u32, u32)>,
    /// Whole-register loads and stores, or one f32 at a time.
    vector: bool,
}

fn split_plan(ir: &KernelIr, p: &Program) -> Result<Option<SplitPlan>, EmitError> {
    if ir.views.is_empty() {
        return Ok(None);
    }
    // One width per level: a loop nest has one counter per level, shared by every leaf.
    let depth = ir.walk_depth().max(1);
    let mut per_level: Vec<Vec<&str>> = vec![Vec::new(); depth as usize];
    for v in &ir.views {
        let lvl = ir.depth(&v.name) as usize - 1;
        if !per_level[lvl].contains(&v.width.as_str()) {
            per_level[lvl].push(v.width.as_str());
        }
    }
    let mut width = Vec::new();
    for (lvl, names) in per_level.iter().enumerate() {
        if names.len() != 1 {
            return Err(refuse(format!(
                "kernel `{}` splits at more than one width at depth {} ({}). This back end walks \
                 every buffer with one loop nest, so it takes one width per level.",
                ir.name,
                lvl + 1,
                names.join(", ")
            )));
        }
        let Some(&w) = p.extents.get(names[0]) else {
            return Err(refuse(format!(
                "kernel `{}` splits at width `{}` and this launch gives none. A kernel that splits \
                 needs its width from a `main`: `emit_program` has no way to invent one.",
                ir.name, names[0]
            )));
        };
        width.push((names[0], w));
    }
    // The launch check, not a second copy of it: every level covers its parent, and the walk is
    // the buffer over 2^depth.
    let walk = ir.split_pairs(&p.extents).map_err(|e| refuse(e.to_string()))?;
    if walk != Some(p.n) {
        return Err(refuse(format!(
            "this launch walks {} elements, but a split kernel at depth {depth} walks {walk:?}",
            p.n
        )));
    }
    let len = p.n << depth;
    let (run, levels) = match width[..] {
        [(_, w)] => (w, vec![(p.n / w, w)]),
        [(an, a), (bn, b)] if b < a => {
            if a % (2 * b) != 0 {
                return Err(refuse(format!(
                    "`{bn} = {b}` runs inside `{an} = {a}` blocks only when 2 * {bn} = {} divides \
                     {a}, but {a} mod {} = {}. The other back ends address this with a division per \
                     element; this one walks a loop nest and refuses it.",
                    2 * b,
                    2 * b,
                    a % (2 * b)
                )));
            }
            (b, vec![(a / (2 * b), b), (len / (2 * a), a)])
        }
        [(an, a), (bn, b)] => {
            if b % a != 0 {
                return Err(refuse(format!(
                    "`{an} = {a}` runs tile `{bn} = {b}` only when {a} divides {b}, but {b} mod {a} \
                     = {}. The other back ends address this with a division per element; this one \
                     walks a loop nest and refuses it.",
                    b % a
                )));
            }
            (a, vec![(b / a, a), (len / (4 * b), 2 * b)])
        }
        _ => {
            return Err(refuse(format!(
                "kernel `{}` splits to depth {depth}. This back end walks depth 1 and 2 with a hand-derived                  nest; deeper nests are derived from `base_index` in ADR-0030 step 4, and until then                  they are refused rather than guessed.",
                ir.name
            )))
        }
    };
    Ok(Some(SplitPlan { run, levels, vector: run % LANES == 0 }))
}

/// The counter and the label of each level of a split's loop nest, innermost first.
const LEVEL_COUNTERS: [&str; 2] = ["s9", "s10"];
const LEVEL_LABELS: [&str; 2] = ["outer", "outer2"];

/// The value `lyth run` gives a scalar that was not `--set`, so the emitted program and the
/// host oracle agree about it. Duplicated from `main.rs` and asserted equal in the tests,
/// which is the honest form of a constant that has to live in two crates.
pub const DEFAULT_SCALAR: f32 = 2.0;

#[derive(Debug, thiserror::Error)]
#[error("{0}")]
pub struct EmitError(pub String);

fn refuse(msg: impl Into<String>) -> EmitError {
    EmitError(msg.into())
}

/// Hand out a free register from the loop body's pool.
fn take(free: &mut Vec<usize>) -> Result<usize, EmitError> {
    free.pop().ok_or_else(|| {
        refuse(
            "the body needs more values live at the same moment than this back end has \
             registers for, and it does not spill.\n  The pool is `t0..t6`, `a0..a7` and every \
             `s1..s7` no buffer base pointer holds; the rest of the register file is `zero`, \
             `ra`, `sp`, `gp`, `tp`, the base pointers and the loop counters, so reaching \
             further means spilling rather than renaming.",
        )
    })
}

/// Registers the loop body may use as temporaries.
///
/// `t0..t6` are the named temporaries. `a0..a7` are argument registers, and they are safe here
/// for a reason worth writing down rather than assuming: this back end issues **no `ecall`
/// inside a loop**. The only syscalls are in the print epilogue, which runs after every loop
/// has finished, so nothing in the body can be clobbered by one. Anything that later emits a
/// syscall mid-loop has to shorten this list.
///
/// `s1..s7` are offered when they are **not** a buffer's base pointer: buffer `i` lives in
/// `s{i}`, so a kernel with four buffers leaves `s4..s7` idle for the whole loop. They were left
/// out until `gate_q` (eight scalars, four loads, outputs held to the store) needed more than 15
/// at once and was refused -- the shape of the allocator that never freed: the machine had the
/// registers and the pool did not offer them. `s8`, `s9` are the loop counters and `s10`, `s11`
/// the epilogue's; none of those is offered. Every register is 256 bits (`Reg256`), so any of
/// them holds a vector.
const TEMPS: [&str; 22] = [
    "t0", "t1", "t2", "t3", "t4", "t5", "t6", //
    "a0", "a1", "a2", "a3", "a4", "a5", "a6", "a7", //
    "s1", "s2", "s3", "s4", "s5", "s6", "s7",
];

/// The pool for a kernel with `buffers` base pointers: `s{i}` for `i < buffers` is taken. In
/// the order `take` hands them out (t, then a, then s), so short-lived values stay in the
/// registers they always had and only a body that needs more reaches the saved ones.
fn pool(buffers: usize) -> Vec<usize> {
    (0..TEMPS.len())
        .filter(|&i| match TEMPS[i].strip_prefix('s') {
            Some(k) => k.parse::<usize>().is_ok_and(|k| k >= buffers),
            None => true,
        })
        .rev()
        .collect()
}


fn tname(i: usize) -> String {
    TEMPS[i].to_string()
}

/// Record which temporary holds a value, so its last read can hand the register back.
fn own(owner: &mut BTreeMap<RegId, usize>, dst: RegId, i: usize) -> String {
    owner.insert(dst, i);
    tname(i)
}

/// The registers an op reads. Its `dst` is not among them, which is what makes it safe to free
/// an operand only after the whole instruction has been written.
fn op_reads(op: &Op) -> Vec<RegId> {
    match op {
        Op::Load { .. } | Op::Param { .. } | Op::Const { .. } => Vec::new(),
        Op::Bin { lhs, rhs, .. } => vec![*lhs, *rhs],
        Op::Fma { a, b, c, .. } => vec![*a, *b, *c],
        Op::Neg { src, .. } => vec![*src],
        Op::Zipper2 { acc, ket, bra, .. } => vec![*acc, *ket, *bra],
    }
}

/// Emit a complete, runnable Unibit program for this kernel at `n` elements, printing every
/// result it produces.
///
/// The `main`-less shape: what a kernel would print if the file did not say. `lyth build` on a
/// source with a `main` goes through [`emit`] instead, and both end in the same code, because
/// a second way to describe one launch is how ADR-0019's two descriptions of a grid drifted.
pub fn emit_program(ir: &KernelIr, n: u32) -> Result<String, EmitError> {
    let mut scalars = BTreeMap::new();
    for p in &ir.params {
        if p.ty == Ty::F32 {
            scalars.insert(p.name.clone(), DEFAULT_SCALAR);
        }
    }
    // Everything the kernel writes: the drained buffers, or the reduction's one value.
    let prints = match &ir.reduction {
        Some(r) => vec![PrintRange {
            buffer: r.into.clone(),
            lo: 0,
            hi: 1,
        }],
        None => ir
            .drains
            .iter()
            .map(|(b, _)| PrintRange {
                buffer: b.clone(),
                lo: 0,
                hi: n,
            })
            .collect(),
    };
    emit(
        ir,
        &Program {
            n,
            extents: BTreeMap::new(),
            scalars,
            prints,
            buffers: BTreeMap::new(),
        },
    )
}

/// Emit the program a resolved `main` describes.
///
/// The inputs are baked into `.data` by `lyth_lang::inputs` — the **same** generator the host
/// oracle uses. A program computing on different numbers from the oracle would make every
/// comparison between them a comparison of two functions.
pub fn emit(ir: &KernelIr, p: &Program) -> Result<String, EmitError> {
    let n = p.n;
    let plan = split_plan(ir, p)?;
    check_scope(ir, n, plan.is_some())?;

    let mut out = String::new();
    let _ = writeln!(out, "; Generated by LYTH from kernel `{}`.", ir.name);
    let _ = writeln!(
        out,
        "; ADR-0025 step 4: n = {n} elements, {LANES} per register, {} iterations.",
        n / LANES
    );
    let _ = writeln!(
        out,
        "; Inputs are {}.",
        if p.buffers.is_empty() {
            "lyth_lang::inputs, the generator the host oracle uses"
        } else {
            "the caller's buffers; the host oracle must be given the same bytes"
        }
    );
    // The contract, in the file the machine runs. A program that carries the intensity it was
    // checked against can be read against its own counters later without a second artifact.
    if let Some(b) = ir.cost.bytes_per_element() {
        let _ = writeln!(
            out,
            "; Contract: {:.4} flop/byte ({} flop / {b} byte per element).",
            ir.cost.intensity_at(1),
            ir.cost.flops_expr().unwrap_or_default()
        );
    }
    let _ = writeln!(out);
    emit_data(ir, p, &mut out)?;
    emit_text(ir, p, plan.as_ref(), &mut out)?;
    Ok(out)
}

/// Everything this back end does not do, said by name.
fn check_scope(ir: &KernelIr, n: u32, splits: bool) -> Result<(), EmitError> {
    if ir.tile.is_some() {
        return Err(refuse(format!(
            "kernel `{}` declares a tile, and `unibit` has no shared memory to stage into.\n  \
             A tile buys reuse by putting a block's worth of elements somewhere every thread \
             can reach; this machine names no `smem` level, so there is nowhere,\n  and the \
             tile would be a declaration that changes nothing. Drop it for this target.",
            ir.name
        )));
    }
    if ir.contract.is_some() {
        return Err(refuse(format!(
            "kernel `{}` contracts, and this back end emits no contraction.\n  Every kernel \
             here walks its space once with one value per element; a contraction needs an \
             inner loop over the contracted axis and a horizontal fold per output,\n  which is \
             a second loop nest and is not built. Not scheduled either -- said plainly rather \
             than deferred to a step that does not exist.",
            ir.name
        )));
    }
    if let Some(r) = &ir.reduction {
        // The path is the machine's answer, and this machine's is the short one: a tree over
        // the eight lanes of one register is staged nowhere, because `VFREDUCE` folds it in
        // place. ADR-0025 step 3 opened the language to both shapes for exactly this.
        if r.path.contains(&Level::Smem) {
            return Err(refuse(format!(
                "kernel `{}` reduces through `smem`, and `unibit` has no shared level.\n  Here \
                 the parallelism is the eight lanes of one register and the tree is \
                 register-internal, so the path is `reg -> dram`.\n  Declare that for this \
                 target.",
                ir.name
            )));
        }
    }
    if let Some(space) = &ir.space {
        if space.vars.len() > 2 {
            return Err(refuse(format!(
                "kernel `{}` walks a rank-{} space; this back end emits rank 1 and rank 2.",
                ir.name,
                space.vars.len()
            )));
        }
        // A rank-2 space whose buffers are all read at `[i, j]` is a linear walk over
        // `rows * cols` elements, so it is the rank-1 loop with a larger bound and the emitter
        // never needs the two extents apart. That is why `n` stays one number here.
        //
        // A **permuted** one is not, and cannot be emitted on this machine at all: `b[j, i]`
        // puts consecutive `j` at a stride of `rows` in `b`, and the only vector store is
        // `SQ`, which writes 32 contiguous bytes. There is no strided store, no scatter, and
        // no lane extract. `LW`/`SW` could copy it a word at a time, and that is refused too:
        // it moves 4 bytes per instruction where the cost model derived 32, so the ceiling
        // ADR-0022 computes would be eight times the machine.
        if let Some(st) = ir
            .streams
            .iter()
            .find(|st| !st.index.is_empty() && st.index != space.vars)
        {
            return Err(refuse(format!(
                "`{}` is accessed at `[{}]` where the space is `[{}]`, and `unibit` has no \
                 strided or scattered vector store.\n  `SQ` writes eight contiguous f32 and \
                 there is no lane extract, so a permuted axis cannot be emitted -- not slowly, \
                 at all.\n  Refused rather than lowered to a word-at-a-time loop, which would \
                 move 4 bytes per instruction where the cost model derived 32.",
                st.buffer,
                st.index.join(", "),
                space.vars.join(", ")
            )));
        }
    }
    // A split kernel walks pairs and chooses whole registers or single f32s by its width, so
    // the ragged-n refusal belongs to the ordinary walk only.
    if !splits && !n.is_multiple_of(LANES) {
        return Err(refuse(format!(
            "n = {n} is not a multiple of {LANES}.\n  One 256-bit register holds {LANES} f32 \
             and this back end emits no tail loop, so a ragged n would silently compute fewer \
             elements than asked for.\n  Refused rather than rounded."
        )));
    }
    if let Some(p) = ir
        .params
        .iter()
        .find(|p| p.ty == Ty::BufF16 || p.ty == Ty::BufBF16)
    {
        return Err(refuse(format!(
            "`{}` is a narrow buffer and `unibit`'s float unit is packed single precision only \
             (ADR-0025 step 0): there is no `cvt.f32.f16`.\n  Refused rather than emitted as \
             f32, which would move twice the bytes the cost model derived.",
            p.name
        )));
    }
    Ok(())
}

fn emit_data(ir: &KernelIr, prog: &Program, out: &mut String) -> Result<(), EmitError> {
    // A split kernel walks `n` pairs and its buffers are the whole vectors (ADR-0028).
    let n = if ir.views.is_empty() { prog.n } else { prog.n << ir.walk_depth().max(1) };
    let _ = writeln!(out, "        .data");
    for p in &ir.params {
        if !p.ty.is_buffer() {
            continue;
        }
        let values = match prog.buffers.get(&p.name) {
            Some(given) if given.len() == n as usize => given.clone(),
            Some(given) => {
                return Err(refuse(format!(
                    "buffer `{}` has {} elements and this program runs {n}",
                    p.name,
                    given.len()
                )))
            }
            None => lyth_lang::inputs::buffer(&p.name, p.ty, n),
        };
        let words: Vec<String> = values
            .iter()
            .map(|v| format!("0x{:08X}", v.to_bits()))
            .collect();
        // One `.word` directive per buffer, never several. Two directives under one label do
        // not land contiguously in this assembler, and `LQ` then reads the end of one buffer
        // followed by the start of whatever sits next -- which assembles, runs, and prints
        // numbers. Found writing `Unibit/programs/saxpy_f32.uasm`.
        let _ = writeln!(out, "{:<8}.word {}", format!("{}:", p.name), words.join(", "));
    }
    // Labels last, and that is placement rather than taste. `.asciiz` advances the data
    // pointer by a byte count that is not a multiple of 32, so a string emitted before the
    // buffers would move every one of them off a register boundary. Put after, they cannot.
    //
    // `PRINT_STRZ` rather than `PRINT_STR`: the latter takes a hand-counted length, and a
    // length counted by hand is one that drifts from the bytes it describes.
    for (i, pr) in prog.prints.iter().enumerate() {
        let _ = writeln!(
            out,
            "{:<8}.asciiz \"{}[{}:{}]\\n\"",
            format!("lbl{i}:"),
            pr.buffer,
            pr.lo,
            pr.hi
        );
    }
    let _ = writeln!(out);
    Ok(())
}

fn emit_text(ir: &KernelIr, prog: &Program, plan: Option<&SplitPlan>, out: &mut String) -> Result<(), EmitError> {
    let n = prog.n;
    let _ = writeln!(out, "        .text");
    let _ = writeln!(out, "        .global _start");
    let _ = writeln!(out, "_start:");

    // Buffers the loop *walks*. A reduction's target is not one of them: it receives one value
    // per program, so giving it a base pointer that advances 32 bytes an iteration would march
    // it off the end of a buffer declared `[f32; blocks]`.
    let reduce_into = ir.reduction.as_ref().map(|r| r.into.as_str());
    let buffers: Vec<&str> = ir
        .params
        .iter()
        .filter(|p| p.ty.is_buffer())
        .map(|p| p.name.as_str())
        .filter(|b| Some(*b) != reduce_into)
        .collect();
    if buffers.len() > 8 {
        return Err(refuse(format!(
            "kernel `{}` has {} buffers; this back end keeps one base pointer per saved \
             register, s0 through s7.",
            ir.name,
            buffers.len()
        )));
    }
    for (i, b) in buffers.iter().enumerate() {
        let _ = writeln!(out, "        la      s{i}, {b}");
    }

    // Where each value is read for the last time, so its register can be handed back.
    //
    // This used to allocate monotonically and never free, and the refusal it produced said "a
    // longer expression is a real limit here, not a bug" -- which was **wrong about its own
    // cause**. A complex multiply over separate real and imaginary buffers never has more than
    // seven values live at once; it makes nine allocations. The machine has thirty-two
    // registers. The limit was the allocator's and the message blamed the ISA.
    let last_use = {
        let mut m: BTreeMap<RegId, usize> = BTreeMap::new();
        for (i, op) in ir.ops.iter().enumerate() {
            for r in op_reads(op) {
                m.insert(r, i);
            }
        }
        m
    };
    // Values the loop's tail still needs after the last op: what is stored, and what the
    // reduction folds. Freeing these would hand out a register holding a result.
    let live_to_end: Vec<RegId> = ir
        .drains
        .iter()
        .map(|(_, r)| *r)
        .chain(ir.reduction.iter().map(|r| r.value))
        .collect();

    let mut free: Vec<usize> = pool(buffers.len());
    // Which temporary holds each body value, so it can be handed back at its last read.
    // `Op::Param` is absent on purpose: a scalar's register is broadcast before the loop and
    // lives for the whole of it, so it is not the body's to free.
    let mut owner: BTreeMap<RegId, usize> = BTreeMap::new();

    // Scalars, broadcast once before the loop. `VSPLAT.w` copies the low 32 bits into all
    // eight lanes, and an f32 is bits, so the integer instruction serves a float unchanged.
    let mut scalar_reg: Vec<(String, String)> = Vec::new();
    for p in &ir.params {
        if p.ty != Ty::F32 {
            continue;
        }
        // From the program, not from a default. A `main` that says `a = 2.0` and a back end
        // that emits its own 2.0 agree until the day the file says 3.
        let v = *prog.scalars.get(&p.name).ok_or_else(|| {
            refuse(format!(
                "`{}` has no value. A scalar the program does not give is a number this file \
                 would have to invent.",
                p.name
            ))
        })?;
        let r = tname(take(&mut free)?);
        let _ = writeln!(
            out,
            "        li      {r}, 0x{:08X}   ; {} = {v}",
            v.to_bits(),
            p.name
        );
        let _ = writeln!(out, "        vsplat.w {r}, {r}");
        scalar_reg.push((p.name.clone(), r));
    }

    // The reduction's accumulator: one running partial **per lane**, live across the whole
    // loop and folded once at the end.
    //
    // That is not a shape this back end chose. It is what `eval_with_launch(ir, n, ., 1, 8)`
    // already describes: grid 1, block 8, so thread `t` folds elements `t, t+8, t+16, ...` in
    // sequence and a tree combines the eight. Lane `l` of this register is thread `t = l`, and
    // the stride is `grid * block = 8` because that is how many f32 one `LQ` brings in. The
    // host oracle for this machine needed no new code, which is the strongest evidence that
    // the launch shape of a Unibit program is a real launch shape and not a metaphor.
    let acc = match &ir.reduction {
        Some(r) => {
            let a = tname(take(&mut free)?);
            let _ = writeln!(
                out,
                "        li      {a}, 0x{:08X}   ; {} identity",
                r.op.identity().to_bits(),
                r.op.name()
            );
            let _ = writeln!(out, "        vsplat.w {a}, {a}");
            Some(a)
        }
        None => None,
    };

    // What one loop iteration moves: a register, or (for the low widths of a split) one f32.
    let (load, store, step): (&str, &str, u32) = match plan {
        Some(sp) if !sp.vector => ("lw", "sw", 4),
        _ => ("lq", "sq", 32),
    };
    // Where a streamed name lives: its buffer's pointer, and the byte offset of the view's first
    // element -- `base_index(view, 0)`, at any depth.
    let slot = |name: &str| -> (usize, u64) {
        match (ir.view(name), plan) {
            (Some(_), Some(_)) => (
                buffers.iter().position(|b| *b == ir.root_of(name)).expect("a view's root is a parameter"),
                ir.base_index(name, &prog.extents, 0).expect("split_plan checked every width") * 4,
            ),
            _ => (
                buffers.iter().position(|b| b == &name).expect("a streamed buffer is a parameter"),
                0,
            ),
        }
    };
    match plan {
        None => {
            let iters = n / LANES;
            let _ = writeln!(out, "        li      s8, {iters}");
        }
        Some(sp) => {
            // Outermost level first; `s9` is the level around the run, `s10` the one around it.
            // `s10` is the epilogue's too, and the epilogue runs after every loop has finished.
            for (depth, (count, _)) in sp.levels.iter().enumerate().rev() {
                let _ = writeln!(out, "        li      {}, {count}", LEVEL_COUNTERS[depth]);
                let _ = writeln!(out, "{}:", LEVEL_LABELS[depth]);
            }
            // Runs of `run` elements: `run / 8` register iterations, or `run` single ones.
            let inner = if sp.vector { sp.run / LANES } else { sp.run };
            let _ = writeln!(out, "        li      s8, {inner}   ; runs of {} elements", sp.run);
        }
    }
    let _ = writeln!(out, "loop:");

    let mut regs: Vec<(RegId, String)> = Vec::new();
    fn get(regs: &[(RegId, String)], id: RegId) -> String {
        regs.iter()
            .find(|(i, _)| *i == id)
            .map(|(_, r)| r.clone())
            .expect("SSA: every use follows its definition")
    }

    for (i, op) in ir.ops.iter().enumerate() {
        match op {
            Op::Load { dst, buffer } => {
                let (i, off) = slot(buffer);
                let r = own(&mut owner, *dst, take(&mut free)?);
                let _ = writeln!(out, "        {load:<7} {r}, {off}(s{i})");
                regs.push((*dst, r));
            }
            Op::Param { dst, name } => {
                let r = scalar_reg
                    .iter()
                    .find(|(n, _)| n == name)
                    .map(|(_, r)| r.clone())
                    .ok_or_else(|| refuse(format!("`{name}` is not a broadcast scalar")))?;
                regs.push((*dst, r));
            }
            Op::Const { dst, value } => {
                let r = own(&mut owner, *dst, take(&mut free)?);
                let _ = writeln!(
                    out,
                    "        li      {r}, 0x{:08X}",
                    (*value as f32).to_bits()
                );
                let _ = writeln!(out, "        vsplat.w {r}, {r}");
                regs.push((*dst, r));
            }
            Op::Bin { dst, op, lhs, rhs } => {
                let m = match op {
                    BinOp::Add => "vfadd",
                    BinOp::Sub => "vfsub",
                    BinOp::Mul => "vfmul",
                    // The one form of choosing this machine can do without a branch, and the
                    // reason `max` is allowed in a body at all (ADR-0027).
                    BinOp::Max => "vfmax",
                    BinOp::Min => "vfmin",
                    BinOp::Div => {
                        return Err(refuse(
                            "this machine has no packed float divide. ADR-0025 step 0 added \
                             add, sub, mul, fma, max and min, and a divide is not among them.",
                        ))
                    }
                };
                let (l, r2) = (get(&regs, *lhs), get(&regs, *rhs));
                let r = own(&mut owner, *dst, take(&mut free)?);
                let _ = writeln!(out, "        {m}   {r}, {l}, {r2}");
                regs.push((*dst, r));
            }
            Op::Fma { dst, a, b, c } => {
                // `VFMA` accumulates into `rd`, so the addend has to be there before the
                // instruction runs. Copying rather than writing into `c`'s register keeps SSA
                // intact: `c` may be read again later in the body.
                let (ra, rb, rc) = (get(&regs, *a), get(&regs, *b), get(&regs, *c));
                let r = own(&mut owner, *dst, take(&mut free)?);
                let _ = writeln!(out, "        mv      {r}, {rc}");
                let _ = writeln!(out, "        vfma    {r}, {ra}, {rb}");
                regs.push((*dst, r));
            }
            Op::Neg { dst, src } => {
                // No packed float negate either. `0 - x` is exact, and it is said out loud
                // because a reader counting instructions should know where the extra two came
                // from.
                let s = get(&regs, *src);
                let zi = take(&mut free)?;
                let z = tname(zi);
                let r = own(&mut owner, *dst, take(&mut free)?);
                let _ = writeln!(out, "        li      {z}, 0");
                let _ = writeln!(out, "        vsplat.w {z}, {z}");
                let _ = writeln!(out, "        vfsub   {r}, {z}, {s}");
                // The zero lives for exactly one instruction and has no `RegId` to be the last
                // reader of, so it is handed back here rather than by the sweep below.
                free.push(zi);
                regs.push((*dst, r));
            }
            Op::Zipper2 { dst, acc, ket, bra } => {
                // `zipper2` reads `rd` as the incoming transfer matrix. Copy the
                // accumulator in first; `mv` moves the whole 256-bit register.
                let (ra, rk, rb) = (get(&regs, *acc), get(&regs, *ket), get(&regs, *bra));
                let r = own(&mut owner, *dst, take(&mut free)?);
                let _ = writeln!(out, "        mv      {r}, {ra}");
                let _ = writeln!(out, "        zipper2 {r}, {rk}, {rb}");
                regs.push((*dst, r));
            }
        }

        // Hand back every register whose last read was this instruction.
        //
        // After the whole op is written, never during it: `vfsub rd, rs1, rs2` reads before it
        // writes, but freeing an operand first would let `rd` be given the same register, and
        // that only happens to be safe. It is safe here because nothing is live to reuse yet.
        for r in op_reads(op) {
            if last_use.get(&r) == Some(&i) && !live_to_end.contains(&r) {
                if let Some(t) = owner.remove(&r) {
                    free.push(t);
                }
            }
        }
    }

    // Fold this iteration's eight values into the eight running partials.
    //
    // `combine(acc, value)`, in that order, because `eval` writes
    // `acc = r.op.combine(acc, reduced[i])`. It makes no difference to `sum` and it makes one
    // to `max` and `min` when an operand is NaN, and an oracle that agrees for a reason the
    // operator happens to supply is one operator away from disagreeing silently (ADR-0013).
    if let (Some(r), Some(a)) = (&ir.reduction, &acc) {
        let m = match r.op {
            ReduceOp::Sum => "vfadd",
            ReduceOp::Max => "vfmax",
            ReduceOp::Min => "vfmin",
        };
        let v = get(&regs, r.value);
        let _ = writeln!(out, "        {m}   {a}, {a}, {v}");
    }

    for (buffer, src) in &ir.drains {
        let (i, off) = slot(buffer);
        let _ = writeln!(out, "        {store:<7} {}, {off}(s{i})", get(&regs, *src));
    }

    for (i, _) in buffers.iter().enumerate() {
        let _ = writeln!(out, "        addi    s{i}, s{i}, {step}");
    }
    let _ = writeln!(out, "        addi    s8, s8, -1");
    let _ = writeln!(out, "        bne     s8, zero, loop");
    // The end of a run: hop over the other block. After `w` elements the pointer has advanced
    // `4w` bytes and the next run starts `8w` past where this one did.
    if let Some(sp) = plan {
        for (depth, (_, hop)) in sp.levels.iter().enumerate() {
            for (i, _) in buffers.iter().enumerate() {
                let _ = writeln!(out, "        addi    s{i}, s{i}, {}", 4 * u64::from(*hop));
            }
            let c = LEVEL_COUNTERS[depth];
            let _ = writeln!(out, "        addi    {c}, {c}, -1");
            let _ = writeln!(out, "        bne     {c}, zero, {}", LEVEL_LABELS[depth]);
        }
    }

    // The tree, and then the one store that leaves. `VFREDUCE` is specified as stride 4, then
    // 2, then 1 -- the same tree `eval`'s `tree_reduce` walks -- because float addition is not
    // associative and a chain would give a different bit pattern over the same eight lanes.
    //
    // `SW` stores lane 0 alone, four bytes. A `SQ` here would write 32 and clobber seven
    // neighbours of a buffer that holds one result per block.
    if let (Some(r), Some(a)) = (&ir.reduction, &acc) {
        let _ = writeln!(out, "        vfreduce {a}, {a}");
        let _ = writeln!(out, "        la      s10, {}", r.into);
        let _ = writeln!(out, "        sw      {a}, 0(s10)");
    }

    // What leaves the machine.
    //
    // Read back from **memory**, element by element, rather than printed out of the register
    // that computed it: a register dump would pass even if the store had gone to the wrong
    // address, which is precisely the thing a store can get wrong.
    //
    // `PRINT_F32` and not `PRINT_REG256`. The dump is four 64-bit lanes of hex, in an order a
    // reader has to decode, and a program whose output nobody can read is not a program
    // anybody runs. `PRINT_F32` prints the shortest string that parses back to the same bits,
    // so this stays a bit-exact comparison as well as a legible one.
    for (i, pr) in prog.prints.iter().enumerate() {
        let count = pr.hi - pr.lo;
        if count == 0 {
            continue;
        }
        let _ = writeln!(out, "        la      a0, lbl{i}");
        let _ = writeln!(out, "        li      a7, 6          ; print_strz");
        let _ = writeln!(out, "        ecall");
        let _ = writeln!(out, "        la      s10, {}", pr.buffer);
        let _ = writeln!(out, "        addi    s10, s10, {}", pr.lo * 4);
        let _ = writeln!(out, "        li      s11, {count}");
        let _ = writeln!(out, "print{i}:");
        // `LW` sign-extends into 64 bits and the syscall truncates back to 32, so the sign of
        // the f32's top bit never reaches the value that is printed.
        let _ = writeln!(out, "        lw      a0, 0(s10)");
        let _ = writeln!(out, "        li      a7, 7          ; print_f32");
        let _ = writeln!(out, "        ecall");
        let _ = writeln!(out, "        li      a0, 10         ; newline");
        let _ = writeln!(out, "        li      a7, 2          ; print_char");
        let _ = writeln!(out, "        ecall");
        let _ = writeln!(out, "        addi    s10, s10, 4");
        let _ = writeln!(out, "        addi    s11, s11, -1");
        let _ = writeln!(out, "        bne     s11, zero, print{i}");
    }

    let _ = writeln!(out, "        halt");
    Ok(())
}

#[cfg(test)]
mod pool_tests {
    use super::*;

    #[test]
    fn a_base_pointer_is_never_offered_as_a_temporary() {
        for n in 0..=8 {
            let names: Vec<&str> = pool(n).into_iter().map(|i| TEMPS[i]).collect();
            for i in 0..n {
                assert!(!names.contains(&format!("s{i}").as_str()), "{n} buffers: s{i} offered");
            }
            // t0..t6, a0..a7, and the saved registers from s{max(n, 1)} to s7.
            assert_eq!(names.len(), 15 + (8 - n.max(1)), "{n} buffers");
            // `take` pops from the end, so the hand-out order is the reverse of the vector.
            let handed: Vec<&str> = names.iter().rev().take(15).copied().collect();
            assert_eq!(handed, TEMPS[..15], "t and a first, in their old order");
        }
    }
}
