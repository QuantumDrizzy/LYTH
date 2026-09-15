//! LYTH IR to PTX text.
//!
//! No dependency on CUDA, on purpose: emission stays testable on a machine with no GPU, and
//! validating the result with `ptxas` is a separate, optional step.
//!
//! One element per thread, bounds-checked. There is no grid-stride loop in v1, so nothing here
//! is tuned for occupancy and no performance claim comes out of it. **This back end exists to
//! be correct and to move exactly the bytes the front end derived.**

use std::fmt::Write as _;

use lyth_lang::ast::Ty;
use lyth_lang::ir::{KernelIr, Op, RegId};
use lyth_lang::BinOp;

/// The PTX ISA version each target first became legal in.
///
/// This is not a constant, because it is a property of the machine. `.version 8.5` with
/// `.target sm_120` is rejected outright by ptxas — found by generating exactly that and
/// reading the JIT log — so the version has to follow the target the machine file names.
/// A too-low version fails loudly; a too-high one fails on an older driver. Both are worse
/// than a table that says what is known.
const TARGET_ISA: &[(&str, &str)] = &[
    ("sm_120", "8.7"),
    ("sm_100", "8.7"),
    ("sm_90", "7.8"),
    ("sm_89", "7.8"),
    ("sm_86", "7.1"),
    ("sm_80", "7.0"),
    ("sm_75", "6.3"),
];

fn isa_for(arch: &str) -> Result<&'static str, EmitError> {
    TARGET_ISA
        .iter()
        .find(|(t, _)| *t == arch)
        .map(|(_, v)| *v)
        .ok_or_else(|| {
            let known: Vec<&str> = TARGET_ISA.iter().map(|(t, _)| *t).collect();
            EmitError::UnknownTarget(arch.to_string(), known.join(", "))
        })
}

#[derive(Debug, thiserror::Error)]
pub enum EmitError {
    #[error(
        "kernel `{0}` has no parameter that bounds the index space; add an `n: u32` parameter"
    )]
    NoBound(String),
    #[error("kernel `{0}` drains nothing, so it computes a value and discards it")]
    NoDrain(String),
    #[error(
        "no PTX ISA version is recorded for target `{0}`. Known: {1}. Add it to TARGET_ISA rather than guessing: `.version` too low is rejected by ptxas, too high by an older driver."
    )]
    UnknownTarget(String, String),
    #[error("{0}")]
    Message(String),
}

/// The generated module, and the facts a caller needs to launch it.
#[derive(Debug, Clone, PartialEq)]
pub struct Module {
    pub ptx: String,
    pub entry: String,
    /// Launch parameters in declaration order, so the host passes them in the right order.
    pub params: Vec<(String, Ty)>,
    /// The `u32` parameter that bounds the element index.
    pub bound: String,
    pub arch: String,
}

/// Emit PTX for one lowered kernel.
///
/// `arch` is the machine's target string, `sm_120` and so on. It comes from the machine file
/// rather than from a flag: ADR-0001 makes the machine a value, and `-arch` is exactly the
/// kind of flag that rots.
pub fn emit(ir: &KernelIr, arch: &str) -> Result<Module, EmitError> {
    // The index space is bounded by the extent the streamed buffers declare, which the front
    // end has already checked they agree on. Taking "the first u32 parameter" instead was
    // right only while a kernel could have exactly one; with shapes a kernel may take a count
    // that is not a length, and picking by type would silently loop over the wrong number.
    // At rank 2 the bound is the product of the space's extents and is computed in the
    // kernel; at rank 1 it is the extent the streamed buffers declare.
    let bound = match &ir.space {
        Some(_) => String::new(),
        None => ir
            .streams
            .iter()
            .find_map(|s| {
                ir.params
                    .iter()
                    .find(|p| p.name == s.buffer)
                    .and_then(|p| p.shape.first())
            })
            .cloned()
            .ok_or_else(|| EmitError::NoBound(ir.name.clone()))?,
    };
    if ir.drains.is_empty() && ir.reduction.is_none() {
        return Err(EmitError::NoDrain(ir.name.clone()));
    }
    // Refused, not ignored. Emitting the untiled kernel for a tiled source would produce
    // working code whose cost is nothing like the one the front end derived for it: the model
    // would say 8 bytes per element and the silicon would move 36. A back end that silently
    // drops a declaration is worse than one that cannot honour it yet.
    if ir.tile.is_some() {
        return Err(EmitError::Message(format!(
            "kernel `{}` declares a tile, which this back end does not emit yet (ADR-0017, step 3). It is refused rather than compiled without staging, because the cost the front end derived would not describe the kernel that came out.",
            ir.name
        )));
    }

    // PTX declares its virtual register banks up front, so the body is emitted first and the
    // header is written once the counts are known.
    let mut e = Emitter {
        n_pred: 0,
        n_f32: 0,
        n_b32: 0,
        n_b64: 0,
    };
    let mut body = String::new();
    e.body(ir, &bound, &mut body)?;

    let mut ptx = String::new();
    let _ = writeln!(ptx, "//");
    let _ = writeln!(ptx, "// Generated by LYTH. Do not edit.");
    let _ = writeln!(ptx, "//   kernel    {}", ir.name);
    let _ = writeln!(ptx, "//   machine   {}", ir.machine);
    let _ = writeln!(
        ptx,
        "//   derived   {:.6} flop/byte = {} flop / {} byte per element",
        ir.cost.intensity,
        ir.cost.flops_per_element,
        ir.cost.bytes_per_element()
    );
    let _ = writeln!(
        ptx,
        "//   traffic   {} read + {} written, per element, at {}",
        ir.cost.read_bytes_per_element(),
        ir.cost.write_bytes_per_element(),
        ir.cost.level().name()
    );
    let _ = writeln!(ptx, "//");
    let _ = writeln!(ptx, ".version {}", isa_for(arch)?);
    let _ = writeln!(ptx, ".target {arch}");
    let _ = writeln!(ptx, ".address_size 64");
    let _ = writeln!(ptx);
    if ir.reduction.is_some() {
        // Dynamic, sized at launch, so the block size is not baked into the module.
        let _ = writeln!(ptx, ".extern .shared .align 4 .b8 lyth_smem[];");
        let _ = writeln!(ptx);
    }
    let _ = writeln!(ptx, ".visible .entry {}(", ir.name);
    let params: Vec<String> = ir
        .params
        .iter()
        .map(|p| {
            let kind = match p.ty {
                Ty::U32 => ".u32",
                Ty::F32 => ".f32",
                Ty::BufF32 => ".u64",
            };
            format!("    .param {kind} {}_{}", ir.name, p.name)
        })
        .collect();
    let _ = writeln!(ptx, "{}", params.join(",\n"));
    let _ = writeln!(ptx, ")");
    let _ = writeln!(ptx, "{{");
    if e.n_pred > 0 {
        let _ = writeln!(ptx, "    .reg .pred  %p<{}>;", e.n_pred + 1);
    }
    if e.n_f32 > 0 {
        let _ = writeln!(ptx, "    .reg .f32   %f<{}>;", e.n_f32 + 1);
    }
    if e.n_b32 > 0 {
        let _ = writeln!(ptx, "    .reg .b32   %r<{}>;", e.n_b32 + 1);
    }
    if e.n_b64 > 0 {
        let _ = writeln!(ptx, "    .reg .b64   %rd<{}>;", e.n_b64 + 1);
    }
    let _ = writeln!(ptx);
    ptx.push_str(&body);
    let _ = writeln!(ptx, "}}");

    Ok(Module {
        ptx,
        entry: ir.name.clone(),
        params: ir.params.iter().map(|p| (p.name.clone(), p.ty)).collect(),
        bound,
        arch: arch.to_string(),
    })
}

struct Emitter {
    n_pred: u32,
    n_f32: u32,
    n_b32: u32,
    n_b64: u32,
}

fn line(o: &mut String, s: &str) {
    let _ = writeln!(o, "    {s}");
}

impl Emitter {
    fn pred(&mut self) -> String {
        self.n_pred += 1;
        format!("%p{}", self.n_pred)
    }

    fn f32(&mut self) -> String {
        self.n_f32 += 1;
        format!("%f{}", self.n_f32)
    }

    fn b32(&mut self) -> String {
        self.n_b32 += 1;
        format!("%r{}", self.n_b32)
    }

    fn b64(&mut self) -> String {
        self.n_b64 += 1;
        format!("%rd{}", self.n_b64)
    }

    fn body(&mut self, ir: &KernelIr, bound: &str, out: &mut String) -> Result<(), EmitError> {
        let k = &ir.name;

        // --- parameters into registers ----------------------------------------------
        //
        // Every u32 is loaded, not only the bound: at rank 2 the extents are needed to
        // decompose the linear index and to stride each buffer, and which u32 is which is a
        // question the shapes answer.
        //
        // Only the ones something uses: an extent named by a buffer or by the space, or the
        // rank-1 bound. A kernel may take a u32 that is a count rather than a length, and
        // loading it would put an instruction in the listing that nothing reads.
        let mut needed: std::collections::BTreeSet<&str> = ir
            .params
            .iter()
            .flat_map(|p| p.shape.iter().map(String::as_str))
            .collect();
        if let Some(sp) = &ir.space {
            needed.extend(sp.extents.iter().map(String::as_str));
        } else {
            needed.insert(bound);
        }
        let mut u32s: Vec<(String, String)> = Vec::new();
        for p in &ir.params {
            if p.ty == Ty::U32 && needed.contains(p.name.as_str()) {
                let r = self.b32();
                line(out, &format!("ld.param.u32 {r}, [{k}_{}];", p.name));
                u32s.push((p.name.clone(), r));
            }
        }
        let u32_of = |name: &str| -> String {
            u32s.iter()
                .find(|(n, _)| n == name)
                .map(|(_, r)| r.clone())
                .unwrap_or_else(|| panic!("`{name}` is not a u32 parameter"))
        };

        let r_bound = match &ir.space {
            None => u32_of(&bound),
            Some(sp) => {
                // rows * cols, the flattened extent.
                //
                // [KNOWN LIMIT] This product is 32-bit and wraps past 2^32 - 1. It is checked
                // once at launch rather than carried in 64-bit index arithmetic on every
                // iteration: 2^32 f32 elements is 17.2 GB, so a buffer cannot reach that
                // index on any device this compiler has a machine file for, and paying wider
                // arithmetic per element for an unreachable case is the wrong trade.
                let b = self.b32();
                line(
                    out,
                    &format!(
                        "mul.lo.u32 {b}, {}, {};",
                        u32_of(&sp.extents[0]),
                        u32_of(&sp.extents[1])
                    ),
                );
                b
            }
        };

        let mut scalars: Vec<(String, String)> = Vec::new();
        for p in &ir.params {
            if p.ty == Ty::F32 {
                let r = self.f32();
                line(out, &format!("ld.param.f32 {r}, [{k}_{}];", p.name));
                scalars.push((p.name.clone(), r));
            }
        }
        let mut buffers: Vec<(String, String)> = Vec::new();
        for p in &ir.params {
            if p.ty == Ty::BufF32 {
                let raw = self.b64();
                let glob = self.b64();
                line(out, &format!("ld.param.u64 {raw}, [{k}_{}];", p.name));
                // Generic to global: a generic address pays an extra translation on every
                // access, and every buffer here is global by construction.
                line(out, &format!("cvta.to.global.u64 {glob}, {raw};"));
                buffers.push((p.name.clone(), glob));
            }
        }

        // --- index, stride, and the loop head --------------------------------------
        //
        // GRID-STRIDE. The grid is no longer a function of `n`, so a thread handles elements
        // i, i+S, i+2S, ... with S = nctaid * ntid. That is what lets the launch shape be
        // varied and therefore measured at all (ADR-0012).
        let ctaid = self.b32();
        let ntid = self.b32();
        let tid = self.b32();
        let nctaid = self.b32();
        let idx = self.b32();
        let stride = self.b32();
        line(out, &format!("mov.u32 {ctaid}, %ctaid.x;"));
        line(out, &format!("mov.u32 {ntid}, %ntid.x;"));
        line(out, &format!("mov.u32 {tid}, %tid.x;"));
        line(out, &format!("mov.u32 {nctaid}, %nctaid.x;"));
        line(out, &format!("mad.lo.s32 {idx}, {ctaid}, {ntid}, {tid};"));
        line(out, &format!("mul.lo.s32 {stride}, {nctaid}, {ntid};"));

        // A reduction accumulates into a register across its own elements before the block
        // tree runs. Starting it at the identity is what makes a thread with no elements
        // correct without a special case -- it simply never enters the loop.
        let acc = if let Some(r) = &ir.reduction {
            let a = self.f32();
            line(out, &format!("mov.f32 {a}, {};", hex_f32(r.op.identity())));
            Some(a)
        } else {
            None
        };

        let _ = writeln!(out, "$L_loop_{k}:");
        let p = self.pred();
        line(out, &format!("setp.ge.u32 {p}, {idx}, {r_bound};"));
        line(out, &format!("@{p} bra $L_loop_end_{k};"));

        // --- the index space, decomposed ----------------------------------------------
        //
        // Row-major, outermost first: `i = k / e1` and `j = k % e1`. The order is part of the
        // contract -- `lyth_lang::eval` walks the same one -- because a reduction folding in a
        // different order is a different bit pattern on a correct kernel.
        //
        // `div.u32` and `rem.u32` are not one instruction on this hardware unless the divisor
        // is a power of two. They are integer work: they move no bytes and retire no flops, so
        // they change neither the traffic model nor the intensity, and `--ncu` cannot be
        // misled by them. What they do change is the instruction count, which ADR-0014
        // measures rather than derives.
        let index_regs: Vec<(String, String)> = match &ir.space {
            None => Vec::new(),
            Some(sp) => {
                let e1 = u32_of(&sp.extents[1]);
                let i = self.b32();
                let j = self.b32();
                line(out, &format!("div.u32 {i}, {idx}, {e1};"));
                line(out, &format!("rem.u32 {j}, {idx}, {e1};"));
                vec![(sp.vars[0].clone(), i), (sp.vars[1].clone(), j)]
            }
        };
        let index_of = |v: &str| -> String {
            index_regs
                .iter()
                .find(|(n, _)| n == v)
                .map(|(_, r)| r.clone())
                .unwrap_or_else(|| panic!("`{v}` is not an index of this space"))
        };

        // --- element addresses, recomputed each iteration -----------------------------
        let reduce_target = ir.reduction.as_ref().map(|r| r.into.as_str());
        // At rank 1 every buffer sits at the same element, so the offset is computed once.
        // At rank 2 each buffer has its own permutation and its own row length.
        let shared_off = match &ir.space {
            None => {
                let off = self.b64();
                line(out, &format!("mul.wide.u32 {off}, {idx}, 4;"));
                Some(off)
            }
            Some(_) => None,
        };
        let mut addrs: Vec<(String, String)> = Vec::new();
        for (name, base) in &buffers {
            // The reduction's target is indexed by block, never by element: computing an
            // element address for it would put an instruction in the listing nothing uses.
            if Some(name.as_str()) == reduce_target {
                continue;
            }
            let off = match &shared_off {
                Some(o) => o.clone(),
                None => {
                    let st = ir
                        .streams
                        .iter()
                        .find(|s| s.buffer == *name)
                        .expect("a buffer with an address is streamed");
                    let shape = &ir
                        .params
                        .iter()
                        .find(|p| p.name == *name)
                        .expect("a streamed buffer is a parameter")
                        .shape;
                    // Row-major: the element at [p, q] of a [_, w] buffer is at p * w + q.
                    // `w` is the buffer's own last extent, not the space's, which is exactly
                    // the difference a transpose turns into strided access.
                    let lin = self.b32();
                    line(
                        out,
                        &format!(
                            "mad.lo.u32 {lin}, {}, {}, {};",
                            index_of(&st.index[0]),
                            u32_of(&shape[1]),
                            index_of(&st.index[1])
                        ),
                    );
                    let off = self.b64();
                    line(out, &format!("mul.wide.u32 {off}, {lin}, 4;"));
                    off
                }
            };
            let a = self.b64();
            line(out, &format!("add.s64 {a}, {base}, {off};"));
            addrs.push((name.clone(), a));
        }
        let addr_of = |name: &str| -> String {
            addrs
                .iter()
                .find(|(n, _)| n == name)
                .map(|(_, a)| a.clone())
                .expect("every buffer parameter got an address")
        };

        // --- the body -----------------------------------------------------------------
        let mut regs: Vec<(RegId, String)> = Vec::new();
        fn get(regs: &[(RegId, String)], id: RegId) -> String {
            regs.iter()
                .find(|(i, _)| *i == id)
                .map(|(_, r)| r.clone())
                .expect("SSA: every use follows its definition")
        }
        for op in &ir.ops {
            match op {
                Op::Load { dst, buffer } => {
                    let r = self.f32();
                    line(out, &format!("ld.global.f32 {r}, [{}];", addr_of(buffer)));
                    regs.push((*dst, r));
                }
                Op::Param { dst, name } => {
                    let r = scalars
                        .iter()
                        .find(|(n, _)| n == name)
                        .map(|(_, r)| r.clone())
                        .ok_or_else(|| {
                            EmitError::Message(format!("no register holds scalar `{name}`"))
                        })?;
                    regs.push((*dst, r));
                }
                Op::Const { dst, value } => {
                    let r = self.f32();
                    line(out, &format!("mov.f32 {r}, {};", hex_f32(*value as f32)));
                    regs.push((*dst, r));
                }
                Op::Bin { dst, op, lhs, rhs } => {
                    let r = self.f32();
                    // `.rn` is round-to-nearest-even, the IEEE default. Spelling it out means
                    // a later reader can see that no fast-math rounding was chosen quietly.
                    let mnemonic = match op {
                        BinOp::Add => "add.rn.f32",
                        BinOp::Sub => "sub.rn.f32",
                        BinOp::Mul => "mul.rn.f32",
                        BinOp::Div => "div.rn.f32",
                    };
                    line(
                        out,
                        &format!(
                            "{mnemonic} {r}, {}, {};",
                            get(&regs, *lhs),
                            get(&regs, *rhs)
                        ),
                    );
                    regs.push((*dst, r));
                }
                Op::Fma { dst, a, b, c } => {
                    let r = self.f32();
                    line(
                        out,
                        &format!(
                            "fma.rn.f32 {r}, {}, {}, {};",
                            get(&regs, *a),
                            get(&regs, *b),
                            get(&regs, *c)
                        ),
                    );
                    regs.push((*dst, r));
                }
                Op::Neg { dst, src } => {
                    let r = self.f32();
                    line(out, &format!("neg.f32 {r}, {};", get(&regs, *src)));
                    regs.push((*dst, r));
                }
            }
        }

        // --- drains --------------------------------------------------------------------
        for (buffer, src) in &ir.drains {
            line(
                out,
                &format!("st.global.f32 [{}], {};", addr_of(buffer), get(&regs, *src)),
            );
        }

        // Fold this element into the accumulator before moving on.
        if let (Some(r), Some(a)) = (&ir.reduction, &acc) {
            let v = get(&regs, r.value);
            match r.op {
                lyth_lang::ast::ReduceOp::Sum => {
                    line(out, &format!("add.rn.f32 {a}, {a}, {v};"));
                }
                // No rounding modifier: a selection has nothing to round.
                lyth_lang::ast::ReduceOp::Max => {
                    line(out, &format!("max.f32 {a}, {a}, {v};"));
                }
                lyth_lang::ast::ReduceOp::Min => {
                    line(out, &format!("min.f32 {a}, {a}, {v};"));
                }
            }
        }

        line(out, &format!("add.s32 {idx}, {idx}, {stride};"));
        line(out, &format!("bra $L_loop_{k};"));
        let _ = writeln!(out, "$L_loop_end_{k}:");

        if let Some(r) = &ir.reduction {
            let a = acc.as_ref().expect("a reduction always has an accumulator");
            self.reduction(ir, r, &buffers, a, &tid, &ctaid, &ntid, out)?;
        }

        let _ = writeln!(out, "$L_done_{k}:");
        line(out, "ret;");
        Ok(())
    }
}

impl Emitter {
    /// The tree, in shared memory, matching `lyth_lang::eval::tree_reduce` step for step.
    ///
    /// Any divergence between the two orders shows up as a wrong last bit on a correct
    /// kernel, so the two are written to be read side by side.
    #[allow(clippy::too_many_arguments)]
    fn reduction(
        &mut self,
        ir: &KernelIr,
        r: &lyth_lang::ir::ReductionIr,
        buffers: &[(String, String)],
        acc: &str,
        tid: &str,
        ctaid: &str,
        ntid: &str,
        out: &mut String,
    ) -> Result<(), EmitError> {
        let k = &ir.name;

        // Every thread arrives here with its accumulator, having folded however many elements
        // the grid-stride loop gave it -- possibly none, in which case it still holds the
        // identity. No idle-thread branch is needed, and every thread reaches every barrier.
        //
        // slot = lyth_smem + tid * 4
        let smem = self.b64();
        let off = self.b64();
        let slot = self.b64();
        line(out, &format!("mov.u64 {smem}, lyth_smem;"));
        line(out, &format!("mul.wide.u32 {off}, {tid}, 4;"));
        line(out, &format!("add.s64 {slot}, {smem}, {off};"));
        line(out, &format!("st.shared.f32 [{slot}], {acc};"));
        line(out, "bar.sync 0;");

        // for (stride = ntid / 2; stride > 0; stride >>= 1)
        let stride = self.b32();
        line(out, &format!("shr.u32 {stride}, {ntid}, 1;"));
        let _ = writeln!(out, "$L_tree_{k}:");
        let done = self.pred();
        line(out, &format!("setp.eq.u32 {done}, {stride}, 0;"));
        line(out, &format!("@{done} bra $L_tree_end_{k};"));

        let active = self.pred();
        let a = self.f32();
        let b = self.f32();
        let combined = self.f32();
        let mate_off = self.b64();
        let mate = self.b64();
        let partner = self.b32();
        line(out, &format!("setp.ge.u32 {active}, {tid}, {stride};"));
        line(out, &format!("@{active} bra $L_skip_{k};"));
        line(out, &format!("add.u32 {partner}, {tid}, {stride};"));
        line(out, &format!("mul.wide.u32 {mate_off}, {partner}, 4;"));
        line(out, &format!("add.s64 {mate}, {smem}, {mate_off};"));
        line(out, &format!("ld.shared.f32 {a}, [{slot}];"));
        line(out, &format!("ld.shared.f32 {b}, [{mate}];"));
        match r.op {
            lyth_lang::ast::ReduceOp::Sum => {
                line(out, &format!("add.rn.f32 {combined}, {a}, {b};"));
            }
            lyth_lang::ast::ReduceOp::Max => {
                line(out, &format!("max.f32 {combined}, {a}, {b};"));
            }
            lyth_lang::ast::ReduceOp::Min => {
                line(out, &format!("min.f32 {combined}, {a}, {b};"));
            }
        }
        line(out, &format!("st.shared.f32 [{slot}], {combined};"));
        let _ = writeln!(out, "$L_skip_{k}:");
        // Every thread reaches this barrier, including the ones that skipped the combine.
        line(out, "bar.sync 0;");
        line(out, &format!("shr.u32 {stride}, {stride}, 1;"));
        line(out, &format!("bra $L_tree_{k};"));
        let _ = writeln!(out, "$L_tree_end_{k}:");

        // Thread 0 writes the block's result.
        let not_zero = self.pred();
        line(out, &format!("setp.ne.u32 {not_zero}, {tid}, 0;"));
        line(out, &format!("@{not_zero} bra $L_done_{k};"));
        let result = self.f32();
        line(out, &format!("ld.shared.f32 {result}, [{smem}];"));
        let into = buffers
            .iter()
            .find(|(n, _)| *n == r.into)
            .map(|(_, base)| base.clone())
            .ok_or_else(|| EmitError::Message(format!("no base address for `{}`", r.into)))?;
        let block_off = self.b64();
        let block_addr = self.b64();
        line(out, &format!("mul.wide.u32 {block_off}, {ctaid}, 4;"));
        line(out, &format!("add.s64 {block_addr}, {into}, {block_off};"));
        line(out, &format!("st.global.f32 [{block_addr}], {result};"));
        Ok(())
    }
}

/// PTX writes float immediates as a hex bit pattern, which is exact. A decimal literal would
/// be re-parsed by `ptxas` and could land on a different value than the one the front end
/// folded.
fn hex_f32(v: f32) -> String {
    format!("0f{:08X}", v.to_bits())
}

#[cfg(test)]
mod tests {
    use super::*;
    use lyth_lang::{ir::lower, parse::parse};

    const SAXPY: &str = "\
machine sm_120

kernel saxpy(n: u32, a: f32, x: [f32; n], y: [f32; n])
    intensity 0.1667

    stream x : dram -> reg
    stream y : dram -> reg, drain

    at reg:
        y = a * x + y
";

    fn saxpy_ptx() -> Module {
        let u = parse(SAXPY).unwrap();
        let ir = lower(&u, &u.kernels[0]).unwrap();
        emit(&ir, "sm_120").unwrap()
    }

    #[test]
    fn an_unknown_target_is_refused_and_lists_the_known_ones() {
        let u = parse(SAXPY).unwrap();
        let ir = lower(&u, &u.kernels[0]).unwrap();
        let e = emit(&ir, "sm_999").unwrap_err();
        assert!(e.to_string().contains("sm_120"), "{e}");
    }

    #[test]
    fn the_isa_version_follows_the_target_rather_than_being_fixed() {
        // Found by generating `.version 8.5` with `.target sm_120` and having ptxas reject it.
        assert_eq!(isa_for("sm_120").unwrap(), "8.7");
        assert_eq!(isa_for("sm_80").unwrap(), "7.0");
    }

    #[test]
    fn the_module_has_the_shape_ptxas_needs() {
        let m = saxpy_ptx();
        assert!(m.ptx.contains(".version 8.7"));
        assert!(m.ptx.contains(".target sm_120"));
        assert!(m.ptx.contains(".address_size 64"));
        assert!(m.ptx.contains(".visible .entry saxpy("));
        assert!(m.ptx.trim_end().ends_with('}'), "{}", m.ptx);
    }

    #[test]
    fn one_fma_and_no_separate_multiply() {
        let m = saxpy_ptx();
        assert_eq!(m.ptx.matches("fma.rn.f32").count(), 1, "{}", m.ptx);
        assert_eq!(m.ptx.matches("mul.rn.f32").count(), 0, "{}", m.ptx);
    }

    #[test]
    fn the_bytes_it_moves_are_the_bytes_the_front_end_derived() {
        let m = saxpy_ptx();
        // Two 4-byte loads and one 4-byte store: exactly the 8 read + 4 written that `Cost`
        // claims. If these two ever disagree, the derived intensity is fiction.
        assert_eq!(m.ptx.matches("ld.global.f32").count(), 2, "{}", m.ptx);
        assert_eq!(m.ptx.matches("st.global.f32").count(), 1, "{}", m.ptx);
    }

    #[test]
    fn the_loop_is_bounded_and_strided() {
        let m = saxpy_ptx();
        // The bound is the loop condition now, not a one-shot guard at the top.
        assert!(m.ptx.contains("setp.ge.u32"), "{}", m.ptx);
        assert!(m.ptx.contains("bra $L_loop_end_saxpy"), "{}", m.ptx);
        assert!(m.ptx.contains("bra $L_loop_saxpy"), "{}", m.ptx);
        // The stride is grid x block, read from the launch rather than assumed.
        assert!(m.ptx.contains("%nctaid.x"), "{}", m.ptx);
        assert!(m.ptx.contains("mul.lo.s32"), "{}", m.ptx);
        assert!(
            m.ptx.contains("add.s32"),
            "advance by the stride:
{}",
            m.ptx
        );
    }

    #[test]
    fn the_header_records_what_the_compiler_derived() {
        let m = saxpy_ptx();
        assert!(m.ptx.contains("0.166667 flop/byte"), "{}", m.ptx);
        assert!(m.ptx.contains("8 read + 4 written"), "{}", m.ptx);
    }

    #[test]
    fn float_constants_are_exact_bit_patterns() {
        assert_eq!(hex_f32(1.0), "0f3F800000");
        assert_eq!(hex_f32(0.0), "0f00000000");
        assert_eq!(hex_f32(-2.5), "0fC0200000");
    }

    #[test]
    fn a_kernel_with_no_bound_is_refused_by_the_front_end_now() {
        // This was a back-end refusal: the emitter looked for a u32 parameter and gave up.
        // With shapes the front end gets there first, because the buffer names an extent that
        // is not a parameter. A better place to be told, and a better thing to be told.
        let src = "machine sm_120\n\nkernel k(x: [f32; n], y: [f32; n])\n    stream x : dram -> reg\n    stream y : dram -> reg, drain\n    at reg:
        y = x\n";
        let u = parse(src).unwrap();
        let e = lower(&u, &u.kernels[0]).unwrap_err();
        assert!(e.to_string().contains("is not a u32 parameter"), "{e}");
    }

    #[test]
    fn the_loop_bound_comes_from_the_shape_and_not_from_the_first_u32() {
        // `stride` is a count, not a length. Bounding the loop by whichever u32 came first
        // would walk the wrong number of elements while compiling cleanly, which is the worst
        // kind of wrong. The shape says which parameter is a length.
        let src = "machine sm_120\n\nkernel k(stride: u32, n: u32, x: [f32; n], y: [f32; n])\n    stream x : dram -> reg\n    stream y : dram -> reg, drain\n    at reg:
        y = x\n";
        let u = parse(src).unwrap();
        let ir = lower(&u, &u.kernels[0]).unwrap();
        let m = emit(&ir, "sm_120").unwrap();
        let bound = m
            .ptx
            .lines()
            .find(|l| l.contains("ld.param.u32") && l.contains("_n]"))
            .unwrap_or_else(|| panic!("the bound should load k_n:\n{}", m.ptx));
        assert!(bound.contains("k_n"), "{bound}");
        assert!(
            !m.ptx.contains("[k_stride]"),
            "stride is not a length and must not bound the loop:\n{}",
            m.ptx
        );
    }

    #[test]
    fn registers_are_declared_before_they_are_used() {
        let m = saxpy_ptx();
        let decl = m.ptx.find(".reg .f32").expect("an f32 bank");
        let first_use = m.ptx.find("ld.param.f32").expect("a parameter load");
        assert!(decl < first_use, "PTX declares its banks up front");
    }
}
