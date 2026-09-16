//! The tiled contraction: two staged tiles, a loop over the contracted axis, an accumulator
//! that survives it (ADR-0018 step 3).
//!
//! This is the body ADR-0000 is about. A transpose moves the same bytes under every schedule;
//! a contraction does not, and the schedule emitted here is what makes `2K/T + 1` per output
//! true rather than a formula. The compiler derives that number in `lyth-lang`; this file is
//! the part that has to earn it.
//!
//! Five things are decided here rather than inherited from `tiled.rs`, and each one is a
//! correctness question the transpose never had to answer:
//!
//! * **The tile is square.** `A`'s tile is `T_i x T_p` and `B`'s is `T_p x T_j`, and one block
//!   of `T_i * T_j` threads loads one element of each. That closes only when the three are the
//!   same number. A rectangular tile has a derivable cost and no schedule here, so it is
//!   refused rather than emitted at the wrong shape.
//! * **The `k` boundary is a loop bound, not a zero fill.** Zero-filling the tail of a tile is
//!   correct for `sum`, where a zero term changes nothing, and wrong for `max`, where it
//!   clamps the result to be at least zero. The inner loop runs `min(T, k - p)` times instead,
//!   which is exact for every operator and does less work.
//! * **Both barriers are outside every branch.** The step loop's bound is `ceil(k / T)`, which
//!   is block-uniform, so no thread can leave while another waits. ADR-0017 measured what the
//!   alternative costs; here it would hang.
//! * **The accumulator is not fused.** `acc = acc + a * b` is emitted as a multiply and an add,
//!   not `fma.rn.f32`. An fma rounds once where two instructions round twice, so fusing would
//!   change the answer -- for the better, and after the host oracle agreed to change with it.
//!   That is ADR-0010's decision to make again, not a free improvement to take here.
//! * **Out-of-range threads still take part.** A thread whose output is past `m` or `n` loads
//!   zeros, accumulates nonsense and is discarded by the store guard. It must not branch out:
//!   its neighbours are waiting for it at `bar.sync`.

use std::fmt::Write;

use lyth_lang::ast::ReduceOp;
use lyth_lang::ir::{KernelIr, RegId};

use crate::{line, EmitError, Emitter};

/// What the emitter needs, once it has refused everything it cannot honour.
pub struct ContractPlan<'a> {
    /// The tile edge. Square: `T_i = T_j = T_p = T`.
    pub t: u32,
    /// The padded row stride in elements, derived to be coprime with the bank count.
    pub stride: u32,
    pub op: ReduceOp,
    /// The free extents, outermost first: `m`, `n`.
    pub free: &'a [String],
    /// The contracted extent: `k`.
    pub k: &'a str,
    /// The two staged operands, each with the register its element lands in and the position
    /// the contracted index occupies in its own index list.
    ///
    /// `a[i, p]` gives `contracted_at = 1`, `b[p, j]` gives `0`. That single number is what
    /// tells the emitter which way round to walk the buffer and which way round to read the
    /// tile back out of shared memory, and it comes from the source rather than from the
    /// operand's name.
    pub operands: [Operand<'a>; 2],
    pub drain: &'a str,
    pub drain_reg: RegId,
    /// Row length of the drained buffer: its own last extent.
    pub drain_row: &'a str,
}

pub struct Operand<'a> {
    pub buffer: &'a str,
    pub reg: RegId,
    pub contracted_at: usize,
    /// Row length: this buffer's own last extent, which is `k` for `a` and `n` for `b`.
    pub row: &'a str,
}

impl<'a> ContractPlan<'a> {
    pub fn of(ir: &'a KernelIr) -> Result<Self, EmitError> {
        let c = ir.contract.as_ref().expect("a contracted body has one");
        let name = &ir.name;
        let refuse = |msg: String| Err(EmitError::Message(msg));

        let Some(tile) = ir.tile.as_ref() else {
            return refuse(format!(
                "kernel `{name}` contracts over `{}` without a tile. Its cost is derived -- two elements read per output per unit of `{}`, which is the untiled control -- but the schedule emitted here is the tiled one, and a contraction with no tile has no shared memory to reuse through. Add `tile 32, 32`.",
                c.extent, c.extent
            ));
        };
        if tile[0] != tile[1] {
            return refuse(format!(
                "kernel `{name}` tiles {} x {}. A contraction stages an operand tile of `T_i x T_p` and another of `T_p x T_j`, one element per thread of a `T_i * T_j` block, which closes only when the three are equal. The cost model derives a rectangular tile correctly; there is no schedule for it here.",
                tile[0], tile[1]
            ));
        };
        let space = ir.space.as_ref().ok_or_else(|| {
            EmitError::Message("a contraction needs a space; the front end refuses otherwise".into())
        })?;
        let layout = ir.shared.as_ref().ok_or_else(|| {
            EmitError::Message(format!(
                "kernel `{name}` contracts and tiles but stages nothing. A tile says the threads exist; shared memory is what makes them share, so without `dram -> smem -> reg` on both operands the reuse is 1 and the derived traffic says so."
            ))
        })?;

        let staged: Vec<_> = ir.streams.iter().filter(|s| s.staged).collect();
        if staged.len() != 2 {
            return refuse(format!(
                "kernel `{name}` stages {} buffers; a contraction stages exactly two. One staged and one not is a schedule with half the reuse, and the cost model derives it -- the emitter does not have it.",
                staged.len()
            ));
        }
        if ir.streams.iter().any(|s| s.read && !s.staged) {
            return refuse(format!(
                "kernel `{name}` reads a buffer that is not staged beside two that are."
            ));
        }
        if ir.drains.len() != 1 {
            return refuse(format!(
                "kernel `{name}` drains {} buffers; a contraction writes one.",
                ir.drains.len()
            ));
        }

        let row_of = |n: &str| -> Result<&'a str, EmitError> {
            ir.params
                .iter()
                .find(|p| p.name == n)
                .and_then(|p| p.shape.last())
                .map(String::as_str)
                .ok_or_else(|| EmitError::Message(format!("`{n}` has no row length")))
        };
        let mut operands = Vec::new();
        for s in &staged {
            let at = s
                .index
                .iter()
                .position(|v| *v == c.var)
                .ok_or_else(|| {
                    EmitError::Message(format!(
                        "kernel `{name}` stages `{}`, which the contraction does not walk. A staged operand that is not contracted is loaded once per output and its tile is never reused.",
                        s.buffer
                    ))
                })?;
            operands.push(Operand {
                buffer: &s.buffer,
                reg: s.loaded.ok_or_else(|| {
                    EmitError::Message(format!("`{}` is staged but never read", s.buffer))
                })?,
                contracted_at: at,
                row: row_of(&s.buffer)?,
            });
        }
        // `a[i, p]` and `b[p, j]`: one operand carries the contracted index second and the
        // other first. Both the same way round is `a[i, p] * b[j, p]`, which is a valid thing
        // to want and a different schedule -- `B` would be walked down its columns.
        if operands[0].contracted_at == operands[1].contracted_at {
            return refuse(format!(
                "kernel `{name}` indexes both operands with `{}` in the same position. The tiled schedule reads one operand along its rows and the other down its columns; two the same way round needs a transposed stage, which is its own decision.",
                c.var
            ));
        }
        let second = operands.pop().expect("two operands");
        let first = operands.pop().expect("two operands");

        let (drain, drain_reg) = &ir.drains[0];
        Ok(ContractPlan {
            t: tile[0],
            stride: layout.tiles[0].2,
            op: c.op,
            free: &space.extents,
            k: &c.extent,
            operands: [first, second],
            drain,
            drain_reg: *drain_reg,
            drain_row: row_of(drain)?,
        })
    }
}

impl Emitter {
    /// Emit the contracted body. `coarsen 1, 1` -- the absent declaration -- is the case every
    /// test before ADR-0021 exercises, and it is this code with both factors set to one rather
    /// than a separate path, so those tests keep covering it.
    pub(crate) fn contracted_body(
        &mut self,
        ir: &KernelIr,
        skewed: bool,
        out: &mut String,
    ) -> Result<(), EmitError> {
        let p = ContractPlan::of(ir)?;
        let k = &ir.name;
        let t = p.t;
        let log2_t = t.trailing_zeros();
        // `--no-skew` builds the unpadded variant, as it does for a transpose. What it is a
        // counterfactual *for* is different here, and that is the point of having it: a
        // transpose reads its tile down a column and the skew is what makes that free, while
        // this schedule reads `A` as a broadcast within a warp and `B` along a row. If the
        // derivation is right, neither variant conflicts -- so the padding is 256 bytes of
        // shared memory this kernel does not need. Measured rather than argued.
        let stride = if skewed { p.stride } else { p.t };

        // How many outputs of the tile one thread owns, and therefore how wide the block is.
        //
        // The tile stays the tile: `t` is still the working set the traffic was derived from,
        // and every address below is still an address into a `t x t` patch. What changes is
        // how many threads cover it -- `bi x bj` instead of `t x t` -- and so how many of its
        // elements each one is responsible for.
        let (ci, cj) = match &ir.coarsen {
            Some(c) => (c[0], c[1]),
            None => (1, 1),
        };
        let bi = t / ci;
        let bj = t / cj;
        let log2_bj = bj.trailing_zeros();

        // --- parameters -------------------------------------------------------------
        let mut u32s: Vec<(String, String)> = Vec::new();
        for param in &ir.params {
            if param.ty == lyth_lang::ast::Ty::U32 {
                let r = self.b32();
                line(out, &format!("ld.param.u32 {r}, [{k}_{}];", param.name));
                u32s.push((param.name.clone(), r));
            }
        }
        let u32_of = |n: &str| -> String {
            u32s.iter()
                .find(|(x, _)| x == n)
                .map(|(_, r)| r.clone())
                .unwrap_or_else(|| panic!("`{n}` is not a u32 parameter"))
        };
        let mut bufs: Vec<(String, String)> = Vec::new();
        for param in &ir.params {
            if param.ty == lyth_lang::ast::Ty::BufF32 {
                let raw = self.b64();
                let glob = self.b64();
                line(out, &format!("ld.param.u64 {raw}, [{k}_{}];", param.name));
                line(out, &format!("cvta.to.global.u64 {glob}, {raw};"));
                bufs.push((param.name.clone(), glob));
            }
        }
        let buf_of = |n: &str| -> String {
            bufs.iter()
                .find(|(x, _)| x == n)
                .map(|(_, r)| r.clone())
                .expect("every buffer parameter was loaded")
        };

        let m = u32_of(&p.free[0]);
        let n = u32_of(&p.free[1]);
        let kk = u32_of(p.k);

        // --- how many output tiles, and where this thread sits in one ----------------
        let tiles_m = self.b32();
        let tiles_n = self.b32();
        let total = self.b32();
        line(out, &format!("add.u32 {tiles_m}, {m}, {};", t - 1));
        line(out, &format!("shr.u32 {tiles_m}, {tiles_m}, {log2_t};"));
        line(out, &format!("add.u32 {tiles_n}, {n}, {};", t - 1));
        line(out, &format!("shr.u32 {tiles_n}, {tiles_n}, {log2_t};"));
        line(out, &format!("mul.lo.u32 {total}, {tiles_m}, {tiles_n};"));

        // The block is `bi x bj`, so the thread's place in it is a mask and a shift by the
        // block's width -- not the tile's. Both are powers of two, which is what `coarsen`
        // refuses a factor for not preserving.
        let tid = self.b32();
        let tx = self.b32();
        let ty = self.b32();
        line(out, &format!("mov.u32 {tid}, %tid.x;"));
        line(out, &format!("and.b32 {tx}, {tid}, {};", bj - 1));
        line(out, &format!("shr.u32 {ty}, {tid}, {log2_bj};"));

        // Two tiles, back to back. The second starts where the first ends, and the first is
        // `T` rows of the padded stride -- the padding is inside the tile, so the offset is
        // the padded size and not the nominal one.
        let smem = self.b64();
        let smem_b = self.b64();
        line(out, &format!("mov.u64 {smem}, lyth_smem;"));
        line(
            out,
            &format!("add.s64 {smem_b}, {smem}, {};", t * stride * 4),
        );
        let tile_base = [smem.clone(), smem_b.clone()];

        // How many steps along the contracted axis. Block-uniform, so the step loop's exit is
        // uniform and both barriers below are reached by every thread or by none.
        let steps = self.b32();
        line(out, &format!("add.u32 {steps}, {kk}, {};", t - 1));
        line(out, &format!("shr.u32 {steps}, {steps}, {log2_t};"));

        // --- the grid-stride loop over output tiles ---------------------------------
        let tile = self.b32();
        let gstep = self.b32();
        let ctaid = self.b32();
        let nctaid = self.b32();
        line(out, &format!("mov.u32 {ctaid}, %ctaid.x;"));
        line(out, &format!("mov.u32 {nctaid}, %nctaid.x;"));
        line(out, &format!("mov.u32 {tile}, {ctaid};"));
        line(out, &format!("mov.u32 {gstep}, {nctaid};"));

        let _ = writeln!(out, "$L_tile_{k}:");
        let done = self.pred();
        line(out, &format!("setp.ge.u32 {done}, {tile}, {total};"));
        line(out, &format!("@{done} bra $L_tile_end_{k};"));

        let trow = self.b32();
        let tcol = self.b32();
        line(out, &format!("div.u32 {trow}, {tile}, {tiles_n};"));
        line(out, &format!("rem.u32 {tcol}, {tile}, {tiles_n};"));

        // This thread's outputs. It owns `ci x cj` of them, spread across the tile by the
        // block's width rather than packed together: thread `tx` takes columns `tx`,
        // `tx + bj`, `tx + 2bj`, so a warp still covers `bj` consecutive columns and the
        // global store stays coalesced.
        let mut gi = Vec::new();
        for u in 0..ci {
            let r = self.b32();
            line(out, &format!("mad.lo.u32 {r}, {trow}, {t}, {ty};"));
            if u > 0 {
                line(out, &format!("add.u32 {r}, {r}, {};", u * bi));
            }
            gi.push(r);
        }
        let mut gj = Vec::new();
        for v in 0..cj {
            let r = self.b32();
            line(out, &format!("mad.lo.u32 {r}, {tcol}, {t}, {tx};"));
            if v > 0 {
                line(out, &format!("add.u32 {r}, {r}, {};", v * bj));
            }
            gj.push(r);
        }

        // One accumulator per output, alive across the whole `p` loop. This is where the
        // arithmetic intensity comes from: the register file is a third level of reuse under
        // shared memory, and `ci + cj` operands feed `ci * cj` multiply-adds.
        let mut acc = Vec::new();
        for _ in 0..ci {
            let mut row = Vec::new();
            for _ in 0..cj {
                let r = self.f32();
                line(out, &format!("mov.f32 {r}, {};", crate::hex_f32(p.op.identity())));
                row.push(r);
            }
            acc.push(row);
        }

        // --- the loop along the contracted axis -------------------------------------
        let s = self.b32();
        let pbase = self.b32();
        line(out, &format!("mov.u32 {s}, 0;"));
        let _ = writeln!(out, "$L_step_{k}:");
        let step_done = self.pred();
        line(out, &format!("setp.ge.u32 {step_done}, {s}, {steps};"));
        line(out, &format!("@{step_done} bra $L_step_end_{k};"));
        line(out, &format!("shl.b32 {pbase}, {s}, {log2_t};"));

        // Phase 1: stage both tiles. The tile is `t x t` and the block is `bi x bj`, so each
        // thread brings `ci * cj` elements of each operand -- one for each of the places it
        // owns. Predicated rather than branched, so the block stays together for the barrier.
        for (idx, op) in p.operands.iter().enumerate() {
            for u in 0..ci {
                for v in 0..cj {
                    let gy = self.b32();
                    let gx = self.b32();
                    // The operand's own two axes, in its own order: one is a free axis of the
                    // output, the other is the contracted one. `contracted_at` says which, and
                    // the coarsening offsets ride along with whichever is which.
                    if op.contracted_at == 1 {
                        // `a[i, p]`: rows are the output's first free axis, columns are `k`.
                        line(out, &format!("mad.lo.u32 {gy}, {trow}, {t}, {ty};"));
                        if u > 0 {
                            line(out, &format!("add.u32 {gy}, {gy}, {};", u * bi));
                        }
                        line(out, &format!("add.u32 {gx}, {pbase}, {tx};"));
                        if v > 0 {
                            line(out, &format!("add.u32 {gx}, {gx}, {};", v * bj));
                        }
                    } else {
                        // `b[p, j]`: rows are `k`, columns are the output's second free axis.
                        line(out, &format!("add.u32 {gy}, {pbase}, {ty};"));
                        if u > 0 {
                            line(out, &format!("add.u32 {gy}, {gy}, {};", u * bi));
                        }
                        line(out, &format!("mad.lo.u32 {gx}, {tcol}, {t}, {tx};"));
                        if v > 0 {
                            line(out, &format!("add.u32 {gx}, {gx}, {};", v * bj));
                        }
                    }

                    let (row_bound, col_bound) = if op.contracted_at == 1 {
                        (m.clone(), kk.clone())
                    } else {
                        (kk.clone(), n.clone())
                    };
                    let in_y = self.pred();
                    let in_x = self.pred();
                    let in_both = self.pred();
                    line(out, &format!("setp.lt.u32 {in_y}, {gy}, {row_bound};"));
                    line(out, &format!("setp.lt.u32 {in_x}, {gx}, {col_bound};"));
                    line(out, &format!("and.pred {in_both}, {in_y}, {in_x};"));

                    let lin = self.b32();
                    let off = self.b64();
                    let addr = self.b64();
                    let val = self.f32();
                    line(
                        out,
                        &format!("mad.lo.u32 {lin}, {gy}, {}, {gx};", u32_of(op.row)),
                    );
                    line(out, &format!("mul.wide.u32 {off}, {lin}, 4;"));
                    line(out, &format!("add.s64 {addr}, {}, {off};", buf_of(op.buffer)));
                    line(out, &format!("mov.f32 {val}, 0f00000000;"));
                    line(out, &format!("@{in_both} ld.global.f32 {val}, [{addr}];"));

                    // Into shared at the thread's own place in the tile, both operands
                    // row-major. The skew is in the stride, so the column reads below land on
                    // 32 distinct banks.
                    let s_at = self.b32();
                    let s_off = self.b64();
                    let s_addr = self.b64();
                    line(out, &format!("mov.u32 {s_at}, {ty};"));
                    if u > 0 {
                        line(out, &format!("add.u32 {s_at}, {s_at}, {};", u * bi));
                    }
                    line(out, &format!("mul.lo.u32 {s_at}, {s_at}, {stride};"));
                    line(out, &format!("add.u32 {s_at}, {s_at}, {tx};"));
                    if v > 0 {
                        line(out, &format!("add.u32 {s_at}, {s_at}, {};", v * bj));
                    }
                    line(out, &format!("mul.wide.u32 {s_off}, {s_at}, 4;"));
                    line(
                        out,
                        &format!("add.s64 {s_addr}, {}, {s_off};", tile_base[idx]),
                    );
                    line(out, &format!("st.shared.f32 [{s_addr}], {val};"));
                }
            }
        }

        // Barrier 1: read-after-write. Nobody walks a tile before it is whole.
        line(out, "bar.sync 0;");

        // Phase 2: the terms of this step.
        //
        // The bound is `min(T, k - p)` rather than `T` with a zero fill. Zero-filling is
        // correct for `sum` and wrong for `max`: a zero term cannot change a sum and does
        // clamp a maximum. Block-uniform either way, so there is no divergence around the
        // barrier that follows.
        let left = self.b32();
        let inner = self.b32();
        line(out, &format!("sub.u32 {left}, {kk}, {pbase};"));
        line(out, &format!("min.u32 {inner}, {left}, {t};"));

        let tt = self.b32();
        line(out, &format!("mov.u32 {tt}, 0;"));
        let _ = writeln!(out, "$L_term_{k}:");
        let term_done = self.pred();
        line(out, &format!("setp.ge.u32 {term_done}, {tt}, {inner};"));
        line(out, &format!("@{term_done} bra $L_term_end_{k};"));

        // `ci` values of one operand and `cj` of the other, read once and used `ci * cj`
        // times. At `coarsen 1, 1` that is two loads for one multiply-add, which is the ratio
        // ADR-0018 was stuck at; at `2, 2` it is four loads for four.
        //
        // Which operand is walked along which axis comes from `contracted_at`, so the two are
        // not two cases of a rule -- they are the same rule read at different positions.
        let (a_idx, b_idx) = if p.operands[0].contracted_at == 1 {
            (0usize, 1usize)
        } else {
            (1usize, 0usize)
        };
        let mut a_val = Vec::new();
        for u in 0..ci {
            let at = self.b32();
            let off = self.b64();
            let addr = self.b64();
            let got = self.f32();
            line(out, &format!("mov.u32 {at}, {ty};"));
            if u > 0 {
                line(out, &format!("add.u32 {at}, {at}, {};", u * bi));
            }
            line(out, &format!("mul.lo.u32 {at}, {at}, {stride};"));
            line(out, &format!("add.u32 {at}, {at}, {tt};"));
            line(out, &format!("mul.wide.u32 {off}, {at}, 4;"));
            line(out, &format!("add.s64 {addr}, {}, {off};", tile_base[a_idx]));
            line(out, &format!("ld.shared.f32 {got}, [{addr}];"));
            a_val.push(got);
        }
        let mut b_val = Vec::new();
        for v in 0..cj {
            let at = self.b32();
            let off = self.b64();
            let addr = self.b64();
            let got = self.f32();
            line(out, &format!("mad.lo.u32 {at}, {tt}, {stride}, {tx};"));
            if v > 0 {
                line(out, &format!("add.u32 {at}, {at}, {};", v * bj));
            }
            line(out, &format!("mul.wide.u32 {off}, {at}, 4;"));
            line(out, &format!("add.s64 {addr}, {}, {off};", tile_base[b_idx]));
            line(out, &format!("ld.shared.f32 {got}, [{addr}];"));
            b_val.push(got);
        }

        for u in 0..ci as usize {
            for v in 0..cj as usize {
                // The body, run once per output this thread owns, on that output's pair of
                // operands. Same `ir.ops`, same `self.op` -- the body is the body wherever the
                // schedule puts it.
                let mut regs: Vec<(RegId, String)> = vec![
                    (p.operands[a_idx].reg, a_val[u].clone()),
                    (p.operands[b_idx].reg, b_val[v].clone()),
                ];
                for op in &ir.ops {
                    self.op(op, &mut regs, &u32s, out)?;
                }
                let term = regs
                    .iter()
                    .find(|(i, _)| *i == p.drain_reg)
                    .map(|(_, r)| r.clone())
                    .ok_or_else(|| {
                        EmitError::Message("the contracted term was never computed".into())
                    })?;
                // Two instructions, not one. See the note at the top of this file.
                let a = &acc[u][v];
                line(
                    out,
                    &format!("{} {a}, {a}, {term};", crate::reduce_mnemonic(p.op)),
                );
            }
        }

        line(out, &format!("add.u32 {tt}, {tt}, 1;"));
        line(out, &format!("bra $L_term_{k};"));
        let _ = writeln!(out, "$L_term_end_{k}:");

        // Barrier 2: write-after-read. The next step's store must not overtake this walk.
        line(out, "bar.sync 0;");
        line(out, &format!("add.u32 {s}, {s}, 1;"));
        line(out, &format!("bra $L_step_{k};"));
        let _ = writeln!(out, "$L_step_end_{k}:");

        // --- the outputs, once each --------------------------------------------------
        for u in 0..ci as usize {
            for v in 0..cj as usize {
                let out_y = self.pred();
                let out_x = self.pred();
                let out_both = self.pred();
                line(out, &format!("setp.lt.u32 {out_y}, {}, {m};", gi[u]));
                line(out, &format!("setp.lt.u32 {out_x}, {}, {n};", gj[v]));
                line(out, &format!("and.pred {out_both}, {out_y}, {out_x};"));

                let olin = self.b32();
                let o_off = self.b64();
                let o_addr = self.b64();
                line(
                    out,
                    &format!(
                        "mad.lo.u32 {olin}, {}, {}, {};",
                        gi[u],
                        u32_of(p.drain_row),
                        gj[v]
                    ),
                );
                line(out, &format!("mul.wide.u32 {o_off}, {olin}, 4;"));
                line(
                    out,
                    &format!("add.s64 {o_addr}, {}, {o_off};", buf_of(p.drain)),
                );
                line(
                    out,
                    &format!("@{out_both} st.global.f32 [{o_addr}], {};", acc[u][v]),
                );
            }
        }

        line(out, &format!("add.u32 {tile}, {tile}, {gstep};"));
        line(out, &format!("bra $L_tile_{k};"));
        let _ = writeln!(out, "$L_tile_end_{k}:");
        line(out, "ret;");
        Ok(())
    }
}
