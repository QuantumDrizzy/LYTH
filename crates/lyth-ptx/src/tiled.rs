//! The four-phase tiled body (ADR-0017, step 3).
//!
//! A block handles one tile at a time, walking tiles with the same grid-stride the flat body
//! uses over elements. Per tile:
//!
//! 1. position: the tile from `%ctaid`, the place inside it from `%tid` by shift and mask;
//! 2. a coalesced global read into the padded shared tile;
//! 3. `bar.sync`, then the body over the value read back from shared, transposed;
//! 4. a coalesced global write, and `bar.sync` again.
//!
//! **Two barriers, and the second is not optional.** The loop is grid-stride over *tiles*, so
//! after a block finishes tile `T` it starts tile `T + gridDim`, whose phase 2 overwrites the
//! shared tile that phase 4 of `T` may still be reading. One barrier leaves a write-after-read
//! hazard that corrupts output only sometimes, which is worse than always.
//!
//! **Boundary guards predicate instructions; they never branch.** A thread that skipped a
//! `bar.sync` while the rest of its block reached one deadlocks the block or aborts the
//! launch. So the loop condition is on the *tile* index, which is uniform across the block,
//! and every out-of-range thread still walks the whole body with its global accesses
//! predicated off.

use std::fmt::Write as _;

use lyth_lang::ir::{KernelIr, Op, RegId};

use crate::{line, EmitError, Emitter};

/// What the tiled path needs that the flat path does not, checked once so the emitter below
/// can assume it.
pub struct Plan<'a> {
    pub tile_h: u32,
    pub tile_w: u32,
    /// The padded row stride in elements: what makes the transposed read conflict-free.
    pub stride: u32,
    /// Extent names, outermost first.
    pub extents: &'a [String],
    /// The single staged buffer, and the register its element lands in.
    pub staged: &'a str,
    pub staged_reg: RegId,
    /// The single drained buffer, and the register holding its value.
    pub drain: &'a str,
    pub drain_reg: RegId,
    /// Shapes, for the row lengths: `(name, last extent)`.
    pub staged_row: &'a str,
    pub drain_row: &'a str,
}

impl<'a> Plan<'a> {
    /// v1 is one staged buffer read, one drained buffer written, rank 2. Anything else is
    /// refused rather than half-emitted.
    pub fn of(ir: &'a KernelIr) -> Result<Self, EmitError> {
        let tile = ir.tile.as_ref().expect("a tiled body has a tile");
        let space = ir.space.as_ref().ok_or_else(|| {
            EmitError::Message("a tile needs a space; the front end should have refused".into())
        })?;
        let layout = ir.shared.as_ref().ok_or_else(|| {
            EmitError::Message(format!(
                "kernel `{}` declares a tile but stages nothing. A tile with no `dram -> smem -> reg` stream changes no traffic; either stage a buffer or drop the tile.",
                ir.name
            ))
        })?;

        let staged: Vec<_> = ir.streams.iter().filter(|s| s.staged).collect();
        if staged.len() != 1 {
            return Err(EmitError::Message(format!(
                "kernel `{}` stages {} buffers; v1 emits one. ADR-0017 covers a single staged tile.",
                ir.name,
                staged.len()
            )));
        }
        let s = staged[0];
        if ir.streams.iter().any(|o| o.read && !o.staged) {
            return Err(EmitError::Message(format!(
                "kernel `{}` reads a buffer that is not staged beside one that is. v1 emits a tile that holds every read, so either stage all of them or none.",
                ir.name
            )));
        }
        if ir.drains.len() != 1 {
            return Err(EmitError::Message(format!(
                "kernel `{}` drains {} buffers; v1 emits one alongside a tile.",
                ir.name,
                ir.drains.len()
            )));
        }
        if ir.reduction.is_some() {
            return Err(EmitError::Message(format!(
                "kernel `{}` reduces and tiles. The tiled reduction is its own ADR: its tree order changes the rounding, so the host oracle has to model the tile before the kernel exists.",
                ir.name
            )));
        }
        let (drain, drain_reg) = &ir.drains[0];
        let row_of = |name: &str| -> Result<&'a str, EmitError> {
            ir.params
                .iter()
                .find(|p| p.name == name)
                .and_then(|p| p.shape.last())
                .map(String::as_str)
                .ok_or_else(|| EmitError::Message(format!("`{name}` has no row length")))
        };

        Ok(Plan {
            tile_h: tile[0],
            tile_w: tile[1],
            stride: layout.tiles[0].2,
            extents: &space.extents,
            staged: &s.buffer,
            staged_reg: s.loaded.ok_or_else(|| {
                EmitError::Message(format!("`{}` is staged but never read", s.buffer))
            })?,
            drain,
            drain_reg: *drain_reg,
            staged_row: row_of(&s.buffer)?,
            drain_row: row_of(drain)?,
        })
    }
}

impl Emitter {
    /// Emit the tiled body. `skew` is the derived padding; passing the unpadded stride is how
    /// ADR-0017's counterfactual fixture is built, and it is the only reason that is possible.
    pub(crate) fn tiled_body(
        &mut self,
        ir: &KernelIr,
        skewed: bool,
        out: &mut String,
    ) -> Result<(), EmitError> {
        let p = Plan::of(ir)?;
        let k = &ir.name;
        let stride = if skewed { p.stride } else { p.tile_w };
        let log2_w = p.tile_w.trailing_zeros();

        // --- parameters ------------------------------------------------------------
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

        let rows = u32_of(&p.extents[0]);
        let cols = u32_of(&p.extents[1]);

        // --- tile counts and the position inside one -------------------------------
        let tiles_y = self.b32();
        let tiles_x = self.b32();
        let total = self.b32();
        line(out, &format!("add.u32 {tiles_y}, {rows}, {};", p.tile_h - 1));
        line(out, &format!("shr.u32 {tiles_y}, {tiles_y}, {};", p.tile_h.trailing_zeros()));
        line(out, &format!("add.u32 {tiles_x}, {cols}, {};", p.tile_w - 1));
        line(out, &format!("shr.u32 {tiles_x}, {tiles_x}, {log2_w};"));
        line(out, &format!("mul.lo.u32 {total}, {tiles_y}, {tiles_x};"));

        // The tile is a power of two wide, so a thread's place in it is a mask and a shift.
        let tid = self.b32();
        let tx = self.b32();
        let ty = self.b32();
        line(out, &format!("mov.u32 {tid}, %tid.x;"));
        line(out, &format!("and.b32 {tx}, {tid}, {};", p.tile_w - 1));
        line(out, &format!("shr.u32 {ty}, {tid}, {log2_w};"));

        let smem = self.b64();
        line(out, &format!("mov.u64 {smem}, lyth_smem;"));

        // --- the loop, over tiles ---------------------------------------------------
        //
        // The bound is the tile index, which every thread of the block shares, so the exit is
        // uniform and no thread can leave while another waits at a barrier.
        let tile = self.b32();
        let step = self.b32();
        let nctaid = self.b32();
        let ctaid = self.b32();
        line(out, &format!("mov.u32 {ctaid}, %ctaid.x;"));
        line(out, &format!("mov.u32 {nctaid}, %nctaid.x;"));
        line(out, &format!("mov.u32 {tile}, {ctaid};"));
        line(out, &format!("mov.u32 {step}, {nctaid};"));

        let _ = writeln!(out, "$L_tile_{k}:");
        let done = self.pred();
        line(out, &format!("setp.ge.u32 {done}, {tile}, {total};"));
        line(out, &format!("@{done} bra $L_tile_end_{k};"));

        let trow = self.b32();
        let tcol = self.b32();
        line(out, &format!("div.u32 {trow}, {tile}, {tiles_x};"));
        line(out, &format!("rem.u32 {tcol}, {tile}, {tiles_x};"));

        // --- phase 2: coalesced global read into the padded tile --------------------
        let gy = self.b32();
        let gx = self.b32();
        line(out, &format!("mad.lo.u32 {gy}, {trow}, {}, {ty};", p.tile_h));
        line(out, &format!("mad.lo.u32 {gx}, {tcol}, {}, {tx};", p.tile_w));

        let in_y = self.pred();
        let in_x = self.pred();
        let in_both = self.pred();
        line(out, &format!("setp.lt.u32 {in_y}, {gy}, {rows};"));
        line(out, &format!("setp.lt.u32 {in_x}, {gx}, {cols};"));
        line(out, &format!("and.pred {in_both}, {in_y}, {in_x};"));

        let lin = self.b32();
        let off = self.b64();
        let addr = self.b64();
        let val = self.f32();
        line(out, &format!("mad.lo.u32 {lin}, {gy}, {}, {gx};", u32_of(p.staged_row)));
        line(out, &format!("mul.wide.u32 {off}, {lin}, 4;"));
        line(out, &format!("add.s64 {addr}, {}, {off};", buf_of(p.staged)));
        // Predicated, never branched: the thread stays in step with its block.
        line(out, &format!("mov.f32 {val}, 0f00000000;"));
        line(out, &format!("@{in_both} ld.global.f32 {val}, [{addr}];"));

        let s_store = self.b32();
        let s_off = self.b64();
        let s_addr = self.b64();
        line(out, &format!("mad.lo.u32 {s_store}, {ty}, {stride}, {tx};"));
        line(out, &format!("mul.wide.u32 {s_off}, {s_store}, 4;"));
        line(out, &format!("add.s64 {s_addr}, {smem}, {s_off};"));
        line(out, &format!("st.shared.f32 [{s_addr}], {val};"));

        // Barrier 1: read-after-write. Nobody reads the tile before it is whole.
        line(out, "bar.sync 0;");

        // --- phase 4: transposed read from shared, coalesced global write -----------
        //
        // The same skewed stride as the store. A read that used the unpadded width would
        // compile, run and produce correct bits while conflicting on every access -- which is
        // why ADR-0017's fixture builds the unpadded variant on purpose.
        let s_load = self.b32();
        let l_off = self.b64();
        let l_addr = self.b64();
        let got = self.f32();
        line(out, &format!("mad.lo.u32 {s_load}, {tx}, {stride}, {ty};"));
        line(out, &format!("mul.wide.u32 {l_off}, {s_load}, 4;"));
        line(out, &format!("add.s64 {l_addr}, {smem}, {l_off};"));
        line(out, &format!("ld.shared.f32 {got}, [{l_addr}];"));

        // The body runs on what came back from shared.
        let mut regs: Vec<(RegId, String)> = vec![(p.staged_reg, got.clone())];
        for op in &ir.ops {
            self.op(op, &mut regs, &u32s, out)?;
        }
        let result = regs
            .iter()
            .find(|(i, _)| *i == p.drain_reg)
            .map(|(_, r)| r.clone())
            .ok_or_else(|| EmitError::Message("the drained value was never computed".into()))?;

        let oy = self.b32();
        let ox = self.b32();
        line(out, &format!("mad.lo.u32 {oy}, {tcol}, {}, {ty};", p.tile_w));
        line(out, &format!("mad.lo.u32 {ox}, {trow}, {}, {tx};", p.tile_h));

        let out_y = self.pred();
        let out_x = self.pred();
        let out_both = self.pred();
        line(out, &format!("setp.lt.u32 {out_y}, {oy}, {cols};"));
        line(out, &format!("setp.lt.u32 {out_x}, {ox}, {rows};"));
        line(out, &format!("and.pred {out_both}, {out_y}, {out_x};"));

        let olin = self.b32();
        let o_off = self.b64();
        let o_addr = self.b64();
        line(out, &format!("mad.lo.u32 {olin}, {oy}, {}, {ox};", u32_of(p.drain_row)));
        line(out, &format!("mul.wide.u32 {o_off}, {olin}, 4;"));
        line(out, &format!("add.s64 {o_addr}, {}, {o_off};", buf_of(p.drain)));
        line(out, &format!("@{out_both} st.global.f32 [{o_addr}], {result};"));

        // Barrier 2: write-after-read. The next tile's load must not overtake this read.
        line(out, "bar.sync 0;");

        line(out, &format!("add.u32 {tile}, {tile}, {step};"));
        line(out, &format!("bra $L_tile_{k};"));
        let _ = writeln!(out, "$L_tile_end_{k}:");
        line(out, "ret;");
        Ok(())
    }
}

/// Emit one straight-line op, shared with the flat body's arithmetic so a tile does not get a
/// second contraction rule.
impl Emitter {
    fn op(
        &mut self,
        op: &Op,
        regs: &mut Vec<(RegId, String)>,
        u32s: &[(String, String)],
        out: &mut String,
    ) -> Result<(), EmitError> {
        let _ = u32s;
        let get = |regs: &[(RegId, String)], id: RegId| -> String {
            regs.iter()
                .find(|(i, _)| *i == id)
                .map(|(_, r)| r.clone())
                .unwrap_or_else(|| panic!("register {id} used before it was defined"))
        };
        match op {
            Op::Load { dst, .. } => {
                // The staged buffer's value is already bound; nothing else is loaded here.
                if !regs.iter().any(|(i, _)| i == dst) {
                    return Err(EmitError::Message(
                        "a tiled kernel loads only the staged buffer".into(),
                    ));
                }
            }
            Op::Const { dst, value } => {
                let r = self.f32();
                line(out, &format!("mov.f32 {r}, {};", crate::hex_f32(*value as f32)));
                regs.push((*dst, r));
            }
            Op::Param { dst, name } => {
                let r = self.f32();
                line(out, &format!("ld.param.f32 {r}, [{name}];"));
                regs.push((*dst, r));
            }
            Op::Neg { dst, src } => {
                let r = self.f32();
                line(out, &format!("neg.f32 {r}, {};", get(regs, *src)));
                regs.push((*dst, r));
            }
            Op::Fma { dst, a, b, c } => {
                let r = self.f32();
                line(
                    out,
                    &format!(
                        "fma.rn.f32 {r}, {}, {}, {};",
                        get(regs, *a),
                        get(regs, *b),
                        get(regs, *c)
                    ),
                );
                regs.push((*dst, r));
            }
            Op::Bin { dst, op, lhs, rhs } => {
                let r = self.f32();
                line(
                    out,
                    &format!(
                        "{} {r}, {}, {};",
                        crate::bin_mnemonic(*op),
                        get(regs, *lhs),
                        get(regs, *rhs)
                    ),
                );
                regs.push((*dst, r));
            }
        }
        Ok(())
    }
}
