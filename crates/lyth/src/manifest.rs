//! What a caller needs to know to launch a kernel it did not compile.
//!
//! The evidence schema (`lyth-intensity/0.1`) carries the accounting and nothing about the
//! signature, because it was written for `lyth-probe` to check traffic rather than for anyone
//! to call the kernel. ADR-0016 needs both, and needs them in one file that a binding
//! generator can read without linking the compiler.
//!
//! Three kinds of fact live here and they are deliberately kept apart:
//!
//! * **the signature** — what to pass, in what order, and which buffer the kernel writes;
//! * **the launch** — grid, block, shared bytes and how the reduction target is sized, all
//!   derived rather than left for a caller to rediscover;
//! * **the contract** — the declared intensity, the derived cost, and the machine it was
//!   checked against, so the claim travels with the artifact instead of living in a comment.
//!
//! The third is the point of the project. A kernel that is bandwidth-bound on sm_120 should
//! say so in the code that calls it, and a build on different hardware should be able to
//! assert against it rather than find out from a profiler months later.

use lyth_lang::ast::Ty;
use lyth_lang::KernelIr;
use serde::{Deserialize, Serialize};

pub const SCHEMA: &str = "lyth-manifest/0.1";

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Manifest {
    pub schema: String,
    /// The PTX entry point, which is also the kernel name in source.
    pub kernel: String,
    pub params: Vec<ParamSpec>,
    pub launch: LaunchSpec,
    pub contract: ContractSpec,
    /// Carried verbatim from the compiler so a caller reads the same warnings the compiler
    /// prints, rather than a summary of them.
    pub known_limits: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ParamSpec {
    pub name: String,
    /// `u32`, `f32` or `buf_f32`. Spelled rather than encoded so the file reads without a key.
    pub ty: String,
    /// A buffer's extents, named: `["n"]`, or `["blocks"]` for a reduction target. Empty for
    /// a scalar. A generator sizes an allocation from this and from nothing else.
    pub shape: Vec<String>,
    /// True for a `u32` that some buffer names as an extent, so a generator can tell a length
    /// from a count that merely happens to be a `u32`.
    pub is_extent: bool,
    /// A buffer the kernel writes. Drives `&mut` in Rust and non-const in C.
    pub written: bool,
    /// A buffer the kernel reads.
    pub read: bool,
    /// The reduction target: one element per block, so it is sized by the grid and **not** by
    /// the extent. Getting this wrong is an out-of-bounds write, so it is stated rather than
    /// inferred from the name.
    pub sized_by_grid: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LaunchSpec {
    pub block: u32,
    /// Dynamic shared memory in bytes. A reduction sizes this by the block; passing 0 to a
    /// kernel that declares `.extern .shared` is not an error the driver reports — the array
    /// is simply empty and the kernel reads whatever is there.
    pub shared_bytes: u32,
    /// How the default grid is chosen, as data a generator evaluates rather than a sentence
    /// it has to parse. A free-text rule is a rule three generators quietly ignore while
    /// hard-coding their own, which is how they drift apart.
    pub grid: GridRule,
    /// Past this the grid-stride loop engages, because a grid is a `u32`.
    pub max_grid: u32,
    /// What the tile costs in shared memory, and the claim the skew makes.
    ///
    /// The caller never computes this: `shared_bytes` is embedded in the generated launch, so
    /// the signature of `launch` does not change when a kernel becomes tiled.
    pub predicted_bank_conflicts: u32,
    /// True when the kernel is a reduction and the block width must be a power of two: the
    /// tree halves its stride to 1, and any other width leaves elements uncombined.
    pub block_must_be_power_of_two: bool,
}

/// The grid the default launch uses.
///
/// Elements a thread must be given before the launch shape stops being pathological.
///
/// One, for an elementwise kernel: ADR-0012 swept `saxpy` and one element per thread was
/// fastest. Eight for a **reduction**, where the block's shared-memory tree runs once per
/// thread and therefore, at one element per thread, once per element. Measured on `sum` at
/// n = 2^26: 213.69 GB/s against 419.68.
///
/// This function is the single definition of that rule. `cmd_run` and the manifest both call
/// it, because they did not: the run path was fixed and the manifest kept emitting the
/// elementwise grid, so `lyth run` and the generated bindings disagreed about how to launch
/// the same kernel. Found by using a binding in another repository, not by a test -- and the
/// test below now asserts the two agree.
/// The scalar parameters that are some buffer's extent, in declaration order -- which is the
/// order a generated signature takes them in.
pub fn extent_params(ir: &KernelIr) -> Vec<String> {
    let named: std::collections::BTreeSet<&str> = ir
        .params
        .iter()
        .flat_map(|p| p.shape.iter().map(String::as_str))
        .collect();
    ir.params
        .iter()
        .filter(|p| p.ty == Ty::U32 && named.contains(p.name.as_str()))
        .map(|p| p.name.clone())
        .collect()
}

impl GridRule {
    /// The rule this kernel launches under. One definition, used by the manifest and by
    /// `lyth run` alike.
    pub fn of(ir: &KernelIr, extents: Vec<String>) -> Self {
        match (&ir.tile, &ir.space) {
            (Some(t), Some(sp)) => GridRule::Tiled {
                tile: t.clone(),
                extents: sp.extents.clone(),
            },
            _ => GridRule::Elementwise {
                extents,
                min_elements_per_thread: min_elements_per_thread(ir),
            },
        }
    }

    /// Blocks the default launch uses, evaluated.
    ///
    /// The three generators emit this arithmetic in Rust, C and Python, and `lyth run`
    /// performs it here; this is the definition all four are checked against. `extent` maps
    /// an extent parameter's name to its value, and any name it does not know is 1 -- which
    /// is what `lyth run` does for an extent nobody passed.
    pub fn blocks(&self, extent: &dyn Fn(&str) -> u32, block: u32, max_grid: u32) -> u32 {
        let n = match self {
            GridRule::Elementwise {
                extents,
                min_elements_per_thread,
            } => {
                let total: u64 = extents.iter().map(|e| extent(e) as u64).product();
                // In 64 bits, because a rank-2 product of 32-bit extents does not fit in 32.
                let per_block = (block as u64) * (*min_elements_per_thread as u64);
                total.div_ceil(per_block.max(1))
            }
            GridRule::Tiled { tile, extents } => extents
                .iter()
                .zip(tile)
                .map(|(e, t)| extent(e).div_ceil(*t) as u64)
                .product(),
        };
        n.clamp(1, max_grid as u64) as u32
    }
}

pub fn min_elements_per_thread(ir: &KernelIr) -> u32 {
    if ir.reduction.is_some() {
        8
    } else {
        1
    }
}

/// `Elementwise` is one element per thread over the product of the extents. `Tiled` is one
/// block per tile, which is a different formula and not a different constant -- a generator
/// that hard-coded the first would launch a fraction of the work on a tiled kernel and
/// compile cleanly doing it.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum GridRule {
    /// `min(ceil(product(extents) / (block * min_elements_per_thread)), max_grid)`
    ///
    /// `min_elements_per_thread` is 1 for an elementwise kernel, where ADR-0012 measured one
    /// element per thread fastest, and 8 for a **reduction**, where it is twice as slow: the
    /// block's shared-memory tree runs once per thread, so one element per thread runs it once
    /// per element. Measured on `sum` at n = 2^26: 213.69 GB/s against 419.68.
    ///
    /// It is a field rather than a second variant because a generator that did not know about
    /// it would silently emit the elementwise rule -- which is exactly what happened, and was
    /// found by using the bindings rather than by a test.
    Elementwise {
        extents: Vec<String>,
        min_elements_per_thread: u32,
    },
    /// `product(ceil(extent[d] / tile[d]))`, one block per tile.
    Tiled { tile: Vec<u32>, extents: Vec<String> },
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ContractSpec {
    pub machine: String,
    /// What the source declared, if it declared anything. `None` is not a failure: a kernel
    /// may stay silent about its intensity and still be compiled.
    pub declared_intensity: Option<f64>,
    pub derived_intensity: f64,
    pub flops_per_element: f64,
    pub bytes_per_element: f64,
    pub read_bytes_per_element: f64,
    pub write_bytes_per_element: f64,
    /// A reduction writes one partial per block rather than per element, so this is reported
    /// beside the per-element figures and never folded into them.
    pub dram_bytes_per_block: f64,
    /// What the bus carries, at 32-byte sector granularity, against what the source asked
    /// for. `1.0` is every fetched byte wanted. Below that, `strided` names the buffers.
    pub coalescence: f64,
    pub sector_read_per_element: f64,
    pub sector_write_per_element: f64,
    pub strided_buffers: Vec<String>,
    /// Traffic per interface, each with the counter that measures it.
    ///
    /// The level names the **interface the figure is a claim about**, which is not always the
    /// level the source declared. `stream x : dram -> reg` says where the data lives and that
    /// is true; the sector model was measured exact at the L1-to-L2 interface and wrong at
    /// DRAM in both directions (ADR-0015), so the figure derived from it is an `l2` claim.
    /// This file is what a CI job asserts against, and a level named after the wrong
    /// interface is a cost contract that lies quietly.
    ///
    /// **Nothing here is a DRAM claim.** DRAM traffic is a function of how far the working set
    /// exceeds the cache, which this compiler does not model and does not pretend to.
    ///
    /// One number per interface, because a byte in shared memory and a byte at DRAM are not
    /// the same byte. A single figure would have to pick one and would be wrong about the
    /// others — which is the mistake ADR-0015 made and had to correct after measuring.
    pub bus: Vec<LevelTraffic>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LevelTraffic {
    pub level: String,
    pub read_per_element: f64,
    pub write_per_element: f64,
    /// The Nsight Compute counter this figure is a claim about. Naming it here is what stops
    /// a number being compared against whatever counter came to hand.
    pub counter: String,
}

impl Manifest {
    pub fn of(
        ir: &KernelIr,
        declared: Option<f64>,
        block: u32,
        max_grid: u32,
        known_limits: Vec<String>,
    ) -> Self {
        let reduce_target = ir.reduction.as_ref().map(|r| r.into.as_str());

        // A u32 is an extent when a buffer says so, not when it is the only one of its type.
        let named: std::collections::BTreeSet<&str> = ir
            .params
            .iter()
            .flat_map(|p| p.shape.iter().map(String::as_str))
            .collect();
        let extents = extent_params(ir);

        let params = ir
            .params
            .iter()
            .map(|p| {
                let stream = ir.streams.iter().find(|s| s.buffer == p.name);
                let is_partial = Some(p.name.as_str()) == reduce_target;
                ParamSpec {
                    name: p.name.clone(),
                    ty: match p.ty {
                        Ty::U32 => "u32",
                        Ty::F32 => "f32",
                        Ty::BufF32 => "buf_f32",
                    }
                    .to_string(),
                    shape: p.shape.clone(),
                    is_extent: p.ty == Ty::U32 && named.contains(p.name.as_str()),
                    written: is_partial
                        || stream.map(|s| s.drain).unwrap_or(false)
                        || ir.drains.iter().any(|(b, _)| *b == p.name),
                    read: stream.map(|s| s.read).unwrap_or(false),
                    sized_by_grid: is_partial,
                }
            })
            .collect();

        Manifest {
            schema: SCHEMA.to_string(),
            kernel: ir.name.clone(),
            params,
            launch: LaunchSpec {
                // A tile fixes the block: one thread per element of the tile. Otherwise the
                // compiler's default, which `--block` may override.
                block: match &ir.tile {
                    Some(t) => t.iter().product(),
                    None => block,
                },
                shared_bytes: match (&ir.shared, ir.reduction.is_some()) {
                    (Some(l), _) => l.bytes,
                    (None, true) => block * 4,
                    (None, false) => 0,
                },
                grid: GridRule::of(ir, extents),
                max_grid,
                predicted_bank_conflicts: ir
                    .shared
                    .as_ref()
                    .map(|l| l.predicted_bank_conflicts)
                    .unwrap_or(0),
                block_must_be_power_of_two: ir.reduction.is_some(),
            },
            contract: ContractSpec {
                machine: ir.machine.clone(),
                declared_intensity: declared,
                derived_intensity: ir.cost.intensity,
                flops_per_element: ir.cost.flops_per_element,
                bytes_per_element: ir.cost.bytes_per_element(),
                read_bytes_per_element: ir.cost.read_bytes_per_element(),
                write_bytes_per_element: ir.cost.write_bytes_per_element(),
                dram_bytes_per_block: ir.cost.dram_bytes_per_block,
                coalescence: ir.cost.coalescence(),
                sector_read_per_element: ir.cost.sector_read_per_element,
                sector_write_per_element: ir.cost.sector_write_per_element,
                bus: ir
                    .cost
                    .levels
                    .iter()
                    .filter(|l| l.total() > 0.0)
                    .map(|l| LevelTraffic {
                        level: match l.level {
                            lyth_lang::ast::Level::Dram | lyth_lang::ast::Level::L2 => "l2",
                            other => other.name(),
                        }
                        .to_string(),
                        read_per_element: if l.level == lyth_lang::ast::Level::Dram {
                            ir.cost.sector_read_per_element
                        } else {
                            l.read
                        },
                        write_per_element: if l.level == lyth_lang::ast::Level::Dram {
                            ir.cost.sector_write_per_element
                        } else {
                            l.write
                        },
                        // The level a figure is a claim about is the level its counter
                        // measures, not the level the source declared the buffer to live in.
                        // `stream x : dram -> reg` is true about where the data is; the sector
                        // model was measured exact at the L1-to-L2 interface and wrong at DRAM
                        // in both directions (ADR-0015), so calling this `dram` while naming
                        // an L2 counter would put a lie in the file CI asserts against.
                        counter: match l.level {
                            lyth_lang::ast::Level::Dram => "lts__t_bytes.sum",
                            lyth_lang::ast::Level::L2 => "lts__t_bytes.sum",
                            lyth_lang::ast::Level::Smem => {
                                "l1tex__data_pipe_lsu_wavefronts_mem_shared.sum"
                            }
                            lyth_lang::ast::Level::Reg => "",
                        }
                        .to_string(),
                    })
                    .collect(),
                strided_buffers: ir
                    .streams
                    .iter()
                    .filter(|s| !s.coalesced)
                    .map(|s| s.buffer.clone())
                    .collect(),
            },
            known_limits,
        }
    }

    /// Buffers a caller allocates by element count, and the one it allocates by block count.
    pub fn sized_by_grid(&self) -> Option<&ParamSpec> {
        self.params.iter().find(|p| p.sized_by_grid)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lyth_lang::{ir::lower, parse::parse};

    fn manifest(src: &str) -> Manifest {
        let unit = parse(src).unwrap();
        let ir = lower(&unit, &unit.kernels[0]).unwrap();
        Manifest::of(&ir, unit.kernels[0].declared_intensity, 256, 1 << 20, vec![])
    }

    #[test]
    fn a_drained_buffer_is_written_and_a_streamed_one_is_read() {
        let m = manifest(
            "machine sm_120\n\nkernel saxpy(n: u32, a: f32, x: [f32; n], y: [f32; n])\n\
             \x20   stream x : dram -> reg\n    stream y : dram -> reg, drain\n\
             \x20   at reg:\n        y = a * x + y\n",
        );
        let by = |n: &str| m.params.iter().find(|p| p.name == n).unwrap().clone();
        assert!(by("x").read && !by("x").written, "x is read only");
        assert!(by("y").read && by("y").written, "y is read and drained");
        assert!(by("n").is_extent, "n bounds the index space");
        assert_eq!(by("x").shape, vec!["n".to_string()]);
        assert!(by("a").shape.is_empty(), "a scalar has no shape");
        assert_eq!(m.launch.shared_bytes, 0, "no reduction, no shared memory");
        assert!(!m.launch.block_must_be_power_of_two);
    }

    #[test]
    fn a_reduction_target_is_sized_by_the_grid_and_never_read() {
        // Getting this wrong is an out-of-bounds write in generated code, so it is asserted
        // rather than left to a generator's reading of the name.
        let m = manifest(
            "machine sm_120\n\nkernel total(n: u32, x: [f32; n], partial: [f32; blocks])\n\
             \x20   stream x : dram -> reg\n    reduce sum v : reg -> smem -> dram into partial\n\
             \x20   at reg:\n        v = x\n",
        );
        let p = m.params.iter().find(|p| p.name == "partial").unwrap();
        assert!(p.sized_by_grid, "one element per block, not per element");
        assert_eq!(p.shape, vec!["blocks".to_string()], "and it says so in the source");
        assert!(p.written && !p.read);
        assert_eq!(m.launch.shared_bytes, 256 * 4);
        assert!(m.launch.block_must_be_power_of_two);
    }

    #[test]
    fn the_contract_travels_with_the_signature() {
        let m = manifest(
            "machine sm_120\n\nkernel saxpy(n: u32, a: f32, x: [f32; n], y: [f32; n])\n\
             \x20   intensity 0.1667\n    stream x : dram -> reg\n\
             \x20   stream y : dram -> reg, drain\n    at reg:\n        y = a * x + y\n",
        );
        assert_eq!(m.contract.machine, "sm_120");
        assert_eq!(m.contract.declared_intensity, Some(0.1667));
        assert_eq!(m.contract.flops_per_element, 2.0);
        assert_eq!(m.contract.bytes_per_element, 12.0);
        assert_eq!(m.contract.read_bytes_per_element, 8.0);
        assert_eq!(m.contract.write_bytes_per_element, 4.0);
    }

    #[test]
    fn a_manifest_round_trips_through_json() {
        let m = manifest(
            "machine sm_120\n\nkernel copy(n: u32, x: [f32; n], y: [f32; n])\n\
             \x20   stream x : dram -> reg\n    stream y : dram -> reg, drain\n\
             \x20   at reg:\n        y = x\n",
        );
        let text = serde_json::to_string_pretty(&m).unwrap();
        let back: Manifest = serde_json::from_str(&text).unwrap();
        assert_eq!(m, back, "a generator reads what the compiler wrote");
    }
}

#[cfg(test)]
mod tiled {
    use super::*;
    use lyth_lang::{ir::lower, parse::parse};

    const SRC: &str = "machine sm_120\n\nkernel t(rows: u32, cols: u32, a: [f32; rows, cols], b: [f32; cols, rows])\n    space i, j : rows, cols\n    tile 32, 32\n    stream a : dram -> smem -> reg\n    stream b : dram -> reg, drain\n    at reg:\n        b[j, i] = a[i, j]\n";

    fn manifest(src: &str) -> Manifest {
        let unit = parse(src).unwrap();
        let ir = lower(&unit, &unit.kernels[0]).unwrap();
        Manifest::of(&ir, None, 256, 1 << 20, vec![])
    }

    #[test]
    fn a_tile_fixes_the_block_and_the_shared_memory() {
        let m = manifest(SRC);
        // One thread per element of the tile, so the compiler's 256 default does not apply.
        assert_eq!(m.launch.block, 1024);
        // 32 rows of 33 elements, which is the skew, not 32 x 32.
        assert_eq!(m.launch.shared_bytes, 4224);
        assert_eq!(m.launch.predicted_bank_conflicts, 0);
    }

    #[test]
    fn the_grid_rule_is_data_and_says_it_is_tiled() {
        // A generator evaluates this. A sentence would be a sentence three generators ignore
        // while hard-coding the elementwise rule, which would launch a fraction of the work.
        let m = manifest(SRC);
        match &m.launch.grid {
            GridRule::Tiled { tile, extents } => {
                assert_eq!(tile, &vec![32, 32]);
                assert_eq!(extents, &vec!["rows".to_string(), "cols".to_string()]);
            }
            other => panic!("expected a tiled rule, got {other:?}"),
        }
    }

    #[test]
    fn an_untiled_kernel_keeps_the_elementwise_rule_and_no_shared_memory() {
        let src = "machine sm_120\n\nkernel k(n: u32, x: [f32; n], y: [f32; n])\n    stream x : dram -> reg\n    stream y : dram -> reg, drain\n    at reg:\n        y = x\n";
        let m = manifest(src);
        assert_eq!(m.launch.block, 256);
        assert_eq!(m.launch.shared_bytes, 0);
        assert!(matches!(m.launch.grid, GridRule::Elementwise { .. }));
    }

    #[test]
    fn the_extents_keep_declaration_order() {
        // A generated signature takes them in this order, so `rows` and `cols` swapping would
        // transpose a different matrix than the caller asked for and compile cleanly.
        let m = manifest(SRC);
        let names: Vec<&str> = m
            .params
            .iter()
            .filter(|p| p.is_extent)
            .map(|p| p.name.as_str())
            .collect();
        assert_eq!(names, vec!["rows", "cols"]);
    }
}

#[cfg(test)]
mod reduction_grid {
    use super::*;
    use lyth_lang::{ir::lower, parse::parse};

    const SUM: &str = "machine sm_120

kernel s(n: u32, x: [f32; n], partial: [f32; blocks])
    intensity 0.25
    stream x : dram -> reg
    reduce sum v : reg -> smem -> dram into partial
    at reg:
        v = x
";
    const SAXPY: &str = "machine sm_120

kernel k(n: u32, a: f32, x: [f32; n], y: [f32; n])
    intensity 0.1667
    stream x : dram -> reg
    stream y : dram -> reg, drain
    at reg:
        y = a * x + y
";

    fn manifest(src: &str) -> Manifest {
        let unit = parse(src).unwrap();
        let ir = lower(&unit, &unit.kernels[0]).unwrap();
        Manifest::of(&ir, None, 256, 1 << 20, vec![])
    }

    fn floor(m: &Manifest) -> u32 {
        match &m.launch.grid {
            GridRule::Elementwise {
                min_elements_per_thread,
                ..
            } => *min_elements_per_thread,
            other => panic!("expected an elementwise rule, got {other:?}"),
        }
    }

    #[test]
    fn a_reduction_does_not_publish_one_element_per_thread() {
        // The defect this test exists for: `cmd_run` was fixed to amortise a reduction's block
        // tree and the manifest was not, so `lyth run` and every generated binding launched the
        // same kernel differently -- the bindings at half the achievable bandwidth. It was found
        // by calling a binding from another repository, which is a worse place to find it.
        assert_eq!(floor(&manifest(SUM)), 8);
        assert_eq!(floor(&manifest(SAXPY)), 1, "an elementwise kernel is unchanged");
    }

    #[test]
    fn the_published_rule_evaluates_to_what_lyth_run_launches() {
        // Both call `GridRule::blocks`. This asserts the arithmetic itself, so a generator
        // emitting the formula by hand has something to be compared against.
        let n = 1u32 << 26;
        let at = |v: u32| move |_: &str| v;
        let sum = manifest(SUM);
        let f = at(n);
        assert_eq!(sum.launch.grid.blocks(&f, 256, 1 << 20), 32768);
        let saxpy = manifest(SAXPY);
        assert_eq!(saxpy.launch.grid.blocks(&f, 256, 1 << 20), 262144);
    }

    #[test]
    fn the_grid_is_at_least_one_block_and_never_above_the_cap() {
        let sum = manifest(SUM);
        // Fewer elements than a single block amortises: still one block, not zero.
        assert_eq!(sum.launch.grid.blocks(&|_| 1, 256, 1 << 20), 1);
        assert_eq!(sum.launch.grid.blocks(&|_| 0, 256, 1 << 20), 1);
        // And the cap binds before the extent does.
        assert_eq!(sum.launch.grid.blocks(&|_| u32::MAX, 256, 64), 64);
    }
}
