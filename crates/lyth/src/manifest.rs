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
    /// True for the `u32` that bounds the index space. Exactly one parameter has this in v1,
    /// and a generator uses it to derive the grid without being told the element count twice.
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
    /// How the default grid is chosen, spelled out so a generator reproduces the compiler's
    /// rule instead of hard-coding one that drifts from it. `ceil(extent / block)` is one
    /// element per thread, which ADR-0012's sweep found faster than filling the machine.
    pub grid_rule: String,
    /// Past this the grid-stride loop engages, because a grid is a `u32`.
    pub max_grid: u32,
    /// True when the kernel is a reduction and the block width must be a power of two: the
    /// tree halves its stride to 1, and any other width leaves elements uncombined.
    pub block_must_be_power_of_two: bool,
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
                    // v1 has exactly one u32 and it bounds the loop. ADR-0015 replaces this
                    // with extents named by a shape, at which point more than one may be true.
                    is_extent: p.ty == Ty::U32,
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
                block,
                shared_bytes: if ir.reduction.is_some() { block * 4 } else { 0 },
                grid_rule: "min(ceil(extent / block), max_grid)".to_string(),
                max_grid,
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
            },
            known_limits,
        }
    }

    /// The parameter that bounds the index space, which a generator needs to size the grid.
    pub fn extent(&self) -> Option<&ParamSpec> {
        self.params.iter().find(|p| p.is_extent)
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
            "machine sm_120\n\nkernel saxpy(n: u32, a: f32, x: [f32], y: [f32])\n\
             \x20   stream x : dram -> reg\n    stream y : dram -> reg, drain\n\
             \x20   at reg:\n        y = a * x + y\n",
        );
        let by = |n: &str| m.params.iter().find(|p| p.name == n).unwrap().clone();
        assert!(by("x").read && !by("x").written, "x is read only");
        assert!(by("y").read && by("y").written, "y is read and drained");
        assert!(by("n").is_extent, "n bounds the index space");
        assert_eq!(m.launch.shared_bytes, 0, "no reduction, no shared memory");
        assert!(!m.launch.block_must_be_power_of_two);
    }

    #[test]
    fn a_reduction_target_is_sized_by_the_grid_and_never_read() {
        // Getting this wrong is an out-of-bounds write in generated code, so it is asserted
        // rather than left to a generator's reading of the name.
        let m = manifest(
            "machine sm_120\n\nkernel total(n: u32, x: [f32], partial: [f32])\n\
             \x20   stream x : dram -> reg\n    reduce sum v : reg -> smem -> dram into partial\n\
             \x20   at reg:\n        v = x\n",
        );
        let p = m.params.iter().find(|p| p.name == "partial").unwrap();
        assert!(p.sized_by_grid, "one element per block, not per element");
        assert!(p.written && !p.read);
        assert_eq!(m.launch.shared_bytes, 256 * 4);
        assert!(m.launch.block_must_be_power_of_two);
    }

    #[test]
    fn the_contract_travels_with_the_signature() {
        let m = manifest(
            "machine sm_120\n\nkernel saxpy(n: u32, a: f32, x: [f32], y: [f32])\n\
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
            "machine sm_120\n\nkernel copy(n: u32, x: [f32], y: [f32])\n\
             \x20   stream x : dram -> reg\n    stream y : dram -> reg, drain\n\
             \x20   at reg:\n        y = x\n",
        );
        let text = serde_json::to_string_pretty(&m).unwrap();
        let back: Manifest = serde_json::from_str(&text).unwrap();
        assert_eq!(m, back, "a generator reads what the compiler wrote");
    }
}
