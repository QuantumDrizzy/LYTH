//! ADR-0021 step 2: `coarsen` parses and resolves, and the block width follows from it.
//!
//! A tile puts one thread on each of its elements, so the block is the tile's area — which is
//! why a tile of 64 asks for 4096 threads against a cap of 1024. `coarsen 2, 2` decouples the
//! two: the tile stays the working set the traffic is derived from, and this says how many of
//! its outputs one thread owns.
//!
//! **The cost model does not appear in this file**, and that absence is the design being right.
//! Reuse is a property of the tile — how many threads want one staged element — and coarsening
//! changes which thread computes what, not what is staged. `tile 64, 64` derived
//! `0.125 * k + 4` before `coarsen` existed and derives it still; `symbolic_cost.rs` owns that.

use lyth_lang::ir::{lower, KernelIr, LowerError};
use lyth_lang::parse::parse;

/// A matmul with the tile and coarsening spliced in.
fn matmul(tile: &str, coarsen: Option<&str>) -> String {
    let line = coarsen.map(|c| format!("    coarsen {c}\n")).unwrap_or_default();
    format!(
        "machine sm_120\n\n\
         kernel mm(m: u32, n: u32, k: u32,\n              \
         a: [f32; m, k], b: [f32; k, n], c: [f32; m, n])\n    \
         space i, j : m, n\n    \
         contract sum p : k\n    \
         tile {tile}\n\
         {line}    \
         stream a : dram -> smem -> reg\n    \
         stream b : dram -> smem -> reg\n    \
         stream c : dram -> reg, drain\n    \
         at reg:\n        \
         c[i, j] = a[i, p] * b[p, j]\n"
    )
}

fn ir(src: &str) -> Result<KernelIr, LowerError> {
    let u = parse(src).expect("should parse");
    lower(&u, &u.kernels[0])
}

fn err(src: &str) -> String {
    match ir(src) {
        Err(e) => e.to_string(),
        Ok(_) => panic!("expected a refusal, got a kernel"),
    }
}

#[test]
fn the_block_is_the_tile_divided_by_what_each_thread_owns() {
    // The whole of step 2 in one assertion. 64 x 64 is 4096 elements; at 2 x 2 each, 1024
    // threads cover it -- which is exactly the cap a tile of 64 could not meet before.
    let k = ir(&matmul("64, 64", Some("2, 2"))).expect("lowers");
    assert_eq!(k.coarsen.as_deref(), Some(&[2u32, 2][..]));
    assert_eq!(k.block_threads(256), 1024);

    // Rectangular coarsening is legal and is not the same number.
    let k = ir(&matmul("64, 64", Some("2, 4"))).expect("lowers");
    assert_eq!(k.block_threads(256), 512);

    // And a tile of 128 needs 4 x 4 to fit a 1024-thread block.
    let k = ir(&matmul("128, 128", Some("4, 4"))).expect("lowers");
    assert_eq!(k.block_threads(256), 1024);
}

#[test]
fn a_tile_without_coarsening_is_unchanged_and_an_untiled_kernel_takes_the_fallback() {
    // The line that is not written has to mean what it meant before.
    let k = ir(&matmul("32, 32", None)).expect("lowers");
    assert!(k.coarsen.is_none());
    assert_eq!(k.block_threads(256), 1024, "one thread per tile element");

    let saxpy = "machine sm_120\n\n\
                 kernel k(n: u32, a: f32, x: [f32; n], y: [f32; n])\n    \
                 intensity 0.1667\n    \
                 stream x : dram -> reg\n    \
                 stream y : dram -> reg, drain\n    \
                 at reg:\n        \
                 y = a * x + y\n";
    let k = ir(saxpy).expect("lowers");
    assert_eq!(k.block_threads(256), 256, "no tile, so the launch chooses");
    assert_eq!(k.block_threads(1024), 1024);
}

#[test]
fn coarsening_does_not_touch_the_derived_traffic() {
    // The property that makes `coarsen` a separate declaration rather than a change to `tile`.
    // Same tile, same bytes, whether one thread owns one output of it or four.
    let plain = ir(&matmul("64, 64", None)).expect("lowers");
    let coarse = ir(&matmul("64, 64", Some("2, 2"))).expect("lowers");
    assert_eq!(plain.cost.bytes_expr(), coarse.cost.bytes_expr());
    assert_eq!(plain.cost.bytes_expr().as_deref(), Some("0.125 * k + 4"));
    assert_eq!(plain.cost.intensity, coarse.cost.intensity);
    assert_eq!(plain.cost.intensity, 16.0);
    // The shared memory is the tile's too, for the same reason.
    assert_eq!(
        plain.shared.as_ref().map(|l| l.bytes),
        coarse.shared.as_ref().map(|l| l.bytes)
    );
}

#[test]
fn a_factor_that_does_not_divide_its_tile_edge_is_refused() {
    // Each thread owns a whole sub-rectangle, so a factor has to split its edge evenly.
    //
    // The factor has to be a power of two as well, and the parser enforces that first -- so a
    // test for *this* rule needs a factor that is a power of two and still does not divide.
    // 64 over a tile edge of 32 is that case. Writing `2, 3` here tested the parser twice.
    let e = err(&matmul("32, 32", Some("2, 64")));
    assert!(e.contains("axis 1"), "{e}");
    assert!(e.contains("tile of 32 coarsened by 64"), "{e}");
}

#[test]
fn a_factor_that_is_not_a_power_of_two_is_refused_where_it_is_written() {
    // Refused in the parser, like a tile dimension, and for the same reason: the block's width
    // is the tile's divided by this, and a thread's position stays a shift and a mask only if
    // every one of them is a power of two.
    let e = parse(&matmul("64, 64", Some("2, 3"))).expect_err("3 is not a power of two");
    // And it is the parser that says so, before lowering ever sees the number.
    assert!(e.to_string().contains("power of two"), "{e}");
    assert!(e.to_string().contains("shift and a mask"), "{e}");
}

#[test]
fn coarsening_without_a_tile_is_refused() {
    let saxpy = "machine sm_120\n\n\
                 kernel k(n: u32, a: f32, x: [f32; n], y: [f32; n])\n    \
                 coarsen 2\n    \
                 stream x : dram -> reg\n    \
                 stream y : dram -> reg, drain\n    \
                 at reg:\n        \
                 y = a * x + y\n";
    let e = err(saxpy);
    assert!(e.contains("without a `tile`"), "{e}");
    assert!(e.contains("nothing to spread"), "{e}");
}

#[test]
fn one_factor_per_tile_axis() {
    let e = err(&matmul("64, 64", Some("2")));
    assert!(e.contains("1 factors and `tile` has 2 dimensions"), "{e}");
}

#[test]
fn the_identity_is_refused_rather_than_accepted_as_a_no_op() {
    // `coarsen 1, 1` is what a tile does without the line. A declaration that changes nothing
    // is a line a reader has to check and then discard.
    let e = err(&matmul("32, 32", Some("1, 1")));
    assert!(e.contains("what a tile does without the line"), "{e}");
}

#[test]
fn a_second_coarsen_is_refused() {
    let src = matmul("64, 64", Some("2, 2")).replace(
        "    coarsen 2, 2\n",
        "    coarsen 2, 2\n    coarsen 4, 4\n",
    );
    let e = parse(&src).expect_err("two coarsen lines");
    assert!(e.to_string().contains("declared twice"), "{e}");
}

#[test]
fn the_shared_traffic_sees_the_coarsening_and_the_global_traffic_does_not() {
    // ADR-0022 step 2. This was a `[KNOWN_LIMIT]` pinned at the wrong value for one commit,
    // because ADR-0021 step 5 measured the compiler deriving `8 * k` shared read per output at
    // `coarsen 2, 2` while the profiler counted **exactly half**: 268,435,456 shared-load
    // instructions against 536,870,912 uncoarsened, at 2048 cubed.
    //
    // The rule is `reuse_of` one level down. Shared memory earns its reuse from the tile --
    // how many threads want one staged element. The register file earns its from the
    // coarsening: a thread owning `cj` outputs along `j` loads `a[i, p]` once and spends it
    // `cj` times. So the divisor is the product of the coarsening of the free axes the index
    // does not mention, which is exactly the tile rule with `coarsen` substituted.
    //
    // `2 * ci * cj / (ci + cj)` flops per shared access is what coarsening actually buys, and
    // ADR-0022 measured that the shared pipe -- not DRAM -- is what a tiled contraction on
    // this device runs out of first.
    let plain = ir(&matmul("64, 64", None)).expect("lowers");
    let sq = ir(&matmul("64, 64", Some("2, 2"))).expect("lowers");
    let wide = ir(&matmul("64, 64", Some("2, 4"))).expect("lowers");
    let deep = ir(&matmul("64, 64", Some("4, 4"))).expect("lowers");

    let smem_read = |k: &lyth_lang::ir::KernelIr| {
        k.cost.traffic_words(lyth_lang::ast::Level::Smem).0
    };

    // `ci + cj` loads feed `ci * cj` bodies, so per output it is `(ci + cj) / (ci * cj)`
    // loads of four bytes, times `k` terms.
    assert_eq!(smem_read(&plain), "8 * k", "2 loads per output per term");
    assert_eq!(smem_read(&sq), "4 * k", "4 loads for 4 outputs: one each");
    assert_eq!(smem_read(&wide), "3 * k", "6 loads for 8 outputs");
    assert_eq!(smem_read(&deep), "2 * k", "8 loads for 16 outputs");

    // And the global figure does not move, which is the ADR-0021 claim that survived: reuse at
    // that level is the tile's, and all four of these have the same tile. Measured to +0.80%.
    for k in [&sq, &wide, &deep] {
        assert_eq!(k.cost.bytes_expr(), plain.cost.bytes_expr());
        assert_eq!(k.cost.bytes_expr().as_deref(), Some("0.125 * k + 4"));
        assert_eq!(k.cost.intensity, 16.0);
    }
}

#[test]
fn an_uncoarsened_kernel_derives_exactly_what_it_derived_before() {
    // The other half of step 2. `register_reuse_of` returns 1 without a `coarsen`, so every
    // kernel written before ADR-0021 has to be bit-identical in its cost -- and a staged
    // transpose, which has no contraction at all, must not acquire one.
    let a = ir(&matmul("32, 32", None)).expect("lowers");
    assert_eq!(
        a.cost.traffic_words(lyth_lang::ast::Level::Smem).0,
        "8 * k",
        "the figure ADR-0018 shipped"
    );
    assert_eq!(a.cost.bytes_expr().as_deref(), Some("0.25 * k + 4"));
    assert_eq!(a.cost.intensity, 8.0);
}
