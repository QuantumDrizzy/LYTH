//! ADR-0024 step 1: the element's width stops being a constant.
//!
//! `derive_cost` opened with `let elem = Ty::BufF32.bytes()` for the project's whole life --
//! the entire cost model's element size, written once, at module scope, because the language
//! had one buffer type. The model was already expressed in terms of *an* element size, so the
//! change at this level is a lookup instead of a literal.
//!
//! These tests are the reason step 1 is worth having on its own: none of them needs a GPU, and
//! every one of them would fail loudly if the width were still assumed anywhere.

use lyth_lang::ast::Level;
use lyth_lang::{ir, parse};

fn cost(src: &str) -> ir::Cost {
    let unit = parse(src).expect("parses");
    ir::lower(&unit, &unit.kernels[0]).expect("lowers").cost
}

fn saxpy(elem: &str, declared: &str) -> String {
    format!(
        "machine sm_120\n\n\
         kernel saxpy(n: u32, a: f32, x: [{elem}; n], y: [{elem}; n])\n    \
         intensity {declared}\n    \
         stream x : dram -> reg\n    \
         stream y : dram -> reg, drain\n    \
         at reg:\n        \
         y = a * x + y\n"
    )
}

#[test]
fn halving_the_element_halves_the_traffic_and_doubles_the_intensity() {
    // Two reads and one write. At four bytes that is 12 per element and the intensity this
    // project has quoted since ADR-0001; at two it is 6 and twice the intensity, and the flops
    // do not move because the arithmetic does not narrow.
    let wide = cost(&saxpy("f32", "0.1667"));
    assert_eq!(wide.bytes_per_element(), Some(12.0));
    assert!((wide.intensity - 1.0 / 6.0).abs() < 1e-4);

    let narrow = cost(&saxpy("f16", "0.3333"));
    assert_eq!(narrow.bytes_per_element(), Some(6.0));
    assert!((narrow.intensity - 1.0 / 3.0).abs() < 1e-4);

    assert_eq!(wide.flops_per_element(), narrow.flops_per_element());
}

#[test]
fn the_compiler_refuses_the_declaration_that_did_not_move_with_the_width() {
    // The contract is the point of the language, and a width change is exactly the kind of
    // edit that leaves a declaration quietly stale. It is refused, with the new figure named.
    let unit = parse(&saxpy("f16", "0.1667")).expect("parses");
    let e = ir::lower(&unit, &unit.kernels[0])
        .map(|_| ())
        .err()
        .map(|e| e.to_string())
        .or_else(|| {
            // The intensity check lives past lowering; drive it the way `lyth check` does.
            let k = ir::lower(&unit, &unit.kernels[0]).unwrap();
            lyth_lang::check::check_intensity(&k, Some(0.1667), None, 0.05)
                .err()
                .map(|e| e.to_string())
        })
        .expect("a stale declaration must not pass");
    assert!(e.contains("0.3333"), "{e}");
}

#[test]
fn bf16_and_f16_are_indistinguishable_to_the_traffic_model() {
    // ADR-0024 predicts this rather than assuming it: two bytes is two bytes, so the model
    // says they will take the same time and differ only in accuracy. Asserting it here is what
    // makes the later measurement a test of the prediction instead of a restatement of it.
    //
    // bf16 keeps f32's exponent range with 8 fewer mantissa bits; f16 keeps 10 mantissa bits
    // and overflows at 65504. Neither fact is visible from here, which is the finding.
    let h = cost(&saxpy("f16", "0.3333"));
    let b = cost(&saxpy("bf16", "0.3333"));
    assert_eq!(h.bytes_per_element(), b.bytes_per_element());
    assert_eq!(h.intensity, b.intensity);
    assert_eq!(h.sector_read_per_element, b.sector_read_per_element);
}

#[test]
fn a_narrow_element_does_not_narrow_the_bus() {
    // ADR-0015's payload-versus-bus distinction, sharpened. A sector is 32 bytes whatever sits
    // in it, so a strided access costs a whole one at any width -- and a strided f16 kernel
    // therefore wastes **twice** the fraction of what it fetches that an f32 one does.
    //
    // Measured here as: the payload halves and the sector cost does not.
    let src = |elem: &str| {
        format!(
            "machine sm_120\n\n\
             kernel t(rows: u32, cols: u32, a: [{elem}; rows, cols], b: [{elem}; cols, rows])\n    \
             space i, j : rows, cols\n    \
             stream a : dram -> reg\n    \
             stream b : dram -> reg, drain\n    \
             at reg:\n        \
             b[j, i] = a[i, j]\n"
        )
    };
    let wide = cost(&src("f32"));
    let narrow = cost(&src("f16"));

    assert_eq!(wide.bytes_per_element(), Some(8.0));
    assert_eq!(narrow.bytes_per_element(), Some(4.0), "the payload halves");

    // The two halves must be checked apart, not summed. Summing gives 36 against 34 and reads
    // like the bus narrowed; it did not. The *coalesced* read narrowed, because there the
    // payload is the bus; the *strided* write did not, because there a whole sector is fetched
    // for one element whatever that element is.
    assert_eq!(wide.sector_read_per_element, 4.0);
    assert_eq!(narrow.sector_read_per_element, 2.0, "coalesced: payload is the bus");
    assert_eq!(wide.sector_write_per_element, 32.0);
    assert_eq!(
        narrow.sector_write_per_element, 32.0,
        "strided: a 32-byte sector does not care how wide the element in it is"
    );
    assert!(
        narrow.coalescence() < wide.coalescence(),
        "so the narrow kernel wastes a larger fraction of the bus: {} vs {}",
        narrow.coalescence(),
        wide.coalescence()
    );
}

#[test]
fn one_kernel_may_mix_widths() {
    // Nothing forces a kernel to be all one width, and the cost model must add them per stream
    // rather than multiply an element count by a single number. Reading half and writing
    // single is the shape every mixed-precision pipeline has.
    let src = "machine sm_120\n\n\
               kernel up(n: u32, x: [f16; n], y: [f32; n])\n    \
               intensity 0.0\n    \
               stream x : dram -> reg\n    \
               stream y : dram -> reg, drain\n    \
               at reg:\n        \
               y = x + y\n";
    // 2 read + 4 read + 4 written = 10, which is neither width's own answer.
    assert_eq!(cost(src).bytes_per_element(), Some(10.0));

    // And it is the **body** that decides whether a drained buffer is also read, not the
    // `drain` keyword. `y = x` never reads `y`, so that kernel moves 6. The first version of
    // this test assumed `drain` implied a read and asserted 10 for it.
    let write_only = src.replace("y = x + y", "y = x");
    assert_eq!(cost(&write_only).bytes_per_element(), Some(6.0));
}

#[test]
fn a_buffer_type_that_does_not_exist_is_refused_by_name() {
    let e = parse(
        "machine sm_120\n\nkernel k(n: u32, x: [f8; n])\n    \
         stream x : dram -> reg\n    at reg:\n        x = x\n",
    )
    .expect_err("f8 is not a type here");
    let msg = e.to_string();
    assert!(msg.contains("[f32], [f16] and [bf16]"), "{msg}");
}

#[test]
fn the_shared_level_narrows_with_the_element_too() {
    // A staged stream crosses shared memory, and shared memory holds whatever width was
    // loaded. ADR-0022 made the shared level a ceiling candidate priced in *accesses*; the
    // bytes still halve, and the two facts are why ADR-0024 predicts a 2x for a streaming
    // kernel and ~1x for the matmul.
    let src = |elem: &str| {
        format!(
            "machine sm_120\n\n\
             kernel t(rows: u32, cols: u32, a: [{elem}; rows, cols], b: [{elem}; cols, rows])\n    \
             space i, j : rows, cols\n    \
             tile 32, 32\n    \
             stream a : dram -> smem -> reg\n    \
             stream b : dram -> reg, drain\n    \
             at reg:\n        \
             b[j, i] = a[i, j]\n"
        )
    };
    let wide = cost(&src("f32")).at(Level::Smem).expect("staged").total();
    let narrow = cost(&src("f16")).at(Level::Smem).expect("staged").total();
    assert_eq!(wide, 8.0);
    assert_eq!(narrow, 4.0);
}
