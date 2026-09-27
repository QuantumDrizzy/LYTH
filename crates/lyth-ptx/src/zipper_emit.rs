//! PTX for one `zipper2` step. Thread 0 does the whole register. The arithmetic
//! is scalar `f32` in the same order as `lyth_lang::zipper2::zipper2_words`.

use crate::{isa_for, EmitError, Module};
use lyth_lang::ast::Ty;
use lyth_lang::ir::KernelIr;

struct Gen {
    f: u32,
    s: u32,
    p: u32,
    body: String,
}

impl Gen {
    fn new() -> Self {
        Self {
            f: 0,
            s: 0,
            p: 0,
            body: String::new(),
        }
    }

    fn f(&mut self) -> String {
        let name = format!("f{}", self.f);
        self.f += 1;
        name
    }

    fn sreg(&mut self) -> String {
        let name = format!("s{}", self.s);
        self.s += 1;
        name
    }

    fn pred(&mut self) -> String {
        let name = format!("p{}", self.p);
        self.p += 1;
        name
    }

    fn line(&mut self, text: &str) {
        self.body.push_str("    ");
        self.body.push_str(text);
        self.body.push('\n');
    }

    fn i8_as_f32(&mut self, word: &str, shift: u32) -> String {
        let bits = self.sreg();
        let shifted = self.sreg();
        let masked = self.sreg();
        let pred = self.pred();
        let wide = self.f();
        let out = self.f();
        self.line(&format!("mov.b32 {bits}, {word};"));
        self.line(&format!("shr.b32 {shifted}, {bits}, {shift};"));
        self.line(&format!("and.b32 {masked}, {shifted}, 255;"));
        self.line(&format!("setp.hs.u32 {pred}, {masked}, 128;"));
        self.line(&format!("cvt.rn.f32.u32 {wide}, {masked};"));
        self.line(&format!("mov.f32 {out}, {wide};"));
        self.line(&format!("@{pred} sub.rn.f32 {out}, {wide}, 0f43800000;"));
        out
    }

    fn mul(&mut self, a: &str, b: &str) -> String {
        let r = self.f();
        self.line(&format!("mul.rn.f32 {r}, {a}, {b};"));
        r
    }

    fn add(&mut self, a: &str, b: &str) -> String {
        let r = self.f();
        self.line(&format!("add.rn.f32 {r}, {a}, {b};"));
        r
    }

    fn sub(&mut self, a: &str, b: &str) -> String {
        let r = self.f();
        self.line(&format!("sub.rn.f32 {r}, {a}, {b};"));
        r
    }
}

pub fn emit(ir: &KernelIr, arch: &str) -> Result<Module, EmitError> {
    let version = isa_for(arch)?;
    let mut params = String::new();
    let mut param_list = Vec::new();
    for p in &ir.params {
        match p.ty {
            Ty::U32 => {
                params.push_str(&format!("    .param .u32 {},\n", p.name));
                param_list.push((p.name.clone(), p.ty));
            }
            Ty::BufF32 => {
                params.push_str(&format!("    .param .u64 {},\n", p.name));
                param_list.push((p.name.clone(), p.ty));
            }
            _ => {
                return Err(EmitError::Message(format!(
                    "zipper2 cannot take a {} parameter",
                    p.name
                )))
            }
        }
    }
    if params.ends_with(",\n") {
        params.truncate(params.len() - 2);
        params.push('\n');
    }
    let mut g = Gen::new();
    g.line("mov.u32 tid, %tid.x;");
    g.line("setp.ne.u32 pstop, tid, 0;");
    g.line("@pstop ret;");
    for (name, letter) in [("acc", "a"), ("ket", "k"), ("bra", "b")] {
        g.line(&format!("ld.param.u64 p{letter}, [{name}];"));
        for i in 0..8 {
            g.line(&format!(
                "ld.global.f32 {letter}{i}, [p{letter}+{}];",
                i * 4
            ));
        }
    }

    fn complex_at(
        g: &mut Gen,
        letter: &str,
        codes: &[String],
        left: usize,
        phys: usize,
        right: usize,
    ) -> (String, String) {
        let idx = 2 * ((2 * left + phys) * 2 + right);
        let scale = format!("{letter}{}", 4 + right);
        let re = g.mul(&codes[idx], &scale);
        let im = g.mul(&codes[idx + 1], &scale);
        (re, im)
    }

    // ket and bra codes: bytes 0..15 of each register. Words 4 and 5 stay the scales.
    let mut ket_codes = Vec::new();
    for idx in 0..16 {
        ket_codes.push(g.i8_as_f32(&format!("k{}", idx / 4), ((idx % 4) * 8) as u32));
    }
    let mut bra_codes = Vec::new();
    for idx in 0..16 {
        bra_codes.push(g.i8_as_f32(&format!("b{}", idx / 4), ((idx % 4) * 8) as u32));
    }

    let mut partial_re = vec![vec![vec![String::new(); 2]; 2]; 2];
    let mut partial_im = vec![vec![vec![String::new(); 2]; 2]; 2];
    for al in 0..2 {
        for phys in 0..2 {
            for br in 0..2 {
                let mut re = g.f();
                g.line(&format!("mov.f32 {re}, 0f00000000;"));
                let mut im = g.f();
                g.line(&format!("mov.f32 {im}, 0f00000000;"));
                for bl in 0..2 {
                    let e_re = format!("a{}", 2 * (2 * al + bl));
                    let e_im = format!("a{}", 2 * (2 * al + bl) + 1);
                    let (b_re, b_im) = complex_at(&mut g, "k", &ket_codes, bl, phys, br);
                    let t1 = g.mul(&e_re, &b_re);
                    let t2 = g.mul(&e_im, &b_im);
                    let t3 = g.sub(&t1, &t2);
                    re = g.add(&re, &t3);
                    let u1 = g.mul(&e_re, &b_im);
                    let u2 = g.mul(&e_im, &b_re);
                    let u3 = g.add(&u1, &u2);
                    im = g.add(&im, &u3);
                }
                partial_re[al][phys][br] = re;
                partial_im[al][phys][br] = im;
            }
        }
    }

    let mut out_words = vec![String::new(); 8];
    for ar in 0..2 {
        for br in 0..2 {
            let mut re = g.f();
            g.line(&format!("mov.f32 {re}, 0f00000000;"));
            let mut im = g.f();
            g.line(&format!("mov.f32 {im}, 0f00000000;"));
            for al in 0..2 {
                for phys in 0..2 {
                    let (a_re, a_im) = complex_at(&mut g, "b", &bra_codes, al, phys, ar);
                    let t_re = &partial_re[al][phys][br];
                    let t_im = &partial_im[al][phys][br];
                    let t1 = g.mul(&a_re, t_re);
                    let t2 = g.mul(&a_im, t_im);
                    let t3 = g.add(&t1, &t2);
                    re = g.add(&re, &t3);
                    let u1 = g.mul(&a_re, t_im);
                    let u2 = g.mul(&a_im, t_re);
                    let u3 = g.sub(&u1, &u2);
                    im = g.add(&im, &u3);
                }
            }
            let w = 2 * (2 * ar + br);
            out_words[w] = re;
            out_words[w + 1] = im;
        }
    }

    g.line("ld.param.u64 po, [out];");
    for (i, word) in out_words.iter().enumerate() {
        g.line(&format!("st.global.f32 [po+{}], {word};", i * 4));
    }
    g.line("ret;");

    let name = &ir.name;
    let ptx = format!(
        ".version {version}\n.target {arch}\n.address_size 64\n.visible .entry {name}(\n{params})\n{{\n    .reg .pred pstop;\n    .reg .pred p<{preds}>;\n    .reg .u32 tid;\n    .reg .u64 pa, pk, pb, po;\n    .reg .f32 a<8>, k<8>, b<8>;\n    .reg .f32 f<{fs}>;\n    .reg .b32 s<{ss}>;\n{body}}}\n",
        preds = g.p.max(1),
        fs = g.f.max(1),
        ss = g.s.max(1),
        body = g.body,
    );
    Ok(Module {
        ptx,
        entry: ir.name.clone(),
        params: ir.params.iter().map(|p| (p.name.clone(), p.ty)).collect(),
        bound: "n".into(),
        arch: arch.to_string(),
    })
}
