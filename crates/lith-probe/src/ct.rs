//! Constant-time taint check — Fase 4 teeth without a parser.
//!
//! Secret-dependent **addresses** and **branches** are refuse. Arithmetic on
//! secrets is allowed (taint propagates). This is the feature ADR-0001 said
//! nobody ships.

use serde::Deserialize;
use std::collections::HashMap;
use thiserror::Error;

pub const CT_SCHEMA: &str = "lith-ct/0.1";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Taint {
    Public,
    Secret,
}

#[derive(Debug, Clone, Deserialize)]
pub struct CtCase {
    pub schema: String,
    pub name: String,
    /// Values that start secret-tainted.
    pub secrets: Vec<String>,
    #[serde(default)]
    pub values: Vec<ValueDecl>,
    pub ops: Vec<CtOp>,
    #[serde(default)]
    pub notes: Vec<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ValueDecl {
    pub id: String,
    #[serde(default = "default_public")]
    pub taint: Taint,
}

fn default_public() -> Taint {
    Taint::Public
}

#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum CtOp {
    /// `dst = f(args...)` — taint = join of args.
    Assign {
        dst: String,
        args: Vec<String>,
        #[serde(default)]
        note: String,
    },
    /// Memory load: address must be public.
    Load {
        addr: String,
        into: String,
        /// Taint of the loaded payload (memory may hold secrets at a public index).
        #[serde(default = "default_public")]
        payload: Taint,
        #[serde(default)]
        note: String,
    },
    /// Memory store: address must be public. Stored value may be secret.
    Store {
        addr: String,
        from: String,
        #[serde(default)]
        note: String,
    },
    /// Control flow: condition must be public.
    Branch {
        cond: String,
        #[serde(default)]
        note: String,
    },
    /// Explicit declassify (rare; must be justified in note).
    Declassify {
        dst: String,
        from: String,
        note: String,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub enum CtVerdict {
    Pass { secrets: Vec<String>, steps: usize },
    Fail { violations: Vec<String> },
}

#[derive(Debug, Error)]
pub enum CtError {
    #[error("schema must be `{CT_SCHEMA}`, got `{0}`")]
    BadSchema(String),
    #[error("{0}")]
    Message(String),
}

pub fn check(case: &CtCase) -> Result<CtVerdict, CtError> {
    if case.schema != CT_SCHEMA {
        return Err(CtError::BadSchema(case.schema.clone()));
    }
    if case.ops.is_empty() {
        return Err(CtError::Message("ct case has no ops".into()));
    }

    let mut taint: HashMap<String, Taint> = HashMap::new();
    for v in &case.values {
        taint.insert(v.id.clone(), v.taint);
    }
    for s in &case.secrets {
        taint.insert(s.clone(), Taint::Secret);
    }

    let mut violations = Vec::new();

    for (i, op) in case.ops.iter().enumerate() {
        let step = i + 1;
        match op {
            CtOp::Assign { dst, args, .. } => {
                let mut acc = Taint::Public;
                for a in args {
                    acc = join(acc, lookup(&taint, a, &mut violations, step, "assign arg"));
                }
                taint.insert(dst.clone(), acc);
            }
            CtOp::Load {
                addr,
                into,
                payload,
                ..
            } => {
                let at = lookup(&taint, addr, &mut violations, step, "load addr");
                if at == Taint::Secret {
                    violations.push(format!(
                        "step {step}: load address `{addr}` is secret-tainted\n  \
                         options: (1) index from a public counter / gather table\n  \
                                  (2) bit-sliced / oblivious access pattern\n  \
                                  (3) declassify only with a documented note (last resort)"
                    ));
                }
                // Payload taint is independent of address taint when the index is public
                // but the cell holds a secret (NTT coefficients).
                let loaded = if at == Taint::Secret {
                    Taint::Secret
                } else {
                    *payload
                };
                taint.insert(into.clone(), loaded);
            }
            CtOp::Store { addr, from, .. } => {
                let at = lookup(&taint, addr, &mut violations, step, "store addr");
                let _ = lookup(&taint, from, &mut violations, step, "store data");
                if at == Taint::Secret {
                    violations.push(format!(
                        "step {step}: store address `{addr}` is secret-tainted\n  \
                         options: (1) write to a public layout slot\n  \
                                  (2) oblivious scatter\n  \
                                  (3) declassify address with a documented note"
                    ));
                }
            }
            CtOp::Branch { cond, .. } => {
                let ct = lookup(&taint, cond, &mut violations, step, "branch cond");
                if ct == Taint::Secret {
                    violations.push(format!(
                        "step {step}: branch on secret-tainted `{cond}`\n  \
                         options: (1) select/cmov / predicated arithmetic\n  \
                                  (2) bit-sliced boolean without control flow\n  \
                                  (3) redesign so the predicate is public"
                    ));
                }
            }
            CtOp::Declassify { dst, from, note } => {
                if note.trim().is_empty() {
                    violations.push(format!(
                        "step {step}: declassify `{from}` → `{dst}` requires a non-empty note"
                    ));
                }
                let _ = lookup(&taint, from, &mut violations, step, "declassify from");
                taint.insert(dst.clone(), Taint::Public);
            }
        }
    }

    // Unreachable ops already walked; also flag unused secrets? skip.

    if violations.is_empty() {
        Ok(CtVerdict::Pass {
            secrets: case.secrets.clone(),
            steps: case.ops.len(),
        })
    } else {
        Ok(CtVerdict::Fail { violations })
    }
}

fn lookup(
    taint: &HashMap<String, Taint>,
    id: &str,
    violations: &mut Vec<String>,
    step: usize,
    ctx: &str,
) -> Taint {
    match taint.get(id) {
        Some(t) => *t,
        None => {
            // Undeclared → public with a soft note? Better: treat as public but warn once.
            // Strict: violation for unknown ids so the IR cannot smuggle secrets.
            violations.push(format!(
                "step {step}: {ctx} `{id}` is undeclared — add it to values[] or secrets[]"
            ));
            Taint::Public
        }
    }
}

fn join(a: Taint, b: Taint) -> Taint {
    match (a, b) {
        (Taint::Secret, _) | (_, Taint::Secret) => Taint::Secret,
        _ => Taint::Public,
    }
}

pub fn format_verdict(case: &CtCase, v: &CtVerdict) -> String {
    let mut out = format!("ct-check: {}\n", case.name);
    match v {
        CtVerdict::Pass { secrets, steps } => {
            out.push_str("verdict: PASS — no secret-dependent addr/branch\n");
            out.push_str(&format!(
                "  secrets: [{}]  ops: {steps}\n",
                secrets.join(", ")
            ));
        }
        CtVerdict::Fail { violations } => {
            out.push_str("verdict: FAIL — constant-time refuse\n");
            for (i, msg) in violations.iter().enumerate() {
                out.push_str(&format!("\n[{}] {msg}\n", i + 1));
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn secret_branch_fails() {
        let case: CtCase = serde_json::from_str(
            r#"{
              "schema":"lith-ct/0.1","name":"bad",
              "secrets":["sk"],
              "values":[{"id":"sk","taint":"secret"}],
              "ops":[{"kind":"branch","cond":"sk"}]
            }"#,
        )
        .unwrap();
        match check(&case).unwrap() {
            CtVerdict::Fail { violations } => assert!(!violations.is_empty()),
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn secret_arith_ok() {
        let case: CtCase = serde_json::from_str(
            r#"{
              "schema":"lith-ct/0.1","name":"ok",
              "secrets":["sk"],
              "values":[
                {"id":"sk","taint":"secret"},
                {"id":"a","taint":"public"},
                {"id":"idx","taint":"public"}
              ],
              "ops":[
                {"kind":"assign","dst":"t","args":["sk","a"]},
                {"kind":"store","addr":"idx","from":"t"}
              ]
            }"#,
        )
        .unwrap();
        match check(&case).unwrap() {
            CtVerdict::Pass { .. } => {}
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn secret_gather_fails() {
        let case: CtCase = serde_json::from_str(
            r#"{
              "schema":"lith-ct/0.1","name":"gather",
              "secrets":["sk"],
              "values":[{"id":"sk"},{"id":"buf"}],
              "ops":[
                {"kind":"assign","dst":"addr","args":["sk"]},
                {"kind":"load","addr":"addr","into":"x"}
              ]
            }"#,
        )
        .unwrap();
        match check(&case).unwrap() {
            CtVerdict::Fail { .. } => {}
            other => panic!("{other:?}"),
        }
    }
}
