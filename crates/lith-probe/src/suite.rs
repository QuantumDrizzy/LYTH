//! Suite runner — one command that refuses to forget a gate.

use serde::Deserialize;
use thiserror::Error;

pub const SUITE_SCHEMA: &str = "lith-suite/0.1";

#[derive(Debug, Clone, Deserialize)]
pub struct Suite {
    pub schema: String,
    pub name: String,
    pub steps: Vec<Step>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "kind", rename_all = "kebab-case")]
pub enum Step {
    Validate {
        path: String,
        #[serde(default = "expect_pass")]
        expect: Expect,
    },
    OracleCheck {
        path: String,
        #[serde(default = "expect_pass")]
        expect: Expect,
    },
    MachineCheck {
        machine: String,
        measurement: String,
        #[serde(default = "expect_pass")]
        expect: Expect,
        #[serde(default = "default_tol")]
        tol: f64,
    },
    IntensityCheck {
        path: String,
        #[serde(default)]
        machine: Option<String>,
        #[serde(default = "expect_pass")]
        expect: Expect,
        #[serde(default = "default_tol")]
        tol: f64,
    },
    KernelCheck {
        path: String,
        machine: String,
        #[serde(default = "expect_pass")]
        expect: Expect,
        #[serde(default = "default_tol")]
        tol: f64,
    },
    CtCheck {
        path: String,
        #[serde(default = "expect_pass")]
        expect: Expect,
    },
    PolyCheck {
        path: String,
        machine: String,
        #[serde(default = "expect_pass")]
        expect: Expect,
        #[serde(default = "default_tol")]
        tol: f64,
    },
}

fn expect_pass() -> Expect {
    Expect::Pass
}

fn default_tol() -> f64 {
    0.05
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Expect {
    Pass,
    Fail,
    /// Oracle only: incomplete silicon.
    Inconclusive,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StepOutcome {
    Ok,
    Unexpected { got: String },
}

#[derive(Debug, Error)]
pub enum SuiteError {
    #[error("schema must be `{SUITE_SCHEMA}`, got `{0}`")]
    BadSchema(String),
    #[error("{0}")]
    Message(String),
}

pub fn check_schema(suite: &Suite) -> Result<(), SuiteError> {
    if suite.schema != SUITE_SCHEMA {
        return Err(SuiteError::BadSchema(suite.schema.clone()));
    }
    if suite.steps.is_empty() {
        return Err(SuiteError::Message("suite has no steps".into()));
    }
    Ok(())
}

pub fn match_expect(expect: Expect, got_pass: bool, got_label: &str) -> StepOutcome {
    let ok = match expect {
        Expect::Pass => got_pass && got_label == "pass",
        Expect::Fail => got_label == "fail",
        Expect::Inconclusive => got_label == "inconclusive",
    };
    if ok {
        StepOutcome::Ok
    } else {
        StepOutcome::Unexpected {
            got: got_label.into(),
        }
    }
}
