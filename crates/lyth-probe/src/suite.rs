//! Suite runner — one command that refuses to forget a gate.

use serde::Deserialize;
use thiserror::Error;

pub const SUITE_SCHEMA: &str = "lyth-suite/0.1";

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

/// What a step actually did.
///
/// This replaces a `(bool, &str)` pair that had to agree by convention and that nothing
/// checked — `got_pass && got_label == "pass"` was the old condition, which says out loud
/// that the two could disagree.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Outcome {
    Pass,
    Fail,
    /// Oracle only: the case did not carry enough silicon to decide.
    Inconclusive,
    /// The step could not run at all — unreadable file, wrong schema, check refused input.
    Error,
}

impl Outcome {
    pub fn label(self) -> &'static str {
        match self {
            Outcome::Pass => "pass",
            Outcome::Fail => "fail",
            Outcome::Inconclusive => "inconclusive",
            Outcome::Error => "error",
        }
    }

    /// A two-valued verdict, where anything that is not a pass is a fail.
    pub fn of(passed: bool) -> Self {
        if passed {
            Outcome::Pass
        } else {
            Outcome::Fail
        }
    }
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

pub fn match_expect(expect: Expect, got: Outcome) -> StepOutcome {
    let ok = matches!(
        (expect, got),
        (Expect::Pass, Outcome::Pass)
            | (Expect::Fail, Outcome::Fail)
            | (Expect::Inconclusive, Outcome::Inconclusive)
    );
    if ok {
        StepOutcome::Ok
    } else {
        StepOutcome::Unexpected {
            got: got.label().into(),
        }
    }
}

impl Step {
    /// Every variant carries an `expect`; reading it should not require matching all seven.
    pub fn expect(&self) -> Expect {
        match self {
            Step::Validate { expect, .. }
            | Step::OracleCheck { expect, .. }
            | Step::MachineCheck { expect, .. }
            | Step::IntensityCheck { expect, .. }
            | Step::KernelCheck { expect, .. }
            | Step::CtCheck { expect, .. }
            | Step::PolyCheck { expect, .. } => *expect,
        }
    }
}

impl crate::document::Document for Suite {
    const SCHEMA: &'static str = SUITE_SCHEMA;
}
