//! The one contract every checkable document carries: it names its own schema.
//!
//! Each command used to spell its schema into its own error message — twelve string
//! literals that no compiler was checking against the twelve constants beside them. A
//! renamed schema left the code correct and the diagnostics lying, which is the failure
//! mode this project exists to refuse.
//!
//! Binding the name to the type means the loader prints what the parser required.

use serde::de::DeserializeOwned;

pub trait Document: DeserializeOwned {
    /// The value this document's `schema` field must carry.
    const SCHEMA: &'static str;
}
