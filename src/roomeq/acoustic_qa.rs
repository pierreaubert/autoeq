//! Analytic acoustic ground truth and acceptance metrics for RoomEQ QA.
//!
//! The fixtures in this module carry their generating parameters, expected
//! complex transfer function, valid correction region, and prohibited
//! behaviours. Candidate DSP is evaluated from its complex transfer function,
//! rather than inferred from plugin presence.
//!
//! CI entry points are intentionally ordinary Rust test filters so they do not
//! depend on a particular task runner:
//!
//! - PR: `cargo test -p autoeq acoustic_qa_pr_ --lib`
//! - Nightly: `cargo test -p autoeq acoustic_qa_nightly_ --lib -- --ignored`

mod fixtures;
mod metrics;
mod scenario;
mod types;

pub use fixtures::*;
pub use metrics::*;
pub use scenario::*;
pub use types::*;

#[cfg(test)]
mod tests;
