//! Analytic acoustic ground truth and acceptance metrics for RoomEQ QA.
//!
//! This is the standalone RoomEQ acoustic-quality boundary.
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

mod acceptance;
mod band_policy;
mod chain_constraints;
mod corpus;
mod final_check;
mod fixtures;
mod inversion_support;
mod metrics;
mod protocol;
mod quality;
mod scenario;
mod seeded;
mod stimuli;
mod types;
mod validation_corpus;

pub use acceptance::*;
pub mod electrical_headroom;
pub use band_policy::*;
pub use chain_constraints::*;
pub use corpus::*;
pub use final_check::*;
pub use fixtures::*;
pub use inversion_support::*;
pub use metrics::*;
pub use protocol::*;
pub use quality::*;
pub use scenario::*;
pub use seeded::*;
pub use stimuli::*;
pub use types::*;
pub use validation_corpus::*;

#[cfg(test)]
mod tests;
