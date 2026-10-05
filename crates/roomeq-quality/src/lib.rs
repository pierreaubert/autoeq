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
mod acceptance_bundle;
mod band_policy;
mod battery;
mod capture;
mod chain_constraints;
mod corpus;
mod final_check;
mod fixtures;
mod inversion_support;
mod joint_sub_scorecard;
mod listening;
mod metrics;
mod promotion;
mod protocol;
mod quality;
mod scenario;
mod seeded;
mod stimuli;
mod trial_import;
mod types;
mod validation_corpus;

pub use acceptance::*;
pub use acceptance_bundle::*;
pub mod electrical_headroom;
pub mod physical_drive;
pub use band_policy::*;
pub use battery::*;
pub use capture::*;
pub use chain_constraints::*;
pub use corpus::*;
pub use final_check::*;
pub use fixtures::*;
pub use inversion_support::*;
pub use joint_sub_scorecard::*;
pub use listening::*;
pub use metrics::*;
pub use promotion::*;
pub use protocol::*;
pub use quality::*;
pub use scenario::*;
pub use seeded::*;
pub use stimuli::*;
pub use trial_import::*;
pub use types::*;
pub use validation_corpus::*;

#[cfg(test)]
mod psycho_fast;
#[cfg(test)]
mod tests;
