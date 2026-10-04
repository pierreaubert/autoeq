//! AutoEQ command-line adapters.

extern crate autoeq_workflow as autoeq;

pub mod autoeq_command;
pub mod benchmark;
pub mod download;
pub mod program_version;

pub use autoeq_workflow::cli::*;
pub use autoeq_workflow::qa_println;
