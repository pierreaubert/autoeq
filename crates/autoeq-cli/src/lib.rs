//! AutoEQ command-line adapters.

extern crate autoeq_workflow as autoeq;

pub mod autoeq_command;
pub mod benchmark;
pub mod download;

pub use autoeq_workflow::cli::*;

/// Conditional debug output used by the historical AutoEQ CLI.
#[macro_export]
macro_rules! qa_println {
    ($fmt:literal) => {
        log::debug!($fmt);
    };
    ($fmt:literal, $($arg:expr),* $(,)?) => {
        log::debug!($fmt, $($arg),*);
    };
    ($args:expr, $fmt:literal) => {
        if $args.qa.is_none() {
            log::debug!($fmt);
        }
    };
    ($args:expr, $fmt:literal, $($arg:expr),* $(,)?) => {
        if $args.qa.is_none() {
            log::debug!($fmt, $($arg),*);
        }
    };
}
