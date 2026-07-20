//! High-level speaker and headphone equalization workflows.

/// Conditional debug output used by historical AutoEQ CLI callers.
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

pub use autoeq_core::iir;
pub use autoeq_core::{AutoeqError, Curve, Result};
pub mod error {
    pub use autoeq_core::error::*;
}
pub mod read {
    pub use autoeq_measurements::read::*;
}
pub use autoeq_measurements::read::Cea2034Data;
pub mod loss {
    pub use autoeq_optim::loss::*;
}
pub mod optim {
    pub use autoeq_optim::optim::*;
}
pub mod cli {
    pub use autoeq_optim::cli::*;
}
pub mod cea2034 {
    pub use autoeq_measurements::cea2034::*;
}
pub use autoeq_optim::LossType;
pub use autoeq_optim::{OptimParams, PeqModel, de};
pub mod x2peq {
    pub use autoeq_core::x2peq::*;
}
pub use autoeq_core::x2peq::x2peq;

pub mod workflow;
pub use workflow::*;
