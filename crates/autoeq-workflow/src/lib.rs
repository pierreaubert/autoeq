//! High-level speaker and headphone equalization workflows.

#![allow(unsafe_code)]

pub use autoeq_core::{AutoeqError, Curve, Result};
pub use autoeq_core::iir;
pub mod error { pub use autoeq_core::error::*; }
pub mod read { pub use autoeq_measurements::read::*; }
pub use autoeq_measurements::read::Cea2034Data;
pub mod loss { pub use autoeq_optim::loss::*; }
pub mod optim { pub use autoeq_optim::optim::*; }
pub mod cli { pub use autoeq_optim::cli::*; }
pub use autoeq_optim::{OptimParams, PeqModel, de};
pub use autoeq_optim::LossType;
pub mod x2peq { pub use autoeq_core::x2peq::*; }
pub use autoeq_core::x2peq::x2peq;

pub mod workflow;
pub use workflow::*;
