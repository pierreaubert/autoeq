//! Pure AutoEQ domain and DSP primitives.
//!
//! This crate intentionally has no filesystem, network, CLI, plotting, or
//! optimizer dependencies. Higher-level crates build measurement and
//! optimization workflows on these types.

// The numerical dependencies use internally audited unsafe implementations
// behind safe APIs (for example ndarray slicing macros).
#![allow(unsafe_code)]

pub use math_audio_iir_fir as iir;

pub mod curve;
pub mod error;
pub mod param_utils;
pub mod peq_model;
pub mod phase_utils;
pub mod response;
pub mod x2peq;

pub use curve::Curve;
pub use error::{AutoeqError, Result};
pub use param_utils::{FilterParams, ParamLayout, PeqLayout};
pub use peq_model::PeqModel;
