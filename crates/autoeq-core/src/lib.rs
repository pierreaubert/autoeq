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
pub mod curve_transforms;
pub mod error;
pub mod measurement_contracts;
pub mod measurement_quality;
pub mod param_utils;
pub mod peq_model;
pub mod phase_utils;
pub mod response;
pub mod x2peq;

#[cfg(test)]
mod curve_interpolation_tests;
#[cfg(test)]
mod curve_normalization_tests;
#[cfg(test)]
mod curve_smoothing_tests;
#[cfg(test)]
mod measurement_quality_tests;

pub use curve::Curve;
pub use curve_transforms::*;
pub use error::{AutoeqError, Result};
pub use measurement_contracts::{
    InlineMeasurement, MeasurementMultiple, MeasurementRef, MeasurementSingle, MeasurementSource,
    SpinoramaBundle,
};
pub use measurement_quality::{
    MeasurementQuality, MeasurementQualityReport, assess_measurement_quality,
    assess_multiple_measurement_quality,
};
pub use param_utils::{FilterParams, ParamLayout, PeqLayout};
pub use peq_model::PeqModel;
