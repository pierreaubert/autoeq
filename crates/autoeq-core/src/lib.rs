//! Pure AutoEQ domain and DSP primitives.
//!
//! This crate intentionally has no filesystem, network, CLI, plotting, or
//! optimizer dependencies. Higher-level crates build measurement and
//! optimization workflows on these types.

pub use math_audio_iir_fir as iir;

pub mod alignment;
pub mod auditory_frequency;
pub mod capture_arrival;
pub mod capture_handoff;
pub mod capture_provenance;
pub mod constraint_envelope;
pub mod curve;
pub mod curve_transforms;
pub mod direct_sound;
pub mod dsp_conventions;
pub mod error;
pub mod evidence;
pub mod fir_response;
pub mod measurement_contracts;
pub mod measurement_quality;
pub mod param_utils;
pub mod peq_model;
pub mod phase_utils;
pub mod resolution;
pub mod response;
pub mod serialized_dsp;
pub mod x2peq;

#[cfg(test)]
mod curve_interpolation_tests;
#[cfg(test)]
mod curve_normalization_tests;
#[cfg(test)]
mod curve_smoothing_tests;
#[cfg(test)]
mod measurement_quality_tests;
#[cfg(test)]
mod psycho_fast;

pub use alignment::{
    align_evidence_support, timing_uncertainty_to_phase_deg, validate_alignment_grid,
};
pub use auditory_frequency::{
    AUDITORY_FREQUENCY_MEASURE_VERSION, erb_rate, erb_rate_cell_widths, erb_rate_weighted_rms,
    try_erb_rate_cell_widths,
};
pub use constraint_envelope::{LocalQEnvelope, LocalQKnot, effective_max_q};
pub use curve::Curve;
pub use curve_transforms::{
    NORMALIZE_HIGH_FREQ, NORMALIZE_LOW_FREQ, PsychoacousticSmoothingConfig,
    ResolutionMagnitudeBasis, build_target_curve_by_name, calculate_variable_n,
    create_log_frequency_grid, curve_slope_per_octave_in_range, interpolate, interpolate_log_space,
    interpolate_log_space_hold_edges, interpolate_response, interpolate_response_hold_edges,
    mean_over_log_frequency, normalize_and_interpolate_response,
    normalize_and_interpolate_response_with_range, normalize_response,
    regression_slope_per_octave_in_range, smooth_correction_to_resolution, smooth_gaussian,
    smooth_one_over_n_octave, smooth_psychoacoustic,
};
pub use error::{AutoeqError, Result};
pub use evidence::{
    CalibrationStatus, CaptureKind, CommonReferenceScope, EVIDENCE_ENVELOPE_VERSION, EvidenceBand,
    EvidenceEnvelope, Uncertainty, UncertaintyKind,
};
pub use measurement_contracts::{
    ESTIMATED_POSITION_MAX_RMS_M, ESTIMATED_POSITION_MAX_SIGMA_M, EstimatedPosition,
    EstimatedRangeCell, InlineMeasurement, MeasurementMultiple, MeasurementProvenance,
    MeasurementRef, MeasurementSingle, MeasurementSource, ProvenanceCaptureKind, SpinoramaBundle,
};
pub use measurement_quality::{
    MeasurementQuality, MeasurementQualityReport, assess_measurement_quality,
    assess_multiple_measurement_quality,
};
pub use param_utils::{FilterParams, ParamLayout, PeqLayout};
pub use peq_model::PeqModel;

pub mod serialized_plugins;

pub mod serialized_chain;

pub mod crossover;

pub mod runtime_limiter;

/// Released product version printed by the command-line executables.
///
/// This preserves the former root-package version independently of library
/// versions serialized in optimizer, model, and report contracts.
pub const PRODUCT_VERSION: &str = "0.5.78";
