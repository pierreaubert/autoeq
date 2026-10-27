//! Shared data types for loss functions.
//!
//! Defines the [`LossType`] enum used by the optimizer dispatcher and
//! the measurement data containers that individual loss functions
//! consume.

use crate::Curve;
use crate::error::{AutoeqError, Result};
use ndarray::Array1;
use std::collections::HashMap;

pub use autoeq_optim_types::LossType;

/// Data required for computing speaker score-based loss
#[derive(Debug, Clone)]
pub struct SpeakerLossData {
    /// On-axis SPL measurements
    pub on: Array1<f64>,
    /// Listening window SPL measurements
    pub lw: Array1<f64>,
    /// Sound power SPL measurements
    pub sp: Array1<f64>,
    /// Predicted in-room SPL measurements
    pub pir: Array1<f64>,
}

impl SpeakerLossData {
    /// Create a new SpeakerLossData instance.
    ///
    /// # Arguments
    /// * `spin` - Map of CEA2034 curves by name ("On Axis", "Listening Window", "Sound Power", "Estimated In-Room Response")
    ///
    /// # Errors
    ///
    /// Returns `AutoeqError::MissingCea2034Curve` if any required curve is missing.
    /// Returns `AutoeqError::CurveLengthMismatch` if curves have different lengths.
    pub fn try_new(spin: &HashMap<String, Curve>) -> Result<Self> {
        let on = spin
            .get("On Axis")
            .ok_or_else(|| AutoeqError::MissingCea2034Curve {
                curve_name: "On Axis".to_string(),
            })?
            .spl
            .clone();
        let lw = spin
            .get("Listening Window")
            .ok_or_else(|| AutoeqError::MissingCea2034Curve {
                curve_name: "Listening Window".to_string(),
            })?
            .spl
            .clone();
        let sp = spin
            .get("Sound Power")
            .ok_or_else(|| AutoeqError::MissingCea2034Curve {
                curve_name: "Sound Power".to_string(),
            })?
            .spl
            .clone();
        let pir = spin
            .get("Estimated In-Room Response")
            .ok_or_else(|| AutoeqError::MissingCea2034Curve {
                curve_name: "Estimated In-Room Response".to_string(),
            })?
            .spl
            .clone();

        // Verify all arrays have the same length
        if on.len() != lw.len() || on.len() != sp.len() || on.len() != pir.len() {
            return Err(AutoeqError::CurveLengthMismatch {
                on_len: on.len(),
                lw_len: lw.len(),
                sp_len: sp.len(),
                pir_len: pir.len(),
            });
        }

        Ok(Self { on, lw, sp, pir })
    }
}

/// Data required for computing headphone loss
#[derive(Debug, Clone)]
pub struct HeadphoneLossData {
    /// Enable smoothing (regularization) of the inverted target curve
    pub smooth: bool,
    /// Smoothing level as 1/N octave (N in [1..24])
    pub smooth_n: usize,
}

impl HeadphoneLossData {
    /// Create a new HeadphoneLossData instance
    ///
    /// # Arguments
    /// * `smooth` - Enable smoothing
    /// * `smooth_n` - Smoothing level as 1/N octave
    pub fn new(smooth: bool, smooth_n: usize) -> Self {
        Self { smooth, smooth_n }
    }
}

/// Whether a loss uses the PEQ parameter layout.
///
/// Driver and multi-sub layouts carry gains/delays/crossovers rather than
/// frequency/Q/gain triplets, so per-filter and composite PEQ envelope rules
/// do not apply to them. Mirrors the guard in `compute.rs`.
pub fn is_peq_layout_loss(loss: LossType) -> bool {
    !matches!(loss, LossType::DriversFlat | LossType::MultiSubFlat)
}
