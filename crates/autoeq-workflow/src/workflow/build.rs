use crate::AutoeqError;
use crate::Curve;
use crate::read;
use ndarray::Array1;

/// Build a target curve from CLI args and the input curve.
///
/// Delegates to [`build_target_curve_by_name`] for predefined curve names,
/// or loads from a CSV file path specified in `args.target`.
///
/// # Errors
///
/// Returns `AutoeqError::TargetCurveLoad` if loading from a CSV file fails.
pub fn build_target_curve(
    target: &crate::workflow::TargetConfig,
    freqs: &Array1<f64>,
    input_curve: &Curve,
) -> Result<Curve, AutoeqError> {
    if let Some(ref target_path) = target.target_path {
        log::debug!(
            "[RUST DEBUG] Loading target curve from path: {}",
            target_path.display()
        );

        let target_curve =
            read::read_curve_from_csv(target_path).map_err(|e| AutoeqError::TargetCurveLoad {
                path: target_path.display().to_string(),
                message: e.to_string(),
            })?;
        Ok(read::normalize_and_interpolate_response(
            freqs,
            &target_curve,
        ))
    } else {
        build_target_curve_by_name(&target.curve_name, freqs, input_curve)
    }
}

/// Build a predefined target curve by name.
///
/// This function is the CLI-independent core of target curve generation.
/// It handles predefined curve names ("Listening Window", "Sound Power", etc.)
/// without requiring a `cli::Args` struct.
///
/// # Arguments
/// * `curve_name` - Name of the predefined curve (e.g. "Listening Window", "On Axis")
/// * `freqs` - Frequency grid for the target curve
/// * `input_curve` - Reference measurement curve (used for slope estimation)
///
/// # Returns
/// A flat (0 dB) target by default; slope-corrected for specific curve names.
pub fn build_target_curve_by_name(
    curve_name: &str,
    freqs: &Array1<f64>,
    input_curve: &Curve,
) -> Result<Curve, AutoeqError> {
    Ok(autoeq_core::build_target_curve_by_name(
        curve_name,
        freqs,
        input_curve,
    ))
}
