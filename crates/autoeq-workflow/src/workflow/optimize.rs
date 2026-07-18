use super::build::build_target_curve;
use super::misc::interpolate_cea2034_data;
use super::types::HeadphoneOptResult;
use super::types::SpeakerOptResult;
use super::types::compute_visualization_curves;
use crate::Curve;
use crate::iir::Biquad;
pub use crate::optim::setup::*;
use crate::read;
use crate::x2peq;
pub use autoeq_optim::{optimize_drivers_crossover, optimize_multisub};
use std::collections::HashMap;
use std::error::Error;
use std::path::PathBuf;

/// Run complete speaker optimization from spinorama data
///
/// # Arguments
/// * `speaker` - Speaker name
/// * `version` - Version (e.g., "asr")
/// * `measurement` - Measurement type (e.g., "CEA2034")
/// * `args` - Optimization arguments (use `Args::speaker_defaults()` as base)
/// * `progress_config` - Optional progress callback configuration
/// * `progress_callback` - Optional progress callback
///
/// # Returns
/// Complete optimization result with all curves
pub async fn optimize_speaker<F>(
    input: &crate::workflow::InputConfig,
    params: &crate::OptimParams,
    progress_config: Option<ProgressCallbackConfig>,
    progress_callback: Option<F>,
) -> Result<SpeakerOptResult, Box<dyn Error>>
where
    F: FnMut(&ProgressUpdate) -> crate::de::CallbackAction + Send + 'static,
{
    optimize_speaker_with_grid(
        input,
        params,
        &crate::workflow::VisualizationGridConfig::default(),
        progress_config,
        progress_callback,
    )
    .await
}

/// Run speaker optimization with an explicit normalization/report grid.
pub async fn optimize_speaker_with_grid<F>(
    input: &crate::workflow::InputConfig,
    params: &crate::OptimParams,
    visualization_grid: &crate::workflow::VisualizationGridConfig,
    progress_config: Option<ProgressCallbackConfig>,
    progress_callback: Option<F>,
) -> Result<SpeakerOptResult, Box<dyn Error>>
where
    F: FnMut(&ProgressUpdate) -> crate::de::CallbackAction + Send + 'static,
{
    // 1. Load measurement with spin data
    let speaker = input.speaker.as_deref().unwrap_or("");
    let version = input.version.as_deref().unwrap_or("");
    let measurement = input.measurement.as_deref().unwrap_or("");
    let (input_curve, spin_data) =
        read::load_spinorama_with_spin(speaker, version, measurement, &input.curve_name).await?;

    // 2. Normalize to standard frequency grid
    let standard_freq = visualization_grid.create_frequency_grid(params)?;
    let input_normalized = read::normalize_and_interpolate_response(&standard_freq, &input_curve);

    // 3. Build target curve
    let target_curve = build_target_curve(
        &crate::workflow::TargetConfig {
            target_path: None,
            curve_name: input.curve_name.clone(),
        },
        &standard_freq,
        &input_normalized,
    )?;

    // 4. Create deviation curve
    let deviation_curve = Curve {
        freq: target_curve.freq.clone(),
        spl: &target_curve.spl - &input_normalized.spl,
        phase: None,
        ..Default::default()
    };

    // 5. Setup objective - normalize spin data to same frequency grid
    let spin_map = spin_data.as_ref().map(|s| {
        s.curves
            .iter()
            .map(|(name, curve)| {
                let normalized = read::normalize_and_interpolate_response(&standard_freq, curve);
                (name.clone(), normalized)
            })
            .collect::<HashMap<String, Curve>>()
    });
    let (objective_data, _) = setup_objective_data(
        params,
        &input_normalized,
        &target_curve,
        &deviation_curve,
        &spin_map,
    )?;

    // 6. Run optimization
    let (opt_params, history) = if let (Some(config), Some(callback)) =
        (progress_config, progress_callback)
    {
        let output = perform_optimization_with_progress(params, &objective_data, config, callback)?;
        (output.params, output.history)
    } else {
        let p = perform_optimization_with_callback(
            params,
            &objective_data,
            Box::new(|_| crate::de::CallbackAction::Continue),
        )?;
        (p, Vec::new())
    };

    // 7. Convert to biquads
    let biquads: Vec<Biquad> = x2peq(&opt_params, params.sample_rate, params.peq_model)
        .into_iter()
        .map(|(_, b)| b)
        .collect();

    // 8. Compute visualization curves
    let frequencies: Vec<f64> = standard_freq.iter().copied().collect();
    let curves =
        compute_visualization_curves(&frequencies, &input_normalized, &target_curve, &biquads)?;

    let initial_loss = history.first().map(|x| x.1).unwrap_or(0.0);
    let final_loss = history.last().map(|x| x.1).unwrap_or(0.0);

    // Interpolate spin_data to standard frequency grid for consistent visualization
    // Note: Does NOT normalize - preserves original dB levels
    let interpolated_spin_data = spin_data.map(|s| interpolate_cea2034_data(&s, &standard_freq));

    Ok(SpeakerOptResult {
        biquads,
        curves,
        spin_data: interpolated_spin_data,
        history,
        initial_loss,
        final_loss,
    })
}

/// Run complete headphone optimization from CSV measurement
///
/// # Arguments
/// * `curve_path` - Path to headphone measurement CSV
/// * `target_curve` - Target curve (use bundled Harman curves or custom)
/// * `args` - Optimization arguments (use `Args::headphone_defaults()` as base)
/// * `progress_config` - Optional progress callback configuration
/// * `progress_callback` - Optional progress callback
///
/// # Returns
/// Complete optimization result with all curves
pub fn optimize_headphone<F>(
    curve_path: &PathBuf,
    target_curve: &Curve,
    params: &crate::OptimParams,
    progress_config: Option<ProgressCallbackConfig>,
    progress_callback: Option<F>,
) -> Result<HeadphoneOptResult, Box<dyn Error>>
where
    F: FnMut(&ProgressUpdate) -> crate::de::CallbackAction + Send + 'static,
{
    optimize_headphone_with_grid(
        curve_path,
        target_curve,
        params,
        &crate::workflow::VisualizationGridConfig::default(),
        progress_config,
        progress_callback,
    )
}

/// Run headphone optimization with an explicit normalization/report grid.
pub fn optimize_headphone_with_grid<F>(
    curve_path: &PathBuf,
    target_curve: &Curve,
    params: &crate::OptimParams,
    visualization_grid: &crate::workflow::VisualizationGridConfig,
    progress_config: Option<ProgressCallbackConfig>,
    progress_callback: Option<F>,
) -> Result<HeadphoneOptResult, Box<dyn Error>>
where
    F: FnMut(&ProgressUpdate) -> crate::de::CallbackAction + Send + 'static,
{
    // 1. Load measurement
    let input_curve = read::read_curve_from_csv(curve_path)?;

    // 2. Normalize to standard frequency grid
    let standard_freq = visualization_grid.create_frequency_grid(params)?;
    let input_normalized = read::normalize_and_interpolate_response(&standard_freq, &input_curve);
    let target_normalized = read::normalize_and_interpolate_response(&standard_freq, target_curve);

    // 3. Create deviation curve
    let deviation_curve = Curve {
        freq: target_normalized.freq.clone(),
        spl: &target_normalized.spl - &input_normalized.spl,
        phase: None,
        ..Default::default()
    };

    // 4. Setup objective
    let (objective_data, _) = setup_objective_data(
        params,
        &input_normalized,
        &target_normalized,
        &deviation_curve,
        &None,
    )?;

    // 5. Run optimization
    let (opt_params, history) = if let (Some(config), Some(callback)) =
        (progress_config, progress_callback)
    {
        let output = perform_optimization_with_progress(params, &objective_data, config, callback)?;
        (output.params, output.history)
    } else {
        let p = perform_optimization_with_callback(
            params,
            &objective_data,
            Box::new(|_| crate::de::CallbackAction::Continue),
        )?;
        (p, Vec::new())
    };

    // 6. Convert to biquads
    let biquads: Vec<Biquad> = x2peq(&opt_params, params.sample_rate, params.peq_model)
        .into_iter()
        .map(|(_, b)| b)
        .collect();

    // 7. Compute visualization curves
    let frequencies: Vec<f64> = standard_freq.iter().copied().collect();
    let curves = compute_visualization_curves(
        &frequencies,
        &input_normalized,
        &target_normalized,
        &biquads,
    )?;

    let initial_loss = history.first().map(|x| x.1).unwrap_or(0.0);
    let final_loss = history.last().map(|x| x.1).unwrap_or(0.0);

    Ok(HeadphoneOptResult {
        biquads,
        curves,
        history,
        initial_loss,
        final_loss,
    })
}
