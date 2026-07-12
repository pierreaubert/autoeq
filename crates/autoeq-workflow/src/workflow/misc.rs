use crate::Cea2034Data;
use crate::Curve;
use crate::read;
use ndarray::Array1;
use std::collections::HashMap;

/// Interpolate all curves in Cea2034Data to a standard frequency grid
/// Note: Does NOT normalize - preserves original dB levels for proper visualization
pub(super) fn interpolate_cea2034_data(
    spin_data: &Cea2034Data,
    standard_freq: &Array1<f64>,
) -> Cea2034Data {
    let interpolate = |curve: &Curve| read::interpolate_response(standard_freq, curve);

    let on_axis = interpolate(&spin_data.on_axis);
    let listening_window = interpolate(&spin_data.listening_window);
    let early_reflections = interpolate(&spin_data.early_reflections);
    let sound_power = interpolate(&spin_data.sound_power);
    let estimated_in_room = interpolate(&spin_data.estimated_in_room);
    let er_di = interpolate(&spin_data.er_di);
    let sp_di = interpolate(&spin_data.sp_di);

    // Build interpolated curves HashMap
    let mut curves = HashMap::new();
    curves.insert("On Axis".to_string(), on_axis.clone());
    curves.insert("Listening Window".to_string(), listening_window.clone());
    curves.insert("Early Reflections".to_string(), early_reflections.clone());
    curves.insert("Sound Power".to_string(), sound_power.clone());
    curves.insert(
        "Estimated In-Room Response".to_string(),
        estimated_in_room.clone(),
    );

    Cea2034Data {
        on_axis,
        listening_window,
        early_reflections,
        sound_power,
        estimated_in_room,
        er_di,
        sp_di,
        curves,
    }
}

/// Create minimal optimization parameters for driver/multi-sub optimization.
///
/// This avoids requiring full CLI args when calling from library code.
#[allow(clippy::too_many_arguments)]
pub(super) fn create_driver_optimization_args(
    min_freq: f64,
    max_freq: f64,
    sample_rate: f64,
    algorithm: &str,
    max_iter: usize,
    population: usize,
    min_db: f64,
    max_db: f64,
    seed: Option<u64>,
) -> crate::OptimParams {
    use crate::LossType;
    use crate::PeqModel;

    crate::OptimParams {
        num_filters: 0, // Not used for driver optimization
        peq_model: PeqModel::Pk,
        sample_rate,
        min_freq,
        max_freq,
        min_q: 0.5,
        max_q: 10.0,
        min_db,
        max_db,
        loss: LossType::DriversFlat,
        smooth: false,
        smooth_n: 1,
        min_spacing_oct: 0.0,
        spacing_weight: 0.0,
        smoothness_penalty: None,
        audibility_deadband: None,
        algo: algorithm.to_string(),
        population,
        maxeval: max_iter,
        refine: false,
        local_algo: "cobyla".to_string(),
        bo_initial_samples: 0,
        bo_batch_size: 0,
        bo_posterior_std_threshold: 0.0,
        bo_acquisition: "qei".to_string(),
        bo_ehvi: false,
        strategy: "currenttobest1bin".to_string(),
        tolerance: 1e-3,
        atolerance: 1e-4,
        recombination: 0.9,
        adaptive_weight_f: 0.9,
        adaptive_weight_cr: 0.9,
        no_parallel: false,
        parallel_threads: num_cpus::get(),
        seed,
        quiet: false,
    }
}
