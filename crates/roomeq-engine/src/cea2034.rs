//! Pure CEA-2034 speaker pre-correction for the three-pass RoomEQ pipeline.

use std::collections::HashMap;

use autoeq_core::{
    AutoeqError, Curve, Result, SpinoramaBundle, interpolate_log_space,
    normalize_and_interpolate_response, response,
};
use log::{debug, info};
use math_audio_iir_fir::{Biquad, BiquadFilterType, DEFAULT_Q_HIGH_LOW_SHELF};
use roomeq_model::{
    Cea2034CorrectionConfig, Cea2034CorrectionMode, OptimizerConfig, UserPreference,
};

/// Speed of sound in m/s at approximately 20 °C.
const SPEED_OF_SOUND: f64 = 343.0;

pub struct SpeakerCorrectionResult {
    pub filters: Vec<Biquad>,
    pub corrected_curve: Curve,
    pub optimizer_evidence: Vec<autoeq_optim::optim::OptimizerRunEvidence>,
}

pub fn compute_speaker_correction(
    cea2034_data: &SpinoramaBundle,
    config: &Cea2034CorrectionConfig,
    room_curve: &Curve,
    schroeder_freq: f64,
    arrival_time_ms: Option<f64>,
    sample_rate: f64,
) -> Result<(Vec<Biquad>, Curve)> {
    compute_speaker_correction_detailed(
        cea2034_data,
        config,
        room_curve,
        schroeder_freq,
        arrival_time_ms,
        sample_rate,
    )
    .map(|result| (result.filters, result.corrected_curve))
}

pub fn compute_speaker_correction_detailed(
    cea2034_data: &SpinoramaBundle,
    config: &Cea2034CorrectionConfig,
    room_curve: &Curve,
    schroeder_freq: f64,
    arrival_time_ms: Option<f64>,
    sample_rate: f64,
) -> Result<SpeakerCorrectionResult> {
    match resolve_correction_mode(config, arrival_time_ms) {
        Cea2034CorrectionMode::Flat => compute_flat_lw_correction(
            cea2034_data,
            config,
            room_curve,
            schroeder_freq,
            sample_rate,
        ),
        Cea2034CorrectionMode::Score => compute_score_correction(
            cea2034_data,
            config,
            room_curve,
            schroeder_freq,
            sample_rate,
        ),
        Cea2034CorrectionMode::Auto => unreachable!("Auto mode should have been resolved"),
    }
}

fn compute_flat_lw_correction(
    cea2034_data: &SpinoramaBundle,
    config: &Cea2034CorrectionConfig,
    room_curve: &Curve,
    schroeder_freq: f64,
    sample_rate: f64,
) -> Result<SpeakerCorrectionResult> {
    if room_curve.freq.is_empty() {
        return Err(AutoeqError::InvalidMeasurement {
            message: "Room curve has no frequency data for CEA2034 correction".to_string(),
        });
    }

    let listening_window =
        normalize_and_interpolate_response(&room_curve.freq, &cea2034_data.listening_window);
    info!(
        "  Flat LW correction: {} filters, {:.0}-{:.0} Hz, max_db={:.1}, min_db={:.1}",
        config.num_filters,
        schroeder_freq,
        room_curve.freq[room_curve.freq.len() - 1],
        config.max_db,
        config.min_db
    );

    let optimizer_config = OptimizerConfig {
        num_filters: config.num_filters,
        min_freq: schroeder_freq,
        max_freq: 20_000.0,
        min_q: 0.5,
        max_q: config.max_q,
        min_db: config.min_db,
        max_db: config.max_db,
        loss_type: "flat".to_string(),
        asymmetric_loss: false,
        psychoacoustic: false,
        refine: true,
        ..OptimizerConfig::default()
    };
    let result = crate::eq::optimize_channel_eq_detailed(
        &listening_window,
        &optimizer_config,
        None,
        sample_rate,
    )
    .map_err(|error| AutoeqError::OptimizationFailed {
        message: format!("CEA2034 flat LW correction failed: {error}"),
    })?;

    info!(
        "  CEA2034 flat LW correction: {} filters, final loss={:.4}",
        result.filters.len(),
        result.loss
    );
    for filter in &result.filters {
        debug!(
            "    {:.0} Hz, Q={:.2}, {:.1} dB",
            filter.freq, filter.q, filter.db_gain
        );
    }

    Ok(SpeakerCorrectionResult {
        corrected_curve: simulate_correction(&result.filters, room_curve, sample_rate),
        filters: result.filters,
        optimizer_evidence: result.optimizer_evidence,
    })
}

fn compute_score_correction(
    cea2034_data: &SpinoramaBundle,
    config: &Cea2034CorrectionConfig,
    room_curve: &Curve,
    schroeder_freq: f64,
    sample_rate: f64,
) -> Result<SpeakerCorrectionResult> {
    room_curve.validate("CEA2034 score room curve")?;
    for (name, curve) in [
        ("On Axis", &cea2034_data.on_axis),
        ("Listening Window", &cea2034_data.listening_window),
        ("Sound Power", &cea2034_data.sound_power),
        (
            "Estimated In-Room Response",
            &cea2034_data.estimated_in_room,
        ),
    ] {
        curve.validate(&format!("CEA2034 {name}"))?;
    }

    let grid = &room_curve.freq;
    let spin_data: HashMap<String, Curve> = [
        ("On Axis", &cea2034_data.on_axis),
        ("Listening Window", &cea2034_data.listening_window),
        ("Sound Power", &cea2034_data.sound_power),
        (
            "Estimated In-Room Response",
            &cea2034_data.estimated_in_room,
        ),
    ]
    .into_iter()
    .map(|(name, curve)| (name.to_string(), interpolate_log_space(grid, curve)))
    .collect();
    let listening_window =
        spin_data
            .get("Listening Window")
            .ok_or_else(|| AutoeqError::MissingCea2034Curve {
                curve_name: "Listening Window".to_string(),
            })?;
    let optimizer_config = OptimizerConfig {
        num_filters: config.num_filters,
        min_freq: schroeder_freq,
        max_freq: 20_000.0,
        min_q: 0.5,
        max_q: config.max_q,
        min_db: config.min_db,
        max_db: config.max_db,
        loss_type: "score".to_string(),
        asymmetric_loss: false,
        psychoacoustic: false,
        refine: true,
        ..OptimizerConfig::default()
    };
    let result = crate::eq::optimize_channel_eq_with_spin_detailed(
        listening_window,
        &spin_data,
        &optimizer_config,
        None,
        sample_rate,
    )
    .map_err(|error| AutoeqError::OptimizationFailed {
        message: format!("CEA2034 score correction failed: {error}"),
    })?;

    info!(
        "  CEA2034 score correction: {} filters, final loss={:.4}",
        result.filters.len(),
        result.loss
    );
    for filter in &result.filters {
        debug!(
            "    {:.0} Hz, Q={:.2}, {:+.1} dB",
            filter.freq, filter.q, filter.db_gain
        );
    }

    Ok(SpeakerCorrectionResult {
        corrected_curve: simulate_correction(&result.filters, room_curve, sample_rate),
        filters: result.filters,
        optimizer_evidence: result.optimizer_evidence,
    })
}

pub fn resolve_correction_mode(
    config: &Cea2034CorrectionConfig,
    arrival_time_ms: Option<f64>,
) -> Cea2034CorrectionMode {
    match config.correction_mode {
        Cea2034CorrectionMode::Flat => Cea2034CorrectionMode::Flat,
        Cea2034CorrectionMode::Score => Cea2034CorrectionMode::Score,
        Cea2034CorrectionMode::Auto => {
            let distance_m = config.listening_distance_m.or_else(|| {
                arrival_time_ms.map(|arrival_ms| {
                    let latency_ms = config.system_latency_ms.unwrap_or(0.0);
                    (arrival_ms - latency_ms).max(0.0) * 0.001 * SPEED_OF_SOUND
                })
            });
            let Some(distance_m) = distance_m else {
                info!("  Auto mode: no distance info available, defaulting to Flat LW correction");
                return Cea2034CorrectionMode::Flat;
            };
            if !distance_m.is_finite() || !config.nearfield_threshold_m.is_finite() {
                info!("  Auto mode: invalid distance metadata, defaulting to Flat LW correction");
                return Cea2034CorrectionMode::Flat;
            }
            if distance_m >= config.nearfield_threshold_m {
                info!(
                    "  Auto mode: distance={:.2}m >= threshold={:.1}m -> CEA2034 score correction",
                    distance_m, config.nearfield_threshold_m
                );
                Cea2034CorrectionMode::Score
            } else {
                info!(
                    "  Auto mode: distance={:.2}m < threshold={:.1}m -> Flat LW correction",
                    distance_m, config.nearfield_threshold_m
                );
                Cea2034CorrectionMode::Flat
            }
        }
    }
}

pub fn simulate_correction(filters: &[Biquad], curve: &Curve, sample_rate: f64) -> Curve {
    if filters.is_empty() {
        return curve.clone();
    }
    let filter_response = response::compute_peq_complex_response(filters, &curve.freq, sample_rate);
    response::apply_complex_response(curve, &filter_response)
}

pub fn generate_preference_filters(preference: &UserPreference, sample_rate: f64) -> Vec<Biquad> {
    let mut filters = Vec::new();
    if preference.bass_shelf_db.abs() > 0.1 {
        filters.push(Biquad::new(
            BiquadFilterType::Lowshelf,
            preference.bass_shelf_freq,
            sample_rate,
            DEFAULT_Q_HIGH_LOW_SHELF,
            preference.bass_shelf_db,
        ));
        info!(
            "  Pass 3 preference: bass shelf {:+.1} dB at {:.0} Hz",
            preference.bass_shelf_db, preference.bass_shelf_freq
        );
    }
    if preference.treble_shelf_db.abs() > 0.1 {
        filters.push(Biquad::new(
            BiquadFilterType::Highshelf,
            preference.treble_shelf_freq,
            sample_rate,
            DEFAULT_Q_HIGH_LOW_SHELF,
            preference.treble_shelf_db,
        ));
        info!(
            "  Pass 3 preference: treble shelf {:+.1} dB at {:.0} Hz",
            preference.treble_shelf_db, preference.treble_shelf_freq
        );
    }
    filters
}

#[cfg(test)]
mod tests {
    use ndarray::Array1;

    use super::*;

    fn flat_curve(points: usize) -> Curve {
        Curve {
            freq: Array1::logspace(10.0, f64::log10(20.0), f64::log10(20_000.0), points),
            spl: Array1::from_elem(points, 85.0),
            ..Curve::default()
        }
    }

    fn spinorama(points: usize) -> SpinoramaBundle {
        SpinoramaBundle {
            on_axis: flat_curve(points),
            listening_window: flat_curve(points),
            early_reflections: flat_curve(points),
            sound_power: flat_curve(points),
            estimated_in_room: flat_curve(points),
            er_di: flat_curve(points),
            sp_di: flat_curve(points),
            curves: HashMap::new(),
        }
    }

    #[test]
    fn preference_filters_cover_both_shelves_and_thresholds() {
        let filters = generate_preference_filters(
            &UserPreference {
                bass_shelf_db: 3.0,
                bass_shelf_freq: 200.0,
                treble_shelf_db: -1.5,
                treble_shelf_freq: 8_000.0,
            },
            48_000.0,
        );
        assert_eq!(filters.len(), 2);
        assert_eq!(filters[0].filter_type, BiquadFilterType::Lowshelf);
        assert_eq!(filters[1].filter_type, BiquadFilterType::Highshelf);

        let filters = generate_preference_filters(
            &UserPreference {
                bass_shelf_db: 5.0,
                bass_shelf_freq: 150.0,
                treble_shelf_db: 0.05,
                treble_shelf_freq: 8_000.0,
            },
            48_000.0,
        );
        assert_eq!(filters.len(), 1);
        assert!(generate_preference_filters(&UserPreference::default(), 48_000.0).is_empty());
    }

    #[test]
    fn correction_mode_resolves_manual_distance_and_arrival_paths() {
        for mode in [Cea2034CorrectionMode::Flat, Cea2034CorrectionMode::Score] {
            assert_eq!(
                resolve_correction_mode(
                    &Cea2034CorrectionConfig {
                        correction_mode: mode.clone(),
                        ..Cea2034CorrectionConfig::default()
                    },
                    None,
                ),
                mode
            );
        }
        let near = Cea2034CorrectionConfig {
            correction_mode: Cea2034CorrectionMode::Auto,
            nearfield_threshold_m: 2.0,
            listening_distance_m: Some(1.5),
            ..Cea2034CorrectionConfig::default()
        };
        assert_eq!(
            resolve_correction_mode(&near, None),
            Cea2034CorrectionMode::Flat
        );
        let far = Cea2034CorrectionConfig {
            listening_distance_m: Some(3.0),
            ..near.clone()
        };
        assert_eq!(
            resolve_correction_mode(&far, None),
            Cea2034CorrectionMode::Score
        );
        let arrival = Cea2034CorrectionConfig {
            listening_distance_m: None,
            system_latency_ms: Some(2.0),
            ..near
        };
        assert_eq!(
            resolve_correction_mode(&arrival, Some(8.83)),
            Cea2034CorrectionMode::Score
        );
        assert_eq!(
            resolve_correction_mode(&arrival, Some(5.0)),
            Cea2034CorrectionMode::Flat
        );
        assert_eq!(
            resolve_correction_mode(&arrival, None),
            Cea2034CorrectionMode::Flat
        );
    }

    #[test]
    fn score_correction_uses_spinorama_objective() {
        let room_curve = flat_curve(32);
        let config = Cea2034CorrectionConfig {
            enabled: true,
            correction_mode: Cea2034CorrectionMode::Score,
            num_filters: 1,
            ..Cea2034CorrectionConfig::default()
        };
        let (filters, corrected) =
            compute_speaker_correction(&spinorama(32), &config, &room_curve, 300.0, None, 48_000.0)
                .unwrap();
        assert!(filters.len() <= config.num_filters);
        assert_eq!(corrected.freq, room_curve.freq);
        assert!(corrected.spl.iter().all(|value| value.is_finite()));
    }

    #[test]
    fn empty_room_curve_is_rejected_and_empty_filter_simulation_is_identity() {
        let empty = Curve {
            freq: Array1::zeros(0),
            spl: Array1::zeros(0),
            ..Curve::default()
        };
        assert!(
            compute_speaker_correction(
                &spinorama(32),
                &Cea2034CorrectionConfig {
                    enabled: true,
                    correction_mode: Cea2034CorrectionMode::Flat,
                    ..Cea2034CorrectionConfig::default()
                },
                &empty,
                300.0,
                None,
                48_000.0,
            )
            .is_err()
        );

        let curve = flat_curve(32);
        assert_eq!(simulate_correction(&[], &curve, 48_000.0).spl, curve.spl);
    }
}
