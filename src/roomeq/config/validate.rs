use super::super::types::{
    AreaPriorKind, AreaQuadratureKind, AreaScalarisationKind, BootstrapScalarisation,
    MultiMeasurementStrategy, MultiSeatStrategy, OptimizerConfig, RoomConfig, SpeakerConfig,
    TargetShape,
};
use super::misc::collect_sources;
use super::misc::is_valid_speaker_name;
use super::misc::source_is_cea2034_shaped;
use super::optimizer_rules::run_optimizer_validation_rules;
use super::validation_result::{ValidationContext, ValidationResult};
use crate::MeasurementSource;
use std::collections::HashMap;

/// Validate a complete room configuration
pub fn validate_room_config(config: &RoomConfig) -> ValidationResult {
    let mut result = ValidationResult::valid();

    // Validate optimizer config
    validate_optimizer_config(&config.optimizer, &mut result);

    // I1 — warn when both target_curve and a non-trivial target_response
    // are set. `target_response` is baked into the measurement during
    // optimization; when both are present, `target_curve` is silently
    // ignored to prevent double-application. Users often don't realise
    // that setting both means only one takes effect.
    if config.target_curve.is_some()
        && let Some(ref tr) = config.optimizer.target_response
    {
        let has_shape = tr.shape != TargetShape::Flat
            || tr.slope_db_per_octave.abs() > 1e-6
            || tr.preference.bass_shelf_db.abs() > 1e-6
            || tr.preference.treble_shelf_db.abs() > 1e-6
            || tr.broadband_precorrection;
        if has_shape {
            result.add_warning(
                "Both target_curve and target_response are configured. \
                 target_response takes precedence — it is baked into the \
                 measurement before EQ optimization, and target_curve is \
                 ignored to avoid double-application. Set only one."
                    .to_string(),
            );
        }
    }

    // Validate speaker configurations
    validate_speakers(&config.speakers, &mut result);
    validate_system_speaker_references(config, &mut result);

    // Validate crossover references
    validate_crossovers(&config.speakers, config.crossovers.as_ref(), &mut result);

    // Cross-validate option interactions that depend on the speaker map
    // (multi-measurement weights, CEA2034 source detection).
    validate_cross_option_interactions(config, &mut result);

    result
}

/// Validate optimizer configuration parameters
fn validate_optimizer_config(opt: &OptimizerConfig, result: &mut ValidationResult) {
    let mut ctx = ValidationContext::new(opt, result);
    run_optimizer_validation_rules(&mut ctx);
}

/// Validate speaker configurations
fn validate_speakers(speakers: &HashMap<String, SpeakerConfig>, result: &mut ValidationResult) {
    if speakers.is_empty() {
        result.add_error("No speakers configured".to_string());
        return;
    }

    for (name, config) in speakers {
        // Validate speaker model name if provided
        if let Some(speaker_name) = config.speaker_name()
            && !is_valid_speaker_name(speaker_name)
        {
            result.add_error(format!(
                "Speaker '{}' has invalid speaker_name '{}'. Only alphanumeric, spaces, and hyphens allowed.",
                name, speaker_name
            ));
        }

        match config {
            SpeakerConfig::Group(group) => {
                if group.measurements.is_empty() {
                    result.add_error(format!("Speaker group '{}' has no measurements", name));
                }
                if group.measurements.len() == 1 {
                    result.add_warning(format!(
                        "Speaker group '{}' has only 1 measurement, consider using Single config",
                        name
                    ));
                }
                if group.crossover.is_none() && group.measurements.len() > 1 {
                    result.add_error(format!(
                        "Speaker group '{}' has multiple drivers but no crossover specified",
                        name
                    ));
                }
            }
            SpeakerConfig::MultiSub(ms) => {
                if ms.subwoofers.is_empty() {
                    result.add_error(format!("Multi-sub '{}' has no subwoofers", name));
                }
                if ms.subwoofers.len() == 1 {
                    result.add_warning(format!(
                        "Multi-sub '{}' has only 1 subwoofer, consider using Single config",
                        name
                    ));
                }
            }
            SpeakerConfig::Dba(dba) => {
                if dba.front.is_empty() {
                    result.add_error(format!("DBA '{}' has no front speakers", name));
                }
                if dba.rear.is_empty() {
                    result.add_error(format!("DBA '{}' has no rear speakers", name));
                }
            }
            SpeakerConfig::Cardioid(cardioid) => {
                if cardioid.separation_meters <= 0.0 {
                    result.add_error(format!(
                        "Cardioid '{}' has invalid separation {:.2}m (must be > 0)",
                        name, cardioid.separation_meters
                    ));
                }
            }
            SpeakerConfig::SupportingSource(s) => {
                // Supporting source requires both primary and support measurements.
                if s.primary.speaker_name().is_some()
                    && s.support.speaker_name().is_some()
                    && s.primary.speaker_name() == s.support.speaker_name()
                {
                    result.add_warning(format!(
                        "Supporting source '{}' uses the same speaker_name for primary and support",
                        name
                    ));
                }

                let cfg = &s.supporting_source;
                if !(1.0..=50.0).contains(&cfg.delay_ms) {
                    result.add_error(format!(
                        "Supporting source '{}' delay_ms must be in [1.0, 50.0], got {:.2}",
                        name, cfg.delay_ms
                    ));
                }
                if cfg.fir_taps == 0 || cfg.fir_taps > 65536 || cfg.fir_taps % 2 != 0 {
                    result.add_error(format!(
                        "Supporting source '{}' fir_taps must be even and in [2, 65536], got {}",
                        name, cfg.fir_taps
                    ));
                }
                if cfg.velvet_noise_taps >= cfg.fir_taps {
                    result.add_warning(format!(
                        "Supporting source '{}' velvet_noise_taps ({}) should be < fir_taps ({})",
                        name, cfg.velvet_noise_taps, cfg.fir_taps
                    ));
                }
                if cfg.freq_range_hz.0 >= cfg.freq_range_hz.1 {
                    result.add_error(format!(
                        "Supporting source '{}' freq_range_hz must satisfy low < high, got {:?}",
                        name, cfg.freq_range_hz
                    ));
                }
                for (i, band) in cfg.precedence_limits.iter().enumerate() {
                    if band.low_hz >= band.high_hz {
                        result.add_error(format!(
                            "Supporting source '{}' precedence_limits[{}]: low_hz must be < high_hz",
                            name, i
                        ));
                    }
                    if band.limit_db <= 0.0 {
                        result.add_warning(format!(
                            "Supporting source '{}' precedence_limits[{}]: limit_db <= 0 dB will prevent compensation",
                            name, i
                        ));
                    }
                }
            }
            SpeakerConfig::Single(_) => {
                // Single speaker - minimal validation, path existence checked at load time
            }
        }
    }
}

/// Validate crossover references
fn validate_system_speaker_references(config: &RoomConfig, result: &mut ValidationResult) {
    let Some(system) = &config.system else {
        return;
    };
    for (role, measurement_key) in &system.speakers {
        if !config.speakers.contains_key(measurement_key) {
            result.add_error(format!(
                "system.speakers role '{role}' references missing speaker measurement '{measurement_key}'"
            ));
        }
    }
}

fn validate_crossovers(
    speakers: &HashMap<String, SpeakerConfig>,
    crossovers: Option<&HashMap<String, super::super::types::CrossoverConfig>>,
    result: &mut ValidationResult,
) {
    if let Some(crossovers) = crossovers {
        for (name, crossover) in crossovers {
            if crossover
                .crossover_type
                .parse::<crate::loss::CrossoverType>()
                .is_err()
            {
                result.add_error(format!(
                    "Crossover '{name}' has unsupported type '{}'",
                    crossover.crossover_type
                ));
            }
        }
    }

    for (name, config) in speakers {
        let SpeakerConfig::Group(group) = config else {
            continue;
        };
        let Some(ref crossover_ref) = group.crossover else {
            continue;
        };

        let Some(crossovers) = crossovers else {
            result.add_error(format!(
                "Speaker '{}' references crossover '{}' but no crossovers defined",
                name, crossover_ref
            ));
            continue;
        };

        if !crossovers.contains_key(crossover_ref) {
            result.add_error(format!(
                "Speaker '{}' references non-existent crossover '{}'",
                name, crossover_ref
            ));
            continue;
        }

        // Validate crossover config
        let crossover = &crossovers[crossover_ref];
        let num_drivers = group.measurements.len();
        let expected_freqs = num_drivers.saturating_sub(1);

        // Check frequency specification
        let has_single = crossover.frequency.is_some();
        let has_multiple = crossover.frequencies.is_some();
        let has_range = crossover.frequency_range.is_some();

        if has_single && num_drivers != 2 {
            result.add_warning(format!(
                "Crossover '{}' has single frequency but speaker '{}' has {} drivers",
                crossover_ref, name, num_drivers
            ));
        }

        if has_multiple
            && let Some(ref freqs) = crossover.frequencies
            && freqs.len() != expected_freqs
        {
            result.add_error(format!(
                "Crossover '{}' has {} frequencies but speaker '{}' needs {} for {} drivers",
                crossover_ref,
                freqs.len(),
                name,
                expected_freqs,
                num_drivers
            ));
        }

        if !has_single && !has_multiple && !has_range {
            // Will be auto-optimized
            result.add_warning(format!(
                "Crossover '{}' has no frequency specified, will be auto-optimized",
                crossover_ref
            ));
        }
    }
}

/// Validate interactions between optimizer options and the resolved speaker map.
///
/// Covers:
/// - B10: `multi_measurement.weights.len()` must match the number of
///   measurements on every `MeasurementSource::Multiple` in the speaker map.
/// - I4: `cea2034_correction.enabled` requires that at least one speaker
///   carries a CEA2034/spinorama-shaped source (speaker_name set, or a path
///   that contains "cea2034"/"spinorama"). Applying the 3-pass pipeline to
///   plain in-room responses silently produces garbage.
fn validate_cross_option_interactions(config: &RoomConfig, result: &mut ValidationResult) {
    validate_multi_measurement_weights(config, result);
    validate_bootstrap_uncertainty(config, result);
    validate_continuous_listening_area(config, result);
    validate_cea2034_source_plausibility(config, result);
    validate_bass_management(config, result);
    validate_role_targets(config, result);
}

/// Validate the bootstrap uncertainty block when
/// `multi_measurement.strategy == MinimaxUncertainty`.
fn validate_bootstrap_uncertainty(config: &RoomConfig, result: &mut ValidationResult) {
    let Some(mm) = config.optimizer.multi_measurement.as_ref() else {
        return;
    };
    if mm.strategy != MultiMeasurementStrategy::MinimaxUncertainty {
        // The block may still be set; we don't reject that — it just won't be
        // consulted unless the strategy switches. Keep validation focused on
        // the active path.
        return;
    }
    // The block is optional (it has a Default); when absent, we use defaults.
    if let Some(b) = mm.bootstrap_uncertainty.as_ref() {
        if b.num_resamples == 0 {
            result.add_error(
                "multi_measurement.bootstrap_uncertainty.num_resamples must be > 0".to_string(),
            );
        }
        if !(0.0..1.0).contains(&b.alpha) || b.alpha <= 0.0 {
            result.add_error(format!(
                "multi_measurement.bootstrap_uncertainty.alpha must be in (0, 1), got {}",
                b.alpha
            ));
        }
        if matches!(b.scalarisation, BootstrapScalarisation::Cvar)
            && (!(0.0..=1.0).contains(&b.cvar_alpha) || b.cvar_alpha <= 0.0)
        {
            result.add_error(format!(
                "multi_measurement.bootstrap_uncertainty.cvar_alpha must be in (0, 1] when \
                 scalarisation = cvar, got {}",
                b.cvar_alpha
            ));
        }
    }
}

/// Validate `multiseat.continuous_area` when the strategy is `ContinuousArea`.
fn validate_continuous_listening_area(config: &RoomConfig, result: &mut ValidationResult) {
    let Some(ms) = config.optimizer.multi_seat.as_ref() else {
        return;
    };
    if ms.strategy != MultiSeatStrategy::ContinuousArea {
        return;
    }
    let Some(area) = ms.continuous_area.as_ref() else {
        result.add_error(
            "multi_seat.strategy = continuous_area requires multi_seat.continuous_area to be set"
                .to_string(),
        );
        return;
    };
    if !(1..=3).contains(&area.dimensions) {
        result.add_error(format!(
            "multi_seat.continuous_area.dimensions must be 1, 2, or 3 (got {})",
            area.dimensions
        ));
    }
    if area.bounds.len() != area.dimensions {
        result.add_error(format!(
            "multi_seat.continuous_area.bounds length {} must equal dimensions {}",
            area.bounds.len(),
            area.dimensions
        ));
    }
    for (i, (lo, hi)) in area.bounds.iter().enumerate() {
        if !(lo.is_finite() && hi.is_finite()) || hi <= lo {
            result.add_error(format!(
                "multi_seat.continuous_area.bounds[{}] = ({}, {}) is degenerate",
                i, lo, hi
            ));
        }
    }
    if area.seat_positions.is_empty() {
        result.add_error(
            "multi_seat.continuous_area.seat_positions must contain at least one position"
                .to_string(),
        );
    } else {
        for (i, row) in area.seat_positions.iter().enumerate() {
            if row.len() != area.dimensions {
                result.add_error(format!(
                    "multi_seat.continuous_area.seat_positions[{}] has length {} (expected {})",
                    i,
                    row.len(),
                    area.dimensions
                ));
            }
        }
    }
    if !area.idw_power.is_finite() || area.idw_power <= 0.0 {
        result.add_error(format!(
            "multi_seat.continuous_area.idw_power must be > 0, got {}",
            area.idw_power
        ));
    }

    match &area.prior {
        AreaPriorKind::Uniform => {}
        AreaPriorKind::Gaussian {
            mean,
            cov_diag,
            truncation_sigmas,
        } => {
            if mean.len() != area.dimensions {
                result.add_error(format!(
                    "multi_seat.continuous_area.prior.gaussian.mean length {} must equal dimensions {}",
                    mean.len(),
                    area.dimensions
                ));
            }
            if cov_diag.len() != area.dimensions {
                result.add_error(format!(
                    "multi_seat.continuous_area.prior.gaussian.cov_diag length {} must equal dimensions {}",
                    cov_diag.len(),
                    area.dimensions
                ));
            }
            for (i, &v) in cov_diag.iter().enumerate() {
                if !v.is_finite() || v <= 0.0 {
                    result.add_error(format!(
                        "multi_seat.continuous_area.prior.gaussian.cov_diag[{}] must be > 0, got {}",
                        i, v
                    ));
                }
            }
            if !truncation_sigmas.is_finite() || *truncation_sigmas <= 0.0 {
                result.add_error(format!(
                    "multi_seat.continuous_area.prior.gaussian.truncation_sigmas must be > 0, got {}",
                    truncation_sigmas
                ));
            }
        }
    }

    match &area.quadrature {
        AreaQuadratureKind::Sobol { num_points, .. }
        | AreaQuadratureKind::LatinHypercube { num_points, .. } => {
            if *num_points == 0 {
                result.add_error(
                    "multi_seat.continuous_area.quadrature.num_points must be > 0".to_string(),
                );
            }
        }
        AreaQuadratureKind::GaussLegendre { points_per_axis } => {
            if *points_per_axis == 0 {
                result.add_error(
                    "multi_seat.continuous_area.quadrature.points_per_axis must be > 0".to_string(),
                );
            }
        }
    }

    match &area.scalarisation {
        AreaScalarisationKind::ExpectedValue => {}
        AreaScalarisationKind::WorstCase { inner_maxiter, .. } => {
            if *inner_maxiter == 0 {
                result.add_error(
                    "multi_seat.continuous_area.scalarisation.worst_case.inner_maxiter must be > 0"
                        .to_string(),
                );
            }
        }
        AreaScalarisationKind::Cvar { alpha } => {
            if !(0.0..=1.0).contains(alpha) || *alpha <= 0.0 {
                result.add_error(format!(
                    "multi_seat.continuous_area.scalarisation.cvar.alpha must be in (0, 1], got {}",
                    alpha
                ));
            }
        }
    }
}

fn validate_multi_measurement_weights(config: &RoomConfig, result: &mut ValidationResult) {
    let Some(mm) = config.optimizer.multi_measurement.as_ref() else {
        return;
    };
    let Some(weights) = mm.weights.as_ref() else {
        return;
    };

    for (channel, speaker) in &config.speakers {
        for source in collect_sources(speaker) {
            let count = match source {
                MeasurementSource::Multiple(m) => m.measurements.len(),
                MeasurementSource::InMemoryMultiple(curves) => curves.len(),
                _ => continue,
            };
            if count != weights.len() {
                result.add_error(format!(
                    "Channel '{}': multi_measurement.weights has {} entries but the channel \
                     has {} measurements. The lengths must match; `optimize_channel_eq_multi` \
                     would otherwise index out of bounds.",
                    channel,
                    weights.len(),
                    count,
                ));
            }
        }
    }
}

fn validate_cea2034_source_plausibility(config: &RoomConfig, result: &mut ValidationResult) {
    let enabled = config
        .optimizer
        .cea2034_correction
        .as_ref()
        .is_some_and(|c| c.enabled);
    if !enabled {
        return;
    }

    let any_plausible = config
        .speakers
        .values()
        .flat_map(collect_sources)
        .any(source_is_cea2034_shaped);

    if !any_plausible {
        result.add_warning(
            "cea2034_correction is enabled but no speaker looks like a CEA2034/spinorama \
             source (no speaker_name set, no path/name hint of 'cea2034' or 'spinorama'). \
             The 3-pass correction pipeline assumes spinorama-shaped data; applying it to \
             plain in-room responses will produce incorrect results. \
             Either disable cea2034_correction or provide a speaker_name so the pipeline \
             can fetch the matching spinorama data."
                .to_string(),
        );
    }
}

fn validate_bass_management(config: &RoomConfig, result: &mut ValidationResult) {
    let Some(system) = config.system.as_ref() else {
        return;
    };
    let Some(bm) = system.bass_management.as_ref() else {
        return;
    };
    if !bm.enabled {
        return;
    }

    if system.subwoofers.is_none() {
        result.add_warning(
            "bass_management is enabled but system.subwoofers is missing; bass management \
             will be reported as unavailable."
                .to_string(),
        );
    }
    if bm.lfe_playback_gain_db.abs() > 24.0 {
        result.add_error(format!(
            "bass_management.lfe_playback_gain_db ({}) is outside the safe +/-24 dB range",
            bm.lfe_playback_gain_db
        ));
    }
    if bm.max_sub_boost_db < 0.0 {
        result.add_error(format!(
            "bass_management.max_sub_boost_db ({}) must be non-negative",
            bm.max_sub_boost_db
        ));
    }
    if bm.headroom_margin_db < 0.0 {
        result.add_error(format!(
            "bass_management.headroom_margin_db ({}) must be non-negative",
            bm.headroom_margin_db
        ));
    }
    if bm.apply_lfe_gain_to_chain && bm.redirect_bass {
        result.add_warning(
            "bass_management.apply_lfe_gain_to_chain=true while redirect_bass=true. \
             The exported RoomEQ chain is per physical sub output, so this also boosts \
             redirected bass; leave it false unless downstream routing separates LFE."
                .to_string(),
        );
    }
}

fn validate_role_targets(config: &RoomConfig, result: &mut ValidationResult) {
    let Some(role_targets) = config
        .optimizer
        .target_response
        .as_ref()
        .and_then(|target| target.role_targets.as_ref())
    else {
        return;
    };
    if !role_targets.enabled {
        return;
    }

    if role_targets.center_dialog_low_hz <= 0.0
        || role_targets.center_dialog_high_hz <= role_targets.center_dialog_low_hz
    {
        result.add_error(format!(
            "target_response.role_targets center dialog band must be positive and ordered; got {}..{} Hz",
            role_targets.center_dialog_low_hz, role_targets.center_dialog_high_hz
        ));
    }
    if role_targets.cinema_x_curve_start_hz <= 0.0 {
        result.add_error(format!(
            "target_response.role_targets.cinema_x_curve_start_hz ({}) must be positive",
            role_targets.cinema_x_curve_start_hz
        ));
    }
    if let Some(distance_m) = role_targets.listening_distance_m
        && distance_m <= 0.0
    {
        result.add_error(format!(
            "target_response.role_targets.listening_distance_m ({distance_m}) must be positive"
        ));
    }
    if role_targets.cinema_reference_distance_m <= 0.0 {
        result.add_error(format!(
            "target_response.role_targets.cinema_reference_distance_m ({}) must be positive",
            role_targets.cinema_reference_distance_m
        ));
    }
}

#[cfg(test)]
mod validate_optimizer_tests {
    use super::*;
    use crate::roomeq::types::OptimizerConfig;

    fn default_config() -> OptimizerConfig {
        OptimizerConfig::default()
    }

    #[test]
    fn validate_normal_config_is_valid() {
        let mut result = ValidationResult::valid();
        let config = default_config();
        validate_optimizer_config(&config, &mut result);
        assert!(
            result.is_valid,
            "default config should be valid: {:?}",
            result.errors
        );
    }

    #[test]
    fn validate_num_filters_zero_warns() {
        let mut result = ValidationResult::valid();
        let mut config = default_config();
        config.num_filters = 0;
        validate_optimizer_config(&config, &mut result);
        assert!(result.is_valid, "zero filters is a warning, not an error");
        assert!(
            result
                .warnings
                .iter()
                .any(|w| w.contains("num_filters is 0"))
        );
    }

    #[test]
    fn validate_min_freq_gte_max_freq_errors() {
        let mut result = ValidationResult::valid();
        let mut config = default_config();
        config.min_freq = 200.0;
        config.max_freq = 100.0;
        validate_optimizer_config(&config, &mut result);
        assert!(!result.is_valid);
        assert!(
            result
                .errors
                .iter()
                .any(|e| e.contains("min_freq") && e.contains("max_freq"))
        );
    }

    #[test]
    fn validate_min_freq_non_positive_errors() {
        let mut result = ValidationResult::valid();
        let mut config = default_config();
        config.min_freq = 0.0;
        validate_optimizer_config(&config, &mut result);
        assert!(!result.is_valid);
        assert!(
            result
                .errors
                .iter()
                .any(|e| e.contains("min_freq") && e.contains("positive"))
        );
    }

    #[test]
    fn validate_max_freq_above_nyquist_warns() {
        let mut result = ValidationResult::valid();
        let mut config = default_config();
        config.max_freq = 30000.0;
        validate_optimizer_config(&config, &mut result);
        assert!(result.is_valid);
        assert!(result.warnings.iter().any(|w| w.contains("Nyquist")));
    }

    #[test]
    fn validate_min_q_gte_max_q_errors() {
        let mut result = ValidationResult::valid();
        let mut config = default_config();
        config.min_q = 5.0;
        config.max_q = 2.0;
        validate_optimizer_config(&config, &mut result);
        assert!(!result.is_valid);
        assert!(
            result
                .errors
                .iter()
                .any(|e| e.contains("min_q") && e.contains("max_q"))
        );
    }

    #[test]
    fn validate_min_q_non_positive_errors() {
        let mut result = ValidationResult::valid();
        let mut config = default_config();
        config.min_q = 0.0;
        validate_optimizer_config(&config, &mut result);
        assert!(!result.is_valid);
        assert!(
            result
                .errors
                .iter()
                .any(|e| e.contains("min_q") && e.contains("positive"))
        );
    }

    #[test]
    fn validate_smooth_n_out_of_range_errors() {
        let mut result = ValidationResult::valid();
        let mut config = default_config();
        config.smooth_n = 0;
        validate_optimizer_config(&config, &mut result);
        assert!(!result.is_valid);
        assert!(result.errors.iter().any(|e| e.contains("smooth_n")));

        let mut result2 = ValidationResult::valid();
        config.smooth_n = 50;
        validate_optimizer_config(&config, &mut result2);
        assert!(!result2.is_valid);
    }

    #[test]
    fn validate_unknown_algorithm_warns() {
        let mut result = ValidationResult::valid();
        let mut config = default_config();
        config.algorithm = "unknown:algo".to_string();
        validate_optimizer_config(&config, &mut result);
        assert!(result.is_valid);
        assert!(
            result
                .warnings
                .iter()
                .any(|w| w.contains("Unknown algorithm"))
        );
    }

    #[test]
    fn validate_unknown_loss_type_errors() {
        let mut result = ValidationResult::valid();
        let mut config = default_config();
        config.loss_type = "invalid".to_string();
        validate_optimizer_config(&config, &mut result);
        assert!(!result.is_valid);
        assert!(
            result
                .errors
                .iter()
                .any(|e| e.contains("Unknown loss_type"))
        );
    }

    #[test]
    fn validate_unknown_peq_model_warns() {
        let mut result = ValidationResult::valid();
        let mut config = default_config();
        config.peq_model = "unknown-model".to_string();
        validate_optimizer_config(&config, &mut result);
        assert!(result.is_valid);
        assert!(
            result
                .warnings
                .iter()
                .any(|w| w.contains("Unknown peq_model"))
        );
    }

    #[test]
    fn validate_max_iter_zero_warns() {
        let mut result = ValidationResult::valid();
        let mut config = default_config();
        config.max_iter = 0;
        validate_optimizer_config(&config, &mut result);
        assert!(result.is_valid);
        assert!(result.warnings.iter().any(|w| w.contains("max_iter is 0")));
    }

    #[test]
    fn validate_auto_optimizer_min_filters_gt_max_errors() {
        let mut result = ValidationResult::valid();
        let mut config = default_config();
        config.auto_optimizer = Some(crate::roomeq::types::AutoOptimizerConfig {
            enabled: true,
            min_filters: 5,
            max_filters: 3,
            ..Default::default()
        });
        validate_optimizer_config(&config, &mut result);
        assert!(!result.is_valid);
        assert!(
            result
                .errors
                .iter()
                .any(|e| e.contains("min_filters") && e.contains("max_filters"))
        );
    }

    #[test]
    fn validate_auto_optimizer_enabled_but_no_flags_warns() {
        let mut result = ValidationResult::valid();
        let mut config = default_config();
        config.auto_optimizer = Some(crate::roomeq::types::AutoOptimizerConfig {
            enabled: true,
            filter_count: false,
            q_bounds: false,
            gain_bounds: false,
            ..Default::default()
        });
        validate_optimizer_config(&config, &mut result);
        assert!(result.is_valid);
        assert!(
            result
                .warnings
                .iter()
                .any(|w| w.contains("all automatic selection flags are disabled"))
        );
    }

    #[test]
    fn validate_min_db_gte_max_db_errors() {
        let mut result = ValidationResult::valid();
        let mut config = default_config();
        config.min_db = 10.0;
        config.max_db = 5.0;
        validate_optimizer_config(&config, &mut result);
        assert!(!result.is_valid);
        assert!(
            result
                .errors
                .iter()
                .any(|e| e.contains("min_db") && e.contains("max_db"))
        );
    }

    #[test]
    fn validate_psychoacoustic_smoothing_zero_n_errors() {
        let mut result = ValidationResult::valid();
        let mut config = default_config();
        config.psychoacoustic_smoothing = Some(crate::read::PsychoacousticSmoothingConfig {
            low_freq_n: 0,
            high_freq_n: 1,
            ..Default::default()
        });
        validate_optimizer_config(&config, &mut result);
        assert!(!result.is_valid);
        assert!(
            result
                .errors
                .iter()
                .any(|e| e.contains("psychoacoustic_smoothing"))
        );
    }

    #[test]
    fn validate_asymmetric_loss_config_negative_weight_errors() {
        let mut result = ValidationResult::valid();
        let mut config = default_config();
        config.asymmetric_loss_config = Some(crate::loss::AsymmetricLossConfig {
            peak_weight: -1.0,
            ..Default::default()
        });
        validate_optimizer_config(&config, &mut result);
        assert!(!result.is_valid);
        assert!(
            result
                .errors
                .iter()
                .any(|e| e.contains("asymmetric_loss_config") && e.contains("weights"))
        );
    }

    #[test]
    fn validate_fir_taps_zero_errors() {
        let mut result = ValidationResult::valid();
        let mut config = default_config();
        config.fir = Some(crate::roomeq::types::FirConfig {
            taps: 0,
            phase: "linear".to_string(),
            ..Default::default()
        });
        validate_optimizer_config(&config, &mut result);
        assert!(!result.is_valid);
        assert!(result.errors.iter().any(|e| e.contains("FIR taps")));
    }

    #[test]
    fn validate_fir_phase_invalid_errors() {
        let mut result = ValidationResult::valid();
        let mut config = default_config();
        config.fir = Some(crate::roomeq::types::FirConfig {
            taps: 256,
            phase: "invalid".to_string(),
            ..Default::default()
        });
        validate_optimizer_config(&config, &mut result);
        assert!(!result.is_valid);
        assert!(
            result
                .errors
                .iter()
                .any(|e| e.contains("Unknown FIR phase"))
        );
    }

    #[test]
    fn validate_cea2034_correction_score_mode_errors() {
        let mut result = ValidationResult::valid();
        let mut config = default_config();
        config.cea2034_correction = Some(crate::roomeq::types::Cea2034CorrectionConfig {
            enabled: true,
            correction_mode: crate::roomeq::types::Cea2034CorrectionMode::Score,
            ..Default::default()
        });
        validate_optimizer_config(&config, &mut result);
        assert!(!result.is_valid);
        assert!(
            result
                .errors
                .iter()
                .any(|e| e.contains("score is not supported in roomeq"))
        );
    }
}

#[cfg(test)]
mod room_config_validation_tests {
    use super::validate_room_config;
    use crate::roomeq::types::{
        AreaPriorKind, AreaQuadratureKind, AreaScalarisationKind, BootstrapScalarisation,
        BootstrapUncertaintyConfig, CardioidConfig, ContinuousListeningAreaConfig, CrossoverConfig,
        DBAConfig, MultiSeatConfig, MultiSeatStrategy, MultiSubGroup, OptimizerConfig, RoomConfig,
        SpeakerConfig, SpeakerGroup, SupportingSourceGroup, SystemConfig, SystemModel,
        TargetResponseConfig, TargetShape,
    };
    use crate::{Curve, MeasurementRef, MeasurementSingle, MeasurementSource};
    use std::collections::HashMap;
    use std::path::PathBuf;

    fn default_room() -> RoomConfig {
        RoomConfig {
            version: crate::roomeq::types::default_config_version(),
            system: None,
            speakers: HashMap::new(),
            crossovers: None,
            target_curve: None,
            optimizer: OptimizerConfig::default(),
            recording_config: None,
            ctc: None,
            cea2034_cache: None,
        }
    }

    fn single_source(path: &str, speaker_name: Option<&str>) -> MeasurementSource {
        MeasurementSource::Single(MeasurementSingle {
            measurement: MeasurementRef::Path(PathBuf::from(path)),
            speaker_name: speaker_name.map(String::from),
        })
    }

    #[test]
    fn target_curve_and_shaped_target_response_warn() {
        let mut config = default_room();
        config.target_curve = Some(crate::roomeq::types::TargetCurveConfig::Predefined(
            "harman".to_string(),
        ));
        config.optimizer.target_response = Some(TargetResponseConfig {
            shape: TargetShape::Harman,
            slope_db_per_octave: 0.0,
            ..Default::default()
        });
        config.speakers.insert(
            "L".to_string(),
            SpeakerConfig::Single(single_source("l.csv", None)),
        );
        let result = validate_room_config(&config);
        assert!(result.is_valid);
        assert!(
            result
                .warnings
                .iter()
                .any(|w| w.contains("target_response takes precedence"))
        );
    }

    #[test]
    fn speaker_group_variants_produce_errors_and_warnings() {
        // Empty group
        let mut config = default_room();
        config.speakers.insert(
            "L".to_string(),
            SpeakerConfig::Group(SpeakerGroup {
                name: "L".to_string(),
                speaker_name: None,
                measurements: vec![],
                crossover: None,
            }),
        );
        let result = validate_room_config(&config);
        assert!(
            result
                .errors
                .iter()
                .any(|e| e.contains("has no measurements"))
        );

        // Single measurement group warns
        let mut config = default_room();
        config.speakers.insert(
            "L".to_string(),
            SpeakerConfig::Group(SpeakerGroup {
                name: "L".to_string(),
                speaker_name: None,
                measurements: vec![single_source("m.csv", None)],
                crossover: None,
            }),
        );
        let result = validate_room_config(&config);
        assert!(
            result
                .warnings
                .iter()
                .any(|w| w.contains("has only 1 measurement"))
        );

        // Two measurements without crossover error
        let mut config = default_room();
        config.speakers.insert(
            "L".to_string(),
            SpeakerConfig::Group(SpeakerGroup {
                name: "L".to_string(),
                speaker_name: None,
                measurements: vec![single_source("w.csv", None), single_source("t.csv", None)],
                crossover: None,
            }),
        );
        let result = validate_room_config(&config);
        assert!(
            result
                .errors
                .iter()
                .any(|e| e.contains("has multiple drivers but no crossover specified"))
        );

        // Multi-sub empty / single
        let mut config = default_room();
        config.speakers.insert(
            "Sub".to_string(),
            SpeakerConfig::MultiSub(MultiSubGroup {
                name: "Sub".to_string(),
                speaker_name: None,
                subwoofers: vec![],
                allpass_optimization: false,
            }),
        );
        let result = validate_room_config(&config);
        assert!(
            result
                .errors
                .iter()
                .any(|e| e.contains("has no subwoofers"))
        );

        let mut config = default_room();
        config.speakers.insert(
            "Sub".to_string(),
            SpeakerConfig::MultiSub(MultiSubGroup {
                name: "Sub".to_string(),
                speaker_name: None,
                subwoofers: vec![single_source("s.csv", None)],
                allpass_optimization: false,
            }),
        );
        let result = validate_room_config(&config);
        assert!(
            result
                .warnings
                .iter()
                .any(|w| w.contains("only 1 subwoofer"))
        );

        // DBA empty front/rear
        let mut config = default_room();
        config.speakers.insert(
            "DBA".to_string(),
            SpeakerConfig::Dba(DBAConfig {
                name: "DBA".to_string(),
                speaker_name: None,
                front: vec![],
                rear: vec![],
            }),
        );
        let result = validate_room_config(&config);
        assert!(
            result
                .errors
                .iter()
                .any(|e| e.contains("no front speakers"))
        );
        assert!(result.errors.iter().any(|e| e.contains("no rear speakers")));

        // Cardioid invalid separation
        let mut config = default_room();
        config.speakers.insert(
            "Card".to_string(),
            SpeakerConfig::Cardioid(Box::new(CardioidConfig {
                name: "Card".to_string(),
                speaker_name: None,
                front: single_source("f.csv", None),
                rear: single_source("r.csv", None),
                separation_meters: 0.0,
            })),
        );
        let result = validate_room_config(&config);
        assert!(
            result
                .errors
                .iter()
                .any(|e| e.contains("separation") && e.contains("must be > 0"))
        );

        // Supporting source same speaker_name warning
        let mut config = default_room();
        config.speakers.insert(
            "Support".to_string(),
            SpeakerConfig::SupportingSource(SupportingSourceGroup {
                name: "Support".to_string(),
                speaker_name: None,
                primary: single_source("p.csv", Some("Same")),
                support: single_source("s.csv", Some("Same")),
                supporting_source: Default::default(),
            }),
        );
        let result = validate_room_config(&config);
        assert!(
            result
                .warnings
                .iter()
                .any(|w| w.contains("same speaker_name for primary and support"))
        );

        // Supporting source invalid delay
        let mut config = default_room();
        config.speakers.insert(
            "Support".to_string(),
            SpeakerConfig::SupportingSource(SupportingSourceGroup {
                name: "Support".to_string(),
                speaker_name: None,
                primary: single_source("p.csv", None),
                support: single_source("s.csv", None),
                supporting_source: crate::roomeq::types::SupportingSourceConfig {
                    delay_ms: 100.0,
                    ..Default::default()
                },
            }),
        );
        let result = validate_room_config(&config);
        assert!(result.errors.iter().any(|e| e.contains("delay_ms")));

        // Supporting source invalid tap count
        let mut config = default_room();
        config.speakers.insert(
            "Support".to_string(),
            SpeakerConfig::SupportingSource(SupportingSourceGroup {
                name: "Support".to_string(),
                speaker_name: None,
                primary: single_source("p.csv", None),
                support: single_source("s.csv", None),
                supporting_source: crate::roomeq::types::SupportingSourceConfig {
                    fir_taps: 3,
                    ..Default::default()
                },
            }),
        );
        let result = validate_room_config(&config);
        assert!(result.errors.iter().any(|e| e.contains("fir_taps")));
    }

    #[test]
    fn crossover_reference_validation() {
        // Missing crossovers map
        let mut config = default_room();
        config.speakers.insert(
            "L".to_string(),
            SpeakerConfig::Group(SpeakerGroup {
                name: "L".to_string(),
                speaker_name: None,
                measurements: vec![single_source("w.csv", None), single_source("t.csv", None)],
                crossover: Some("xo".to_string()),
            }),
        );
        let result = validate_room_config(&config);
        assert!(
            result
                .errors
                .iter()
                .any(|e| e.contains("no crossovers defined"))
        );

        // Non-existent crossover
        let mut config = default_room();
        config.speakers.insert(
            "L".to_string(),
            SpeakerConfig::Group(SpeakerGroup {
                name: "L".to_string(),
                speaker_name: None,
                measurements: vec![single_source("w.csv", None), single_source("t.csv", None)],
                crossover: Some("xo".to_string()),
            }),
        );
        config.crossovers = Some(HashMap::new());
        let result = validate_room_config(&config);
        assert!(
            result
                .errors
                .iter()
                .any(|e| e.contains("non-existent crossover"))
        );

        // Single frequency with 3 drivers -> warning
        let mut config = default_room();
        config.speakers.insert(
            "L".to_string(),
            SpeakerConfig::Group(SpeakerGroup {
                name: "L".to_string(),
                speaker_name: None,
                measurements: vec![
                    single_source("w.csv", None),
                    single_source("m.csv", None),
                    single_source("t.csv", None),
                ],
                crossover: Some("xo".to_string()),
            }),
        );
        config.crossovers = Some(HashMap::from([(
            "xo".to_string(),
            CrossoverConfig {
                crossover_type: "LR24".to_string(),
                frequency: Some(1000.0),
                frequencies: None,
                frequency_range: None,
            },
        )]));
        let result = validate_room_config(&config);
        assert!(
            result
                .warnings
                .iter()
                .any(|w| w.contains("single frequency but speaker") && w.contains("has 3 drivers"))
        );

        // Wrong number of frequencies for 3 drivers (need 2)
        let mut config = default_room();
        config.speakers.insert(
            "L".to_string(),
            SpeakerConfig::Group(SpeakerGroup {
                name: "L".to_string(),
                speaker_name: None,
                measurements: vec![
                    single_source("w.csv", None),
                    single_source("m.csv", None),
                    single_source("t.csv", None),
                ],
                crossover: Some("xo".to_string()),
            }),
        );
        config.crossovers = Some(HashMap::from([(
            "xo".to_string(),
            CrossoverConfig {
                crossover_type: "LR24".to_string(),
                frequency: None,
                frequencies: Some(vec![1000.0]),
                frequency_range: None,
            },
        )]));
        let result = validate_room_config(&config);
        assert!(
            result
                .errors
                .iter()
                .any(|e| e.contains("needs 2 for 3 drivers"))
        );

        // Auto-optimized crossover warns
        let mut config = default_room();
        config.speakers.insert(
            "L".to_string(),
            SpeakerConfig::Group(SpeakerGroup {
                name: "L".to_string(),
                speaker_name: None,
                measurements: vec![single_source("w.csv", None), single_source("t.csv", None)],
                crossover: Some("xo".to_string()),
            }),
        );
        config.crossovers = Some(HashMap::from([(
            "xo".to_string(),
            CrossoverConfig {
                crossover_type: "LR24".to_string(),
                frequency: None,
                frequencies: None,
                frequency_range: None,
            },
        )]));
        let result = validate_room_config(&config);
        assert!(
            result
                .warnings
                .iter()
                .any(|w| w.contains("will be auto-optimized"))
        );
    }

    #[test]
    fn multi_measurement_weight_length_mismatch() {
        let mut config = default_room();
        config.speakers.insert(
            "L".to_string(),
            SpeakerConfig::Single(MeasurementSource::InMemoryMultiple(vec![
                Curve {
                    freq: ndarray::array![20.0, 100.0],
                    spl: ndarray::array![0.0, 0.0],
                    phase: None,
                    ..Default::default()
                },
                Curve {
                    freq: ndarray::array![20.0, 100.0],
                    spl: ndarray::array![0.0, 0.0],
                    phase: None,
                    ..Default::default()
                },
                Curve {
                    freq: ndarray::array![20.0, 100.0],
                    spl: ndarray::array![0.0, 0.0],
                    phase: None,
                    ..Default::default()
                },
            ])),
        );
        config.optimizer.multi_measurement = Some(crate::roomeq::types::MultiMeasurementConfig {
            weights: Some(vec![0.5, 0.5]),
            ..Default::default()
        });
        let result = validate_room_config(&config);
        assert!(
            result
                .errors
                .iter()
                .any(|e| e.contains("multi_measurement.weights has 2 entries")
                    && e.contains("has 3 measurements"))
        );
    }

    #[test]
    fn bootstrap_uncertainty_validation_errors() {
        let mut config = default_room();
        config.speakers.insert(
            "L".to_string(),
            SpeakerConfig::Single(single_source("l.csv", None)),
        );
        config.optimizer.multi_measurement = Some(crate::roomeq::types::MultiMeasurementConfig {
            strategy: crate::roomeq::types::MultiMeasurementStrategy::MinimaxUncertainty,
            bootstrap_uncertainty: Some(BootstrapUncertaintyConfig {
                num_resamples: 0,
                alpha: 0.0,
                scalarisation: BootstrapScalarisation::Cvar,
                cvar_alpha: 0.0,
                ..Default::default()
            }),
            ..Default::default()
        });
        let result = validate_room_config(&config);
        assert!(
            result
                .errors
                .iter()
                .any(|e| e.contains("num_resamples must be > 0"))
        );
        assert!(
            result
                .errors
                .iter()
                .any(|e| e.contains("alpha must be in (0, 1)") && e.contains("got 0"))
        );
        assert!(
            result
                .errors
                .iter()
                .any(|e| e.contains("cvar_alpha must be in (0, 1]") && e.contains("got 0"))
        );
    }

    #[test]
    fn continuous_listening_area_validation_errors() {
        let make_config = |area: ContinuousListeningAreaConfig| {
            let mut config = default_room();
            config.speakers.insert(
                "L".to_string(),
                SpeakerConfig::Single(single_source("l.csv", None)),
            );
            config.optimizer.multi_seat = Some(MultiSeatConfig {
                enabled: true,
                strategy: MultiSeatStrategy::ContinuousArea,
                continuous_area: Some(area),
                ..Default::default()
            });
            config
        };

        // Missing continuous_area
        let mut config = default_room();
        config.speakers.insert(
            "L".to_string(),
            SpeakerConfig::Single(single_source("l.csv", None)),
        );
        config.optimizer.multi_seat = Some(MultiSeatConfig {
            enabled: true,
            strategy: MultiSeatStrategy::ContinuousArea,
            continuous_area: None,
            ..Default::default()
        });
        let result = validate_room_config(&config);
        assert!(
            result
                .errors
                .iter()
                .any(|e| e.contains("continuous_area to be set"))
        );

        // Bad dimensions
        let result = validate_room_config(&make_config(ContinuousListeningAreaConfig {
            dimensions: 4,
            bounds: vec![(0.0, 1.0)],
            seat_positions: vec![vec![0.5]],
            prior: AreaPriorKind::Uniform,
            quadrature: AreaQuadratureKind::Sobol {
                num_points: 1,
                seed: 0,
            },
            scalarisation: AreaScalarisationKind::ExpectedValue,
            idw_power: 2.0,
        }));
        assert!(
            result
                .errors
                .iter()
                .any(|e| e.contains("dimensions must be 1, 2, or 3"))
        );

        // Bounds/seats mismatch
        let result = validate_room_config(&make_config(ContinuousListeningAreaConfig {
            dimensions: 2,
            bounds: vec![(0.0, 1.0)],
            seat_positions: vec![vec![0.5, 0.5]],
            prior: AreaPriorKind::Uniform,
            quadrature: AreaQuadratureKind::Sobol {
                num_points: 1,
                seed: 0,
            },
            scalarisation: AreaScalarisationKind::ExpectedValue,
            idw_power: 2.0,
        }));
        assert!(
            result
                .errors
                .iter()
                .any(|e| e.contains("bounds length 1 must equal dimensions 2"))
        );

        // Degenerate bounds
        let result = validate_room_config(&make_config(ContinuousListeningAreaConfig {
            dimensions: 1,
            bounds: vec![(1.0, 1.0)],
            seat_positions: vec![vec![0.5]],
            prior: AreaPriorKind::Uniform,
            quadrature: AreaQuadratureKind::Sobol {
                num_points: 1,
                seed: 0,
            },
            scalarisation: AreaScalarisationKind::ExpectedValue,
            idw_power: 2.0,
        }));
        assert!(
            result
                .errors
                .iter()
                .any(|e| e.contains("bounds[0] = (1, 1) is degenerate"))
        );

        // No seat positions
        let result = validate_room_config(&make_config(ContinuousListeningAreaConfig {
            dimensions: 1,
            bounds: vec![(0.0, 1.0)],
            seat_positions: vec![],
            prior: AreaPriorKind::Uniform,
            quadrature: AreaQuadratureKind::Sobol {
                num_points: 1,
                seed: 0,
            },
            scalarisation: AreaScalarisationKind::ExpectedValue,
            idw_power: 2.0,
        }));
        assert!(
            result
                .errors
                .iter()
                .any(|e| e.contains("seat_positions must contain at least one position"))
        );

        // Seat row length mismatch
        let result = validate_room_config(&make_config(ContinuousListeningAreaConfig {
            dimensions: 1,
            bounds: vec![(0.0, 1.0)],
            seat_positions: vec![vec![0.5, 0.5]],
            prior: AreaPriorKind::Uniform,
            quadrature: AreaQuadratureKind::Sobol {
                num_points: 1,
                seed: 0,
            },
            scalarisation: AreaScalarisationKind::ExpectedValue,
            idw_power: 2.0,
        }));
        assert!(
            result
                .errors
                .iter()
                .any(|e| e.contains("seat_positions[0] has length 2 (expected 1)"))
        );

        // Non-positive idw_power
        let result = validate_room_config(&make_config(ContinuousListeningAreaConfig {
            dimensions: 1,
            bounds: vec![(0.0, 1.0)],
            seat_positions: vec![vec![0.5]],
            prior: AreaPriorKind::Uniform,
            quadrature: AreaQuadratureKind::Sobol {
                num_points: 1,
                seed: 0,
            },
            scalarisation: AreaScalarisationKind::ExpectedValue,
            idw_power: 0.0,
        }));
        assert!(
            result
                .errors
                .iter()
                .any(|e| e.contains("idw_power must be > 0"))
        );

        // Gaussian prior mismatches / bad cov_diag
        let result = validate_room_config(&make_config(ContinuousListeningAreaConfig {
            dimensions: 1,
            bounds: vec![(0.0, 1.0)],
            seat_positions: vec![vec![0.5]],
            prior: AreaPriorKind::Gaussian {
                mean: vec![0.5, 0.5],
                cov_diag: vec![0.1],
                truncation_sigmas: 4.0,
            },
            quadrature: AreaQuadratureKind::Sobol {
                num_points: 1,
                seed: 0,
            },
            scalarisation: AreaScalarisationKind::ExpectedValue,
            idw_power: 2.0,
        }));
        assert!(
            result
                .errors
                .iter()
                .any(|e| e.contains("prior.gaussian.mean length 2 must equal dimensions 1"))
        );

        let result = validate_room_config(&make_config(ContinuousListeningAreaConfig {
            dimensions: 1,
            bounds: vec![(0.0, 1.0)],
            seat_positions: vec![vec![0.5]],
            prior: AreaPriorKind::Gaussian {
                mean: vec![0.5],
                cov_diag: vec![-0.1],
                truncation_sigmas: 4.0,
            },
            quadrature: AreaQuadratureKind::Sobol {
                num_points: 1,
                seed: 0,
            },
            scalarisation: AreaScalarisationKind::ExpectedValue,
            idw_power: 2.0,
        }));
        assert!(
            result
                .errors
                .iter()
                .any(|e| e.contains("prior.gaussian.cov_diag[0] must be > 0"))
        );

        let result = validate_room_config(&make_config(ContinuousListeningAreaConfig {
            dimensions: 1,
            bounds: vec![(0.0, 1.0)],
            seat_positions: vec![vec![0.5]],
            prior: AreaPriorKind::Gaussian {
                mean: vec![0.5],
                cov_diag: vec![0.1],
                truncation_sigmas: 0.0,
            },
            quadrature: AreaQuadratureKind::Sobol {
                num_points: 1,
                seed: 0,
            },
            scalarisation: AreaScalarisationKind::ExpectedValue,
            idw_power: 2.0,
        }));
        assert!(
            result
                .errors
                .iter()
                .any(|e| e.contains("prior.gaussian.truncation_sigmas must be > 0"))
        );

        // Quadrature num_points == 0
        let result = validate_room_config(&make_config(ContinuousListeningAreaConfig {
            dimensions: 1,
            bounds: vec![(0.0, 1.0)],
            seat_positions: vec![vec![0.5]],
            prior: AreaPriorKind::Uniform,
            quadrature: AreaQuadratureKind::Sobol {
                num_points: 0,
                seed: 0,
            },
            scalarisation: AreaScalarisationKind::ExpectedValue,
            idw_power: 2.0,
        }));
        assert!(
            result
                .errors
                .iter()
                .any(|e| e.contains("quadrature.num_points must be > 0"))
        );

        let result = validate_room_config(&make_config(ContinuousListeningAreaConfig {
            dimensions: 1,
            bounds: vec![(0.0, 1.0)],
            seat_positions: vec![vec![0.5]],
            prior: AreaPriorKind::Uniform,
            quadrature: AreaQuadratureKind::GaussLegendre { points_per_axis: 0 },
            scalarisation: AreaScalarisationKind::ExpectedValue,
            idw_power: 2.0,
        }));
        assert!(
            result
                .errors
                .iter()
                .any(|e| e.contains("quadrature.points_per_axis must be > 0"))
        );

        // Worst-case inner_maxiter == 0
        let result = validate_room_config(&make_config(ContinuousListeningAreaConfig {
            dimensions: 1,
            bounds: vec![(0.0, 1.0)],
            seat_positions: vec![vec![0.5]],
            prior: AreaPriorKind::Uniform,
            quadrature: AreaQuadratureKind::Sobol {
                num_points: 1,
                seed: 0,
            },
            scalarisation: AreaScalarisationKind::WorstCase {
                inner_maxiter: 0,
                inner_seed: 0,
            },
            idw_power: 2.0,
        }));
        assert!(
            result
                .errors
                .iter()
                .any(|e| e.contains("worst_case.inner_maxiter must be > 0"))
        );

        // CVaR alpha out of range
        let result = validate_room_config(&make_config(ContinuousListeningAreaConfig {
            dimensions: 1,
            bounds: vec![(0.0, 1.0)],
            seat_positions: vec![vec![0.5]],
            prior: AreaPriorKind::Uniform,
            quadrature: AreaQuadratureKind::Sobol {
                num_points: 1,
                seed: 0,
            },
            scalarisation: AreaScalarisationKind::Cvar { alpha: 0.0 },
            idw_power: 2.0,
        }));
        assert!(
            result
                .errors
                .iter()
                .any(|e| e.contains("scalarisation.cvar.alpha must be in (0, 1]"))
        );
    }

    #[test]
    fn cea2034_correction_plausibility_warning() {
        let mut config = default_room();
        config.speakers.insert(
            "L".to_string(),
            SpeakerConfig::Single(single_source("inroom.csv", None)),
        );
        config.optimizer.cea2034_correction = Some(crate::roomeq::types::Cea2034CorrectionConfig {
            enabled: true,
            ..Default::default()
        });
        let result = validate_room_config(&config);
        assert!(result.warnings.iter().any(|w| w.contains(
            "cea2034_correction is enabled but no speaker looks like a CEA2034/spinorama"
        )));
    }

    #[test]
    fn bass_management_validation() {
        let mut config = default_room();
        config.speakers.insert(
            "Sub".to_string(),
            SpeakerConfig::Single(single_source("sub.csv", None)),
        );
        config.system = Some(SystemConfig {
            model: SystemModel::HomeCinema,
            speakers: HashMap::new(),
            subwoofers: None,
            bass_management: Some(crate::roomeq::types::BassManagementConfig {
                enabled: true,
                lfe_playback_gain_db: 30.0,
                max_sub_boost_db: -1.0,
                headroom_margin_db: -2.0,
                apply_lfe_gain_to_chain: true,
                redirect_bass: true,
                ..Default::default()
            }),
            supporting_source_outputs: None,
        });
        let result = validate_room_config(&config);
        assert!(
            result
                .warnings
                .iter()
                .any(|w| w.contains("bass_management is enabled but system.subwoofers is missing"))
        );
        assert!(
            result
                .warnings
                .iter()
                .any(|w| w.contains("apply_lfe_gain_to_chain=true while redirect_bass=true"))
        );
        assert!(
            result
                .errors
                .iter()
                .any(|e| e.contains("lfe_playback_gain_db") && e.contains("outside the safe"))
        );
        assert!(
            result
                .errors
                .iter()
                .any(|e| e.contains("max_sub_boost_db") && e.contains("must be non-negative"))
        );
        assert!(
            result
                .errors
                .iter()
                .any(|e| e.contains("headroom_margin_db") && e.contains("must be non-negative"))
        );
    }

    #[test]
    fn role_targets_validation() {
        let mut config = default_room();
        config.speakers.insert(
            "C".to_string(),
            SpeakerConfig::Single(single_source("c.csv", None)),
        );
        config.optimizer.target_response = Some(TargetResponseConfig {
            role_targets: Some(crate::roomeq::types::RoleTargetConfig {
                enabled: true,
                center_dialog_low_hz: 500.0,
                center_dialog_high_hz: 200.0,
                cinema_x_curve_start_hz: 0.0,
                listening_distance_m: Some(-1.0),
                cinema_reference_distance_m: 0.0,
                ..Default::default()
            }),
            ..Default::default()
        });
        let result = validate_room_config(&config);
        assert!(
            result
                .errors
                .iter()
                .any(|e| e.contains("center dialog band"))
        );
        assert!(
            result
                .errors
                .iter()
                .any(|e| e.contains("cinema_x_curve_start_hz") && e.contains("must be positive"))
        );
        assert!(
            result
                .errors
                .iter()
                .any(|e| e.contains("listening_distance_m") && e.contains("must be positive"))
        );
        assert!(
            result.errors.iter().any(
                |e| e.contains("cinema_reference_distance_m") && e.contains("must be positive")
            )
        );
    }

    #[test]
    fn audit_optimizer_numeric_bounds_must_be_finite() {
        let mut config = default_room();
        config.optimizer.min_freq = f64::NAN;
        config.optimizer.max_q = f64::INFINITY;
        let result = validate_room_config(&config);

        assert!(
            result
                .errors
                .iter()
                .any(|e| e.contains("min_freq") && e.contains("finite"))
        );
        assert!(
            result
                .errors
                .iter()
                .any(|e| e.contains("max_q") && e.contains("finite"))
        );
    }

    #[test]
    fn audit_boost_envelope_must_be_finite_positive_and_strictly_sorted() {
        let mut config = default_room();
        config.optimizer.max_boost_envelope = Some(vec![(20.0, 6.0), (200.0, 3.0), (200.0, 2.0)]);
        let result = validate_room_config(&config);

        assert!(
            result
                .errors
                .iter()
                .any(|e| e.contains("max_boost_envelope") && e.contains("strictly increasing"))
        );
    }

    #[test]
    fn audit_system_speaker_mapping_must_reference_existing_measurement() {
        let mut config = default_room();
        config.speakers.insert(
            "LeftMeasurement".to_string(),
            SpeakerConfig::Single(single_source("left.csv", None)),
        );
        config.system = Some(SystemConfig {
            model: SystemModel::Stereo,
            speakers: HashMap::from([("L".to_string(), "Typo".to_string())]),
            subwoofers: None,
            bass_management: None,
            supporting_source_outputs: None,
        });
        let result = validate_room_config(&config);

        assert!(
            result
                .errors
                .iter()
                .any(|e| e.contains("system.speakers") && e.contains("Typo"))
        );
    }

    #[test]
    fn audit_crossover_type_is_validated_at_config_load() {
        let mut config = default_room();
        config.crossovers = Some(HashMap::from([(
            "main".to_string(),
            CrossoverConfig {
                crossover_type: "not-a-crossover".to_string(),
                frequency: Some(80.0),
                frequencies: None,
                frequency_range: None,
            },
        )]));
        let result = validate_room_config(&config);

        assert!(
            result
                .errors
                .iter()
                .any(|e| e.contains("Crossover 'main'") && e.contains("unsupported type"))
        );
    }
}
