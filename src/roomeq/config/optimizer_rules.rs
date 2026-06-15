// Independent validation rules for `OptimizerConfig`.
//
// Each rule is a pure function of the config that pushes errors/warnings into a
// shared `ValidationContext`.  Rules are individually unit-testable and are
// orchestrated by `validate_optimizer_config` in `validate.rs`.

use super::misc::PHASE_LINEAR_RECOMMENDED_MAX_FREQ_HZ;
use super::validation_result::ValidationContext;
use crate::roomeq::types::{Cea2034CorrectionMode, ProcessingMode};

pub fn rule_num_filters(ctx: &mut ValidationContext<'_>) {
    if ctx.opt.num_filters == 0 {
        ctx.add_warning("num_filters is 0, no EQ will be applied");
    }
}

pub fn rule_freq_range(ctx: &mut ValidationContext<'_>) {
    if ctx.opt.min_freq >= ctx.opt.max_freq {
        ctx.add_error(format!(
            "min_freq ({}) must be less than max_freq ({})",
            ctx.opt.min_freq, ctx.opt.max_freq
        ));
    }
    if ctx.opt.min_freq <= 0.0 {
        ctx.add_error(format!("min_freq ({}) must be positive", ctx.opt.min_freq));
    }
    if ctx.opt.max_freq > 24000.0 {
        ctx.add_warning(format!(
            "max_freq ({}) is above Nyquist for 48kHz sample rate",
            ctx.opt.max_freq
        ));
    }
}

pub fn rule_q_range(ctx: &mut ValidationContext<'_>) {
    if ctx.opt.min_q > ctx.opt.max_q {
        ctx.add_error(format!(
            "min_q ({}) must be less than or equal to max_q ({})",
            ctx.opt.min_q, ctx.opt.max_q
        ));
    }
    if ctx.opt.min_q <= 0.0 {
        ctx.add_error(format!("min_q ({}) must be positive", ctx.opt.min_q));
    }
}

pub fn rule_smooth_n(ctx: &mut ValidationContext<'_>) {
    if !(1..=48).contains(&ctx.opt.smooth_n) {
        ctx.add_error(format!(
            "smooth_n ({}) must be in range [1..48]",
            ctx.opt.smooth_n
        ));
    }
}

pub fn rule_psychoacoustic_smoothing(ctx: &mut ValidationContext<'_>) {
    let Some(smoothing) = ctx.opt.psychoacoustic_smoothing else {
        return;
    };
    if smoothing.low_freq_n == 0 || smoothing.high_freq_n == 0 {
        ctx.add_error("psychoacoustic_smoothing low_freq_n/high_freq_n must be at least 1");
    }
    if smoothing.low_freq <= 0.0 || smoothing.high_freq <= smoothing.low_freq {
        ctx.add_error(format!(
            "psychoacoustic_smoothing requires 0 < low_freq < high_freq (got {:.1}..{:.1})",
            smoothing.low_freq, smoothing.high_freq
        ));
    }
}

pub fn rule_asymmetric_loss(ctx: &mut ValidationContext<'_>) {
    let Some(asym) = ctx.opt.asymmetric_loss_config else {
        return;
    };
    if asym.transition_freq <= 0.0 {
        ctx.add_error(format!(
            "asymmetric_loss_config.transition_freq ({}) must be positive",
            asym.transition_freq
        ));
    }
    if asym.peak_weight < 0.0
        || asym.dip_weight < 0.0
        || asym.bass_peak_weight < 0.0
        || asym.bass_dip_weight < 0.0
    {
        ctx.add_error("asymmetric_loss_config weights must be non-negative");
    }
}

pub fn rule_audibility_deadband(ctx: &mut ValidationContext<'_>) {
    let Some(deadband) = ctx.opt.audibility_deadband else {
        return;
    };
    if deadband.bass_db < 0.0 || deadband.mid_db < 0.0 || deadband.treble_db < 0.0 {
        ctx.add_error("audibility_deadband thresholds must be non-negative");
    }
    if deadband.bass_mid_hz <= 0.0 || deadband.mid_treble_hz <= deadband.bass_mid_hz {
        ctx.add_error(format!(
            "audibility_deadband requires 0 < bass_mid_hz < mid_treble_hz (got {:.1}..{:.1})",
            deadband.bass_mid_hz, deadband.mid_treble_hz
        ));
    }
    if deadband.schroeder_hz <= 0.0 {
        ctx.add_error(format!(
            "audibility_deadband.schroeder_hz ({}) must be positive",
            deadband.schroeder_hz
        ));
    }
}

pub fn rule_high_frequency_correction(ctx: &mut ValidationContext<'_>) {
    let Some(hf) = ctx.opt.high_frequency_correction else {
        return;
    };
    if hf.start_hz <= 0.0 {
        ctx.add_error(format!(
            "high_frequency_correction.start_hz ({}) must be positive",
            hf.start_hz
        ));
    }
    if hf.extra_deadband_db < 0.0 {
        ctx.add_error(format!(
            "high_frequency_correction.extra_deadband_db ({}) must be non-negative",
            hf.extra_deadband_db
        ));
    }
    if hf.smoothing_n == 0 {
        ctx.add_error("high_frequency_correction.smoothing_n must be at least 1");
    }
    if hf.max_q <= 0.0 {
        ctx.add_error(format!(
            "high_frequency_correction.max_q ({}) must be positive",
            hf.max_q
        ));
    }
}

pub fn rule_early_late_correction(ctx: &mut ValidationContext<'_>) {
    let Some(early_late) = ctx.opt.early_late_correction else {
        return;
    };
    if early_late.direct_window_ms <= 0.0
        || early_late.early_window_ms <= early_late.direct_window_ms
        || early_late.late_window_ms <= early_late.early_window_ms
    {
        ctx.add_error(format!(
            "early_late_correction windows must satisfy 0 < direct < early < late (got {:.1}, {:.1}, {:.1} ms)",
            early_late.direct_window_ms,
            early_late.early_window_ms,
            early_late.late_window_ms
        ));
    }
    if !early_late.early_cue_risk_db.is_finite() {
        ctx.add_error("early_late_correction.early_cue_risk_db must be finite");
    }
}

pub fn rule_validation_bundle(ctx: &mut ValidationContext<'_>) {
    if let Some(bundle) = ctx.opt.validation_bundle
        && !bundle.target_lufs.is_finite()
    {
        ctx.add_error("validation_bundle.target_lufs must be finite");
    }
}

pub fn rule_gain_bounds(ctx: &mut ValidationContext<'_>) {
    if ctx.opt.min_db > ctx.opt.max_db {
        ctx.add_error(format!(
            "min_db ({}) must be less than or equal to max_db ({})",
            ctx.opt.min_db, ctx.opt.max_db
        ));
    }
}

pub fn rule_excursion_protection(ctx: &mut ValidationContext<'_>) {
    if let Some(excursion) = &ctx.opt.excursion_protection
        && (excursion.f3_reference_min_hz <= 0.0
            || excursion.f3_reference_max_hz <= excursion.f3_reference_min_hz)
    {
        ctx.add_error(format!(
            "excursion_protection F3 reference band must satisfy 0 < min < max (got {:.1}..{:.1})",
            excursion.f3_reference_min_hz, excursion.f3_reference_max_hz
        ));
    }
}

pub fn rule_auto_optimizer(ctx: &mut ValidationContext<'_>) {
    let Some(auto) = &ctx.opt.auto_optimizer else {
        return;
    };
    if !auto.enabled {
        return;
    }

    if auto.min_filters == 0 {
        ctx.add_error("auto_optimizer.min_filters must be at least 1");
    }
    if auto.max_filters == 0 {
        ctx.add_error("auto_optimizer.max_filters must be at least 1");
    }
    if auto.min_filters > auto.max_filters {
        ctx.add_error(format!(
            "auto_optimizer.min_filters ({}) must be <= max_filters ({})",
            auto.min_filters, auto.max_filters
        ));
    }
    if !auto.filter_count && !auto.q_bounds && !auto.gain_bounds {
        ctx.add_warning("auto_optimizer is enabled but all automatic selection flags are disabled");
    }
    if auto.filter_count
        && auto.max_filters < 2
        && ctx
            .opt
            .schroeder_split
            .as_ref()
            .is_some_and(|split| split.enabled)
    {
        ctx.add_error(
            "auto_optimizer.max_filters must be at least 2 when schroeder_split is enabled",
        );
    }
}

pub fn rule_multi_seat(ctx: &mut ValidationContext<'_>) {
    let Some(multi_seat) = &ctx.opt.multi_seat else {
        return;
    };
    if !multi_seat.enabled {
        return;
    }

    if multi_seat.max_deviation_db < 0.0 {
        ctx.add_error(format!(
            "multi_seat.max_deviation_db ({}) must be non-negative",
            multi_seat.max_deviation_db
        ));
    }
    if !multi_seat.primary_seat_weight.is_finite() || multi_seat.primary_seat_weight <= 0.0 {
        ctx.add_error(format!(
            "multi_seat.primary_seat_weight ({}) must be positive",
            multi_seat.primary_seat_weight
        ));
    }
    if let Some(weights) = &multi_seat.seat_weights {
        if weights.is_empty() {
            ctx.add_error("multi_seat.seat_weights must not be empty");
        }
        for (idx, weight) in weights.iter().enumerate() {
            if !weight.is_finite() || *weight < 0.0 {
                ctx.add_error(format!(
                    "multi_seat.seat_weights[{}] ({}) must be finite and non-negative",
                    idx, weight
                ));
            }
        }
    }
}

pub fn rule_max_iter(ctx: &mut ValidationContext<'_>) {
    if ctx.opt.max_iter == 0 {
        ctx.add_warning("max_iter is 0, optimization will not run");
    }
}

pub fn rule_algorithm(ctx: &mut ValidationContext<'_>) {
    let valid_prefixes = ["nlopt:", "mh:", "autoeq:"];
    let valid_bare = ["cobyla", "de"];
    let algo = ctx.opt.algorithm.as_str();
    let is_known = valid_prefixes.iter().any(|p| algo.starts_with(p)) || valid_bare.contains(&algo);
    if !is_known {
        ctx.add_warning(format!(
            "Unknown algorithm '{}', may not be supported",
            ctx.opt.algorithm
        ));
    }
}

pub fn rule_loss_type(ctx: &mut ValidationContext<'_>) {
    let valid_loss_types = ["flat", "score", "epa"];
    if !valid_loss_types.contains(&ctx.opt.loss_type.as_str()) {
        ctx.add_error(format!(
            "Unknown loss_type '{}', must be one of {:?}",
            ctx.opt.loss_type, valid_loss_types
        ));
    }
}

pub fn rule_peq_model(ctx: &mut ValidationContext<'_>) {
    let valid_peq_models = [
        "pk",
        "hp-pk",
        "ls-pk",
        "hp-pk-lp",
        "ls-pk-hs",
        "free-pk-free",
        "free",
    ];
    if !valid_peq_models.contains(&ctx.opt.peq_model.as_str()) {
        ctx.add_warning(format!(
            "Unknown peq_model '{}', may not be supported",
            ctx.opt.peq_model
        ));
    }
}

pub fn rule_cea2034_correction(ctx: &mut ValidationContext<'_>) {
    let Some(ref cea) = ctx.opt.cea2034_correction else {
        return;
    };
    if !cea.enabled {
        return;
    }

    if cea.num_filters == 0 || cea.num_filters > 20 {
        ctx.add_error(format!(
            "cea2034_correction.num_filters ({}) must be in range [1..20]",
            cea.num_filters
        ));
    }
    if cea.max_q <= 0.0 {
        ctx.add_error(format!(
            "cea2034_correction.max_q ({}) must be positive",
            cea.max_q
        ));
    }
    if cea.min_db >= 0.0 {
        ctx.add_warning(format!(
            "cea2034_correction.min_db ({}) is non-negative; speaker correction typically needs cuts",
            cea.min_db
        ));
    }
    if cea.max_db < cea.min_db {
        ctx.add_error(format!(
            "cea2034_correction.max_db ({}) must be >= min_db ({})",
            cea.max_db, cea.min_db
        ));
    }
    if cea.correction_mode == Cea2034CorrectionMode::Score {
        ctx.add_error(
            "cea2034_correction.correction_mode=score is not supported in roomeq; \
             Harman/Olive speaker score is defined for anechoic spinorama data, while \
             roomeq CEA2034 correction only supports flat Listening Window pre-correction",
        );
    }
    if cea.nearfield_threshold_m <= 0.0 {
        ctx.add_error(format!(
            "cea2034_correction.nearfield_threshold_m ({}) must be positive",
            cea.nearfield_threshold_m
        ));
    }
}

pub fn rule_phase_linear_max_freq(ctx: &mut ValidationContext<'_>) {
    if ctx.opt.processing_mode == ProcessingMode::PhaseLinear
        && ctx.opt.max_freq > PHASE_LINEAR_RECOMMENDED_MAX_FREQ_HZ
    {
        ctx.add_warning(format!(
            "processing_mode=phase_linear with max_freq={:.0} Hz exceeds the recommended \
             ceiling of {:.0} Hz for reasonable FIR tap counts. Consider capping max_freq \
             or increasing fir.taps; the resulting correction will otherwise be accurate \
             only in the bass/low-mid range.",
            ctx.opt.max_freq, PHASE_LINEAR_RECOMMENDED_MAX_FREQ_HZ
        ));
    }
}

pub fn rule_schroeder_split(ctx: &mut ValidationContext<'_>) {
    if !ctx.opt.schroeder_split.as_ref().is_some_and(|s| s.enabled) {
        return;
    }

    let has_slope = ctx
        .opt
        .target_response
        .as_ref()
        .map(|t| t.slope_db_per_octave.abs() > f64::EPSILON)
        .unwrap_or(false);
    if has_slope {
        ctx.add_warning(
            "schroeder_split is enabled together with a non-zero target slope \
             (target_response.slope_db_per_octave). The modal and diffuse regions \
             are optimized independently, so the requested slope will be \
             approximated rather than matched exactly across the crossover.",
        );
    }
}

pub fn rule_fir_config(ctx: &mut ValidationContext<'_>) {
    if matches!(
        ctx.opt.processing_mode,
        ProcessingMode::PhaseLinear | ProcessingMode::Hybrid | ProcessingMode::MixedPhase
    ) && ctx.opt.fir.is_none()
    {
        ctx.add_warning(format!(
            "processing_mode={:?} requires FIR configuration; using defaults",
            ctx.opt.processing_mode
        ));
    }

    let Some(ref fir) = ctx.opt.fir else {
        return;
    };
    if fir.taps == 0 {
        ctx.add_error("FIR taps must be greater than 0");
    }
    if fir.taps < 256 {
        ctx.add_warning(format!(
            "FIR taps ({}) is low, may result in poor frequency resolution",
            fir.taps
        ));
    }
    let valid_phases = ["linear", "minimum", "kirkeby"];
    if !valid_phases.contains(&fir.phase.to_lowercase().as_str()) {
        ctx.add_error(format!(
            "Unknown FIR phase '{}', must be one of {:?}",
            fir.phase, valid_phases
        ));
    }
}

pub fn rule_mixed_config(ctx: &mut ValidationContext<'_>) {
    let Some(ref mixed_config) = ctx.opt.mixed_config else {
        return;
    };

    if ctx.opt.processing_mode != ProcessingMode::Hybrid {
        ctx.add_warning(
            "mixed_config specified but processing_mode is not Hybrid, configuration will be ignored",
        );
    }

    if mixed_config.crossover_freq <= 0.0 {
        ctx.add_error(format!(
            "mixed_config.crossover_freq ({}) must be positive",
            mixed_config.crossover_freq
        ));
    }
    if mixed_config.crossover_freq < ctx.opt.min_freq {
        ctx.add_warning(format!(
            "mixed_config.crossover_freq ({}) is below min_freq ({}), some frequencies may not be optimized",
            mixed_config.crossover_freq, ctx.opt.min_freq
        ));
    }
    if mixed_config.crossover_freq > ctx.opt.max_freq {
        ctx.add_warning(format!(
            "mixed_config.crossover_freq ({}) is above max_freq ({}), some frequencies may not be optimized",
            mixed_config.crossover_freq, ctx.opt.max_freq
        ));
    }

    let valid_crossover_types = ["LR24", "LR48", "LR4", "LR8"];
    if !valid_crossover_types
        .iter()
        .any(|&t| t.eq_ignore_ascii_case(&mixed_config.crossover_type))
    {
        ctx.add_error(format!(
            "Unknown mixed_config.crossover_type '{}', must be one of {:?}",
            mixed_config.crossover_type, valid_crossover_types
        ));
    }

    let valid_fir_bands = ["low", "high"];
    if !valid_fir_bands
        .iter()
        .any(|&b| b.eq_ignore_ascii_case(&mixed_config.fir_band))
    {
        ctx.add_error(format!(
            "Unknown mixed_config.fir_band '{}', must be 'low' or 'high'",
            mixed_config.fir_band
        ));
    }

    let fir_uses_low = mixed_config.fir_band.eq_ignore_ascii_case("low");
    let crossover = mixed_config.crossover_freq;

    if fir_uses_low {
        if crossover <= ctx.opt.min_freq {
            ctx.add_error(format!(
                "In mixed mode with fir_band='low', crossover_freq ({}) must be greater than min_freq ({}) \
                to give the FIR band a valid range",
                crossover, ctx.opt.min_freq
            ));
        }
        if crossover >= ctx.opt.max_freq {
            ctx.add_error(format!(
                "In mixed mode with fir_band='low', crossover_freq ({}) must be less than max_freq ({}) \
                to give the IIR band a valid range",
                crossover, ctx.opt.max_freq
            ));
        }
    } else {
        if crossover <= ctx.opt.min_freq {
            ctx.add_error(format!(
                "In mixed mode with fir_band='high', crossover_freq ({}) must be greater than min_freq ({}) \
                to give the IIR band a valid range",
                crossover, ctx.opt.min_freq
            ));
        }
        if crossover >= ctx.opt.max_freq {
            ctx.add_error(format!(
                "In mixed mode with fir_band='high', crossover_freq ({}) must be less than max_freq ({}) \
                to give the FIR band a valid range",
                crossover, ctx.opt.max_freq
            ));
        }
    }
}

/// Runs the full set of optimizer validation rules against `ctx`.
///
/// This is the orchestrator used by `validate_optimizer_config`.
pub fn run_optimizer_validation_rules(ctx: &mut ValidationContext<'_>) {
    rule_num_filters(ctx);
    rule_freq_range(ctx);
    rule_q_range(ctx);
    rule_smooth_n(ctx);
    rule_psychoacoustic_smoothing(ctx);
    rule_asymmetric_loss(ctx);
    rule_audibility_deadband(ctx);
    rule_high_frequency_correction(ctx);
    rule_early_late_correction(ctx);
    rule_validation_bundle(ctx);
    rule_gain_bounds(ctx);
    rule_excursion_protection(ctx);
    rule_auto_optimizer(ctx);
    rule_multi_seat(ctx);
    rule_max_iter(ctx);
    rule_algorithm(ctx);
    rule_loss_type(ctx);
    rule_peq_model(ctx);
    rule_cea2034_correction(ctx);
    rule_phase_linear_max_freq(ctx);
    rule_schroeder_split(ctx);
    rule_fir_config(ctx);
    rule_mixed_config(ctx);
}

#[cfg(test)]
mod optimizer_rule_tests {
    use super::*;
    use crate::roomeq::config::validation_result::{ValidationContext, ValidationResult};
    use crate::roomeq::types::OptimizerConfig;

    fn run_rule(f: fn(&mut ValidationContext<'_>), config: &OptimizerConfig) -> ValidationResult {
        let mut result = ValidationResult::valid();
        let mut ctx = ValidationContext::new(config, &mut result);
        f(&mut ctx);
        result
    }

    fn default_config() -> OptimizerConfig {
        OptimizerConfig::default()
    }

    #[test]
    fn rule_num_filters_default_is_silent() {
        let result = run_rule(rule_num_filters, &default_config());
        assert!(result.is_valid);
        assert!(result.warnings.is_empty());
    }

    #[test]
    fn rule_num_filters_zero_warns() {
        let mut config = default_config();
        config.num_filters = 0;
        let result = run_rule(rule_num_filters, &config);
        assert!(result.is_valid);
        assert!(
            result
                .warnings
                .iter()
                .any(|w| w.contains("num_filters is 0"))
        );
    }

    #[test]
    fn rule_freq_range_default_is_valid() {
        let result = run_rule(rule_freq_range, &default_config());
        assert!(result.is_valid);
    }

    #[test]
    fn rule_freq_range_reversed_errors() {
        let mut config = default_config();
        config.min_freq = 200.0;
        config.max_freq = 100.0;
        let result = run_rule(rule_freq_range, &config);
        assert!(!result.is_valid);
    }

    #[test]
    fn rule_freq_range_non_positive_min_errors() {
        let mut config = default_config();
        config.min_freq = 0.0;
        let result = run_rule(rule_freq_range, &config);
        assert!(!result.is_valid);
    }

    #[test]
    fn rule_freq_range_above_nyquist_warns() {
        let mut config = default_config();
        config.max_freq = 30000.0;
        let result = run_rule(rule_freq_range, &config);
        assert!(result.is_valid);
        assert!(result.warnings.iter().any(|w| w.contains("Nyquist")));
    }

    #[test]
    fn rule_q_range_default_is_valid() {
        let result = run_rule(rule_q_range, &default_config());
        assert!(result.is_valid);
    }

    #[test]
    fn rule_q_range_reversed_errors() {
        let mut config = default_config();
        config.min_q = 5.0;
        config.max_q = 2.0;
        let result = run_rule(rule_q_range, &config);
        assert!(!result.is_valid);
    }

    #[test]
    fn rule_q_range_non_positive_errors() {
        let mut config = default_config();
        config.min_q = 0.0;
        let result = run_rule(rule_q_range, &config);
        assert!(!result.is_valid);
    }

    #[test]
    fn rule_smooth_n_default_is_valid() {
        let result = run_rule(rule_smooth_n, &default_config());
        assert!(result.is_valid);
    }

    #[test]
    fn rule_smooth_n_out_of_range_errors() {
        let mut config = default_config();
        config.smooth_n = 0;
        let result = run_rule(rule_smooth_n, &config);
        assert!(!result.is_valid);
    }

    #[test]
    fn rule_algorithm_known_is_silent() {
        let result = run_rule(rule_algorithm, &default_config());
        assert!(result.is_valid);
        assert!(result.warnings.is_empty());
    }

    #[test]
    fn rule_algorithm_unknown_warns() {
        let mut config = default_config();
        config.algorithm = "unknown:algo".to_string();
        let result = run_rule(rule_algorithm, &config);
        assert!(result.is_valid);
        assert!(
            result
                .warnings
                .iter()
                .any(|w| w.contains("Unknown algorithm"))
        );
    }

    #[test]
    fn rule_loss_type_known_is_valid() {
        let result = run_rule(rule_loss_type, &default_config());
        assert!(result.is_valid);
    }

    #[test]
    fn rule_loss_type_unknown_errors() {
        let mut config = default_config();
        config.loss_type = "invalid".to_string();
        let result = run_rule(rule_loss_type, &config);
        assert!(!result.is_valid);
    }

    #[test]
    fn rule_peq_model_known_is_silent() {
        let result = run_rule(rule_peq_model, &default_config());
        assert!(result.is_valid);
        assert!(result.warnings.is_empty());
    }

    #[test]
    fn rule_peq_model_unknown_warns() {
        let mut config = default_config();
        config.peq_model = "unknown-model".to_string();
        let result = run_rule(rule_peq_model, &config);
        assert!(result.is_valid);
        assert!(
            result
                .warnings
                .iter()
                .any(|w| w.contains("Unknown peq_model"))
        );
    }

    #[test]
    fn rule_max_iter_default_is_silent() {
        let result = run_rule(rule_max_iter, &default_config());
        assert!(result.is_valid);
        assert!(result.warnings.is_empty());
    }

    #[test]
    fn rule_max_iter_zero_warns() {
        let mut config = default_config();
        config.max_iter = 0;
        let result = run_rule(rule_max_iter, &config);
        assert!(result.is_valid);
        assert!(result.warnings.iter().any(|w| w.contains("max_iter is 0")));
    }

    #[test]
    fn rule_gain_bounds_default_is_valid() {
        let result = run_rule(rule_gain_bounds, &default_config());
        assert!(result.is_valid);
    }

    #[test]
    fn rule_gain_bounds_reversed_errors() {
        let mut config = default_config();
        config.min_db = 10.0;
        config.max_db = 5.0;
        let result = run_rule(rule_gain_bounds, &config);
        assert!(!result.is_valid);
    }

    #[test]
    fn rule_phase_linear_max_freq_default_is_silent() {
        let result = run_rule(rule_phase_linear_max_freq, &default_config());
        assert!(result.is_valid);
        assert!(result.warnings.is_empty());
    }

    #[test]
    fn rule_phase_linear_max_freq_exceeds_recommended_warns() {
        let mut config = default_config();
        config.processing_mode = crate::roomeq::types::ProcessingMode::PhaseLinear;
        config.max_freq = 5000.0;
        let result = run_rule(rule_phase_linear_max_freq, &config);
        assert!(result.is_valid);
        assert!(result.warnings.iter().any(|w| w.contains("phase_linear")));
    }

    #[test]
    fn rule_schroeder_split_no_slope_is_silent() {
        let result = run_rule(rule_schroeder_split, &default_config());
        assert!(result.is_valid);
        assert!(result.warnings.is_empty());
    }

    #[test]
    fn rule_fir_config_default_is_valid() {
        let result = run_rule(rule_fir_config, &default_config());
        assert!(result.is_valid);
    }

    #[test]
    fn rule_fir_config_zero_taps_errors() {
        let mut config = default_config();
        config.fir = Some(crate::roomeq::types::FirConfig {
            taps: 0,
            phase: "linear".to_string(),
            ..Default::default()
        });
        let result = run_rule(rule_fir_config, &config);
        assert!(!result.is_valid);
    }

    #[test]
    fn rule_fir_config_invalid_phase_errors() {
        let mut config = default_config();
        config.fir = Some(crate::roomeq::types::FirConfig {
            taps: 256,
            phase: "invalid".to_string(),
            ..Default::default()
        });
        let result = run_rule(rule_fir_config, &config);
        assert!(!result.is_valid);
    }

    #[test]
    fn rule_mixed_config_default_is_silent() {
        let result = run_rule(rule_mixed_config, &default_config());
        assert!(result.is_valid);
        assert!(result.warnings.is_empty());
    }

    #[test]
    fn rule_cea2034_correction_disabled_is_silent() {
        let result = run_rule(rule_cea2034_correction, &default_config());
        assert!(result.is_valid);
        assert!(result.warnings.is_empty());
    }

    #[test]
    fn run_optimizer_validation_rules_default_is_valid() {
        let config = default_config();
        let mut result = ValidationResult::valid();
        let mut ctx = ValidationContext::new(&config, &mut result);
        run_optimizer_validation_rules(&mut ctx);
        assert!(result.is_valid);
    }

    #[test]
    fn rule_psychoacoustic_smoothing_errors() {
        let mut config = default_config();
        config.psychoacoustic_smoothing = Some(crate::read::PsychoacousticSmoothingConfig {
            low_freq_n: 0,
            high_freq_n: 1,
            low_freq: 100.0,
            high_freq: 1000.0,
        });
        let result = run_rule(rule_psychoacoustic_smoothing, &config);
        assert!(
            result
                .errors
                .iter()
                .any(|e| e.contains("low_freq_n/high_freq_n must be at least 1"))
        );

        let mut config = default_config();
        config.psychoacoustic_smoothing = Some(crate::read::PsychoacousticSmoothingConfig {
            low_freq_n: 1,
            high_freq_n: 1,
            low_freq: 200.0,
            high_freq: 100.0,
        });
        let result = run_rule(rule_psychoacoustic_smoothing, &config);
        assert!(
            result
                .errors
                .iter()
                .any(|e| e.contains("0 < low_freq < high_freq"))
        );
    }

    #[test]
    fn rule_asymmetric_loss_errors() {
        let mut config = default_config();
        config.asymmetric_loss_config = Some(crate::loss::AsymmetricLossConfig {
            transition_freq: 0.0,
            peak_weight: -1.0,
            dip_weight: 1.0,
            bass_peak_weight: 1.0,
            bass_dip_weight: 1.0,
        });
        let result = run_rule(rule_asymmetric_loss, &config);
        assert!(
            result
                .errors
                .iter()
                .any(|e| e.contains("transition_freq") && e.contains("must be positive"))
        );
        assert!(
            result
                .errors
                .iter()
                .any(|e| e.contains("weights must be non-negative"))
        );
    }

    #[test]
    fn rule_audibility_deadband_errors() {
        let mut config = default_config();
        config.audibility_deadband = Some(crate::roomeq::types::AudibilityDeadbandConfig {
            bass_db: -1.0,
            mid_db: 1.0,
            treble_db: 1.0,
            bass_mid_hz: 200.0,
            mid_treble_hz: 100.0,
            schroeder_hz: 0.0,
            ..Default::default()
        });
        let result = run_rule(rule_audibility_deadband, &config);
        assert!(
            result
                .errors
                .iter()
                .any(|e| e.contains("thresholds must be non-negative"))
        );
        assert!(
            result
                .errors
                .iter()
                .any(|e| e.contains("0 < bass_mid_hz < mid_treble_hz"))
        );
        assert!(
            result
                .errors
                .iter()
                .any(|e| e.contains("schroeder_hz") && e.contains("must be positive"))
        );
    }

    #[test]
    fn rule_high_frequency_correction_errors() {
        let mut config = default_config();
        config.high_frequency_correction =
            Some(crate::roomeq::types::HighFrequencyCorrectionConfig {
                enabled: true,
                start_hz: 0.0,
                extra_deadband_db: -1.0,
                smoothing_n: 0,
                max_q: 0.0,
            });
        let result = run_rule(rule_high_frequency_correction, &config);
        assert!(
            result
                .errors
                .iter()
                .any(|e| e.contains("start_hz") && e.contains("must be positive"))
        );
        assert!(
            result
                .errors
                .iter()
                .any(|e| e.contains("extra_deadband_db") && e.contains("must be non-negative"))
        );
        assert!(
            result
                .errors
                .iter()
                .any(|e| e.contains("smoothing_n must be at least 1"))
        );
        assert!(
            result
                .errors
                .iter()
                .any(|e| e.contains("max_q") && e.contains("must be positive"))
        );
    }

    #[test]
    fn rule_early_late_correction_errors() {
        let mut config = default_config();
        config.early_late_correction = Some(crate::roomeq::types::EarlyLateCorrectionConfig {
            direct_window_ms: 2.0,
            early_window_ms: 1.0,
            late_window_ms: 0.5,
            early_cue_risk_db: f64::NAN,
            ..Default::default()
        });
        let result = run_rule(rule_early_late_correction, &config);
        assert!(
            result
                .errors
                .iter()
                .any(|e| e.contains("0 < direct < early < late"))
        );
        assert!(
            result
                .errors
                .iter()
                .any(|e| e.contains("early_cue_risk_db must be finite"))
        );
    }

    #[test]
    fn rule_validation_bundle_errors() {
        let mut config = default_config();
        config.validation_bundle = Some(crate::roomeq::types::ValidationBundleConfig {
            target_lufs: f64::NAN,
            ..Default::default()
        });
        let result = run_rule(rule_validation_bundle, &config);
        assert!(
            result
                .errors
                .iter()
                .any(|e| e.contains("target_lufs must be finite"))
        );
    }

    #[test]
    fn rule_excursion_protection_errors() {
        let mut config = default_config();
        config.excursion_protection = Some(crate::roomeq::types::ExcursionProtectionConfig {
            f3_reference_min_hz: 200.0,
            f3_reference_max_hz: 100.0,
            ..Default::default()
        });
        let result = run_rule(rule_excursion_protection, &config);
        assert!(
            result
                .errors
                .iter()
                .any(|e| e.contains("F3 reference band") && e.contains("0 < min < max"))
        );
    }

    #[test]
    fn rule_auto_optimizer_errors_and_warnings() {
        let mut config = default_config();
        config.auto_optimizer = Some(crate::roomeq::types::AutoOptimizerConfig {
            enabled: true,
            min_filters: 0,
            max_filters: 0,
            filter_count: false,
            q_bounds: false,
            gain_bounds: false,
        });
        let result = run_rule(rule_auto_optimizer, &config);
        assert!(
            result
                .errors
                .iter()
                .any(|e| e.contains("min_filters must be at least 1"))
        );
        assert!(
            result
                .errors
                .iter()
                .any(|e| e.contains("max_filters must be at least 1"))
        );
        assert!(
            result
                .warnings
                .iter()
                .any(|w| w.contains("all automatic selection flags are disabled"))
        );

        // max_filters < 2 with schroeder_split enabled
        let mut config = default_config();
        config.auto_optimizer = Some(crate::roomeq::types::AutoOptimizerConfig {
            enabled: true,
            min_filters: 1,
            max_filters: 1,
            filter_count: true,
            q_bounds: true,
            gain_bounds: true,
        });
        config.schroeder_split = Some(crate::roomeq::types::SchroederSplitConfig {
            enabled: true,
            ..Default::default()
        });
        let result = run_rule(rule_auto_optimizer, &config);
        assert!(
            result
                .errors
                .iter()
                .any(|e| e
                    .contains("max_filters must be at least 2 when schroeder_split is enabled"))
        );

        // min_filters > max_filters
        let mut config = default_config();
        config.auto_optimizer = Some(crate::roomeq::types::AutoOptimizerConfig {
            enabled: true,
            min_filters: 5,
            max_filters: 2,
            filter_count: true,
            q_bounds: true,
            gain_bounds: true,
        });
        let result = run_rule(rule_auto_optimizer, &config);
        assert!(
            result
                .errors
                .iter()
                .any(|e| e.contains("min_filters (5) must be <= max_filters (2)"))
        );
    }

    #[test]
    fn rule_multi_seat_errors() {
        let mut config = default_config();
        config.multi_seat = Some(crate::roomeq::types::MultiSeatConfig {
            enabled: true,
            max_deviation_db: -1.0,
            primary_seat_weight: 0.0,
            seat_weights: Some(vec![0.5, f64::NAN]),
            ..Default::default()
        });
        let result = run_rule(rule_multi_seat, &config);
        assert!(
            result
                .errors
                .iter()
                .any(|e| e.contains("max_deviation_db") && e.contains("must be non-negative"))
        );
        assert!(
            result
                .errors
                .iter()
                .any(|e| e.contains("primary_seat_weight") && e.contains("must be positive"))
        );
        assert!(
            result
                .errors
                .iter()
                .any(|e| e.contains("seat_weights[1]") && e.contains("must be finite"))
        );

        // empty seat_weights
        let mut config = default_config();
        config.multi_seat = Some(crate::roomeq::types::MultiSeatConfig {
            enabled: true,
            seat_weights: Some(vec![]),
            ..Default::default()
        });
        let result = run_rule(rule_multi_seat, &config);
        assert!(
            result
                .errors
                .iter()
                .any(|e| e.contains("seat_weights must not be empty"))
        );
    }

    #[test]
    fn rule_cea2034_correction_enabled_errors() {
        let mut config = default_config();
        config.cea2034_correction = Some(crate::roomeq::types::Cea2034CorrectionConfig {
            enabled: true,
            num_filters: 0,
            max_q: 0.0,
            min_db: 1.0,
            max_db: -1.0,
            correction_mode: crate::roomeq::types::Cea2034CorrectionMode::Score,
            nearfield_threshold_m: 0.0,
            ..Default::default()
        });
        let result = run_rule(rule_cea2034_correction, &config);
        assert!(
            result
                .errors
                .iter()
                .any(|e| e.contains("num_filters") && e.contains("range [1..20]"))
        );
        assert!(
            result
                .errors
                .iter()
                .any(|e| e.contains("max_q") && e.contains("must be positive"))
        );
        assert!(
            result
                .warnings
                .iter()
                .any(|w| w.contains("min_db") && w.contains("non-negative"))
        );
        assert!(
            result
                .errors
                .iter()
                .any(|e| e.contains("max_db") && e.contains("must be >= min_db"))
        );
        assert!(
            result
                .errors
                .iter()
                .any(|e| e.contains("score is not supported"))
        );
        assert!(
            result
                .errors
                .iter()
                .any(|e| e.contains("nearfield_threshold_m") && e.contains("must be positive"))
        );
    }

    #[test]
    fn rule_fir_config_warnings_and_errors() {
        let mut config = default_config();
        config.processing_mode = crate::roomeq::types::ProcessingMode::PhaseLinear;
        config.fir = None;
        let result = run_rule(rule_fir_config, &config);
        assert!(
            result
                .warnings
                .iter()
                .any(|w| w.contains("requires FIR configuration"))
        );

        let mut config = default_config();
        config.fir = Some(crate::roomeq::types::FirConfig {
            taps: 128,
            phase: "linear".to_string(),
            ..Default::default()
        });
        let result = run_rule(rule_fir_config, &config);
        assert!(
            result
                .warnings
                .iter()
                .any(|w| w.contains("FIR taps (128) is low"))
        );
    }

    #[test]
    fn rule_schroeder_split_slope_warning() {
        let mut config = default_config();
        config.schroeder_split = Some(crate::roomeq::types::SchroederSplitConfig {
            enabled: true,
            ..Default::default()
        });
        config.target_response = Some(crate::roomeq::types::TargetResponseConfig {
            slope_db_per_octave: 1.0,
            ..Default::default()
        });
        let result = run_rule(rule_schroeder_split, &config);
        assert!(result.warnings.iter().any(|w| {
            w.contains("schroeder_split is enabled together with a non-zero target slope")
        }));
    }

    #[test]
    fn rule_mixed_config_validation() {
        let mut config = default_config();
        config.mixed_config = Some(crate::roomeq::types::MixedModeConfig {
            crossover_freq: 0.0,
            crossover_type: "LR24".to_string(),
            fir_band: "low".to_string(),
        });
        config.processing_mode = crate::roomeq::types::ProcessingMode::Hybrid;
        config.min_freq = 20.0;
        config.max_freq = 20000.0;
        let result = run_rule(rule_mixed_config, &config);
        assert!(
            result
                .errors
                .iter()
                .any(|e| e.contains("crossover_freq (0) must be positive"))
        );

        let mut config = default_config();
        config.mixed_config = Some(crate::roomeq::types::MixedModeConfig {
            crossover_freq: 100.0,
            crossover_type: "unknown".to_string(),
            fir_band: "low".to_string(),
        });
        config.processing_mode = crate::roomeq::types::ProcessingMode::Hybrid;
        config.min_freq = 20.0;
        config.max_freq = 20000.0;
        let result = run_rule(rule_mixed_config, &config);
        assert!(
            result
                .errors
                .iter()
                .any(|e| e.contains("Unknown mixed_config.crossover_type"))
        );

        let mut config = default_config();
        config.mixed_config = Some(crate::roomeq::types::MixedModeConfig {
            crossover_freq: 100.0,
            crossover_type: "LR24".to_string(),
            fir_band: "unknown".to_string(),
        });
        config.processing_mode = crate::roomeq::types::ProcessingMode::Hybrid;
        config.min_freq = 20.0;
        config.max_freq = 20000.0;
        let result = run_rule(rule_mixed_config, &config);
        assert!(
            result
                .errors
                .iter()
                .any(|e| e.contains("Unknown mixed_config.fir_band"))
        );

        // warnings for crossover outside frequency range
        let mut config = default_config();
        config.mixed_config = Some(crate::roomeq::types::MixedModeConfig {
            crossover_freq: 10.0,
            crossover_type: "LR24".to_string(),
            fir_band: "low".to_string(),
        });
        config.processing_mode = crate::roomeq::types::ProcessingMode::Hybrid;
        config.min_freq = 20.0;
        config.max_freq = 20000.0;
        let result = run_rule(rule_mixed_config, &config);
        assert!(
            result
                .warnings
                .iter()
                .any(|w| w.contains("crossover_freq (10) is below min_freq"))
        );

        let mut config = default_config();
        config.mixed_config = Some(crate::roomeq::types::MixedModeConfig {
            crossover_freq: 25000.0,
            crossover_type: "LR24".to_string(),
            fir_band: "low".to_string(),
        });
        config.processing_mode = crate::roomeq::types::ProcessingMode::Hybrid;
        config.min_freq = 20.0;
        config.max_freq = 20000.0;
        let result = run_rule(rule_mixed_config, &config);
        assert!(
            result
                .warnings
                .iter()
                .any(|w| w.contains("crossover_freq (25000) is above max_freq"))
        );

        // non-hybrid warning
        let mut config = default_config();
        config.mixed_config = Some(crate::roomeq::types::MixedModeConfig::default());
        let result = run_rule(rule_mixed_config, &config);
        assert!(
            result
                .warnings
                .iter()
                .any(|w| w.contains("mixed_config specified but processing_mode is not Hybrid"))
        );
    }
}
