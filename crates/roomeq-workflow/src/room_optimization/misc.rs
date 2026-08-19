use log::{debug, info, warn};
use math_audio_dsp::analysis::compute_average_response;
use roomeq_engine::analysis::slope;
use roomeq_engine::pipeline::PipelineStepId;
use roomeq_model::validation_rules::RoomValidationContext;
use roomeq_model::{AutoeqError, Result};
use roomeq_model::{MeasurementSource, RoomConfig, SpeakerConfig, TargetShape};
use std::collections::HashMap;

/// Detect passband and compute mean SPL for normalization.
///
/// Smooths the measurement at 1 octave to suppress room modes and comb
/// filtering, then uses a log-frequency weighted average (not the raw
/// median) as the reference level. The passband is the full span between
/// the first and last samples whose smoothed SPL is within 10 dB of that
/// reference.
///
/// Using a log-frequency weighted reference is essential: a raw median on
/// linearly sampled curves biases toward the high-frequency region, and a
/// single strong bass mode can inflate the median enough that the true
/// passband falls below the -10 dB threshold, collapsing detection to a
/// tiny bass window. Similarly, searching only for the first crossing from
/// each end misreports the high edge when the curve does not roll off
/// within the measurement range and is tricked into returning a deep
/// mid-band null. First-above / last-above indices are robust to both
/// failure modes.
/// Threshold in dB above which to warn about channel level differences
pub(super) const LEVEL_DIFFERENCE_WARNING_THRESHOLD: f64 = 6.0;

/// Threshold in ms above which to warn about arrival time differences
pub(super) const ARRIVAL_TIME_WARNING_THRESHOLD_MS: f64 = 50.0;

pub(super) fn is_subwoofer_channel(config: &RoomConfig, channel_name: &str) -> bool {
    if let Some(sys) = &config.system
        && let Some(subs) = &sys.subwoofers
    {
        if channel_name.eq_ignore_ascii_case("lfe") {
            return true;
        }

        if let Some(measurement_key) = sys.speakers.get(channel_name) {
            return subs.mapping.contains_key(measurement_key);
        }
    }

    let lower = channel_name.to_lowercase();
    lower == "lfe" || lower == "sub" || lower.starts_with("sub")
}

/// Find subwoofer-to-main-speaker pairings using system config or heuristic fallback.
///
/// Returns `(sub_name, main_name)` pairs where names are keys into the curves/chains maps.
/// Used by both phase alignment and GD-Opt v2.
pub(super) fn find_sub_main_pairings(
    config: &RoomConfig,
    curves: &HashMap<String, roomeq_model::Curve>,
) -> Vec<(String, String)> {
    let mut pairings = Vec::new();

    if let Some(sys) = &config.system {
        // Use explicit system configuration
        if let Some(subs) = &sys.subwoofers {
            // Invert speakers map to find roles from measurement keys
            // measurement_key -> role
            let meas_to_role: HashMap<&String, &String> =
                sys.speakers.iter().map(|(r, m)| (m, r)).collect();

            for (sub_meas_key, main_role) in &subs.mapping {
                if let Some(sub_role) = meas_to_role.get(sub_meas_key) {
                    pairings.push((sub_role.to_string(), main_role.clone()));
                } else {
                    warn!(
                        "Subwoofer measurement '{}' not mapped to any output channel",
                        sub_meas_key
                    );
                }
            }
        }
    } else {
        // Legacy heuristic: find "lfe" or "sub*" channel, pair with all non-sub channels
        let sub_channel = curves
            .keys()
            .find(|name| is_subwoofer_channel(config, name))
            .cloned();
        if let Some(sub_name) = sub_channel {
            let main_channels: Vec<String> = curves
                .keys()
                .filter(|name| *name != &sub_name && !is_subwoofer_channel(config, name))
                .cloned()
                .collect();
            for main in main_channels {
                pairings.push((sub_name.clone(), main));
            }
        }
    }

    pairings
}

pub(super) fn pipeline_stopped_error(step_id: PipelineStepId) -> AutoeqError {
    AutoeqError::OptimizationFailed {
        message: format!("Room optimization stopped by observer during {:?}", step_id),
    }
}

pub(super) fn optimizer_progress_iterations(config: &RoomConfig) -> usize {
    let params_per_filter = match config.optimizer.peq_model.as_str() {
        "free" | "ls-pk-hs" => 4,
        _ => 3,
    };
    let n_params = config.optimizer.num_filters * params_per_filter;
    let n_free = n_params.max(1);
    let desired_pop = config
        .optimizer
        .population
        .max(1)
        .min(config.optimizer.max_iter.max(1));
    let pop_multiplier = desired_pop.div_ceil(n_free).max(4);
    let population_size = pop_multiplier * n_free;
    const DE_GENERATIONS_FLOOR: usize = 5000;
    let computed_generations =
        config.optimizer.max_iter.saturating_sub(population_size) / population_size;
    if config.optimizer.max_iter >= DE_GENERATIONS_FLOOR.saturating_mul(population_size) {
        computed_generations.max(DE_GENERATIONS_FLOOR)
    } else {
        computed_generations.max(1)
    }
}

pub(super) fn prepare_room_config_with_frequency_samples(
    config: &RoomConfig,
    frequency_samples: usize,
) -> RoomConfig {
    let mut config = config.clone();

    config.optimizer.apply_perceptual_policy_defaults();
    config
        .optimizer
        .apply_high_frequency_correction_defaults(false);

    // Resolve `TargetShape::FromMeasurement` slope once, system-wide,
    // from a full-range reference channel — see
    // `resolve_from_measurement_slope` for the picking rules. Lifting
    // this out of the per-channel loop prevents band-limited channels
    // (LFE, sub) from deriving a junk slope from their own rolled-off
    // skirts.
    if config
        .optimizer
        .target_response
        .as_ref()
        .is_some_and(|t| t.shape == TargetShape::FromMeasurement)
        && config.optimizer.from_measurement_slope_override.is_none()
    {
        let resolved =
            resolve_from_measurement_slope_with_frequency_samples(&config, frequency_samples);
        config.optimizer.from_measurement_slope_override = Some(resolved);
    }

    // Pre-fetch CEA2034 data for all speakers when speaker pre-correction is enabled
    if config
        .optimizer
        .cea2034_correction
        .as_ref()
        .is_some_and(|c| c.enabled)
    {
        let cache = crate::cea2034::pre_fetch_all_cea2034(&config);
        if !cache.is_empty() {
            info!(
                "  CEA2034 cache: loaded data for {} speaker(s)",
                cache.len()
            );
            config.cea2034_cache = Some(cache);
        }
    }

    config
}

pub(super) fn validate_room_config_or_fail_with_frequency_samples(
    config: &RoomConfig,
    frequency_samples: usize,
) -> Result<()> {
    let validation = crate::config_loader::validate_room_config_for_workflow_with_frequency_samples(
        config,
        RoomValidationContext::production(),
        frequency_samples,
    );
    for warning in validation.warnings() {
        eprintln!("Warning: {warning}");
    }
    for error in validation.errors() {
        eprintln!("Error: {error}");
    }
    if !validation.production_ready() {
        let errors = validation.errors().map(String::as_str).collect::<Vec<_>>();
        return Err(AutoeqError::OptimizationFailed {
            message: format!(
                "Configuration validation failed with {} errors: {}",
                errors.len(),
                errors.join("; ")
            ),
        });
    }
    Ok(())
}

pub(super) fn channels_for_generic_optimization(
    config: &RoomConfig,
) -> Vec<(String, SpeakerConfig)> {
    if let Some(sys) = &config.system {
        info!("Using SystemConfig for channel mapping");
        sys.speakers
            .iter()
            .filter_map(|(role, key)| match config.speakers.get(key) {
                Some(cfg) => Some((role.clone(), cfg.clone())),
                None => {
                    warn!(
                        "System config references missing speaker key '{}' for role '{}'",
                        key, role
                    );
                    None
                }
            })
            .collect()
    } else {
        config
            .speakers
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect()
    }
}

pub(super) fn compute_shared_mean_spl_with_frequency_samples(
    config: &RoomConfig,
    channels_to_process: &[(String, SpeakerConfig)],
    frequency_samples: usize,
) -> Option<f64> {
    if channels_to_process.len() <= 1 {
        return None;
    }

    let min_freq = config.optimizer.min_freq;
    let max_freq = config.optimizer.max_freq;
    let mut channel_means: Vec<f64> = Vec::new();
    let mut excluded_group_count = 0_usize;

    for (_name, speaker_config) in channels_to_process {
        if let SpeakerConfig::Single(source) = speaker_config
            && let Ok(curve) =
                crate::measurement::load_source_with_frequency_samples(source, frequency_samples)
        {
            let freqs_f32: Vec<f32> = curve.freq.iter().map(|&f| f as f32).collect();
            let spl_f32: Vec<f32> = curve.spl.iter().map(|&s| s as f32).collect();
            let mean = compute_average_response(
                &freqs_f32,
                &spl_f32,
                Some((min_freq as f32, max_freq as f32)),
            ) as f64;
            channel_means.push(mean);
        } else if !matches!(speaker_config, SpeakerConfig::Single(_)) {
            excluded_group_count += 1;
        }
    }

    if excluded_group_count > 0 {
        info!(
            "Shared mean pre-pass: {} non-Single speaker(s) excluded (Group/MultiSub/DBA/Cardioid)",
            excluded_group_count
        );
    }

    if channel_means.len() > 1 {
        let avg = shared_target_level(&channel_means);
        info!(
            "Shared target level: {:.1} dB (robust center of {} channels)",
            avg,
            channel_means.len()
        );
        Some(avg)
    } else {
        None
    }
}

pub(super) fn generic_channel_progress_iterations(config: &RoomConfig) -> usize {
    let params_per_filter = match config.optimizer.peq_model.as_str() {
        "free" | "ls-pk-hs" => 4,
        _ => 3,
    };
    let n_params = config.optimizer.num_filters * params_per_filter;
    let n_free = n_params.max(1); // all params are free in standard EQ
    let desired_pop = config
        .optimizer
        .population
        .max(1)
        .min(config.optimizer.max_iter.max(1));
    let pop_multiplier = desired_pop.div_ceil(n_free).max(4);
    let population_size = pop_multiplier * n_free;
    // Only apply the 5000-generation floor when the user's budget actually
    // supports it; otherwise honour the requested max_iter so QA / benchmark
    // runs don't silently exceed their evaluation budget.
    // Mirrors `derive_de_budget` in `optim_de.rs`.
    const DE_GENERATIONS_FLOOR: usize = 5000;
    let computed_generations =
        config.optimizer.max_iter.saturating_sub(population_size) / population_size;
    let budget_supports_floor =
        config.optimizer.max_iter >= DE_GENERATIONS_FLOOR.saturating_mul(population_size);
    let max_iterations = if budget_supports_floor {
        computed_generations.max(DE_GENERATIONS_FLOOR)
    } else {
        let capped = computed_generations.max(1);
        if config.optimizer.max_iter > 0 && capped < DE_GENERATIONS_FLOOR {
            warn!(
                "Optimizer budget: max_iter={} with population_size={} is below the {} generation floor × pop. \
                 Running {} generations — expect degraded convergence. Raise max_iter to {} to regain the floor.",
                config.optimizer.max_iter,
                population_size,
                DE_GENERATIONS_FLOOR,
                capped,
                DE_GENERATIONS_FLOOR.saturating_mul(population_size),
            );
        }
        capped
    };
    info!(
        "Optimizer budget: {} params, population_size={}, max_generations={} (from max_iter={}, floor={} when budget allows)",
        n_params, population_size, max_iterations, config.optimizer.max_iter, DE_GENERATIONS_FLOOR,
    );
    max_iterations
}

/// Fixed iteration order for bed channels feeding the room-level
/// `FromMeasurement` slope average. HashMap iteration is
/// non-deterministic; sorting by this priority makes the resolved
/// slope reproducible across runs of the same configuration. Front
/// L/R lead because they are the most consistently positioned and
/// most-measured pair in any layout.
pub(super) fn bed_channel_priority(role: roomeq_engine::home_cinema::HomeCinemaRole) -> Option<u8> {
    use roomeq_engine::home_cinema::HomeCinemaRole as R;
    match role {
        R::FrontLeft => Some(0),
        R::FrontRight => Some(1),
        R::Center => Some(2),
        R::SideSurroundLeft => Some(3),
        R::SideSurroundRight => Some(4),
        R::RearSurroundLeft => Some(5),
        R::RearSurroundRight => Some(6),
        R::WideLeft => Some(7),
        R::WideRight => Some(8),
        _ => None,
    }
}

/// Resolve a single, system-wide `FromMeasurement` slope (dB/octave)
/// by averaging the regression slope across every available bed
/// channel.
///
/// Why averaging instead of "pick one":
/// - Asymmetric L/R in-room slopes of 1–2 dB/oct are common in real
///   rooms (one speaker near a corner, one in free space — Toole on
///   SBIR). Picking the first channel via HashMap iteration order
///   makes the resolved slope non-deterministic across runs of the
///   same room.
/// - Averaging across all bed channels reflects the room's
///   collective tonal balance — what Dirac, Audyssey, and ARC do
///   internally for their reference tilt estimation.
///
/// Channels backed by `Group`/`MultiSub`/`Dba`/`Cardioid` are skipped
/// here because their per-driver curves are not the right input for
/// a system-wide tilt regression. Sub/LFE channels are excluded — see
/// `detect_sub_passband_3db` for why their bandwidth makes per-channel
/// regression unreliable.
///
/// Picking rules:
/// 1. Average slopes from every bed channel (front L/R, center,
///    surrounds, wides) sorted by `bed_channel_priority`.
/// 2. If no bed channels, fall back to the first non-sub channel
///    (sorted alphabetically for determinism).
/// 3. If no curve is loadable / regressable, return 0.0 (flat).
pub(super) fn resolve_from_measurement_slope_with_frequency_samples(
    config: &RoomConfig,
    frequency_samples: usize,
) -> f64 {
    let mut bed_channels: Vec<(u8, &str, &MeasurementSource)> = Vec::new();
    let mut other_channels: Vec<(&str, &MeasurementSource)> = Vec::new();
    for (channel_name, speaker_config) in &config.speakers {
        let SpeakerConfig::Single(source) = speaker_config else {
            continue;
        };
        let role = roomeq_engine::home_cinema::role_for_channel(channel_name);
        if role.is_sub_or_lfe() {
            continue;
        }
        if let Some(prio) = bed_channel_priority(role) {
            bed_channels.push((prio, channel_name.as_str(), source));
        } else {
            other_channels.push((channel_name.as_str(), source));
        }
    }
    bed_channels.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(b.1)));
    other_channels.sort_by(|a, b| a.0.cmp(b.0));

    let mut slopes: Vec<f64> = Vec::with_capacity(bed_channels.len());
    for (_, name, source) in &bed_channels {
        match crate::measurement::load_source_with_frequency_samples(source, frequency_samples) {
            Ok(curve) => {
                if let Some(s) = slope::estimate_slope_db_per_octave(
                    &curve,
                    slope::DEFAULT_SLOPE_MIN_FREQ,
                    slope::DEFAULT_SLOPE_MAX_FREQ,
                ) {
                    info!(
                        "  FromMeasurement: '{}' contributes slope = {:.2} dB/octave",
                        name, s
                    );
                    slopes.push(s);
                } else {
                    debug!(
                        "  FromMeasurement: '{}' produced no valid slope — skipped from average",
                        name
                    );
                }
            }
            Err(e) => {
                debug!(
                    "  FromMeasurement: failed to load '{}' for slope averaging: {}",
                    name, e
                );
            }
        }
    }

    if !slopes.is_empty() {
        let avg = slopes.iter().sum::<f64>() / slopes.len() as f64;
        info!(
            "  FromMeasurement: averaged room-level slope = {:.2} dB/octave from {} bed channel(s)",
            avg,
            slopes.len()
        );
        return avg;
    }

    // No bed channels usable — fall back to the first non-sub channel
    // we can load (sorted for determinism).
    for (name, source) in &other_channels {
        match crate::measurement::load_source_with_frequency_samples(source, frequency_samples) {
            Ok(curve) => {
                let s = slope::estimate_slope_db_per_octave(
                    &curve,
                    slope::DEFAULT_SLOPE_MIN_FREQ,
                    slope::DEFAULT_SLOPE_MAX_FREQ,
                )
                .unwrap_or(0.0);
                info!(
                    "  FromMeasurement: fallback slope = {:.2} dB/octave from non-bed channel '{}'",
                    s, name
                );
                return s;
            }
            Err(e) => {
                debug!(
                    "  FromMeasurement: failed to load fallback '{}': {}",
                    name, e
                );
            }
        }
    }

    info!("  FromMeasurement: no usable reference channel — defaulting to 0.0 dB/octave");
    0.0
}

/// Identify Acoustic Groups from RoomConfig
///
/// Acoustic Groups are speakers expected to be acoustically similar (e.g., L/R pair).
/// Uses explicit speaker_name metadata if available, otherwise falls back to
/// positional heuristics (L/R, SL/SR, etc.).
pub(super) fn identify_acoustic_groups(config: &RoomConfig) -> HashMap<String, Vec<String>> {
    let mut groups: HashMap<String, Vec<String>> = HashMap::new();
    let mut positioned_channels: HashMap<String, String> = HashMap::new();

    // 1. Group by explicit speaker_name
    for (channel_name, speaker_cfg) in &config.speakers {
        if let Some(speaker_name) = speaker_cfg.speaker_name() {
            groups
                .entry(speaker_name.to_string())
                .or_default()
                .push(channel_name.clone());
        } else {
            positioned_channels.insert(channel_name.clone(), channel_name.clone());
        }
    }

    // 2. Positional heuristics for remaining channels
    let pairs = [
        ("L", "R"),
        ("SL", "SR"),
        ("SBL", "SBR"),
        ("TFL", "TFR"),
        ("TRL", "TRR"),
        ("FWL", "FWR"),
    ];

    for (p1, p2) in pairs {
        if positioned_channels.contains_key(p1) && positioned_channels.contains_key(p2) {
            let group_name = format!("{}-{}", p1, p2);
            let mut group = Vec::new();
            if let Some(c1) = positioned_channels.remove(p1) {
                group.push(c1);
            }
            if let Some(c2) = positioned_channels.remove(p2) {
                group.push(c2);
            }
            groups.insert(group_name, group);
        }
    }

    groups
}

pub(super) fn shared_target_level(channel_means: &[f64]) -> f64 {
    let mut finite_means: Vec<f64> = channel_means
        .iter()
        .copied()
        .filter(|mean| mean.is_finite())
        .collect();
    if finite_means.is_empty() {
        return 0.0;
    }

    finite_means.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mid = finite_means.len() / 2;
    if finite_means.len().is_multiple_of(2) {
        (finite_means[mid - 1] + finite_means[mid]) / 2.0
    } else {
        finite_means[mid]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;
    use roomeq_model::MeasurementSource;
    use roomeq_model::{InlineMeasurement, MeasurementRef, MeasurementSingle};
    use roomeq_model::{OptimizerConfig, SubwooferStrategy, SubwooferSystemConfig, SystemConfig};

    fn small_curve() -> roomeq_model::Curve {
        roomeq_model::Curve {
            freq: Array1::logspace(10.0, f64::log10(20.0), f64::log10(20_000.0), 32),
            spl: Array1::from_elem(32, 80.0),
            phase: None,
            ..Default::default()
        }
    }

    fn single_source() -> MeasurementSource {
        MeasurementSource::InMemory(small_curve())
    }

    fn room_config_with_speakers(speakers: HashMap<String, SpeakerConfig>) -> RoomConfig {
        RoomConfig {
            version: roomeq_model::default_config_version(),
            system: None,
            speakers,
            crossovers: None,
            target_curve: None,
            optimizer: OptimizerConfig::default(),
            provenance: Default::default(),
            recording_config: None,
            ctc: None,
            cea2034_cache: None,
        }
    }

    #[test]
    fn is_subwoofer_channel_by_name() {
        let config = room_config_with_speakers(HashMap::new());
        assert!(is_subwoofer_channel(&config, "lfe"));
        assert!(is_subwoofer_channel(&config, "Sub"));
        assert!(is_subwoofer_channel(&config, "subwoofer"));
        assert!(!is_subwoofer_channel(&config, "left"));
    }

    #[test]
    fn is_subwoofer_channel_by_system_mapping() {
        let mut speakers = HashMap::new();
        speakers.insert("left".to_string(), SpeakerConfig::Single(single_source()));
        speakers.insert("lfe".to_string(), SpeakerConfig::Single(single_source()));
        let config = RoomConfig {
            system: Some(SystemConfig {
                model: roomeq_model::SystemModel::HomeCinema,
                speakers: HashMap::from([
                    ("Left".to_string(), "left".to_string()),
                    ("LFE".to_string(), "lfe".to_string()),
                ]),
                subwoofers: Some(SubwooferSystemConfig {
                    config: SubwooferStrategy::Single,
                    crossover: None,
                    mapping: [("lfe".to_string(), "Left".to_string())].into(),
                }),
                bass_management: None,
                ..Default::default()
            }),
            ..room_config_with_speakers(speakers)
        };
        assert!(is_subwoofer_channel(&config, "lfe"));
        assert!(!is_subwoofer_channel(&config, "left"));
    }

    #[test]
    fn find_sub_main_pairings_explicit_system() {
        let mut speakers = HashMap::new();
        speakers.insert("left".to_string(), SpeakerConfig::Single(single_source()));
        speakers.insert("lfe".to_string(), SpeakerConfig::Single(single_source()));
        let config = RoomConfig {
            system: Some(SystemConfig {
                model: roomeq_model::SystemModel::HomeCinema,
                speakers: HashMap::from([
                    ("Left".to_string(), "left".to_string()),
                    ("LFE".to_string(), "lfe".to_string()),
                ]),
                subwoofers: Some(SubwooferSystemConfig {
                    config: SubwooferStrategy::Single,
                    crossover: None,
                    mapping: [("lfe".to_string(), "Left".to_string())].into(),
                }),
                bass_management: None,
                ..Default::default()
            }),
            ..room_config_with_speakers(speakers)
        };
        let curves = HashMap::from([
            ("left".to_string(), small_curve()),
            ("lfe".to_string(), small_curve()),
        ]);
        let pairings = find_sub_main_pairings(&config, &curves);
        assert_eq!(pairings, vec![("LFE".to_string(), "Left".to_string())]);
    }

    #[test]
    fn find_sub_main_pairings_legacy_heuristic() {
        let mut speakers = HashMap::new();
        speakers.insert("left".to_string(), SpeakerConfig::Single(single_source()));
        speakers.insert("sub".to_string(), SpeakerConfig::Single(single_source()));
        let config = room_config_with_speakers(speakers);
        let curves = HashMap::from([
            ("left".to_string(), small_curve()),
            ("sub".to_string(), small_curve()),
        ]);
        let mut pairings = find_sub_main_pairings(&config, &curves);
        pairings.sort();
        assert_eq!(pairings, vec![("sub".to_string(), "left".to_string())]);
    }

    #[test]
    fn pipeline_stopped_error_includes_step() {
        let err = pipeline_stopped_error(roomeq_engine::pipeline::PipelineStepId::Validation);
        let msg = format!("{:?}", err);
        assert!(msg.contains("Validation"));
        assert!(msg.contains("stopped by observer"));
    }

    #[test]
    fn optimizer_progress_iterations_is_positive() {
        let mut speakers = HashMap::new();
        speakers.insert("left".to_string(), SpeakerConfig::Single(single_source()));
        let mut config = room_config_with_speakers(speakers);
        config.optimizer.num_filters = 2;
        config.optimizer.max_iter = 1_000;
        config.optimizer.population = 10;
        let iters = optimizer_progress_iterations(&config);
        assert!(iters > 0);
    }

    #[test]
    fn channels_for_generic_optimization_without_system() {
        let mut speakers = HashMap::new();
        speakers.insert("left".to_string(), SpeakerConfig::Single(single_source()));
        speakers.insert("right".to_string(), SpeakerConfig::Single(single_source()));
        let config = room_config_with_speakers(speakers);
        let channels = channels_for_generic_optimization(&config);
        let names: Vec<_> = channels.iter().map(|(n, _)| n.clone()).collect();
        assert!(names.contains(&"left".to_string()));
        assert!(names.contains(&"right".to_string()));
    }

    #[test]
    fn channels_for_generic_optimization_with_system() {
        let mut speakers = HashMap::new();
        speakers.insert("left".to_string(), SpeakerConfig::Single(single_source()));
        speakers.insert("right".to_string(), SpeakerConfig::Single(single_source()));
        let config = RoomConfig {
            system: Some(SystemConfig {
                model: roomeq_model::SystemModel::Stereo,
                speakers: HashMap::from([
                    ("Left".to_string(), "left".to_string()),
                    ("Right".to_string(), "right".to_string()),
                ]),
                subwoofers: None,
                bass_management: None,
                ..Default::default()
            }),
            ..room_config_with_speakers(speakers)
        };
        let channels = channels_for_generic_optimization(&config);
        let names: Vec<_> = channels.iter().map(|(n, _)| n.clone()).collect();
        assert!(names.contains(&"Left".to_string()));
        assert!(names.contains(&"Right".to_string()));
    }

    #[test]
    fn compute_shared_mean_spl_with_two_channels() {
        let channels = vec![
            ("left".to_string(), SpeakerConfig::Single(single_source())),
            ("right".to_string(), SpeakerConfig::Single(single_source())),
        ];
        let config = room_config_with_speakers(HashMap::new());
        let mean = compute_shared_mean_spl_with_frequency_samples(
            &config,
            &channels,
            crate::DEFAULT_FREQUENCY_SAMPLES,
        );
        assert!(mean.is_some_and(|m| (m - 80.0).abs() < 1.0));
    }

    #[test]
    fn compute_shared_mean_spl_single_channel_returns_none() {
        let channels = vec![("left".to_string(), SpeakerConfig::Single(single_source()))];
        let config = room_config_with_speakers(HashMap::new());
        assert!(
            compute_shared_mean_spl_with_frequency_samples(
                &config,
                &channels,
                crate::DEFAULT_FREQUENCY_SAMPLES,
            )
            .is_none()
        );
    }

    #[test]
    fn generic_channel_progress_iterations_positive() {
        let mut speakers = HashMap::new();
        speakers.insert("left".to_string(), SpeakerConfig::Single(single_source()));
        let mut config = room_config_with_speakers(speakers);
        config.optimizer.num_filters = 1;
        config.optimizer.max_iter = 100;
        config.optimizer.population = 8;
        let iters = generic_channel_progress_iterations(&config);
        assert!(iters > 0);
    }

    #[test]
    fn bed_channel_priority_front_left_highest() {
        use roomeq_engine::home_cinema::HomeCinemaRole;
        assert_eq!(bed_channel_priority(HomeCinemaRole::FrontLeft), Some(0));
        assert_eq!(bed_channel_priority(HomeCinemaRole::Center), Some(2));
        assert_eq!(bed_channel_priority(HomeCinemaRole::Lfe), None);
    }

    #[test]
    fn resolve_from_measurement_slope_flat_bed_channels_is_zero() {
        let mut speakers = HashMap::new();
        speakers.insert("left".to_string(), SpeakerConfig::Single(single_source()));
        speakers.insert("right".to_string(), SpeakerConfig::Single(single_source()));
        speakers.insert("lfe".to_string(), SpeakerConfig::Single(single_source()));
        let config = room_config_with_speakers(speakers);
        let slope = resolve_from_measurement_slope_with_frequency_samples(
            &config,
            crate::DEFAULT_FREQUENCY_SAMPLES,
        );
        assert!(slope.abs() < 0.1, "expected ~0 dB/oct, got {}", slope);
    }

    #[test]
    fn identify_acoustic_groups_explicit_speaker_name() {
        let mut speakers = HashMap::new();
        speakers.insert(
            "left".to_string(),
            SpeakerConfig::Single(MeasurementSource::Single(MeasurementSingle {
                measurement: MeasurementRef::Inline(InlineMeasurement {
                    frequencies: vec![100.0],
                    magnitude_db: vec![80.0],
                    phase_deg: None,
                    name: None,
                    wav_path: None,
                    csv_path: None,
                }),
                speaker_name: Some("MySpeaker".to_string()),
            })),
        );
        speakers.insert(
            "right".to_string(),
            SpeakerConfig::Single(MeasurementSource::Single(MeasurementSingle {
                measurement: MeasurementRef::Inline(InlineMeasurement {
                    frequencies: vec![100.0],
                    magnitude_db: vec![80.0],
                    phase_deg: None,
                    name: None,
                    wav_path: None,
                    csv_path: None,
                }),
                speaker_name: Some("MySpeaker".to_string()),
            })),
        );
        let config = room_config_with_speakers(speakers);
        let groups = identify_acoustic_groups(&config);
        assert_eq!(groups.get("MySpeaker").map(|v| v.len()), Some(2));
    }

    #[test]
    fn identify_acoustic_groups_positional_lr_pair() {
        let mut speakers = HashMap::new();
        speakers.insert("L".to_string(), SpeakerConfig::Single(single_source()));
        speakers.insert("R".to_string(), SpeakerConfig::Single(single_source()));
        let config = room_config_with_speakers(speakers);
        let groups = identify_acoustic_groups(&config);
        assert!(groups.contains_key("L-R"));
    }

    #[test]
    fn shared_target_level_median_odd() {
        assert_eq!(shared_target_level(&[70.0, 80.0, 90.0]), 80.0);
    }

    #[test]
    fn shared_target_level_average_even() {
        assert_eq!(shared_target_level(&[70.0, 90.0]), 80.0);
    }

    #[test]
    fn shared_target_level_ignores_non_finite() {
        assert_eq!(shared_target_level(&[70.0, f64::NAN, 90.0]), 80.0);
    }

    #[test]
    fn shared_target_level_empty_returns_zero() {
        assert_eq!(shared_target_level(&[]), 0.0);
    }

    #[test]
    fn validate_room_config_or_fail_empty_speakers_fails() {
        let config = room_config_with_speakers(HashMap::new());
        assert!(
            validate_room_config_or_fail_with_frequency_samples(
                &config,
                crate::DEFAULT_FREQUENCY_SAMPLES,
            )
            .is_err()
        );
    }

    #[test]
    fn validate_room_config_or_fail_loads_file_sources_at_runtime_boundary() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("invalid-measurement.csv");
        std::fs::write(&path, "not,a,measurement\n").unwrap();
        let source = MeasurementSource::Single(MeasurementSingle {
            measurement: MeasurementRef::Path(path),
            speaker_name: None,
        });
        let config = room_config_with_speakers(HashMap::from([(
            "left".to_string(),
            SpeakerConfig::Single(source),
        )]));

        let error = validate_room_config_or_fail_with_frequency_samples(
            &config,
            crate::DEFAULT_FREQUENCY_SAMPLES,
        )
        .unwrap_err();
        assert!(error.to_string().contains("failed acoustic validation"));
    }
}
