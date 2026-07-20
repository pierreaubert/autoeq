pub use super::types::*;
use crate::Curve;
use roomeq_model::{MeasurementSource, SpatialRobustnessSerdeConfig, SpeakerConfig};

pub fn group_id_for_role(role: HomeCinemaRole) -> &'static str {
    match role {
        HomeCinemaRole::FrontLeft | HomeCinemaRole::FrontRight | HomeCinemaRole::Center => "lcr",
        HomeCinemaRole::SideSurroundLeft
        | HomeCinemaRole::SideSurroundRight
        | HomeCinemaRole::RearSurroundLeft
        | HomeCinemaRole::RearSurroundRight => "surround",
        HomeCinemaRole::WideLeft | HomeCinemaRole::WideRight => "wide",
        role if role.is_height() => "height",
        HomeCinemaRole::Lfe => "lfe",
        HomeCinemaRole::Subwoofer => "sub",
        HomeCinemaRole::Unknown => "unknown",
        _ => "unknown",
    }
}

pub(super) fn optimization_group_result<'a>(
    optimization: Option<&'a BassManagementOptimizationReport>,
    group_id: &str,
) -> Option<&'a BassManagementGroupReport> {
    optimization?
        .group_results
        .iter()
        .find(|group| group.group_id == group_id)
}

pub fn is_linear_phase_crossover_type(crossover_type: &str) -> bool {
    matches!(
        crossover_type.to_ascii_lowercase().as_str(),
        "linearphase" | "linear_phase" | "linear-phase" | "linearphasefir" | "fir" | "lpfir"
    )
}

pub fn linear_to_db(value: f64) -> f64 {
    if !value.is_finite() || value < 0.0 {
        return f64::NAN;
    }
    20.0 * value.max(1e-12).log10()
}

pub fn home_cinema_role_sort_index(role: HomeCinemaRole) -> usize {
    match role {
        HomeCinemaRole::FrontLeft => 0,
        HomeCinemaRole::FrontRight => 1,
        HomeCinemaRole::Center => 2,
        HomeCinemaRole::Lfe | HomeCinemaRole::Subwoofer => 3,
        HomeCinemaRole::SideSurroundLeft => 4,
        HomeCinemaRole::SideSurroundRight => 5,
        HomeCinemaRole::RearSurroundLeft => 6,
        HomeCinemaRole::RearSurroundRight => 7,
        HomeCinemaRole::WideLeft => 8,
        HomeCinemaRole::WideRight => 9,
        HomeCinemaRole::TopFrontLeft => 10,
        HomeCinemaRole::TopFrontRight => 11,
        HomeCinemaRole::TopMiddleLeft => 12,
        HomeCinemaRole::TopMiddleRight => 13,
        HomeCinemaRole::TopRearLeft => 14,
        HomeCinemaRole::TopRearRight => 15,
        HomeCinemaRole::Unknown => 99,
    }
}

pub fn limited_sub_gain(
    requested_gain_db: f64,
    bass_management: Option<&EffectiveBassManagement>,
) -> (f64, bool) {
    let Some(bm) = bass_management else {
        return (requested_gain_db, false);
    };
    let with_trim = requested_gain_db + bm.config.sub_trim_db;
    let max_boost = bm.config.max_sub_boost_db.max(0.0);
    if with_trim > max_boost {
        (max_boost, true)
    } else {
        (with_trim, false)
    }
}

pub fn default_all_channel_spatial_robustness() -> SpatialRobustnessSerdeConfig {
    SpatialRobustnessSerdeConfig {
        variance_threshold_db: 3.0,
        transition_width_db: 2.0,
        min_correction_depth: 0.1,
        mask_smoothing_octaves: 1.0 / 6.0,
    }
}

pub fn spatial_robustness_config_from(
    config: &SpatialRobustnessSerdeConfig,
) -> roomeq_analysis::spatial_robustness::SpatialRobustnessConfig {
    roomeq_analysis::spatial_robustness::SpatialRobustnessConfig {
        variance_threshold_db: config.variance_threshold_db,
        transition_width_db: config.transition_width_db,
        min_correction_depth: config.min_correction_depth,
        mask_smoothing_octaves: config.mask_smoothing_octaves,
    }
}

pub fn single_measurement_source(speaker: &SpeakerConfig) -> Option<&MeasurementSource> {
    match speaker {
        SpeakerConfig::Single(source) => Some(source),
        _ => None,
    }
}

pub fn curves_share_frequency_grid(curves: &[Curve]) -> bool {
    let Some(first) = curves.first() else {
        return true;
    };
    curves.iter().all(|curve| {
        curve.freq.len() == first.freq.len()
            && curve
                .freq
                .iter()
                .zip(first.freq.iter())
                .all(|(a, b)| (a - b).abs() <= (1e-6 * b.abs().max(1.0)))
    })
}

pub fn band_metrics(curve: &Curve, band_hz: (f64, f64)) -> Option<(f64, f64, f64, f64)> {
    let values: Vec<f64> = curve
        .freq
        .iter()
        .zip(curve.spl.iter())
        .filter_map(|(freq, spl)| {
            (*freq >= band_hz.0 && *freq <= band_hz.1 && spl.is_finite()).then_some(*spl)
        })
        .collect();
    if values.is_empty() {
        return None;
    }
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let deviations: Vec<f64> = values.iter().map(|v| v - mean).collect();
    let rms = (deviations.iter().map(|v| v * v).sum::<f64>() / deviations.len() as f64).sqrt();
    let max_abs = deviations.iter().map(|v| v.abs()).fold(0.0, f64::max);
    let min_dev = deviations.iter().cloned().fold(f64::INFINITY, f64::min);
    Some((rms, max_abs, min_dev, mean))
}

pub fn optional_max(values: impl Iterator<Item = f64>) -> Option<f64> {
    values.reduce(f64::max)
}

pub(super) fn detect_layout_name(
    bed_channels: usize,
    lfe_channels: usize,
    height_channels: usize,
) -> String {
    if height_channels > 0 {
        format!("{bed_channels}.{lfe_channels}.{height_channels}")
    } else {
        format!("{bed_channels}.{lfe_channels}")
    }
}

pub fn speaker_measurement_count(speaker: &SpeakerConfig) -> Option<usize> {
    match speaker {
        SpeakerConfig::SupportingSource(group) => [measurement_source_count(&group.primary)]
            .into_iter()
            .chain([measurement_source_count(&group.support)])
            .flatten()
            .max(),
        SpeakerConfig::Single(source) => measurement_source_count(source),
        SpeakerConfig::Group(group) => group
            .measurements
            .iter()
            .filter_map(measurement_source_count)
            .max(),
        SpeakerConfig::Topology(topology) => topology
            .drivers
            .iter()
            .filter_map(|driver| measurement_source_count(&driver.measurement))
            .max(),
        SpeakerConfig::MultiSub(group) => group
            .subwoofers
            .iter()
            .filter_map(measurement_source_count)
            .max(),
        SpeakerConfig::Dba(config) => config
            .front
            .iter()
            .chain(config.rear.iter())
            .filter_map(measurement_source_count)
            .max(),
        SpeakerConfig::Cardioid(config) => [measurement_source_count(&config.front)]
            .into_iter()
            .chain([measurement_source_count(&config.rear)])
            .flatten()
            .max(),
    }
}

pub fn measurement_source_count(source: &MeasurementSource) -> Option<usize> {
    match source {
        MeasurementSource::Single(_) | MeasurementSource::InMemory(_) => Some(1),
        MeasurementSource::Multiple(m) => Some(m.measurements.len()),
        MeasurementSource::InMemoryMultiple(curves) => Some(curves.len()),
    }
}
