//! Role-aware alignment for overhead/height channels.

use crate::Curve;
use crate::error::{AutoeqError, Result};
use std::collections::HashMap;

use super::home_cinema::{HomeCinemaRole, HomeCinemaRoleGroup, role_for_channel};
use super::inter_channel_timbre_matching::{
    apply_alignment_to_curve, pairwise_normalized_timbre_spread_db,
};
use super::spectral_align::{SpectralAlignmentResult, compute_target_alignment};
use super::types::HeightChannelAlignmentConfig;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HeightAlignmentStatus {
    Applied,
    Skipped,
    Failed,
}

#[derive(Debug, Clone)]
pub struct HeightChannelAlignmentResult {
    pub channel_name: String,
    pub reference_channel: Option<String>,
    pub alignment: Option<SpectralAlignmentResult>,
    pub delay_ms: f64,
    pub timbre_spread_before_db: Option<f64>,
    pub timbre_spread_after_db: Option<f64>,
    pub level_difference_before_db: Option<f64>,
    pub level_difference_after_db: Option<f64>,
    pub phase_difference_before_deg: Option<f64>,
    pub phase_difference_after_deg: Option<f64>,
    pub phase_objective_evaluated: bool,
    pub status: HeightAlignmentStatus,
    pub advisories: Vec<String>,
}

fn role_group_key(role: HomeCinemaRole) -> &'static str {
    match role.group() {
        HomeCinemaRoleGroup::TopFront => "top_front",
        HomeCinemaRoleGroup::TopMiddle => "top_middle",
        HomeCinemaRoleGroup::TopRear => "top_rear",
        _ => "height",
    }
}

fn preferred_reference_roles(role: HomeCinemaRole) -> &'static [HomeCinemaRole] {
    use HomeCinemaRole::*;
    match role {
        TopFrontLeft => &[FrontLeft],
        TopFrontRight => &[FrontRight],
        TopMiddleLeft => &[SideSurroundLeft, FrontLeft],
        TopMiddleRight => &[SideSurroundRight, FrontRight],
        TopRearLeft => &[RearSurroundLeft, SideSurroundLeft, FrontLeft],
        TopRearRight => &[RearSurroundRight, SideSurroundRight, FrontRight],
        _ => &[],
    }
}

fn resolve_reference(
    channel_name: &str,
    role: HomeCinemaRole,
    curves: &HashMap<String, Curve>,
    config: &HeightChannelAlignmentConfig,
) -> Option<String> {
    if let Some(reference) = config
        .reference_channels
        .get(channel_name)
        .or_else(|| config.reference_channels.get(role_group_key(role)))
    {
        return curves.contains_key(reference).then(|| reference.clone());
    }
    for &preferred in preferred_reference_roles(role) {
        if let Some(name) = curves
            .iter()
            .filter(|(name, _)| role_for_channel(name) == preferred)
            .map(|(name, _)| name)
            .min()
        {
            return Some(name.clone());
        }
    }
    None
}

fn mean_level_difference_db(
    channel: &Curve,
    reference: &Curve,
    min_freq: f64,
    max_freq: f64,
) -> Option<f64> {
    if !super::frequency_grid::is_valid_frequency_grid(&channel.freq)
        || !super::frequency_grid::is_valid_frequency_grid(&reference.freq)
        || channel.spl.len() != channel.freq.len()
        || reference.spl.len() != reference.freq.len()
    {
        return None;
    }
    let overlap_min = min_freq.max(channel.freq[0]).max(reference.freq[0]);
    let overlap_max = max_freq
        .min(channel.freq[channel.freq.len() - 1])
        .min(reference.freq[reference.freq.len() - 1]);
    if overlap_max <= overlap_min {
        return None;
    }
    let reference = if super::frequency_grid::same_frequency_grid(&channel.freq, &reference.freq) {
        reference.clone()
    } else {
        crate::read::interpolate_log_space(&channel.freq, reference)
    };
    let differences = channel
        .freq
        .iter()
        .enumerate()
        .filter(|(_, frequency)| **frequency >= overlap_min && **frequency <= overlap_max)
        .filter_map(|(index, _)| {
            let channel = channel.spl[index];
            let reference = reference.spl[index];
            (channel.is_finite() && reference.is_finite()).then_some(channel - reference)
        })
        .collect::<Vec<_>>();
    (!differences.is_empty()).then(|| differences.iter().sum::<f64>() / differences.len() as f64)
}

fn has_trustworthy_phase(curve: &Curve, min_freq: f64, max_freq: f64) -> bool {
    let (Some(phase), Some(coherence)) = (&curve.phase, &curve.coherence) else {
        return false;
    };
    if phase.len() != curve.freq.len()
        || coherence.len() != curve.freq.len()
        || phase.iter().any(|value| !value.is_finite())
        || coherence.iter().any(|value| !value.is_finite())
    {
        return false;
    }
    let values = curve
        .freq
        .iter()
        .zip(coherence)
        .filter(|(frequency, _)| **frequency >= min_freq && **frequency <= max_freq)
        .map(|(_, coherence)| *coherence)
        .collect::<Vec<_>>();
    !values.is_empty()
        && values.iter().sum::<f64>() / values.len() as f64
            >= super::bass_phase_confidence::DEFAULT_COHERENCE_THRESHOLD
}

fn mean_phase_difference_deg(
    channel: &Curve,
    reference: &Curve,
    min_freq: f64,
    max_freq: f64,
) -> Option<f64> {
    let channel_phase = channel.phase.as_ref()?;
    let reference_phase = reference.phase.as_ref()?;
    let overlap_min = min_freq.max(channel.freq[0]).max(reference.freq[0]);
    let overlap_max = max_freq
        .min(channel.freq[channel.freq.len() - 1])
        .min(reference.freq[reference.freq.len() - 1]);
    if overlap_max <= overlap_min {
        return None;
    }
    let mut unwrapped_reference = reference.clone();
    unwrapped_reference.phase = Some(crate::loss::phase_aware::unwrap_phase_degrees(
        reference_phase,
    ));
    let reference = if super::frequency_grid::same_frequency_grid(&channel.freq, &reference.freq) {
        unwrapped_reference
    } else {
        crate::read::interpolate_log_space(&channel.freq, &unwrapped_reference)
    };
    let unwrapped_channel = crate::loss::phase_aware::unwrap_phase_degrees(channel_phase);
    let reference_phase = reference.phase.as_ref()?;
    let differences = channel
        .freq
        .iter()
        .enumerate()
        .filter(|(_, frequency)| **frequency >= overlap_min && **frequency <= overlap_max)
        .map(|(index, _)| {
            let wrapped = (unwrapped_channel[index] - reference_phase[index]).rem_euclid(360.0);
            wrapped.min(360.0 - wrapped)
        })
        .collect::<Vec<_>>();
    (!differences.is_empty()).then(|| differences.iter().sum::<f64>() / differences.len() as f64)
}

#[allow(clippy::too_many_arguments)]
pub fn compute_height_channel_alignment(
    corrected_curves: &HashMap<String, Curve>,
    arrivals_ms: &HashMap<String, f64>,
    config: &HeightChannelAlignmentConfig,
    sample_rate: f64,
    min_freq: f64,
    max_freq: f64,
) -> Result<HashMap<String, HeightChannelAlignmentResult>> {
    if !sample_rate.is_finite() || sample_rate <= 0.0 {
        return Err(AutoeqError::InvalidConfiguration {
            message: format!("height alignment requires positive sample rate, got {sample_rate}"),
        });
    }
    if !min_freq.is_finite() || !max_freq.is_finite() || min_freq <= 0.0 || max_freq <= min_freq {
        return Err(AutoeqError::InvalidConfiguration {
            message: format!(
                "height alignment requires 0 < min_freq < max_freq, got {min_freq}..{max_freq}"
            ),
        });
    }
    if !config.min_timbre_improvement_db.is_finite() || config.min_timbre_improvement_db < 0.0 {
        return Err(AutoeqError::InvalidConfiguration {
            message: format!(
                "height alignment min_timbre_improvement_db must be finite and non-negative, got {}",
                config.min_timbre_improvement_db
            ),
        });
    }
    if !config.max_delay_ms.is_finite() || config.max_delay_ms <= 0.0 {
        return Err(AutoeqError::InvalidConfiguration {
            message: format!(
                "height alignment max_delay_ms must be finite and positive, got {}",
                config.max_delay_ms
            ),
        });
    }
    if !(config.match_timbre || config.match_level || config.match_arrival_time) {
        return Err(AutoeqError::InvalidConfiguration {
            message: "height alignment requires at least one actionable timbre, level, or arrival-time objective"
                .to_string(),
        });
    }
    let mut results = HashMap::new();
    for (channel_name, channel_curve) in corrected_curves {
        let role = role_for_channel(channel_name);
        if !role.is_height() {
            continue;
        }
        let Some(reference_channel) =
            resolve_reference(channel_name, role, corrected_curves, config)
        else {
            results.insert(
                channel_name.clone(),
                HeightChannelAlignmentResult {
                    channel_name: channel_name.clone(),
                    reference_channel: None,
                    alignment: None,
                    delay_ms: 0.0,
                    timbre_spread_before_db: None,
                    timbre_spread_after_db: None,
                    level_difference_before_db: None,
                    level_difference_after_db: None,
                    phase_difference_before_deg: None,
                    phase_difference_after_deg: None,
                    phase_objective_evaluated: false,
                    status: HeightAlignmentStatus::Failed,
                    advisories: vec!["height_reference_not_found".to_string()],
                },
            );
            continue;
        };
        let reference_curve = &corrected_curves[&reference_channel];
        let timbre_before = pairwise_normalized_timbre_spread_db(
            channel_curve,
            reference_curve,
            min_freq,
            max_freq,
        );
        let level_before =
            mean_level_difference_db(channel_curve, reference_curve, min_freq, max_freq);
        let phase_objective_evaluated = config.match_phase
            && has_trustworthy_phase(channel_curve, min_freq, max_freq)
            && has_trustworthy_phase(reference_curve, min_freq, max_freq);
        let phase_before = phase_objective_evaluated
            .then(|| mean_phase_difference_deg(channel_curve, reference_curve, min_freq, max_freq))
            .flatten();
        let candidate = (config.match_timbre || config.match_level)
            .then(|| {
                compute_target_alignment(
                    channel_curve,
                    reference_curve,
                    min_freq,
                    max_freq,
                    sample_rate,
                )
            })
            .flatten();
        let had_candidate = candidate.is_some();
        let (alignment, timbre_after, level_after, phase_after) = if let Some(candidate) = candidate
        {
            let corrected = apply_alignment_to_curve(channel_curve, &candidate, sample_rate);
            let timbre_after = pairwise_normalized_timbre_spread_db(
                &corrected,
                reference_curve,
                min_freq,
                max_freq,
            );
            let level_after =
                mean_level_difference_db(&corrected, reference_curve, min_freq, max_freq);
            let timbre_pass = !config.match_timbre
                || timbre_before
                    .zip(timbre_after)
                    .is_some_and(|(before, after)| {
                        before - after >= config.min_timbre_improvement_db
                    });
            let level_pass = !config.match_level
                || level_before
                    .zip(level_after)
                    .is_some_and(|(before, after)| after.abs() <= before.abs() + 1e-6);
            let phase_after = phase_objective_evaluated
                .then(|| mean_phase_difference_deg(&corrected, reference_curve, min_freq, max_freq))
                .flatten();
            let phase_pass = !phase_objective_evaluated
                || phase_before
                    .zip(phase_after)
                    .is_some_and(|(before, after)| after <= before + 1.0);
            (
                (timbre_pass && level_pass && phase_pass).then_some(candidate),
                timbre_after,
                level_after,
                phase_after,
            )
        } else {
            (None, timbre_before, level_before, phase_before)
        };

        let mut advisories = Vec::new();
        if had_candidate && alignment.is_none() {
            advisories.push("height_objective_acceptance_failed".to_string());
        }
        let delay_ms = if config.match_arrival_time {
            match (
                arrivals_ms.get(channel_name),
                arrivals_ms.get(&reference_channel),
            ) {
                (Some(channel_arrival), Some(reference_arrival)) => {
                    let needed = reference_arrival - channel_arrival;
                    if needed > 0.01 && needed <= config.max_delay_ms {
                        needed
                    } else {
                        if needed < -0.01 {
                            advisories.push("height_arrives_after_reference".to_string());
                        } else if needed > config.max_delay_ms {
                            advisories.push("height_delay_limit_exceeded".to_string());
                        }
                        0.0
                    }
                }
                _ => {
                    advisories.push("height_arrival_data_missing".to_string());
                    0.0
                }
            }
        } else {
            0.0
        };
        if config.match_phase && !phase_objective_evaluated {
            advisories.push("height_phase_or_coherence_untrustworthy".to_string());
        }
        let status = if alignment.is_some() || delay_ms > 0.01 {
            HeightAlignmentStatus::Applied
        } else {
            HeightAlignmentStatus::Skipped
        };
        results.insert(
            channel_name.clone(),
            HeightChannelAlignmentResult {
                channel_name: channel_name.clone(),
                reference_channel: Some(reference_channel),
                alignment,
                delay_ms,
                timbre_spread_before_db: timbre_before,
                timbre_spread_after_db: timbre_after,
                level_difference_before_db: level_before,
                level_difference_after_db: level_after,
                phase_difference_before_deg: phase_before,
                phase_difference_after_deg: phase_after,
                phase_objective_evaluated,
                status,
                advisories,
            },
        );
    }
    Ok(results)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;

    fn curve(bass_db: f64, level_db: f64, phase: bool) -> Curve {
        let freq = Array1::from(vec![80.0, 160.0, 500.0, 2_000.0, 8_000.0, 16_000.0]);
        Curve {
            spl: freq.mapv(|frequency| level_db + if frequency < 300.0 { bass_db } else { 0.0 }),
            phase: phase.then(|| freq.mapv(|frequency| -frequency * 0.01)),
            freq,
            ..Default::default()
        }
    }

    #[test]
    fn aligns_height_channel_to_role_appropriate_bed_reference() {
        let curves = HashMap::from([
            ("L".to_string(), curve(0.0, 0.0, true)),
            ("TFL".to_string(), curve(3.0, 2.0, true)),
        ]);
        let arrivals = HashMap::from([("L".to_string(), 5.0), ("TFL".to_string(), 3.0)]);
        let result = compute_height_channel_alignment(
            &curves,
            &arrivals,
            &HeightChannelAlignmentConfig {
                enabled: true,
                min_timbre_improvement_db: 0.0,
                ..Default::default()
            },
            48_000.0,
            80.0,
            16_000.0,
        )
        .unwrap();
        let height = &result["TFL"];
        assert_eq!(height.reference_channel.as_deref(), Some("L"));
        assert_eq!(height.status, HeightAlignmentStatus::Applied);
        assert!(height.alignment.is_some());
        assert!((height.delay_ms - 2.0).abs() < 1e-9);
        assert!(height.timbre_spread_after_db.unwrap() < height.timbre_spread_before_db.unwrap());
    }

    #[test]
    fn reports_missing_height_reference() {
        let curves = HashMap::from([("TFL".to_string(), curve(0.0, 0.0, false))]);
        let result = compute_height_channel_alignment(
            &curves,
            &HashMap::new(),
            &HeightChannelAlignmentConfig::default(),
            48_000.0,
            80.0,
            16_000.0,
        )
        .unwrap();
        assert_eq!(result["TFL"].status, HeightAlignmentStatus::Failed);
    }

    #[test]
    fn invalid_reference_override_does_not_silently_fall_back() {
        let curves = HashMap::from([
            ("L".to_string(), curve(0.0, 0.0, false)),
            ("TFL".to_string(), curve(0.0, 0.0, false)),
        ]);
        let result = compute_height_channel_alignment(
            &curves,
            &HashMap::new(),
            &HeightChannelAlignmentConfig {
                reference_channels: HashMap::from([("TFL".to_string(), "missing".to_string())]),
                ..Default::default()
            },
            48_000.0,
            80.0,
            16_000.0,
        )
        .unwrap();

        assert_eq!(result["TFL"].status, HeightAlignmentStatus::Failed);
        assert_eq!(result["TFL"].reference_channel, None);
    }

    #[test]
    fn optional_phase_objective_requires_coherence() {
        let curves = HashMap::from([
            ("L".to_string(), curve(0.0, 0.0, true)),
            ("TFL".to_string(), curve(2.0, 0.0, true)),
        ]);
        let result = compute_height_channel_alignment(
            &curves,
            &HashMap::new(),
            &HeightChannelAlignmentConfig {
                match_arrival_time: false,
                match_phase: true,
                min_timbre_improvement_db: 0.0,
                ..Default::default()
            },
            48_000.0,
            80.0,
            16_000.0,
        )
        .unwrap();

        assert!(!result["TFL"].phase_objective_evaluated);
        assert!(
            result["TFL"]
                .advisories
                .contains(&"height_phase_or_coherence_untrustworthy".to_string())
        );
    }

    #[test]
    fn optional_phase_objective_reports_trustworthy_evidence() {
        let mut bed = curve(0.0, 0.0, true);
        bed.coherence = Some(Array1::ones(bed.freq.len()));
        let mut height = curve(2.0, 0.0, true);
        height.coherence = Some(Array1::ones(height.freq.len()));
        let curves = HashMap::from([("L".to_string(), bed), ("TFL".to_string(), height)]);
        let result = compute_height_channel_alignment(
            &curves,
            &HashMap::new(),
            &HeightChannelAlignmentConfig {
                match_arrival_time: false,
                match_phase: true,
                min_timbre_improvement_db: 0.0,
                ..Default::default()
            },
            48_000.0,
            80.0,
            16_000.0,
        )
        .unwrap();

        assert!(result["TFL"].phase_objective_evaluated);
        assert!(result["TFL"].phase_difference_before_deg.is_some());
        assert!(result["TFL"].phase_difference_after_deg.is_some());
    }
}
