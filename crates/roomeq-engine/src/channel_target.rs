//! Target-context preparation for one RoomEQ channel.

use autoeq_core::{AutoeqError, Curve, Result};
use log::{debug, info, warn};
use roomeq_model::{RoomConfig, TargetCurveConfig, TargetShape, UserPreference};

use crate::eq::PreparedEqTarget;

/// Prepared target state shared by channel preprocessing and optimization.
#[derive(Clone, Debug)]
pub struct TargetContext {
    pub target_tilt_curve: Option<Curve>,
    pub min_freq: f64,
    pub max_freq: f64,
    pub pre_score: f64,
    pub mean_spl: f64,
    pub cea2034_active: bool,
}

impl TargetContext {
    /// Configured target resource, unless an in-memory response tilt is already
    /// incorporated into the channel optimization curve.
    pub fn effective_target<'a>(
        &self,
        room_config: &'a RoomConfig,
    ) -> Option<&'a TargetCurveConfig> {
        if self.target_tilt_curve.is_some() {
            None
        } else {
            room_config.target_curve.as_ref()
        }
    }
}

/// Build the channel's target response, including role-specific shaping.
pub fn build_target_tilt_curve(
    channel_name: &str,
    room_config: &RoomConfig,
    curve: &Curve,
    _cea2034_active: bool,
) -> Option<Curve> {
    let target_response = room_config.optimizer.target_response.as_ref()?;

    // User voicing is always a separately bypassable output-filter layer and
    // must not contaminate the neutral correction objective or quality score.
    let mut effective_target = target_response.clone();
    effective_target.preference = UserPreference::default();
    effective_target.role_targets = None;
    if effective_target.shape == TargetShape::Harman {
        effective_target.shape = TargetShape::Flat;
    }
    // Role voicing is realized exclusively by channel_preference filters.
    // Keep it out of the neutral optimization target to avoid double application.

    if effective_target.shape == TargetShape::FromMeasurement {
        let is_sub_or_lfe =
            roomeq_model::home_cinema::role_for_channel(channel_name).is_sub_or_lfe();
        let measured_slope = if let Some(override_slope) =
            room_config.optimizer.from_measurement_slope_override
        {
            info!(
                "  FromMeasurement: using room-level slope = {:.2} dB/octave (resolved from reference channel) for '{}'",
                override_slope, channel_name
            );
            override_slope
        } else if is_sub_or_lfe {
            info!(
                "  FromMeasurement: '{}' is band-limited (sub/LFE) and no reference slope is available — defaulting to flat (0.0 dB/octave)",
                channel_name
            );
            0.0
        } else {
            let slope = roomeq_analysis::slope::estimate_slope_db_per_octave(
                curve,
                roomeq_analysis::slope::DEFAULT_SLOPE_MIN_FREQ,
                roomeq_analysis::slope::DEFAULT_SLOPE_MAX_FREQ,
            )
            .unwrap_or(0.0);
            info!(
                "  FromMeasurement: estimated slope = {:.2} dB/octave from '{}'",
                slope, channel_name
            );
            slope
        };
        if measured_slope.abs() <= 1e-9 {
            effective_target.shape = TargetShape::Flat;
            effective_target.slope_db_per_octave = 0.0;
        } else {
            effective_target.shape = TargetShape::Custom;
            effective_target.slope_db_per_octave = measured_slope;
        }
    }

    if effective_target.shape != TargetShape::Flat
        || effective_target.preference.bass_shelf_db.abs() > 1e-6
        || effective_target.preference.treble_shelf_db.abs() > 1e-6
    {
        info!(
            "  Building target curve: shape={:?}, slope={:.2} dB/oct, bass={:+.1}dB, treble={:+.1}dB{}",
            effective_target.shape,
            match effective_target.shape {
                TargetShape::Harman => -0.8,
                TargetShape::Custom => effective_target.slope_db_per_octave,
                _ => 0.0,
            },
            effective_target.preference.bass_shelf_db,
            effective_target.preference.treble_shelf_db,
            " (preferences extracted to output layer)",
        );
        Some(roomeq_model::target_tilt::build_complete_target_curve(
            &curve.freq,
            &effective_target,
        ))
    } else {
        None
    }
}

/// Build the initial target context before feature preprocessing.
pub fn build_target_context(
    channel_name: &str,
    room_config: &RoomConfig,
    curve: &Curve,
    shared_mean_spl: Option<f64>,
) -> TargetContext {
    build_target_context_with_prepared_target(
        channel_name,
        room_config,
        curve,
        shared_mean_spl,
        None,
    )
    .expect("non-file target context must be constructible without a prepared resource")
}

/// Build a target context with a workflow-resolved file target.
pub fn build_target_context_with_prepared_target(
    channel_name: &str,
    room_config: &RoomConfig,
    curve: &Curve,
    shared_mean_spl: Option<f64>,
    prepared_target: Option<&PreparedEqTarget>,
) -> Result<TargetContext> {
    let cea2034_active = room_config
        .optimizer
        .cea2034_correction
        .as_ref()
        .is_some_and(|config| config.enabled);
    let target_tilt_curve = if room_config
        .optimizer
        .target_response
        .as_ref()
        .is_some_and(|target| target.shape == TargetShape::File)
    {
        let Some(PreparedEqTarget::Curve(target)) = prepared_target else {
            return Err(AutoeqError::InvalidConfiguration {
                message: format!(
                    "channel '{channel_name}' uses target_response.shape='file' but its target curve was not prepared"
                ),
            });
        };
        if target.freq.is_empty() || target.spl.is_empty() {
            return Err(AutoeqError::InvalidConfiguration {
                message: format!("channel '{channel_name}' has an empty file target"),
            });
        }
        let mut aligned = autoeq_core::interpolate_log_space(&curve.freq, target);
        let target_min = target.freq[0];
        let target_max = target.freq[target.freq.len() - 1];
        let low_value = target.spl[0];
        let high_value = target.spl[target.spl.len() - 1];
        for (&frequency, value) in curve.freq.iter().zip(aligned.spl.iter_mut()) {
            if frequency < target_min {
                *value = low_value;
            } else if frequency > target_max {
                *value = high_value;
            }
        }
        aligned.phase = None;
        Some(aligned)
    } else {
        build_target_tilt_curve(channel_name, room_config, curve, cea2034_active)
    };

    if target_tilt_curve.is_some() && room_config.target_curve.is_some() {
        warn!(
            "  Both target_curve and target_response are configured for '{}'. target_response is baked into the measurement; target_curve will be ignored to avoid double-application.",
            channel_name
        );
    }

    let min_freq = room_config.optimizer.min_freq;
    let max_freq = room_config.optimizer.max_freq;
    let pre_score = target_tilt_curve
        .as_ref()
        .map(|tilt| {
            let neutral = Curve {
                freq: curve.freq.clone(),
                spl: &curve.spl - &tilt.spl,
                phase: curve.phase.clone(),
                ..Curve::default()
            };
            flatness_score_in_range(&neutral, min_freq, max_freq)
        })
        .unwrap_or_else(|| flatness_score_in_range(curve, min_freq, max_freq));
    let channel_mean_spl =
        roomeq_analysis::response_metrics::mean_response_in_range(curve, min_freq, max_freq);

    Ok(TargetContext {
        target_tilt_curve,
        min_freq,
        max_freq,
        pre_score,
        mean_spl: target_mean_spl(channel_name, channel_mean_spl, shared_mean_spl),
        cea2034_active,
    })
}

pub fn flatness_score_in_range(curve: &Curve, min_freq: f64, max_freq: f64) -> f64 {
    let mean = roomeq_analysis::response_metrics::mean_response_in_range(curve, min_freq, max_freq);
    let normalized_spl = &curve.spl - mean;
    autoeq_optim::loss::flat_loss(&curve.freq, &normalized_spl, min_freq, max_freq)
}

pub fn target_mean_spl(
    channel_name: &str,
    channel_mean_spl: f64,
    shared_mean_spl: Option<f64>,
) -> f64 {
    if let Some(shared) = shared_mean_spl {
        debug!(
            "  Using shared target level {:.1} dB (channel mean was {:.1} dB, delta {:.1} dB)",
            shared,
            channel_mean_spl,
            shared - channel_mean_spl
        );
        shared
    } else {
        debug!(
            "  Using channel '{}' target level {:.1} dB",
            channel_name, channel_mean_spl
        );
        channel_mean_spl
    }
}

#[cfg(test)]
mod tests {
    use ndarray::Array1;
    use roomeq_model::{Cea2034CorrectionConfig, OptimizerConfig, TargetResponseConfig};

    use super::*;

    fn curve() -> Curve {
        Curve {
            freq: Array1::logspace(10.0, f64::log10(20.0), f64::log10(20_000.0), 96),
            spl: Array1::from_elem(96, 80.0),
            ..Curve::default()
        }
    }

    fn config(target_response: TargetResponseConfig) -> RoomConfig {
        RoomConfig {
            optimizer: OptimizerConfig {
                min_freq: 20.0,
                max_freq: 500.0,
                target_response: Some(target_response),
                ..OptimizerConfig::default()
            },
            ..RoomConfig::default()
        }
    }

    #[test]
    fn flat_context_has_no_tilt_and_honors_shared_mean() {
        let context = build_target_context(
            "left",
            &config(TargetResponseConfig::default()),
            &curve(),
            Some(82.0),
        );
        assert!(context.target_tilt_curve.is_none());
        assert_eq!((context.min_freq, context.max_freq), (20.0, 500.0));
        assert_eq!(context.mean_spl, 82.0);
        assert!(context.pre_score.abs() < 0.1);
    }

    #[test]
    fn harman_is_extracted_while_custom_neutral_target_keeps_expected_direction() {
        let response = curve();
        let harman = build_target_tilt_curve(
            "left",
            &config(TargetResponseConfig {
                shape: TargetShape::Harman,
                ..TargetResponseConfig::default()
            }),
            &response,
            false,
        );
        assert!(
            harman.is_none(),
            "Harman house curve must be extracted from the neutral optimizer target"
        );

        let custom = build_target_tilt_curve(
            "left",
            &config(TargetResponseConfig {
                shape: TargetShape::Custom,
                slope_db_per_octave: -1.5,
                ..TargetResponseConfig::default()
            }),
            &response,
            false,
        )
        .unwrap();
        assert!(custom.spl[0] > custom.spl[custom.spl.len() - 1]);
    }

    #[test]
    fn from_measurement_uses_flat_sub_fallback_or_room_override() {
        let response = curve();
        let mut room_config = config(TargetResponseConfig {
            shape: TargetShape::FromMeasurement,
            ..TargetResponseConfig::default()
        });
        assert!(build_target_tilt_curve("LFE", &room_config, &response, false).is_none());

        room_config.optimizer.from_measurement_slope_override = Some(-1.25);
        let overridden = build_target_tilt_curve("left", &room_config, &response, false).unwrap();
        assert!(overridden.spl[0] > overridden.spl[overridden.spl.len() - 1]);
    }

    #[test]
    fn file_target_context_uses_prepared_curve_and_requires_resource() {
        let response = curve();
        let room_config = config(TargetResponseConfig {
            shape: TargetShape::File,
            curve_path: Some("target.csv".into()),
            ..TargetResponseConfig::default()
        });
        let prepared = PreparedEqTarget::Curve(Box::new(Curve {
            freq: Array1::from_vec(vec![20.0, 20_000.0]),
            spl: Array1::from_vec(vec![3.0, -3.0]),
            ..Curve::default()
        }));

        let context = build_target_context_with_prepared_target(
            "left",
            &room_config,
            &response,
            None,
            Some(&prepared),
        )
        .expect("prepared file target");
        let tilt = context.target_tilt_curve.expect("file tilt");
        assert!((tilt.spl[0] - 3.0).abs() < 1e-9);
        assert!((tilt.spl[tilt.spl.len() - 1] + 3.0).abs() < 1e-9);

        let error =
            build_target_context_with_prepared_target("left", &room_config, &response, None, None)
                .expect_err("missing file target must fail");
        assert!(error.to_string().contains("was not prepared"));
    }

    #[test]
    fn file_target_context_clamps_outside_points_to_endpoint_levels() {
        let response = curve();
        let room_config = config(TargetResponseConfig {
            shape: TargetShape::File,
            curve_path: Some("target.csv".into()),
            ..TargetResponseConfig::default()
        });
        let prepared = PreparedEqTarget::Curve(Box::new(Curve {
            freq: Array1::from_vec(vec![100.0, 1_000.0]),
            spl: Array1::from_vec(vec![10.0, 20.0]),
            ..Curve::default()
        }));

        let context = build_target_context_with_prepared_target(
            "left",
            &room_config,
            &response,
            None,
            Some(&prepared),
        )
        .expect("prepared file target");
        let target = context.target_tilt_curve.expect("file target");

        assert_eq!(target.spl[0], 10.0);
        assert_eq!(target.spl[target.spl.len() - 1], 20.0);
    }

    #[test]
    fn house_curve_and_user_preferences_are_extracted_from_neutral_target() {
        let response = curve();
        let mut room_config = config(TargetResponseConfig {
            shape: TargetShape::Harman,
            preference: UserPreference {
                bass_shelf_db: 2.0,
                ..UserPreference::default()
            },
            ..TargetResponseConfig::default()
        });
        assert!(build_target_tilt_curve("left", &room_config, &response, false).is_none());
        assert!(build_target_tilt_curve("left", &room_config, &response, true).is_none());

        room_config.optimizer.cea2034_correction = Some(Cea2034CorrectionConfig {
            enabled: true,
            ..Cea2034CorrectionConfig::default()
        });
        assert!(build_target_context("left", &room_config, &response, None).cea2034_active);
    }
}
