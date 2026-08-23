//! Target-context preparation for one RoomEQ channel.

use autoeq_core::Curve;
use log::{debug, info, warn};
use roomeq_model::{RoomConfig, TargetCurveConfig, TargetShape, UserPreference};

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
    effective_target =
        roomeq_model::home_cinema::role_adjusted_target_response(channel_name, &effective_target);

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
        effective_target.shape = TargetShape::Custom;
        effective_target.slope_db_per_octave = measured_slope;
    }

    if effective_target.shape != TargetShape::Flat
        || effective_target.preference.bass_shelf_db.abs() > 1e-6
        || effective_target.preference.treble_shelf_db.abs() > 1e-6
        || roomeq_model::home_cinema::role_target_curve_shape_active(
            channel_name,
            &effective_target,
        )
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
        let mut target_curve =
            roomeq_model::target_tilt::build_complete_target_curve(&curve.freq, &effective_target);
        roomeq_model::target_tilt::apply_role_target_curve_shape(
            channel_name,
            &mut target_curve,
            &effective_target,
        );
        Some(target_curve)
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
    let cea2034_active = room_config
        .optimizer
        .cea2034_correction
        .as_ref()
        .is_some_and(|config| config.enabled);
    let target_tilt_curve =
        build_target_tilt_curve(channel_name, room_config, curve, cea2034_active);

    if target_tilt_curve.is_some() && room_config.target_curve.is_some() {
        warn!(
            "  Both target_curve and target_response are configured for '{}'. target_response is baked into the measurement; target_curve will be ignored to avoid double-application.",
            channel_name
        );
    }

    let min_freq = room_config.optimizer.min_freq;
    let max_freq = room_config.optimizer.max_freq;
    let pre_score = flatness_score_in_range(curve, min_freq, max_freq);
    let channel_mean_spl =
        roomeq_analysis::response_metrics::mean_response_in_range(curve, min_freq, max_freq);

    TargetContext {
        target_tilt_curve,
        min_freq,
        max_freq,
        pre_score,
        mean_spl: target_mean_spl(channel_name, channel_mean_spl, shared_mean_spl),
        cea2034_active,
    }
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
        let sub = build_target_tilt_curve("LFE", &room_config, &response, false).unwrap();
        assert!(sub.spl.iter().all(|value| value.abs() < 1e-9));

        room_config.optimizer.from_measurement_slope_override = Some(-1.25);
        let overridden = build_target_tilt_curve("left", &room_config, &response, false).unwrap();
        assert!(overridden.spl[0] > overridden.spl[overridden.spl.len() - 1]);
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
