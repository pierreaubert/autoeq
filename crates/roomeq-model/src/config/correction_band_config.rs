use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// Optional, explicit frequency range in which RoomEQ is allowed to shape
/// the response.
///
/// The optimizer's observation/evaluation band remains unchanged. When the
/// range is narrower than that band, bins outside it are retained in the
/// measured graph and scorecard; with `allow_natural_rolloff`, the existing
/// source response is allowed to remain there instead of being flattened.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct CorrectionBandPolicy {
    /// Lower edge of the correction support in Hz.
    pub min_hz: f64,
    /// Upper edge of the correction support in Hz.
    pub max_hz: f64,
    /// Permit measured response outside the correction support to remain
    /// naturally rolled off. This never installs a high-pass or low-pass.
    #[serde(default)]
    pub allow_natural_rolloff: bool,
}

impl CorrectionBandPolicy {
    /// Validate the policy against the configured observation band.
    pub fn validate_against(
        &self,
        configured_min_hz: f64,
        configured_max_hz: f64,
    ) -> Result<(), String> {
        if !self.min_hz.is_finite()
            || !self.max_hz.is_finite()
            || self.min_hz <= 0.0
            || self.max_hz <= self.min_hz
        {
            return Err(
                "correction_band min_hz/max_hz must be finite with 0 < min_hz < max_hz".into(),
            );
        }
        if !configured_min_hz.is_finite()
            || !configured_max_hz.is_finite()
            || configured_min_hz <= 0.0
            || configured_max_hz <= configured_min_hz
        {
            return Err("optimizer observation band is invalid".into());
        }
        if self.min_hz < configured_min_hz || self.max_hz > configured_max_hz {
            return Err(format!(
                "correction_band [{:.1}, {:.1}] Hz must lie inside optimizer observation band [{:.1}, {:.1}] Hz",
                self.min_hz, self.max_hz, configured_min_hz, configured_max_hz
            ));
        }
        if !self.allow_natural_rolloff
            && (self.min_hz > configured_min_hz || self.max_hz < configured_max_hz)
        {
            return Err(
                "narrow correction_band requires allow_natural_rolloff=true (otherwise it would silently leave scored bins uncorrected)"
                    .into(),
            );
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn full_observation_band_is_legacy_compatible() {
        let policy = CorrectionBandPolicy {
            min_hz: 20.0,
            max_hz: 20_000.0,
            allow_natural_rolloff: false,
        };
        assert!(policy.validate_against(20.0, 20_000.0).is_ok());
    }

    #[test]
    fn narrowed_band_requires_explicit_natural_rolloff() {
        let policy = CorrectionBandPolicy {
            min_hz: 40.0,
            max_hz: 16_000.0,
            allow_natural_rolloff: false,
        };
        let error = policy.validate_against(20.0, 20_000.0).unwrap_err();
        assert!(error.contains("allow_natural_rolloff"));

        assert!(
            CorrectionBandPolicy {
                allow_natural_rolloff: true,
                ..policy
            }
            .validate_against(20.0, 20_000.0)
            .is_ok()
        );
    }

    #[test]
    fn policy_cannot_escape_observation_band() {
        let policy = CorrectionBandPolicy {
            min_hz: 10.0,
            max_hz: 16_000.0,
            allow_natural_rolloff: true,
        };
        assert!(policy.validate_against(20.0, 20_000.0).is_err());
    }

    #[test]
    fn serde_round_trip_keeps_rolloff_opt_in() {
        let config = crate::OptimizerConfig {
            min_freq: 20.0,
            max_freq: 20_000.0,
            correction_band: Some(CorrectionBandPolicy {
                min_hz: 40.0,
                max_hz: 16_000.0,
                allow_natural_rolloff: true,
            }),
            ..crate::OptimizerConfig::default()
        };
        let value = serde_json::to_value(&config).expect("optimizer config serializes");
        assert_eq!(value["correction_band"]["min_hz"], 40.0);
        assert_eq!(value["correction_band"]["allow_natural_rolloff"], true);
        let decoded: crate::OptimizerConfig = serde_json::from_value(value).unwrap();
        assert_eq!(decoded.active_correction_band(), [40.0, 16_000.0]);
    }

    #[test]
    fn room_validation_rejects_implicit_narrowing() {
        let mut room = crate::RoomConfig::default();
        room.speakers.insert(
            "left".into(),
            crate::SpeakerConfig::Single(crate::MeasurementSource::InMemory(crate::Curve {
                freq: ndarray::arr1(&[20.0, 20_000.0]),
                spl: ndarray::arr1(&[0.0, 0.0]),
                ..Default::default()
            })),
        );
        room.optimizer.correction_band = Some(CorrectionBandPolicy {
            min_hz: 40.0,
            max_hz: 16_000.0,
            allow_natural_rolloff: false,
        });
        let report = crate::validation_rules::validate_room_config(&room);
        assert!(
            report
                .errors
                .iter()
                .any(|error| error.contains("correction_band"))
        );
    }
}
