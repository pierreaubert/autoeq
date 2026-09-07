use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// Configuration for a single supporting-source loudspeaker channel.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct SupportingSourceConfig {
    /// Delay of the supporting source relative to the primary at the listening position.
    #[serde(default = "default_support_delay_ms")]
    pub delay_ms: f64,

    /// Unfiltered support arrival minus primary arrival at the reference seat,
    /// measured in milliseconds using a common acquisition time reference.
    /// Electrical delay is `delay_ms - acoustic_arrival_offset_ms`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub acoustic_arrival_offset_ms: Option<f64>,

    /// Both transfer-function phases retain the same acquisition time reference.
    #[serde(default)]
    pub shared_phase_reference: bool,

    /// Explicit experimental opt-in when arrival or coherent-sum evidence is
    /// unavailable. Power-average design is not a coherent or perceptual guarantee.
    #[serde(default)]
    pub allow_unverified_acoustics: bool,

    /// Engineering budget for a coherent dip below the louder branch, not an
    /// audibility threshold.
    #[serde(default = "default_max_coherent_cancellation_db")]
    pub max_coherent_cancellation_db: f64,

    /// Frequency-dependent precedence limits. The first matching band wins.
    #[serde(default = "default_precedence_limits")]
    pub precedence_limits: Vec<PrecedenceLimitBand>,

    /// Compensation band. Outside this band the supporting source is muted.
    #[serde(default = "default_support_freq_range")]
    pub freq_range_hz: (f64, f64),

    /// Decorrelator applied to the supporting source path.
    #[serde(default)]
    pub decorrelation: SupportingSourceDecorrelation,

    /// Target response to match. Defaults to the room-level `target_curve`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub target_response: Option<String>,

    /// FIR length in taps. Default follows Brooks-Park (≈186 ms at 44.1 kHz).
    #[serde(default = "default_support_fir_taps")]
    pub fir_taps: usize,

    /// Velvet-noise sequence length when decorrelation is enabled.
    #[serde(default = "default_velvet_noise_taps")]
    pub velvet_noise_taps: usize,
}

fn default_max_coherent_cancellation_db() -> f64 {
    3.0
}

impl Default for SupportingSourceConfig {
    fn default() -> Self {
        Self {
            delay_ms: default_support_delay_ms(),
            acoustic_arrival_offset_ms: None,
            shared_phase_reference: false,
            allow_unverified_acoustics: false,
            max_coherent_cancellation_db: default_max_coherent_cancellation_db(),
            precedence_limits: default_precedence_limits(),
            freq_range_hz: default_support_freq_range(),
            decorrelation: SupportingSourceDecorrelation::default(),
            target_response: None,
            fir_taps: default_support_fir_taps(),
            velvet_noise_taps: default_velvet_noise_taps(),
        }
    }
}

/// A single precedence-limit band.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct PrecedenceLimitBand {
    /// Lower edge of the band in Hz (inclusive).
    pub low_hz: f64,
    /// Upper edge of the band in Hz (inclusive).
    pub high_hz: f64,
    /// Maximum lagging-source level above the primary in this band, in dB.
    pub limit_db: f64,
}

/// Decorrelation strategy for the supporting source path.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum SupportingSourceDecorrelation {
    /// Apply a velvet-noise sequence to the supporting source.
    #[default]
    VelvetNoise,
    /// No decorrelation.
    None,
}

fn default_support_delay_ms() -> f64 {
    10.0
}

fn default_support_freq_range() -> (f64, f64) {
    (70.0, 20000.0)
}

fn default_support_fir_taps() -> usize {
    8192
}

fn default_velvet_noise_taps() -> usize {
    4096
}

fn default_precedence_limits() -> Vec<PrecedenceLimitBand> {
    vec![
        PrecedenceLimitBand {
            low_hz: 70.0,
            high_hz: 500.0,
            limit_db: 10.0,
        },
        PrecedenceLimitBand {
            low_hz: 500.0,
            high_hz: 20000.0,
            limit_db: 6.0,
        },
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_match_brooks_park() {
        let cfg = SupportingSourceConfig::default();
        assert_eq!(cfg.delay_ms, 10.0);
        assert_eq!(cfg.freq_range_hz, (70.0, 20000.0));
        assert_eq!(cfg.fir_taps, 8192);
        assert_eq!(cfg.velvet_noise_taps, 4096);
        assert_eq!(cfg.precedence_limits.len(), 2);
        assert_eq!(cfg.precedence_limits[0].limit_db, 10.0);
        assert_eq!(cfg.precedence_limits[1].limit_db, 6.0);
    }

    #[test]
    fn json_roundtrip() {
        let cfg = SupportingSourceConfig::default();
        let json = serde_json::to_string(&cfg).unwrap();
        let back: SupportingSourceConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(back.delay_ms, cfg.delay_ms);
        assert_eq!(back.fir_taps, cfg.fir_taps);
        assert_eq!(back.decorrelation, cfg.decorrelation);
    }

    #[test]
    fn deserialize_partial() {
        let json = r#"{"delay_ms": 12.5, "fir_taps": 4096}"#;
        let cfg: SupportingSourceConfig = serde_json::from_str(json).unwrap();
        assert_eq!(cfg.delay_ms, 12.5);
        assert_eq!(cfg.fir_taps, 4096);
        assert_eq!(cfg.freq_range_hz, (70.0, 20000.0));
    }
}
