//! Neutral, serializable optimizer settings owned by the RoomEQ contract.
//!
//! Runtime adapters into concrete optimizer and measurement types live in
//! `roomeq-engine`; this module must remain backend- and I/O-independent.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// Psychoacoustic variable smoothing configuration
///
/// Different frequency ranges benefit from different smoothing levels:
/// - Low frequencies (< 100 Hz): Fine resolution (1/48 octave) to preserve room modes
/// - High frequencies (> 1 kHz): Coarse resolution (1/6 octave) to ignore comb filtering
/// - Transition region (100 Hz - 1 kHz): Gradual interpolation between the two
#[derive(Debug, Clone, Copy, Serialize, Deserialize, JsonSchema, PartialEq)]
pub struct PsychoacousticSmoothingConfig {
    /// Smoothing resolution below low_freq (bands per octave, e.g., 48 for 1/48 octave)
    pub low_freq_n: usize,
    /// Smoothing resolution above high_freq (bands per octave, e.g., 6 for 1/6 octave)
    pub high_freq_n: usize,
    /// Lower transition frequency in Hz (default: 100 Hz)
    pub low_freq: f64,
    /// Upper transition frequency in Hz (default: 1000 Hz)
    pub high_freq: f64,
}

impl Default for PsychoacousticSmoothingConfig {
    fn default() -> Self {
        Self {
            low_freq_n: 48,
            high_freq_n: 6,
            low_freq: 100.0,
            high_freq: 1_000.0,
        }
    }
}

/// Configuration for asymmetric loss weighting.
///
/// Weights apply per sample as a multiplier on the squared error. The
/// peak / dip split uses a sigmoid crossfade in log frequency around
/// `transition_freq` so the transition from bass weighting to mid/treble
/// weighting is smooth. Narrow-null suppression is a separate mask passed
/// to the loss function at call time (see
/// [`crate::roomeq::impulse_analysis::build_null_suppression_mask`]).
#[derive(Debug, Clone, Copy, Serialize, Deserialize, JsonSchema, PartialEq)]
pub struct AsymmetricLossConfig {
    /// Weight for positive errors (peaks above `transition_freq`). Default: 2.0
    pub peak_weight: f64,
    /// Weight for negative errors (dips above `transition_freq`). Default: 1.0
    pub dip_weight: f64,
    /// Weight for bass peaks (below `transition_freq`). Default: 5.0
    pub bass_peak_weight: f64,
    /// Weight for bass dips (below `transition_freq`). Default: 1.0
    ///
    /// Historically this defaulted to 0.2 (near-ignore) as a crude
    /// proxy for "do not fight acoustic nulls". With explicit narrow-null
    /// suppression in place, broad bass dips (SBIR, baffle step) become
    /// legitimate correction targets and the dip weight is aligned with
    /// the mid/treble default.
    pub bass_dip_weight: f64,
    /// Transition frequency between bass and mid/treble weighting. Default: 300.0 Hz
    pub transition_freq: f64,
}

impl Default for AsymmetricLossConfig {
    fn default() -> Self {
        Self {
            peak_weight: 2.0,
            dip_weight: 1.0,
            bass_peak_weight: 5.0,
            bass_dip_weight: 1.0,
            transition_freq: 300.0,
        }
    }
}

/// Frequency band configuration for weighted loss
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct FrequencyBandWeights {
    /// Bass band minimum frequency (Hz)
    pub bass_min: f64,
    /// Bass band maximum frequency (Hz)
    pub bass_max: f64,
    /// Midrange band minimum frequency (Hz)
    pub mid_min: f64,
    /// Midrange band maximum frequency (Hz)
    pub mid_max: f64,
    /// Treble band minimum frequency (Hz)
    pub treble_min: f64,
    /// Treble band maximum frequency (Hz)
    pub treble_max: f64,
    /// Weight for bass band (default: 2.0 - bass is more critical for room correction)
    pub bass_weight: f64,
    /// Weight for midrange band (default: 1.0)
    pub mid_weight: f64,
    /// Weight for treble band (default: 0.8 - less critical for room issues)
    pub treble_weight: f64,
}

impl Default for FrequencyBandWeights {
    fn default() -> Self {
        Self {
            bass_min: 20.0,
            bass_max: 200.0,
            mid_min: 200.0,
            mid_max: 4_000.0,
            treble_min: 4_000.0,
            treble_max: 20_000.0,
            bass_weight: 2.0,
            mid_weight: 1.0,
            treble_weight: 0.8,
        }
    }
}

/// Program-material bias for temporal masking.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum TemporalMaskingProfile {
    /// Percussive material: modal ringing is least masked and should be cut
    /// more decisively.
    Transient,
    /// General music / film content.
    #[default]
    Mixed,
    /// Sustained material: late modal decay is partly masked by ongoing tone.
    Sustained,
}

fn default_temporal_masking_enabled() -> bool {
    true
}
fn default_temporal_masking_weight() -> f64 {
    0.15
}
fn default_ir_temporal_masking_enabled() -> bool {
    true
}
fn default_ir_temporal_masking_weight() -> f64 {
    0.05
}
fn default_pre_mask_ms() -> f64 {
    3.0
}
fn default_post_mask_ms() -> f64 {
    120.0
}
fn default_pre_ringing_weight() -> f64 {
    2.0
}
fn default_post_ringing_weight() -> f64 {
    1.0
}
fn default_ir_audibility_threshold_db() -> f64 {
    -45.0
}

/// Temporal masking penalty configuration for EPA optimization.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct TemporalMaskingConfig {
    /// Enable the temporal masking penalty when modal data is available.
    #[serde(default = "default_temporal_masking_enabled")]
    pub enabled: bool,
    /// Weight applied to the normalized temporal masking penalty.
    #[serde(default = "default_temporal_masking_weight")]
    pub weight: f64,
    /// Material profile used to scale modal-ringing audibility.
    #[serde(default)]
    pub profile: TemporalMaskingProfile,
    /// Enable true FIR impulse-response pre/post masking analysis when FIR
    /// coefficients are available.
    #[serde(default = "default_ir_temporal_masking_enabled")]
    pub ir_enabled: bool,
    /// Weight applied to the IR masking penalty metric.
    #[serde(default = "default_ir_temporal_masking_weight")]
    pub ir_weight: f64,
    /// Pre-masking window before the main impulse. Pre-ringing inside this
    /// window is partially masked; earlier energy is fully audible.
    #[serde(default = "default_pre_mask_ms")]
    pub pre_mask_ms: f64,
    /// Post-masking window after the main impulse. Ringing grows more audible
    /// as it decays beyond this window.
    #[serde(default = "default_post_mask_ms")]
    pub post_mask_ms: f64,
    /// Relative weight for audible pre-ringing. Usually higher than post
    /// because pre-echo before a transient is especially objectionable.
    #[serde(default = "default_pre_ringing_weight")]
    pub pre_ringing_weight: f64,
    /// Relative weight for audible post-ringing.
    #[serde(default = "default_post_ringing_weight")]
    pub post_ringing_weight: f64,
    /// Audibility floor for weighted pre/post ringing energy, in dB relative
    /// to the main impulse peak.
    #[serde(default = "default_ir_audibility_threshold_db")]
    pub ir_audibility_threshold_db: f64,
}

impl Default for TemporalMaskingConfig {
    fn default() -> Self {
        Self {
            enabled: default_temporal_masking_enabled(),
            weight: default_temporal_masking_weight(),
            profile: TemporalMaskingProfile::Mixed,
            ir_enabled: default_ir_temporal_masking_enabled(),
            ir_weight: default_ir_temporal_masking_weight(),
            pre_mask_ms: default_pre_mask_ms(),
            post_mask_ms: default_post_mask_ms(),
            pre_ringing_weight: default_pre_ringing_weight(),
            post_ringing_weight: default_post_ringing_weight(),
            ir_audibility_threshold_db: default_ir_audibility_threshold_db(),
        }
    }
}

fn default_flatness_erb_weight() -> f64 {
    1.0
}

/// Configuration for EPA scoring.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct EpaConfig {
    /// Listening level in phon (affects loudness computation)
    pub listening_level_phon: f64,
    /// Target sharpness in acum (1.0 = natural broadband noise character)
    pub target_sharpness: f64,
    /// Maximum acceptable roughness (above this, penalty increases)
    pub max_roughness: f64,
    /// Weights for the three EPA dimensions in the composite score
    pub evaluation_weight: f64,
    pub potency_weight: f64,
    pub activity_weight: f64,
    /// Band weights used for the flatness component of the EPA loss.
    /// Only consulted when `flatness_band_weight > 0`.
    #[serde(default)]
    pub flatness_band_weights: FrequencyBandWeights,
    /// ERB weight for the flatness component of the EPA loss.
    /// Default 1.0 (pure ERB) because EPA already has band-sensitive
    /// sharpness / roughness / loudness_balance terms — adding band
    /// weighting on top of flatness would double-count frequency bias.
    #[serde(default = "default_flatness_erb_weight")]
    pub flatness_erb_weight: f64,
    /// Band weight for the flatness component of the EPA loss.
    /// Default 0.0 (see `flatness_erb_weight`).
    #[serde(default)]
    pub flatness_band_weight: f64,
    /// Temporal-masking penalties for modal ringing and FIR phase behavior.
    ///
    /// Modal data is used as an optimizer-cheap proxy for post-masked room
    /// decay audibility. When FIR coefficients are exported, the FIR impulse
    /// response is also analyzed directly for pre/post ringing audibility.
    #[serde(default)]
    pub temporal_masking: TemporalMaskingConfig,
}

impl Default for EpaConfig {
    fn default() -> Self {
        Self {
            listening_level_phon: 75.0,
            target_sharpness: 1.2,
            max_roughness: 0.5,
            evaluation_weight: 0.6,
            potency_weight: 0.2,
            activity_weight: 0.2,
            flatness_band_weights: FrequencyBandWeights::default(),
            flatness_erb_weight: 1.0,
            flatness_band_weight: 0.0,
            temporal_masking: TemporalMaskingConfig::default(),
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, JsonSchema, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum MultiMeasurementStrategy {
    #[default]
    Average,
    WeightedSum,
    Minimax,
    VariancePenalized,
    SpatialRobustness,
    MinimaxUncertainty,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub enum CrossoverType {
    Butterworth2,
    LinkwitzRiley2,
    #[default]
    #[serde(alias = "LR24")]
    LinkwitzRiley4,
    #[serde(alias = "LR48")]
    LinkwitzRiley8,
    #[serde(alias = "LinearPhase")]
    LinearPhase,
    None,
}

impl std::str::FromStr for CrossoverType {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value.to_lowercase().as_str() {
            "butterworth2" | "bw2" | "butterworth12" | "bw12" => Ok(Self::Butterworth2),
            "lr2" | "lr12" | "linkwitzriley2" | "linkwitzriley12" => Ok(Self::LinkwitzRiley2),
            "lr4" | "lr24" | "linkwitzriley4" | "linkwitzriley24" => Ok(Self::LinkwitzRiley4),
            "lr8" | "lr48" | "linkwitzriley8" | "linkwitzriley48" => Ok(Self::LinkwitzRiley8),
            "linearphase" | "linear_phase" | "linear-phase" | "linearphasefir" | "fir"
            | "lpfir" => Ok(Self::LinearPhase),
            "none" => Ok(Self::None),
            _ => Err(format!("Unknown crossover type: {value}")),
        }
    }
}
