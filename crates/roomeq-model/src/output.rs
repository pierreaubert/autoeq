//! Room EQ Output Types
//!
//! Types for returning optimization results and DSP chain outputs.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize, Serializer};
use std::collections::{BTreeMap, HashMap};

// Re-export Curve for reference in output docs
pub use crate::Curve;

// ============================================================================
// Frequency Response Curve Data
// ============================================================================

/// Frequency response curve data for serialization
///
/// Represents a curve with frequency points and SPL values.
/// Conversion preserves the source level; it does not normalize captures.
/// Display normalization, when present, is identified by `norm_range`.
#[derive(Debug, Clone, Default, Serialize, Deserialize, JsonSchema)]
pub struct CurveData {
    /// Frequency points in Hz
    pub freq: Vec<f64>,
    /// Sound pressure level in dB, in the source curve's reference convention.
    pub spl: Vec<f64>,
    /// Phase in degrees (optional)
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        serialize_with = "serialize_wrapped_phase"
    )]
    pub phase: Option<Vec<f64>>,
    /// Optional frequency range used for normalization
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub norm_range: Option<(f64, f64)>,
    /// Measured noise floor on this grid, in the same level reference as `spl`.
    /// A level-reference shift must shift this array by the same amount.
    /// Absent for legacy curves or when capture noise was not measured.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub noise_floor_db: Option<Vec<f64>>,
    /// Measured coherence on this grid; absence is not perfect coherence.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub coherence: Option<Vec<f64>>,
}

fn wrap_phase_degrees(phase: f64) -> f64 {
    if phase.is_finite() {
        (phase + 180.0).rem_euclid(360.0) - 180.0
    } else {
        phase
    }
}

fn serialize_wrapped_phase<S>(phase: &Option<Vec<f64>>, serializer: S) -> Result<S::Ok, S::Error>
where
    S: Serializer,
{
    phase
        .as_ref()
        .map(|values| {
            values
                .iter()
                .copied()
                .map(wrap_phase_degrees)
                .collect::<Vec<_>>()
        })
        .serialize(serializer)
}

impl From<Curve> for CurveData {
    fn from(curve: Curve) -> Self {
        CurveData {
            freq: curve.freq.to_vec(),
            spl: curve.spl.to_vec(),
            phase: curve.phase.map(|p| p.to_vec()),
            norm_range: None,
            noise_floor_db: curve.noise_floor_db.map(|values| values.to_vec()),
            coherence: curve.coherence.map(|values| values.to_vec()),
        }
    }
}

impl From<&Curve> for CurveData {
    fn from(curve: &Curve) -> Self {
        CurveData {
            freq: curve.freq.to_vec(),
            spl: curve.spl.to_vec(),
            phase: curve.phase.as_ref().map(|p| p.to_vec()),
            norm_range: None,
            noise_floor_db: curve.noise_floor_db.as_ref().map(|values| values.to_vec()),
            coherence: curve.coherence.as_ref().map(|values| values.to_vec()),
        }
    }
}

impl From<CurveData> for Curve {
    fn from(data: CurveData) -> Self {
        Curve {
            freq: ndarray::Array1::from(data.freq),
            spl: ndarray::Array1::from(data.spl),
            phase: data.phase.map(ndarray::Array1::from),
            noise_floor_db: data.noise_floor_db.map(ndarray::Array1::from),
            coherence: data.coherence.map(ndarray::Array1::from),
            ..Default::default()
        }
    }
}

// ============================================================================
// Impulse Response Waveform
// ============================================================================

pub use crate::IrWaveform;

// ============================================================================
// DSP Chain Types
// ============================================================================

/// Backwards-compatible name for the canonical DSP execution/export graph.
pub type DspChainOutput = crate::DspGraph;

/// Third-octave early/late energy report for one channel's measured room IR.
///
/// The viewer renders this field only when `method`, `reference`,
/// `smoothing`, and `split_ms` carry the exact values below and the three
/// curves share one finite grid covering 1–8 kHz. Absence renders as
/// "pending", never as success.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct ChannelEarlyLateCurves {
    /// Energy-decomposition method; always `"incoherent_band_energy"`.
    pub method: String,
    /// Band-energy reference; always `"full_peak_band"`.
    pub reference: String,
    /// Frequency smoothing; always `"third_octave"`.
    pub smoothing: String,
    /// Early/late split after the direct reference, in milliseconds.
    pub split_ms: f64,
    /// Evidence basis; always `"measured_room_ir"`.
    pub basis: String,
    /// Direct-sound reference: `"broadband envelope peak"`, or
    /// `"120 Hz lowpass envelope peak"` for subwoofer/LFE channels.
    pub direct_reference: String,
    /// Third-octave coverage of the emitted curves in Hz.
    pub valid_band_hz: [f64; 2],
    /// Incoherent sum of early and late band energies (not a complex sum).
    pub full: CurveData,
    /// Band energy in the first 20 ms after the direct reference.
    pub early: CurveData,
    /// Band energy after the first 20 ms until the IR ends.
    pub late: CurveData,
}

/// One detected early reflection from a measured room IR.
///
/// Gains are dBFS relative to the 1–8 kHz filtered direct peak; times are
/// post-direct milliseconds. The viewer cross-checks distance against
/// 34.3 cm/ms and the first dip against 500/time_ms, and requires gains in
/// `[-15, 0]` dBFS within 15 ms.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct ChannelReflectionEvent {
    /// Event gain in dBFS relative to the filtered direct peak.
    pub gain_dbfs: f64,
    /// Post-direct arrival time in milliseconds.
    pub time_ms: f64,
    /// Path difference in centimeters.
    pub distance_cm: f64,
    /// First comb dip in Hz.
    pub first_dip_hz: f64,
    /// Comb ripple in dB; null when the gain equals direct (unbounded).
    pub ripple_db: Option<f64>,
}

/// Band-limited early-reflection table for one channel's measured room IR.
///
/// The viewer renders this field only when `method`, `band_hz`,
/// `threshold_dbfs`, and a nonempty `direct_reference` carry the exact
/// values below. Absence renders as pending, never as success.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct ChannelEarlyReflections {
    /// Evidence basis; always `"measured_room_ir"`.
    pub basis: String,
    /// Table method; always `"bandlimited_early_reflection_table_v1"`.
    pub method: String,
    /// Analysis band in Hz; always `[1000, 8000]`.
    pub band_hz: [f64; 2],
    /// Detection threshold in dBFS; always `-15`.
    pub threshold_dbfs: f64,
    /// Direct-sound reference description (nonempty).
    pub direct_reference: String,
    /// Pre-correction events from the measured IR.
    pub pre: Vec<ChannelReflectionEvent>,
    /// Post-correction events; empty until a post-correction IR is measured.
    pub post: Vec<ChannelReflectionEvent>,
}

/// One octave-band Schroeder/T60 fit from a measured room IR.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct ChannelT60Band {
    /// Octave centre frequency in Hz.
    pub centre_hz: f64,
    /// Selected T60 in seconds; null when invalid.
    pub t60_s: Option<f64>,
    /// Fit range that produced the value (`T30`/`T20`; also `EDT` on rows
    /// that stay invalid because EDT alone is not late-decay T60).
    pub fit_range: Option<String>,
    /// `r²` of the selected fit (0.0 when invalid).
    pub r2: f64,
    /// Whether the value passed the fit-range + `min_r2` policy.
    pub valid: bool,
    /// Machine-readable reason when invalid (nonempty).
    pub reason: String,
}

/// Nine-band octave T60 report for one channel's measured room IR.
///
/// The viewer requires all nine ordered bands with valid rows carrying
/// `T20`/`T30` fits above `min_r2`, and invalid rows carrying a nonempty
/// reason with no plotted value. Absence renders as pending.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct ChannelOctaveT60 {
    /// Evidence basis; always `"measured_room_ir"`.
    pub basis: String,
    /// Minimum fit `r²` for validity.
    pub min_r2: f64,
    /// Nine octave bands, 63 Hz–16 kHz in order.
    pub bands: Vec<ChannelT60Band>,
}

/// STFT waterfall decay grid for one channel's measured room IR (pre-
/// correction only; there is no post-correction IR at optimization time).
///
/// The viewer renders this field only when `basis`, `method`, and
/// `reference` carry the exact values below and the grid is a nonempty
/// finite `times × freqs` matrix matching `mags_db`. Absence renders as
/// pending, never as success.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct ChannelWaterfall {
    /// Evidence basis; always `"measured_room_ir"`.
    pub basis: String,
    /// Analysis method; always `"hann_stft_waterfall_v1"`.
    pub method: String,
    /// Level reference; always `"full_grid_peak"` (each grid is relative
    /// to its own peak, so two channels' grids do not compare absolutely).
    pub reference: String,
    /// Emitted frequency coverage in Hz.
    pub valid_band_hz: [f64; 2],
    /// STFT window in milliseconds; always `32`.
    pub window_ms: f64,
    /// STFT hop in milliseconds; always `2`.
    pub hop_ms: f64,
    /// Post-direct span in milliseconds; always `500`.
    pub post_ms: f64,
    /// Frame centre times relative to the direct peak in milliseconds.
    pub times_ms: Vec<f64>,
    /// Bin centre frequencies in Hz.
    pub freqs_hz: Vec<f64>,
    /// Magnitudes in dB relative to the grid peak, `mags_db[frame][bin]`.
    pub mags_db: Vec<Vec<f32>>,
    /// Machine-readable scope and non-comparison disclaimer.
    pub scope: String,
}

/// One prominent resonance decay slice from a measured room IR.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct ChannelResonanceDecay {
    /// Resonance frequency in Hz.
    pub freq_hz: f64,
    /// Level at the 60 ms slice in dB relative to the waterfall grid peak.
    pub level_db: f64,
    /// Fitted decay time in seconds; null when the bin has no fittable
    /// 20–200 ms decay.
    pub decay_time_s: Option<f64>,
}

/// Prominent-resonance decay slices for one channel's measured room IR.
///
/// The viewer renders this field only when `basis`, `method`, and
/// `reference` carry the exact values below. An empty `decays` list means
/// the analysis ran and found no prominent resonances, not a failure.
/// Absence of the field renders as pending.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct ChannelResonanceDecays {
    /// Evidence basis; always `"measured_room_ir"`.
    pub basis: String,
    /// Detection method; always `"hann_stft_waterfall_v1"`.
    pub method: String,
    /// Level reference; always `"full_grid_peak"`.
    pub reference: String,
    /// Slice offset after the direct peak in milliseconds; always `60`.
    pub slice_ms: f64,
    /// Detected resonances with fitted decay times.
    pub decays: Vec<ChannelResonanceDecay>,
}

/// Three-cycle wavelet heatmap for one channel's measured room IR (pre-
/// correction only).
///
/// The viewer renders this field only when `basis`, `method`, and
/// `reference` carry the exact values below and the heatmap is a nonempty
/// finite `freqs × times` matrix matching `mags_db` with at least two
/// frequency rows and two time columns. Absence renders as pending.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct ChannelWavelet {
    /// Evidence basis; always `"measured_room_ir"`.
    pub basis: String,
    /// Analysis method; always `"complex_morlet_three_cycle_v1"`.
    pub method: String,
    /// Level reference; always `"full_grid_peak"`.
    pub reference: String,
    /// Emitted frequency coverage in Hz.
    pub valid_band_hz: [f64; 2],
    /// Wavelet cycles; always `3`.
    pub cycles: f64,
    /// Log-grid density: `48` in dense measured exports, `6` in legacy reports.
    pub freqs_per_octave: f64,
    /// Minimum nominal hop in milliseconds; use `times_ms` for exact coordinates.
    /// Dense exports sample at 0.1 ms through 15 ms and 1 ms thereafter.
    pub hop_ms: f64,
    /// Display range in dB; always `[-30, 0]`.
    pub display_range_db: [f64; 2],
    /// Centre frequencies in Hz.
    pub freqs_hz: Vec<f64>,
    /// Frame centre times relative to the direct peak in milliseconds.
    pub times_ms: Vec<f64>,
    /// Magnitudes in dB relative to the grid peak, `mags_db[freq][time]`.
    pub mags_db: Vec<Vec<f32>>,
}

/// DSP chain for a single channel
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct ChannelDspChain {
    /// Channel name
    pub channel: String,
    /// Ordered list of plugins (AudioEngine PluginConfig format)
    pub plugins: Vec<PluginConfigWrapper>,
    /// Per-driver DSP chains for active crossover (optional)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub drivers: Option<Vec<DriverDspChain>>,
    /// Initial frequency response curve before optimization (optional)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub initial_curve: Option<CurveData>,
    /// Final frequency response curve after applying correction (optional)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub final_curve: Option<CurveData>,
    /// EQ filter response curve (correction magnitude in dB) (optional)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub eq_response: Option<CurveData>,
    /// Effective target curve the optimizer worked against (optional)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub target_curve: Option<CurveData>,
    /// Impulse response before correction (optional, requires phase data)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pre_ir: Option<IrWaveform>,
    /// Impulse response after correction (optional, requires phase data)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub post_ir: Option<IrWaveform>,
    /// FIR impulse-response temporal masking metrics (optional, FIR/phase modes)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fir_temporal_masking: Option<crate::TemporalIrMaskingMetrics>,
    /// Direct/early/late correction-energy metrics (optional, FIR/phase modes
    /// or any channel with phase-derived IRs).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub direct_early_late_correction: Option<DirectEarlyLateCorrectionMetrics>,
    /// Stage-bound joint subwoofer diagnostics, when that method ran.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub joint_sub: Option<crate::JointSubDiagnostics>,
    /// Third-octave early/late energy from the channel's measured room IR,
    /// when a complete IR was declared at optimization time.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub early_late_curves: Option<ChannelEarlyLateCurves>,
    /// Band-limited early-reflection table from the channel's measured room
    /// IR, when a complete IR was declared at optimization time.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub early_reflections: Option<ChannelEarlyReflections>,
    /// Nine-band octave T60 from the channel's measured room IR, when a
    /// complete IR was declared at optimization time.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub t60_octaves: Option<ChannelOctaveT60>,
    /// STFT waterfall decay grid from the channel's measured room IR, when
    /// a complete 500 ms post-peak window was declared at optimization
    /// time.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub waterfall: Option<ChannelWaterfall>,
    /// Prominent-resonance decay slices from the channel's measured room
    /// IR, accompanying `waterfall`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub resonance_decays: Option<ChannelResonanceDecays>,
    /// Three-cycle wavelet heatmap from the channel's measured room IR,
    /// when a complete post-peak window was declared at optimization time.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub wavelet: Option<ChannelWavelet>,
}

/// DSP chain for an individual driver in a multi-driver speaker
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct DriverDspChain {
    /// Driver name (e.g. "woofer", "tweeter")
    pub name: String,
    /// Driver index in the array (0 = lowest frequency)
    pub index: usize,
    /// Ordered list of plugins for this driver (gain, crossover filters)
    pub plugins: Vec<PluginConfigWrapper>,
    /// Initial frequency response curve for this driver before optimization (optional)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub initial_curve: Option<CurveData>,
    /// Measured frequency span of this driver's capture in Hz, when the
    /// stored display curve was extended past it for full-range DSP replay.
    /// Plots of the raw measurement should clip to this band; points outside
    /// it are trend continuation, not data.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub measured_band_hz: Option<[f64; 2]>,
}

/// Backend-neutral serialized plugin descriptor. Native adapters translate this
/// contract into their runtime types; the model does not depend on AudioEngine.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct PluginConfigWrapper {
    pub plugin_type: String,
    pub parameters: serde_json::Value,
}

/// Per-channel EPA psychoacoustic metrics computed on the initial
/// (pre-EQ) and final (post-EQ) frequency responses.
///
/// See [`crate::loss::epa::score::EpaScore`] for the individual fields.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct EpaChannelMetrics {
    /// EPA score computed from the initial (pre-EQ) response.
    pub pre: crate::EpaScore,
    /// EPA score computed from the final (post-EQ) response.
    pub post: crate::EpaScore,
}

/// Aggregate EPA metrics for the whole reproduced channel set.
///
/// Channels are combined with BS.1770-style energy weights before EPA scoring:
/// front/main channels use unit weight, surround channels use +1.5 dB energy
/// weight, and LFE/subwoofer channels are excluded.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct EpaMultichannelMetrics {
    /// EPA score computed from the aggregate initial (pre-EQ) response.
    pub pre: crate::EpaScore,
    /// EPA score computed from the aggregate final (post-EQ) response.
    pub post: crate::EpaScore,
    /// Human-readable aggregation standard/approximation.
    pub standard: String,
}

/// Correction-energy split across direct, early, and late IR windows.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct DirectEarlyLateCorrectionMetrics {
    /// Direct-sound window end in milliseconds.
    pub direct_window_ms: f64,
    /// Early-reflection window end in milliseconds.
    pub early_window_ms: f64,
    /// Late summary window end in milliseconds.
    pub late_window_ms: f64,
    /// Correction energy in the direct window, dB relative to total correction energy.
    pub direct_energy_db: f64,
    /// Correction energy in the early window, dB relative to total correction energy.
    pub early_energy_db: f64,
    /// Correction energy in the late window, dB relative to total correction energy.
    pub late_energy_db: f64,
    /// Direct + early correction energy, dB relative to total correction energy.
    pub direct_plus_early_energy_db: f64,
    /// Advisory when FIR/mixed-phase correction may be altering direct/early cues.
    pub advisory: String,
}

/// Resolved preference-layer metadata.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct PreferenceLayerReport {
    /// Optional house curve extracted from the neutral optimizer target.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub house_curve: Option<crate::TargetShape>,
    /// User bass/treble voicing realized after neutral correction.
    pub user_preference: crate::UserPreference,
    /// Optional role/content voicing realized after neutral correction.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub role_targets: Option<crate::RoleTargetConfig>,
    /// Preference filters are excluded from correction quality scores.
    pub excluded_from_neutral_quality_score: bool,
}

/// Resolved correction/preference policy metadata.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct PerceptualPolicyReport {
    /// Policy preset requested by the user or UI.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub preset: Option<crate::PerceptualPolicyPreset>,
    /// Effective loss type after policy resolution.
    pub loss_type: String,
    /// Effective target response after policy resolution.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub target_response: Option<crate::TargetResponseConfig>,
    /// Target used by the physical correction optimizer, with voicing stripped.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub neutral_target_response: Option<crate::TargetResponseConfig>,
    /// Separately bypassable post-correction listener/content voicing.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub preference_layer: Option<PreferenceLayerReport>,
    /// Effective audibility deadband.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub audibility_deadband: Option<crate::AudibilityDeadbandConfig>,
    /// Effective high-frequency safeguard.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub high_frequency_correction: Option<crate::HighFrequencyCorrectionConfig>,
}

/// Bootstrap uncertainty population.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum BootstrapUncertaintySource {
    /// Case-bootstrap over listening positions, not measurement noise.
    SpatialSeatSampling,
}

/// Bootstrap uncertainty reporting summary.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct BootstrapUncertaintyReport {
    /// Population whose variability is estimated.
    pub source: BootstrapUncertaintySource,
    /// Whether the estimator treats listening positions as independent cases.
    pub assumes_independent_positions: bool,
    /// Effective spatial sample size under the stated independence model.
    ///
    /// This equals the nominal seat count by default. A configured correlation
    /// adjustment may reduce it; this still is not a covariance/block model.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub effective_spatial_sample_size: Option<f64>,
    /// Repeat-sweep noise, absent unless separately measured.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub repeat_sweep_noise_std_db: Option<f64>,
    /// Calibration uncertainty, absent unless separately supplied.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub calibration_uncertainty_std_db: Option<f64>,
    /// Number of case-bootstrap resamples.
    pub num_resamples: usize,
    /// Two-sided alpha used for confidence bands.
    pub alpha: f64,
    /// Scalarisation used for uncertainty-aware optimization.
    pub scalarisation: crate::BootstrapScalarisation,
    /// CVaR tail fraction when scalarisation is CVaR.
    pub cvar_alpha: f64,
    /// Whether bootstrap confidence width was folded into correction-depth masks.
    pub used_for_correction_depth_mask: bool,
}

/// Continuous listening-area evaluation metrics.
///
/// All scalar metrics are `Option`: a variance (or loss) of exactly `0.0`
/// must mean "evaluated and found to be zero", never "not evaluated".
/// `None` means the quantity was not evaluated for this run (e.g. the
/// strategy was not `ContinuousArea`, or the producer predates area
/// metrics). Consumers must not substitute `0.0` for `None`.
///
/// Contract for producers (`roomeq-workflow`, `roomeq-engine`): serialize
/// this struct alongside the run output and echo the canonical seat IDs so
/// the seat correspondence is auditable. A future optional field on
/// `OptimizationMetadata` may carry it directly; until then it stands alone
/// as the additive output contract.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
pub struct ContinuousAreaMetrics {
    /// Canonical seat IDs (explicit [`SeatIdentityMap`](crate::SeatIdentityMap)
    /// IDs or the positional `seat-{i}` fallback), parallel to the evaluated
    /// seat order. `None` when the seat correspondence is unknown.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub seat_ids: Option<Vec<String>>,
    /// Number of quadrature points actually evaluated. `None` = not evaluated.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub num_quadrature_points: Option<usize>,
    /// Probability-weighted mean (expected) loss over the area.
    /// `None` = not evaluated.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub expected_loss: Option<f64>,
    /// Spatial variance of the per-point loss over the area.
    /// `None` = not evaluated (never default to `0.0`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub area_variance: Option<f64>,
    /// Worst-case per-point loss over the area's bounding box.
    /// `None` = not evaluated (only meaningful for worst-case scalarisation).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub worst_case_loss: Option<f64>,
    /// CVaR tail-mean loss. `None` = not evaluated (only meaningful for
    /// CVaR scalarisation).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cvar_loss: Option<f64>,
}

impl ContinuousAreaMetrics {
    /// Unevaluated metrics: every scalar is `None`.
    pub fn not_evaluated() -> Self {
        Self {
            seat_ids: None,
            num_quadrature_points: None,
            expected_loss: None,
            area_variance: None,
            worst_case_loss: None,
            cvar_loss: None,
        }
    }

    /// Whether any scalar metric was evaluated. A zero variance with
    /// `is_evaluated() == true` means "evaluated and zero".
    pub fn is_evaluated(&self) -> bool {
        self.expected_loss.is_some()
            || self.area_variance.is_some()
            || self.worst_case_loss.is_some()
            || self.cvar_loss.is_some()
    }
}

/// Validation/listening-test bundle descriptor.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct ValidationBundleReport {
    /// JSON artifact path.
    pub artifact: String,
    /// Target loudness for matched validation assets.
    pub target_lufs: f64,
    /// ABX descriptor included.
    pub abx: bool,
    /// MUSHRA descriptor included.
    pub mushra: bool,
    /// Perceptual regression summary included.
    pub perceptual_regression_summary: bool,
    /// Bundle advisories.
    pub advisories: Vec<String>,
}

/// Compact perceptual scorecard for downstream QA and UIs.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct PerceptualMetrics {
    /// Average EPA preference before correction.
    pub epa_preference_pre: f64,
    /// Average EPA preference after correction.
    pub epa_preference_post: f64,
    /// EPA preference delta, positive means perceptual improvement.
    pub epa_preference_delta: f64,
    /// Midrange inter-channel deviation in dB when more than one comparable
    /// channel exists.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub channel_matching_midrange_rms_db: Option<f64>,
    /// Role-aware channel matching RMS, computed only inside comparable
    /// channel groups such as L/R, surrounds, or matching height pairs.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub role_channel_matching_rms_db: Option<f64>,
    /// Bass-seat/output consistency RMS across sub/LFE outputs in the
    /// modal/crossover band. Lower is better.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bass_consistency_rms_db: Option<f64>,
    /// Center-channel dialog-band roughness after removing the local mean.
    /// Lower is better for speech intelligibility.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dialog_band_roughness_rms_db: Option<f64>,
    /// Peak positive gain requested by exported gain/EQ plugins. High values
    /// are a clipping/headroom risk even when the final curve looks smooth.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub headroom_peak_boost_db: Option<f64>,
    /// Advisory derived from `headroom_peak_boost_db`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub headroom_risk: Option<String>,
    /// Human-readable timing/GD confidence label.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub timing_confidence: Option<String>,
    /// Maximum FIR pre-ringing audible energy across channels, dB relative to
    /// each FIR's main impulse peak.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fir_pre_ringing_audible_db: Option<f64>,
    /// Maximum FIR post-ringing audible energy across channels, dB relative to
    /// each FIR's main impulse peak.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fir_post_ringing_audible_db: Option<f64>,
    /// Maximum FIR temporal masking penalty across channels.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fir_temporal_masking_penalty: Option<f64>,
    /// Maximum direct + early correction energy across channels, dB relative
    /// to total correction energy.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub direct_plus_early_correction_energy_db: Option<f64>,
    /// Worst direct/early cue advisory across channels.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub early_cue_advisory: Option<String>,
}

/// Simple statistical summary for reporting.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct StatisticalSummary {
    /// Arithmetic mean.
    pub mean: f64,
    /// Standard deviation.
    pub std: f64,
}

/// Per-channel supporting-source report.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct SupportingSourceReport {
    /// Design uses power averaging; optional prediction retains coherent phase.
    #[serde(default)]
    pub summation_model: String,
    /// Propagation plus electrical delay at the reference seat, excluding FIR
    /// energy spread. Not a measured onset or a perceptual fusion guarantee.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub propagation_relative_arrival_ms: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub coherent_sum: Option<CurveData>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_coherent_cancellation_db: Option<f64>,
    /// Whether supporting-source processing was enabled for this logical channel.
    pub enabled: bool,
    /// Name of the primary output channel.
    pub primary_output: String,
    /// Name of the supporting output channel.
    pub support_output: String,
    /// Delay applied to the supporting source in ms.
    pub delay_ms: f64,
    /// Length of the supporting-source FIR in taps.
    pub fir_length: usize,
    /// Compensation band in Hz.
    pub compensation_band_hz: (f64, f64),
    /// DRR before compensation (dB) summary, available only with time-gated
    /// impulse-response evidence.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub drr_before_db: Option<StatisticalSummary>,
    /// DRR after compensation (dB) summary, available only with time-gated
    /// impulse-response evidence.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub drr_after_db: Option<StatisticalSummary>,
    /// Whether target constraints (floor/ceiling) were active.
    pub target_constraints_active: bool,
    /// Number of frequency bins where the precedence ceiling was hit.
    pub precedence_limit_hits: usize,
    /// Optional advisories raised during processing (e.g. spatial robustness).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub advisories: Vec<String>,
}

/// Final outcome of an optional RoomEQ processing stage.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum StageStatus {
    /// The stage applied a material change.
    Applied,
    /// The stage was not applicable or no material change was needed.
    Skipped,
    /// The stage ran with reduced capabilities.
    Degraded,
    /// Invalid data or an internal error prevented the stage from running.
    Failed,
}

/// Classification of a stage check.  Structural and safety checks are cheap
/// enough for production; quality checks are intended for QA and diagnostics.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum StageCheckKind {
    Structural,
    Safety,
    Quality,
}

/// A machine-readable assertion made at a pipeline boundary.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
pub struct StageCheck {
    /// Stable identifier used by QA and mutation tests.
    pub id: String,
    /// Enforcement/diagnostic category.
    pub kind: StageCheckKind,
    /// Whether the assertion passed.
    pub passed: bool,
    /// Optional scalar observed by the check.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub observed: Option<f64>,
    /// Optional scalar limit associated with the check.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub limit: Option<f64>,
    /// Human-readable diagnostic, including context on failure.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub diagnostic: Option<String>,
}

impl StageCheck {
    pub fn pass(id: impl Into<String>, kind: StageCheckKind) -> Self {
        Self {
            id: id.into(),
            kind,
            passed: true,
            observed: None,
            limit: None,
            diagnostic: None,
        }
    }

    pub fn fail(
        id: impl Into<String>,
        kind: StageCheckKind,
        diagnostic: impl Into<String>,
    ) -> Self {
        Self {
            id: id.into(),
            kind,
            passed: false,
            observed: None,
            limit: None,
            diagnostic: Some(diagnostic.into()),
        }
    }
}

/// Structured stage status and machine-readable advisories.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
pub struct StageOutcome {
    /// Stable snake_case stage identifier.
    pub stage: String,
    /// Final stage status.
    pub status: StageStatus,
    /// Machine-readable reasons, warnings, or migration notices.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub advisories: Vec<String>,
    /// Assertions evaluated at this stage boundary.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub checks: Vec<StageCheck>,
}

/// Versioned optimizer-confidence evidence used by final production
/// acceptance. Superseded attempts remain visible in `runs_by_channel`, while
/// `confidence` is derived only from runs marked `selected_for_output`.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
pub struct RoomOptimizerEvidence {
    pub policy_version: String,
    pub confidence: crate::OptimizerConfidence,
    pub runs_by_channel: BTreeMap<String, Vec<crate::OptimizerRunEvidence>>,
}

/// Durable outcome of one QA seed, including rejected and reverted attempts.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
pub struct QaSeedOutcome {
    pub seed: u64,
    pub pre_score: f64,
    pub post_score: f64,
    /// Explicit acceptance plus a material improvement of the delivered score.
    pub accepted_useful: bool,
    /// Finite output with explicit final acceptance and no Failed stage.
    /// False includes unverified/reverted output, not only proven unsafe DSP.
    /// This does not imply useful EQ or independent backend certification.
    pub safe_output: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub acceptance: Option<crate::CorrectionAcceptanceReport>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub optimizer_evidence: Option<RoomOptimizerEvidence>,
    pub stage_outcomes: Vec<StageOutcome>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
pub struct QaSeedDistribution {
    pub selected_seed: u64,
    pub accepted_useful_rate: f64,
    pub safe_output_rate: f64,
    pub post_score_spread: f64,
    pub outcomes: Vec<QaSeedOutcome>,
}

/// Optimization metadata
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct OptimizationMetadata {
    /// Raw-measurement score before optimization, evaluated over the same
    /// role/crossover-aware band as `post_score`.
    pub pre_score: f64,
    /// Final deployed-response score, evaluated over the channel's
    /// role/crossover-aware band (lower is better).
    pub post_score: f64,
    /// Optimization algorithm used
    pub algorithm: String,
    /// Loss function that the optimizer minimized.
    /// One of `"flat"`, `"score"`, `"epa"`.
    ///
    /// Note: `pre_score` and `post_score` are *not* values of this loss
    /// function — they are always computed by
    /// `crate::roomeq::workflows::compute_flat_loss` over the
    /// role/crossover-aware evaluation window so that routes do not penalize
    /// intentional crossover rolloff. Runs with different `loss_type` values
    /// stay on the same scale within that channel role. To compare *perceptual* outcomes across
    /// loss types use `epa_per_channel.{pre,post}.preference` instead,
    /// which is computed identically for every run.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub loss_type: Option<String>,
    /// Number of iterations
    pub iterations: usize,
    /// Timestamp
    pub timestamp: String,
    /// Inter-channel deviation metric (computed when >1 channel)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub inter_channel_deviation: Option<crate::InterChannelDeviation>,
    /// Per-channel EPA psychoacoustic metrics (pre-EQ and post-EQ).
    /// Computed from each channel's initial and final frequency responses
    /// using the configured `EpaConfig` (or defaults when unset).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub epa_per_channel: Option<HashMap<String, EpaChannelMetrics>>,
    /// Whole-system EPA psychoacoustic metrics computed after BS.1770-style
    /// channel-energy aggregation.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub epa_multichannel: Option<EpaMultichannelMetrics>,
    /// Provenance for shipped EPA scores: model identity and the
    /// calibration it assumed. Present whenever EPA scores ship.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub epa_provenance: Option<crate::EpaProvenance>,
    /// Claim-level playback summary: what shipped, what benefit was
    /// demonstrated, and which limits apply. Rendered from the evidence
    /// blocks at conversion; auditors re-derive it from the report.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub playback_summary: Option<crate::PlaybackSummary>,
    /// Group delay optimisation summary (GD-Opt v2, Phase GD-4).
    /// Present when GD-Opt was attempted (success or skip with advisory).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub group_delay: Option<crate::GroupDelayOptSummary>,
    /// Per-channel mixed-phase decomposition and excess-phase FIR summaries.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mixed_phase_per_channel: Option<HashMap<String, crate::MixedPhaseCorrectionReport>>,
    /// Perceptual scorecard computed from final exported curves/DSP.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub perceptual_metrics: Option<PerceptualMetrics>,
    /// Home-cinema role/layout interpretation used by role-aware scoring.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub home_cinema_layout: Option<crate::HomeCinemaLayoutReport>,
    /// Coverage summary for multi-position measurements beyond sub-only MSO.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub multi_seat_coverage: Option<crate::MultiSeatCoverageReport>,
    /// All-channel multi-seat correction summary for non-sub home-cinema channels.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub multi_seat_correction: Option<crate::MultiSeatCorrectionReport>,
    /// Bass-management policy and applied trim/headroom summary.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bass_management: Option<crate::BassManagementReport>,
    /// Timing/localization diagnostics derived from measured arrivals and
    /// final exported delay plugins.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub timing_diagnostics: Option<crate::TimingDiagnosticsReport>,
    /// Cross-talk cancellation / binaural-aware correction artifact summary.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ctc: Option<crate::CtcReport>,
    /// Resolved perceptual policy metadata.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub perceptual_policy: Option<PerceptualPolicyReport>,
    /// Bootstrap uncertainty summary.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bootstrap_uncertainty: Option<BootstrapUncertaintyReport>,
    /// Validation/listening-test bundle descriptor.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub validation_bundle: Option<ValidationBundleReport>,
    /// Final workflow sidecar byte identities. None is legacy/unbound; a null
    /// member means the final artifact was unavailable. This is integrity
    /// evidence, not an acoustic acceptance or listening-success certificate.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub final_convolution_sha256: Option<std::collections::BTreeMap<String, Option<String>>>,
    /// Supporting-source room-compensation reports, keyed by logical channel.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub supporting_source: Option<HashMap<String, SupportingSourceReport>>,
    /// Audibility-first acceptance decision for the final correction chain.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub correction_acceptance: Option<crate::CorrectionAcceptanceReport>,
    /// Provisional decision records collected during the run (evidence
    /// checks, constraints, target policy, phase/crossover/sub operations,
    /// pruning, fallback decisions). In-memory threading only, never
    /// serialized: observation stays separate from delivery claims, and
    /// the finalized graph-bound ledger is attached to
    /// `DspChainOutput.correction_decisions` by workflow reconciliation.
    #[serde(skip)]
    pub provisional_decisions: Vec<crate::decision_ledger::DecisionRecord>,
    /// Per-filter audibility-veto verdicts keyed by channel.  Report-only
    /// decisions remain visible even when no filter was removed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub audibility_veto: Option<std::collections::BTreeMap<String, Vec<crate::FilterVetoVerdict>>>,
    /// Frozen-chain adjudication summaries keyed by channel.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub veto_adjudication:
        Option<std::collections::BTreeMap<String, crate::VetoAdjudicationReport>>,
    /// Backend termination, convergence, budget, seed, constraint, and
    /// confidence evidence for the optimizer runs that produced each channel.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub optimizer_evidence: Option<RoomOptimizerEvidence>,
    /// Structured outcomes for optional/degradable processing stages.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub stage_outcomes: Vec<StageOutcome>,
    /// Full QA seed population; the selected median is not a reliability rate.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub qa_seed_distribution: Option<QaSeedDistribution>,
    /// Fully merged configuration used for this run. The CLI records this
    /// when `--override-config` is supplied so QA and display tooling can
    /// verify that an override did not silently replace unrelated sections.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub effective_config: Option<Box<crate::RoomConfig>>,
    /// Per-channel operation-boundary verdicts from the evidence intake.
    /// Phase dispatch, target policy, and the final ledger read these
    /// records; a missing gate means the channel ran before intake gating.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub operation_gates: Option<Vec<crate::eligibility::ChannelOperationGate>>,
    /// Declared ±tolerance in seconds for the report Section 1 T60 flatness
    /// share, carried from the input `reporting` policy. When absent the
    /// viewer applies its documented standard default; it changes no
    /// acceptance math.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub t60_flatness_tolerance_s: Option<f64>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;

    fn sample_curve() -> Curve {
        Curve {
            freq: Array1::from(vec![100.0, 1000.0, 10000.0]),
            spl: Array1::from(vec![80.0, 82.0, 81.0]),
            phase: Some(Array1::from(vec![0.0, 45.0, 90.0])),
            ..Default::default()
        }
    }

    #[test]
    fn curve_data_from_curve_owned() {
        let curve = sample_curve();
        let data = CurveData::from(curve.clone());
        assert_eq!(data.freq, vec![100.0, 1000.0, 10000.0]);
        assert_eq!(data.spl, vec![80.0, 82.0, 81.0]);
        assert_eq!(data.phase, Some(vec![0.0, 45.0, 90.0]));
        assert!(data.norm_range.is_none());
    }

    #[test]
    fn curve_data_from_curve_ref() {
        let curve = sample_curve();
        let data = CurveData::from(&curve);
        assert_eq!(data.freq, vec![100.0, 1000.0, 10000.0]);
        assert_eq!(data.spl, vec![80.0, 82.0, 81.0]);
        assert_eq!(data.phase, Some(vec![0.0, 45.0, 90.0]));
    }

    #[test]
    fn curve_data_wraps_phase_at_serialization_boundary() {
        let curve = Curve {
            freq: Array1::from(vec![100.0, 1_000.0, 10_000.0]),
            spl: Array1::from(vec![80.0, 82.0, 81.0]),
            phase: Some(Array1::from(vec![-540.0, 181.0, 540.0])),
            ..Default::default()
        };

        for data in [CurveData::from(curve.clone()), CurveData::from(&curve)] {
            assert_eq!(data.freq, curve.freq.to_vec());
            assert_eq!(data.spl, curve.spl.to_vec());
            assert_eq!(data.phase, Some(vec![-540.0, 181.0, 540.0]));
            let serialized = serde_json::to_value(&data).unwrap();
            let serialized_phase = serialized["phase"].as_array().unwrap();
            assert_eq!(serialized_phase[0].as_f64(), Some(-180.0));
            assert_eq!(serialized_phase[1].as_f64(), Some(-179.0));
            assert_eq!(serialized_phase[2].as_f64(), Some(-180.0));
            assert!(
                serialized_phase
                    .iter()
                    .all(|phase| (-180.0..180.0).contains(&phase.as_f64().unwrap()))
            );
        }

        // Serialization must not mutate the internal unwrapped phase used by
        // the DSP and impulse-response calculations.
        assert_eq!(
            curve.phase.as_ref().unwrap().to_vec(),
            vec![-540.0, 181.0, 540.0]
        );
    }

    #[test]
    fn curve_data_roundtrips_to_curve() {
        let data = CurveData {
            freq: vec![100.0, 1000.0, 10000.0],
            spl: vec![80.0, 82.0, 81.0],
            phase: Some(vec![0.0, 45.0, 90.0]),
            norm_range: Some((1000.0, 2000.0)),
            ..Default::default()
        };
        let curve: Curve = data.clone().into();
        assert_eq!(curve.freq.to_vec(), data.freq);
        assert_eq!(curve.spl.to_vec(), data.spl);
        assert_eq!(curve.phase.as_ref().map(|p| p.to_vec()), data.phase);
    }

    #[test]
    fn curve_data_preserves_capture_quality_without_inventing_missing_evidence() {
        let capture = Curve {
            freq: ndarray::Array1::from_vec(vec![50.0, 100.0, 200.0]),
            spl: ndarray::Array1::from_vec(vec![72.0, 81.0, 76.0]),
            phase: Some(ndarray::Array1::from_vec(vec![10.0, 20.0, 30.0])),
            noise_floor_db: Some(ndarray::Array1::from_vec(vec![50.0, 60.0, 55.0])),
            coherence: Some(ndarray::Array1::from_vec(vec![0.95, 0.7, 0.99])),
            ..Default::default()
        };
        for data in [CurveData::from(&capture), CurveData::from(capture.clone())] {
            let json = serde_json::to_value(&data).unwrap();
            let restored: Curve = serde_json::from_value::<CurveData>(json).unwrap().into();
            assert_eq!(restored.noise_floor_db, capture.noise_floor_db);
            assert_eq!(restored.coherence, capture.coherence);
            assert_eq!(restored.spl, capture.spl);
            restored.validate("round-tripped capture").unwrap();
        }
        let legacy: CurveData = serde_json::from_value(serde_json::json!({
            "freq": [50.0, 100.0, 200.0], "spl": [72.0, 81.0, 76.0]
        }))
        .unwrap();
        let restored: Curve = legacy.into();
        assert!(restored.noise_floor_db.is_none());
        assert!(restored.coherence.is_none());
    }

    #[test]
    fn curve_data_json_roundtrip() {
        let data = CurveData {
            freq: vec![100.0, 1000.0],
            spl: vec![80.0, 82.0],
            phase: None,
            norm_range: None,
            ..Default::default()
        };
        let json = serde_json::to_string(&data).unwrap();
        let back: CurveData = serde_json::from_str(&json).unwrap();
        assert_eq!(back.freq, data.freq);
        assert_eq!(back.spl, data.spl);
        assert_eq!(back.phase, data.phase);
    }
}
