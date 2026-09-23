//! Multi-view acceptance bundle for RoomEQ correction.
//!
//! Extends spectral comparisons with the standard view set: magnitude,
//! common-reference IR/step, octave-band ETC, frequency-resolved decay,
//! calibrated ambient noise, and full routed-chain headroom. Every view is
//! computed under one shared [`ViewSettings`] and carries its own
//! [`ViewProvenance`]; a smoother plot alone never passes a gate, and reduced
//! playback ringing is never reported as changed passive room damping.

// Rust guideline compliant 2026-02-21

use rustfft::{FftPlanner, num_complex::Complex};
use serde::{Deserialize, Serialize};

mod captured_decay;
mod captured_etc;
mod captured_noise;
mod production;
mod validation;
#[doc(inline)]
pub use captured_decay::{
    DecayFit, MatchedDecayBand, MatchedDecaySettings, MatchedDecayView, matched_capture_decay,
};
#[doc(inline)]
pub use captured_etc::{MatchedEtcView, matched_capture_etc};
#[doc(inline)]
pub use captured_noise::{
    CapturedNoiseSettings, CapturedNoiseView, NoiseAnalysisError, NoisePressureCalibration,
    calibrated_capture_noise,
};
#[doc(inline)]
pub use production::graph_acceptance_evidence;

/// Matched analysis settings shared by every view in a bundle.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ViewSettings {
    /// General magnitude smoothing in bands per octave (6 or 12).
    pub general_smoothing_bands_per_octave: u32,
    /// Bass view resolution: raw measurements or 1/24-octave detail.
    pub bass_detail: BassDetail,
    /// Whether a 1/3-octave view is shown (only for 1/3-specified targets).
    pub third_octave_for_third_octave_targets: bool,
    /// Frequency limits in Hz applied to every frequency-domain view.
    pub freq_limits_hz: [f64; 2],
    /// ETC octave-band centers in Hz (standard set spans 500-4000).
    pub etc_bands_hz: Vec<f64>,
    /// ETC time window in ms (standard view spans 0-40).
    pub etc_window_ms: [f64; 2],
    /// Reference level in dB shared by every view.
    pub reference_level_db: f64,
    /// Window name and parameters shared by every time-domain view.
    pub window: String,
    /// Normalization ledger shared by every view.
    pub normalization: String,
    /// Sample rate in Hz shared by every view.
    pub sample_rate_hz: f64,
}

/// Bass magnitude resolution.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BassDetail {
    /// Unsmoothed raw bass response.
    Raw,
    /// 1/24-octave bass detail.
    Fine24,
}

impl Default for ViewSettings {
    /// Standard view set: 1/12 general smoothing, raw-plus-1/24 bass, common
    /// ETC bands and window, explicit reference level and ledger.
    fn default() -> Self {
        Self {
            general_smoothing_bands_per_octave: 12,
            bass_detail: BassDetail::Fine24,
            third_octave_for_third_octave_targets: false,
            freq_limits_hz: [20.0, 20000.0],
            etc_bands_hz: vec![500.0, 1000.0, 2000.0, 4000.0],
            etc_window_ms: [0.0, 40.0],
            reference_level_db: 0.0,
            window: "rectangular".to_string(),
            normalization: "absolute identical pre/post".to_string(),
            sample_rate_hz: 48000.0,
        }
    }
}

impl ViewSettings {
    /// Canonical hash identifying these exact settings.
    ///
    /// Every view stores the hash of the settings it was computed with; the
    /// bundle gate compares them so mismatched views cannot be combined.
    pub fn settings_hash(&self) -> String {
        let json = serde_json::to_string(self).unwrap_or_default();
        crate::stimuli::sha256_hex(json.as_bytes())
    }
}

/// Processing provenance carried by every view.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ViewProvenance {
    /// Measurement identities the view derives from.
    pub measurement_ids: Vec<String>,
    /// Immutable DSP graph identity under test.
    pub graph_identity: String,
    /// Sample rate in Hz.
    pub sample_rate_hz: f64,
    /// Calibration identity (`uncalibrated` forces unsupported downstream).
    pub calibration: String,
    /// Processing-chain identity (window, filters, routing stages).
    pub processing_chain: String,
}

/// Magnitude view with one smoothing policy for both curves.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MagnitudeView {
    /// Provenance of this view.
    pub provenance: ViewProvenance,
    /// Settings hash this view was computed with.
    pub settings_hash: String,
    /// Frequency grid in Hz.
    pub freqs: Vec<f64>,
    /// Pre-correction magnitude in dB.
    pub pre_db: Vec<f64>,
    /// Post-correction magnitude in dB.
    pub post_db: Vec<f64>,
}

/// Common-reference IR and step view.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct IrStepView {
    /// Provenance of this view.
    pub provenance: ViewProvenance,
    /// Settings hash this view was computed with.
    pub settings_hash: String,
    /// Time axis in ms with a documented common `t = 0` definition.
    pub times_ms: Vec<f64>,
    /// Definition of the shared time origin.
    pub common_reference: String,
    /// Pre-correction impulse response.
    pub pre_ir: Vec<f64>,
    /// Post-correction impulse response.
    pub post_ir: Vec<f64>,
    /// Pre-correction step response (cumulative sum of `pre_ir`).
    pub pre_step: Vec<f64>,
    /// Post-correction step response (cumulative sum of `post_ir`).
    pub post_step: Vec<f64>,
}

/// One octave-band ETC curve.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EtcBand {
    /// Octave-band center in Hz.
    pub center_hz: f64,
    /// Time axis in ms.
    pub times_ms: Vec<f64>,
    /// Pre-correction envelope in dB.
    pub pre_db: Vec<f64>,
    /// Post-correction envelope in dB.
    pub post_db: Vec<f64>,
}

/// Octave-band ETC view with left/right similarity.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EtcView {
    /// Provenance of this view.
    pub provenance: ViewProvenance,
    /// Settings hash this view was computed with.
    pub settings_hash: String,
    /// One curve per octave band.
    pub bands: Vec<EtcBand>,
    /// Maximum absolute L/R envelope difference across bands, in dB.
    pub lr_similarity_db: f64,
}

/// Frequency-resolved decay view.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DecayView {
    /// Provenance of this view.
    pub provenance: ViewProvenance,
    /// Settings hash this view was computed with.
    pub settings_hash: String,
    /// Analysis center frequencies in Hz.
    pub center_freqs: Vec<f64>,
    /// Absolute Schroeder tail energy in dB.
    pub absolute_tail_db: Vec<f64>,
    /// Normalized Schroeder tail in dB (0 dB at direct sound).
    pub normalized_tail_db: Vec<f64>,
    /// T20 fit in seconds with linear-fit quality.
    pub t20_s: (f64, f64),
    /// T30 fit in seconds with linear-fit quality.
    pub t30_s: (f64, f64),
    /// Valid analysis range in Hz.
    pub valid_range_hz: [f64; 2],
    /// Playback tail change after correction, in dB (not room damping).
    pub playback_tail_change_db: f64,
    /// Passive room RT estimate in seconds, never modified by EQ claims.
    pub passive_room_rt_s: f64,
}

/// Calibrated ambient-noise view.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AmbientNoiseView {
    /// Provenance of this view.
    pub provenance: ViewProvenance,
    /// Settings hash this view was computed with.
    pub settings_hash: String,
    /// Frequency grid in Hz.
    pub freqs: Vec<f64>,
    /// Calibrated ambient noise spectrum in dB SPL.
    pub noise_spl_db: Vec<f64>,
    /// Calibration identity authorizing absolute SPL.
    pub calibration: String,
}

/// Full routed-chain headroom view.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RoutedHeadroomView {
    /// Provenance of this view.
    pub provenance: ViewProvenance,
    /// Settings hash this view was computed with.
    pub settings_hash: String,
    /// Legacy single-output true peak in dBFS (kept readable; no verdict).
    pub true_peak_dbfs: Option<f64>,
    /// Legacy single-output amplifier limit in dB (kept readable; no verdict).
    pub amplifier_limit_db: Option<f64>,
    /// Legacy single-output driver limit in dB (kept readable; no verdict).
    pub driver_limit_db: Option<f64>,
    /// Maximum per-filter gain in dB (insufficient on its own).
    pub per_filter_max_db: f64,
    /// Explicit digital ceiling the demand was compared against, in dBFS.
    #[serde(default = "default_digital_ceiling")]
    pub digital_ceiling_dbfs: f64,
    /// Per-output assessments with retained evidence.
    #[serde(default)]
    pub outputs: Vec<AssessedOutput>,
    /// Whether full-chain headroom is demonstrated.
    pub passes: bool,
    /// Machine-readable reason when `passes` is false.
    pub note: String,
}

/// Default digital ceiling for deserializing views written before the
/// explicit ceiling field existed.
fn default_digital_ceiling() -> f64 {
    DIGITAL_CEILING_DBFS
}

/// Final per-driver DSP disposition derived once from the finalized graph.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DriverDisposition {
    /// Driver name.
    pub name: String,
    /// Total gain in dB summed once over driver gain stages.
    pub gain_db: f64,
    /// Total delay in ms summed once over driver delay stages.
    pub delay_ms: f64,
    /// Effective driver polarity after composing all gain-stage inversions.
    pub inverted: bool,
    /// Ordered driver processing route (plugin types in chain order).
    pub route: Vec<String>,
}

/// Final per-channel DSP disposition derived once from the finalized graph.
///
/// Each plugin contributes to exactly one disposition fact: gains sum once,
/// delays sum once, inversion is a single flag with its stage count, and
/// routing/FIR references are descriptive only. Reports and DSP exports
/// built from the same graph bytes therefore tell the same story.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChannelDisposition {
    /// Channel name.
    pub channel: String,
    /// Total gain in dB summed once over channel gain stages.
    pub gain_db: f64,
    /// Effective channel polarity after composing all gain-stage inversions.
    pub inverted: bool,
    /// Channel gain stages carrying an inversion flag.
    pub invert_stage_count: usize,
    /// Total delay in ms summed once over channel delay stages.
    pub delay_ms: f64,
    /// Mute is not a graph stage: always false, with the reason recorded
    /// on the view. Muting lives outside the delivered graph.
    pub muted: bool,
    /// Ordered processing route (plugin types in chain order).
    pub route: Vec<String>,
    /// Matrix routing summaries (`label: inputs -> outputs`).
    pub matrix_routes: Vec<String>,
    /// Convolution FIR references (`ir_file` values). Tap counts need
    /// sidecar bytes and stay unavailable, never zero-filled.
    pub fir_references: Vec<String>,
    /// Snapshot (pre-correction) curve present for gain labeling.
    pub has_snapshot_curve: bool,
    /// Optimizer target curve present for gain labeling.
    pub has_target_curve: bool,
    /// Final (post-correction) curve present for gain labeling.
    pub has_final_curve: bool,
    /// Per-driver dispositions for multi-driver chains.
    pub drivers: Vec<DriverDisposition>,
}

/// DSP disposition view over every channel of the finalized graph.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DspDispositionView {
    /// Provenance of this view.
    pub provenance: ViewProvenance,
    /// Settings hash this view was computed with.
    pub settings_hash: String,
    /// Bound graph fingerprint the dispositions were derived from.
    pub graph_identity: String,
    /// One disposition per channel, in sorted channel order.
    pub channels: Vec<ChannelDisposition>,
    /// Machine-readable reasons for unavailable facts (mute stage, FIR
    /// taps, nonfinite stage parameters). Absent facts stay unavailable,
    /// never zero-filled or hidden by omission.
    pub unavailable: Vec<String>,
}

fn finite_stage_param(
    parameters: &serde_json::Value,
    key: &str,
    channel: &str,
    plugin_type: &str,
) -> Result<f64, String> {
    parameters
        .get(key)
        .and_then(|value| value.as_f64())
        .filter(|value| value.is_finite())
        .ok_or_else(|| {
            format!(
                "channel '{channel}' {plugin_type} stage lacks a finite '{key}'; disposition refused"
            )
        })
}

/// Derive final per-channel DSP dispositions from a finalized graph.
///
/// The derivation is total and single-counting: rebuilding from the same
/// graph bytes yields the same view (last write wins; no stale merges),
/// and every gain/delay/inversion fact comes from exactly one stage sum.
///
/// # Errors
///
/// Returns a reason when the graph holds no channels or a gain/delay
/// stage lacks a finite parameter: an unmeasurable chain is unavailable,
/// never half-derived.
pub fn derive_dsp_dispositions(
    graph: &roomeq_model::DspGraph,
    provenance: ViewProvenance,
    settings_hash: String,
) -> Result<DspDispositionView, String> {
    if graph.channels.is_empty() {
        return Err(String::from(
            "no channels in the finalized graph; disposition unavailable",
        ));
    }
    let value = serde_json::to_value(graph)
        .map_err(|error| format!("finalized graph does not serialize: {error}"))?;
    let mut cleared = value.clone();
    if let Some(object) = cleared.as_object_mut() {
        object.remove("correction_decisions");
    }
    let graph_identity =
        roomeq_model::decision_ledger::canonical_value_identity(&cleared).fingerprint;
    let mut channels: Vec<String> = graph.channels.keys().cloned().collect();
    channels.sort();
    let mut dispositions = Vec::with_capacity(channels.len());
    let unavailable = vec![
        String::from("mute is not a graph stage; muting lives outside the delivered graph"),
        String::from("FIR tap counts need sidecar bytes; references only"),
    ];
    for channel in channels {
        let chain = graph.channels.get(&channel).expect("channel listed");
        dispositions.push(disposition_for_chain(channel, chain, None)?);
    }
    if provenance.graph_identity != graph_identity {
        return Err(String::from(
            "disposition provenance does not identify the supplied graph",
        ));
    }
    Ok(DspDispositionView {
        provenance,
        settings_hash,
        graph_identity,
        channels: dispositions,
        unavailable,
    })
}

fn disposition_for_chain(
    channel: String,
    chain: &roomeq_model::ChannelDspChain,
    driver: Option<&roomeq_model::DriverDspChain>,
) -> Result<ChannelDisposition, String> {
    let plugins = match driver {
        Some(driver) => &driver.plugins,
        None => &chain.plugins,
    };
    let mut gain_db = 0.0;
    let mut delay_ms = 0.0;
    let mut inverted = false;
    let mut invert_stage_count = 0_usize;
    let mut route = Vec::with_capacity(plugins.len());
    let mut matrix_routes = Vec::new();
    let mut fir_references = Vec::new();
    for plugin in plugins {
        route.push(plugin.plugin_type.clone());
        match plugin.plugin_type.as_str() {
            "gain" => {
                gain_db += finite_stage_param(&plugin.parameters, "gain_db", &channel, "gain")?;
                if plugin
                    .parameters
                    .get("invert")
                    .and_then(|value| value.as_bool())
                    .unwrap_or(false)
                {
                    inverted = !inverted;
                    invert_stage_count += 1;
                }
            }
            "delay" => {
                delay_ms += finite_stage_param(&plugin.parameters, "delay_ms", &channel, "delay")?;
            }
            "matrix" => {
                let label = plugin
                    .parameters
                    .get("label")
                    .and_then(|value| value.as_str())
                    .unwrap_or("matrix");
                matrix_routes.push(format!("{label}: route recorded on graph"));
            }
            "convolution" => {
                let reference = plugin
                    .parameters
                    .get("ir_file")
                    .and_then(|value| value.as_str())
                    .unwrap_or("reference unavailable");
                fir_references.push(reference.to_string());
            }
            _ => {}
        }
    }
    let drivers = match driver {
        Some(_) => Vec::new(),
        None => chain
            .drivers
            .as_ref()
            .map(|drivers| {
                drivers
                    .iter()
                    .map(|driver| {
                        let inner =
                            disposition_for_chain(driver.name.clone(), chain, Some(driver))?;
                        Ok(DriverDisposition {
                            name: driver.name.clone(),
                            gain_db: inner.gain_db,
                            delay_ms: inner.delay_ms,
                            inverted: inner.inverted,
                            route: inner.route,
                        })
                    })
                    .collect::<Result<Vec<_>, String>>()
            })
            .transpose()?
            .unwrap_or_default(),
    };
    if !gain_db.is_finite() || !delay_ms.is_finite() || delay_ms < 0.0 {
        return Err(format!(
            "channel '{channel}' has a nonfinite aggregate gain/delay or noncausal delay"
        ));
    }
    Ok(ChannelDisposition {
        channel,
        gain_db,
        inverted,
        invert_stage_count,
        delay_ms,
        muted: false,
        route,
        matrix_routes,
        fir_references,
        has_snapshot_curve: chain.initial_curve.is_some(),
        has_target_curve: chain.target_curve.is_some(),
        has_final_curve: chain.final_curve.is_some(),
        drivers,
    })
}

/// Complete multi-view acceptance bundle.
///
/// Every view is optional at the type level so partial bundles can be
/// assembled and honestly reported; the gate fails them.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct AcceptanceBundle {
    /// Matched settings every included view must carry.
    pub settings: ViewSettings,
    /// Physical outputs the routed headroom assessment must cover.
    /// Derived from the route plan by producers, never inferred from the
    /// subset of outputs that happened to supply successful measurements.
    /// Legacy absence is readable but cannot establish complete coverage.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub required_output_ids: Vec<String>,
    /// Magnitude view.
    pub magnitude: Option<MagnitudeView>,
    /// IR/step view.
    pub ir_step: Option<IrStepView>,
    /// ETC view.
    pub etc: Option<EtcView>,
    /// Decay view.
    pub decay: Option<DecayView>,
    /// Ambient-noise view.
    pub ambient_noise: Option<AmbientNoiseView>,
    /// Routed headroom view.
    pub headroom: Option<RoutedHeadroomView>,
    /// Final per-channel DSP dispositions derived from the finalized graph.
    pub disposition: Option<DspDispositionView>,
}

/// Bundle gate outcome.
#[derive(Debug, Clone, PartialEq)]
pub struct BundleGate {
    /// Whether required views are internally valid and supplied headroom passes.
    /// This is not acquisition authentication or proof of correction benefit.
    pub passed: bool,
    /// Views absent from the bundle.
    pub missing_views: Vec<&'static str>,
    /// Machine-readable reasons for failure.
    pub failures: Vec<String>,
}

fn check_view(hash: &str, expected: &str, name: &'static str, failures: &mut Vec<String>) {
    if hash != expected {
        failures.push(format!("{name} was computed with different settings"));
    }
}

/// Evaluate the acceptance gate for a bundle.
///
/// The gate passes only when every view is present under matched settings
/// with provenance, and the headroom view demonstrates full-chain margin. A
/// smoother magnitude plot alone never passes.
///
/// # Examples
///
/// ```
/// use roomeq_quality::{AcceptanceBundle, evaluate_bundle_gate};
/// let gate = evaluate_bundle_gate(&AcceptanceBundle::default());
/// assert!(!gate.passed);
/// assert_eq!(gate.missing_views.len(), 7);
/// ```
pub fn evaluate_bundle_gate(bundle: &AcceptanceBundle) -> BundleGate {
    let mut missing = Vec::new();
    let mut failures = validation::validate_bundle_contents(bundle);
    let expected = bundle.settings.settings_hash();

    match &bundle.magnitude {
        Some(view) => check_view(&view.settings_hash, &expected, "magnitude", &mut failures),
        None => missing.push("magnitude"),
    }
    match &bundle.ir_step {
        Some(view) => check_view(&view.settings_hash, &expected, "ir_step", &mut failures),
        None => missing.push("ir_step"),
    }
    match &bundle.etc {
        Some(view) => check_view(&view.settings_hash, &expected, "etc", &mut failures),
        None => missing.push("etc"),
    }
    match &bundle.decay {
        Some(view) => check_view(&view.settings_hash, &expected, "decay", &mut failures),
        None => missing.push("decay"),
    }
    match &bundle.ambient_noise {
        Some(view) => {
            check_view(
                &view.settings_hash,
                &expected,
                "ambient_noise",
                &mut failures,
            );
            if view.calibration == "uncalibrated" {
                failures.push("ambient noise is uncalibrated".to_string());
            }
        }
        None => missing.push("ambient_noise"),
    }
    match &bundle.headroom {
        Some(view) => {
            check_view(&view.settings_hash, &expected, "headroom", &mut failures);
            // The stored flag is never trusted: the verdict is re-derived
            // from the retained evidence, and any mismatch with the stored
            // assessment fails as tampered or stale.
            let (verdict, consistent) = derive_headroom_verdict(view);
            if verdict != HeadroomVerdict::Pass {
                failures.push(format!(
                    "headroom not demonstrated ({verdict:?}): {}",
                    view.note
                ));
            }
            if !consistent {
                failures.push("headroom stored assessment disagrees with its evidence".to_string());
            }
        }
        None => missing.push("headroom"),
    }

    match &bundle.disposition {
        Some(view) => {
            check_view(&view.settings_hash, &expected, "disposition", &mut failures);
            let derived: Vec<String> = view.channels.iter().map(|c| c.channel.clone()).collect();
            let mut sorted = derived.clone();
            sorted.sort();
            if derived != sorted {
                failures.push("disposition channels are not in sorted order".to_string());
            }
        }
        None => missing.push("disposition"),
    }

    if bundle.magnitude.is_some() && bundle.ir_step.is_none() {
        failures.push("magnitude alone never passes without time-domain views".to_string());
    }

    BundleGate {
        passed: missing.is_empty() && failures.is_empty(),
        missing_views: missing,
        failures,
    }
}

/// Step response as the cumulative sum of an impulse response.
pub fn step_response(ir: &[f64]) -> Vec<f64> {
    let mut sum = 0.0;
    ir.iter()
        .map(|sample| {
            sum += *sample;
            sum
        })
        .collect()
}

/// Analytic-signal envelope via the FFT Hilbert transform.
///
/// Forward transform uses the unnormalized FFT convention with `1/N` on the
/// inverse; the envelope is the magnitude of the analytic signal. Output
/// units match the input sample units.
pub fn hilbert_envelope(samples: &[f64]) -> Vec<f64> {
    if samples.is_empty() {
        return Vec::new();
    }
    let n = samples.len();
    let mut planner = FftPlanner::<f64>::new();
    let forward = planner.plan_fft_forward(n);
    let inverse = planner.plan_fft_inverse(n);
    let mut spectrum: Vec<Complex<f64>> = samples.iter().map(|s| Complex::new(*s, 0.0)).collect();
    forward.process(&mut spectrum);
    for (bin, value) in spectrum.iter_mut().enumerate() {
        if bin == 0 || (n.is_multiple_of(2) && bin == n / 2) {
            continue;
        } else if bin < n.div_ceil(2) {
            *value *= 2.0;
        } else {
            *value = Complex::new(0.0, 0.0);
        }
    }
    inverse.process(&mut spectrum);
    spectrum.iter().map(|z| z.norm() / n as f64).collect()
}

/// Octave-band ETC envelope in dB.
///
/// The bandpass is an analytic brick-wall octave mask in the frequency
/// domain, documented here so fixtures stay reproducible; it is not a claim
/// about any measurement filter. Reference is the envelope peak (0 dB).
pub fn octave_etc_envelope(ir: &[f64], sample_rate_hz: f64, center_hz: f64) -> Vec<f64> {
    if ir.is_empty() || !sample_rate_hz.is_finite() || sample_rate_hz <= 0.0 {
        return Vec::new();
    }
    let n = ir.len();
    let mut planner = FftPlanner::<f64>::new();
    let forward = planner.plan_fft_forward(n);
    let inverse = planner.plan_fft_inverse(n);
    let mut spectrum: Vec<Complex<f64>> = ir.iter().map(|s| Complex::new(*s, 0.0)).collect();
    forward.process(&mut spectrum);
    let low = center_hz / std::f64::consts::SQRT_2;
    let high = center_hz * std::f64::consts::SQRT_2;
    for (bin, value) in spectrum.iter_mut().enumerate() {
        let frequency = bin as f64 * sample_rate_hz / n as f64;
        let aliased = if frequency > sample_rate_hz / 2.0 {
            sample_rate_hz - frequency
        } else {
            frequency
        };
        if aliased < low || aliased > high {
            *value = Complex::new(0.0, 0.0);
        }
    }
    inverse.process(&mut spectrum);
    let band: Vec<f64> = spectrum.iter().map(|z| z.re / n as f64).collect();
    let envelope = hilbert_envelope(&band);
    let peak = envelope.iter().fold(0.0_f64, |a, b| a.max(*b)).max(1e-30);
    envelope
        .iter()
        .map(|e| 20.0 * (e.max(1e-30) / peak).log10())
        .collect()
}

/// Schroeder reverse-integrated decay in dB, normalized to 0 dB at direct.
pub fn schroeder_decay_db(ir: &[f64]) -> Vec<f64> {
    if ir.is_empty() {
        return Vec::new();
    }
    let total: f64 = ir.iter().map(|s| s * s).sum();
    if total <= 0.0 {
        return vec![f64::NEG_INFINITY; ir.len()];
    }
    let mut tail = 0.0;
    let mut decay: Vec<f64> = ir
        .iter()
        .rev()
        .map(|s| {
            tail += s * s;
            10.0 * (tail / total).max(1e-30).log10()
        })
        .collect();
    decay.reverse();
    decay
}

/// Linear T60 fit over one decay segment; returns seconds and R-squared.
///
/// `from_db` and `to_db` select the fit interval (for example -5/-25 for T20
/// extrapolated to 60 dB, or -5/-35 for T30). No pass/fail threshold lives
/// here; fit quality is reported for the reviewer.
pub fn fit_t60(times_s: &[f64], decay_db: &[f64], from_db: f64, to_db: f64) -> (f64, f64) {
    let points: Vec<(f64, f64)> = times_s
        .iter()
        .zip(decay_db.iter())
        .filter(|(_, decay)| **decay <= from_db && **decay >= to_db)
        .map(|(time, decay)| (*time, *decay))
        .collect();
    if points.len() < 2 {
        return (f64::NAN, 0.0);
    }
    let n = points.len() as f64;
    let mean_t = points.iter().map(|p| p.0).sum::<f64>() / n;
    let mean_d = points.iter().map(|p| p.1).sum::<f64>() / n;
    let slope_num: f64 = points.iter().map(|p| (p.0 - mean_t) * (p.1 - mean_d)).sum();
    let slope_den: f64 = points.iter().map(|p| (p.0 - mean_t).powi(2)).sum();
    if slope_den <= 0.0 {
        return (f64::NAN, 0.0);
    }
    let slope = slope_num / slope_den;
    let intercept = mean_d - slope * mean_t;
    let ss_total: f64 = points.iter().map(|p| (p.1 - mean_d).powi(2)).sum();
    let ss_res: f64 = points
        .iter()
        .map(|p| (p.1 - (slope * p.0 + intercept)).powi(2))
        .sum();
    let r_squared = if ss_total > 0.0 {
        1.0 - ss_res / ss_total
    } else {
        0.0
    };
    let t60 = if slope < 0.0 {
        -60.0 / slope
    } else {
        f64::INFINITY
    };
    (t60, r_squared.clamp(0.0, 1.0))
}

/// Equality tolerance for demand-versus-limit comparisons, in dB.
///
/// A demand exactly at the limit meets the limit; anything beyond the limit
/// plus this epsilon fails. The value is far below measurement resolution so
/// it only absorbs floating-point rounding, never real overload.
pub const HEADROOM_EPSILON_DB: f64 = 1e-9;

/// Default digital ceiling for true-peak demand, in dBFS.
///
/// Inter-sample overs can exceed 0 dBFS; the ceiling itself is explicit per
/// assessment rather than hard-coded at the call site.
pub const DIGITAL_CEILING_DBFS: f64 = 0.0;

/// Trial identity of a headroom demand or limit.
///
/// Small-signal demand must never be compared against a limiter or
/// compressor trial limit; mismatched trials yield unknown, not pass.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HeadroomTrial {
    /// Linear small-signal comparison.
    SmallSignal,
    /// Limiter-engaged trial.
    LimiterTrial,
    /// Compressor-engaged trial.
    CompressorTrial,
}

/// Calibrated physical level with the identity authorizing absolute comparison.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CalibratedLevel {
    /// Level in dB SPL.
    pub value_db_spl: f64,
    /// Calibration identity; `"uncalibrated"` never authorizes comparison.
    pub calibration_id: String,
    /// Trial the level was observed under.
    pub trial: HeadroomTrial,
}

/// Safety verdict for one headroom comparison.
///
/// Unknown is never a pass: missing or incomparable evidence fails the gate
/// with a reason instead of passing silently.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HeadroomVerdict {
    /// Demand is demonstrated within every applicable limit.
    Pass,
    /// Demand demonstrably exceeds a limit.
    Fail,
    /// Evidence is missing, non-finite, or incomparable.
    Unknown,
}

/// Headroom demand and limits for one physical output.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OutputHeadroomInput {
    /// Physical output name, e.g. `"sub1"`.
    pub output_name: String,
    /// True-peak demand of the full routed chain in dBFS, if measured.
    pub true_peak_dbfs: Option<f64>,
    /// Calibrated peak demand in dB SPL, if measured.
    pub demand_db_spl: Option<CalibratedLevel>,
    /// Amplifier output limit in dB SPL under stated conditions, if known.
    pub amplifier_limit_db_spl: Option<CalibratedLevel>,
    /// Driver excursion/thermal limit in dB SPL under stated conditions, if known.
    pub driver_limit_db_spl: Option<CalibratedLevel>,
}

/// Assessed headroom for one physical output with surviving margins.
///
/// The raw `input` is retained so the gate can re-derive the verdict from
/// evidence instead of trusting a deserialized flag.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AssessedOutput {
    /// Physical output name.
    pub output_name: String,
    /// Evidence this assessment was derived from.
    pub input: OutputHeadroomInput,
    /// Digital margin (`ceiling - true peak`) in dB, when a peak was measured.
    pub digital_margin_db: Option<f64>,
    /// Amplifier margin (`limit - demand`) in dB, when comparable.
    pub amplifier_margin_db: Option<f64>,
    /// Driver margin (`limit - demand`) in dB, when comparable.
    pub driver_margin_db: Option<f64>,
    /// Combined verdict for this output.
    pub verdict: HeadroomVerdict,
    /// Machine-readable reasons, one per failed or unknown comparison.
    pub reasons: Vec<String>,
}

fn assess_digital(
    output_name: &str,
    ceiling_dbfs: f64,
    true_peak_dbfs: Option<f64>,
    reasons: &mut Vec<String>,
) -> (Option<f64>, HeadroomVerdict) {
    match true_peak_dbfs {
        None => (None, HeadroomVerdict::Unknown),
        Some(peak) => {
            if !peak.is_finite() || !ceiling_dbfs.is_finite() {
                reasons.push(format!("{output_name}: non-finite true peak or ceiling"));
                return (None, HeadroomVerdict::Unknown);
            }
            let margin = ceiling_dbfs - peak;
            if !margin.is_finite() {
                reasons.push(format!("{output_name}: non-finite digital margin"));
                return (None, HeadroomVerdict::Unknown);
            }
            if margin >= -HEADROOM_EPSILON_DB {
                (Some(margin), HeadroomVerdict::Pass)
            } else {
                reasons.push(format!(
                    "{output_name}: true peak {peak:.3} dBFS exceeds ceiling {ceiling_dbfs:.3} dBFS"
                ));
                (Some(margin), HeadroomVerdict::Fail)
            }
        }
    }
}

fn assess_physical(
    output_name: &str,
    kind: &str,
    demand: &Option<CalibratedLevel>,
    limit: &Option<CalibratedLevel>,
    reasons: &mut Vec<String>,
) -> (Option<f64>, HeadroomVerdict) {
    match (demand, limit) {
        (Some(demand), Some(limit)) => {
            if !demand.value_db_spl.is_finite() || !limit.value_db_spl.is_finite() {
                reasons.push(format!("{output_name}: non-finite {kind} demand or limit"));
                return (None, HeadroomVerdict::Unknown);
            }
            if [&demand.calibration_id, &limit.calibration_id]
                .iter()
                .any(|id| id.trim().is_empty() || id.trim().eq_ignore_ascii_case("uncalibrated"))
            {
                reasons.push(format!(
                    "{output_name}: {kind} comparison lacks calibration"
                ));
                return (None, HeadroomVerdict::Unknown);
            }
            if demand.calibration_id != limit.calibration_id {
                reasons.push(format!(
                    "{output_name}: {kind} calibration mismatch (demand '{}' vs limit '{}')",
                    demand.calibration_id, limit.calibration_id
                ));
                return (None, HeadroomVerdict::Unknown);
            }
            if demand.trial != limit.trial {
                reasons.push(format!(
                    "{output_name}: {kind} trial mismatch ({:?} vs {:?})",
                    demand.trial, limit.trial
                ));
                return (None, HeadroomVerdict::Unknown);
            }
            let margin = limit.value_db_spl - demand.value_db_spl;
            if !margin.is_finite() {
                reasons.push(format!("{output_name}: non-finite {kind} margin"));
                return (None, HeadroomVerdict::Unknown);
            }
            if margin >= -HEADROOM_EPSILON_DB {
                (Some(margin), HeadroomVerdict::Pass)
            } else {
                reasons.push(format!(
                    "{output_name}: {kind} demand {:.3} dB SPL exceeds limit {:.3} dB SPL",
                    demand.value_db_spl, limit.value_db_spl
                ));
                (Some(margin), HeadroomVerdict::Fail)
            }
        }
        (Some(_), None) => {
            reasons.push(format!("{output_name}: no applicable {kind} limit"));
            (None, HeadroomVerdict::Unknown)
        }
        (None, _) => {
            reasons.push(format!("{output_name}: no calibrated {kind} demand"));
            (None, HeadroomVerdict::Unknown)
        }
    }
}

fn combine_verdicts(verdicts: &[HeadroomVerdict]) -> HeadroomVerdict {
    if verdicts.contains(&HeadroomVerdict::Fail) {
        HeadroomVerdict::Fail
    } else if verdicts.contains(&HeadroomVerdict::Unknown) {
        HeadroomVerdict::Unknown
    } else {
        HeadroomVerdict::Pass
    }
}

/// Assess one physical output against the digital ceiling and physical limits.
///
/// Digital demand is compared to the explicit ceiling; physical demand is
/// compared to a limit only when both share units (dB SPL), calibration
/// identity, and trial. Anything else yields unknown for that comparison.
pub fn assess_output_headroom(ceiling_dbfs: f64, input: &OutputHeadroomInput) -> AssessedOutput {
    let mut reasons = Vec::new();
    let mut verdicts = Vec::new();
    let had_any_demand = input.true_peak_dbfs.is_some() || input.demand_db_spl.is_some();

    let (digital_margin_db, digital) = assess_digital(
        &input.output_name,
        ceiling_dbfs,
        input.true_peak_dbfs,
        &mut reasons,
    );
    if input.true_peak_dbfs.is_some() {
        verdicts.push(digital);
    }
    let (amplifier_margin_db, amplifier) = assess_physical(
        &input.output_name,
        "amplifier",
        &input.demand_db_spl,
        &input.amplifier_limit_db_spl,
        &mut reasons,
    );
    verdicts.push(amplifier);
    let (driver_margin_db, driver) = assess_physical(
        &input.output_name,
        "driver",
        &input.demand_db_spl,
        &input.driver_limit_db_spl,
        &mut reasons,
    );
    verdicts.push(driver);

    let mut verdict = combine_verdicts(&verdicts);
    if !had_any_demand {
        reasons.push(format!("{}: no demand evidence", input.output_name));
        verdict = HeadroomVerdict::Unknown;
    }
    AssessedOutput {
        output_name: input.output_name.clone(),
        input: input.clone(),
        digital_margin_db,
        amplifier_margin_db,
        driver_margin_db,
        verdict,
        reasons,
    }
}

/// Build a routed headroom view from full-chain measurements.
///
/// `passes` requires every assessed output to demonstrate its digital demand
/// within the explicit ceiling and its physical demand within every
/// applicable limit in compatible units, calibration, and trial. A per-filter
/// maximum alone is informational and never authorizes a pass. Missing,
/// non-finite, or incomparable evidence yields unknown, which fails the gate.
pub fn assess_routed_headroom(
    provenance: ViewProvenance,
    settings_hash: String,
    digital_ceiling_dbfs: f64,
    per_filter_max_db: f64,
    outputs: Vec<OutputHeadroomInput>,
) -> RoutedHeadroomView {
    let assessed: Vec<AssessedOutput> = outputs
        .iter()
        .map(|input| assess_output_headroom(digital_ceiling_dbfs, input))
        .collect();
    let verdict = if assessed.is_empty() {
        HeadroomVerdict::Unknown
    } else {
        combine_verdicts(
            &assessed
                .iter()
                .map(|output| output.verdict)
                .collect::<Vec<_>>(),
        )
    };
    let mut note = String::new();
    if assessed.is_empty() {
        note.push_str("no output assessed; ");
    }
    for output in &assessed {
        for reason in &output.reasons {
            note.push_str(reason);
            note.push_str("; ");
        }
    }
    // Legacy scalar fields stay populated from the first output so old
    // readers keep working; they no longer decide the verdict.
    let (true_peak_dbfs, amplifier_limit_db, driver_limit_db) = match outputs.first() {
        Some(first) => (
            first.true_peak_dbfs,
            first
                .amplifier_limit_db_spl
                .as_ref()
                .map(|limit| limit.value_db_spl),
            first
                .driver_limit_db_spl
                .as_ref()
                .map(|limit| limit.value_db_spl),
        ),
        None => (None, None, None),
    };
    RoutedHeadroomView {
        provenance,
        settings_hash,
        true_peak_dbfs,
        amplifier_limit_db,
        driver_limit_db,
        per_filter_max_db,
        digital_ceiling_dbfs,
        outputs: assessed,
        passes: verdict == HeadroomVerdict::Pass,
        note: note.trim_end_matches("; ").to_string(),
    }
}

/// Re-derive the headroom verdict from a view's stored evidence.
///
/// The bundle gate re-runs the assessment over the retained per-output
/// inputs instead of trusting deserialized flags, so tampering with
/// `passes` or a per-output verdict over failing evidence is rejected.
/// Returns the verdict plus whether the stored assessment matches the
/// re-derived one.
pub fn derive_headroom_verdict(view: &RoutedHeadroomView) -> (HeadroomVerdict, bool) {
    if view.outputs.is_empty() {
        // Legacy views predate per-output evidence: the bare dB limits carry
        // no calibration or trial, so physical margin is unknowable.
        let verdict = match view.true_peak_dbfs {
            Some(peak)
                if peak.is_finite()
                    && view.digital_ceiling_dbfs.is_finite()
                    && peak <= view.digital_ceiling_dbfs + HEADROOM_EPSILON_DB =>
            {
                HeadroomVerdict::Unknown
            }
            Some(_) => HeadroomVerdict::Fail,
            None => HeadroomVerdict::Unknown,
        };
        return (verdict, !view.passes);
    }
    let mut consistent = true;
    let mut verdicts = Vec::with_capacity(view.outputs.len());
    for output in &view.outputs {
        let recomputed = assess_output_headroom(view.digital_ceiling_dbfs, &output.input);
        consistent = consistent && recomputed == *output;
        verdicts.push(recomputed.verdict);
    }
    let verdict = combine_verdicts(&verdicts);
    consistent = consistent && (view.passes == (verdict == HeadroomVerdict::Pass));
    (verdict, consistent)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn provenance() -> ViewProvenance {
        ViewProvenance {
            measurement_ids: vec!["seat-loop".to_string()],
            graph_identity: roomeq_model::decision_ledger::canonical_graph_identity(
                &fixture_graph(),
            )
            .fingerprint,
            sample_rate_hz: 48000.0,
            calibration: "calibrated".to_string(),
            processing_chain: "rectangular identical".to_string(),
        }
    }

    fn magnitude_view(settings: &ViewSettings) -> MagnitudeView {
        MagnitudeView {
            provenance: provenance(),
            settings_hash: settings.settings_hash(),
            freqs: vec![30.0, 60.0, 120.0],
            pre_db: vec![80.0, 84.0, 80.0],
            post_db: vec![80.0, 81.0, 80.0],
        }
    }

    fn full_bundle() -> AcceptanceBundle {
        let settings = ViewSettings::default();
        let hash = settings.settings_hash();
        let times_ms: Vec<f64> = (0..8)
            .map(|i| i as f64 * 1000.0 / settings.sample_rate_hz)
            .collect();
        let ir = vec![1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.0078125];
        AcceptanceBundle {
            required_output_ids: vec![String::from("sub1")],
            magnitude: Some(magnitude_view(&settings)),
            ir_step: Some(IrStepView {
                provenance: provenance(),
                settings_hash: hash.clone(),
                times_ms,
                common_reference: "t=0 at IR peak".to_string(),
                pre_ir: ir.clone(),
                post_ir: ir.clone(),
                pre_step: step_response(&ir),
                post_step: step_response(&ir),
            }),
            etc: Some(EtcView {
                provenance: provenance(),
                settings_hash: hash.clone(),
                bands: settings
                    .etc_bands_hz
                    .iter()
                    .map(|center| EtcBand {
                        center_hz: *center,
                        times_ms: (0..=1920).map(|index| index as f64 / 48.0).collect(),
                        pre_db: (0..=1920)
                            .map(|index| -(index as f64) / 1920.0 * 6.0)
                            .collect(),
                        post_db: (0..=1920)
                            .map(|index| -(index as f64) / 1920.0 * 9.0)
                            .collect(),
                    })
                    .collect(),
                lr_similarity_db: 1.0,
            }),
            decay: Some(DecayView {
                provenance: provenance(),
                settings_hash: hash.clone(),
                center_freqs: vec![1000.0],
                absolute_tail_db: vec![-20.0],
                normalized_tail_db: vec![-20.0],
                t20_s: (0.4, 0.99),
                t30_s: (0.4, 0.99),
                valid_range_hz: [500.0, 4000.0],
                playback_tail_change_db: -3.0,
                passive_room_rt_s: 0.4,
            }),
            ambient_noise: Some(AmbientNoiseView {
                provenance: provenance(),
                settings_hash: hash.clone(),
                freqs: vec![100.0, 1000.0],
                noise_spl_db: vec![30.0, 25.0],
                calibration: "calibrated".to_string(),
            }),
            headroom: Some(assess_routed_headroom(
                provenance(),
                hash,
                DIGITAL_CEILING_DBFS,
                0.0,
                vec![safe_output("sub1")],
            )),
            disposition: Some(fixture_disposition(&settings)),
            settings,
        }
    }

    fn fixture_graph() -> roomeq_model::DspGraph {
        use roomeq_model::Plugin;
        let mut graph = roomeq_model::DspGraph::new("test");
        graph.add_channel(
            "left",
            vec![
                Plugin {
                    kind: "gain".to_string(),
                    parameters: serde_json::json!({"gain_db": -3.0, "invert": true}),
                },
                Plugin {
                    kind: "delay".to_string(),
                    parameters: serde_json::json!({"delay_ms": 2.5}),
                },
                Plugin {
                    kind: "convolution".to_string(),
                    parameters: serde_json::json!({"ir_file": "left.wav"}),
                },
            ],
        );
        graph.add_channel(
            "right",
            vec![Plugin {
                kind: "gain".to_string(),
                parameters: serde_json::json!({"gain_db": 1.5}),
            }],
        );
        graph
    }

    fn fixture_disposition(settings: &ViewSettings) -> DspDispositionView {
        derive_dsp_dispositions(&fixture_graph(), provenance(), settings.settings_hash())
            .expect("fixture graph derives dispositions")
    }

    #[test]
    fn bundle_empty_fails_with_all_views_missing() {
        let gate = evaluate_bundle_gate(&AcceptanceBundle::default());
        assert!(!gate.passed);
        assert_eq!(gate.missing_views.len(), 7);
    }

    #[test]
    fn bundle_full_passes_with_matched_settings() {
        let bundle = full_bundle();
        let gate = evaluate_bundle_gate(&bundle);
        assert!(gate.passed, "failures: {:?}", gate.failures);
    }

    #[test]
    fn roadmap_correction_bundle_rejects_empty_magnitude_despite_matching_hash() {
        let mut bundle = full_bundle();
        let view = bundle.magnitude.as_mut().unwrap();
        view.freqs.clear();
        view.pre_db.clear();
        view.post_db.clear();
        assert!(
            !evaluate_bundle_gate(&bundle).passed,
            "empty traces must not establish acceptance"
        );
    }

    #[test]
    fn roadmap_correction_bundle_rejects_mixed_graph_provenance() {
        let mut bundle = full_bundle();
        bundle.ir_step.as_mut().unwrap().provenance.graph_identity = "different-graph".into();
        assert!(
            !evaluate_bundle_gate(&bundle).passed,
            "matching settings cannot bind different processing graphs"
        );
    }

    #[test]
    fn roadmap_correction_bundle_rejects_invalid_trace_and_provenance_fields() {
        type Mutation = fn(&mut AcceptanceBundle);
        let cases: &[(&str, Mutation)] = &[
            ("magnitude length", |b| {
                b.magnitude.as_mut().unwrap().post_db.pop();
            }),
            ("magnitude NaN", |b| {
                b.magnitude.as_mut().unwrap().pre_db[0] = f64::NAN
            }),
            ("magnitude frequency order", |b| {
                b.magnitude.as_mut().unwrap().freqs.swap(0, 1)
            }),
            ("magnitude outside band", |b| {
                b.magnitude.as_mut().unwrap().freqs[0] = 1.0
            }),
            ("IR sampling rate", |b| {
                b.ir_step.as_mut().unwrap().times_ms[1] += 1.0
            }),
            ("IR reference", |b| {
                b.ir_step.as_mut().unwrap().common_reference.clear()
            }),
            ("step not integral", |b| {
                b.ir_step.as_mut().unwrap().post_step[2] += 0.1
            }),
            ("ETC band missing", |b| {
                b.etc.as_mut().unwrap().bands.pop();
            }),
            ("ETC band duplicate", |b| {
                b.etc.as_mut().unwrap().bands[1].center_hz = 500.0
            }),
            ("ETC window", |b| {
                b.etc.as_mut().unwrap().bands[0].times_ms[1920] = 39.0
            }),
            ("ETC NaN", |b| {
                b.etc.as_mut().unwrap().bands[0].post_db[0] = f64::NAN
            }),
            ("decay fit quality", |b| {
                b.decay.as_mut().unwrap().t20_s.1 = 1.1
            }),
            ("decay negative time", |b| {
                b.decay.as_mut().unwrap().t30_s.0 = -1.0
            }),
            ("decay missing band", |b| {
                b.decay.as_mut().unwrap().center_freqs.clear()
            }),
            ("decay nonfinite", |b| {
                b.decay.as_mut().unwrap().passive_room_rt_s = f64::INFINITY
            }),
            ("noise calibration mismatch", |b| {
                b.ambient_noise.as_mut().unwrap().calibration = "another-calibration".into()
            }),
            ("noise NaN", |b| {
                b.ambient_noise.as_mut().unwrap().noise_spl_db[0] = f64::NAN
            }),
            ("no measurement IDs", |b| {
                b.ir_step
                    .as_mut()
                    .unwrap()
                    .provenance
                    .measurement_ids
                    .clear()
            }),
            ("blank measurement ID", |b| {
                b.etc.as_mut().unwrap().provenance.measurement_ids[0] = " ".into()
            }),
            ("duplicate measurement ID", |b| {
                b.decay
                    .as_mut()
                    .unwrap()
                    .provenance
                    .measurement_ids
                    .push("seat-loop".into())
            }),
            ("sample-rate mismatch", |b| {
                b.decay.as_mut().unwrap().provenance.sample_rate_hz = 44100.0
            }),
            ("missing processing provenance", |b| {
                b.magnitude
                    .as_mut()
                    .unwrap()
                    .provenance
                    .processing_chain
                    .clear()
            }),
            ("missing calibration status", |b| {
                b.magnitude.as_mut().unwrap().provenance.calibration.clear()
            }),
            ("missing graph", |b| {
                b.headroom
                    .as_mut()
                    .unwrap()
                    .provenance
                    .graph_identity
                    .clear()
            }),
            ("missing output plan", |b| b.required_output_ids.clear()),
            ("unassessed required output", |b| {
                b.required_output_ids.push("sub2".into())
            }),
            ("duplicate output plan", |b| {
                b.required_output_ids.push("sub1".into())
            }),
            ("duplicate assessed output", |b| {
                let v = b.headroom.as_mut().unwrap();
                v.outputs.push(v.outputs[0].clone());
            }),
            ("empty disposition", |b| {
                b.disposition.as_mut().unwrap().channels.clear()
            }),
            ("duplicate disposition", |b| {
                let v = b.disposition.as_mut().unwrap();
                v.channels.push(v.channels[1].clone());
            }),
            ("negative delivered delay", |b| {
                b.disposition.as_mut().unwrap().channels[0].delay_ms = -1.0
            }),
            ("invalid settings rate", |b| b.settings.sample_rate_hz = 0.0),
            ("invalid settings reference", |b| {
                b.settings.reference_level_db = f64::NAN
            }),
        ];
        for (name, mutate) in cases {
            let mut bundle = full_bundle();
            mutate(&mut bundle);
            // Rebinding the settings hash is not a substitute for validation.
            let hash = bundle.settings.settings_hash();
            bundle.magnitude.as_mut().unwrap().settings_hash = hash.clone();
            bundle.ir_step.as_mut().unwrap().settings_hash = hash.clone();
            bundle.etc.as_mut().unwrap().settings_hash = hash.clone();
            bundle.decay.as_mut().unwrap().settings_hash = hash.clone();
            bundle.ambient_noise.as_mut().unwrap().settings_hash = hash.clone();
            bundle.headroom.as_mut().unwrap().settings_hash = hash.clone();
            bundle.disposition.as_mut().unwrap().settings_hash = hash;
            let gate = evaluate_bundle_gate(&bundle);
            assert!(
                !gate.passed && !gate.failures.is_empty(),
                "{name} must fail on content, not only missing views or hash mismatch"
            );
        }
    }

    #[test]
    fn roadmap_correction_bundle_legacy_output_plan_is_unknown_not_pass() {
        let mut value = serde_json::to_value(full_bundle()).unwrap();
        value.as_object_mut().unwrap().remove("required_output_ids");
        let legacy: AcceptanceBundle = serde_json::from_value(value).unwrap();
        assert!(legacy.required_output_ids.is_empty());
        let gate = evaluate_bundle_gate(&legacy);
        assert!(!gate.passed);
        assert!(
            gate.failures
                .iter()
                .any(|reason| reason.contains("physical-output plan"))
        );
    }

    /// Each view carries processing provenance with matched settings.
    #[test]
    fn bundle_matched_settings_enforced() {
        let mut bundle = full_bundle();
        bundle.magnitude.as_mut().unwrap().settings_hash = "other".to_string();
        let gate = evaluate_bundle_gate(&bundle);
        assert!(!gate.passed);
        assert!(gate.failures.iter().any(|f| f.contains("magnitude")));
    }

    /// A smoother magnitude plot alone never passes a gate.
    #[test]
    fn bundle_smoother_plot_alone_never_passes() {
        let settings = ViewSettings::default();
        let bundle = AcceptanceBundle {
            magnitude: Some(magnitude_view(&settings)),
            settings,
            ..Default::default()
        };
        let gate = evaluate_bundle_gate(&bundle);
        assert!(!gate.passed);
        assert_eq!(gate.missing_views.len(), 6);
        assert!(gate.failures.iter().any(|f| f.contains("alone")));
    }

    /// Each stage fact is counted exactly once: gains and delays sum over
    /// their own stages, inversion is one flag with its stage count, and
    /// FIR references are descriptive. Reports and DSP built from the same
    /// bytes therefore agree.
    #[test]
    fn roadmap_correction_disposition_sums_each_stage_once() {
        let settings = ViewSettings::default();
        let view = fixture_disposition(&settings);
        assert_eq!(view.channels.len(), 2);
        let left = view
            .channels
            .iter()
            .find(|disposition| disposition.channel == "left")
            .expect("left disposition");
        assert_eq!(left.gain_db, -3.0);
        assert!(left.inverted);
        assert_eq!(left.invert_stage_count, 1);
        assert_eq!(left.delay_ms, 2.5);
        assert!(!left.muted);
        assert_eq!(left.route, vec!["gain", "delay", "convolution"]);
        assert_eq!(left.fir_references, vec!["left.wav"]);
        assert!(!left.has_snapshot_curve);
        assert!(!left.has_target_curve);
        let right = view
            .channels
            .iter()
            .find(|disposition| disposition.channel == "right")
            .expect("right disposition");
        assert_eq!(right.gain_db, 1.5);
        assert!(!right.inverted);
        assert!(right.fir_references.is_empty());
        assert!(
            view.unavailable
                .iter()
                .any(|reason| reason.contains("mute is not a graph stage"))
        );
        assert!(
            view.unavailable
                .iter()
                .any(|reason| reason.contains("tap counts"))
        );
        assert!(!view.graph_identity.is_empty());
    }

    #[test]
    fn roadmap_correction_disposition_composes_polarity_and_rejects_overflow() {
        let mut graph = fixture_graph();
        graph
            .channels
            .get_mut("left")
            .unwrap()
            .plugins
            .push(roomeq_model::PluginConfigWrapper {
                plugin_type: "gain".into(),
                parameters: serde_json::json!({"gain_db": 0.0, "invert": true}),
            });
        let mut capture = provenance();
        capture.graph_identity =
            roomeq_model::decision_ledger::canonical_graph_identity(&graph).fingerprint;
        let view = derive_dsp_dispositions(
            &graph,
            capture.clone(),
            ViewSettings::default().settings_hash(),
        )
        .unwrap();
        assert!(
            !view.channels[0].inverted,
            "two inversions cancel in the delivered channel"
        );
        assert_eq!(view.channels[0].invert_stage_count, 2);
        for _ in 0..2 {
            graph.channels.get_mut("left").unwrap().plugins.push(
                roomeq_model::PluginConfigWrapper {
                    plugin_type: "gain".into(),
                    parameters: serde_json::json!({"gain_db": f64::MAX}),
                },
            );
        }
        capture.graph_identity =
            roomeq_model::decision_ledger::canonical_graph_identity(&graph).fingerprint;
        let error =
            derive_dsp_dispositions(&graph, capture, ViewSettings::default().settings_hash())
                .unwrap_err();
        assert!(error.contains("nonfinite aggregate"));
    }

    /// Unmeasurable chains are unavailable, never half-derived: an empty
    /// graph and a gain stage without a finite parameter both refuse.
    #[test]
    fn roadmap_correction_disposition_refuses_unmeasurable_chain() {
        use roomeq_model::Plugin;
        let settings = ViewSettings::default();
        let hash = settings.settings_hash();
        let empty = roomeq_model::DspGraph::new("test");
        assert!(derive_dsp_dispositions(&empty, provenance(), hash.clone()).is_err());
        let mut broken = roomeq_model::DspGraph::new("test");
        broken.add_channel(
            "left",
            vec![Plugin {
                kind: "gain".to_string(),
                parameters: serde_json::json!({"gain_db": "loud"}),
            }],
        );
        let error = derive_dsp_dispositions(&broken, provenance(), hash)
            .expect_err("nonfinite gain must refuse");
        assert!(error.contains("lacks a finite 'gain_db'"), "{error}");
    }

    /// Rebuilding replaces the view: the later graph wins with no stale
    /// merge, so reports always describe the bytes they ship.
    #[test]
    fn roadmap_correction_disposition_rebuild_replaces() {
        let settings = ViewSettings::default();
        let hash = settings.settings_hash();
        let first = derive_dsp_dispositions(&fixture_graph(), provenance(), hash.clone())
            .expect("first derivation");
        let mut changed = fixture_graph();
        changed
            .channels
            .get_mut("left")
            .expect("left channel")
            .plugins
            .push(roomeq_model::PluginConfigWrapper {
                plugin_type: "gain".to_string(),
                parameters: serde_json::json!({"gain_db": 2.0}),
            });
        assert!(
            derive_dsp_dispositions(&changed, provenance(), hash.clone()).is_err(),
            "stale provenance must be rejected"
        );
        let mut updated_provenance = provenance();
        updated_provenance.graph_identity =
            roomeq_model::decision_ledger::canonical_graph_identity(&changed).fingerprint;
        let second =
            derive_dsp_dispositions(&changed, updated_provenance, hash).expect("second derivation");
        let first_left = first
            .channels
            .iter()
            .find(|disposition| disposition.channel == "left")
            .unwrap();
        let second_left = second
            .channels
            .iter()
            .find(|disposition| disposition.channel == "left")
            .unwrap();
        assert_eq!(first_left.gain_db, -3.0);
        assert_eq!(second_left.gain_db, -1.0);
        assert_ne!(first.graph_identity, second.graph_identity);
    }

    /// An absent disposition view is named by the gate, never hidden by
    /// omission.
    #[test]
    fn roadmap_correction_disposition_absent_view_named() {
        let mut bundle = full_bundle();
        bundle.disposition = None;
        let gate = evaluate_bundle_gate(&bundle);
        assert!(!gate.passed);
        assert!(gate.missing_views.contains(&"disposition"));
    }

    /// Reduced playback ringing is never reported as changed passive damping.
    #[test]
    fn bundle_playback_ringing_separate_from_room_damping() {
        let bundle = full_bundle();
        let decay = bundle.decay.as_ref().unwrap();
        assert!(decay.playback_tail_change_db < 0.0);
        // The passive room estimate is an independent input: halving the
        // reported tail must not rewrite it.
        let mut quieter = decay.clone();
        quieter.playback_tail_change_db *= 2.0;
        assert_eq!(quieter.passive_room_rt_s, decay.passive_room_rt_s);
    }

    /// Shared safe output: digital peak below the ceiling and physical
    /// demand inside both limits under one calibration and trial.
    fn safe_output(name: &str) -> OutputHeadroomInput {
        let level = |value_db_spl: f64| CalibratedLevel {
            value_db_spl,
            calibration_id: "spl-cal-94db".to_string(),
            trial: HeadroomTrial::SmallSignal,
        };
        OutputHeadroomInput {
            output_name: name.to_string(),
            true_peak_dbfs: Some(-3.0),
            demand_db_spl: Some(level(100.0)),
            amplifier_limit_db_spl: Some(level(106.0)),
            driver_limit_db_spl: Some(level(103.0)),
        }
    }

    #[test]
    fn roadmap_correction_headroom_rejects_empty_calibration_and_overflow() {
        for id in ["", " ", " Uncalibrated "] {
            let mut input = safe_output("sub1");
            for value in [
                &mut input.demand_db_spl,
                &mut input.amplifier_limit_db_spl,
                &mut input.driver_limit_db_spl,
            ] {
                value.as_mut().unwrap().calibration_id = id.into();
            }
            assert_eq!(
                assess_output_headroom(0.0, &input).verdict,
                HeadroomVerdict::Unknown,
                "{id:?}"
            );
        }
        let mut input = safe_output("sub1");
        input.true_peak_dbfs = Some(-f64::MAX);
        let assessment = assess_output_headroom(f64::MAX, &input);
        assert_eq!(assessment.verdict, HeadroomVerdict::Unknown);
        assert_eq!(assessment.digital_margin_db, None);
        let mut input = safe_output("sub1");
        input.demand_db_spl.as_mut().unwrap().value_db_spl = -f64::MAX;
        input.amplifier_limit_db_spl.as_mut().unwrap().value_db_spl = f64::MAX;
        let assessment = assess_output_headroom(0.0, &input);
        assert_eq!(assessment.verdict, HeadroomVerdict::Unknown);
        assert_eq!(assessment.amplifier_margin_db, None);
    }

    #[test]
    fn roadmap_correction_noise_rejects_uncalibrated_sentinel_variants() {
        for calibration in ["", " ", "uncalibrated", " Uncalibrated "] {
            let mut bundle = full_bundle();
            let noise = bundle.ambient_noise.as_mut().unwrap();
            noise.calibration = calibration.into();
            noise.provenance.calibration = calibration.into();
            assert!(!evaluate_bundle_gate(&bundle).passed, "{calibration:?}");
        }
    }

    /// Full routed-chain headroom needs true peaks and physical limits.
    #[test]
    fn bundle_headroom_needs_true_peaks_and_limits() {
        let settings = ViewSettings::default();
        let hash = settings.settings_hash();
        let per_filter_only = assess_routed_headroom(
            provenance(),
            hash.clone(),
            DIGITAL_CEILING_DBFS,
            0.0,
            vec![],
        );
        assert!(!per_filter_only.passes);
        let with_limits = assess_routed_headroom(
            provenance(),
            hash,
            DIGITAL_CEILING_DBFS,
            0.0,
            vec![safe_output("sub1")],
        );
        assert!(with_limits.passes, "note: {}", with_limits.note);
        let mut bundle = full_bundle();
        bundle.headroom = Some(per_filter_only);
        assert!(!evaluate_bundle_gate(&bundle).passed);
    }

    #[test]
    fn bundle_uncalibrated_noise_fails() {
        let mut bundle = full_bundle();
        let noise = bundle.ambient_noise.as_mut().unwrap();
        noise.calibration = "uncalibrated".to_string();
        let gate = evaluate_bundle_gate(&bundle);
        assert!(!gate.passed);
        assert!(gate.failures.iter().any(|f| f.contains("uncalibrated")));
    }

    fn assess_one(input: OutputHeadroomInput) -> RoutedHeadroomView {
        let settings = ViewSettings::default();
        assess_routed_headroom(
            provenance(),
            settings.settings_hash(),
            DIGITAL_CEILING_DBFS,
            0.0,
            vec![input],
        )
    }

    /// A1 regression: a finite true peak above the digital ceiling must not
    /// pass, even when physical limits are present.
    #[test]
    fn roadmap_correction_headroom_over_ceiling_fails() {
        let mut input = safe_output("sub1");
        input.true_peak_dbfs = Some(3.0);
        let view = assess_one(input);
        assert!(!view.passes, "over-ceiling peak passed: {}", view.note);
        assert_eq!(derive_headroom_verdict(&view).0, HeadroomVerdict::Fail);
    }

    /// Below, exactly at, and above the explicit ceiling.
    #[test]
    fn roadmap_correction_headroom_ceiling_boundaries() {
        let mut below = safe_output("sub1");
        below.true_peak_dbfs = Some(-0.5);
        assert!(assess_one(below).passes);
        // Exact equality meets the ceiling.
        let mut equal = safe_output("sub1");
        equal.true_peak_dbfs = Some(DIGITAL_CEILING_DBFS);
        assert!(assess_one(equal).passes);
        // Epsilon-scale excess still meets the ceiling; real excess fails.
        let mut epsilon = safe_output("sub1");
        epsilon.true_peak_dbfs = Some(DIGITAL_CEILING_DBFS + HEADROOM_EPSILON_DB / 2.0);
        assert!(assess_one(epsilon).passes);
        let mut above = safe_output("sub1");
        above.true_peak_dbfs = Some(DIGITAL_CEILING_DBFS + 2.0 * HEADROOM_EPSILON_DB);
        assert!(!assess_one(above).passes);
    }

    /// Physical demand over either limit fails; exact equality passes.
    #[test]
    fn roadmap_correction_headroom_physical_boundaries() {
        let mut over_amp = safe_output("sub1");
        over_amp.demand_db_spl.as_mut().unwrap().value_db_spl = 107.0;
        let view = assess_one(over_amp);
        assert!(!view.passes);
        assert!(view.note.contains("amplifier"));
        let mut equal = safe_output("sub1");
        equal.demand_db_spl.as_mut().unwrap().value_db_spl = 103.0;
        assert!(assess_one(equal).passes);
    }

    /// Missing calibration, missing limit, and NaN/infinity never pass.
    #[test]
    fn roadmap_correction_headroom_incomplete_evidence_unknown() {
        let mut no_cal = safe_output("sub1");
        no_cal.demand_db_spl.as_mut().unwrap().calibration_id = "uncalibrated".to_string();
        let view = assess_one(no_cal);
        assert!(!view.passes);
        assert_eq!(derive_headroom_verdict(&view).0, HeadroomVerdict::Unknown);

        let mut no_limit = safe_output("sub1");
        no_limit.amplifier_limit_db_spl = None;
        no_limit.driver_limit_db_spl = None;
        let view = assess_one(no_limit);
        assert!(!view.passes);
        assert_eq!(derive_headroom_verdict(&view).0, HeadroomVerdict::Unknown);

        let mut nan_peak = safe_output("sub1");
        nan_peak.true_peak_dbfs = Some(f64::NAN);
        assert!(!assess_one(nan_peak).passes);
        let mut inf_demand = safe_output("sub1");
        inf_demand.demand_db_spl.as_mut().unwrap().value_db_spl = f64::INFINITY;
        assert!(!assess_one(inf_demand).passes);

        // Calibration or trial mismatch is unknown, never a pass.
        let mut cal_mismatch = safe_output("sub1");
        cal_mismatch
            .amplifier_limit_db_spl
            .as_mut()
            .unwrap()
            .calibration_id = "other-cal".to_string();
        assert!(!assess_one(cal_mismatch).passes);
        let mut trial_mismatch = safe_output("sub1");
        trial_mismatch.driver_limit_db_spl.as_mut().unwrap().trial = HeadroomTrial::LimiterTrial;
        assert!(!assess_one(trial_mismatch).passes);
    }

    /// One overloaded output among safe outputs fails the whole view.
    #[test]
    fn roadmap_correction_headroom_one_bad_output_fails_all() {
        let settings = ViewSettings::default();
        let mut bad = safe_output("sub2");
        bad.true_peak_dbfs = Some(1.5);
        let view = assess_routed_headroom(
            provenance(),
            settings.settings_hash(),
            DIGITAL_CEILING_DBFS,
            0.0,
            vec![safe_output("sub1"), bad],
        );
        assert!(!view.passes);
        assert!(view.note.contains("sub2"));
        // Margins of the safe output still survive for the report.
        let safe = view
            .outputs
            .iter()
            .find(|o| o.output_name == "sub1")
            .unwrap();
        assert_eq!(safe.verdict, HeadroomVerdict::Pass);
        assert!(safe.digital_margin_db.unwrap() > 0.0);
        assert!(safe.amplifier_margin_db.unwrap() > 0.0);
    }

    /// A tampered `passes: true` over failing evidence is rejected by the gate.
    #[test]
    fn roadmap_correction_headroom_tampered_pass_rejected() {
        let mut input = safe_output("sub1");
        input.true_peak_dbfs = Some(3.0);
        let mut view = assess_one(input);
        assert!(!view.passes);
        view.passes = true;
        view.note = "tampered".to_string();
        let (verdict, consistent) = derive_headroom_verdict(&view);
        assert_eq!(verdict, HeadroomVerdict::Fail);
        assert!(!consistent);
        let settings = ViewSettings::default();
        let bundle = AcceptanceBundle {
            headroom: Some(view),
            settings,
            ..Default::default()
        };
        let gate = evaluate_bundle_gate(&bundle);
        assert!(!gate.passed);
    }

    /// A safe fully specified example passes the whole bundle gate, and its
    /// values, units, margins, and reasons survive a serialization round trip.
    #[test]
    fn roadmap_correction_headroom_safe_example_round_trip() {
        let bundle = full_bundle();
        let gate = evaluate_bundle_gate(&bundle);
        assert!(gate.passed, "failures: {:?}", gate.failures);
        let json = serde_json::to_string(&bundle).unwrap();
        let back: AcceptanceBundle = serde_json::from_str(&json).unwrap();
        assert_eq!(back, bundle);
        let gate = evaluate_bundle_gate(&back);
        assert!(gate.passed, "failures: {:?}", gate.failures);
        let headroom = back.headroom.as_ref().unwrap();
        let output = &headroom.outputs[0];
        assert_eq!(
            output.input.demand_db_spl.as_ref().unwrap().calibration_id,
            "spl-cal-94db"
        );
        assert!(output.digital_margin_db.unwrap() > 0.0);
    }

    #[test]
    fn bundle_step_response_is_cumulative_ir() {
        let step = step_response(&[1.0, 0.5, 0.25]);
        assert_eq!(step, vec![1.0, 1.5, 1.75]);
        assert!(step_response(&[]).is_empty());
    }

    #[test]
    fn bundle_hilbert_envelope_of_tone_is_flat() {
        let tone: Vec<f64> = (0..256)
            .map(|i| (2.0 * std::f64::consts::PI * i as f64 / 32.0).sin())
            .collect();
        let envelope = hilbert_envelope(&tone);
        assert_eq!(envelope.len(), tone.len());
        let interior = &envelope[32..224];
        let mean = interior.iter().sum::<f64>() / interior.len() as f64;
        assert!((mean - 1.0).abs() < 0.05, "envelope mean was {mean}");
        let spread = interior
            .iter()
            .fold(0.0_f64, |a, b| a.max((b - mean).abs()));
        assert!(spread < 0.1, "envelope ripple was {spread}");
    }

    #[test]
    fn bundle_etc_peak_at_direct_sound() {
        let sample_rate = 48000.0;
        let mut ir = vec![0.0; 512];
        ir[10] = 1.0;
        ir[100] = 0.3;
        let envelope = octave_etc_envelope(&ir, sample_rate, 1000.0);
        assert_eq!(envelope.len(), ir.len());
        let peak = envelope
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap();
        assert!((peak as i64 - 10).abs() <= 4, "ETC peak at {peak}");
        assert!(envelope[0] <= 0.5);
    }

    #[test]
    fn bundle_schroeder_decay_fits_known_rt() {
        // Exponential decay with T60 = 0.5 s sampled at 1 kHz.
        let sample_rate = 1000.0;
        let t60 = 0.5;
        let ir: Vec<f64> = (0..1000)
            .map(|i| {
                let t = i as f64 / sample_rate;
                10.0_f64.powf(-3.0 * t / t60) * (2.0 * std::f64::consts::PI * 50.0 * t).sin()
            })
            .collect();
        let decay = schroeder_decay_db(&ir);
        let times: Vec<f64> = (0..1000).map(|i| i as f64 / sample_rate).collect();
        let (fit, r_squared) = fit_t60(&times, &decay, -5.0, -25.0);
        assert!(r_squared > 0.98, "R² was {r_squared}");
        assert!((fit - t60).abs() < 0.05, "T60 fit was {fit}");
    }
}
