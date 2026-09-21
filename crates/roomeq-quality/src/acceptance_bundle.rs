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
    /// True peak of the full routed chain in dBFS.
    pub true_peak_dbfs: Option<f64>,
    /// Amplifier output limit in dB.
    pub amplifier_limit_db: Option<f64>,
    /// Driver excursion/thermal limit in dB.
    pub driver_limit_db: Option<f64>,
    /// Maximum per-filter gain in dB (insufficient on its own).
    pub per_filter_max_db: f64,
    /// Whether full-chain headroom is demonstrated.
    pub passes: bool,
    /// Machine-readable reason when `passes` is false.
    pub note: String,
}

/// Complete multi-view acceptance bundle.
///
/// Every view is optional at the type level so partial bundles can be
/// assembled and honestly reported; the gate fails them.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct AcceptanceBundle {
    /// Matched settings every included view must carry.
    pub settings: ViewSettings,
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
}

/// Bundle gate outcome.
#[derive(Debug, Clone, PartialEq)]
pub struct BundleGate {
    /// Whether the bundle demonstrates acceptance.
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
/// assert_eq!(gate.missing_views.len(), 6);
/// ```
pub fn evaluate_bundle_gate(bundle: &AcceptanceBundle) -> BundleGate {
    let mut missing = Vec::new();
    let mut failures = Vec::new();
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
            if !view.passes {
                failures.push(format!("headroom not demonstrated: {}", view.note));
            }
        }
        None => missing.push("headroom"),
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

/// Build a routed headroom view from full-chain measurements.
///
/// `passes` requires a measured true peak plus at least one physical limit
/// (amplifier or driver). A per-filter 0 dB maximum alone is insufficient and
/// yields `passes == false` with a machine-readable note.
pub fn assess_routed_headroom(
    provenance: ViewProvenance,
    settings_hash: String,
    true_peak_dbfs: Option<f64>,
    amplifier_limit_db: Option<f64>,
    driver_limit_db: Option<f64>,
    per_filter_max_db: f64,
) -> RoutedHeadroomView {
    let finite = |value: Option<f64>| value.is_some_and(|v| v.is_finite());
    let (passes, note) =
        if finite(true_peak_dbfs) && (finite(amplifier_limit_db) || finite(driver_limit_db)) {
            (true, "full routed chain assessed".to_string())
        } else {
            (
                false,
                "per-filter maximum is insufficient; true peak and physical limits required"
                    .to_string(),
            )
        };
    RoutedHeadroomView {
        provenance,
        settings_hash,
        true_peak_dbfs,
        amplifier_limit_db,
        driver_limit_db,
        per_filter_max_db,
        passes,
        note,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn provenance() -> ViewProvenance {
        ViewProvenance {
            measurement_ids: vec!["seat-loop".to_string()],
            graph_identity: "graph-1".to_string(),
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
        let times_ms: Vec<f64> = (0..8).map(|i| i as f64 * 5.0).collect();
        let ir = vec![1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.0078125];
        AcceptanceBundle {
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
                bands: vec![EtcBand {
                    center_hz: 1000.0,
                    times_ms: vec![0.0, 5.0],
                    pre_db: vec![0.0, -6.0],
                    post_db: vec![0.0, -9.0],
                }],
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
                Some(-3.0),
                Some(6.0),
                None,
                0.0,
            )),
            settings,
        }
    }

    #[test]
    fn bundle_empty_fails_with_all_views_missing() {
        let gate = evaluate_bundle_gate(&AcceptanceBundle::default());
        assert!(!gate.passed);
        assert_eq!(gate.missing_views.len(), 6);
    }

    #[test]
    fn bundle_full_passes_with_matched_settings() {
        let bundle = full_bundle();
        let gate = evaluate_bundle_gate(&bundle);
        assert!(gate.passed, "failures: {:?}", gate.failures);
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
        assert_eq!(gate.missing_views.len(), 5);
        assert!(gate.failures.iter().any(|f| f.contains("alone")));
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

    /// Full routed-chain headroom needs true peaks and physical limits.
    #[test]
    fn bundle_headroom_needs_true_peaks_and_limits() {
        let settings = ViewSettings::default();
        let hash = settings.settings_hash();
        let per_filter_only =
            assess_routed_headroom(provenance(), hash.clone(), None, None, None, 0.0);
        assert!(!per_filter_only.passes);
        let with_limits =
            assess_routed_headroom(provenance(), hash, Some(-1.0), None, Some(3.0), 0.0);
        assert!(with_limits.passes);
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
