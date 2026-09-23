//! Pure prediction-vs-capture comparison over K5 playback bindings.
//!
//! Scores independently supplied playback evidence against a prediction
//! without recomputing DSP: identity, calibration, routing and alignment
//! compatibility is checked first, and only then are matched supported
//! bands compared. A mismatch yields an unassessed report, never a pass.
//!
//! Only declared gain/delay adjustments are applied, and they are recorded
//! in the ledger. The comparison never searches for fitting values, so
//! output loss and timing error cannot be fitted away to pass.
//!
//! Evidence classes stay distinct: a passing simulated or backend-rendered
//! comparison is software behavior only. Only a passing acoustic capture at
//! the claimed level counts toward playback verification, and a small-signal
//! transfer check never supports a maximum-output claim.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use autoeq_core::Curve;

use super::metrics::weighted_mean;

/// What the playback evidence is. Distinct from the K1 acquisition
/// `CaptureKind` in `roomeq-model` (stationary IR, spatial magnitude,
/// direct sound, unknown): this names the verification class of the
/// comparison, not how the measurement was acquired. Simulated and
/// backend-rendered evidence test software behavior only; neither is
/// acoustic verification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum PlaybackEvidenceKind {
    /// Numerically simulated capture (e.g. convolved model output).
    Simulated,
    /// Rendered through the real export backend without acoustic playback.
    BackendRendered,
    /// Measured acoustic playback in the room.
    Acoustic,
}

impl PlaybackEvidenceKind {
    /// Evidence-class label recorded on every report.
    pub fn as_str(self) -> &'static str {
        match self {
            PlaybackEvidenceKind::Simulated => "simulated",
            PlaybackEvidenceKind::BackendRendered => "backend-rendered",
            PlaybackEvidenceKind::Acoustic => "acoustic",
        }
    }
}

/// Signal level the capture was taken at. Level-dependent compression and
/// limiter behavior are outside a small-signal transfer check.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum PlaybackLevel {
    /// Linear transfer measurement; says nothing about maximum output.
    SmallSignal,
    /// Driven level assessment for compression/limiter behavior.
    MaximumOutput,
}

/// Immutable K5 binding of one side of the comparison. Both sides must
/// agree on every field before any metric is scored.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct PlaybackBinding {
    /// Immutable delivered-graph identity the prediction was rendered from.
    pub graph_id: String,
    /// Logical source driving the playback.
    pub source_id: String,
    /// Seat the capture was taken at.
    pub seat_id: String,
    /// Hash of the stimulus actually played.
    pub stimulus_hash: String,
    /// Sample rate of prediction and capture chains.
    pub sample_rate_hz: f64,
    /// Calibration identity applied to the capture chain.
    pub calibration_id: String,
    /// Processing state, e.g. `"final-delivered"` vs `"preview"`.
    pub processing_state: String,
}

impl PlaybackBinding {
    /// Fail closed on the first incompatible field. A mismatch never
    /// degrades to a scored comparison with a warning.
    pub fn check_compatible(&self, other: &Self) -> Result<(), String> {
        for binding in [self, other] {
            for (name, value) in [
                ("source", binding.source_id.as_str()),
                ("seat", binding.seat_id.as_str()),
                ("calibration", binding.calibration_id.as_str()),
                ("processing state", binding.processing_state.as_str()),
            ] {
                if value.trim().is_empty() {
                    return Err(format!("capture binding needs a nonempty {name} identity"));
                }
            }
            if !binding.sample_rate_hz.is_finite() || binding.sample_rate_hz <= 0.0 {
                return Err("capture binding needs a finite positive sample rate".to_owned());
            }
        }
        if self.graph_id.trim().is_empty() || other.graph_id.trim().is_empty() {
            return Err(String::from(
                "capture binding needs graph identities on both sides",
            ));
        }
        if self.graph_id != other.graph_id {
            return Err(String::from("mismatched graph identity"));
        }
        if self.source_id != other.source_id {
            return Err(String::from("mismatched source mapping"));
        }
        if self.seat_id != other.seat_id {
            return Err(String::from("mismatched seat mapping"));
        }
        if self.stimulus_hash.trim().is_empty() || other.stimulus_hash.trim().is_empty() {
            return Err(String::from(
                "capture binding needs stimulus hashes on both sides",
            ));
        }
        if self.stimulus_hash != other.stimulus_hash {
            return Err(String::from("mismatched stimulus hash"));
        }
        if self.sample_rate_hz != other.sample_rate_hz {
            return Err(String::from("mismatched sample rate"));
        }
        if self.calibration_id != other.calibration_id {
            return Err(String::from("mismatched calibration identity"));
        }
        if self.processing_state != other.processing_state {
            return Err(String::from("mismatched processing state"));
        }
        Ok(())
    }
}

/// The only alignment the comparison may apply. Declared up front by the
/// operator, recorded in the ledger, never fitted to the data.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct DeclaredAlignment {
    /// Authorized broadband gain trim in dB.
    pub gain_db: f64,
    /// Authorized bulk delay in ms.
    pub delay_ms: f64,
}

/// Per-metric tolerances fixed before the comparison. There are no default
/// tolerances: every comparison declares what it will accept.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct CaptureTolerances {
    /// Largest accepted peak magnitude residual in dB after declared gain.
    pub max_magnitude_deviation_db: f64,
    /// Largest accepted bulk-delay error in ms after declared delay.
    pub max_timing_error_ms: f64,
    /// Largest accepted broadband output loss in dB beyond declared gain.
    pub max_output_loss_db: f64,
}

/// Requested band without support on both sides, retained with its reason.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct ExcludedBand {
    /// Requested sub-band with no matched support.
    pub band_hz: [f64; 2],
    /// Why the band was excluded, e.g. `"no capture support above 8000 Hz"`.
    pub reason: String,
}

/// Outcome of one comparison metric.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct CaptureMetricOutcome {
    /// Metric name, e.g. `"magnitude_agreement"`.
    pub metric: String,
    /// False when the evidence for this metric is missing; the metric is
    /// then unassessed, never zero or passing by default.
    pub assessed: bool,
    /// Observed value in the metric's units (`None` when unassessed).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub observed: Option<f64>,
    /// Tolerance applied (`None` when unassessed).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tolerance: Option<f64>,
    /// True only when assessed and within tolerance.
    pub passed: bool,
    /// Supporting detail, e.g. sample counts or the unassessed reason.
    #[serde(default)]
    pub detail: String,
}

/// Pure prediction-vs-capture comparison report.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct CaptureComparisonReport {
    /// False when identities mismatch or no band is jointly supported.
    /// Unassessed evidence must not promote any playback claim.
    pub assessed: bool,
    /// Why the comparison is unassessed (`None` when assessed).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub unassessed_reason: Option<String>,
    /// Evidence class: `"simulated"`, `"backend-rendered"` or `"acoustic"`.
    pub evidence_class: String,
    /// Level the capture was taken at: `"small_signal"` or `"maximum_output"`.
    pub playback_level: String,
    /// Jointly supported band actually compared (`None` when unassessed).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub evaluated_band_hz: Option<[f64; 2]>,
    /// Requested band portions without matched support.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub excluded_bands: Vec<ExcludedBand>,
    /// Declared adjustments actually applied, plus the no-fit statement.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub ledger: Vec<String>,
    /// Per-metric outcomes; unassessed metrics stay visible here.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub metric_outcomes: Vec<CaptureMetricOutcome>,
    /// True only when assessed and every assessed metric passed.
    pub passed: bool,
}

impl CaptureComparisonReport {
    /// Require complete supported evidence before promoting a comparison result.
    ///
    /// `passed` alone only summarizes assessed metrics. Missing timing, excluded
    /// bands, absent numeric results, or inconsistent serialized pass flags must
    /// not authorize playback verification.
    pub fn is_complete_pass(&self) -> bool {
        self.assessed
            && self.passed
            && self.unassessed_reason.is_none()
            && self.excluded_bands.is_empty()
            && self.evaluated_band_hz.is_some_and(|band| {
                band[0].is_finite() && band[1].is_finite() && band[0] > 0.0 && band[1] > band[0]
            })
            && ["magnitude_agreement", "useful_output", "timing_agreement"]
                .iter()
                .all(|name| {
                    self.metric_outcomes
                        .iter()
                        .filter(|metric| metric.metric == *name)
                        .count()
                        == 1
                })
            && self.metric_outcomes.iter().all(|metric| {
                metric.assessed
                    && metric.passed
                    && metric
                        .observed
                        .zip(metric.tolerance)
                        .is_some_and(|(observed, tolerance)| {
                            observed.is_finite()
                                && tolerance.is_finite()
                                && observed >= 0.0
                                && tolerance >= 0.0
                                && observed <= tolerance
                        })
            })
    }

    /// True only for a passing acoustic capture. Simulated and
    /// backend-rendered passes test software behavior, never the room.
    pub fn counts_as_acoustic_verification(&self) -> bool {
        self.is_complete_pass() && self.evidence_class == PlaybackEvidenceKind::Acoustic.as_str()
    }

    /// True only for a passing maximum-output acoustic capture. A
    /// small-signal transfer check never supports a maximum-output claim.
    pub fn supports_maximum_output_claim(&self) -> bool {
        self.counts_as_acoustic_verification() && self.playback_level == "maximum_output"
    }
}

/// Compare a prediction against an independently supplied capture.
///
/// Compatibility first: any binding mismatch returns an unassessed report.
/// Only the jointly supported overlap of `band_hz` is compared; the rest is
/// retained in `excluded_bands`. Magnitude and output residuals apply the
/// declared gain only, and timing is checked against the declared delay
/// only — no value is fitted to make the comparison pass.
#[allow(clippy::too_many_arguments)]
pub fn compare_prediction_capture(
    prediction: &Curve,
    capture: &Curve,
    predicted: &PlaybackBinding,
    captured: &PlaybackBinding,
    capture_kind: PlaybackEvidenceKind,
    level: PlaybackLevel,
    declared: &DeclaredAlignment,
    tolerances: &CaptureTolerances,
    band_hz: [f64; 2],
) -> Result<CaptureComparisonReport, String> {
    if !band_hz[0].is_finite()
        || !band_hz[1].is_finite()
        || band_hz[0] <= 0.0
        || band_hz[1] <= band_hz[0]
    {
        return Err(String::from("invalid capture comparison band"));
    }
    for (name, value) in [
        ("declared gain", declared.gain_db),
        ("declared delay", declared.delay_ms),
    ] {
        if !value.is_finite() {
            return Err(format!("invalid capture comparison {name}"));
        }
    }
    for (name, value) in [
        ("magnitude tolerance", tolerances.max_magnitude_deviation_db),
        ("timing tolerance", tolerances.max_timing_error_ms),
        ("output tolerance", tolerances.max_output_loss_db),
    ] {
        if !value.is_finite() || value < 0.0 {
            return Err(format!("invalid capture comparison {name}"));
        }
    }
    let unassessed = |reason: String| {
        Ok(CaptureComparisonReport {
            assessed: false,
            unassessed_reason: Some(reason),
            evidence_class: capture_kind.as_str().into(),
            playback_level: match level {
                PlaybackLevel::SmallSignal => "small_signal".into(),
                PlaybackLevel::MaximumOutput => "maximum_output".into(),
            },
            evaluated_band_hz: None,
            excluded_bands: Vec::new(),
            ledger: Vec::new(),
            metric_outcomes: Vec::new(),
            passed: false,
        })
    };
    if let Err(mismatch) = predicted.check_compatible(captured) {
        return unassessed(format!("capture identity mismatch: {mismatch}"));
    }
    prediction
        .validate("prediction")
        .map_err(|error| error.to_string())?;
    capture
        .validate("capture")
        .map_err(|error| error.to_string())?;

    let overlap_low = band_hz[0]
        .max(first_freq(prediction)?)
        .max(first_freq(capture)?);
    let overlap_high = band_hz[1]
        .min(last_freq(prediction)?)
        .min(last_freq(capture)?);
    if overlap_high <= overlap_low {
        return unassessed(String::from(
            "no jointly supported band in the requested range",
        ));
    }
    let mut frequencies: Vec<f64> = prediction
        .freq
        .iter()
        .chain(capture.freq.iter())
        .copied()
        .filter(|frequency| *frequency >= overlap_low && *frequency <= overlap_high)
        .collect();
    frequencies.sort_by(f64::total_cmp);
    frequencies.dedup();
    if frequencies.len() < 2 {
        return unassessed(String::from(
            "jointly supported band needs at least two bins",
        ));
    }

    let mut excluded_bands = Vec::new();
    if overlap_low > band_hz[0] {
        excluded_bands.push(ExcludedBand {
            band_hz: [band_hz[0], overlap_low],
            reason: String::from("no matched prediction/capture support below the overlap"),
        });
    }
    if overlap_high < band_hz[1] {
        excluded_bands.push(ExcludedBand {
            band_hz: [overlap_high, band_hz[1]],
            reason: String::from("no matched prediction/capture support above the overlap"),
        });
    }

    let predicted_mag: Vec<f64> = frequencies
        .iter()
        .map(|frequency| sample_log(prediction, *frequency, &prediction.spl))
        .collect();
    let captured_mag: Vec<f64> = frequencies
        .iter()
        .map(|frequency| sample_log(capture, *frequency, &capture.spl))
        .collect();
    let residuals: Vec<f64> = captured_mag
        .iter()
        .zip(&predicted_mag)
        .map(|(capture, prediction)| capture - prediction - declared.gain_db)
        .collect();
    let worst_magnitude = residuals
        .iter()
        .map(|value| value.abs())
        .fold(0.0, f64::max);
    let mean_change =
        weighted_mean(&frequencies, &captured_mag) - weighted_mean(&frequencies, &predicted_mag);
    // Loss beyond the declared trim only: extra level is not lost output,
    // and the declared trim itself is authorized, not fitted.
    let unexplained_loss = (declared.gain_db - mean_change).max(0.0);

    let mut metric_outcomes = vec![
        CaptureMetricOutcome {
            metric: String::from("magnitude_agreement"),
            assessed: true,
            observed: Some(worst_magnitude),
            tolerance: Some(tolerances.max_magnitude_deviation_db),
            passed: worst_magnitude <= tolerances.max_magnitude_deviation_db,
            detail: format!(
                "peak residual over {} matched bins after declared gain {:+.3} dB; no fitted alignment",
                frequencies.len(),
                declared.gain_db,
            ),
        },
        CaptureMetricOutcome {
            metric: String::from("useful_output"),
            assessed: true,
            observed: Some(unexplained_loss),
            tolerance: Some(tolerances.max_output_loss_db),
            passed: unexplained_loss <= tolerances.max_output_loss_db,
            detail: format!(
                "mean capture change {mean_change:+.3} dB against declared gain {:+.3} dB",
                declared.gain_db,
            ),
        },
    ];
    match bulk_delay_difference_ms(prediction, capture, &frequencies)? {
        TimingSupport::Available(estimated_delay_ms) => {
            let timing_error = (estimated_delay_ms - declared.delay_ms).abs();
            metric_outcomes.push(CaptureMetricOutcome {
                metric: String::from("timing_agreement"),
                assessed: true,
                observed: Some(timing_error),
                tolerance: Some(tolerances.max_timing_error_ms),
                passed: timing_error <= tolerances.max_timing_error_ms,
                detail: format!(
                    "estimated bulk-delay difference {estimated_delay_ms:+.3} ms checked against declared {:+.3} ms; estimate never applied as a correction",
                    declared.delay_ms,
                ),
            });
        }
        TimingSupport::Unassessed(reason) => metric_outcomes.push(CaptureMetricOutcome {
            metric: String::from("timing_agreement"),
            assessed: false,
            observed: None,
            tolerance: None,
            passed: false,
            detail: reason,
        }),
    }

    let passed = metric_outcomes
        .iter()
        .filter(|outcome| outcome.assessed)
        .all(|outcome| outcome.passed);
    Ok(CaptureComparisonReport {
        assessed: true,
        unassessed_reason: None,
        evidence_class: capture_kind.as_str().into(),
        playback_level: match level {
            PlaybackLevel::SmallSignal => "small_signal".into(),
            PlaybackLevel::MaximumOutput => "maximum_output".into(),
        },
        evaluated_band_hz: Some([overlap_low, overlap_high]),
        excluded_bands,
        ledger: vec![
            format!("declared gain {:+.3} dB applied", declared.gain_db),
            format!("declared delay {:+.3} ms checked", declared.delay_ms),
            String::from("no fitted gain or delay: residuals use declared values only"),
        ],
        metric_outcomes,
        passed,
    })
}

/// Timing support for the bulk-delay check.
enum TimingSupport {
    /// Unambiguous slope estimate in ms.
    Available(f64),
    /// Timing left unassessed with its reason, never folded to zero.
    Unassessed(String),
}

/// Bulk-delay difference of capture relative to prediction in ms, from the
/// slope of the unwrapped capture-minus-prediction phase.
///
/// Unassessed when either side lacks phase, and also when adjacent bins
/// rotate past half a cycle on the comparison grid: ordinary unwrapping
/// cannot recover multi-cycle bulk delay on a sparse grid (a long FIR
/// through sparse bins aliases to a small delay), so a folded estimate
/// must never score as agreement. Compare on a grid dense enough for the
/// delays at stake instead.
fn bulk_delay_difference_ms(
    prediction: &Curve,
    capture: &Curve,
    frequencies: &[f64],
) -> Result<TimingSupport, String> {
    let (Some(prediction_phase), Some(capture_phase)) =
        (prediction.phase.as_ref(), capture.phase.as_ref())
    else {
        return Ok(TimingSupport::Unassessed(String::from(
            "phase evidence unavailable on one or both sides: timing is unassessed, not zero",
        )));
    };
    let mut difference: Vec<f64> = frequencies
        .iter()
        .map(|frequency| {
            sample_log(capture, *frequency, capture_phase).to_radians()
                - sample_log(prediction, *frequency, prediction_phase).to_radians()
        })
        .collect();
    let mut aliased = false;
    for index in 1..difference.len() {
        if (difference[index] - difference[index - 1]).abs() > std::f64::consts::PI {
            aliased = true;
        }
        while difference[index] - difference[index - 1] > std::f64::consts::PI {
            difference[index] -= 2.0 * std::f64::consts::PI;
        }
        while difference[index] - difference[index - 1] < -std::f64::consts::PI {
            difference[index] += 2.0 * std::f64::consts::PI;
        }
    }
    if aliased {
        return Ok(TimingSupport::Unassessed(String::from(
            "adjacent phase steps exceed half a cycle on this grid: bulk delay is ambiguous, timing unassessed rather than folded",
        )));
    }
    // Least-squares slope of phase vs frequency; delay is its scaled
    // negation. The estimate is compared against the declared delay and
    // never applied, so it cannot fit timing error away.
    let count = frequencies.len() as f64;
    let sum_x: f64 = frequencies.iter().sum();
    let sum_y: f64 = difference.iter().sum();
    let sum_xx: f64 = frequencies.iter().map(|value| value * value).sum();
    let sum_xy: f64 = frequencies
        .iter()
        .zip(&difference)
        .map(|(x, y)| x * y)
        .sum();
    let denominator = count * sum_xx - sum_x * sum_x;
    if !denominator.is_finite() || denominator <= 0.0 {
        return Err(String::from("capture timing fit is degenerate"));
    }
    let slope = (count * sum_xy - sum_x * sum_y) / denominator;
    if !slope.is_finite() {
        return Err(String::from("capture timing fit is non-finite"));
    }
    Ok(TimingSupport::Available(
        -slope / (2.0 * std::f64::consts::PI) * 1000.0,
    ))
}

fn first_freq(curve: &Curve) -> Result<f64, String> {
    curve
        .freq
        .first()
        .copied()
        .ok_or_else(|| String::from("capture comparison curve has no frequency bins"))
}

fn last_freq(curve: &Curve) -> Result<f64, String> {
    curve
        .freq
        .last()
        .copied()
        .ok_or_else(|| String::from("capture comparison curve has no frequency bins"))
}

fn sample_log(curve: &Curve, frequency: f64, values: &ndarray::Array1<f64>) -> f64 {
    let mut low = 0usize;
    let mut high = curve.freq.len();
    while low < high {
        let middle = low + (high - low) / 2;
        match curve.freq[middle].total_cmp(&frequency) {
            std::cmp::Ordering::Less => low = middle + 1,
            std::cmp::Ordering::Greater => high = middle,
            std::cmp::Ordering::Equal => return values[middle],
        }
    }
    let upper = low.min(curve.freq.len() - 1);
    let lower = upper.saturating_sub(1);
    if lower == upper {
        return values[lower];
    }
    let low_log = curve.freq[lower].ln();
    let high_log = curve.freq[upper].ln();
    let t = (frequency.ln() - low_log) / (high_log - low_log);
    values[lower] + t * (values[upper] - values[lower])
}

#[cfg(test)]
mod capture_tests {
    use super::*;
    use ndarray::Array1;

    #[test]
    fn roadmap_correction_capture_incomplete_metrics_cannot_verify_playback() {
        let frequencies = grid();
        let magnitude_only = curve(&frequencies, &flat(80.0));
        let complex = curve_with_phase(&frequencies, &flat(80.0), &vec![0.0; frequencies.len()]);
        for (capture, band) in [
            (&magnitude_only, [20.0, 20_000.0]),
            (&complex, [10.0, 20_000.0]),
        ] {
            let report = compare_prediction_capture(
                &complex,
                capture,
                &binding(),
                &binding(),
                PlaybackEvidenceKind::Acoustic,
                PlaybackLevel::SmallSignal,
                &DeclaredAlignment {
                    gain_db: 0.0,
                    delay_ms: 0.0,
                },
                &tolerances(),
                band,
            )
            .unwrap();
            assert!(report.passed, "supported magnitude metrics still agree");
            assert!(
                !report.counts_as_acoustic_verification(),
                "missing timing or support is not a full playback pass: {report:?}"
            );
        }
    }

    #[test]
    fn roadmap_correction_capture_equal_missing_bindings_are_not_compatible() {
        for field in [
            "source",
            "seat",
            "calibration",
            "processing",
            "zero_rate",
            "infinite_rate",
        ] {
            let mut value = binding();
            match field {
                "source" => value.source_id.clear(),
                "seat" => value.seat_id = " ".to_owned(),
                "calibration" => value.calibration_id.clear(),
                "processing" => value.processing_state.clear(),
                "zero_rate" => value.sample_rate_hz = 0.0,
                _ => value.sample_rate_hz = f64::INFINITY,
            }
            assert!(value.check_compatible(&value).is_err(), "{field}");
        }
    }

    fn curve(freq: &[f64], spl: &[f64]) -> Curve {
        Curve {
            freq: Array1::from(freq.to_vec()),
            spl: Array1::from(spl.to_vec()),
            ..Default::default()
        }
    }

    fn curve_with_phase(freq: &[f64], spl: &[f64], phase_deg: &[f64]) -> Curve {
        Curve {
            freq: Array1::from(freq.to_vec()),
            spl: Array1::from(spl.to_vec()),
            phase: Some(Array1::from(phase_deg.to_vec())),
            ..Default::default()
        }
    }

    fn binding() -> PlaybackBinding {
        PlaybackBinding {
            graph_id: String::from("graph-final-001"),
            source_id: String::from("front-left"),
            seat_id: String::from("seat-1"),
            stimulus_hash: String::from("stimulus-hash-abc"),
            sample_rate_hz: 48_000.0,
            calibration_id: String::from("mic-cal-2026-09"),
            processing_state: String::from("final-delivered"),
        }
    }

    fn tolerances() -> CaptureTolerances {
        CaptureTolerances {
            max_magnitude_deviation_db: 1.0,
            max_timing_error_ms: 0.25,
            max_output_loss_db: 1.0,
        }
    }

    fn grid() -> Vec<f64> {
        vec![
            20.0, 50.0, 100.0, 200.0, 500.0, 1000.0, 2000.0, 5000.0, 10_000.0, 20_000.0,
        ]
    }

    fn flat(level: f64) -> Vec<f64> {
        vec![level; 10]
    }

    #[test]
    fn quality_capture_identity_mismatch_unassessed() {
        // F13: a valid capture under mismatched identities is unassessed.
        let frequencies = grid();
        let prediction = curve(&frequencies, &flat(80.0));
        let capture = curve(&frequencies, &flat(80.0));
        let declared = DeclaredAlignment {
            gain_db: 0.0,
            delay_ms: 0.0,
        };
        for mutate in [
            |binding: &mut PlaybackBinding| binding.graph_id = String::from("graph-other"),
            |binding: &mut PlaybackBinding| binding.stimulus_hash = String::from("other-hash"),
            |binding: &mut PlaybackBinding| binding.sample_rate_hz = 44_100.0,
            |binding: &mut PlaybackBinding| binding.calibration_id = String::from("other-cal"),
            |binding: &mut PlaybackBinding| {
                binding.processing_state = String::from("preview");
            },
            |binding: &mut PlaybackBinding| binding.seat_id = String::from("seat-2"),
        ] {
            let mut captured = binding();
            mutate(&mut captured);
            let report = compare_prediction_capture(
                &prediction,
                &capture,
                &binding(),
                &captured,
                PlaybackEvidenceKind::Acoustic,
                PlaybackLevel::SmallSignal,
                &declared,
                &tolerances(),
                [20.0, 20_000.0],
            )
            .unwrap();
            assert!(!report.assessed, "mismatch must not assess");
            assert!(!report.passed, "mismatch must not pass");
            assert!(!report.counts_as_acoustic_verification());
            assert!(report.unassessed_reason.is_some());
        }
        // Matched identities assess the same data.
        let report = compare_prediction_capture(
            &prediction,
            &capture,
            &binding(),
            &binding(),
            PlaybackEvidenceKind::Acoustic,
            PlaybackLevel::SmallSignal,
            &declared,
            &tolerances(),
            [20.0, 20_000.0],
        )
        .unwrap();
        assert!(report.assessed);
        assert!(report.passed);
    }

    #[test]
    fn quality_prediction_capture_declared_alignment_only() {
        // A +2 dB capture offset passes only with +2 dB declared: the
        // comparison applies declared values and never fits the offset away.
        let frequencies = grid();
        let prediction = curve(&frequencies, &flat(80.0));
        let capture = curve(&frequencies, &flat(82.0));
        let undeclared = DeclaredAlignment {
            gain_db: 0.0,
            delay_ms: 0.0,
        };
        let report = compare_prediction_capture(
            &prediction,
            &capture,
            &binding(),
            &binding(),
            PlaybackEvidenceKind::BackendRendered,
            PlaybackLevel::SmallSignal,
            &undeclared,
            &tolerances(),
            [20.0, 20_000.0],
        )
        .unwrap();
        assert!(report.assessed);
        assert!(!report.passed, "undeclared offset must fail, not be fitted");
        assert!(
            report
                .metric_outcomes
                .iter()
                .any(|outcome| outcome.metric == "magnitude_agreement" && !outcome.passed)
        );
        let declared = DeclaredAlignment {
            gain_db: 2.0,
            delay_ms: 0.0,
        };
        let report = compare_prediction_capture(
            &prediction,
            &capture,
            &binding(),
            &binding(),
            PlaybackEvidenceKind::BackendRendered,
            PlaybackLevel::SmallSignal,
            &declared,
            &tolerances(),
            [20.0, 20_000.0],
        )
        .unwrap();
        assert!(report.passed);
        assert!(
            report
                .ledger
                .iter()
                .any(|entry| entry.contains("+2.000 dB"))
        );
    }

    #[test]
    fn quality_capture_known_gain_delay_error_fails() {
        // Known +2 dB / +1 ms capture errors beyond declared zeros fail
        // both magnitude and timing; declaring the true values passes.
        // The timing grid is dense (100 Hz steps: 36 degrees per bin at
        // 1 ms) so the bulk-delay slope is unambiguous; a sparse grid
        // that aliases multi-cycle delay stays unassessed instead.
        let frequencies: Vec<f64> = (0..200).map(|index| 20.0 + index as f64 * 100.0).collect();
        let zeros = vec![0.0; frequencies.len()];
        let levels = vec![80.0; frequencies.len()];
        let hot_levels = vec![82.0; frequencies.len()];
        let prediction = curve_with_phase(&frequencies, &levels, &zeros);
        let delay_ms = 1.0;
        let phase: Vec<f64> = frequencies
            .iter()
            .map(|frequency| -360.0 * frequency * delay_ms / 1000.0)
            .collect();
        let capture = curve_with_phase(&frequencies, &hot_levels, &phase);
        let zero = DeclaredAlignment {
            gain_db: 0.0,
            delay_ms: 0.0,
        };
        let report = compare_prediction_capture(
            &prediction,
            &capture,
            &binding(),
            &binding(),
            PlaybackEvidenceKind::Acoustic,
            PlaybackLevel::SmallSignal,
            &zero,
            &tolerances(),
            [20.0, 20_000.0],
        )
        .unwrap();
        assert!(report.assessed);
        assert!(!report.passed);
        let timing = report
            .metric_outcomes
            .iter()
            .find(|outcome| outcome.metric == "timing_agreement")
            .unwrap();
        assert!(timing.assessed);
        assert!(!timing.passed);
        assert!((timing.observed.unwrap() - 1.0).abs() < 1e-6);
        let truthful = DeclaredAlignment {
            gain_db: 2.0,
            delay_ms: 1.0,
        };
        let report = compare_prediction_capture(
            &prediction,
            &capture,
            &binding(),
            &binding(),
            PlaybackEvidenceKind::Acoustic,
            PlaybackLevel::SmallSignal,
            &truthful,
            &tolerances(),
            [20.0, 20_000.0],
        )
        .unwrap();
        assert!(report.passed);

        // The same 1 ms delay on a sparse grid aliases across bins, so
        // timing stays unassessed instead of folding to a passing estimate.
        let sparse = grid();
        let sparse_prediction = curve_with_phase(&sparse, &flat(80.0), &vec![0.0; sparse.len()]);
        let sparse_phase: Vec<f64> = sparse
            .iter()
            .map(|frequency| -360.0 * frequency * delay_ms / 1000.0)
            .collect();
        let sparse_capture = curve_with_phase(&sparse, &flat(82.0), &sparse_phase);
        let report = compare_prediction_capture(
            &sparse_prediction,
            &sparse_capture,
            &binding(),
            &binding(),
            PlaybackEvidenceKind::Acoustic,
            PlaybackLevel::SmallSignal,
            &truthful,
            &tolerances(),
            [20.0, 20_000.0],
        )
        .unwrap();
        let timing = report
            .metric_outcomes
            .iter()
            .find(|outcome| outcome.metric == "timing_agreement")
            .unwrap();
        assert!(!timing.assessed, "aliased timing must not score");
    }

    #[test]
    fn quality_simulation_not_acoustic_verification() {
        // Identical data passes as a simulation but never verifies the room.
        let frequencies = grid();
        let prediction = curve_with_phase(&frequencies, &flat(80.0), &vec![0.0; frequencies.len()]);
        let capture = prediction.clone();
        let declared = DeclaredAlignment {
            gain_db: 0.0,
            delay_ms: 0.0,
        };
        let report = compare_prediction_capture(
            &prediction,
            &capture,
            &binding(),
            &binding(),
            PlaybackEvidenceKind::Simulated,
            PlaybackLevel::SmallSignal,
            &declared,
            &tolerances(),
            [20.0, 20_000.0],
        )
        .unwrap();
        assert!(report.passed, "simulation checks software behavior");
        assert_eq!(report.evidence_class, "simulated");
        assert!(
            !report.counts_as_acoustic_verification(),
            "simulation is not acoustic verification"
        );
        let acoustic = compare_prediction_capture(
            &prediction,
            &capture,
            &binding(),
            &binding(),
            PlaybackEvidenceKind::Acoustic,
            PlaybackLevel::SmallSignal,
            &declared,
            &tolerances(),
            [20.0, 20_000.0],
        )
        .unwrap();
        assert!(acoustic.counts_as_acoustic_verification());
    }

    #[test]
    fn quality_small_signal_not_maximum_output() {
        // A passing small-signal transfer check supports no max-output claim.
        let frequencies = grid();
        let prediction = curve_with_phase(&frequencies, &flat(80.0), &vec![0.0; frequencies.len()]);
        let capture = prediction.clone();
        let declared = DeclaredAlignment {
            gain_db: 0.0,
            delay_ms: 0.0,
        };
        let report = compare_prediction_capture(
            &prediction,
            &capture,
            &binding(),
            &binding(),
            PlaybackEvidenceKind::Acoustic,
            PlaybackLevel::SmallSignal,
            &declared,
            &tolerances(),
            [20.0, 20_000.0],
        )
        .unwrap();
        assert!(report.passed);
        assert!(!report.supports_maximum_output_claim());
        let driven = compare_prediction_capture(
            &prediction,
            &capture,
            &binding(),
            &binding(),
            PlaybackEvidenceKind::Acoustic,
            PlaybackLevel::MaximumOutput,
            &declared,
            &tolerances(),
            [20.0, 20_000.0],
        )
        .unwrap();
        assert!(driven.supports_maximum_output_claim());
    }
}
