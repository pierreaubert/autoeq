//! Pressure-calibrated Welch noise estimates from a separate silent-playback recording.
//!
//! Periodic Hann windows, half-frame hops, per-frame mean removal, and power
//! averaging are explicit. The last frame is anchored at the record end when
//! needed: no samples are discarded, but this is not uniform time weighting.
//! Calibration is supplied evidence, not authenticated hardware acquisition.

// Rust guideline compliant 2026-02-21
use super::{AmbientNoiseView, ViewProvenance};
use rustfft::{FftPlanner, num_complex::Complex};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Unsupported noise analysis with a diagnostic reason and optional development backtrace.
#[derive(Debug)]
pub struct NoiseAnalysisError {
    reason: String,
    backtrace: std::backtrace::Backtrace,
}

impl NoiseAnalysisError {
    /// Stable diagnostic text suitable for an unavailable report view.
    pub fn reason(&self) -> &str {
        &self.reason
    }
}

impl From<String> for NoiseAnalysisError {
    fn from(reason: String) -> Self {
        Self {
            reason,
            backtrace: std::backtrace::Backtrace::capture(),
        }
    }
}

impl From<&str> for NoiseAnalysisError {
    fn from(reason: &str) -> Self {
        reason.to_owned().into()
    }
}

impl std::fmt::Display for NoiseAnalysisError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{}", self.reason)?;
        if self.backtrace.status() == std::backtrace::BacktraceStatus::Captured {
            write!(formatter, "\n{}", self.backtrace)?;
        }
        Ok(())
    }
}

impl std::error::Error for NoiseAnalysisError {}

/// Numeric pressure and frequency-response calibration for one acquisition gain.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NoisePressureCalibration {
    /// Identity of this calibration resource.
    pub calibration_id: String,
    /// Pascals per decoded full-scale sample, before frequency-response correction.
    pub pascals_per_sample: f64,
    /// Individual microphone identity.
    pub microphone_id: String,
    /// Microphone orientation to which the response correction applies.
    pub orientation: String,
    /// Acquisition gain/path identity; must match the noise handoff.
    pub acquisition_gain_id: String,
    /// Reference source, level, gain, and conditions supporting the numeric scale.
    pub reference_conditions: String,
    /// Strictly increasing positive calibration frequencies; no extrapolation allowed.
    pub response_freqs_hz: Vec<f64>,
    /// Pressure-level corrections to add, interpolated linearly in log frequency.
    pub response_correction_db: Vec<f64>,
    /// Declared calibration uncertainty, not an estimated confidence interval.
    pub uncertainty_db: Option<f64>,
    /// Microphone/interface self-noise limits, explicitly unknown when uncharacterized.
    pub self_noise_note: String,
}

/// Explicit finite-record noise-estimator settings.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CapturedNoiseSettings {
    /// Power-of-two FFT length between 256 and 65536, without zero padding.
    pub frame_samples: usize,
    /// Operator-declared usable acquisition band, within calibration support.
    pub valid_band_hz: [f64; 2],
}

/// Calibrated diagnostic with explicit spectral bandwidth and observation support.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CapturedNoiseView {
    /// Canonical octave-summed SPL view; unavailable bands are omitted, not floored.
    pub spectrum: AmbientNoiseView,
    /// Versioned estimator convention.
    pub method: String,
    /// Exact estimator settings.
    pub settings: CapturedNoiseSettings,
    /// Numeric calibration and its limitations, retained for reproducibility.
    pub calibration: NoisePressureCalibration,
    /// Complete record length; frames cover every sample without truncation.
    pub record_samples: usize,
    /// Complete observation duration, not a stationarity claim.
    pub duration_seconds: f64,
    /// Actual frame starts, including a possible nonuniform final hop.
    pub frame_starts: Vec<usize>,
    /// FFT-bin spacing, not true resolving power.
    pub bin_spacing_hz: f64,
    /// Periodic Hann equivalent noise bandwidth, 1.5 bin spacings.
    pub window_enbw_hz: f64,
    /// Supported positive bin centers, including Nyquist only when declared usable.
    pub bin_freqs_hz: Vec<f64>,
    /// One-sided corrected pressure PSD in Pa²/Hz; zeros remain zero.
    pub pressure_psd_pa2_per_hz: Vec<f64>,
    /// Nominal octave boundaries for each retained canonical spectrum entry.
    pub octave_edges_hz: Vec<[f64; 2]>,
    /// First and last bin centers actually summed for each retained octave.
    pub octave_bin_centers_hz: Vec<[f64; 2]>,
    /// Unavailable nominal octave centers mapped to explicit reasons.
    pub unavailable_bands: BTreeMap<String, String>,
    /// Limits on inference, weighting, and calibration authority.
    pub scope: String,
}

fn declared(value: &str) -> bool {
    !matches!(
        value.trim().to_ascii_lowercase().as_str(),
        "" | "unknown" | "uncalibrated" | "unavailable" | "none"
    )
}

fn validate(
    cal: &NoisePressureCalibration,
    settings: &CapturedNoiseSettings,
    rate: f64,
) -> Result<(), NoiseAnalysisError> {
    let [low, high] = settings.valid_band_hz;
    if !rate.is_finite()
        || rate <= 0.0
        || !low.is_finite()
        || !high.is_finite()
        || low <= 0.0
        || high <= low
        || high > rate / 2.0
    {
        return Err("noise band must be positive, finite, and below or at Nyquist".into());
    }
    // Computational/report-size bounds, not acoustic-validity thresholds.
    if !settings.frame_samples.is_power_of_two()
        || !(256..=65_536).contains(&settings.frame_samples)
    {
        return Err("noise FFT frame must be a power of two between 256 and 65536".into());
    }
    if [
        &cal.calibration_id,
        &cal.microphone_id,
        &cal.orientation,
        &cal.acquisition_gain_id,
        &cal.reference_conditions,
    ]
    .iter()
    .any(|value| !declared(value))
        || cal.self_noise_note.trim().is_empty()
        || !cal.pascals_per_sample.is_finite()
        || cal.pascals_per_sample <= 0.0
        || !cal.pascals_per_sample.powi(2).is_finite()
        || cal.pascals_per_sample.powi(2) == 0.0
        || cal
            .uncertainty_db
            .is_some_and(|value| !value.is_finite() || value < 0.0)
    {
        return Err(
            "noise needs numeric pressure calibration and explicit acquisition/reference metadata"
                .into(),
        );
    }
    let freqs = &cal.response_freqs_hz;
    if freqs.len() < 2
        || freqs.len() != cal.response_correction_db.len()
        || freqs.iter().any(|f| !f.is_finite() || *f <= 0.0)
        || freqs.windows(2).any(|pair| pair[0] >= pair[1])
        || cal.response_correction_db.iter().any(|db| !db.is_finite())
        || freqs[0] > low
        || freqs[freqs.len() - 1] < high
    {
        return Err("noise frequency-response calibration must cover the declared band without extrapolation".into());
    }
    Ok(())
}

fn response_gain(cal: &NoisePressureCalibration, frequency: f64) -> f64 {
    let index = cal
        .response_freqs_hz
        .partition_point(|value| *value < frequency)
        .max(1)
        .min(cal.response_freqs_hz.len() - 1);
    let weight = (frequency.ln() - cal.response_freqs_hz[index - 1].ln())
        / (cal.response_freqs_hz[index].ln() - cal.response_freqs_hz[index - 1].ln());
    let db = cal.response_correction_db[index - 1] * (1.0 - weight)
        + cal.response_correction_db[index] * weight;
    10.0_f64.powf(db / 10.0)
}

fn welch(samples: &[f64], rate: f64, size: usize) -> (Vec<f64>, Vec<usize>) {
    let hop = size / 2;
    let mut starts: Vec<_> = (0..=samples.len() - size).step_by(hop).collect();
    if starts.last().copied() != Some(samples.len() - size) {
        starts.push(samples.len() - size);
    }
    let window: Vec<_> = (0..size)
        .map(|i| 0.5 - 0.5 * (std::f64::consts::TAU * i as f64 / size as f64).cos())
        .collect();
    let normalization = rate * window.iter().map(|w| w * w).sum::<f64>() * starts.len() as f64;
    let fft = FftPlanner::<f64>::new().plan_fft_forward(size);
    let mut buffer = vec![Complex::new(0.0, 0.0); size];
    let mut psd = vec![0.0; size / 2 + 1];
    for &start in &starts {
        let frame = &samples[start..start + size];
        let mean = frame.iter().sum::<f64>() / size as f64;
        for ((value, sample), window) in buffer.iter_mut().zip(frame).zip(&window) {
            *value = Complex::new((sample - mean) * window, 0.0);
        }
        fft.process(&mut buffer);
        for (bin, power) in psd.iter_mut().enumerate() {
            // DC and Nyquist have no negative-frequency partner.
            let sidedness = if bin == 0 || bin == size / 2 {
                1.0
            } else {
                2.0
            };
            *power += sidedness * buffer[bin].norm_sqr() / normalization;
        }
    }
    (psd, starts)
}

/// Estimate calibrated pressure PSD and bin-summed octave levels from recorded noise.
///
/// Samples are decoded full-scale mono values, not IR samples or dBFS spectra.
/// The caller must bind raw capture/calibration bytes and establish silent-playback
/// conditions separately. SPL uses RMS pressure relative to 20 µPa. Octaves are
/// FFT-bin sums, not certified octave filters, NC/NR ratings, or loudness estimates.
///
/// # Errors
/// Rejects malformed calibration, unsupported bands/frames, insufficient samples,
/// nonfinite or full-scale samples, identity mismatch, or numeric overflow.
pub fn calibrated_capture_noise(
    samples: impl AsRef<[f64]>,
    settings: CapturedNoiseSettings,
    calibration: NoisePressureCalibration,
    provenance: ViewProvenance,
    settings_hash: String,
) -> Result<CapturedNoiseView, NoiseAnalysisError> {
    let samples = samples.as_ref();
    let rate = provenance.sample_rate_hz;
    validate(&calibration, &settings, rate)?;
    if samples.len() < settings.frame_samples
        || samples
            .iter()
            .any(|value| !value.is_finite() || value.abs() >= 1.0)
    {
        return Err(
            "noise needs at least one complete frame of finite, below-full-scale samples".into(),
        );
    }
    if provenance.calibration != calibration.calibration_id
        || !declared(&settings_hash)
        || !declared(&provenance.graph_identity)
        || provenance.measurement_ids.is_empty()
    {
        return Err("noise calibration, settings, or capture provenance mismatch".into());
    }
    let (raw_psd, frame_starts) = welch(samples, rate, settings.frame_samples);
    let spacing = rate / settings.frame_samples as f64;
    let [low, high] = settings.valid_band_hz;
    let mut bin_freqs_hz = Vec::new();
    let mut pressure_psd_pa2_per_hz = Vec::new();
    for (index, value) in raw_psd.into_iter().enumerate().skip(1) {
        let frequency = index as f64 * spacing;
        if frequency < low || frequency > high {
            continue;
        }
        let gain = calibration.pascals_per_sample.powi(2) * response_gain(&calibration, frequency);
        let power = value * gain;
        if !gain.is_finite() || gain <= 0.0 || !power.is_finite() || (value > 0.0 && power == 0.0) {
            return Err("noise pressure calibration or power overflow/underflow".into());
        }
        bin_freqs_hz.push(frequency);
        pressure_psd_pa2_per_hz.push(power);
    }
    if bin_freqs_hz.is_empty() {
        return Err("declared noise band contains no positive FFT bin".into());
    }
    let mut spectrum = AmbientNoiseView {
        provenance,
        settings_hash,
        freqs: Vec::new(),
        noise_spl_db: Vec::new(),
        calibration: calibration.calibration_id.clone(),
    };
    let mut octave_edges_hz = Vec::new();
    let mut octave_bin_centers_hz = Vec::new();
    let mut unavailable_bands = BTreeMap::new();
    for center in [
        31.5, 63.0, 125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0, 16000.0,
    ] {
        let edges = [center / 2.0_f64.sqrt(), center * 2.0_f64.sqrt()];
        let indices: Vec<_> = bin_freqs_hz
            .iter()
            .enumerate()
            .filter_map(|(i, f)| (*f >= edges[0] && *f < edges[1]).then_some(i))
            .collect();
        // Three bin centers are a display-support rule, not proof of adequate
        // resolving power or conformance with an octave-filter standard.
        let reason = if edges[0] < low || edges[1] > high {
            Some("octave extends outside declared acquisition/calibration support")
        } else if indices.len() < 3 {
            Some("octave has fewer than three FFT bin centers")
        } else {
            None
        };
        if let Some(reason) = reason {
            unavailable_bands.insert(center.to_string(), reason.into());
            continue;
        }
        let power = indices
            .iter()
            .map(|&i| pressure_psd_pa2_per_hz[i] * spacing)
            .sum::<f64>();
        if !power.is_finite() {
            return Err("noise octave power overflow".into());
        }
        if power <= 0.0 {
            unavailable_bands.insert(
                center.to_string(),
                "zero estimated power; no finite SPL, no noise floor inferred".into(),
            );
            continue;
        }
        spectrum.freqs.push(center);
        // Log subtraction avoids overflowing the ratio for large finite powers.
        spectrum
            .noise_spl_db
            .push(10.0 * (power.log10() - (20e-6_f64).powi(2).log10()));
        octave_edges_hz.push(edges);
        octave_bin_centers_hz.push([
            bin_freqs_hz[indices[0]],
            bin_freqs_hz[*indices.last().unwrap()],
        ]);
    }
    Ok(CapturedNoiseView {
        spectrum, method: "periodic_hann_welch_pressure_v1".into(), settings, calibration,
        record_samples: samples.len(), duration_seconds: samples.len() as f64 / rate,
        frame_starts, bin_spacing_hz: spacing, window_enbw_hz: 1.5 * spacing,
        bin_freqs_hz, pressure_psd_pa2_per_hz, octave_edges_hz, octave_bin_centers_hz,
        unavailable_bands,
        scope: "separate operator-declared silent-playback recording; periodic Hann, 50% nominal overlap, end-anchored final frame, per-frame mean removal, unnormalized forward FFT, one-sided power averaging; no zero padding or tail truncation; bin-summed nominal octaves, no certified octave-filter response; RMS pressure reference 20 µPa; no A/C weighting, loudness, NC/NR, stationarity, acoustic acceptance, or authenticated calibration claim; calibration correction applied once in log frequency; no self-noise subtraction; window leakage limits band isolation; record duration does not prove representative ambient conditions".into(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn calibration() -> NoisePressureCalibration {
        NoisePressureCalibration {
            calibration_id: "numeric-test".into(),
            pascals_per_sample: 2.0,
            microphone_id: "synthetic".into(),
            orientation: "omnidirectional model".into(),
            acquisition_gain_id: "fixed-test-gain".into(),
            reference_conditions: "numeric oracle, not a real calibrator".into(),
            response_freqs_hz: vec![1.0, 24_000.0],
            response_correction_db: vec![0.0, 0.0],
            uncertainty_db: None,
            self_noise_note: "not characterized in this synthetic test".into(),
        }
    }

    fn analyze(
        samples: &[f64],
        cal: NoisePressureCalibration,
    ) -> Result<CapturedNoiseView, NoiseAnalysisError> {
        let provenance = ViewProvenance {
            measurement_ids: vec!["synthetic-noise".into()],
            graph_identity: "graph".into(),
            sample_rate_hz: 48_000.0,
            calibration: cal.calibration_id.clone(),
            processing_chain: "test".into(),
        };
        calibrated_capture_noise(
            samples,
            CapturedNoiseSettings {
                frame_samples: 4096,
                valid_band_hz: [1.0, 24_000.0],
            },
            cal,
            provenance,
            "settings".into(),
        )
    }

    #[test]
    fn calibrated_noise_known_pressure_and_response_correction() {
        let samples: Vec<_> = (0..8192)
            .map(|i| 0.01 * (std::f64::consts::TAU * 80.0 * i as f64 / 4096.0).sin())
            .collect();
        let view = analyze(&samples, calibration()).unwrap();
        assert_eq!(view.frame_starts, [0, 2048, 4096]);
        let index = view
            .spectrum
            .freqs
            .iter()
            .position(|f| *f == 1000.0)
            .unwrap();
        let expected = 20.0 * (0.02_f64 / 2.0_f64.sqrt() / 20e-6).log10();
        assert!((view.spectrum.noise_spl_db[index] - expected).abs() < 1e-10);
        let mut corrected = calibration();
        corrected.response_correction_db = vec![6.0, 6.0];
        let corrected = analyze(&samples, corrected).unwrap();
        assert!((corrected.spectrum.noise_spl_db[index] - expected - 6.0).abs() < 1e-10);
        let doubled = analyze(
            &samples.iter().map(|v| v * 2.0).collect::<Vec<_>>(),
            calibration(),
        )
        .unwrap();
        assert!(
            (doubled.spectrum.noise_spl_db[index] - expected - 20.0 * 2.0_f64.log10()).abs()
                < 1e-10
        );
        let mut slope = calibration();
        slope.response_freqs_hz = vec![100.0, 10_000.0];
        slope.response_correction_db = vec![-6.0, 6.0];
        assert!((response_gain(&slope, 1000.0) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn calibrated_noise_parseval_nyquist_and_final_frame() {
        let samples: Vec<_> = (0..5000)
            .map(|i| 0.1 + 0.02 * ((i * 73 % 257) as f64 / 257.0 - 0.5))
            .collect();
        let (psd, starts) = welch(&samples, 48_000.0, 4096);
        assert_eq!(starts, [0, 904]);
        let mut expected = 0.0;
        for start in &starts {
            let frame = &samples[*start..*start + 4096];
            let mean = frame.iter().sum::<f64>() / 4096.0;
            let mut power = 0.0;
            let mut weight = 0.0;
            for (i, sample) in frame.iter().enumerate() {
                let w = 0.5 - 0.5 * (std::f64::consts::TAU * i as f64 / 4096.0).cos();
                power += ((sample - mean) * w).powi(2);
                weight += w * w;
            }
            expected += power / weight / starts.len() as f64;
        }
        assert!((psd.iter().sum::<f64>() * 48_000.0 / 4096.0 - expected).abs() < 1e-16);
        let alternating: Vec<_> = (0..4096)
            .map(|i| if i % 2 == 0 { 0.01 } else { -0.01 })
            .collect();
        let view = analyze(&alternating, calibration()).unwrap();
        let total = view.pressure_psd_pa2_per_hz.iter().sum::<f64>() * view.bin_spacing_hz;
        assert!(
            (total - 0.02_f64.powi(2)).abs() < 1e-15,
            "Nyquist must not be doubled"
        );
    }

    #[test]
    fn calibrated_noise_normalization_is_sample_rate_independent() {
        let samples: Vec<_> = (0..4096)
            .map(|i| 0.01 * (std::f64::consts::TAU * 80.0 * i as f64 / 4096.0).sin())
            .collect();
        for rate in [44_100.0, 48_000.0, 96_000.0] {
            let mut cal = calibration();
            cal.response_freqs_hz = vec![1.0, rate / 2.0];
            let view = calibrated_capture_noise(
                &samples,
                CapturedNoiseSettings {
                    frame_samples: 4096,
                    valid_band_hz: [1.0, rate / 2.0],
                },
                cal,
                ViewProvenance {
                    measurement_ids: vec!["synthetic-noise".into()],
                    graph_identity: "graph".into(),
                    sample_rate_hz: rate,
                    calibration: "numeric-test".into(),
                    processing_chain: "test".into(),
                },
                "settings".into(),
            )
            .unwrap();
            let power = view.pressure_psd_pa2_per_hz.iter().sum::<f64>() * view.bin_spacing_hz;
            assert!(
                (power - 0.02_f64.powi(2) / 2.0).abs() < 1e-15,
                "rate {rate}"
            );
            assert_eq!(view.duration_seconds, 4096.0 / rate);
        }
    }

    #[test]
    fn calibrated_noise_zero_power_is_not_a_fabricated_floor() {
        let view = analyze(&vec![0.0; 4096], calibration()).unwrap();
        assert!(view.spectrum.freqs.is_empty());
        assert!(view.pressure_psd_pa2_per_hz.iter().all(|p| *p == 0.0));
        assert!(view.unavailable_bands["1000"].contains("zero estimated power"));
    }

    #[test]
    fn calibrated_noise_refuses_unsupported_or_invalid_inputs() {
        for samples in [
            vec![0.0; 4095],
            vec![f64::NAN; 4096],
            vec![1.0; 4096],
            vec![-1.0; 4096],
        ] {
            assert!(analyze(&samples, calibration()).is_err());
        }
        for case in 0..7 {
            let mut cal = calibration();
            match case {
                0 => cal.pascals_per_sample = 0.0,
                1 => cal.pascals_per_sample = f64::MAX,
                2 => cal.response_freqs_hz = vec![10.0, 1000.0],
                3 => cal.response_freqs_hz = vec![24_000.0, 1.0],
                4 => cal.uncertainty_db = Some(-1.0),
                5 => cal.calibration_id = " Uncalibrated ".into(),
                _ => cal.response_correction_db = vec![4000.0, 4000.0],
            }
            assert!(analyze(&vec![0.01; 4096], cal).is_err(), "case {case}");
        }
        let cal = calibration();
        for settings in [
            CapturedNoiseSettings {
                frame_samples: 255,
                valid_band_hz: [1.0, 2000.0],
            },
            CapturedNoiseSettings {
                frame_samples: 4096,
                valid_band_hz: [0.0, 2000.0],
            },
            CapturedNoiseSettings {
                frame_samples: 4096,
                valid_band_hz: [1.0, 24_001.0],
            },
        ] {
            assert!(validate(&cal, &settings, 48_000.0).is_err());
        }
    }
}
