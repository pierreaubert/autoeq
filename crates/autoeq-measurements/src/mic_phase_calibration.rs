//! Microphone phase calibration loader — GD-Opt v2 Phase GD-1f.
//!
//! Below ~50 Hz a USB measurement mic's own phase can drift ±30°, so
//! any bass-phase extraction that doesn't subtract the mic's response
//! attributes mic artefacts to the room. See
//! `docs/gd_opt_v2_plan.md` §2.6 and §2.8 (`"mic_phase_uncalibrated"`
//! advisory).
//!
//! This module provides:
//! 1. [`MicPhaseCalibration`] — the loaded 4-column calibration
//!    `(freq, mag_db, phase_deg, coherence)`.
//! 2. [`load_mic_phase_calibration`] — CSV loader with the same
//!    header-driven column discovery as
//!    [`crate::read::load_driver_measurement`], so callers can drop a
//!    calibration file authored by any tool that produces those four
//!    named columns.
//! 3. [`MicPhaseCalibration::apply_to_curve`] — in-place correction
//!    that subtracts the mic's magnitude and phase from a measured
//!    `Curve` and attenuates the curve's own coherence by the mic
//!    coherence.
//!
//! The struct, loader, and application are all read-only with
//! respect to the filesystem — nothing is mutated outside of the
//! `&mut Curve` the caller passes in.

use crate::Curve;
use ndarray::Array1;
use std::error::Error;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

/// Microphone calibration carrying both magnitude **and** phase
/// deviations at each frequency, plus a per-bin coherence that
/// bounds how much we trust each correction.
///
/// A "flat" mic has `mag_db[k] == 0.0` and `phase_deg[k] == 0.0`
/// across all bins; any deviation reports the mic's bias relative
/// to an ideal reference.
///
/// Frequencies must be strictly increasing so that
/// [`MicPhaseCalibration::apply_to_curve`] can do monotonic
/// interpolation. The loader enforces this.
#[derive(Debug, Clone, PartialEq)]
pub struct MicPhaseCalibration {
    /// Frequency points in Hz, strictly increasing.
    pub freq: Array1<f64>,
    /// Magnitude deviation in dB (positive = mic is louder than flat).
    pub mag_db: Array1<f64>,
    /// Phase deviation in degrees (positive = mic leads the reference).
    pub phase_deg: Array1<f64>,
    /// Per-bin coherence γ² from the calibration capture, in `[0, 1]`.
    /// Callers use this to down-weight corrections where the cal
    /// itself was noisy.
    pub coherence: Array1<f64>,
}

impl MicPhaseCalibration {
    /// Build an identity calibration (no deviation) on a given grid.
    /// Only useful for tests and as a no-op when a calibration file is
    /// missing — the caller still has to **choose** to apply this
    /// rather than treat the mic as uncalibrated.
    pub fn identity(freq: Array1<f64>) -> Self {
        let n = freq.len();
        Self {
            freq,
            mag_db: Array1::zeros(n),
            phase_deg: Array1::zeros(n),
            coherence: Array1::ones(n),
        }
    }

    /// Interpolate within measured calibration support, or return None for
    /// malformed calibration, nonfinite queries or out-of-band frequencies.
    /// Uses linear-frequency interpolation, not acoustic extrapolation.
    pub fn sample_at(&self, freq_hz: f64) -> Option<(f64, f64, f64)> {
        self.validate().ok()?;
        self.sample_validated(freq_hz)
    }

    fn validate(&self) -> Result<(), String> {
        let n = self.freq.len();
        if n == 0
            || self.mag_db.len() != n
            || self.phase_deg.len() != n
            || self.coherence.len() != n
        {
            return Err("microphone calibration has empty or mismatched arrays".into());
        }
        if self.freq.iter().any(|f| !f.is_finite() || *f <= 0.0)
            || self
                .freq
                .iter()
                .zip(self.freq.iter().skip(1))
                .any(|(a, b)| b <= a)
            || self
                .mag_db
                .iter()
                .chain(self.phase_deg.iter())
                .any(|v| !v.is_finite())
            || self
                .coherence
                .iter()
                .any(|v| !v.is_finite() || !(0.0..=1.0).contains(v))
        {
            return Err("microphone calibration has invalid frequency, magnitude, phase or coherence evidence".into());
        }
        Ok(())
    }

    fn sample_validated(&self, freq_hz: f64) -> Option<(f64, f64, f64)> {
        let n = self.freq.len();
        if n == 0 {
            return None;
        }
        if !freq_hz.is_finite() {
            return None;
        }
        let tolerance = 8.0 * f64::EPSILON * freq_hz.abs().max(self.freq[n - 1]);
        if freq_hz <= 0.0
            || freq_hz < self.freq[0] - tolerance
            || freq_hz > self.freq[n - 1] + tolerance
        {
            return None;
        }
        if freq_hz <= self.freq[0] {
            return Some((self.mag_db[0], self.phase_deg[0], self.coherence[0]));
        }
        if freq_hz >= self.freq[n - 1] {
            return Some((
                self.mag_db[n - 1],
                self.phase_deg[n - 1],
                self.coherence[n - 1],
            ));
        }
        // Binary search for the bracket without requiring contiguous storage.
        let mut lo = 0;
        let mut hi = n;
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            if self.freq[mid] < freq_hz {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        let idx = lo;
        let x0 = self.freq[idx - 1];
        let x1 = self.freq[idx];
        let dx = x1 - x0;
        if dx.abs() < f64::EPSILON {
            return Some((self.mag_db[idx], self.phase_deg[idx], self.coherence[idx]));
        }
        let t = (freq_hz - x0) / dx;
        Some((
            self.mag_db[idx - 1] * (1.0 - t) + self.mag_db[idx] * t,
            self.phase_deg[idx - 1] * (1.0 - t) + self.phase_deg[idx] * t,
            self.coherence[idx - 1] * (1.0 - t) + self.coherence[idx] * t,
        ))
    }

    /// Atomically subtract microphone magnitude/phase and multiply coherence.
    /// Malformed evidence, unsupported frequencies, and nonfinite results are
    /// errors. No fields are changed on error. Cached phase decomposition is
    /// invalidated on success; missing measured phase/coherence stay absent.
    pub fn apply_to_curve(&self, curve: &mut Curve) -> Result<(), String> {
        self.validate()?;
        curve
            .validate("microphone calibration input")
            .map_err(|e| e.to_string())?;
        if curve
            .coherence
            .as_ref()
            .is_some_and(|values| values.iter().any(|v| !(0.0..=1.0).contains(v)))
        {
            return Err("measurement coherence must be in [0, 1]".into());
        }
        let mut corrected = curve.clone();
        for i in 0..curve.freq.len() {
            let (mag, phase, coh) = self.sample_validated(curve.freq[i]).ok_or_else(|| {
                format!("microphone calibration does not cover {} Hz", curve.freq[i])
            })?;
            corrected.spl[i] -= mag;
            if let Some(ref mut values) = corrected.phase {
                values[i] -= phase;
            }
            if let Some(ref mut values) = corrected.coherence {
                values[i] *= coh;
            }
        }
        corrected.min_phase = None;
        corrected.excess_phase = None;
        corrected.excess_delay_ms = None;
        corrected
            .validate("microphone calibration output")
            .map_err(|e| e.to_string())?;
        *curve = corrected;
        Ok(())
    }
}

/// Load a 4-column microphone phase calibration CSV.
///
/// # CSV format
/// Header row required. Column discovery is header-name driven
/// (case-insensitive) so authors can reorder columns or name them in
/// any of the canonical SOTF spellings:
///
/// | Column | Recognised header names |
/// |---|---|
/// | freq | `frequency_hz`, `frequency`, `freq`, `hz` |
/// | mag_db | `mag_db`, `magnitude_db`, `magnitude`, `spl`, `spl_db`, `db` |
/// | phase_deg | `phase_deg`, `phase` |
/// | coherence | `coherence` |
///
/// Missing columns trigger an error — this is a strict 4-column
/// loader. For 2-column magnitude-only calibrations use the
/// pre-existing [`math_audio_dsp::analysis::MicrophoneCompensation`].
///
/// Frequencies must be positive and strictly increasing; coherence must be
/// in [0, 1]. Malformed, short, or non-finite data rows reject the file with
/// a row number. Dropping them would manufacture calibration by interpolating
/// across missing evidence. Blank lines and comments remain permitted.
pub fn load_mic_phase_calibration(path: &Path) -> Result<MicPhaseCalibration, Box<dyn Error>> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);

    let mut freq_col: Option<usize> = None;
    let mut mag_col: Option<usize> = None;
    let mut phase_col: Option<usize> = None;
    let mut coh_col: Option<usize> = None;
    let mut header_parsed = false;

    let mut freqs = Vec::new();
    let mut mags = Vec::new();
    let mut phases = Vec::new();
    let mut cohs = Vec::new();

    for (line_num, line) in reader.lines().enumerate() {
        let line = line?;
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') || trimmed.starts_with("//") {
            continue;
        }

        let parts: Vec<&str> = if trimmed.contains(',') {
            trimmed.split(',').map(|s| s.trim()).collect()
        } else {
            trimmed.split_whitespace().collect()
        };

        if !header_parsed {
            // A valid 4-column calibration MUST start with a named
            // header — without it we can't tell `phase_deg` from
            // `coherence`.
            for (idx, col_name) in parts.iter().enumerate() {
                let lower = col_name.to_lowercase();
                if coh_col.is_none() && lower == "coherence" {
                    coh_col = Some(idx);
                } else if phase_col.is_none() && (lower.contains("phase") || lower == "phase_deg") {
                    phase_col = Some(idx);
                } else if freq_col.is_none()
                    && (lower.contains("freq") || lower == "hz" || lower == "frequency_hz")
                {
                    freq_col = Some(idx);
                } else if mag_col.is_none()
                    && (lower.contains("mag_db")
                        || lower.contains("magnitude")
                        || lower.contains("spl")
                        || lower == "db"
                        || lower == "spl_db")
                {
                    mag_col = Some(idx);
                }
            }
            if freq_col.is_none() || mag_col.is_none() || phase_col.is_none() || coh_col.is_none() {
                return Err(format!(
                    "mic phase calibration at {path:?} must have named columns for \
                     frequency / magnitude_db / phase_deg / coherence; got header {parts:?}"
                )
                .into());
            }
            header_parsed = true;
            continue;
        }

        let (Some(freq_idx), Some(mag_idx), Some(phase_idx), Some(coh_idx)) =
            (freq_col, mag_col, phase_col, coh_col)
        else {
            return Err("mic phase calibration header was not parsed".into());
        };
        let max_idx = freq_idx.max(mag_idx).max(phase_idx).max(coh_idx);
        let invalid_row = || {
            format!(
                "invalid mic phase calibration row {} in {path:?}",
                line_num + 1
            )
        };
        if parts.len() <= max_idx {
            return Err(invalid_row().into());
        }

        let (Ok(f), Ok(m), Ok(p), Ok(c)) = (
            parts[freq_idx].parse::<f64>(),
            parts[mag_idx].parse::<f64>(),
            parts[phase_idx].parse::<f64>(),
            parts[coh_idx].parse::<f64>(),
        ) else {
            return Err(invalid_row().into());
        };
        if !(f.is_finite() && m.is_finite() && p.is_finite() && c.is_finite()) {
            return Err(invalid_row().into());
        }
        if f <= 0.0 || !(0.0..=1.0).contains(&c) {
            return Err(invalid_row().into());
        }
        freqs.push(f);
        mags.push(m);
        phases.push(p);
        cohs.push(c);
    }

    if freqs.is_empty() {
        return Err(
            format!("mic phase calibration at {path:?} contained no valid data rows").into(),
        );
    }

    for pair in freqs.windows(2) {
        if pair[0] >= pair[1] {
            return Err(format!(
                "mic phase calibration at {path:?} frequencies are not strictly increasing \
                 (found {} before {})",
                pair[0], pair[1]
            )
            .into());
        }
    }

    Ok(MicPhaseCalibration {
        freq: Array1::from_vec(freqs),
        mag_db: Array1::from_vec(mags),
        phase_deg: Array1::from_vec(phases),
        coherence: Array1::from_vec(cohs),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    fn write_cal_csv(csv: &str) -> NamedTempFile {
        let mut f = NamedTempFile::new().unwrap();
        f.write_all(csv.as_bytes()).unwrap();
        f.flush().unwrap();
        f
    }

    #[test]
    fn loads_canonical_four_column_csv() {
        let csv = "\
frequency_hz,mag_db,phase_deg,coherence
20.0,2.0,-10.0,0.95
50.0,1.0,-5.0,0.98
200.0,0.0,0.0,0.99
2000.0,-0.5,2.0,0.99
20000.0,-3.0,15.0,0.90
";
        let f = write_cal_csv(csv);
        let cal = load_mic_phase_calibration(f.path()).unwrap();
        assert_eq!(cal.freq.len(), 5);
        assert!((cal.freq[0] - 20.0).abs() < 1e-9);
        assert!((cal.mag_db[0] - 2.0).abs() < 1e-9);
        assert!((cal.phase_deg[1] + 5.0).abs() < 1e-9);
        assert!((cal.coherence[4] - 0.90).abs() < 1e-9);
    }

    #[test]
    fn column_order_is_header_driven() {
        // Shuffled columns: the loader must key off names, not position.
        let csv = "\
phase_deg,coherence,frequency_hz,mag_db
-10.0,0.95,20.0,2.0
-5.0,0.98,50.0,1.0
";
        let f = write_cal_csv(csv);
        let cal = load_mic_phase_calibration(f.path()).unwrap();
        assert!((cal.freq[0] - 20.0).abs() < 1e-9);
        assert!((cal.phase_deg[0] + 10.0).abs() < 1e-9);
        assert!((cal.mag_db[0] - 2.0).abs() < 1e-9);
    }

    #[test]
    fn missing_column_rejects_file() {
        // No coherence column → the 4-col contract is violated.
        let csv = "\
frequency_hz,mag_db,phase_deg
20.0,2.0,-10.0
";
        let f = write_cal_csv(csv);
        assert!(load_mic_phase_calibration(f.path()).is_err());
    }

    #[test]
    fn non_monotonic_frequencies_reject_file() {
        let csv = "\
frequency_hz,mag_db,phase_deg,coherence
200.0,0.0,0.0,0.99
20.0,2.0,-10.0,0.95
";
        let f = write_cal_csv(csv);
        let err = load_mic_phase_calibration(f.path())
            .unwrap_err()
            .to_string();
        assert!(err.contains("not strictly increasing"), "got: {err}");
    }

    #[test]
    fn invalid_calibration_values_cannot_be_interpolated_across() {
        for row in [
            "50,NaN,0,1",
            "50,0,NaN,1",
            "50,0,0,NaN",
            "50,0,0,-0.1",
            "50,0,0,1.1",
            "0,0,0,1",
            "-50,0,0,1",
        ] {
            let csv =
                format!("frequency_hz,mag_db,phase_deg,coherence\n20,0,0,1\n{row}\n200,0,0,1\n");
            let file = write_cal_csv(&csv);
            let error = load_mic_phase_calibration(file.path())
                .unwrap_err()
                .to_string();
            assert!(error.contains("row 3"), "{row}: {error}");
        }
    }

    #[test]
    fn malformed_row_rejects_file() {
        let csv = "\
frequency_hz,mag_db,phase_deg,coherence
20.0,2.0,-10.0,0.95
50.0,not-a-number,-5.0,0.98
200.0,0.0,0.0,0.99
";
        let f = write_cal_csv(csv);
        let error = load_mic_phase_calibration(f.path())
            .unwrap_err()
            .to_string();
        assert!(error.contains("row 3"), "{error}");
    }

    #[test]
    fn sample_at_exact_node_matches_stored_value() {
        let cal = MicPhaseCalibration {
            freq: Array1::from_vec(vec![20.0, 200.0, 2000.0]),
            mag_db: Array1::from_vec(vec![2.0, 0.0, -3.0]),
            phase_deg: Array1::from_vec(vec![-10.0, 0.0, 5.0]),
            coherence: Array1::from_vec(vec![0.95, 0.99, 0.90]),
        };
        let (m, p, c) = cal.sample_at(200.0).unwrap();
        assert!((m - 0.0).abs() < 1e-9);
        assert!((p - 0.0).abs() < 1e-9);
        assert!((c - 0.99).abs() < 1e-9);
    }

    #[test]
    fn sample_at_midpoint_interpolates() {
        let cal = MicPhaseCalibration {
            freq: Array1::from_vec(vec![100.0, 200.0]),
            mag_db: Array1::from_vec(vec![0.0, 4.0]),
            phase_deg: Array1::from_vec(vec![0.0, 20.0]),
            coherence: Array1::from_vec(vec![1.0, 0.5]),
        };
        // At 150 Hz we're halfway between 100 and 200 → all three
        // values should interpolate linearly.
        let (m, p, c) = cal.sample_at(150.0).unwrap();
        assert!(
            (m - 2.0).abs() < 1e-9,
            "mag @ 150 Hz should be 2 dB, got {m}"
        );
        assert!(
            (p - 10.0).abs() < 1e-9,
            "phase @ 150 Hz should be 10°, got {p}"
        );
        assert!(
            (c - 0.75).abs() < 1e-9,
            "coherence @ 150 Hz should be 0.75, got {c}"
        );
    }

    #[test]
    fn sample_at_interpolates_non_contiguous_calibration_arrays() {
        let strided = |values: Vec<f64>| {
            values
                .into_iter()
                .flat_map(|value| [value, -1.0])
                .collect::<Array1<f64>>()
                .slice_axis_move(ndarray::Axis(0), ndarray::Slice::new(0, None, 2))
        };
        let cal = MicPhaseCalibration {
            freq: strided(vec![100.0, 200.0]),
            mag_db: strided(vec![0.0, 4.0]),
            phase_deg: strided(vec![0.0, 20.0]),
            coherence: strided(vec![1.0, 0.5]),
        };
        assert!(cal.freq.as_slice().is_none());

        let (magnitude, phase, coherence) = cal.sample_at(150.0).unwrap();

        assert!((magnitude - 2.0).abs() < 1e-9);
        assert!((phase - 10.0).abs() < 1e-9);
        assert!((coherence - 0.75).abs() < 1e-9);
    }

    #[test]
    fn sample_at_below_min_returns_none() {
        let cal = MicPhaseCalibration::identity(vec![50.0, 500.0].into());
        assert!(cal.sample_at(10.0).is_none());
    }

    #[test]
    fn sample_at_above_max_returns_none() {
        let cal = MicPhaseCalibration::identity(vec![50.0, 500.0].into());
        assert!(cal.sample_at(50_000.0).is_none());
    }

    #[test]
    fn identity_is_transparent() {
        let freq = Array1::from_vec(vec![20.0, 200.0, 2000.0]);
        let cal = MicPhaseCalibration::identity(freq.clone());
        let mut curve = Curve {
            freq: freq.clone(),
            spl: Array1::from_vec(vec![60.0, 80.0, 90.0]),
            phase: Some(Array1::from_vec(vec![-30.0, 0.0, 45.0])),
            coherence: Some(Array1::from_vec(vec![0.98, 0.99, 0.97])),
            ..Default::default()
        };
        let before = curve.clone();
        cal.apply_to_curve(&mut curve).unwrap();
        assert_eq!(before.spl, curve.spl);
        assert_eq!(before.phase, curve.phase);
        assert_eq!(before.coherence, curve.coherence);
    }

    #[test]
    fn unsupported_calibration_band_does_not_partially_modify_curve() {
        let cal = MicPhaseCalibration {
            freq: vec![20.0, 200.0].into(),
            mag_db: vec![3.0, 4.0].into(),
            phase_deg: vec![10.0, 20.0].into(),
            coherence: vec![0.8, 0.9].into(),
        };
        let mut curve = Curve {
            freq: vec![20.0, 100.0, 500.0].into(),
            spl: vec![80.0; 3].into(),
            phase: Some(vec![0.0; 3].into()),
            coherence: Some(vec![1.0; 3].into()),
            ..Default::default()
        };
        let before = curve.clone();
        assert!(
            cal.apply_to_curve(&mut curve)
                .unwrap_err()
                .contains("does not cover")
        );
        assert_eq!(
            curve.spl, before.spl,
            "unsupported calibration must be atomic"
        );
        assert_eq!(curve.phase, before.phase);
        assert_eq!(curve.coherence, before.coherence);
    }

    #[test]
    fn malformed_calibration_and_overflow_reject_without_mutation() {
        let good = MicPhaseCalibration::identity(vec![20.0, 200.0].into());
        let curve = Curve {
            freq: good.freq.clone(),
            spl: vec![80.0; 2].into(),
            phase: Some(vec![0.0; 2].into()),
            ..Default::default()
        };
        let mut cases = vec![good.clone(); 4];
        cases[0].mag_db = vec![0.0].into();
        cases[1].freq[1] = 20.0;
        cases[2].coherence[1] = 1.1;
        cases[3].phase_deg[1] = f64::NAN;
        for cal in cases {
            assert!(cal.sample_at(20.0).is_none());
            let mut input = curve.clone();
            assert!(cal.apply_to_curve(&mut input).is_err());
            assert_eq!(input.spl, curve.spl);
            assert_eq!(input.phase, curve.phase);
        }
        let mut cal = good;
        cal.mag_db.fill(-f64::MAX);
        let mut input = curve;
        input.spl.fill(f64::MAX);
        assert!(cal.apply_to_curve(&mut input).is_err());
        assert!(input.spl.iter().all(|value| *value == f64::MAX));
    }

    #[test]
    fn calibrated_curve_invalidates_derived_phase_cache() {
        let mut cal = MicPhaseCalibration::identity(vec![20.0, 200.0].into());
        cal.mag_db.fill(2.0);
        cal.phase_deg.fill(10.0);
        let mut curve = Curve {
            freq: cal.freq.clone(),
            spl: vec![80.0; 2].into(),
            phase: Some(vec![0.0; 2].into()),
            min_phase: Some(vec![1.0; 2].into()),
            excess_phase: Some(vec![2.0; 2].into()),
            excess_delay_ms: Some(3.0),
            ..Default::default()
        };
        cal.apply_to_curve(&mut curve).unwrap();
        assert_eq!(curve.spl.to_vec(), vec![78.0; 2]);
        assert_eq!(curve.phase.unwrap().to_vec(), vec![-10.0; 2]);
        assert!(curve.min_phase.is_none());
        assert!(curve.excess_phase.is_none());
        assert!(curve.excess_delay_ms.is_none());
    }

    #[test]
    fn apply_subtracts_mag_and_phase() {
        // Curve freq grid = cal freq grid → exact subtraction.
        let freq = Array1::from_vec(vec![20.0, 200.0, 2000.0]);
        let cal = MicPhaseCalibration {
            freq: freq.clone(),
            mag_db: Array1::from_vec(vec![2.0, 0.0, -3.0]),
            phase_deg: Array1::from_vec(vec![-10.0, 0.0, 5.0]),
            coherence: Array1::from_vec(vec![0.5, 1.0, 0.8]),
        };
        let mut curve = Curve {
            freq: freq.clone(),
            spl: Array1::from_vec(vec![60.0, 80.0, 90.0]),
            phase: Some(Array1::from_vec(vec![0.0, 0.0, 0.0])),
            coherence: Some(Array1::from_vec(vec![1.0, 1.0, 1.0])),
            ..Default::default()
        };
        cal.apply_to_curve(&mut curve).unwrap();
        // spl: subtract mic's magnitude
        assert!((curve.spl[0] - 58.0).abs() < 1e-9); // 60 - 2
        assert!((curve.spl[1] - 80.0).abs() < 1e-9); // 80 - 0
        assert!((curve.spl[2] - 93.0).abs() < 1e-9); // 90 - (-3)
        // phase: subtract mic's phase
        let phase = curve.phase.as_ref().unwrap();
        assert!((phase[0] - 10.0).abs() < 1e-9); // 0 - (-10)
        assert!((phase[1] - 0.0).abs() < 1e-9);
        assert!((phase[2] + 5.0).abs() < 1e-9); // 0 - 5
        // coherence: multiply by mic's coherence
        let coh = curve.coherence.as_ref().unwrap();
        assert!((coh[0] - 0.5).abs() < 1e-9);
        assert!((coh[1] - 1.0).abs() < 1e-9);
        assert!((coh[2] - 0.8).abs() < 1e-9);
    }

    #[test]
    fn apply_skips_when_phase_or_coherence_absent() {
        // Without phase/coherence on the Curve, the cal should only
        // modify `spl` and not panic.
        let freq = Array1::from_vec(vec![20.0, 200.0]);
        let cal = MicPhaseCalibration {
            freq: freq.clone(),
            mag_db: Array1::from_vec(vec![2.0, 0.0]),
            phase_deg: Array1::from_vec(vec![-10.0, 0.0]),
            coherence: Array1::from_vec(vec![0.5, 1.0]),
        };
        let mut curve = Curve {
            freq,
            spl: Array1::from_vec(vec![60.0, 80.0]),
            phase: None,
            coherence: None,
            ..Default::default()
        };
        cal.apply_to_curve(&mut curve).unwrap();
        assert_eq!(curve.phase, None);
        assert_eq!(curve.coherence, None);
        assert!((curve.spl[0] - 58.0).abs() < 1e-9);
    }

    #[test]
    fn load_mic_phase_calibration_empty_file_errors() {
        let csv = "frequency_hz,mag_db,phase_deg,coherence\n";
        let f = write_cal_csv(csv);
        assert!(load_mic_phase_calibration(f.path()).is_err());
    }

    #[test]
    fn load_mic_phase_calibration_comments_and_blanks_skipped() {
        let csv = "\
# This is a comment
frequency_hz,mag_db,phase_deg,coherence

20.0,2.0,-10.0,0.95
// Another comment
50.0,1.0,-5.0,0.98
";
        let f = write_cal_csv(csv);
        let cal = load_mic_phase_calibration(f.path()).unwrap();
        assert_eq!(cal.freq.len(), 2);
        assert!((cal.freq[1] - 50.0).abs() < 1e-9);
    }

    #[test]
    fn load_mic_phase_calibration_short_row_rejected() {
        let csv = "\
frequency_hz,mag_db,phase_deg,coherence
20.0,2.0,-10.0,0.95
50.0,1.0\n200.0,0.0,0.0,0.99
";
        let f = write_cal_csv(csv);
        let error = load_mic_phase_calibration(f.path())
            .unwrap_err()
            .to_string();
        assert!(error.contains("row 3"), "{error}");
    }

    #[test]
    fn load_mic_phase_calibration_non_finite_value_rejected() {
        let csv = "\
frequency_hz,mag_db,phase_deg,coherence
20.0,2.0,-10.0,0.95
50.0,inf,-5.0,0.98
200.0,0.0,0.0,0.99
";
        let f = write_cal_csv(csv);
        let error = load_mic_phase_calibration(f.path())
            .unwrap_err()
            .to_string();
        assert!(error.contains("row 3"), "{error}");
    }

    #[test]
    fn sample_at_empty_cal_returns_none() {
        let cal = MicPhaseCalibration {
            freq: Array1::from_vec(vec![]),
            mag_db: Array1::from_vec(vec![]),
            phase_deg: Array1::from_vec(vec![]),
            coherence: Array1::from_vec(vec![]),
        };
        assert!(cal.sample_at(100.0).is_none());
    }

    #[test]
    fn sample_at_non_finite_freq_returns_none() {
        let cal = MicPhaseCalibration {
            freq: Array1::from_vec(vec![20.0, 200.0]),
            mag_db: Array1::from_vec(vec![0.0, 0.0]),
            phase_deg: Array1::from_vec(vec![0.0, 0.0]),
            coherence: Array1::from_vec(vec![1.0, 1.0]),
        };
        assert!(cal.sample_at(f64::NAN).is_none());
        assert!(cal.sample_at(f64::INFINITY).is_none());
    }

    #[test]
    fn sample_at_exact_first_node_returns_first_value() {
        let cal = MicPhaseCalibration {
            freq: Array1::from_vec(vec![100.0, 200.0]),
            mag_db: Array1::from_vec(vec![1.0, 2.0]),
            phase_deg: Array1::from_vec(vec![0.0, 10.0]),
            coherence: Array1::from_vec(vec![0.9, 0.95]),
        };
        let (m, p, c) = cal.sample_at(100.0).unwrap();
        assert!((m - 1.0).abs() < 1e-9);
        assert!((p - 0.0).abs() < 1e-9);
        assert!((c - 0.9).abs() < 1e-9);
    }

    #[test]
    fn apply_to_curve_empty_curve_no_panic() {
        let cal = MicPhaseCalibration::identity(Array1::from_vec(vec![20.0, 200.0]));
        let mut curve = Curve {
            freq: Array1::from_vec(vec![]),
            spl: Array1::from_vec(vec![]),
            phase: None,
            coherence: None,
            ..Default::default()
        };
        assert!(cal.apply_to_curve(&mut curve).is_err());
        assert!(curve.spl.is_empty());
    }

    #[test]
    fn apply_to_curve_mismatched_lengths_no_panic() {
        let cal = MicPhaseCalibration::identity(Array1::from_vec(vec![20.0, 200.0]));
        let mut curve = Curve {
            freq: Array1::from_vec(vec![20.0, 200.0]),
            spl: Array1::from_vec(vec![80.0]), // mismatched
            phase: None,
            coherence: None,
            ..Default::default()
        };
        assert!(cal.apply_to_curve(&mut curve).is_err());
        // Explicit failure must leave the input unchanged
        assert_eq!(curve.spl.len(), 1);
    }

    #[test]
    fn apply_to_curve_phase_mismatched_length_skipped() {
        let freq = Array1::from_vec(vec![20.0, 200.0]);
        let cal = MicPhaseCalibration::identity(freq.clone());
        let mut curve = Curve {
            freq,
            spl: Array1::from_vec(vec![80.0, 80.0]),
            phase: Some(Array1::from_vec(vec![0.0])), // wrong length
            coherence: Some(Array1::from_vec(vec![1.0, 1.0])),
            ..Default::default()
        };
        assert!(cal.apply_to_curve(&mut curve).is_err());
        // Invalid optional fields reject the entire correction
        assert_eq!(curve.phase.as_ref().unwrap().len(), 1);
        assert_eq!(curve.spl[0], 80.0);
    }
}
