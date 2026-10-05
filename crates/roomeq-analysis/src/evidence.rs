//! Band-local measurement confidence (plan task A1).
//!
//! Pure computations on supplied curves: per-band magnitude repeatability
//! across repeats of one setup, timing uncertainty with its
//! frequency-dependent phase conversion, and band-local support from
//! coherence/SNR evidence.
//!
//! Repeat-to-repeat variation is never mixed with seat-to-seat variation.
//! Validity gaps and source grids are preserved: unalignable grids are
//! rejected, never zipped by index.

use serde::{Deserialize, Serialize};

use autoeq_core::{EvidenceBand, Uncertainty, UncertaintyKind, timing_uncertainty_to_phase_deg};

use crate::Curve;
use crate::error::{AutoeqError, Result};

/// How a reported spread was estimated. The kind travels with the number so
/// downstream code cannot silently combine different estimators, and bounded
/// errors are never treated as independent variances.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EstimatorKind {
    /// Full max-min range across repeats inside the band.
    MaxMinRange,
    /// Spread of arrival-time estimates across repeats.
    ArrivalSpread,
    /// No estimate: single repeat, coverage gap, or missing input.
    Unknown,
}

/// Per-band magnitude repeatability across repeats of the SAME setup.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BandRepeatability {
    pub band_lo_hz: f64,
    pub band_hi_hz: f64,
    /// Worst-bin max-min spread across repeats in dB. `None` means unknown,
    /// never zero error; see `reason`.
    pub spread_db: Option<f64>,
    /// Repeats contributing to this band (0 inside a coverage gap).
    pub repeat_count: usize,
    pub estimator: EstimatorKind,
    pub evidence_refs: Vec<String>,
    pub reason: Option<String>,
}

/// Per-band seat-to-seat spread. A deliberately separate type from
/// [`BandRepeatability`]: a spatial difference is not measurement noise.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BandSeatSpread {
    pub band_lo_hz: f64,
    pub band_hi_hz: f64,
    pub spread_db: Option<f64>,
    pub seat_count: usize,
    pub estimator: EstimatorKind,
    pub evidence_refs: Vec<String>,
    pub reason: Option<String>,
}

/// Timing uncertainty with the sample count and estimator that produced it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TimingUncertainty {
    /// One-sided timing uncertainty in seconds (bounded error, not a sigma).
    pub seconds: f64,
    pub sample_count: usize,
    pub estimator: EstimatorKind,
    pub evidence_refs: Vec<String>,
}

/// Timing uncertainty converted to phase uncertainty at one frequency.
///
/// Both terms are reported individually: bounded timing error and frequency
/// are not statistically combined with anything else.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct PhaseUncertainty {
    pub frequency_hz: f64,
    pub timing_s: f64,
    pub phase_deg: f64,
}

/// Convert timing uncertainty (seconds) to phase uncertainty (degrees).
///
/// Delegates to the core C2 conversion
/// ([`timing_uncertainty_to_phase_deg`]): 0.5 ms gives 18 deg at 100 Hz
/// and 180 deg at 1 kHz (F02). Returns `None` for non-physical input.
pub fn phase_uncertainty_deg(timing_s: f64, frequency_hz: f64) -> Option<f64> {
    timing_uncertainty_to_phase_deg(frequency_hz, timing_s).ok()
}

/// Convert one timing uncertainty into per-frequency phase terms.
pub fn phase_uncertainty_terms(
    timing: &TimingUncertainty,
    frequencies_hz: &[f64],
) -> Vec<PhaseUncertainty> {
    frequencies_hz
        .iter()
        .copied()
        .filter_map(|frequency_hz| {
            phase_uncertainty_deg(timing.seconds, frequency_hz).map(|phase_deg| PhaseUncertainty {
                frequency_hz,
                timing_s: timing.seconds,
                phase_deg,
            })
        })
        .collect()
}

/// Versioned policy for band-local support judgments. Thresholds are explicit
/// configuration, never universal acoustic constants.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ConfidencePolicy {
    pub version: String,
    pub min_coherence: f64,
    pub min_snr_db: f64,
}

impl ConfidencePolicy {
    pub fn v1() -> Self {
        Self {
            version: "analysis-confidence-v1".to_string(),
            min_coherence: 0.7,
            min_snr_db: 10.0,
        }
    }

    pub fn validate(&self) -> Result<()> {
        if !self.min_coherence.is_finite()
            || !(0.0..=1.0).contains(&self.min_coherence)
            || !self.min_snr_db.is_finite()
        {
            return Err(AutoeqError::InvalidConfiguration {
                message: format!(
                    "confidence policy {} has non-physical thresholds",
                    self.version
                ),
            });
        }
        Ok(())
    }
}

/// Band-local support of one curve: each band is judged on its own bins, so a
/// good broadband median can never hide a bad crossover band (F06).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BandSupport {
    Supported,
    Restricted,
    /// No coherence/SNR evidence supplied: unknown, never good quality.
    Unknown,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BandSupportReport {
    pub band_lo_hz: f64,
    pub band_hi_hz: f64,
    pub support: BandSupport,
    pub min_coherence: Option<f64>,
    pub min_snr_db: Option<f64>,
    pub policy_version: String,
    pub reason: Option<String>,
}

fn check_aligned(curves: &[Curve], context: &str) -> Result<()> {
    if curves.is_empty() {
        return Err(AutoeqError::InvalidMeasurement {
            message: format!("{context} needs at least one curve"),
        });
    }
    curves
        .first()
        .expect("non-empty")
        .validate(context)
        .map_err(|error| AutoeqError::InvalidMeasurement {
            message: error.to_string(),
        })?;
    if !crate::frequency_grid::is_valid_frequency_grid(&curves[0].freq) {
        return Err(AutoeqError::InvalidMeasurement {
            message: format!("{context} reference curve has an invalid frequency grid"),
        });
    }
    for (index, curve) in curves.iter().enumerate().skip(1) {
        if curve.validate(context).is_err()
            || !crate::frequency_grid::is_valid_frequency_grid(&curve.freq)
            || curve.spl.len() != curve.freq.len()
            || !crate::frequency_grid::same_frequency_grid(&curves[0].freq, &curve.freq)
        {
            // F05: grids that cannot be aligned are rejected outright.
            return Err(AutoeqError::InvalidMeasurement {
                message: format!("{context} curve {index} cannot be aligned to the reference grid"),
            });
        }
    }
    Ok(())
}

fn band_bins(freq: &ndarray::Array1<f64>, band_lo_hz: f64, band_hi_hz: f64) -> Vec<usize> {
    freq.iter()
        .enumerate()
        .filter(|(_, frequency)| **frequency >= band_lo_hz && **frequency <= band_hi_hz)
        .map(|(index, _)| index)
        .collect()
}

fn max_min_spread(curves: &[Curve], bins: &[usize]) -> Option<f64> {
    let mut worst = 0.0_f64;
    for &bin in bins {
        let mut min = f64::INFINITY;
        let mut max = f64::NEG_INFINITY;
        for curve in curves {
            let level = curve.spl[bin];
            if !level.is_finite() {
                return None;
            }
            min = min.min(level);
            max = max.max(level);
        }
        worst = worst.max(max - min);
    }
    Some(worst)
}

/// Per-band magnitude repeatability across repeats of one setup.
///
/// Bands with no covered bins keep their gap: `spread_db` is `None` with
/// reason `coverage_gap`, and the band bounds are preserved.
pub fn compute_band_repeatability(
    repeats: &[Curve],
    bands: &[(f64, f64)],
    evidence_ref: &str,
) -> Result<Vec<BandRepeatability>> {
    check_aligned(repeats, "repeatability")?;
    let mut out = Vec::with_capacity(bands.len());
    for &(band_lo_hz, band_hi_hz) in bands {
        let bins = band_bins(&repeats[0].freq, band_lo_hz, band_hi_hz);
        if bins.is_empty() {
            out.push(BandRepeatability {
                band_lo_hz,
                band_hi_hz,
                spread_db: None,
                repeat_count: 0,
                estimator: EstimatorKind::Unknown,
                evidence_refs: vec![evidence_ref.to_string()],
                reason: Some("coverage_gap".to_string()),
            });
            continue;
        }
        if repeats.len() < 2 {
            out.push(BandRepeatability {
                band_lo_hz,
                band_hi_hz,
                spread_db: None,
                repeat_count: repeats.len(),
                estimator: EstimatorKind::Unknown,
                evidence_refs: vec![evidence_ref.to_string()],
                reason: Some("single_repeat_unknown".to_string()),
            });
            continue;
        }
        let spread_db = max_min_spread(repeats, &bins);
        out.push(BandRepeatability {
            band_lo_hz,
            band_hi_hz,
            spread_db,
            repeat_count: repeats.len(),
            estimator: if spread_db.is_some() {
                EstimatorKind::MaxMinRange
            } else {
                EstimatorKind::Unknown
            },
            evidence_refs: vec![evidence_ref.to_string()],
            reason: if spread_db.is_some() {
                None
            } else {
                Some("nonfinite_level".to_string())
            },
        });
    }
    Ok(out)
}

/// Per-band seat-to-seat spread. Same mechanics as repeatability but a
/// distinct type and evidence role: spatial difference is not noise.
pub fn compute_band_seat_spread(
    seats: &[Curve],
    bands: &[(f64, f64)],
    evidence_ref: &str,
) -> Result<Vec<BandSeatSpread>> {
    check_aligned(seats, "seat spread")?;
    let mut out = Vec::with_capacity(bands.len());
    for &(band_lo_hz, band_hi_hz) in bands {
        let bins = band_bins(&seats[0].freq, band_lo_hz, band_hi_hz);
        if bins.is_empty() {
            out.push(BandSeatSpread {
                band_lo_hz,
                band_hi_hz,
                spread_db: None,
                seat_count: 0,
                estimator: EstimatorKind::Unknown,
                evidence_refs: vec![evidence_ref.to_string()],
                reason: Some("coverage_gap".to_string()),
            });
            continue;
        }
        if seats.len() < 2 {
            out.push(BandSeatSpread {
                band_lo_hz,
                band_hi_hz,
                spread_db: None,
                seat_count: seats.len(),
                estimator: EstimatorKind::Unknown,
                evidence_refs: vec![evidence_ref.to_string()],
                reason: Some("single_seat_unknown".to_string()),
            });
            continue;
        }
        let spread_db = max_min_spread(seats, &bins);
        out.push(BandSeatSpread {
            band_lo_hz,
            band_hi_hz,
            spread_db,
            seat_count: seats.len(),
            estimator: if spread_db.is_some() {
                EstimatorKind::MaxMinRange
            } else {
                EstimatorKind::Unknown
            },
            evidence_refs: vec![evidence_ref.to_string()],
            reason: if spread_db.is_some() {
                None
            } else {
                Some("nonfinite_level".to_string())
            },
        });
    }
    Ok(out)
}

/// Convert band repeatability reports into core K1 evidence bands.
///
/// A known max-min spread becomes a [`UncertaintyKind::Bound`] spread: the
/// observed range bounds the repeat variation without claiming a statistical
/// distribution. Unknown spreads (gaps, single repeats) stay `None`, and the
/// band bounds, reason codes, and evidence references are preserved.
pub fn repeatability_to_evidence_bands(reports: &[BandRepeatability]) -> Vec<EvidenceBand> {
    reports
        .iter()
        .map(|report| EvidenceBand {
            id: format!("{}-{}hz", report.band_lo_hz, report.band_hi_hz),
            low_hz: report.band_lo_hz,
            high_hz: report.band_hi_hz,
            snr_db: None,
            coherence: None,
            spread: report.spread_db.map(|spread_db| Uncertainty {
                kind: UncertaintyKind::Bound,
                magnitude_db: spread_db,
            }),
            timing_uncertainty_s: None,
            reasons: report.reason.clone().into_iter().collect(),
            references: report.evidence_refs.clone(),
        })
        .collect()
}

/// Band-local support from per-bin coherence and SNR evidence.
///
/// A band is `Restricted` when any of its bins falls below policy; `Unknown`
/// when the curve carries no coherence/SNR evidence at all. There is no
/// broadband roll-up: the caller sees every band.
pub fn assess_band_support(
    curve: &Curve,
    bands: &[(f64, f64)],
    policy: &ConfidencePolicy,
) -> Result<Vec<BandSupportReport>> {
    policy.validate()?;
    curve
        .validate("band support")
        .map_err(|error| AutoeqError::InvalidMeasurement {
            message: error.to_string(),
        })?;
    let mut out = Vec::with_capacity(bands.len());
    for &(band_lo_hz, band_hi_hz) in bands {
        let bins = band_bins(&curve.freq, band_lo_hz, band_hi_hz);
        if bins.is_empty() {
            out.push(BandSupportReport {
                band_lo_hz,
                band_hi_hz,
                support: BandSupport::Unknown,
                min_coherence: None,
                min_snr_db: None,
                policy_version: policy.version.clone(),
                reason: Some("coverage_gap".to_string()),
            });
            continue;
        }
        let coherence_ok = curve
            .coherence
            .as_ref()
            .filter(|values| values.len() == curve.spl.len());
        let noise_ok = curve
            .noise_floor_db
            .as_ref()
            .filter(|values| values.len() == curve.spl.len());
        let (Some(coherence), Some(noise)) = (coherence_ok, noise_ok) else {
            out.push(BandSupportReport {
                band_lo_hz,
                band_hi_hz,
                support: BandSupport::Unknown,
                min_coherence: None,
                min_snr_db: None,
                policy_version: policy.version.clone(),
                reason: Some("missing_coherence_or_noise_evidence".to_string()),
            });
            continue;
        };
        let mut min_coherence = f64::INFINITY;
        let mut min_snr = f64::INFINITY;
        let mut finite = true;
        for &bin in &bins {
            let coherence_value = coherence[bin];
            let snr = curve.spl[bin] - noise[bin];
            if !coherence_value.is_finite() || !snr.is_finite() {
                finite = false;
                break;
            }
            min_coherence = min_coherence.min(coherence_value);
            min_snr = min_snr.min(snr);
        }
        if !finite {
            out.push(BandSupportReport {
                band_lo_hz,
                band_hi_hz,
                support: BandSupport::Unknown,
                min_coherence: None,
                min_snr_db: None,
                policy_version: policy.version.clone(),
                reason: Some("nonfinite_band_evidence".to_string()),
            });
            continue;
        }
        let restricted = min_coherence < policy.min_coherence || min_snr < policy.min_snr_db;
        out.push(BandSupportReport {
            band_lo_hz,
            band_hi_hz,
            support: if restricted {
                BandSupport::Restricted
            } else {
                BandSupport::Supported
            },
            min_coherence: Some(min_coherence),
            min_snr_db: Some(min_snr),
            policy_version: policy.version.clone(),
            reason: if restricted {
                Some("band_below_policy".to_string())
            } else {
                None
            },
        });
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;

    fn grid() -> Array1<f64> {
        Array1::from_vec(vec![
            20.0, 40.0, 80.0, 160.0, 320.0, 640.0, 1280.0, 2560.0, 5120.0, 10240.0,
        ])
    }

    fn flat_curve(offset_db: f64) -> Curve {
        Curve {
            freq: grid(),
            spl: Array1::from_vec(vec![80.0 + offset_db; 10]),
            ..Curve::default()
        }
    }

    fn curve_with_band(coherence: f64, snr_db: f64) -> Curve {
        let spl = Array1::from_vec(vec![80.0; 10]);
        Curve {
            freq: grid(),
            spl: spl.clone(),
            coherence: Some(Array1::from_vec(vec![coherence; 10])),
            noise_floor_db: Some(spl.mapv(|level| level - snr_db)),
            ..Curve::default()
        }
    }

    #[test]
    fn analysis_good_median_does_not_hide_bad_crossover() {
        // F06: nine good bins must not rescue one bad crossover band.
        let mut curve = curve_with_band(0.95, 25.0);
        let mut coherence = curve.coherence.clone().expect("coherence");
        coherence[3] = 0.4; // 160 Hz crossover bin
        let mut noise = curve.noise_floor_db.clone().expect("noise");
        noise[3] = 80.0 - 3.0;
        curve.coherence = Some(coherence);
        curve.noise_floor_db = Some(noise);
        let bands = vec![(20.0, 100.0), (100.0, 300.0), (300.0, 12000.0)];
        let reports =
            assess_band_support(&curve, &bands, &ConfidencePolicy::v1()).expect("band support");
        assert_eq!(reports[0].support, BandSupport::Supported);
        assert_eq!(reports[1].support, BandSupport::Restricted);
        assert_eq!(reports[2].support, BandSupport::Supported);
        assert_eq!(reports[1].policy_version, ConfidencePolicy::v1().version);
    }

    #[test]
    fn analysis_repeatability_is_not_seat_variance() {
        // Tightly repeated measurement at one position.
        let repeats = vec![flat_curve(0.0), flat_curve(0.1), flat_curve(-0.1)];
        // Seats 3 dB apart: spatial difference, not noise.
        let seats = vec![flat_curve(0.0), flat_curve(3.0)];
        let bands = vec![(20.0, 12000.0)];
        let repeatability =
            compute_band_repeatability(&repeats, &bands, "repeat-ev").expect("repeatability");
        let seat_spread = compute_band_seat_spread(&seats, &bands, "seat-ev").expect("seat spread");
        let repeat_spread = repeatability[0].spread_db.expect("repeat spread");
        let seat_value = seat_spread[0].spread_db.expect("seat spread");
        assert!((repeat_spread - 0.2).abs() < 1e-9, "{repeat_spread}");
        assert!((seat_value - 3.0).abs() < 1e-9, "{seat_value}");
        assert_eq!(repeatability[0].repeat_count, 3);
        assert_eq!(repeatability[0].estimator, EstimatorKind::MaxMinRange);
        assert_eq!(repeatability[0].evidence_refs, vec!["repeat-ev"]);
        assert_ne!(seat_value, repeat_spread);
    }

    #[test]
    fn analysis_phase_uncertainty_tracks_frequency() {
        // F02 exact conversion: 0.5 ms -> 18 deg at 100 Hz, 180 deg at 1 kHz.
        let timing = TimingUncertainty {
            seconds: 0.0005,
            sample_count: 5,
            estimator: EstimatorKind::ArrivalSpread,
            evidence_refs: vec!["arrival-ev".to_string()],
        };
        let terms = phase_uncertainty_terms(&timing, &[100.0, 1000.0]);
        assert!((terms[0].phase_deg - 18.0).abs() < 1e-9);
        assert!((terms[1].phase_deg - 180.0).abs() < 1e-9);
        // Terms are reported individually: timing survives alongside phase.
        assert_eq!(terms[0].timing_s, 0.0005);
        assert!(phase_uncertainty_deg(-1.0, 100.0).is_none());
        assert!(phase_uncertainty_deg(0.0005, 0.0).is_none());
    }

    #[test]
    fn analysis_repeatability_bridges_to_core_evidence_bands() {
        let repeats = vec![flat_curve(0.0), flat_curve(0.2)];
        let bands = vec![(20.0, 100.0), (30000.0, 40000.0)];
        let reports = compute_band_repeatability(&repeats, &bands, "ev").expect("repeatability");
        let evidence = repeatability_to_evidence_bands(&reports);
        assert_eq!(evidence.len(), 2);
        let known = &evidence[0];
        assert_eq!((known.low_hz, known.high_hz), (20.0, 100.0));
        let spread = known.spread.expect("known spread");
        assert_eq!(spread.kind, UncertaintyKind::Bound);
        assert!((spread.magnitude_db - 0.2).abs() < 1e-9);
        assert_eq!(known.references, vec!["ev"]);
        assert!(known.validate("bridge").is_ok());
        let gap = &evidence[1];
        assert_eq!(gap.spread, None);
        assert_eq!(gap.reasons, vec!["coverage_gap"]);
    }

    #[test]
    fn analysis_grid_mismatch_and_gap_preserved() {
        // F05: same-length arrays on different grids must be rejected.
        let mut shifted = flat_curve(0.0);
        shifted.freq = Array1::from_vec(vec![
            21.0, 41.0, 81.0, 161.0, 321.0, 641.0, 1281.0, 2561.0, 5121.0, 10241.0,
        ]);
        let repeats = vec![flat_curve(0.0), shifted];
        let bands = vec![(20.0, 12000.0)];
        assert!(compute_band_repeatability(&repeats, &bands, "ev").is_err());
        // A coverage gap is preserved as unknown, not interpolated.
        let repeats = vec![flat_curve(0.0), flat_curve(0.2)];
        let gapped = vec![(20.0, 100.0), (30000.0, 40000.0)];
        let reports = compute_band_repeatability(&repeats, &gapped, "ev").expect("gap report");
        assert!(reports[0].spread_db.is_some());
        assert_eq!(reports[1].spread_db, None);
        assert_eq!(reports[1].repeat_count, 0);
        assert_eq!(reports[1].estimator, EstimatorKind::Unknown);
        assert_eq!(reports[1].reason.as_deref(), Some("coverage_gap"));
        assert_eq!(
            (reports[1].band_lo_hz, reports[1].band_hi_hz),
            (30000.0, 40000.0)
        );
    }
}
