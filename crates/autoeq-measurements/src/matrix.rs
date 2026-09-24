//! Per-take, per-source, per-seat evidence retention (lane L3).
//!
//! A [`TakeMatrix`] keeps the complete source-by-seat matrix: accepted,
//! rejected and zero-weight takes all stay present with explicit reasons,
//! so bad takes are never silently discarded because they make the result
//! worse. Loaders never invent missing source/seat entries; gaps are
//! reported by [`CompletenessReport`].
//!
//! Averaging is explicit about its kind ([`AverageKind`]). Spatial
//! magnitude averages never carry measured phase. Coherent averaging
//! additionally requires a common timing/gain reference across every
//! averaged take, on top of the calibration/confidence contract enforced
//! by [`crate::coherent_average_measurement`].
//!
//! Coverage masks travel with resampling: bins outside measured support
//! or inside a coverage gap stay invalid, so resampling cannot extend
//! valid-band claims. The mask stays lane-local (core C2 aligns
//! `EvidenceBand`s, not bool masks), but every resample validates the
//! target grid with core C2 [`autoeq_core::alignment::validate_alignment_grid`]
//! before arithmetic, so non-finite or unordered grids fail instead of
//! silently producing support. The invariant (gaps preserved, no invented
//! support) is asserted by `measurement_resampling_preserves_invalid_band`.

use ndarray::Array1;
use serde::{Deserialize, Serialize};

use crate::{CoherentAverageContract, Curve};

/// How repeated takes are combined. Explicit on every matrix.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AverageKind {
    /// Power-domain (RMS) magnitude average. No phase.
    #[default]
    Power,
    /// Magnitude average of the same RMS family, kept as a distinct
    /// declared kind. No phase.
    Magnitude,
    /// Arithmetic mean in dB. No phase.
    Decibel,
    /// Complex-pressure mean. Carries phase; requires a common
    /// timing/gain reference plus a satisfied coherent contract.
    Coherent,
}

/// Acceptance state of one take. Rejected and zero-weight takes are
/// retained with reasons, never dropped.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TakeDecision {
    Accepted,
    Rejected { reason: String },
    ZeroWeight { reason: String },
}

impl TakeDecision {
    pub fn is_accepted(&self) -> bool {
        matches!(self, TakeDecision::Accepted)
    }

    pub fn reason(&self) -> Option<&str> {
        match self {
            TakeDecision::Accepted => None,
            TakeDecision::Rejected { reason } | TakeDecision::ZeroWeight { reason } => Some(reason),
        }
    }
}

/// One retained take: a single capture of one source at one seat.
///
/// `Curve` carries float arrays without `PartialEq`, so takes compare
/// through their serialized form (see the handoff roundtrip test).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Take {
    pub take_id: String,
    pub source_id: String,
    pub seat_id: String,
    pub weight: f64,
    pub decision: TakeDecision,
    pub curve: Option<Curve>,
    /// Timing/gain reference this take is bound to. Coherent averaging
    /// requires every averaged take to share one value.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reference_id: Option<String>,
}

impl Take {
    pub fn is_averaging_eligible(&self) -> bool {
        self.decision.is_accepted() && self.weight > 0.0 && self.curve.is_some()
    }
}

/// Complete source-by-seat matrix with an explicit average kind.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TakeMatrix {
    pub takes: Vec<Take>,
    #[serde(default)]
    pub average_kind: AverageKind,
}

impl TakeMatrix {
    /// Curves eligible for averaging: accepted, positive weight, present.
    pub fn averaging_set(&self) -> Vec<&Curve> {
        self.takes
            .iter()
            .filter(|take| take.is_averaging_eligible())
            .filter_map(|take| take.curve.as_ref())
            .collect()
    }

    /// Average the eligible takes under the declared [`AverageKind`].
    ///
    /// Never invents entries: an empty eligible set is an error, not a
    /// fabricated curve. Spatial kinds (`Power`, `Magnitude`, `Decibel`)
    /// never carry phase. `Coherent` requires every eligible take to
    /// share one non-empty `reference_id` and delegates phase/confidence
    /// gating to the caller's [`CoherentAverageContract`] (seat ids must
    /// match take ids in eligible order).
    pub fn average(
        &self,
        contract: Option<&CoherentAverageContract>,
    ) -> Result<Curve, crate::MeasurementError> {
        const OPERATION: &str = "take_matrix_average";
        let eligible: Vec<&Take> = self
            .takes
            .iter()
            .filter(|take| take.is_averaging_eligible())
            .collect();
        if eligible.is_empty() {
            return Err(crate::MeasurementError::InvalidEvidence {
                measurement: "take-matrix".into(),
                operation: OPERATION.into(),
                message: "no eligible takes: refusing to invent an average".into(),
            });
        }
        let curves: Vec<Curve> = eligible
            .iter()
            .map(|take| take.curve.clone().expect("eligibility guarantees a curve"))
            .collect();
        match self.average_kind {
            AverageKind::Power | AverageKind::Magnitude => Ok(power_average(&curves)),
            AverageKind::Decibel => Ok(decibel_average(&curves)),
            AverageKind::Coherent => {
                let reference = eligible[0]
                    .reference_id
                    .as_deref()
                    .filter(|id| !id.trim().is_empty());
                let Some(reference) = reference else {
                    return Err(crate::MeasurementError::InvalidEvidence {
                        measurement: "take-matrix".into(),
                        operation: OPERATION.into(),
                        message: "coherent average requires a common timing/gain reference \
                                  (first take has none)"
                            .into(),
                    });
                };
                for take in &eligible[1..] {
                    if take.reference_id.as_deref() != Some(reference) {
                        return Err(crate::MeasurementError::InvalidEvidence {
                            measurement: "take-matrix".into(),
                            operation: OPERATION.into(),
                            message: format!(
                                "coherent average requires a common timing/gain reference: \
                                 take '{}' does not share reference '{reference}'",
                                take.take_id
                            ),
                        });
                    }
                }
                let contract =
                    contract.ok_or_else(|| crate::MeasurementError::InvalidEvidence {
                        measurement: "take-matrix".into(),
                        operation: OPERATION.into(),
                        message: "coherent average requires an explicit coherent contract".into(),
                    })?;
                let ids: Vec<&str> = contract
                    .seats
                    .iter()
                    .map(|seat| seat.seat_id.as_str())
                    .collect();
                let expected: Vec<&str> =
                    eligible.iter().map(|take| take.take_id.as_str()).collect();
                if ids != expected {
                    return Err(crate::MeasurementError::InvalidEvidence {
                        measurement: "take-matrix".into(),
                        operation: OPERATION.into(),
                        message: format!(
                            "coherent contract seats {ids:?} do not cover the eligible takes {expected:?}"
                        ),
                    });
                }
                crate::coherent_average_measurement(&curves, contract).map_err(|error| {
                    crate::MeasurementError::InvalidEvidence {
                        measurement: "take-matrix".into(),
                        operation: OPERATION.into(),
                        message: error.to_string(),
                    }
                })
            }
        }
    }

    /// Report matrix completeness against expected (source, seat) pairs.
    ///
    /// Missing pairs are reported, never filled with invented entries.
    /// Rejected and zero-weight takes are listed with their reasons.
    pub fn completeness_report(&self, expected: &[MatrixKey]) -> CompletenessReport {
        let mut missing = Vec::new();
        for key in expected {
            let present = self
                .takes
                .iter()
                .any(|take| take.source_id == key.source_id && take.seat_id == key.seat_id);
            if !present {
                missing.push(key.clone());
            }
        }
        let mut rejected = Vec::new();
        let mut accepted = 0;
        for take in &self.takes {
            match &take.decision {
                TakeDecision::Accepted => {
                    if take.weight > 0.0 {
                        accepted += 1;
                    } else {
                        rejected.push(RejectedTake {
                            take_id: take.take_id.clone(),
                            reason: "accepted take has non-positive weight".into(),
                        });
                    }
                }
                TakeDecision::Rejected { reason } | TakeDecision::ZeroWeight { reason } => {
                    rejected.push(RejectedTake {
                        take_id: take.take_id.clone(),
                        reason: reason.clone(),
                    });
                }
            }
        }
        CompletenessReport {
            missing,
            rejected,
            accepted,
            total: self.takes.len(),
        }
    }
}

/// Expected (source, seat) entry for completeness reporting.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MatrixKey {
    pub source_id: String,
    pub seat_id: String,
}

/// One retained rejection with its handoff reason.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RejectedTake {
    pub take_id: String,
    pub reason: String,
}

/// Completeness outcome: what is present, missing, or rejected.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CompletenessReport {
    pub missing: Vec<MatrixKey>,
    pub rejected: Vec<RejectedTake>,
    pub accepted: usize,
    pub total: usize,
}

impl CompletenessReport {
    pub fn is_complete(&self) -> bool {
        self.missing.is_empty() && self.rejected.is_empty()
    }
}

fn grids_match(left: &Array1<f64>, right: &Array1<f64>) -> bool {
    left.len() == right.len() && left.iter().zip(right).all(|(a, b)| (a - b).abs() <= 1e-9)
}

/// Power-domain (RMS) magnitude average. Phase is always `None`: an RMS
/// magnitude must never be paired with an averaged angle.
fn power_average(curves: &[Curve]) -> Curve {
    let freq = curves[0].freq.clone();
    debug_assert!(curves.iter().all(|curve| grids_match(&freq, &curve.freq)));
    let mut power_sum = Array1::<f64>::zeros(freq.len());
    for curve in curves {
        let grid = if grids_match(&freq, &curve.freq) {
            curve.spl.clone()
        } else {
            crate::interpolate_log_space(&freq, curve).spl
        };
        power_sum = power_sum + grid.mapv(|spl| 10.0_f64.powf(spl / 10.0));
    }
    let average = power_sum / curves.len() as f64;
    Curve {
        freq,
        spl: average.mapv(|power| 10.0 * power.log10()),
        phase: None,
        coherence: None,
        ..Default::default()
    }
}

/// Arithmetic mean in dB. Phase is always `None`.
fn decibel_average(curves: &[Curve]) -> Curve {
    let freq = curves[0].freq.clone();
    let mut sum = Array1::<f64>::zeros(freq.len());
    for curve in curves {
        let grid = if grids_match(&freq, &curve.freq) {
            curve.spl.clone()
        } else {
            crate::interpolate_log_space(&freq, curve).spl
        };
        sum = sum + grid;
    }
    Curve {
        freq,
        spl: sum / curves.len() as f64,
        phase: None,
        coherence: None,
        ..Default::default()
    }
}

/// Per-bin validity mask over a frequency grid.
///
/// `false` marks bins outside measured support or inside a coverage gap.
/// Masks are carried through resampling so downstream consumers cannot
/// mistake interpolated values for measured support.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CoverageMask {
    pub freq: Vec<f64>,
    pub valid: Vec<bool>,
}

impl CoverageMask {
    pub fn new(freq: Vec<f64>, valid: Vec<bool>) -> Result<Self, crate::MeasurementError> {
        if freq.len() != valid.len() {
            return Err(crate::MeasurementError::InvalidEvidence {
                measurement: "coverage-mask".into(),
                operation: "coverage_mask_new".into(),
                message: format!(
                    "frequency and validity lengths differ ({} vs {})",
                    freq.len(),
                    valid.len()
                ),
            });
        }
        if freq.is_empty() {
            return Err(crate::MeasurementError::InvalidEvidence {
                measurement: "coverage-mask".into(),
                operation: "coverage_mask_new".into(),
                message: "coverage mask must not be empty".into(),
            });
        }
        autoeq_core::alignment::validate_alignment_grid(
            &ndarray::Array1::from_vec(freq.clone()),
            "coverage-mask source",
        )
        .map_err(|error| crate::MeasurementError::InvalidEvidence {
            measurement: "coverage-mask".into(),
            operation: "coverage_mask_new".into(),
            message: error.to_string(),
        })?;
        Ok(Self { freq, valid })
    }

    pub fn all_valid(freq: Vec<f64>) -> Result<Self, crate::MeasurementError> {
        let valid = vec![true; freq.len()];
        Self::new(freq, valid)
    }

    /// Resample the mask onto a target grid without extending support.
    ///
    /// A target bin is valid only when it falls inside a source interval
    /// whose endpoints are both valid (or exactly on a valid source
    /// point). Gap bins and out-of-support targets stay invalid.
    pub fn resample(&self, target_freq: &[f64]) -> Result<Self, crate::MeasurementError> {
        // Public fields and deserialization can bypass `new`; validate the
        // source before `partition_point` assumes sorted frequencies.
        Self::new(self.freq.clone(), self.valid.clone())?;
        if target_freq.is_empty() {
            return Err(crate::MeasurementError::InvalidEvidence {
                measurement: "coverage-mask".into(),
                operation: "coverage_mask_resample".into(),
                message: "target grid must not be empty".into(),
            });
        }
        // Core C2 grid contract: finite, positive, strictly increasing
        // bins are checked before any support arithmetic.
        autoeq_core::alignment::validate_alignment_grid(
            &ndarray::Array1::from_vec(target_freq.to_vec()),
            "coverage-mask resample",
        )
        .map_err(|error| crate::MeasurementError::InvalidEvidence {
            measurement: "coverage-mask".into(),
            operation: "coverage_mask_resample".into(),
            message: error.to_string(),
        })?;
        let valid = target_freq.iter().map(|t| self.valid_at(*t)).collect();
        Ok(Self {
            freq: target_freq.to_vec(),
            valid,
        })
    }

    fn valid_at(&self, target: f64) -> bool {
        if !target.is_finite() || self.freq.is_empty() {
            return false;
        }
        if target < self.freq[0] || target > self.freq[self.freq.len() - 1] {
            return false;
        }
        let tolerance = 1e-9 * target.abs().max(1.0);
        for (index, point) in self.freq.iter().enumerate() {
            if (point - target).abs() <= tolerance {
                return self.valid[index];
            }
        }
        let upper = self.freq.partition_point(|point| *point < target);
        if upper == 0 || upper >= self.freq.len() {
            return false;
        }
        self.valid[upper - 1] && self.valid[upper]
    }
}

/// Resample a curve onto a target grid, carrying its coverage mask.
///
/// The interpolated values outside valid support exist only so array
/// shapes line up; the returned mask is authoritative about which bins
/// are measured. Uncertainty companions (`coherence`, `noise_floor_db`)
/// are interpolated alongside SPL.
pub fn resample_curve_with_coverage(
    curve: &Curve,
    coverage: &CoverageMask,
    target_freq: &Array1<f64>,
) -> Result<(Curve, CoverageMask), crate::MeasurementError> {
    CoverageMask::new(coverage.freq.clone(), coverage.valid.clone())?;
    autoeq_core::alignment::validate_alignment_grid(target_freq, "coverage-resample target")
        .map_err(|error| crate::MeasurementError::InvalidEvidence {
            measurement: "coverage-resample".into(),
            operation: "resample_curve_with_coverage".into(),
            message: error.to_string(),
        })?;
    curve
        .validate("coverage-resample source")
        .map_err(|error| crate::MeasurementError::InvalidEvidence {
            measurement: "coverage-resample".into(),
            operation: "resample_curve_with_coverage".into(),
            message: error.to_string(),
        })?;
    if coverage.freq.len() != curve.freq.len()
        || coverage
            .freq
            .iter()
            .zip(curve.freq.iter())
            .any(|(a, b)| (a - b).abs() > 1e-9)
    {
        return Err(crate::MeasurementError::InvalidEvidence {
            measurement: "coverage-resample".into(),
            operation: "resample_curve_with_coverage".into(),
            message: "coverage mask grid does not match the curve grid".into(),
        });
    }
    let resampled = crate::interpolate_log_space(target_freq, curve);
    let mask = coverage.resample(&target_freq.to_vec())?;
    Ok((resampled, mask))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{CoherentAverageContract, SeatProvenance};
    use ndarray::Array1;

    fn grid() -> Array1<f64> {
        Array1::from_vec(vec![100.0, 1000.0, 10000.0])
    }

    fn phased_curve(spl: f64, phase: f64) -> Curve {
        Curve {
            freq: grid(),
            spl: Array1::from_vec(vec![spl, spl, spl]),
            phase: Some(Array1::from_vec(vec![phase, phase, phase])),
            ..Default::default()
        }
    }

    fn take(id: &str, source: &str, seat: &str, curve: Curve, reference: Option<&str>) -> Take {
        Take {
            take_id: id.into(),
            source_id: source.into(),
            seat_id: seat.into(),
            weight: 1.0,
            decision: TakeDecision::Accepted,
            curve: Some(curve),
            reference_id: reference.map(String::from),
        }
    }

    fn coherent_contract(ids: &[&str]) -> CoherentAverageContract {
        CoherentAverageContract {
            seats: ids
                .iter()
                .map(|id| SeatProvenance {
                    seat_id: (*id).to_string(),
                    calibration_id: Some("mic-1".to_string()),
                    phase_confidence: Some(1.0),
                    ..Default::default()
                })
                .collect(),
            min_phase_confidence: 0.0,
            require_calibration: true,
        }
    }

    #[test]
    fn measurement_spatial_magnitude_has_no_measured_phase() {
        for kind in [
            AverageKind::Power,
            AverageKind::Magnitude,
            AverageKind::Decibel,
        ] {
            let matrix = TakeMatrix {
                takes: vec![
                    take("t0", "L", "seat-0", phased_curve(80.0, 0.0), Some("ref-a")),
                    take(
                        "t1",
                        "L",
                        "seat-1",
                        phased_curve(80.0, 180.0),
                        Some("ref-a"),
                    ),
                ],
                average_kind: kind,
            };
            let average = matrix.average(None).unwrap();
            assert!(
                average.phase.is_none(),
                "kind {kind:?} must not carry phase"
            );
            assert!(average.min_phase.is_none());
            assert!(average.excess_phase.is_none());
        }
        // Power RMS of two equal 80 dB takes stays 80 dB without an angle.
        let matrix = TakeMatrix {
            takes: vec![
                take("t0", "L", "seat-0", phased_curve(80.0, 0.0), Some("ref-a")),
                take(
                    "t1",
                    "L",
                    "seat-1",
                    phased_curve(80.0, 180.0),
                    Some("ref-a"),
                ),
            ],
            average_kind: AverageKind::Power,
        };
        let average = matrix.average(None).unwrap();
        assert!((average.spl[0] - 80.0).abs() < 1e-9);
    }

    #[test]
    fn measurement_coherent_average_requires_common_reference() {
        // Shared reference plus a satisfied contract: genuine complex mean.
        let matrix = TakeMatrix {
            takes: vec![
                take("t0", "L", "seat-0", phased_curve(80.0, 10.0), Some("ref-a")),
                take("t1", "L", "seat-1", phased_curve(80.0, 12.0), Some("ref-a")),
            ],
            average_kind: AverageKind::Coherent,
        };
        let coherent = matrix
            .average(Some(&coherent_contract(&["t0", "t1"])))
            .unwrap();
        assert!(coherent.phase.is_some());

        // Divergent references: rejected even with a satisfied contract.
        let split = TakeMatrix {
            takes: vec![
                take("t0", "L", "seat-0", phased_curve(80.0, 10.0), Some("ref-a")),
                take("t1", "L", "seat-1", phased_curve(80.0, 12.0), Some("ref-b")),
            ],
            average_kind: AverageKind::Coherent,
        };
        let error = split
            .average(Some(&coherent_contract(&["t0", "t1"])))
            .unwrap_err();
        assert!(
            error.to_string().contains("common timing/gain reference"),
            "unexpected error: {error}"
        );

        // Missing reference: rejected, not treated as an implicit common one.
        let missing = TakeMatrix {
            takes: vec![
                take("t0", "L", "seat-0", phased_curve(80.0, 10.0), None),
                take("t1", "L", "seat-1", phased_curve(80.0, 12.0), None),
            ],
            average_kind: AverageKind::Coherent,
        };
        assert!(
            missing
                .average(Some(&coherent_contract(&["t0", "t1"])))
                .is_err()
        );

        // No contract at all: rejected.
        assert!(matrix.average(None).is_err());
    }

    #[test]
    fn measurement_matrix_missing_source_is_reported() {
        let matrix = TakeMatrix {
            takes: vec![take(
                "t0",
                "L",
                "seat-0",
                phased_curve(80.0, 0.0),
                Some("ref-a"),
            )],
            average_kind: AverageKind::Power,
        };
        let report = matrix.completeness_report(&[
            MatrixKey {
                source_id: "L".into(),
                seat_id: "seat-0".into(),
            },
            MatrixKey {
                source_id: "R".into(),
                seat_id: "seat-0".into(),
            },
        ]);
        assert_eq!(report.missing.len(), 1);
        assert_eq!(report.missing[0].source_id, "R");
        assert!(!report.is_complete());
        // The loader reports the gap; it does not fill it.
        assert_eq!(matrix.takes.len(), 1);
        let average = matrix.average(None).unwrap();
        assert_eq!(average.spl.len(), 3);
    }

    #[test]
    fn measurement_resampling_preserves_invalid_band() {
        let freq = vec![100.0, 200.0, 400.0, 800.0, 1600.0, 3200.0];
        let mask =
            CoverageMask::new(freq.clone(), vec![true, true, false, false, true, true]).unwrap();
        // Targets inside the gap and outside support stay invalid.
        let target = vec![100.0, 300.0, 600.0, 1600.0, 6400.0];
        let resampled = mask.resample(&target).unwrap();
        assert_eq!(resampled.valid, vec![true, false, false, true, false]);
        // Exact valid points survive.
        let exact = mask.resample(&[200.0, 3200.0]).unwrap();
        assert_eq!(exact.valid, vec![true, true]);
        // Core C2 grid contract: non-finite, non-positive, or unordered
        // target grids fail instead of silently producing support.
        assert!(mask.resample(&[100.0, f64::NAN, 400.0]).is_err());
        assert!(mask.resample(&[400.0, 100.0]).is_err());
        assert!(mask.resample(&[100.0, -200.0]).is_err());
        assert!(CoverageMask::new(vec![400.0, 100.0], vec![true, true]).is_err());
        assert!(CoverageMask::new(vec![100.0, f64::NAN], vec![true, true]).is_err());

        let curve = Curve {
            freq: Array1::from_vec(freq),
            spl: Array1::from_vec(vec![80.0, 81.0, 82.0, 83.0, 84.0, 85.0]),
            coherence: Some(Array1::from_vec(vec![0.9; 6])),
            ..Default::default()
        };
        let target_grid = Array1::from_vec(target);
        let (resampled_curve, resampled_mask) =
            resample_curve_with_coverage(&curve, &mask, &target_grid).unwrap();
        assert_eq!(resampled_curve.spl.len(), 5);
        assert_eq!(resampled_mask.valid, vec![true, false, false, true, false]);
        assert!(resampled_curve.coherence.is_some());
        let invalid_grid = Array1::from_vec(vec![400.0, 100.0]);
        assert!(resample_curve_with_coverage(&curve, &mask, &invalid_grid).is_err());
        let malformed = CoverageMask {
            freq: vec![400.0, 100.0],
            valid: vec![true, true],
        };
        assert!(malformed.resample(&[200.0]).is_err());
        assert!(resample_curve_with_coverage(&curve, &malformed, &target_grid).is_err());
    }

    #[test]
    fn measurement_rejected_take_reason_survives_handoff() {
        let mut clipped = take("t1", "L", "seat-1", phased_curve(90.0, 0.0), Some("ref-a"));
        clipped.decision = TakeDecision::Rejected {
            reason: "clipping detected in sweep 3".into(),
        };
        let mut ignored = take("t2", "L", "seat-2", phased_curve(70.0, 0.0), Some("ref-a"));
        ignored.weight = 0.0;
        ignored.decision = TakeDecision::ZeroWeight {
            reason: "seat excluded from training set".into(),
        };
        let matrix = TakeMatrix {
            takes: vec![
                take("t0", "L", "seat-0", phased_curve(80.0, 0.0), Some("ref-a")),
                clipped,
                ignored,
            ],
            average_kind: AverageKind::Power,
        };
        // Bad takes do not move the average.
        let average = matrix.average(None).unwrap();
        assert!((average.spl[0] - 80.0).abs() < 1e-9);

        let report = matrix.completeness_report(&[MatrixKey {
            source_id: "L".into(),
            seat_id: "seat-0".into(),
        }]);
        assert_eq!(report.rejected.len(), 2);
        assert!(
            report
                .rejected
                .iter()
                .any(|rejected| rejected.reason.contains("clipping"))
        );

        // Reasons survive a serialized handoff to workflow/CLI.
        let json = serde_json::to_value(&matrix).unwrap();
        let loaded: TakeMatrix = serde_json::from_value(json.clone()).unwrap();
        assert_eq!(serde_json::to_value(&loaded).unwrap(), json);
        assert_eq!(
            loaded.takes[1].decision.reason(),
            Some("clipping detected in sweep 3")
        );
        assert_eq!(
            loaded.takes[2].decision.reason(),
            Some("seat excluded from training set")
        );
    }
}
