//! Cumulative pruning audit and source-summation checks (E3).
//!
//! Reuses the committed [`ConditionEvaluator`](crate::eq::audibility_veto::conditions::ConditionEvaluator)
//! and the frozen full-chain (F0) reference: every accepted removal is
//! recomputed against F0, and the walk stops at the first budget failure.
//! All measurements, programmes, and levels stay in the condition set,
//! including seats with zero optimizer weight (F08/F09). Rollback restores
//! coefficients by stable pre-removal index and records the reverted
//! proposal. FIR and hybrid branches require their own realization
//! evidence (an IIR-only approximation never judges them), and isolated
//! versus shared-bass inputs stay distinct: redirected-bass and LFE gain
//! conventions are never merged. Whole-graph pruning belongs to workflow
//! reconciliation; kernel success alone is not final-chain acceptance.

// Rust guideline compliant 2026-02-21

use crate::eq::audibility_veto::conditions::{VetoCondition, VetoConditionSet};
use crate::eq::audibility_veto::{AdjudicationConfig, RemovedFilter, VetoAdjudication};
use math_audio_iir_fir::Biquad;
use ndarray::Array1;
use roomeq_model::{BudgetAggregation, FilterVetoVerdict};

/// Whether a conditioned input is an isolated logical input or part of an
/// explicitly correlated shared-bass group.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum BassInputKind {
    /// Independently corrected logical input.
    #[default]
    Isolated,
    /// Member of an explicitly correlated shared-bass group.
    SharedBass,
}

/// One declared seat/programme/level condition for cumulative pruning.
#[derive(Debug, Clone)]
pub struct SeatCondition {
    /// Stable identifier binding evidence to the declared condition set.
    pub id: String,
    /// Seat plus programme magnitude spectrum in dB on the adjudication
    /// grid: the uncorrected background.
    pub background_db: Vec<f64>,
    /// Nominal full-chain playback level, not a physical SPL calibration.
    pub listening_phon: f64,
    /// Optimizer training weight. Zero-weight seats remain in the condition
    /// set: a guarded seat can still reject a candidate (F08).
    pub optimizer_weight: f64,
    /// Isolated logical input or shared-bass group member.
    pub bass: BassInputKind,
}

/// Cumulative-pruning audit failure: malformed inputs. Missing, duplicate,
/// or invalid condition evidence inside adjudication retains all filters
/// with an insufficient-evidence record instead of erroring.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AuditError(pub String);

impl std::fmt::Display for AuditError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "pruning audit failed: {}", self.0)
    }
}

impl std::error::Error for AuditError {}

/// Declared condition identifiers for a seat set, in order.
///
/// Every seat — including zero-weight guard seats — contributes exactly one
/// condition; optimizer weight never excludes a measurement.
pub fn declared_condition_ids(seats: &[SeatCondition]) -> Vec<String> {
    seats.iter().map(|seat| seat.id.clone()).collect()
}

/// Audit cumulative removals against every declared condition and frozen F0.
///
/// Builds one condition per seat/programme/level (zero-weight seats
/// included), freezes the full-chain composite, then adjudicates each
/// accepted removal with recomputation after every step. Returns the
/// adjudication with kept filters, stable-index removals, and the F0
/// reference identity for rollback and reporting.
pub fn audit_cumulative_pruning(
    filters: Vec<Biquad>,
    verdicts: &mut [FilterVetoVerdict],
    freqs: &Array1<f64>,
    seats: &[SeatCondition],
    config: &AdjudicationConfig,
    aggregation: BudgetAggregation,
) -> Result<VetoAdjudication, AuditError> {
    let fail = |message: String| AuditError(message);
    if seats.is_empty() {
        return Err(fail(String::from("at least one condition is required")));
    }
    let mut seen = std::collections::BTreeSet::new();
    for seat in seats {
        if seat.id.trim().is_empty() {
            return Err(fail(String::from("condition ID must not be empty")));
        }
        if !seen.insert(seat.id.clone()) {
            return Err(fail(format!("duplicate condition ID '{}'", seat.id)));
        }
        if seat.background_db.len() != freqs.len() {
            return Err(fail(format!(
                "condition '{}' background length {} does not match grid {}",
                seat.id,
                seat.background_db.len(),
                freqs.len()
            )));
        }
        if !seat.listening_phon.is_finite() || seat.listening_phon <= 0.0 {
            return Err(fail(format!(
                "condition '{}' needs a finite positive nominal level",
                seat.id
            )));
        }
        if !seat.optimizer_weight.is_finite() || seat.optimizer_weight < 0.0 {
            return Err(fail(format!(
                "condition '{}' needs a finite nonnegative optimizer weight",
                seat.id
            )));
        }
    }
    let conditions: Vec<VetoCondition> = seats
        .iter()
        .map(|seat| VetoCondition {
            id: seat.id.clone(),
            background_db: Array1::from(seat.background_db.clone()),
            listening_phon: seat.listening_phon,
        })
        .collect();
    let declared: Vec<String> = declared_condition_ids(seats);
    let set = VetoConditionSet {
        declared_ids: &declared,
        conditions: &conditions,
        aggregation,
    };
    Ok(
        crate::eq::audibility_veto::adjudicate_veto_removals_for_conditions(
            filters, verdicts, freqs, config, &set,
        ),
    )
}

/// Stored record of a rollback: the F0 reference plus everything needed to
/// restore deletions and cite them in reports.
#[derive(Debug, Clone, PartialEq)]
pub struct RollbackRecord {
    /// Frozen-full-chain reference identity the rollback restores.
    pub f0_reference_id: String,
    /// Stable pre-removal indices restored, in ascending order.
    pub restored_indices: Vec<usize>,
    /// Number of restored filters.
    pub restored_count: usize,
}

/// Roll back an adjudicated removal: re-insert removed filters at their
/// stable pre-removal indices so the composite reproduces F0 exactly, and
/// record the reverted proposal for the ledger history.
pub fn rollback_restore(
    kept: &[Biquad],
    removed: &[RemovedFilter],
    expected_total: usize,
    f0_reference_id: impl Into<String>,
) -> Result<(Vec<Biquad>, RollbackRecord), AuditError> {
    let fail = |message: String| AuditError(message);
    if kept.len() + removed.len() != expected_total {
        return Err(fail(format!(
            "kept ({}) plus removed ({}) does not match expected total {expected_total}",
            kept.len(),
            removed.len()
        )));
    }
    let mut ordered: Vec<&RemovedFilter> = removed.iter().collect();
    ordered.sort_by_key(|entry| entry.index);
    let mut seen = std::collections::BTreeSet::new();
    for entry in &ordered {
        if entry.index >= expected_total || !seen.insert(entry.index) {
            return Err(fail(format!(
                "removed index {} is out of range or duplicated",
                entry.index
            )));
        }
    }
    let removed_at: std::collections::BTreeMap<usize, &Biquad> = ordered
        .iter()
        .map(|entry| (entry.index, &entry.filter))
        .collect();
    let mut kept_queue = kept.iter();
    let mut full = Vec::with_capacity(expected_total);
    for index in 0..expected_total {
        if let Some(filter) = removed_at.get(&index) {
            full.push((*filter).clone());
        } else {
            full.push(
                kept_queue
                    .next()
                    .cloned()
                    .ok_or_else(|| fail(String::from("kept filters are misaligned")))?,
            );
        }
    }
    if kept_queue.next().is_some() {
        return Err(fail(String::from("kept filters are misaligned")));
    }
    let restored_indices: Vec<usize> = ordered.iter().map(|entry| entry.index).collect();
    let restored_count = restored_indices.len();
    Ok((
        full,
        RollbackRecord {
            f0_reference_id: f0_reference_id.into(),
            restored_indices,
            restored_count,
        },
    ))
}

/// Realization branch of a correction path.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum RealizationKind {
    /// Minimum-phase IIR path.
    #[default]
    Iir,
    /// Convolution (FIR) path.
    Fir,
    /// Combined IIR plus convolution path.
    Hybrid,
}

/// Missing realization evidence: retain the candidate/reference safely.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RealizationGap {
    /// Branch lacking evidence.
    pub kind: RealizationKind,
    /// What is missing, e.g. `"convolution_ir"` or `"measured_phase"`.
    pub missing: &'static str,
}

/// Require branch-specific realization evidence before judging a path.
///
/// FIR and hybrid branches cannot be judged through an IIR-only
/// approximation: without the convolution IR (and, for hybrid
/// excess-phase content, measured phase) the candidate is retained.
pub fn require_realization_evidence(
    kind: RealizationKind,
    has_convolution_ir: bool,
    has_measured_phase: bool,
) -> Result<(), RealizationGap> {
    match kind {
        RealizationKind::Iir => Ok(()),
        RealizationKind::Fir if has_convolution_ir => Ok(()),
        RealizationKind::Fir => Err(RealizationGap {
            kind,
            missing: "convolution_ir",
        }),
        RealizationKind::Hybrid if has_convolution_ir && has_measured_phase => Ok(()),
        RealizationKind::Hybrid if has_convolution_ir => Err(RealizationGap {
            kind,
            missing: "measured_phase",
        }),
        RealizationKind::Hybrid => Err(RealizationGap {
            kind,
            missing: "convolution_ir",
        }),
    }
}

/// Ideal coherent summation gain of `equal_sources` equal in-phase sources
/// in dB: 20 log10(n) (F03).
pub fn coherent_sum_gain_db(equal_sources: u32) -> f64 {
    f64::from(equal_sources).log10() * 20.0
}

/// Outcome of a shared-bass summation check.
#[derive(Debug, Clone, PartialEq)]
pub enum SummationVerdict {
    /// Measured sum matches the coherent prediction within tolerance.
    Consistent {
        /// Measured minus expected sum in dB.
        delta_db: f64,
    },
    /// Measured sum falls short: cancellation flagged, never clamped into
    /// a good score.
    CancellationFlagged {
        /// Expected minus measured sum in dB.
        deficit_db: f64,
    },
}

/// Check explicitly correlated shared-bass inputs against the coherent
/// summation prediction. Opposite-polarity playback flags cancellation.
pub fn check_shared_bass_summation(
    expected_sum_db: f64,
    measured_sum_db: f64,
    tolerance_db: f64,
) -> Result<SummationVerdict, AuditError> {
    if !expected_sum_db.is_finite() || !measured_sum_db.is_finite() {
        return Err(AuditError(String::from("summation levels must be finite")));
    }
    if !tolerance_db.is_finite() || tolerance_db < 0.0 {
        return Err(AuditError(String::from(
            "summation tolerance must be finite and nonnegative",
        )));
    }
    let delta = measured_sum_db - expected_sum_db;
    if delta >= -tolerance_db {
        Ok(SummationVerdict::Consistent { delta_db: delta })
    } else {
        Ok(SummationVerdict::CancellationFlagged { deficit_db: -delta })
    }
}

/// Bass gain convention: redirected bass and LFE gains are never merged.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum BassGainConvention {
    /// Low-frequency-effects channel gain.
    #[default]
    Lfe,
    /// Redirected (bass-managed) gain.
    Redirected,
}

/// Combine two bass gains only when their conventions match. Mixing LFE and
/// redirected conventions is rejected instead of silently summed.
pub fn merge_bass_gains(
    first_db: f64,
    first_convention: BassGainConvention,
    second_db: f64,
    second_convention: BassGainConvention,
) -> Result<f64, AuditError> {
    if !first_db.is_finite() || !second_db.is_finite() {
        return Err(AuditError(String::from("bass gains must be finite")));
    }
    if first_convention != second_convention {
        return Err(AuditError(format!(
            "refusing to merge {first_convention:?} and {second_convention:?} bass gains"
        )));
    }
    Ok(first_db + second_db)
}

#[cfg(test)]
mod pruning_audit_tests {
    use super::*;
    use math_audio_iir_fir::BiquadFilterType;
    use roomeq_model::{AssessmentRecord, VetoDecision, VetoReason};

    const SAMPLE_RATE: f64 = 48_000.0;

    fn grid() -> Array1<f64> {
        Array1::from(
            (0..200)
                .map(|i| 20.0 * 1000.0_f64.powf(i as f64 / 199.0))
                .collect::<Vec<_>>(),
        )
    }

    fn peak(gain_db: f64, freq_hz: f64, q: f64) -> Biquad {
        Biquad::new(BiquadFilterType::Peak, freq_hz, SAMPLE_RATE, q, gain_db)
    }

    fn remove_verdict(index: usize, filter: &Biquad) -> FilterVetoVerdict {
        FilterVetoVerdict {
            index,
            center_hz: filter.freq,
            q: filter.q,
            gain_db: filter.db_gain,
            peak_delta_db: filter.db_gain.abs(),
            affected_erb_width: 1.0,
            loudness_delta_sones: 0.05,
            decision: VetoDecision::Remove,
            reason: VetoReason::Audible,
            enforced: false,
            acceptance: AssessmentRecord::default(),
        }
    }

    fn keep_verdict(index: usize, filter: &Biquad) -> FilterVetoVerdict {
        let mut verdict = remove_verdict(index, filter);
        verdict.decision = VetoDecision::Keep;
        verdict.reason = VetoReason::Audible;
        verdict
    }

    fn generous_config() -> AdjudicationConfig {
        AdjudicationConfig {
            listening_phon: 75.0,
            per_step_quantum_sones: 1e9,
            cumulative_cap_sones: None,
            local_deviation_cap_db: 1e9,
            enforce: true,
            model_version: String::from("test"),
        }
    }

    fn flat_seat(id: &str, weight: f64, grid_len: usize) -> SeatCondition {
        SeatCondition {
            id: String::from(id),
            background_db: vec![0.0; grid_len],
            listening_phon: 75.0,
            optimizer_weight: weight,
            bass: BassInputKind::Isolated,
        }
    }

    fn composite(filters: &[Biquad], freqs: &Array1<f64>) -> Array1<f64> {
        filters
            .iter()
            .fold(Array1::zeros(freqs.len()), |mut sum, filter| {
                sum += &filter.np_log_result(freqs);
                sum
            })
    }

    #[test]
    fn engine_cumulative_removals_compare_frozen_reference() {
        // F09: two individually small removals with a harmful cumulative
        // effect. The first run (open budget) measures the frozen-reference
        // distance; the second run caps the cumulative budget at half that
        // distance, so at least one filter is retained.
        let freqs = grid();
        let filters = vec![peak(4.0, 200.0, 1.0), peak(-3.0, 3000.0, 2.0)];
        let seats = vec![flat_seat("train", 1.0, freqs.len())];
        let mut verdicts: Vec<FilterVetoVerdict> = filters
            .iter()
            .enumerate()
            .map(|(index, filter)| remove_verdict(index, filter))
            .collect();
        let open = audit_cumulative_pruning(
            filters.clone(),
            &mut verdicts,
            &freqs,
            &seats,
            &generous_config(),
            BudgetAggregation::Sum,
        )
        .expect("valid audit");
        assert_eq!(
            open.removed.len(),
            2,
            "open budget removes both nominations"
        );
        assert!(open.cumulative_loudness_delta_sones > 0.0);
        assert!(!open.f0_reference_id.is_empty());

        let cap = open.cumulative_loudness_delta_sones / 2.0;
        let mut tight = generous_config();
        tight.cumulative_cap_sones = Some(cap);
        let mut verdicts2: Vec<FilterVetoVerdict> = filters
            .iter()
            .enumerate()
            .map(|(index, filter)| remove_verdict(index, filter))
            .collect();
        let capped = audit_cumulative_pruning(
            filters.clone(),
            &mut verdicts2,
            &freqs,
            &seats,
            &tight,
            BudgetAggregation::Sum,
        )
        .expect("valid audit");
        assert!(
            capped.kept.len() > open.kept.len(),
            "cumulative budget must retain at least one filter"
        );
        assert!(capped.cumulative_loudness_delta_sones <= cap);
        // Same frozen reference across both runs.
        assert_eq!(capped.f0_reference_id, open.f0_reference_id);
    }

    #[test]
    fn engine_zero_weight_seat_still_guarded() {
        // F08: a zero-weight training seat stays in the condition set and
        // can reject a candidate the training seat alone would accept.
        let freqs = grid();
        let filters = vec![peak(4.0, 200.0, 1.0)];
        let training = vec![flat_seat("train", 1.0, freqs.len())];
        let mut verdicts: Vec<FilterVetoVerdict> = filters
            .iter()
            .enumerate()
            .map(|(index, filter)| remove_verdict(index, filter))
            .collect();
        let alone = audit_cumulative_pruning(
            filters.clone(),
            &mut verdicts,
            &freqs,
            &training,
            &generous_config(),
            BudgetAggregation::Sum,
        )
        .expect("valid audit");
        assert_eq!(alone.removed.len(), 1);
        let accepted_total = alone.cumulative_loudness_delta_sones;
        assert!(accepted_total > 0.0);

        // Guarded seat with zero optimizer weight but a strongly different
        // background operating point.
        let mut guarded_background = vec![0.0; freqs.len()];
        for (index, frequency) in freqs.iter().enumerate() {
            if *frequency < 150.0 {
                guarded_background[index] = 12.0;
            }
        }
        let seats = vec![
            flat_seat("train", 1.0, freqs.len()),
            SeatCondition {
                id: String::from("held-out"),
                background_db: guarded_background,
                listening_phon: 75.0,
                optimizer_weight: 0.0,
                bass: BassInputKind::Isolated,
            },
        ];
        assert_eq!(
            declared_condition_ids(&seats),
            vec![String::from("train"), String::from("held-out")],
            "zero-weight seats stay declared"
        );
        let mut tight = generous_config();
        tight.cumulative_cap_sones = Some(accepted_total);
        let mut verdicts2: Vec<FilterVetoVerdict> = filters
            .iter()
            .enumerate()
            .map(|(index, filter)| remove_verdict(index, filter))
            .collect();
        let guarded = audit_cumulative_pruning(
            filters.clone(),
            &mut verdicts2,
            &freqs,
            &seats,
            &tight,
            BudgetAggregation::Sum,
        )
        .expect("valid audit");
        assert_eq!(
            guarded.kept.len(),
            1,
            "held-out seat distance must exceed the training-only budget"
        );
        assert!(guarded.removed.is_empty());
    }

    #[test]
    fn engine_opposed_shared_bass_detected() {
        // F03: two equal coherent sources sum to 20 log10(2); opposite
        // polarity flags cancellation instead of a good score.
        let expected = coherent_sum_gain_db(2);
        assert!((expected - 6.020599913279624).abs() < 1e-9);
        assert_eq!(coherent_sum_gain_db(1), 0.0);
        let consistent =
            check_shared_bass_summation(expected, expected - 0.1, 0.5).expect("valid check");
        assert!(matches!(consistent, SummationVerdict::Consistent { .. }));
        let opposed = check_shared_bass_summation(expected, -40.0, 0.5).expect("valid check");
        match opposed {
            SummationVerdict::CancellationFlagged { deficit_db } => {
                assert!(deficit_db > 40.0);
            }
            SummationVerdict::Consistent { .. } => {
                panic!("opposed shared bass must flag cancellation, never consistency")
            }
        }
        // Shared conventions combine; LFE and redirected gains never merge.
        assert_eq!(
            merge_bass_gains(3.0, BassGainConvention::Lfe, 2.0, BassGainConvention::Lfe)
                .expect("same convention"),
            5.0
        );
        assert!(
            merge_bass_gains(
                3.0,
                BassGainConvention::Lfe,
                2.0,
                BassGainConvention::Redirected
            )
            .is_err()
        );
    }

    #[test]
    fn engine_pruning_rollback_restores_coefficients_and_history() {
        // Accepted removal followed by rollback: coefficients restored by
        // stable index reproduce F0 exactly, with the reverted proposal in
        // history.
        let freqs = grid();
        let filters = vec![peak(4.0, 200.0, 1.0), peak(-3.0, 3000.0, 2.0)];
        let f0 = composite(&filters, &freqs);
        let seats = vec![flat_seat("train", 1.0, freqs.len())];
        let verdicts: Vec<FilterVetoVerdict> = filters
            .iter()
            .enumerate()
            .map(|(index, filter)| {
                if index == 1 {
                    remove_verdict(index, filter)
                } else {
                    keep_verdict(index, filter)
                }
            })
            .collect();
        let mut verdicts = verdicts;
        let adjudication = audit_cumulative_pruning(
            filters.clone(),
            &mut verdicts,
            &freqs,
            &seats,
            &generous_config(),
            BudgetAggregation::Sum,
        )
        .expect("valid audit");
        assert_eq!(adjudication.removed.len(), 1);
        assert_eq!(adjudication.removed[0].index, 1);

        let (restored, history) = rollback_restore(
            &adjudication.kept,
            &adjudication.removed,
            filters.len(),
            adjudication.f0_reference_id.clone(),
        )
        .expect("valid rollback");
        assert_eq!(restored.len(), filters.len());
        let recomposed = composite(&restored, &freqs);
        let worst: f64 = recomposed
            .iter()
            .zip(f0.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max);
        assert_eq!(
            worst, 0.0,
            "rollback must reproduce the frozen chain exactly"
        );
        assert_eq!(history.restored_indices, vec![1]);
        assert_eq!(history.restored_count, 1);
        assert_eq!(history.f0_reference_id, adjudication.f0_reference_id);
        // Misaligned inputs fail instead of silently reshuffling.
        assert!(rollback_restore(&adjudication.kept, &adjudication.removed, 5, "f0").is_err());
    }

    #[test]
    fn engine_fir_hybrid_requires_realization_evidence() {
        // FIR/hybrid branches need their own realization evidence; an
        // IIR-only approximation never judges them.
        assert!(require_realization_evidence(RealizationKind::Iir, false, false).is_ok());
        assert_eq!(
            require_realization_evidence(RealizationKind::Fir, false, true),
            Err(RealizationGap {
                kind: RealizationKind::Fir,
                missing: "convolution_ir",
            })
        );
        assert!(require_realization_evidence(RealizationKind::Fir, true, false).is_ok());
        assert_eq!(
            require_realization_evidence(RealizationKind::Hybrid, true, false),
            Err(RealizationGap {
                kind: RealizationKind::Hybrid,
                missing: "measured_phase",
            })
        );
        assert!(require_realization_evidence(RealizationKind::Hybrid, true, true).is_ok());
    }
}
