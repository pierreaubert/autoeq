//! Staged validation corpora with deterministic holdout splits.
//!
//! Stage 2 runs the same comparison across four stages — seeded
//! measurement corpora, a small intake corpus, targeted synthetic cases,
//! and full room corpora — holding out seats, programmes, levels, and
//! rooms per stage. Splits derive deterministically from item ids and a
//! seed, so staged results stay comparable without a split registry.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// Validation stage, in staging order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum ValidationStage {
    /// Seeded measurement corpora (deterministic resampling/noise).
    SeededMeasurement,
    /// Small intake corpus for fast iteration.
    SmallIntake,
    /// Targeted synthetic cases (implementation behavior, not thresholds).
    TargetedSynthetic,
    /// Full room corpora with held-out rooms.
    FullRoom,
}

impl ValidationStage {
    /// Stable staging order index.
    pub fn order(self) -> u8 {
        match self {
            Self::SeededMeasurement => 0,
            Self::SmallIntake => 1,
            Self::TargetedSynthetic => 2,
            Self::FullRoom => 3,
        }
    }
}

/// Case kinds a staged corpus must cover. The inclusion list is staging
/// metadata: missing kinds fail closed at staging time, not at the
/// listening chair.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum ValidationCaseKind {
    /// Repeated measurements of the same setup (repeatability).
    MeasurementRepeatability,
    /// Unprocessed baseline (no-correction control).
    NoCorrection,
    /// Sub/main crossover phase interactions.
    SubMainPhase,
    /// Channel balance and timing preservation checks.
    ChannelBalanceTiming,
    /// Pruning/correction detectability arm.
    CorrectionDetectability,
    /// Correction preference arm (kept separate from detectability).
    CorrectionPreference,
}

/// Case kinds every stage must cover.
const BASE_CASE_KINDS: [ValidationCaseKind; 4] = [
    ValidationCaseKind::MeasurementRepeatability,
    ValidationCaseKind::NoCorrection,
    ValidationCaseKind::SubMainPhase,
    ValidationCaseKind::ChannelBalanceTiming,
];

/// Full required list for the targeted-synthetic stage, which additionally
/// stages the detectability and preference arms separately.
const TARGETED_CASE_KINDS: [ValidationCaseKind; 6] = [
    ValidationCaseKind::MeasurementRepeatability,
    ValidationCaseKind::NoCorrection,
    ValidationCaseKind::SubMainPhase,
    ValidationCaseKind::ChannelBalanceTiming,
    ValidationCaseKind::CorrectionDetectability,
    ValidationCaseKind::CorrectionPreference,
];

/// Required case kinds for `stage`, in coverage order.
pub fn required_case_kinds(stage: ValidationStage) -> &'static [ValidationCaseKind] {
    match stage {
        ValidationStage::TargetedSynthetic => &TARGETED_CASE_KINDS,
        ValidationStage::SeededMeasurement
        | ValidationStage::SmallIntake
        | ValidationStage::FullRoom => &BASE_CASE_KINDS,
    }
}

/// Required kinds absent from `present`, in coverage order. Empty means
/// the stage's inclusion list is fully staged.
pub fn missing_required_kinds(
    stage: ValidationStage,
    present: &[ValidationCaseKind],
) -> Vec<ValidationCaseKind> {
    required_case_kinds(stage)
        .iter()
        .copied()
        .filter(|kind| !present.contains(kind))
        .collect()
}

/// Deterministic train/held-out split of item ids.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct HoldoutSplit {
    /// Split seed (recorded in sidecars).
    pub seed: u64,
    /// Training item ids, in deterministic order.
    pub train: Vec<String>,
    /// Held-out item ids, in deterministic order.
    pub held_out: Vec<String>,
}

/// Split `ids` by hashing each id with `seed`.
///
/// Sorts by `(hash, id)` and holds out the top `held_out_fraction`. At
/// least one item stays held out whenever `ids` has two or more entries,
/// and at least one stays in training — degenerate splits silently
/// removing a whole side are worse than no split. Empty input yields an
/// empty split; `held_out_fraction` outside `[0, 1]` is an error.
pub fn deterministic_split(
    ids: &[String],
    held_out_fraction: f64,
    seed: u64,
) -> Result<HoldoutSplit, String> {
    if !(0.0..=1.0).contains(&held_out_fraction) {
        return Err(String::from("held_out_fraction must lie in [0, 1]"));
    }
    let mut keyed: Vec<([u8; 32], &String)> = ids
        .iter()
        .map(|id| {
            let mut hasher = Sha256::new();
            hasher.update(seed.to_le_bytes());
            hasher.update(id.as_bytes());
            (hasher.finalize().into(), id)
        })
        .collect();
    keyed.sort();
    let ordered: Vec<String> = keyed.into_iter().map(|(_, id)| id.clone()).collect();
    let mut held_count = (ordered.len() as f64 * held_out_fraction).round() as usize;
    if ordered.len() >= 2 {
        held_count = held_count.clamp(1, ordered.len() - 1);
    } else {
        held_count = 0;
    }
    let boundary = ordered.len() - held_count;
    Ok(HoldoutSplit {
        seed,
        train: ordered[..boundary].to_vec(),
        held_out: ordered[boundary..].to_vec(),
    })
}

#[cfg(test)]
mod validation_corpus_tests {
    use super::*;

    fn ids(names: &[&str]) -> Vec<String> {
        names.iter().map(|name| String::from(*name)).collect()
    }

    #[test]
    fn split_is_deterministic_and_complete() {
        let items = ids(&["seat-a", "seat-b", "seat-c", "seat-d", "seat-e"]);
        let first = deterministic_split(&items, 0.4, 11).unwrap();
        let second = deterministic_split(&items, 0.4, 11).unwrap();
        assert_eq!(first, second);
        assert_eq!(first.held_out.len(), 2);
        assert_eq!(first.train.len(), 3);
        let mut all = first.train.clone();
        all.extend(first.held_out.clone());
        all.sort();
        let mut expected = items.clone();
        expected.sort();
        assert_eq!(all, expected);
    }

    #[test]
    fn split_never_degenerates() {
        // Two items always split one and one, whatever the fraction.
        for fraction in [0.0, 0.25, 0.9, 1.0] {
            let split = deterministic_split(&ids(&["a", "b"]), fraction, 3).unwrap();
            assert_eq!(split.held_out.len(), 1, "fraction {fraction}");
            assert_eq!(split.train.len(), 1, "fraction {fraction}");
        }
        // One item cannot split: everything trains, nothing held out.
        let solo = deterministic_split(&ids(&["only"]), 0.5, 3).unwrap();
        assert!(solo.held_out.is_empty());
        assert_eq!(solo.train.len(), 1);
        // Fractions outside [0, 1] are rejected, not clamped.
        assert!(deterministic_split(&ids(&["a", "b"]), 1.5, 3).is_err());
        assert!(deterministic_split(&ids(&["a", "b"]), -0.1, 3).is_err());
    }

    #[test]
    fn required_kinds_cover_the_inclusion_list() {
        use ValidationCaseKind as Kind;
        // Non-targeted stages require the four inclusion-list kinds.
        for stage in [
            ValidationStage::SeededMeasurement,
            ValidationStage::SmallIntake,
            ValidationStage::FullRoom,
        ] {
            assert_eq!(
                required_case_kinds(stage),
                &[
                    Kind::MeasurementRepeatability,
                    Kind::NoCorrection,
                    Kind::SubMainPhase,
                    Kind::ChannelBalanceTiming,
                ]
            );
            assert!(missing_required_kinds(stage, required_case_kinds(stage)).is_empty());
        }
        // Targeted synthetic additionally stages both arms separately.
        let targeted = required_case_kinds(ValidationStage::TargetedSynthetic);
        assert!(targeted.contains(&Kind::CorrectionDetectability));
        assert!(targeted.contains(&Kind::CorrectionPreference));
        // Nothing present: everything missing, in coverage order.
        let missing = missing_required_kinds(ValidationStage::FullRoom, &[]);
        assert_eq!(missing.len(), 4);
        assert_eq!(missing[0], Kind::MeasurementRepeatability);
        // Partial coverage reports exactly the gap.
        let missing = missing_required_kinds(
            ValidationStage::TargetedSynthetic,
            &[
                Kind::MeasurementRepeatability,
                Kind::NoCorrection,
                Kind::SubMainPhase,
                Kind::ChannelBalanceTiming,
                Kind::CorrectionPreference,
            ],
        );
        assert_eq!(missing, vec![Kind::CorrectionDetectability]);
    }

    #[test]
    fn stages_order_stably() {
        let mut stages = [
            ValidationStage::FullRoom,
            ValidationStage::SmallIntake,
            ValidationStage::TargetedSynthetic,
            ValidationStage::SeededMeasurement,
        ];
        stages.sort_by_key(|stage| stage.order());
        assert_eq!(
            stages,
            [
                ValidationStage::SeededMeasurement,
                ValidationStage::SmallIntake,
                ValidationStage::TargetedSynthetic,
                ValidationStage::FullRoom,
            ]
        );
    }
}
