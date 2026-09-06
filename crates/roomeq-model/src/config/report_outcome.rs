//! Stage 0 contract types for audibility assessment reporting.
//!
//! These types record assessment *outcomes* — they never authorize a
//! change. Prediction (what the model says), enforcement (whether anything
//! was applied), and confidence (how much the evidence supports) travel in
//! separate fields so a report can never conflate "model predicts no
//! difference" with "proven inaudible" or "nothing was changed".
//!
//! All assessment travels advisory until a later stage validates it:
//! [`PruningBudget`] tracking is disabled unless explicitly configured,
//! and [`AssessmentRecord`] defaults to an unassessed state rather than a
//! passing one.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

fn unspecified() -> String {
    String::from("unspecified")
}

fn unknown() -> String {
    String::from("unknown")
}

fn unstated() -> String {
    String::from("unstated")
}

/// Outcome of assessing one proposed simplification or correction.
///
/// These are the only report outcomes a later stage may emit: assessment
/// code maps its internal verdicts onto this vocabulary instead of
/// inventing per-stage reason strings.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ReportOutcome {
    /// The full-chain comparison supports keeping the current processing.
    Keep,
    /// A simplification is nominated but not yet accepted: it needs the
    /// cumulative check against the frozen full chain before anything
    /// is applied.
    CandidateRemoval,
    /// A nominated simplification passed its acceptance checks and was
    /// applied. Compare against the frozen full chain, not the
    /// pre-removal step.
    AcceptedRemoval,
    /// Correction proceeds under explicit limits (gain caps, band
    /// restrictions, seat subsets) because evidence is partial.
    /// Records the limits, not just the decision.
    RiskLimitedCorrection,
    /// Evidence is missing or too weak to decide. Uncertain cases retain
    /// the filter; "unknown" is never evidence of safety or inaudibility.
    #[default]
    InsufficientEvidence,
}

/// How much the available evidence supports an assessment.
///
/// `Unknown` is the default: missing data is reported as unknown, never
/// as confidence.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum AssessmentConfidence {
    /// Independent or held-out evidence supports the outcome.
    High,
    /// Consistent evidence within the calibration set only.
    Moderate,
    /// Weak, contradictory, or single-source evidence.
    Low,
    /// No assessment ran, or inputs were missing. The default.
    #[default]
    Unknown,
}

/// Whether an assessment drove an applied change.
///
/// Separating this from the outcome keeps "evaluated and recorded" visibly
/// distinct from "evaluated and applied".
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum EnforcementState {
    /// Evaluated and recorded; nothing was changed. The default and the
    /// only state the heuristic path may report until a later stage
    /// validates enforcement.
    #[default]
    Advisory,
    /// The outcome was applied to the processing chain.
    Enforced,
    /// The assessment path was disabled or skipped for this item.
    NotEvaluated,
}

/// Model, calibration, and reference behind one assessment.
///
/// Every applied or proposed change reports these four so a reader can
/// reproduce the comparison. Sentinel defaults (`"unspecified"` /
/// `"unknown"`) are honest placeholders, not claims.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
pub struct AssessmentProvenance {
    /// Assessment model name, e.g. `"heuristic-erb-proxy"`.
    #[serde(default = "unspecified")]
    pub model: String,
    /// Model or code version the assessment ran under.
    #[serde(default = "unspecified")]
    pub model_version: String,
    /// Calibration or level assumption, e.g. `"nominal-75phon"` or
    /// `"unknown-spl"`. A nominal loudness-level setting is not an SPL
    /// calibration.
    #[serde(default = "unknown")]
    pub calibration: String,
    /// Comparison reference identity: the frozen full-chain id for pruning
    /// comparisons, or the declared desired reference for quality
    /// comparisons.
    #[serde(default = "unspecified")]
    pub reference: String,
}

impl Default for AssessmentProvenance {
    fn default() -> Self {
        Self {
            model: unspecified(),
            model_version: unspecified(),
            calibration: unknown(),
            reference: unspecified(),
        }
    }
}

/// One threshold applied during an assessment, with units.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
pub struct AppliedThreshold {
    /// Threshold name, e.g. `"jnd_db"`.
    #[serde(default = "unstated")]
    pub name: String,
    /// Threshold value in `unit`.
    #[serde(default)]
    pub value: f64,
    /// Unit string, e.g. `"db"`, `"erb"`, `"sones-experimental-proxy"`.
    /// Proxy units are labeled experimental; they are not portable
    /// audibility thresholds.
    #[serde(default = "unspecified")]
    pub unit: String,
}

impl Default for AppliedThreshold {
    fn default() -> Self {
        Self {
            name: unstated(),
            value: 0.0,
            unit: unspecified(),
        }
    }
}

/// One assessment report: outcome, confidence, enforcement, and provenance.
///
/// The default is an unassessed record (`InsufficientEvidence` /
/// `Unknown` / `Advisory`), never a passing one.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
pub struct AssessmentRecord {
    /// The report outcome from [`ReportOutcome`].
    #[serde(default)]
    pub outcome: ReportOutcome,
    /// Evidence strength from [`AssessmentConfidence`].
    #[serde(default)]
    pub confidence: AssessmentConfidence,
    /// Whether anything was applied from [`EnforcementState`].
    #[serde(default)]
    pub enforcement: EnforcementState,
    /// Model, calibration, and reference from [`AssessmentProvenance`].
    #[serde(default)]
    pub provenance: AssessmentProvenance,
    /// Thresholds applied, with units. Empty when none were applied.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub thresholds: Vec<AppliedThreshold>,
    /// Machine-readable reason summary. `"unstated"` when not recorded.
    #[serde(default = "unstated")]
    pub reason: String,
}

impl Default for AssessmentRecord {
    fn default() -> Self {
        Self {
            outcome: ReportOutcome::default(),
            confidence: AssessmentConfidence::default(),
            enforcement: EnforcementState::default(),
            provenance: AssessmentProvenance::default(),
            thresholds: Vec::new(),
            reason: unstated(),
        }
    }
}

/// How per-condition differences aggregate against a cumulative cap.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum BudgetAggregation {
    /// Sum of per-condition differences must stay under the cap.
    #[default]
    Sum,
    /// Worst single-condition difference must stay under the cap.
    Max,
}

/// Cumulative pruning budget over declared conditions.
///
/// Stage 0 resolution of the cumulative-budget open decision: budgets are
/// declared up front, span explicit condition ids (seats, programmes,
/// levels), and are disabled (`None` cap) unless configured. A configured
/// budget is validated for shape here; *enforcement* of the budget arrives
/// with the Stage 1 cumulative checks, so a set budget is currently
/// tracked in reports, not used to accept or reject removals.
///
/// Differences are measured in approximate masked-loudness delta (sones)
/// under the experimental heuristic proxy until Stage 2 validates a
/// replacement distance.
#[derive(Debug, Clone, Default, Serialize, Deserialize, JsonSchema, PartialEq)]
pub struct PruningBudget {
    /// Cumulative cap in approximate masked-loudness delta (sones,
    /// experimental proxy units). `None` (default) disables budget
    /// tracking.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_cumulative_delta: Option<f64>,
    /// How per-condition differences aggregate against the cap.
    #[serde(default)]
    pub aggregation: BudgetAggregation,
    /// Declared condition ids the budget spans (seats, programmes,
    /// levels). Empty means the budget is not bound to conditions yet.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub conditions: Vec<String>,
}

#[cfg(test)]
mod report_outcome_tests {
    use super::*;

    #[test]
    fn outcome_wire_format_is_pinned() {
        // Canonical on-the-wire strings; schema and sidecars depend on them.
        let cases = [
            (ReportOutcome::Keep, "\"keep\""),
            (ReportOutcome::CandidateRemoval, "\"candidate_removal\""),
            (ReportOutcome::AcceptedRemoval, "\"accepted_removal\""),
            (ReportOutcome::RiskLimitedCorrection, "\"risk_limited_correction\""),
            (
                ReportOutcome::InsufficientEvidence,
                "\"insufficient_evidence\"",
            ),
        ];
        for (outcome, expected) in cases {
            let serialized = serde_json::to_string(&outcome).unwrap();
            assert_eq!(serialized, expected, "serialize {outcome:?}");
            let round_tripped: ReportOutcome = serde_json::from_str(&serialized).unwrap();
            assert_eq!(round_tripped, outcome, "round-trip {outcome:?}");
        }
    }

    #[test]
    fn default_record_is_unassessed_not_passing() {
        // Acceptance: a fresh record must never read as a passing verdict.
        let record = AssessmentRecord::default();
        assert_eq!(record.outcome, ReportOutcome::InsufficientEvidence);
        assert_eq!(record.confidence, AssessmentConfidence::Unknown);
        assert_eq!(record.enforcement, EnforcementState::Advisory);
        assert!(record.thresholds.is_empty());
    }

    #[test]
    fn budget_defaults_to_disabled() {
        let budget = PruningBudget::default();
        assert!(budget.max_cumulative_delta.is_none());
        assert!(budget.conditions.is_empty());
    }

    #[test]
    fn partial_json_fills_honest_sentinels() {
        // Missing fields degrade to explicit unknowns, never to claims.
        let provenance: AssessmentProvenance = serde_json::from_value(serde_json::json!({})).unwrap();
        assert_eq!(provenance.model, "unspecified");
        assert_eq!(provenance.calibration, "unknown");
        let record: AssessmentRecord = serde_json::from_value(serde_json::json!({})).unwrap();
        assert_eq!(record.outcome, ReportOutcome::InsufficientEvidence);
        assert_eq!(record.reason, "unstated");
    }

    #[test]
    fn schema_advertises_new_vocabulary() {
        // Guards schema drift for the additive contract (mirrors the
        // seat/search parity test style in validation_rules).
        let outcome_schema =
            serde_json::to_value(schemars::schema_for!(ReportOutcome)).unwrap().to_string();
        for variant in [
            "keep",
            "candidate_removal",
            "accepted_removal",
            "risk_limited_correction",
            "insufficient_evidence",
        ] {
            assert!(outcome_schema.contains(variant), "{outcome_schema}");
        }
        let budget_schema =
            serde_json::to_value(schemars::schema_for!(PruningBudget)).unwrap().to_string();
        assert!(budget_schema.contains("max_cumulative_delta"), "{budget_schema}");
        let record_schema =
            serde_json::to_value(schemars::schema_for!(AssessmentRecord)).unwrap().to_string();
        for field in ["outcome", "confidence", "enforcement", "provenance"] {
            assert!(record_schema.contains(field), "{record_schema}");
        }
    }
}
