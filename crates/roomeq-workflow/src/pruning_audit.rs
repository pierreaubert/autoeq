//! W4 — cumulative pruning audit and listening handoff.
//!
//! This module audits the existing routed-pruning and pruning-QA machinery
//! rather than rebuilding it:
//!
//! - [`audit_pruning_matrix`] runs over matrix rows (processing mode,
//!   filter class, routed variant, conditions) and records ran, stale, or
//!   skipped paths with reasons. Stale paths stay visible; they are never
//!   silently dropped from the report.
//! - [`check_cumulative_pruning`] compares candidate removals against the
//!   frozen full chain over every measurement, programme, and level,
//!   including zero-weight training positions and held-out constraints.
//!   Better mean quality with a worse held-out or zero-weight seat stays
//!   visible and can reject the candidate (F08); individually small
//!   removals with a harmful cumulative effect restore the filters (F09).
//! - [`check_removal_applicability`] keeps filters when a declared
//!   condition or its IR resource is missing: absent evidence never
//!   removes a filter.
//! - [`check_metadata_matches_export`] verifies every accepted removal and
//!   rollback reaches final metadata *and* the serialized/exported chain,
//!   not just an intermediate filter vector.
//! - [`bind_listening_stimuli`] renders V3 stimuli from frozen graphs only:
//!   a stimulus bound to a stale graph is rejected, never rebound quietly.
//! - [`approve_listening_bundle`] never approves a listening-validation
//!   bundle when final playback validation rejected the graph; diagnostic
//!   artifacts stay advisory-labelled until G7.

use roomeq_model::{BudgetAggregation, VetoAdjudicationReport};
use serde::{Deserialize, Serialize};

use crate::verification::PlaybackStatus;

/// Pruning matrix audit version pinned by this workflow lane.
pub const PRUNING_AUDIT_VERSION: &str = "workflow-pruning-audit-v1";

/// Processing-mode axis of the pruning matrix.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum PruningMode {
    Ordinary,
    Adaptive,
    Pareto,
    Refinement,
}

/// Filter-class axis of the pruning matrix.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum PruningFilterClass {
    Iir,
    Fir,
    Hybrid,
}

/// Routing axis of the pruning matrix.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum PruningRoutedVariant {
    Unrouted,
    Routed,
}

/// Outcome of one matrix row.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum MatrixRowOutcome {
    /// Row ran and its evidence is current.
    Ran,
    /// Row ran but its evidence predates the current chain: re-run needed.
    Stale { reason: String },
    /// Row did not run, with its reason.
    Skipped { reason: String },
}

/// One audited matrix row: mode x filter class x routed variant x conditions.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PruningMatrixRow {
    pub mode: PruningMode,
    pub filter_class: PruningFilterClass,
    pub routed_variant: PruningRoutedVariant,
    pub condition_ids: Vec<String>,
    pub outcome: MatrixRowOutcome,
}

/// Matrix audit report: stale and skipped paths stay listed with reasons.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MatrixAuditReport {
    pub audit_version: String,
    pub ran: usize,
    pub stale: Vec<String>,
    pub skipped: Vec<String>,
}

fn describe_row(row: &PruningMatrixRow) -> String {
    format!(
        "{:?}/{:?}/{:?}:{}",
        row.mode,
        row.filter_class,
        row.routed_variant,
        row.condition_ids.join("+")
    )
}

/// Audit the pruning matrix: count ran rows, list stale and skipped paths.
///
/// A row with no conditions is stale (nothing was actually exercised); rows
/// that report stale or skipped keep their reasons in the report.
pub fn audit_pruning_matrix(rows: &[PruningMatrixRow]) -> MatrixAuditReport {
    let mut report = MatrixAuditReport {
        audit_version: PRUNING_AUDIT_VERSION.to_string(),
        ran: 0,
        stale: Vec::new(),
        skipped: Vec::new(),
    };
    for row in rows {
        if row.condition_ids.is_empty() {
            report
                .stale
                .push(format!("{}: row declares no conditions", describe_row(row)));
            continue;
        }
        match &row.outcome {
            MatrixRowOutcome::Ran => report.ran += 1,
            MatrixRowOutcome::Stale { reason } => {
                report
                    .stale
                    .push(format!("{}: {reason}", describe_row(row)));
            }
            MatrixRowOutcome::Skipped { reason } => {
                report
                    .skipped
                    .push(format!("{}: {reason}", describe_row(row)));
            }
        }
    }
    report
}

/// One condition score: frozen full chain vs pruned candidate, in dB.
///
/// `delta_db` is pruned minus full (positive means the removal hurt).
/// Zero-weight training seats and held-out seats are conditions like any
/// other: the policy decides their weight, never their visibility.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ConditionScore {
    pub condition_id: String,
    /// Pruned-minus-full difference in dB (positive = regression).
    pub delta_db: f64,
    /// Policy weight; zero-weight seats are evaluated, not hidden.
    pub weight: f64,
    /// True for held-out generalization seats.
    pub held_out: bool,
}

/// Cumulative pruning verdict against the frozen full chain.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum CumulativeVerdict {
    /// Every condition, including zero-weight and held-out seats, is within
    /// budget. Carries the aggregate for the ledger.
    RetainPruned { aggregate_db: f64 },
    /// The candidate regresses: restore the frozen full-chain filters.
    /// Carries the offending condition and its delta.
    RestoreFullChain { condition_id: String, delta_db: f64 },
}

impl CumulativeVerdict {
    pub fn restores_filters(&self) -> bool {
        matches!(self, CumulativeVerdict::RestoreFullChain { .. })
    }
}

/// Check candidate removals against the frozen full chain (F08/F09).
///
/// Every supplied condition is evaluated: a better weighted mean with a
/// worse held-out or zero-weight seat stays visible and rejects the
/// candidate when it exceeds `max_condition_regression_db` (F08). The
/// aggregate (sum or max per `aggregation`) must stay under
/// `max_cumulative_delta_db`, so individually small removals with a harmful
/// cumulative effect restore the filters (F09). Nonfinite deltas are a
/// missing-evidence error, never a pass.
pub fn check_cumulative_pruning(
    scores: &[ConditionScore],
    aggregation: BudgetAggregation,
    max_cumulative_delta_db: f64,
    max_condition_regression_db: f64,
) -> Result<CumulativeVerdict, String> {
    if scores.is_empty() {
        return Err(String::from(
            "cumulative pruning needs at least one condition; refusing to approve an empty check",
        ));
    }
    if !max_cumulative_delta_db.is_finite()
        || !max_condition_regression_db.is_finite()
        || max_cumulative_delta_db < 0.0
        || max_condition_regression_db < 0.0
    {
        return Err(String::from(
            "pruning budgets must be finite and non-negative",
        ));
    }
    // F08: no guarded seat regression may hide behind a better mean — not a
    // zero-weight training seat, not a held-out seat, not any seat.
    for score in scores {
        if !score.delta_db.is_finite() || !score.weight.is_finite() || score.weight < 0.0 {
            return Err(format!(
                "condition '{}' has nonfinite delta or invalid weight; retaining filters",
                score.condition_id
            ));
        }
        if score.delta_db > max_condition_regression_db {
            return Ok(CumulativeVerdict::RestoreFullChain {
                condition_id: score.condition_id.clone(),
                delta_db: score.delta_db,
            });
        }
    }
    let aggregate_db = match aggregation {
        BudgetAggregation::Sum => scores
            .iter()
            .map(|score| (score.delta_db * score.weight).max(0.0))
            .sum(),
        BudgetAggregation::Max => scores
            .iter()
            .map(|score| score.delta_db)
            .fold(0.0_f64, f64::max),
    };
    // F09: individually small removals can still fail the cumulative budget.
    if aggregate_db > max_cumulative_delta_db {
        return Ok(CumulativeVerdict::RestoreFullChain {
            condition_id: String::from("cumulative_budget"),
            delta_db: aggregate_db,
        });
    }
    Ok(CumulativeVerdict::RetainPruned { aggregate_db })
}

/// Whether one declared removal may proceed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RemovalApplicability {
    /// All declared conditions and IR resources are present.
    Applies,
    /// Missing condition or IR resource: filters are retained, with reason.
    Retain { reason: String },
}

/// A missing condition or IR resource must not remove filters.
///
/// Every declared condition needs matching evidence, and every filter under
/// test needs its IR resource. Anything missing retains the filters with an
/// explicit reason — removal is never the default.
pub fn check_removal_applicability(
    removal_id: &str,
    declared_conditions: &[String],
    evidenced_conditions: &[String],
    filters_have_ir: bool,
) -> RemovalApplicability {
    for declared in declared_conditions {
        if !evidenced_conditions.iter().any(|seen| seen == declared) {
            return RemovalApplicability::Retain {
                reason: format!(
                    "removal '{removal_id}' declares condition '{declared}' with no evidence; retaining filters"
                ),
            };
        }
    }
    if !filters_have_ir {
        return RemovalApplicability::Retain {
            reason: format!(
                "removal '{removal_id}' has no IR resource for the filters under test; retaining filters"
            ),
        };
    }
    RemovalApplicability::Applies
}

/// Verify accepted removals and rollbacks reach metadata and export.
///
/// `adjudication` is the final per-channel adjudication;
/// `exported_filter_count` is the EQ section count in the serialized chain;
/// `fitted_filter_count` is the pre-pruning fitted count. Removed filters
/// must be absent from the export; retained filters must be present. An
/// advisory (report-only) row must export the retained EQ unchanged.
pub fn check_metadata_matches_export(
    adjudication: &VetoAdjudicationReport,
    fitted_filter_count: usize,
    exported_filter_count: usize,
) -> Result<(), String> {
    let removed = adjudication.removed_filter_indices.len();
    if removed > fitted_filter_count {
        return Err(format!(
            "adjudication removes {removed} filters but only {fitted_filter_count} were fitted"
        ));
    }
    let expected_exported = if adjudication.enforced {
        fitted_filter_count - removed
    } else {
        fitted_filter_count
    };
    if exported_filter_count != expected_exported {
        return Err(format!(
            "exported chain carries {exported_filter_count} EQ sections but metadata implies \
             {expected_exported} (fitted {fitted_filter_count}, removed {removed}, enforced {})",
            adjudication.enforced
        ));
    }
    Ok(())
}

/// Bind V3 listening stimuli to frozen graphs only.
///
/// Each stimulus carries the graph fingerprint it was rendered from. A
/// stimulus bound to any other graph is stale and rejected; it is never
/// rebound quietly to the newly finalized graph.
pub fn bind_listening_stimuli(
    frozen_graph_fingerprint: &str,
    stimuli: &[(String, String)],
) -> Result<Vec<String>, String> {
    if frozen_graph_fingerprint.trim().is_empty() {
        return Err(String::from("frozen graph fingerprint must not be empty"));
    }
    let mut bound = Vec::new();
    for (stimulus_hash, graph_fingerprint) in stimuli {
        if graph_fingerprint != frozen_graph_fingerprint {
            return Err(format!(
                "stimulus '{stimulus_hash}' was rendered from graph '{graph_fingerprint}', \
                 not the frozen graph '{frozen_graph_fingerprint}'; refusing stale stimuli"
            ));
        }
        bound.push(stimulus_hash.clone());
    }
    Ok(bound)
}

/// Listening-bundle approval gate.
///
/// A listening-validation bundle is approved only when final playback
/// validation verified the graph. Rejected playback never yields an
/// approved bundle; diagnostic artifacts stay advisory-labelled (G7 keeps
/// perceptual proxies advisory).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ListeningBundleApproval {
    Approved,
    /// Diagnostic artifacts only, with their advisory label.
    Advisory {
        label: String,
    },
    /// Rejected playback: no approved bundle.
    Rejected {
        reason: String,
    },
}

pub fn approve_listening_bundle(
    playback: PlaybackStatus,
    validated_stimuli: &[String],
) -> ListeningBundleApproval {
    match playback {
        PlaybackStatus::Verified if !validated_stimuli.is_empty() => {
            ListeningBundleApproval::Approved
        }
        PlaybackStatus::Verified => ListeningBundleApproval::Advisory {
            label: String::from("verified graph but no bound stimuli; diagnostics only"),
        },
        PlaybackStatus::SimulatedPass => ListeningBundleApproval::Advisory {
            label: String::from(
                "backend simulation only; perceptual diagnostics stay advisory until G7",
            ),
        },
        PlaybackStatus::Failed => ListeningBundleApproval::Rejected {
            reason: String::from(
                "required playback comparison failed; no approved listening bundle",
            ),
        },
        PlaybackStatus::InsufficientEvidence => ListeningBundleApproval::Rejected {
            reason: String::from(
                "required playback evidence is incomplete; no approved listening bundle",
            ),
        },
        PlaybackStatus::Unassessed => ListeningBundleApproval::Rejected {
            reason: String::from(
                "final playback validation rejected or never assessed the graph; \
                 no approved listening bundle",
            ),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roadmap_correction_listening_rejects_failed_or_incomplete_playback() {
        for status in [PlaybackStatus::Failed, PlaybackStatus::InsufficientEvidence] {
            assert!(matches!(
                approve_listening_bundle(status, &["bound-stimulus".to_owned()]),
                ListeningBundleApproval::Rejected { .. }
            ));
        }
    }

    fn condition(id: &str, delta_db: f64, weight: f64, held_out: bool) -> ConditionScore {
        ConditionScore {
            condition_id: id.to_string(),
            delta_db,
            weight,
            held_out,
        }
    }

    fn adjudication(removed: &[usize], enforced: bool) -> VetoAdjudicationReport {
        VetoAdjudicationReport {
            f0_reference_id: String::from("conditions-v1:test"),
            removed_filter_indices: removed.to_vec(),
            cumulative_loudness_delta_sones: 0.0,
            max_local_deviation_db: 0.0,
            enforced,
        }
    }

    #[test]
    fn workflow_cumulative_pruning_all_conditions_final_graph() {
        // F08: better mean, worse held-out seat. The guarded regression
        // stays visible through the zero-weight training seat and rejects.
        let guarded = vec![
            condition("seat-main", -1.0, 1.0, false),
            condition("seat-zero-weight", 0.0, 0.0, false),
            condition("seat-held-out", 2.0, 1.0, true),
        ];
        let verdict =
            check_cumulative_pruning(&guarded, BudgetAggregation::Sum, 1.0, 0.25).unwrap();
        assert!(verdict.restores_filters());
        assert_eq!(
            verdict,
            CumulativeVerdict::RestoreFullChain {
                condition_id: String::from("seat-held-out"),
                delta_db: 2.0,
            }
        );
        // A zero-weight training regression rejects even when it carries no
        // weight in the aggregate.
        let zero_weight_regression = vec![
            condition("seat-main", -2.0, 1.0, false),
            condition("seat-zero-weight", 1.5, 0.0, false),
        ];
        let verdict =
            check_cumulative_pruning(&zero_weight_regression, BudgetAggregation::Sum, 5.0, 0.25)
                .unwrap();
        assert!(verdict.restores_filters());
        // F09: two individually small removals (0.2 dB each) with a harmful
        // cumulative effect against the frozen full chain.
        let cumulative = vec![
            condition("music-55phon", 0.2, 1.0, false),
            condition("music-85phon", 0.2, 1.0, false),
            condition("flat-55phon", 0.2, 1.0, false),
            condition("flat-85phon", 0.2, 1.0, false),
        ];
        let verdict =
            check_cumulative_pruning(&cumulative, BudgetAggregation::Sum, 0.5, 0.25).unwrap();
        assert!(verdict.restores_filters());
        assert_eq!(
            verdict,
            CumulativeVerdict::RestoreFullChain {
                condition_id: String::from("cumulative_budget"),
                delta_db: 0.8,
            }
        );
        // Max aggregation judges the worst condition instead of the sum.
        let verdict =
            check_cumulative_pruning(&cumulative, BudgetAggregation::Max, 0.5, 0.5).unwrap();
        assert_eq!(
            verdict,
            CumulativeVerdict::RetainPruned { aggregate_db: 0.2 }
        );
        // A clean sweep retains the pruned chain with its aggregate.
        let clean = vec![
            condition("seat-main", -0.5, 1.0, false),
            condition("seat-held-out", 0.1, 1.0, true),
        ];
        let verdict = check_cumulative_pruning(&clean, BudgetAggregation::Sum, 1.0, 0.25).unwrap();
        assert_eq!(
            verdict,
            CumulativeVerdict::RetainPruned { aggregate_db: 0.1 }
        );
        // Empty conditions and nonfinite deltas never approve.
        assert!(check_cumulative_pruning(&[], BudgetAggregation::Sum, 1.0, 0.25).is_err());
        assert!(
            check_cumulative_pruning(
                &[condition("bad", f64::NAN, 1.0, false)],
                BudgetAggregation::Sum,
                1.0,
                0.25
            )
            .is_err()
        );
    }

    #[test]
    fn workflow_missing_condition_retains_filters() {
        assert_eq!(
            check_removal_applicability(
                "removal-0",
                &[String::from("music-85phon")],
                &[String::from("flat-55phon")],
                true
            ),
            RemovalApplicability::Retain {
                reason: String::from(
                    "removal 'removal-0' declares condition 'music-85phon' with no evidence; retaining filters"
                ),
            }
        );
        assert_eq!(
            check_removal_applicability(
                "removal-0",
                &[String::from("music-85phon")],
                &[String::from("music-85phon")],
                false
            ),
            RemovalApplicability::Retain {
                reason: String::from(
                    "removal 'removal-0' has no IR resource for the filters under test; retaining filters"
                ),
            }
        );
        assert_eq!(
            check_removal_applicability(
                "removal-0",
                &[String::from("music-85phon")],
                &[String::from("music-85phon")],
                true
            ),
            RemovalApplicability::Applies
        );
    }

    use crate::optimize_room;
    use ndarray::Array1;
    use roomeq_model::{
        FilterAudibilityConfig, MeasurementSource, MultiMeasurementConfig,
        MultiMeasurementStrategy, OptimizerConfig, ProcessingMode, PruningBudget, RoomConfig,
        SpeakerConfig, default_config_version,
    };
    use std::collections::HashMap;

    fn advisory_pruning_config(measurements: usize) -> RoomConfig {
        let ids: Vec<_> = (0..measurements)
            .map(|index| format!("seat-{index}"))
            .collect();
        let evaluation = serde_json::from_value(serde_json::json!({
            "version": "spectral-v1",
            "measurement_ids": ids,
            "programmes": [
                {"id": "flat", "frequencies_hz": [20, 20000], "spectrum_db": [0, 0]},
                {"id": "music", "frequencies_hz": [20, 20000], "spectrum_db": [0, -9]}
            ],
            "listening_levels_phon": [55, 85]
        }))
        .unwrap();
        let measurement = |seat: usize| {
            let freq = Array1::logspace(10.0, 20.0_f64.log10(), 20_000.0_f64.log10(), 96);
            let spl = freq.mapv(|frequency| {
                80.0 + (6.0 + seat as f64) * (-((frequency / 80.0).log2() / 0.5).powi(2)).exp()
            });
            autoeq_core::Curve {
                freq,
                spl,
                ..Default::default()
            }
        };
        let source = if measurements == 1 {
            MeasurementSource::InMemory(measurement(0))
        } else {
            MeasurementSource::InMemoryMultiple((0..measurements).map(measurement).collect())
        };
        RoomConfig {
            version: default_config_version(),
            system: None,
            speakers: HashMap::from([(String::from("left"), SpeakerConfig::Single(source))]),
            crossovers: None,
            target_curve: None,
            optimizer: OptimizerConfig {
                processing_mode: ProcessingMode::LowLatency,
                algorithm: String::from("autoeq:de"),
                strategy: String::from("lshade"),
                num_filters: 2,
                max_iter: 30,
                population: 6,
                seed: Some(7),
                parallel_threads: Some(1),
                min_freq: 20.0,
                max_freq: 500.0,
                min_db: -0.5,
                max_db: 0.5,
                multi_measurement: (measurements > 1).then_some(MultiMeasurementConfig {
                    strategy: MultiMeasurementStrategy::WeightedSum,
                    weights: Some(vec![1.0, 0.0]),
                    ..Default::default()
                }),
                filter_audibility: Some(FilterAudibilityConfig {
                    report_only: true,
                    allow_enforcement_with_experimental_proxy: false,
                    ..Default::default()
                }),
                pruning_budget: Some(PruningBudget {
                    evaluation: Some(evaluation),
                    aggregation: roomeq_model::BudgetAggregation::Max,
                    ..Default::default()
                }),
                ..Default::default()
            },
            provenance: Default::default(),
            recording_config: None,
            ctc: None,
            cea2034_cache: None,
        }
    }

    #[test]
    fn workflow_final_filter_metadata_agrees_with_export() {
        // Enforced removal: two fitted, one removed, one exported.
        assert!(check_metadata_matches_export(&adjudication(&[1], true), 2, 1).is_ok());
        // Removed EQ must not reappear in the exported chain ...
        assert!(check_metadata_matches_export(&adjudication(&[1], true), 2, 2).is_err());
        // ... and an advisory row must export the retained EQ unchanged.
        assert!(check_metadata_matches_export(&adjudication(&[], false), 2, 2).is_ok());
        assert!(check_metadata_matches_export(&adjudication(&[], false), 2, 1).is_err());
        // Over-removal is contradictory, never approved.
        assert!(check_metadata_matches_export(&adjudication(&[0, 1, 2], true), 2, 0).is_err());

        // Real entry point: an advisory pruning run must reach final
        // metadata and the serialized chain consistently — retained EQ
        // exported, adjudication bound to the frozen evaluation, and the
        // agreement check passing on real artifacts (single and multi
        // measurement, zero-weight seat included).
        for measurements in [1, 2] {
            let config = advisory_pruning_config(measurements);
            let directory = tempfile::tempdir().unwrap();
            let result = optimize_room(&config, 48_000.0, None, Some(directory.path())).unwrap();
            let graph = result.to_dsp_chain_output();
            assert!(graph.validate().is_ok());
            let report =
                roomeq_export::roundtrip::verify_biquad_json_roundtrip(&graph, 48_000.0, 1e-10)
                    .unwrap();
            let channel_adjudication = graph
                .metadata
                .as_ref()
                .expect("native export must retain optimization metadata")
                .veto_adjudication
                .as_ref()
                .expect("workflow must preserve cumulative pruning evidence")
                .get("left")
                .expect("left adjudication present");
            assert!(!channel_adjudication.enforced);
            assert!(channel_adjudication.removed_filter_indices.is_empty());
            assert!(
                channel_adjudication
                    .f0_reference_id
                    .contains("conditions-v1:")
            );
            assert!(report.sections > 0, "advisory row exports the retained EQ");
            let fitted = result.channel_results["left"].biquads.len();
            check_metadata_matches_export(channel_adjudication, fitted, report.sections)
                .unwrap_or_else(|error| {
                    panic!("measurements={measurements}: metadata/export agreement failed: {error}")
                });
        }
    }

    #[test]
    fn workflow_pruning_audit_records_stale_paths() {
        let rows = vec![
            PruningMatrixRow {
                mode: PruningMode::Ordinary,
                filter_class: PruningFilterClass::Iir,
                routed_variant: PruningRoutedVariant::Unrouted,
                condition_ids: vec![String::from("flat-55phon")],
                outcome: MatrixRowOutcome::Ran,
            },
            PruningMatrixRow {
                mode: PruningMode::Adaptive,
                filter_class: PruningFilterClass::Fir,
                routed_variant: PruningRoutedVariant::Routed,
                condition_ids: vec![String::from("music-85phon")],
                outcome: MatrixRowOutcome::Stale {
                    reason: String::from("chain changed after the row ran"),
                },
            },
            PruningMatrixRow {
                mode: PruningMode::Pareto,
                filter_class: PruningFilterClass::Hybrid,
                routed_variant: PruningRoutedVariant::Routed,
                condition_ids: vec![String::from("flat-55phon")],
                outcome: MatrixRowOutcome::Skipped {
                    reason: String::from("hybrid Pareto needs the G4 engine commit"),
                },
            },
            PruningMatrixRow {
                mode: PruningMode::Refinement,
                filter_class: PruningFilterClass::Iir,
                routed_variant: PruningRoutedVariant::Unrouted,
                condition_ids: Vec::new(),
                outcome: MatrixRowOutcome::Ran,
            },
        ];
        let report = audit_pruning_matrix(&rows);
        assert_eq!(report.ran, 1);
        assert_eq!(report.stale.len(), 2);
        assert!(
            report
                .stale
                .iter()
                .any(|entry| entry.contains("chain changed"))
        );
        assert!(
            report
                .stale
                .iter()
                .any(|entry| entry.contains("declares no conditions"))
        );
        assert_eq!(report.skipped.len(), 1);
        let json = serde_json::to_value(&report).unwrap();
        let back: MatrixAuditReport = serde_json::from_value(json).unwrap();
        assert_eq!(back, report);
    }

    #[test]
    fn workflow_listening_handoff_needs_verified_graph_and_frozen_stimuli() {
        // Stimuli render from frozen graphs only: a newly finalized graph
        // cannot bind stale earlier stimuli.
        let bound = bind_listening_stimuli(
            "graph-final",
            &[("stim-1".to_string(), "graph-final".to_string())],
        );
        assert_eq!(bound.unwrap(), vec![String::from("stim-1")]);
        assert!(
            bind_listening_stimuli(
                "graph-final-2",
                &[("stim-1".to_string(), "graph-final".to_string())]
            )
            .is_err()
        );
        // Rejected playback never yields an approved listening bundle.
        assert_eq!(
            approve_listening_bundle(PlaybackStatus::Unassessed, &["stim-1".to_string()]),
            ListeningBundleApproval::Rejected {
                reason: String::from(
                    "final playback validation rejected or never assessed the graph; \
                     no approved listening bundle"
                ),
            }
        );
        assert_eq!(
            approve_listening_bundle(PlaybackStatus::Verified, &["stim-1".to_string()]),
            ListeningBundleApproval::Approved
        );
        assert!(matches!(
            approve_listening_bundle(PlaybackStatus::SimulatedPass, &["stim-1".to_string()]),
            ListeningBundleApproval::Advisory { .. }
        ));
    }
}
