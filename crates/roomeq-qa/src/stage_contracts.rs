//! Structural validation of RoomEQ stage contracts.
//!
//! This deliberately does not inspect optimizer scores.  It catches the
//! cheaper and more damaging class of failures where a stage disappears,
//! runs twice, or reports a failed machine check while the final scalar score
//! still looks plausible.

use roomeq_engine::PipelineStepId;
use roomeq_model::{StageOutcome, StageStatus};
use std::collections::HashSet;

/// Validate that every stage applicable to a run has an explicit outcome.
/// `applicable` is supplied by the workflow because optional DSP stages vary
/// by topology and processing mode.
pub fn validate_trace(outcomes: &[StageOutcome], applicable: &[&str]) -> Vec<String> {
    let mut failures = Vec::new();
    let mut seen = HashSet::new();

    for outcome in outcomes {
        if !seen.insert(outcome.stage.as_str()) {
            failures.push(format!("duplicate stage outcome: {}", outcome.stage));
        }
        for check in &outcome.checks {
            if !check.passed
                && matches!(
                    check.kind,
                    roomeq_model::StageCheckKind::Structural | roomeq_model::StageCheckKind::Safety
                )
            {
                failures.push(format!(
                    "{} check {} failed{}",
                    outcome.stage,
                    check.id,
                    check
                        .diagnostic
                        .as_deref()
                        .map(|d| format!(": {d}"))
                        .unwrap_or_default()
                ));
            }
        }
        if outcome.status == StageStatus::Failed {
            failures.push(format!("stage failed: {}", outcome.stage));
        }
    }

    for stage in applicable {
        if !seen.contains(stage) {
            failures.push(format!("missing applicable stage outcome: {stage}"));
        }
    }

    failures
}

/// The stable names corresponding to the engine's canonical pipeline order.
pub fn canonical_stage_names() -> Vec<&'static str> {
    PipelineStepId::ALL
        .iter()
        .map(|step| match step {
            PipelineStepId::ConfigPreparation => "config_preparation",
            PipelineStepId::Validation => "validation",
            PipelineStepId::TopologyRouteSelection => "topology_route_selection",
            PipelineStepId::TopologyWorkflowExecution => "topology_workflow_execution",
            PipelineStepId::GenericChannelOptimization => "generic_channel_optimization",
            PipelineStepId::FirGeneration => "fir_generation",
            PipelineStepId::MixedPhaseFirGeneration => "mixed_phase_fir_generation",
            PipelineStepId::PhaseCorrection => "phase_correction",
            PipelineStepId::TimeAlignment => "time_alignment",
            PipelineStepId::SpectralAlignment => "spectral_alignment",
            PipelineStepId::InterChannelTimbreMatching => "inter_channel_timbre_matching",
            PipelineStepId::HeightChannelAlignment => "height_channel_alignment",
            PipelineStepId::PhaseAlignment => "phase_alignment",
            PipelineStepId::GroupDelayOptimization => "group_delay_optimization",
            PipelineStepId::ImpulseResponseComputation => "impulse_response_computation",
            PipelineStepId::ChannelMatching => "channel_matching",
            PipelineStepId::MetadataRefresh => "metadata_refresh",
            PipelineStepId::SanityCheck => "sanity_check",
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use roomeq_model::{StageCheck, StageCheckKind};

    fn outcome(stage: &str) -> StageOutcome {
        StageOutcome {
            stage: stage.into(),
            status: StageStatus::Applied,
            advisories: vec![],
            checks: vec![],
        }
    }

    #[test]
    fn missing_and_duplicate_stages_are_blocking() {
        let outcomes = vec![outcome("validation"), outcome("validation")];
        let failures = validate_trace(&outcomes, &["validation", "metadata_refresh"]);
        assert!(failures.iter().any(|f| f.contains("duplicate")));
        assert!(failures.iter().any(|f| f.contains("metadata_refresh")));
    }

    #[test]
    fn failed_structural_check_is_reported() {
        let mut validation = outcome("validation");
        validation.checks.push(StageCheck::fail(
            "finite_sample_rate",
            StageCheckKind::Structural,
            "NaN",
        ));
        assert!(
            validate_trace(&[validation], &["validation"])
                .iter()
                .any(|f| f.contains("finite_sample_rate"))
        );
    }
}
