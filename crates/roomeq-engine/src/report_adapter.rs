//! Conversion from runtime/backend results to stable RoomEQ report contracts.

pub fn to_epa_score(value: autoeq_optim::loss::epa::score::EpaScore) -> roomeq_model::EpaScore {
    roomeq_model::EpaScore {
        evaluation: value.evaluation,
        potency: value.potency,
        activity: value.activity,
        preference: value.preference,
        sharpness_acum: value.sharpness_acum,
        roughness: value.roughness,
        total_loudness_sone: value.total_loudness_sone,
        loudness_balance: value.loudness_balance,
    }
}

pub fn to_temporal_ir_masking(
    value: autoeq_optim::loss::epa::score::TemporalIrMaskingMetrics,
) -> roomeq_model::TemporalIrMaskingMetrics {
    roomeq_model::TemporalIrMaskingMetrics {
        main_index: value.main_index,
        main_time_ms: value.main_time_ms,
        pre_ringing_peak_db: value.pre_ringing_peak_db,
        post_ringing_peak_db: value.post_ringing_peak_db,
        pre_ringing_audible_db: value.pre_ringing_audible_db,
        post_ringing_audible_db: value.post_ringing_audible_db,
        penalty: value.penalty,
    }
}

pub fn to_optimizer_confidence(
    value: autoeq_optim::optim::OptimizerConfidence,
) -> roomeq_model::OptimizerConfidence {
    match value {
        autoeq_optim::optim::OptimizerConfidence::High => roomeq_model::OptimizerConfidence::High,
        autoeq_optim::optim::OptimizerConfidence::Low => roomeq_model::OptimizerConfidence::Low,
        autoeq_optim::optim::OptimizerConfidence::Unusable => {
            roomeq_model::OptimizerConfidence::Unusable
        }
    }
}

fn to_optimizer_termination(
    value: autoeq_optim::optim::OptimizerTermination,
) -> roomeq_model::OptimizerTermination {
    match value {
        autoeq_optim::optim::OptimizerTermination::Converged => {
            roomeq_model::OptimizerTermination::Converged
        }
        autoeq_optim::optim::OptimizerTermination::EvaluationLimit => {
            roomeq_model::OptimizerTermination::EvaluationLimit
        }
        autoeq_optim::optim::OptimizerTermination::NonConverged => {
            roomeq_model::OptimizerTermination::NonConverged
        }
        autoeq_optim::optim::OptimizerTermination::UserStopped => {
            roomeq_model::OptimizerTermination::UserStopped
        }
        autoeq_optim::optim::OptimizerTermination::BackendFailure => {
            roomeq_model::OptimizerTermination::BackendFailure
        }
        autoeq_optim::optim::OptimizerTermination::InvalidResult => {
            roomeq_model::OptimizerTermination::InvalidResult
        }
    }
}

pub fn to_optimizer_run_evidence(
    value: &autoeq_optim::optim::OptimizerRunEvidence,
) -> roomeq_model::OptimizerRunEvidence {
    roomeq_model::OptimizerRunEvidence {
        algorithm: value.algorithm.clone(),
        termination: to_optimizer_termination(value.termination),
        converged: value.converged,
        best_effort: value.best_effort,
        status: value.status.clone(),
        objective: value.objective,
        evaluation_count: value.evaluation_count,
        evaluation_limit: value.evaluation_limit,
        seed: value.seed,
        max_constraint_violation: value.max_constraint_violation,
        confidence: to_optimizer_confidence(value.confidence),
        selected_for_output: value.selected_for_output,
        restart_history: value
            .restart_history
            .iter()
            .map(|restart| roomeq_model::OptimizerRestartEvidence {
                attempt: restart.attempt,
                seed: restart.seed,
                termination: to_optimizer_termination(restart.termination),
                objective: restart.objective,
            })
            .collect(),
    }
}
