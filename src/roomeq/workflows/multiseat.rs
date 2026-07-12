// Multi-seat workflow executor.
//
// Currently a high-level dispatcher: when multi-seat optimization is disabled
// it returns an empty/unchanged result immediately, preserving the original
// channel state. When enabled it delegates to `crate::roomeq::multiseat`.

use super::super::optimize::RoomOptimizationResult;
use super::super::types::OptimizationMetadata;
use super::types::{WorkflowAssembly, WorkflowExecutor};
use crate::error::{AutoeqError, Result};
use log::info;

#[allow(dead_code)]
pub(in super::super) struct MultiseatExecutor;

impl WorkflowExecutor for MultiseatExecutor {
    fn execute<'cfg, 'p, 's>(
        &self,
        assembly: &mut WorkflowAssembly<'cfg, 'p, 's>,
    ) -> Result<RoomOptimizationResult> {
        let config = assembly.config;

        info!("Running Multi-seat Optimization Workflow");

        let enabled = config
            .optimizer
            .multi_seat
            .as_ref()
            .map(|m| m.enabled)
            .unwrap_or(false);

        if !enabled {
            return Ok(RoomOptimizationResult {
                channels: Default::default(),
                channel_results: Default::default(),
                combined_pre_score: 0.0,
                combined_post_score: 0.0,
                metadata: OptimizationMetadata {
                    pre_score: 0.0,
                    post_score: 0.0,
                    algorithm: config.optimizer.algorithm.clone(),
                    loss_type: Some(config.optimizer.loss_type.clone()),
                    iterations: config.optimizer.max_iter,
                    timestamp: chrono::Utc::now().to_rfc3339(),
                    inter_channel_deviation: None,
                    epa_per_channel: Default::default(),
                    epa_multichannel: Default::default(),
                    group_delay: None,
                    perceptual_metrics: None,
                    home_cinema_layout: None,
                    multi_seat_coverage: None,
                    multi_seat_correction: None,
                    bass_management: None,
                    timing_diagnostics: None,
                    ctc: None,
                    perceptual_policy: None,
                    bootstrap_uncertainty: None,
                    validation_bundle: None,
                    supporting_source: None,
                    correction_acceptance: None,
                    stage_outcomes: Vec::new(),
                },
            });
        }

        Err(AutoeqError::InvalidConfiguration {
            message: "Enabled multi-seat workflow is not yet implemented as a standalone executor"
                .to_string(),
        })
    }
}
