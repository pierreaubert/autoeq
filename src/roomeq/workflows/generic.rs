// Generic/custom workflow executor.
//
// This is a minimal, testable executor for ad-hoc/custom layouts. It
// processes every `Single` speaker in `config.speakers` independently
// through the generic per-channel path (the same path used by stereo and
// home-cinema workflows) and aggregates the results.

use super::super::optimize::RoomOptimizationResult;
use super::super::pipeline::{PipelineStepId, PipelineStepStatus};
use super::super::types::{OptimizationMetadata, SpeakerConfig};
use super::run::run_channel_via_generic_path;
use super::types::{WorkflowAssembly, WorkflowExecutor};
use super::workflow::workflow_stage_event;
use crate::error::{AutoeqError, Result};
use log::info;
use std::collections::HashMap;

#[allow(dead_code)]
pub(in super::super) struct GenericExecutor;

impl WorkflowExecutor for GenericExecutor {
    fn execute<'cfg, 'p, 's>(
        &self,
        assembly: &mut WorkflowAssembly<'cfg, 'p, 's>,
    ) -> Result<RoomOptimizationResult> {
        let config = assembly.config;
        let sys = assembly.sys;
        let sample_rate = assembly.sample_rate;
        let output_dir = assembly.output_dir;

        info!("Running Generic Optimization Workflow");

        let mut channel_chains = HashMap::new();
        let mut channel_results = HashMap::new();
        let mut pre_scores = Vec::new();
        let mut post_scores = Vec::new();

        // Collect roles from the system mapping, falling back to speaker keys.
        let roles: Vec<String> = if sys.speakers.is_empty() {
            config
                .speakers
                .keys()
                .filter(|k| matches!(config.speakers.get(*k), Some(SpeakerConfig::Single(_))))
                .cloned()
                .collect()
        } else {
            sys.speakers.keys().cloned().collect()
        };

        if roles.is_empty() {
            return Err(AutoeqError::InvalidConfiguration {
                message: "Generic workflow requires at least one Single speaker".to_string(),
            });
        }

        let total_channels = roles.len();
        let max_iterations = config.optimizer.max_iter;
        for (channel_index, role) in roles.iter().enumerate() {
            let speaker_key = sys.speakers.get(role).unwrap_or(role);
            let source = match config.speakers.get(speaker_key) {
                Some(SpeakerConfig::Single(s)) => s,
                _ => {
                    return Err(AutoeqError::InvalidConfiguration {
                        message: format!(
                            "'{}' must be a Single speaker config in generic workflow",
                            role
                        ),
                    });
                }
            };

            info!("  Optimizing '{}' via generic path", role);
            let (chain, ch_result, pre_score, post_score, _fir, _multiseat_rejection) =
                run_channel_via_generic_path(
                    role,
                    source,
                    config,
                    0.0,
                    sample_rate,
                    output_dir,
                    &mut assembly.progress_factory,
                    channel_index,
                    total_channels,
                    max_iterations,
                )?;

            info!(
                "  '{}' pre_score={:.4} post_score={:.4}",
                role, pre_score, post_score
            );

            channel_chains.insert(role.clone(), chain);
            channel_results.insert(role.clone(), ch_result);
            pre_scores.push(pre_score);
            post_scores.push(post_score);
        }

        workflow_stage_event(
            &mut assembly.stage_callback,
            PipelineStepId::GenericChannelOptimization,
            PipelineStepStatus::Completed,
            "Optimized generic/custom channels",
            0.90,
        )?;

        let avg_pre = pre_scores.iter().sum::<f64>() / pre_scores.len() as f64;
        let avg_post = post_scores.iter().sum::<f64>() / post_scores.len() as f64;

        info!(
            "Average pre-score: {:.4}, post-score: {:.4}",
            avg_pre, avg_post
        );

        let epa_cfg = config.optimizer.epa_config.clone().unwrap_or_default();
        let epa_per_channel =
            crate::roomeq::output::compute_epa_per_channel(&channel_chains, &epa_cfg);
        let epa_multichannel =
            crate::roomeq::output::compute_epa_multichannel(&channel_chains, &epa_cfg);

        Ok(RoomOptimizationResult {
            channels: channel_chains,
            channel_results,
            combined_pre_score: avg_pre,
            combined_post_score: avg_post,
            metadata: OptimizationMetadata {
                pre_score: avg_pre,
                post_score: avg_post,
                algorithm: config.optimizer.algorithm.clone(),
                loss_type: Some(config.optimizer.loss_type.clone()),
                iterations: config.optimizer.max_iter,
                timestamp: chrono::Utc::now().to_rfc3339(),
                inter_channel_deviation: None,
                epa_per_channel,
                epa_multichannel,
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
            },
        })
    }
}
