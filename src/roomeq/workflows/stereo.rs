// Stereo 2.0 workflow executor.

use super::super::optimize::RoomOptimizationResult;
use super::super::pipeline::{PipelineStepId, PipelineStepStatus};
use super::super::types::OptimizationMetadata;
use super::bass_management::resolve_single_source;
use super::misc::align_channels_to_lowest;
use super::run::run_channel_via_generic_path;
use super::supporting_source::{
    load_single_source_curves, partition_roles, process_supporting_source_channels,
};
use super::types::{WorkflowAssembly, WorkflowExecutor};
use super::workflow::workflow_stage_event;
use crate::error::Result;
use log::info;
use std::collections::HashMap;

pub(in super::super) struct Stereo20Executor;

impl WorkflowExecutor for Stereo20Executor {
    fn execute<'cfg, 'p, 's>(
        &self,
        assembly: &mut WorkflowAssembly<'cfg, 'p, 's>,
    ) -> Result<RoomOptimizationResult> {
        let config = assembly.config;
        let sys = assembly.sys;
        let sample_rate = assembly.sample_rate;
        let output_dir = assembly.output_dir;

        info!("Running Stereo 2.0 Optimization Workflow");

        // 1. Partition roles into single-source and supporting-source channels.
        let (single_roles, _supporting_roles) = partition_roles(config, sys)?;

        // 2. Load single-source measurements
        let curves = load_single_source_curves(config, sys, &single_roles)?;

        // 3. Alignment
        let mut ranges = HashMap::new();
        for role in curves.keys() {
            ranges.insert(role.clone(), (100.0, 2000.0));
        }
        let gains = align_channels_to_lowest(&curves, &ranges);

        // 4. Optimization — delegate each single channel to the generic path so features apply.
        let mut channel_chains = HashMap::new();
        let mut channel_results = HashMap::new();
        let mut pre_scores = Vec::new();
        let mut post_scores = Vec::new();

        let total_channels = single_roles.len();
        let max_iterations = config.optimizer.max_iter;
        for (channel_index, role) in single_roles.iter().enumerate() {
            let gain = *gains.get(role).unwrap_or(&0.0);
            let source = resolve_single_source(role, config, sys)?;

            info!("  Optimizing '{}' with alignment gain {:.2} dB", role, gain);

            let (chain, ch_result, pre_score, post_score, _fir, _multiseat_rejection) =
                run_channel_via_generic_path(
                    role,
                    source,
                    config,
                    gain,
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
            "Optimized stereo channels",
            0.90,
        )?;

        let avg_pre = if pre_scores.is_empty() {
            0.0
        } else {
            pre_scores.iter().sum::<f64>() / pre_scores.len() as f64
        };
        let avg_post = if post_scores.is_empty() {
            0.0
        } else {
            post_scores.iter().sum::<f64>() / post_scores.len() as f64
        };

        info!(
            "Average pre-score: {:.4}, post-score: {:.4}",
            avg_pre, avg_post
        );

        let epa_cfg = config.optimizer.epa_config.clone().unwrap_or_default();
        let epa_per_channel =
            crate::roomeq::output::compute_epa_per_channel(&channel_chains, &epa_cfg);
        let epa_multichannel =
            crate::roomeq::output::compute_epa_multichannel(&channel_chains, &epa_cfg);

        let mut metadata = OptimizationMetadata {
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
            stage_outcomes: Vec::new(),
        };

        // 5. Process supporting-source channels.
        process_supporting_source_channels(
            config,
            sys,
            sample_rate,
            output_dir,
            &mut channel_chains,
            &mut channel_results,
            &mut metadata,
        )?;

        Ok(RoomOptimizationResult {
            channels: channel_chains,
            channel_results,
            combined_pre_score: avg_pre,
            combined_post_score: avg_post,
            metadata,
        })
    }
}
