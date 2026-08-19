// This module is now a thin facade over route-specific workflow executors.
// The implementation bodies live in `stereo.rs`, `stereo_sub.rs`,
// `home_cinema.rs`, and `generic.rs`. Multi-seat, multi-sub, and
// bass-management behavior is composed inside those supported topology
// executors rather than exposed as standalone workflows.

use super::home_cinema::HomeCinemaExecutor;
use super::stereo::Stereo20Executor;
use super::stereo_sub::Stereo21Executor;
use super::types::{
    WorkflowAssembly, WorkflowExecutor, WorkflowProgressCallbackFactory, WorkflowStageCallback,
};
use crate::DEFAULT_FREQUENCY_SAMPLES;
use roomeq_engine::error::Result;
use roomeq_engine::room_result::RoomOptimizationResult;
use roomeq_model::{RoomConfig, SystemConfig};
use std::collections::HashMap;
use std::path::Path;

#[allow(clippy::too_many_arguments)]
pub(crate) fn optimize_stereo_2_0_with_progress_and_probe_arrivals<'a>(
    config: &RoomConfig,
    sys: &SystemConfig,
    sample_rate: f64,
    output_dir: &Path,
    probe_arrival_overrides: Option<&'a HashMap<String, f64>>,
    progress_factory: Option<&'a mut WorkflowProgressCallbackFactory<'a>>,
    stage_callback: Option<&'a mut WorkflowStageCallback<'a>>,
    frequency_samples: usize,
) -> Result<RoomOptimizationResult> {
    let executor = Stereo20Executor;
    let mut assembly = WorkflowAssembly {
        config,
        sys,
        sample_rate,
        frequency_samples,
        output_dir,
        probe_arrival_overrides,
        progress_factory,
        stage_callback,
    };
    executor.execute(&mut assembly)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn optimize_stereo_2_1_with_progress_and_probe_arrivals<'a>(
    config: &RoomConfig,
    sys: &SystemConfig,
    sample_rate: f64,
    output_dir: &Path,
    probe_arrival_overrides: Option<&'a HashMap<String, f64>>,
    progress_factory: Option<&'a mut WorkflowProgressCallbackFactory<'a>>,
    stage_callback: Option<&'a mut WorkflowStageCallback<'a>>,
    frequency_samples: usize,
) -> Result<RoomOptimizationResult> {
    let executor = Stereo21Executor;
    let mut assembly = WorkflowAssembly {
        config,
        sys,
        sample_rate,
        frequency_samples,
        output_dir,
        probe_arrival_overrides,
        progress_factory,
        stage_callback,
    };
    executor.execute(&mut assembly)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn optimize_home_cinema_with_progress_and_probe_arrivals<'a>(
    config: &RoomConfig,
    sys: &SystemConfig,
    sample_rate: f64,
    output_dir: &Path,
    probe_arrival_overrides: Option<&'a HashMap<String, f64>>,
    progress_factory: Option<&'a mut WorkflowProgressCallbackFactory<'a>>,
    stage_callback: Option<&'a mut WorkflowStageCallback<'a>>,
    frequency_samples: usize,
) -> Result<RoomOptimizationResult> {
    let executor = HomeCinemaExecutor;
    let mut assembly = WorkflowAssembly {
        config,
        sys,
        sample_rate,
        frequency_samples,
        output_dir,
        probe_arrival_overrides,
        progress_factory,
        stage_callback,
    };
    executor.execute(&mut assembly)
}

/// Workflow for Stereo 2.0 (No Subwoofer)
///
/// Per-channel EQ is delegated to the crate-owned channel workflow so that
/// `excursion_protection`, `target_response`, and `cea2034_correction`
/// all apply inside the workflow. An alignment-gain plugin is prepended
/// to the returned DSP chain without affecting feature decisions
/// (F3 detection, passband estimation, and target shaping all use
/// relative-to-peak thresholds that are gain-invariant).
pub fn optimize_stereo_2_0(
    config: &RoomConfig,
    sys: &SystemConfig,
    sample_rate: f64,
    output_dir: &Path,
) -> Result<RoomOptimizationResult> {
    optimize_stereo_2_0_with_progress(config, sys, sample_rate, output_dir, None, None)
}

pub fn optimize_stereo_2_0_with_progress<'a>(
    config: &RoomConfig,
    sys: &SystemConfig,
    sample_rate: f64,
    output_dir: &Path,
    progress_factory: Option<&'a mut WorkflowProgressCallbackFactory<'a>>,
    stage_callback: Option<&'a mut WorkflowStageCallback<'a>>,
) -> Result<RoomOptimizationResult> {
    let executor = Stereo20Executor;
    let mut assembly = WorkflowAssembly {
        config,
        sys,
        sample_rate,
        frequency_samples: DEFAULT_FREQUENCY_SAMPLES,
        output_dir,
        probe_arrival_overrides: None,
        progress_factory,
        stage_callback,
    };
    executor.execute(&mut assembly)
}

/// Workflow for Stereo 2.1 (With Subwoofer)
///
/// Phase 3b: per-channel features (`excursion_protection`, `target_response`,
/// `cea2034_correction`) are applied by the channel workflow at the
/// Pre-EQ stage and the resulting plugin stack is inserted before the
/// crossover HP/LP in the final DSP chain. Post-EQ remains a plain cleanup
/// pass on the post-crossover curve, with the "do no harm" guard from
/// Phase 3a.
pub fn optimize_stereo_2_1(
    config: &RoomConfig,
    sys: &SystemConfig,
    sample_rate: f64,
    output_dir: &Path,
) -> Result<RoomOptimizationResult> {
    optimize_stereo_2_1_with_progress(config, sys, sample_rate, output_dir, None, None)
}

pub fn optimize_stereo_2_1_with_progress<'a>(
    config: &RoomConfig,
    sys: &SystemConfig,
    sample_rate: f64,
    output_dir: &Path,
    progress_factory: Option<&'a mut WorkflowProgressCallbackFactory<'a>>,
    stage_callback: Option<&'a mut WorkflowStageCallback<'a>>,
) -> Result<RoomOptimizationResult> {
    let executor = Stereo21Executor;
    let mut assembly = WorkflowAssembly {
        config,
        sys,
        sample_rate,
        frequency_samples: DEFAULT_FREQUENCY_SAMPLES,
        output_dir,
        probe_arrival_overrides: None,
        progress_factory,
        stage_callback,
    };
    executor.execute(&mut assembly)
}

/// Workflow for Home Cinema X.0 / X.1 (any channel count)
///
/// Handles all standard layouts: 5.0, 5.1, 7.1, 9.1, 5.1.2, 5.1.4, 7.1.2, 7.1.4, 9.1.4, 9.1.6.
/// The workflow is layout-agnostic: channels are classified as "main" (everything except LFE)
/// and "sub" (LFE if present). The specific channel names don't affect the algorithm.
pub fn optimize_home_cinema(
    config: &RoomConfig,
    sys: &SystemConfig,
    sample_rate: f64,
    _output_dir: &Path,
) -> Result<RoomOptimizationResult> {
    optimize_home_cinema_with_progress(config, sys, sample_rate, _output_dir, None, None)
}

pub fn optimize_home_cinema_with_progress<'a>(
    config: &RoomConfig,
    sys: &SystemConfig,
    sample_rate: f64,
    _output_dir: &Path,
    progress_factory: Option<&'a mut WorkflowProgressCallbackFactory<'a>>,
    stage_callback: Option<&'a mut WorkflowStageCallback<'a>>,
) -> Result<RoomOptimizationResult> {
    let executor = HomeCinemaExecutor;
    let mut assembly = WorkflowAssembly {
        config,
        sys,
        sample_rate,
        frequency_samples: DEFAULT_FREQUENCY_SAMPLES,
        output_dir: _output_dir,
        probe_arrival_overrides: None,
        progress_factory,
        stage_callback,
    };
    executor.execute(&mut assembly)
}
