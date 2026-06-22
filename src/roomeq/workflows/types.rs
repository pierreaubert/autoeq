use super::super::optimize::RoomOptimizationResult;
use super::super::pipeline::{PipelineStepId, PipelineStepStatus};
use super::super::types::{RoomConfig, SystemConfig};
use super::misc::{
    create_crossover_filters, is_linear_phase_crossover_type, linear_phase_crossover_coefficients,
};
use crate::error::Result;
use crate::response;
use ndarray::Array1;
use std::path::Path;
use std::sync::{Arc, atomic::AtomicBool};

pub(in super::super) struct WorkflowProgressCallback {
    pub callback: crate::optim::OptimProgressCallback,
    pub stopped: Arc<AtomicBool>,
}

pub(in super::super) type WorkflowProgressCallbackFactory<'a> =
    dyn FnMut(&str, usize, usize, usize) -> Option<WorkflowProgressCallback> + 'a;

pub(in super::super) type WorkflowStageCallback<'a> =
    dyn FnMut(PipelineStepId, PipelineStepStatus, &str, f64) -> Result<()> + 'a;

/// Assembled inputs passed to a topology-specific workflow executor.
pub(in super::super) struct WorkflowAssembly<'cfg, 'p, 's> {
    pub config: &'cfg RoomConfig,
    pub sys: &'cfg SystemConfig,
    pub sample_rate: f64,
    pub output_dir: &'cfg Path,
    pub progress_factory: Option<&'p mut WorkflowProgressCallbackFactory<'p>>,
    pub stage_callback: Option<&'s mut WorkflowStageCallback<'s>>,
}

/// Route-specific executor for a RoomEQ workflow topology.
pub(in super::super) trait WorkflowExecutor {
    fn execute<'cfg, 'p, 's>(
        &self,
        assembly: &mut WorkflowAssembly<'cfg, 'p, 's>,
    ) -> Result<RoomOptimizationResult>;
}

pub(super) fn compute_crossover_complex_response(
    type_str: &str,
    freq: f64,
    sample_rate: f64,
    is_lowpass: bool,
    freqs: &Array1<f64>,
) -> Vec<num_complex::Complex64> {
    if is_linear_phase_crossover_type(type_str) {
        let coeffs = linear_phase_crossover_coefficients(freq, sample_rate, is_lowpass);
        response::compute_fir_complex_response(&coeffs, freqs, sample_rate)
    } else {
        let filters = create_crossover_filters(type_str, freq, sample_rate, is_lowpass);
        response::compute_peq_complex_response(&filters, freqs, sample_rate)
    }
}
