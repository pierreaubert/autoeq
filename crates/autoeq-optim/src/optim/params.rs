//! Shared optimization parameters used by both AutoEQ CLI and RoomEQ.
//!
//! [`OptimParams`] decouples the optimization infrastructure from the CLI
//! argument struct (`cli::Args`), allowing roomeq to use the same
//! optimization functions without constructing fake `Args` values.

use crate::cli::Args;
use crate::loss::LossType;
use crate::optim::SmoothnessPenaltyConfig;

pub use autoeq_core::PeqModel;

/// Optimization-relevant parameters extracted from either `cli::Args`
/// (for the autoeq binary) or `roomeq::OptimizerConfig` (for room EQ).
///
/// The optimization functions (`setup_objective_data`, `setup_bounds`,
/// `initial_guess`, `perform_optimization`, etc.) accept this struct
/// instead of the full CLI `Args`.
#[derive(Debug, Clone)]
pub struct OptimParams {
    // -- Filter topology --
    pub num_filters: usize,
    pub peq_model: PeqModel,
    pub sample_rate: f64,

    // -- Bounds --
    pub min_freq: f64,
    pub max_freq: f64,
    pub min_q: f64,
    pub max_q: f64,
    pub min_db: f64,
    pub max_db: f64,

    // -- Loss / objective --
    pub loss: LossType,
    pub smooth: bool,
    pub smooth_n: usize,
    pub min_spacing_oct: f64,
    pub spacing_weight: f64,
    pub smoothness_penalty: Option<SmoothnessPenaltyConfig>,
    pub audibility_deadband: Option<crate::roomeq::AudibilityDeadbandConfig>,

    // -- Algorithm --
    pub algo: String,
    pub population: usize,
    pub maxeval: usize,
    pub refine: bool,
    pub local_algo: String,
    pub bo_initial_samples: usize,
    pub bo_batch_size: usize,
    pub bo_posterior_std_threshold: f64,
    pub bo_acquisition: String,
    pub bo_ehvi: bool,

    // -- DE-specific --
    pub strategy: String,
    pub tolerance: f64,
    pub atolerance: f64,
    pub recombination: f64,
    pub adaptive_weight_f: f64,
    pub adaptive_weight_cr: f64,

    // -- Execution --
    pub no_parallel: bool,
    pub parallel_threads: usize,
    pub seed: Option<u64>,

    /// Suppress non-essential logging (replaces `args.qa.is_some()`).
    pub quiet: bool,
}

impl From<&Args> for OptimParams {
    fn from(args: &Args) -> Self {
        Self {
            num_filters: args.num_filters,
            peq_model: args.effective_peq_model(),
            sample_rate: args.sample_rate,
            min_freq: args.min_freq,
            max_freq: args.max_freq,
            min_q: args.min_q,
            max_q: args.max_q,
            min_db: args.min_db,
            max_db: args.max_db,
            loss: args.loss,
            smooth: args.smooth,
            smooth_n: args.smooth_n,
            min_spacing_oct: args.min_spacing_oct,
            spacing_weight: args.spacing_weight,
            smoothness_penalty: if args.smoothness_weight > 0.0 {
                Some(SmoothnessPenaltyConfig {
                    tv2_weight: args.smoothness_weight,
                    schroeder_hz: args.smoothness_schroeder_hz,
                    modal_weight_scale: args.smoothness_modal_scale,
                    exponent: args.smoothness_exponent,
                })
            } else {
                None
            },
            audibility_deadband: None,
            algo: args.algo.clone(),
            population: args.population,
            maxeval: args.maxeval,
            refine: args.refine,
            local_algo: args.local_algo.clone(),
            bo_initial_samples: args.bo_initial_samples,
            bo_batch_size: args.bo_batch_size,
            bo_posterior_std_threshold: args.bo_posterior_std_threshold,
            bo_acquisition: args.bo_acquisition.clone(),
            bo_ehvi: args.bo_ehvi,
            strategy: args.strategy.clone(),
            tolerance: args.tolerance,
            atolerance: args.atolerance,
            recombination: args.recombination,
            adaptive_weight_f: args.adaptive_weight_f,
            adaptive_weight_cr: args.adaptive_weight_cr,
            no_parallel: args.no_parallel,
            parallel_threads: args.parallel_threads,
            seed: args.seed,
            quiet: args.qa.is_some(),
        }
    }
}
