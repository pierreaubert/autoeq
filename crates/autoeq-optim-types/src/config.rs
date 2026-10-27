//! Parser-independent configuration values shared by library clients.

use crate::LossType;
use autoeq_core::PeqModel;
use std::path::PathBuf;

/// Configuration values for loading measurements and running AutoEQ.
///
/// `Default` preserves the plain command's historical values. The speaker,
/// headphone, and room constructors preserve their tuned library defaults.
/// Parsing, validation messages, and process control belong to terminal adapters.
#[derive(Debug, Clone, PartialEq)]
pub struct OptimizationConfig {
    /// Number of IIR filters to use for optimization.
    pub num_filters: usize,

    /// Path to the input curve CSV file (format: frequency,spl).
    /// Required unless speaker, version, and measurement are provided for API data.
    pub curve: Option<PathBuf>,

    /// Path to the optional target curve CSV file (format: frequency,spl).
    /// If not provided, a flat 0 dB target is assumed.
    pub target: Option<PathBuf>,

    /// Product workflow JSON describing source, independent target support,
    /// and an explicit playback device profile.
    pub product_config: Option<PathBuf>,

    /// The sample rate for the IIR filters.
    pub sample_rate: f64,

    /// Maximum absolute dB gain allowed for each filter.
    pub max_db: f64,

    /// Minimum signed dB gain allowed for each filter (the cut bound).
    pub min_db: f64,

    /// Maximum Q factor allowed for each filter.
    pub max_q: f64,

    /// Minimum Q factor allowed for each filter.
    pub min_q: f64,

    /// Minimum frequency allowed for each filter.
    pub min_freq: f64,

    /// Maximum frequency allowed for each filter.
    pub max_freq: f64,

    /// Output PNG file for plotting results.
    pub output: Option<PathBuf>,

    /// Speaker name for API data fetching.
    pub speaker: Option<String>,

    /// Version for API data fetching.
    pub version: Option<String>,

    /// Measurement type for API data fetching.
    pub measurement: Option<String>,

    /// Curve name inside CEA2034 plots to use (only when --measurement CEA2034)
    /// e.g., "Listening Window", "On Axis", "Early Reflections". Default: Listening Window
    pub curve_name: String,

    /// Optimization algorithm to use (e.g., autoeq:isres, autoeq:cobyla)
    pub algo: String,

    /// Optional population size for population-based algorithms (e.g., ISRES)
    pub population: usize,

    /// Maximum number of evaluations for the optimizer
    pub maxeval: usize,

    /// Load a checked warm-start candidate from this checkpoint file.
    pub resume_state: Option<PathBuf>,

    /// Atomically save a recoverable warm-start checkpoint to this file.
    pub checkpoint_state: Option<PathBuf>,

    /// Resume the complete seeded AutoEQ DE optimizer state from this file.
    /// Exact continuation requires the same measurement, config, seed, executable,
    /// and numeric runtime environment.
    pub resume_exact: Option<PathBuf>,

    /// Atomically save complete AutoEQ DE state after each safe generation barrier.
    pub checkpoint_exact: Option<PathBuf>,

    /// Whether to run a local refinement after global optimization
    pub refine: bool,

    /// Local optimizer to use for refinement (e.g., cobyla)
    pub local_algo: String,

    /// Bayesian optimization Sobol hot-start samples (0 = automatic)
    pub bo_initial_samples: usize,

    /// Bayesian optimization batch size for parallel expensive evaluations (0 = automatic)
    pub bo_batch_size: usize,

    /// Bayesian optimization posterior-std threshold for local-refiner handoff (0 disables)
    pub bo_posterior_std_threshold: f64,

    /// Bayesian optimization acquisition: ei, qei, or thompson
    pub bo_acquisition: String,

    /// Use Monte-Carlo qEHVI Bayesian optimization for multi-objective data
    pub bo_ehvi: bool,

    /// Minimum spacing between filter center frequencies in octaves (0 disables)
    pub min_spacing_oct: f64,

    /// Weight for the spacing penalty in the objective function
    pub spacing_weight: f64,

    /// Second-difference penalty weight on the correction curve (dB/decade^2).
    /// 0 disables. Suggested starting point: 0.05.
    pub smoothness_weight: f64,

    /// L_p exponent for smoothness penalty (1.0 = TV^2/sparsifier, 2.0 = L2).
    pub smoothness_exponent: f64,

    /// Schroeder cutoff for relaxed smoothness in the modal region.
    pub smoothness_schroeder_hz: Option<f64>,

    /// Modal-region weight multiplier (0 = exempt modal region entirely).
    pub smoothness_modal_scale: f64,

    /// Enable smoothing (regularization) of the inverted target curve
    pub smooth: bool,

    /// Smoothing level as 1/N octave (N in [1..24]). Example: N=6 => 1/6 octave smoothing
    pub smooth_n: usize,

    /// Loss function to optimize (flat or score).
    pub loss: LossType,

    /// PEQ model that defines the filter structure
    pub peq_model: PeqModel,

    /// Optimization tolerance (tol parameter for DE algorithm)
    pub tolerance: f64,

    /// Absolute tolerance (atol parameter for DE algorithm)
    pub atolerance: f64,

    /// Recombination probability for DE algorithm (0.0 to 1.0)
    pub recombination: f64,

    /// DE strategy to use (e.g., best1bin, rand1bin, currenttobest1bin, adaptive)
    pub strategy: String,

    /// Adaptive weight for F parameter (DE adaptive strategies only)
    pub adaptive_weight_f: f64,

    /// Adaptive weight for CR parameter (DE adaptive strategies only)
    pub adaptive_weight_cr: f64,

    /// Disable parallel evaluation for DE algorithm (default: parallel is enabled)
    pub no_parallel: bool,

    /// Number of threads to use for parallel evaluation (0 = use all available cores)
    pub parallel_threads: usize,

    /// Random seed for deterministic optimization (default: random seed for each run)
    /// Setting this makes optimization results reproducible
    pub seed: Option<u64>,

    /// Quality assurance mode with optional threshold: suppress normal output, show summary line and analysis
    /// Bare `--qa` requires strict improvement at threshold zero, convergence, and spacing.
    /// A supplied threshold (e.g., `--qa 0.4`) keeps the same strict predicate.
    pub qa: Option<f64>,

    /// Path to first driver measurement CSV file (for multi-driver optimization with --loss drivers-flat)
    pub driver1: Option<PathBuf>,

    /// Path to second driver measurement CSV file (for multi-driver optimization with --loss drivers-flat)
    pub driver2: Option<PathBuf>,

    /// Path to third driver measurement CSV file (for multi-driver optimization with --loss drivers-flat)
    pub driver3: Option<PathBuf>,

    /// Path to fourth driver measurement CSV file (for multi-driver optimization with --loss drivers-flat)
    pub driver4: Option<PathBuf>,

    /// Crossover type for multi-driver optimization (butterworth2, linkwitzriley2, linkwitzriley4)
    pub crossover_type: String,
}

impl OptimizationConfig {
    /// Get the effective PEQ model
    pub fn effective_peq_model(&self) -> PeqModel {
        self.peq_model
    }

    /// Check if the first filter should be a highpass (for compatibility)
    pub fn uses_highpass_first(&self) -> bool {
        matches!(
            self.effective_peq_model(),
            PeqModel::HpPk | PeqModel::HpPkLp
        )
    }

    /// Create configuration with speaker optimization defaults
    ///
    /// These defaults are tuned for typical speaker EQ optimization using
    /// CEA2034 measurements from spinorama.org.
    pub fn speaker_defaults() -> Self {
        Self {
            num_filters: 5,
            sample_rate: 48000.0,
            loss: LossType::SpeakerFlat,
            algo: "autoeq:de".to_string(),
            // Bumped from the previous (population=50, maxeval=2000,
            // strategy="currenttobest1bin") defaults after some runs
            // failed to converge. L-SHADE adapts its mutation and
            // crossover rates per generation, so it handles noisy
            // room-EQ loss landscapes much better than the fixed-rate
            // strategies. 300/50000 is conservative for a 7-filter
            // problem — cheaper runs can still pick `--preset quick`.
            population: 300,
            maxeval: 50000,
            strategy: "lshade".to_string(),
            min_db: -12.0,
            max_db: 12.0,
            min_q: 0.5,
            max_q: 10.0,
            min_freq: 20.0,
            max_freq: 20000.0,
            min_spacing_oct: 0.5,
            spacing_weight: 20.0,
            smoothness_weight: 0.0,
            smoothness_exponent: 1.0,
            smoothness_schroeder_hz: None,
            smoothness_modal_scale: 0.1,
            smooth: true,
            smooth_n: 1,
            refine: false,
            local_algo: "cobyla".to_string(),
            bo_initial_samples: 0,
            bo_batch_size: 0,
            bo_posterior_std_threshold: 0.0,
            bo_acquisition: "qei".to_string(),
            bo_ehvi: false,
            tolerance: 1e-3,
            atolerance: 1e-4,
            recombination: 0.9,
            adaptive_weight_f: 0.8,
            adaptive_weight_cr: 0.7,
            peq_model: PeqModel::Pk,
            curve_name: "Listening Window".to_string(),
            // File paths/flags default to None/false
            curve: None,
            target: None,
            product_config: None,
            output: None,
            resume_state: None,
            checkpoint_state: None,
            resume_exact: None,
            checkpoint_exact: None,
            speaker: None,
            version: None,
            measurement: None,
            no_parallel: false,
            parallel_threads: 0,
            seed: None,
            qa: None,
            driver1: None,
            driver2: None,
            driver3: None,
            driver4: None,
            crossover_type: "linkwitzriley4".to_string(),
        }
    }

    /// Create configuration with headphone optimization defaults
    ///
    /// These defaults are tuned for headphone EQ optimization using
    /// frequency response measurements and Harman target curves.
    pub fn headphone_defaults() -> Self {
        Self {
            loss: LossType::HeadphoneScore,
            num_filters: 7,
            ..Self::speaker_defaults()
        }
    }

    /// Create configuration with room EQ optimization defaults
    ///
    /// These defaults are tuned for room correction, focusing on
    /// low frequencies where room modes are most problematic.
    pub fn roomeq_defaults() -> Self {
        Self {
            num_filters: 10,
            max_freq: 500.0, // Room EQ focuses on low frequencies
            ..Self::speaker_defaults()
        }
    }
}

impl Default for OptimizationConfig {
    fn default() -> Self {
        Self {
            num_filters: 7,
            curve: None,
            target: None,
            product_config: None,
            sample_rate: 48000.0,
            max_db: 3.0,
            min_db: -12.0,
            max_q: 3.0,
            min_q: 1.0,
            min_freq: 60.0,
            max_freq: 16000.0,
            output: None,
            speaker: None,
            version: None,
            measurement: None,
            curve_name: "Listening Window".to_string(),
            algo: "autoeq:cobyla".to_string(),
            population: 300,
            maxeval: 2_000,
            resume_state: None,
            checkpoint_state: None,
            resume_exact: None,
            checkpoint_exact: None,
            refine: false,
            local_algo: "cobyla".to_string(),
            bo_initial_samples: 0,
            bo_batch_size: 0,
            bo_posterior_std_threshold: 0.0,
            bo_acquisition: "qei".to_string(),
            bo_ehvi: false,
            min_spacing_oct: 0.2,
            spacing_weight: 20.0,
            smoothness_weight: 0.0,
            smoothness_exponent: 1.0,
            smoothness_schroeder_hz: None,
            smoothness_modal_scale: 0.1,
            smooth: true,
            smooth_n: 2,
            loss: LossType::SpeakerFlat,
            peq_model: PeqModel::Pk,
            tolerance: 1e-3,
            atolerance: 1e-4,
            recombination: 0.9,
            strategy: "currenttobest1bin".to_string(),
            adaptive_weight_f: 0.9,
            adaptive_weight_cr: 0.9,
            no_parallel: false,
            parallel_threads: 0,
            seed: None,
            qa: None,
            driver1: None,
            driver2: None,
            driver3: None,
            driver4: None,
            crossover_type: "linkwitzriley4".to_string(),
        }
    }
}
