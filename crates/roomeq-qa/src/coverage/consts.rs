use roomeq_model::{FirConfig, ProcessingMode, RoomConfig};
use std::sync::atomic::AtomicUsize;

pub(super) const SAMPLE_RATE: f64 = 48000.0;

pub(super) const SEED: u64 = 42;

/// Default budget for the CMA-ES-backed coverage path.
pub(super) const QA_MAXEVAL: usize = 600_000;
/// Budget for QA runs that deliberately retain Differential Evolution.
pub(super) const DE_QA_MAXEVAL: usize = 15_000;
pub(super) const QA_ALGORITHM: &str = "autoeq:cmaes";
pub(super) const CMAES_QA_POPULATION: usize = 20;
pub(super) const CMAES_QA_NUM_FILTERS: usize = 9;
pub(super) const QA_MAX_FILTER_BOOST_DB: f64 = 12.0;

pub(super) fn qa_budget_for_algorithm(algorithm: &str, requested: usize) -> usize {
    if algorithm.to_ascii_lowercase().contains("de") {
        requested.min(DE_QA_MAXEVAL)
    } else {
        requested
    }
}

pub(super) const FEM_DIR: &str = "data_tests/roomeq/generate/fem";

pub(super) const OPTIM_CONFIG_DIR: &str = "data_tests/roomeq/generate/optimiser-config";

pub(super) fn apply_qa_overrides(config: &mut RoomConfig, maxeval: usize) {
    // Keep the scenario's tuned optimizer (algorithm, filter count, refine,
    // population): forcing a bare local optimizer with few filters makes the
    // multi-measurement scenarios physically unable to improve, which the
    // final safety gate then reports as "no improvement". QA only pins
    // determinism and caps the evaluation budget.
    // Replace AutoDE with CMA-ES while preserving the scenario's other
    // optimizer settings.
    let replaced_auto_de = config.optimizer.algorithm.eq_ignore_ascii_case("autoeq:de");
    if replaced_auto_de {
        config.optimizer.algorithm = QA_ALGORITHM.to_string();
        config.optimizer.population = CMAES_QA_POPULATION;
        config.optimizer.asymmetric_loss = false;
        config.optimizer.refine = true;
        config.optimizer.num_filters = CMAES_QA_NUM_FILTERS;
        config.optimizer.max_db = config.optimizer.max_db.min(QA_MAX_FILTER_BOOST_DB);
    }
    config.optimizer.max_iter = if replaced_auto_de {
        maxeval
    } else {
        qa_budget_for_algorithm(&config.optimizer.algorithm, config.optimizer.max_iter.min(maxeval))
    };
    config.optimizer.seed = Some(SEED);
    // Pin evaluation to a single thread: parallel optimizer evaluation order
    // depends on machine load, which makes seeded runs non-reproducible
    // when many QA cases run concurrently. The harness's own case-level parallelism
    // provides the throughput.
    config.optimizer.parallel_threads = Some(1);

    // Ensure FIR config exists when processing mode requires it
    ensure_fir_config(config);
}

pub(super) fn apply_legacy_fast_overrides(
    config: &mut RoomConfig,
    maxeval: usize,
    num_filters: Option<usize>,
) {
    // Calibrated fast setup for home-cinema feature-validation cases: these
    // expectations were tuned against a cheap local optimizer, and their
    // scenario configs do not carry a tuned optimizer of their own.
    config.optimizer.algorithm = "cobyla".to_string();
    config.optimizer.max_iter = maxeval;
    config.optimizer.population = 50;
    config.optimizer.refine = false;
    config.optimizer.seed = Some(SEED);
    // See apply_qa_overrides: pin single-threaded evaluation for reproducible
    // seeded runs under case-level parallelism.
    config.optimizer.parallel_threads = Some(1);
    if let Some(num_filters) = num_filters {
        config.optimizer.num_filters = num_filters;
    }

    // Ensure FIR config exists when processing mode requires it
    ensure_fir_config(config);
}

fn ensure_fir_config(config: &mut RoomConfig) {
    match config.optimizer.processing_mode {
        ProcessingMode::PhaseLinear | ProcessingMode::Hybrid => {
            if config.optimizer.fir.is_none() {
                config.optimizer.fir = Some(FirConfig {
                    taps: 4096,
                    phase: "kirkeby".to_string(),
                    correct_excess_phase: false,
                    phase_smoothing: 0.167,
                    pre_ringing: None,
                    max_boost_db: None,
                });
            }
        }
        ProcessingMode::LowLatency
        | ProcessingMode::MixedPhase
        | ProcessingMode::WarpedIir
        | ProcessingMode::KautzModal => {}
    }
}

pub(super) static TEMP_DIR_COUNTER: AtomicUsize = AtomicUsize::new(0);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn qa_overrides_replace_auto_de_and_pin_determinism() {
        let mut config = RoomConfig::default();
        config.optimizer.algorithm = "autoeq:de".to_string();
        config.optimizer.num_filters = 7;
        config.optimizer.refine = true;
        config.optimizer.population = 60;
        config.optimizer.max_db = 12.0;
        config.optimizer.max_iter = 30000;
        config.optimizer.seed = None;

        apply_qa_overrides(&mut config, QA_MAXEVAL);

        assert_eq!(config.optimizer.algorithm, QA_ALGORITHM);
        assert_eq!(config.optimizer.num_filters, CMAES_QA_NUM_FILTERS);
        assert!(config.optimizer.refine);
        assert_eq!(config.optimizer.population, CMAES_QA_POPULATION);
        assert_eq!(config.optimizer.max_db, QA_MAX_FILTER_BOOST_DB);
        assert_eq!(config.optimizer.max_iter, QA_MAXEVAL);
        assert_eq!(config.optimizer.seed, Some(SEED));
        assert_eq!(config.optimizer.parallel_threads, Some(1));
    }

    #[test]
    fn qa_overrides_keep_smaller_scenario_budget() {
        let mut config = RoomConfig::default();
        config.optimizer.max_iter = 5000;

        apply_qa_overrides(&mut config, QA_MAXEVAL);

        assert_eq!(config.optimizer.max_iter, 5000);
    }

    #[test]
    fn qa_budget_distinguishes_de_from_cmaes() {
        assert_eq!(qa_budget_for_algorithm("autoeq:de", QA_MAXEVAL), DE_QA_MAXEVAL);
        assert_eq!(qa_budget_for_algorithm("autoeq:cmaes", QA_MAXEVAL), QA_MAXEVAL);
    }
}
