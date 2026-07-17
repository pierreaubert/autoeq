use super::consts::SAMPLE_RATE;
use super::consts::TEMP_DIR_COUNTER;
use super::consts::apply_qa_overrides;
use super::counting_semaphore::CountingSemaphore;
use super::misc::avg_epa_preference;
use super::processing_method::validate_result;
use super::test_case::TestCase;
use super::test_case::load_config_for_test;
use super::test_result::TestResult;
use anyhow::{Result, anyhow};
use autoeq::roomeq::{CallbackAction, RoomConfig, RoomOptimizationResult, optimize_room};
use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::sync::mpsc::channel;
use std::thread;

pub(super) fn run_optimization(config: &RoomConfig) -> Result<RoomOptimizationResult> {
    let id = TEMP_DIR_COUNTER.fetch_add(1, Ordering::Relaxed);
    let temp_dir = std::env::temp_dir().join(format!("roomeq_qa_{}_{}", std::process::id(), id));
    std::fs::create_dir_all(&temp_dir)?;
    let callback =
        Box::new(|_: &autoeq::roomeq::RoomOptimizationProgress| CallbackAction::Continue);
    let result = optimize_room(config, SAMPLE_RATE, Some(callback), Some(&temp_dir))
        .map_err(|e| anyhow!("{}", e));
    let _ = std::fs::remove_dir_all(&temp_dir);
    result
}

pub(super) fn run_test_case(tc: &TestCase, maxeval: usize) -> TestResult {
    let start = std::time::Instant::now();

    let name = tc.name();
    let scenario = tc.scenario.clone();
    let solver = tc.solver.name().to_string();
    let method = tc.method.name().to_string();

    // Check if config exists
    if !tc.config_path().exists() {
        let err = format!("Config not found: {:?}", tc.config_path());
        return TestResult::failure(
            &name,
            &scenario,
            &solver,
            &method,
            err,
            start.elapsed().as_millis() as u64,
        );
    }

    // Load and configure
    let mut config = match load_config_for_test(tc) {
        Ok((c, _)) => c,
        Err(e) => {
            return TestResult::failure(
                &name,
                &scenario,
                &solver,
                &method,
                format!("{:#}", e),
                start.elapsed().as_millis() as u64,
            );
        }
    };

    apply_qa_overrides(&mut config, maxeval);

    // Run optimization
    let result = match run_optimization(&config) {
        Ok(r) => r,
        Err(e) => {
            return TestResult::failure(
                &name,
                &scenario,
                &solver,
                &method,
                format!("{:#}", e),
                start.elapsed().as_millis() as u64,
            );
        }
    };

    let pre = result.combined_pre_score;
    let post = result.combined_post_score;
    let epa_pref = avg_epa_preference(&result);
    let dur = start.elapsed().as_millis() as u64;

    let validation_failures = validate_result(&result, tc.room_size(), tc.method, &config);

    TestResult::success(
        &name,
        &scenario,
        &solver,
        &method,
        pre,
        post,
        epa_pref,
        validation_failures,
        dur,
    )
}

pub(super) fn run_parallel(
    test_cases: Vec<TestCase>,
    maxeval: usize,
    num_jobs: usize,
) -> Vec<TestResult> {
    let (tx, rx) = channel::<TestResult>();
    let semaphore = Arc::new(CountingSemaphore::new(num_jobs));
    let mut handles: Vec<std::thread::JoinHandle<()>> = Vec::new();

    for tc in test_cases {
        let tx = tx.clone();
        let sem = Arc::clone(&semaphore);

        let handle = thread::spawn(move || {
            sem.acquire();
            let result = run_test_case(&tc, maxeval);
            sem.release();
            let _ = tx.send(result);
        });
        handles.push(handle);
    }

    drop(tx);

    let mut results = Vec::new();
    while let Ok(result) = rx.recv() {
        results.push(result);
    }

    for handle in handles {
        let _ = handle.join();
    }

    results
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::processing_method::ProcessingMethod;
    use crate::solver::Solver;

    fn test_case(scenario: &str, method: ProcessingMethod) -> TestCase {
        TestCase {
            scenario: scenario.to_string(),
            description: scenario.to_string(),
            solver: Solver::Fem,
            method,
        }
    }

    #[test]
    fn grouped_topology_modes_accept_final_realization() {
        for method in [
            ProcessingMethod::Iir,
            ProcessingMethod::Fir,
            ProcessingMethod::Mixed,
            ProcessingMethod::MixedPhase,
        ] {
            let result = run_test_case(
                &test_case("small_stereo_2_2_group", method),
                super::super::consts::QA_MAXEVAL,
            );
            assert!(
                result.passed,
                "{} failed: {}",
                method.name(),
                result.error.as_deref().unwrap_or("unknown failure")
            );
        }
    }

    #[test]
    fn mso_realization_counts_as_coverage_when_flatness_is_unchanged() {
        let result = run_test_case(
            &test_case("small_stereo_2_2_mso", ProcessingMethod::Iir),
            super::super::consts::QA_MAXEVAL,
        );
        assert!(
            result.passed,
            "MSO coverage failed: {}",
            result.error.as_deref().unwrap_or("unknown failure")
        );
    }
}
