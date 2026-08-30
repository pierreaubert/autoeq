use super::consts::SAMPLE_RATE;
use super::consts::TEMP_DIR_COUNTER;
use super::consts::{apply_legacy_fast_overrides, apply_qa_overrides};
use super::home_cinema::validate_home_cinema_result;
use super::is::qa_primary_score_pair;
use super::misc::avg_epa_preference;
use super::processing_method::ProcessingMethod;
use super::processing_method::validate_result;
use super::test_case::TestCase;
use super::test_case::load_config_for_test;
use super::test_result::TestResult;
use anyhow::Result;
use roomeq_engine::room_result::RoomOptimizationResult;
use roomeq_model::{ChannelDspChain, CorrectionDecision, PluginConfigWrapper, RoomConfig};
use serde_json::Value;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc::channel;
use std::thread;

// The QA optimizers intentionally run with bounded evaluation budgets and can
// stop before stochastic solvers fully converge. Keep metamorphic checks tight
// enough to catch material regressions while tolerating that residual noise.
const METAMORPHIC_SCORE_TOLERANCE_FRACTION: f64 = 0.02;

pub(super) fn run_optimization(
    config: &RoomConfig,
) -> Result<(RoomOptimizationResult, PathBuf, u64)> {
    let id = TEMP_DIR_COUNTER.fetch_add(1, Ordering::Relaxed);
    let temp_dir = std::env::temp_dir().join(format!("roomeq_qa_{}_{}", std::process::id(), id));
    std::fs::create_dir_all(&temp_dir)?;
    match crate::optimize_room_with_selected_seed(config, SAMPLE_RATE, Some(&temp_dir)) {
        Ok((result, selected_seed)) => Ok((result, temp_dir, selected_seed)),
        Err(error) => {
            let _ = std::fs::remove_dir_all(&temp_dir);
            Err(error)
        }
    }
}

fn resolve_convolution_paths(result: &mut RoomOptimizationResult, output_dir: &Path) {
    for chain in result.channels.values_mut() {
        resolve_chain_convolution_paths(chain, output_dir);
    }
}

fn resolve_chain_convolution_paths(chain: &mut ChannelDspChain, output_dir: &Path) {
    for plugin in &mut chain.plugins {
        resolve_plugin_convolution_path(plugin, output_dir);
    }
    if let Some(drivers) = &mut chain.drivers {
        for driver in drivers {
            for plugin in &mut driver.plugins {
                resolve_plugin_convolution_path(plugin, output_dir);
            }
        }
    }
}

fn resolve_plugin_convolution_path(plugin: &mut PluginConfigWrapper, output_dir: &Path) {
    if plugin.plugin_type != "convolution" {
        return;
    }
    let Some(filename) = plugin.parameters.get("ir_file").and_then(Value::as_str) else {
        return;
    };
    let path = output_dir.join(filename);
    if path.is_file() {
        plugin.parameters["ir_file"] = Value::String(path.to_string_lossy().into_owned());
    }
}

fn validate_metamorphic_optimization(
    config: &RoomConfig,
    baseline: &RoomOptimizationResult,
    selected_seed: u64,
) -> Vec<String> {
    let mut failures = Vec::new();
    let (_, baseline_post) = qa_primary_score_pair(baseline, config);
    let material_tolerance =
        (baseline_post.abs() * METAMORPHIC_SCORE_TOLERANCE_FRACTION).max(1.0e-6);

    let mut higher_budget = config.clone();
    higher_budget.optimizer.seed = Some(selected_seed);
    higher_budget.optimizer.max_iter = higher_budget.optimizer.max_iter.saturating_mul(2).max(2);
    match crate::optimize_room_single_seed(&higher_budget, SAMPLE_RATE) {
        Ok(result) => {
            if !correction_reverted(&result) && optimizer_runs_converged(&result) {
                let (_, post) = qa_primary_score_pair(&result, &higher_budget);
                if post > baseline_post + material_tolerance {
                    failures.push(format!(
                        "metamorphic max_iter regression: baseline {:.4}, doubled-budget {:.4}",
                        baseline_post, post
                    ));
                }
            }
        }
        Err(error) => failures.push(format!("metamorphic max_iter run failed: {error:#}")),
    }

    let mut more_filters = higher_budget;
    more_filters.optimizer.num_filters = more_filters.optimizer.num_filters.saturating_add(2);
    // Adding filters increases CMA-ES dimensionality. Give the larger search
    // space a safer budget; this metamorphic run checks successful, accepted
    // output rather than score monotonicity across non-convex local optima.
    more_filters.optimizer.max_iter = more_filters.optimizer.max_iter.saturating_mul(2);
    match crate::optimize_room_single_seed(&more_filters, SAMPLE_RATE) {
        Ok(_) => {}
        Err(error) => failures.push(format!("metamorphic filter-count run failed: {error:#}")),
    }
    failures
}

fn correction_reverted(result: &RoomOptimizationResult) -> bool {
    result
        .metadata
        .correction_acceptance
        .as_ref()
        .is_some_and(|report| {
            matches!(
                report.decision,
                CorrectionDecision::RevertedStage | CorrectionDecision::IdentityFallback
            )
        })
}

fn optimizer_runs_converged(result: &RoomOptimizationResult) -> bool {
    result
        .metadata
        .optimizer_evidence
        .as_ref()
        .is_some_and(|evidence| {
            let mut runs = evidence.runs_by_channel.values().flatten().peekable();
            runs.peek().is_some() && runs.all(|run| run.converged)
        })
}

fn should_validate_generic_acoustics(
    is_home_cinema: bool,
    reverted: bool,
    allow_reverted: bool,
) -> bool {
    !is_home_cinema && !(reverted && allow_reverted)
}

fn rejected_identity_fallback(identity_fallback: bool, allow_reverted: bool) -> bool {
    identity_fallback && !allow_reverted
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

    if tc.home_cinema_expectations.is_some() {
        // Home-cinema cases validate workflow features (routing, alignment,
        // MSO) and their expectations were calibrated against the legacy fast
        // setup. Fast-hybrid cases keep their configured filter count; FEM
        // cases historically ran with 3 filters.
        let num_filters = if tc.solver.name() == "fast-hybrid" {
            None
        } else {
            Some(3)
        };
        apply_legacy_fast_overrides(&mut config, maxeval, num_filters);
    } else {
        apply_qa_overrides(&mut config, maxeval);
    }

    // Run optimization
    let (mut result, temp_dir, selected_seed) = match run_optimization(&config) {
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
    resolve_convolution_paths(&mut result, &temp_dir);

    let (pre, post) = if tc.home_cinema_expectations.is_some() {
        result
            .metadata
            .correction_acceptance
            .as_ref()
            .and_then(|acceptance| acceptance.acoustic_quality.as_ref())
            .map(|quality| {
                (
                    quality.training.pre_weighted_rms_median_db,
                    quality.training.post_weighted_rms_median_db,
                )
            })
            .unwrap_or((result.combined_pre_score, result.combined_post_score))
    } else {
        (result.combined_pre_score, result.combined_post_score)
    };
    let epa_pref = avg_epa_preference(&result);
    let dur = start.elapsed().as_millis() as u64;
    let reverted = correction_reverted(&result);
    let identity_fallback = result
        .metadata
        .correction_acceptance
        .as_ref()
        .is_some_and(|report| report.decision == CorrectionDecision::IdentityFallback);
    let allow_reverted = tc.expect.accepts_safe_revert()
        || tc
            .home_cinema_expectations
            .as_ref()
            .is_some_and(|expectations| expectations.allow_safe_revert);

    // Home-cinema cases use the canonical, passband-aware runtime acceptance
    // report below. The generic scalar scores mix unlike channel targets and
    // graph-routed bass, so treating them as an additional acoustic gate can
    // reject a physically accepted realization.
    let mut validation_failures = if should_validate_generic_acoustics(
        tc.home_cinema_expectations.is_some(),
        reverted,
        allow_reverted,
    ) {
        validate_result(&result, tc.expect, tc.method, &config)
    } else {
        Vec::new()
    };
    if let Some(expectations) = tc.home_cinema_expectations.clone() {
        validation_failures.extend(validate_home_cinema_result(
            &result,
            &config,
            tc.method,
            expectations,
        ));
    }
    validation_failures.extend(crate::registry::verify_result_claims(&result, &tc.claims));
    validation_failures.extend(validate_metamorphic_optimization(
        &config,
        &result,
        selected_seed,
    ));
    if rejected_identity_fallback(identity_fallback, allow_reverted) {
        let acceptance = result.metadata.correction_acceptance.as_ref();
        let violations = acceptance
            .map(|report| report.violations.join(", "))
            .filter(|violations| !violations.is_empty())
            .unwrap_or_else(|| "none reported".to_string());
        let reverted_stages = acceptance
            .map(|report| report.reverted_stages.join(", "))
            .filter(|stages| !stages.is_empty())
            .unwrap_or_else(|| "none reported".to_string());
        validation_failures.push(format!(
            "runtime acceptance reverted the proposed correction; this scenario does not declare expect.allow_safe_revert; violations: {violations}; reverted stages: {reverted_stages}"
        ));
    }

    let _ = std::fs::remove_dir_all(temp_dir);

    TestResult::success(
        &name,
        &scenario,
        &solver,
        &method,
        pre,
        post,
        epa_pref,
        validation_failures,
        reverted,
        dur,
    )
}

pub(super) fn run_parallel(
    test_cases: Vec<TestCase>,
    maxeval: usize,
    num_jobs: usize,
) -> Vec<TestResult> {
    let (tx, rx) = channel::<TestResult>();
    let worker_count = num_jobs.max(1).min(test_cases.len());
    let test_cases = Arc::new(test_cases);
    let next_case = Arc::new(AtomicUsize::new(0));
    let mut handles = Vec::with_capacity(worker_count);

    for _ in 0..worker_count {
        let tx = tx.clone();
        let test_cases = Arc::clone(&test_cases);
        let next_case = Arc::clone(&next_case);
        let handle = thread::spawn(move || {
            loop {
                let index = next_case.fetch_add(1, Ordering::Relaxed);
                let Some(test_case) = test_cases.get(index) else {
                    break;
                };
                let result = run_test_case(test_case, maxeval);
                if tx.send(result).is_err() {
                    break;
                }
            }
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
    use super::{rejected_identity_fallback, should_validate_generic_acoustics};

    #[test]
    fn safe_revert_skips_only_generic_acoustic_gate_when_explicitly_allowed() {
        assert!(!should_validate_generic_acoustics(false, true, true));
        assert!(should_validate_generic_acoustics(false, true, false));
        assert!(should_validate_generic_acoustics(false, false, true));
        assert!(!should_validate_generic_acoustics(true, false, false));
    }

    #[test]
    fn only_unapproved_identity_fallback_is_rejected() {
        assert!(!rejected_identity_fallback(false, false));
        assert!(rejected_identity_fallback(true, false));
        assert!(!rejected_identity_fallback(true, true));
    }
}

/// Execute one production-backed regression scenario through the canonical
/// RoomEQ workflow.
#[doc(hidden)]
pub fn run_regression_case(
    scenario: &str,
    method: ProcessingMethod,
    maxeval: usize,
) -> std::result::Result<(), String> {
    let test_case = TestCase {
        registry_id: format!("adhoc/{scenario}/{}", method.name()),
        scenario: scenario.to_string(),
        description: scenario.to_string(),
        solver: super::solver::Solver::Fem,
        method,
        case_name: None,
        override_file: None,
        home_cinema_expectations: None,
        claims: Vec::new(),
        expect: crate::registry::ScenarioExpect {
            improvement_min_pct: super::room_size::RoomSize::from_scenario(scenario)
                .min_improvement_pct(),
            max_post_score: super::room_size::RoomSize::from_scenario(scenario).max_post_score(),
            max_boost_db: 12.0,
            allow_safe_revert: false,
            gate_purpose: crate::registry::QaGatePurpose::Quality,
        },
    };
    let result = run_test_case(&test_case, maxeval);
    if result.passed {
        Ok(())
    } else {
        Err(result
            .error
            .unwrap_or_else(|| "coverage scenario failed without a diagnostic".to_string()))
    }
}
