//! Deterministic integration tests for DE CLI parameter propagation and behavior.

use autoeq::OptimParams;
use autoeq::PeqModel;
use autoeq::cli::Args;
use autoeq::de::Strategy;
use autoeq::optim::optimize_filters;
use autoeq::workflow::{initial_guess, setup_bounds};
use clap::Parser;
use ndarray::Array1;
use std::str::FromStr;

fn create_test_objective_data() -> autoeq::optim::ObjectiveData {
    let freqs = Array1::from(vec![
        80.0, 120.0, 250.0, 500.0, 1_000.0, 2_000.0, 5_000.0, 10_000.0,
    ]);
    let target = Array1::zeros(freqs.len());
    let deviation = Array1::from(vec![4.0, 2.0, -3.0, 5.0, -2.0, 3.5, -4.0, 1.5]);

    autoeq::optim::ObjectiveDataBuilder::speaker_flat(
        freqs,
        target,
        deviation,
        48_000.0,
        PeqModel::Pk,
    )
    .min_spacing_oct(0.2)
    .spacing_weight(20.0)
    .max_db(3.0)
    .min_db(0.5)
    .freq_range(60.0, 16_000.0)
    .smoothing(false, 3)
    .build()
    .expect("valid test objective data")
}

#[derive(Debug)]
struct DeRun {
    params: Vec<f64>,
    initial: Vec<f64>,
    fitness: f64,
}

fn run_seeded_de(mut args: Args) -> DeRun {
    args.seed = Some(0x5eed);
    args.no_parallel = true;
    args.population = 12;
    args.maxeval = 24;
    args.num_filters = 2;

    let params = OptimParams::from(&args);
    let (lower, upper) = setup_bounds(&params);
    let initial = initial_guess(&params, &lower, &upper);
    let mut x = initial.clone();
    let (_, fitness) = optimize_filters(
        &mut x,
        &lower,
        &upper,
        create_test_objective_data(),
        &params,
    )
    .expect("seeded DE optimization");

    assert!(fitness.is_finite());
    assert_eq!(x.len(), lower.len());
    assert!(
        x.iter()
            .zip(lower.iter().zip(&upper))
            .all(|(value, (lower, upper))| value.is_finite() && value >= lower && value <= upper)
    );

    DeRun {
        params: x,
        initial,
        fitness,
    }
}

#[test]
fn tolerance_parameters_propagate_to_optimizer_params() {
    let args = Args::parse_from([
        "autoeq-test",
        "--algo",
        "autoeq:de",
        "--tolerance",
        "0.0001",
        "--atolerance",
        "0.000001",
    ]);
    let params = OptimParams::from(&args);

    assert_eq!(params.tolerance, 0.0001);
    assert_eq!(params.atolerance, 0.000001);
}

#[test]
fn strategy_parameter_changes_seeded_search_trajectory() {
    let runs: Vec<DeRun> = ["best1bin", "rand1bin", "currenttobest1bin"]
        .into_iter()
        .map(|strategy| {
            assert!(Strategy::from_str(strategy).is_ok());
            run_seeded_de(Args::parse_from([
                "autoeq-test",
                "--algo",
                "autoeq:de",
                "--strategy",
                strategy,
            ]))
        })
        .collect();

    assert!(
        runs.windows(2).any(|pair| {
            pair[0].params != pair[1].params || (pair[0].fitness - pair[1].fitness).abs() > 1e-12
        }),
        "different DE strategies produced identical seeded trajectories: {runs:?}"
    );
}

#[test]
fn recombination_parameter_changes_seeded_search_trajectory() {
    let low = run_seeded_de(Args::parse_from([
        "autoeq-test",
        "--algo",
        "autoeq:de",
        "--strategy",
        "rand1bin",
        "--recombination",
        "0.1",
    ]));
    let high = run_seeded_de(Args::parse_from([
        "autoeq-test",
        "--algo",
        "autoeq:de",
        "--strategy",
        "rand1bin",
        "--recombination",
        "0.9",
    ]));

    assert!(
        low.params != high.params || (low.fitness - high.fitness).abs() > 1e-12,
        "recombination did not reach the seeded DE solver"
    );
}

#[test]
fn adaptive_strategy_produces_a_finite_bounded_candidate() {
    let run = run_seeded_de(Args::parse_from([
        "autoeq-test",
        "--algo",
        "autoeq:de",
        "--strategy",
        "adaptivebin",
        "--adaptive-weight-f",
        "0.8",
        "--adaptive-weight-cr",
        "0.7",
    ]));

    assert_ne!(
        run.params, run.initial,
        "adaptive DE should evaluate and select a candidate"
    );
}

#[test]
fn invalid_de_parameters_are_rejected_with_specific_errors() {
    let test_cases = [
        (
            vec![
                "autoeq-test",
                "--algo",
                "autoeq:de",
                "--strategy",
                "invalid",
            ],
            "Invalid DE strategy",
        ),
        (
            vec!["autoeq-test", "--algo", "autoeq:de", "--tolerance", "0"],
            "Tolerance must be > 0",
        ),
        (
            vec!["autoeq-test", "--algo", "autoeq:de", "--atolerance=-0.1"],
            "Absolute tolerance must be >= 0",
        ),
        (
            vec![
                "autoeq-test",
                "--algo",
                "autoeq:de",
                "--adaptive-weight-f",
                "1.1",
            ],
            "Adaptive weight for F must be between 0.0 and 1.0",
        ),
        (
            vec![
                "autoeq-test",
                "--algo",
                "autoeq:de",
                "--adaptive-weight-cr=-0.1",
            ],
            "Adaptive weight for CR must be between 0.0 and 1.0",
        ),
    ];

    for (args, expected_error) in test_cases {
        let error = autoeq::cli::validate_args(&Args::parse_from(args.clone()))
            .expect_err("invalid DE arguments must fail validation");
        assert!(
            error.contains(expected_error),
            "expected '{expected_error}' for {args:?}, got '{error}'"
        );
    }
}
