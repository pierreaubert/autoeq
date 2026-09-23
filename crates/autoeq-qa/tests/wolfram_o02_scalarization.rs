//! Wolfram cross-check: per-curve loss + multi-curve scalarization (O02).
//!
//! Oracle: `wolfram/o02_scalarization.wls` — per-curve flat losses from the
//! constant-residual theorem plus direct-sum Average / WeightedSum / Minimax
//! / VariancePenalized / fractional-tail CVaR reductions, including a
//! zero-weight-seat scenario. Compared through both the realized-response
//! entry (`compute_response_fitness`) and the parameter-vector entry
//! (`compute_base_fitness` with a neutral filter). Tolerance 1e-9 absolute.

use autoeq_optim::PeqModel;
use autoeq_optim::loss::LossType;
use autoeq_optim::optim::{
    MultiObjectiveData, ObjectiveDataBuilder, compute_base_fitness, compute_pareto_objectives,
    compute_response_fitness,
};
use autoeq_optim::roomeq::MultiMeasurementStrategy;
use autoeq_qa::require_reference;
use autoeq_qa::{QaResult, assert_case_id, assert_close_abs, emit_result, provenance};
use ndarray::Array1;

const CASE: &str = "o02_scalarization";
const CASE_ID: &str = "autoeq-qa.o02-scalarization.v1";
const TOL: f64 = 1e-9;

fn num(value: &serde_json::Value, key: &str) -> f64 {
    value[key]
        .as_f64()
        .unwrap_or_else(|| panic!("{CASE}: golden lacks `{key}`"))
}

fn base_curve(freqs: &Array1<f64>, deviation_db: f64) -> autoeq_optim::optim::ObjectiveData {
    let n = freqs.len();
    ObjectiveDataBuilder::new(
        freqs.clone(),
        Array1::zeros(n),
        Array1::from_elem(n, deviation_db),
        48000.0,
        PeqModel::Pk,
        LossType::SpeakerFlat,
    )
    .freq_range(20.0, 20000.0)
    .build()
    .expect("valid test objective")
}

#[test]
fn wolfram_o02_scalarization() {
    let ref_json = require_reference(CASE, "o02_scalarization.wls");
    assert_case_id(&ref_json, CASE_ID, CASE);

    let grid: Vec<f64> = serde_json::from_value(ref_json["grid_hz"].clone()).unwrap();
    assert_eq!(grid, vec![50.0, 200.0, 1000.0, 5000.0, 15000.0]);
    let freqs = Array1::from_vec(grid);
    let deviations: Vec<f64> =
        serde_json::from_value(ref_json["seat_deviations_db"].clone()).unwrap();
    assert_eq!(deviations, vec![0.0, 4.0, 10.0]);
    let per_curve: Vec<f64> = serde_json::from_value(ref_json["per_curve_losses"].clone()).unwrap();

    // Each per-curve loss independently: zero correction against a constant
    // deviation scores exactly the absolute constant (grid invariance).
    let zeros = Array1::zeros(freqs.len());
    let mut max_err = 0.0f64;
    let objectives: Vec<_> = deviations.iter().map(|d| base_curve(&freqs, *d)).collect();
    for (objective, expected) in objectives.iter().zip(per_curve.iter()) {
        let single = ObjectiveDataBuilder::new(
            freqs.clone(),
            Array1::zeros(freqs.len()),
            objective.deviation.as_ref().clone(),
            48000.0,
            PeqModel::Pk,
            LossType::SpeakerFlat,
        )
        .freq_range(20.0, 20000.0)
        .build()
        .unwrap();
        let actual = compute_response_fitness(std::slice::from_ref(&zeros), &single)
            .expect("single response fitness");
        assert_close_abs(actual, *expected, TOL, "per-curve loss");
        max_err = max_err.max((actual - *expected).abs());
    }

    // Neutral single-peak filter (gain 0 dB): identical correction, so the
    // parameter-vector entry must reproduce the same per-curve losses.
    let neutral_x = vec![500f64.log10(), 1.0, 0.0];
    for (objective, expected) in objectives.iter().zip(per_curve.iter()) {
        let actual = compute_base_fitness(&neutral_x, objective);
        assert_close_abs(actual, *expected, TOL, "parameter-vector per-curve loss");
        max_err = max_err.max((actual - *expected).abs());
    }

    let responses = vec![zeros.clone(), zeros.clone(), zeros.clone()];
    let check = |strategy: MultiMeasurementStrategy,
                 weights: Vec<f64>,
                 lambda: f64,
                 alpha: Option<f64>,
                 expected: f64,
                 what: &str| {
        let mut data = objectives[0].clone();
        data.multi_objective = Some(MultiObjectiveData {
            objectives: objectives.clone(),
            strategy,
            weights,
            variance_lambda: lambda,
            uncertainty_cvar_alpha: alpha,
        });
        let realized = compute_response_fitness(&responses, &data).expect("scalar fitness");
        assert_close_abs(realized, expected, TOL, &format!("response entry {what}"));
        let parametric = compute_base_fitness(&neutral_x, &data);
        assert_close_abs(
            parametric,
            expected,
            TOL,
            &format!("parameter entry {what}"),
        );
        (realized - expected)
            .abs()
            .max((parametric - expected).abs())
    };

    let weights = vec![0.5, 0.25, 0.25];
    max_err = max_err.max(check(
        MultiMeasurementStrategy::Average,
        weights.clone(),
        0.5,
        None,
        num(&ref_json, "average"),
        "average",
    ));
    max_err = max_err.max(check(
        MultiMeasurementStrategy::WeightedSum,
        weights.clone(),
        0.5,
        None,
        num(&ref_json, "weighted_sum"),
        "weighted-sum",
    ));
    max_err = max_err.max(check(
        MultiMeasurementStrategy::Minimax,
        weights.clone(),
        0.5,
        None,
        num(&ref_json, "minimax"),
        "minimax",
    ));
    max_err = max_err.max(check(
        MultiMeasurementStrategy::VariancePenalized,
        weights.clone(),
        0.5,
        None,
        num(&ref_json, "variance_penalized"),
        "variance-penalized",
    ));
    max_err = max_err.max(check(
        MultiMeasurementStrategy::MinimaxUncertainty,
        weights.clone(),
        0.5,
        Some(0.5),
        num(&ref_json, "cvar"),
        "fractional-tail CVaR",
    ));
    max_err = max_err.max(check(
        MultiMeasurementStrategy::MinimaxUncertainty,
        weights.clone(),
        0.5,
        None,
        num(&ref_json, "minimax"),
        "worst-case",
    ));

    // Zero-weight seat: excluded from the variance reduction, kept in the sum.
    let zero_weights = vec![0.0, 0.5, 0.5];
    max_err = max_err.max(check(
        MultiMeasurementStrategy::WeightedSum,
        zero_weights.clone(),
        0.5,
        None,
        num(&ref_json, "zero_weight_weighted_sum"),
        "zero-weight weighted-sum",
    ));
    max_err = max_err.max(check(
        MultiMeasurementStrategy::VariancePenalized,
        zero_weights.clone(),
        0.5,
        None,
        num(&ref_json, "zero_weight_variance_penalized"),
        "zero-weight variance-penalized",
    ));

    // Pareto path exposes the same per-curve losses before scalarization.
    let mut pareto_data = objectives[0].clone();
    pareto_data.multi_objective = Some(MultiObjectiveData {
        objectives: objectives.clone(),
        strategy: MultiMeasurementStrategy::Average,
        weights: weights.clone(),
        variance_lambda: 0.5,
        uncertainty_cvar_alpha: None,
    });
    let pareto = compute_pareto_objectives(&neutral_x, &pareto_data);
    assert_eq!(pareto.len(), 3, "{CASE}: one Pareto component per curve");
    for (actual, expected) in pareto.iter().zip(per_curve.iter()) {
        assert_close_abs(*actual, *expected, TOL, "pareto per-curve loss");
        max_err = max_err.max((actual - expected).abs());
    }

    emit_result(&QaResult {
        case: CASE_ID.to_string(),
        pass: true,
        max_rel_error: 0.0,
        max_abs_error: max_err,
        tolerance: TOL,
        tolerance_kind: "abs".to_string(),
        provenance: provenance(),
    });
}
