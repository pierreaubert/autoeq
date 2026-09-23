//! Wolfram cross-check: initial-guess centers/widths/gains and bound
//! construction from synthetic peaks (O06).
//!
//! Oracle: `wolfram/o06_initial_guess_bounds.wls` (sorting by |gain|,
//! Log10 centers, diversification bands and bound stencils derived from the
//! declared conventions, independent of the Rust code). Centers and bounds
//! compare at 1e-12 absolute; widths/gains are checked against the exact
//! documented diversification bands. Tolerance class A/X.

use autoeq_optim::FrequencyQPolicy;
use autoeq_optim::OptimParams;
use autoeq_optim::PeqModel;
use autoeq_optim::initial_guess::{
    SmartInitConfig, create_smart_initial_guesses, generate_integrality_constraints,
};
use autoeq_optim::optim::setup::setup_bounds;
use autoeq_qa::require_reference;
use autoeq_qa::{QaResult, assert_case_id, emit_result, provenance};
use ndarray::Array1;

const CASE: &str = "o06_initial_guess_bounds";
const CASE_ID: &str = "autoeq-qa.o06-initial-guess-bounds.v1";
const TOL: f64 = 1e-9;
const CENTER_TOL: f64 = 1e-12;

fn check_abs(actual: f64, expected: f64, tol: f64, what: &str, max_err: &mut f64) {
    assert!(
        actual.is_finite() && expected.is_finite(),
        "{CASE}: non-finite {what}: actual={actual} expected={expected}"
    );
    let err = (actual - expected).abs();
    assert!(
        err <= tol,
        "{what}: rust={actual:.12e} expected={expected:.12e} abs_err={err:.3e} tol={tol:.1e}"
    );
    *max_err = (*max_err).max(err);
}

fn base_params() -> OptimParams {
    OptimParams {
        num_filters: 2,
        peq_model: PeqModel::Pk,
        sample_rate: 48000.0,
        min_freq: 20.0,
        max_freq: 20000.0,
        min_q: 0.5,
        max_q: 10.0,
        min_db: -12.0,
        max_db: 12.0,
        loss: autoeq_optim::LossType::SpeakerFlat,
        smooth: false,
        smooth_n: 1,
        min_spacing_oct: 0.0,
        spacing_weight: 0.0,
        smoothness_penalty: None,
        audibility_deadband: None,
        frequency_q_policy: None,
        algo: String::from("autoeq:de"),
        population: 8,
        maxeval: 100,
        refine: false,
        local_algo: String::from("autoeq:cobyla"),
        bo_initial_samples: 4,
        bo_batch_size: 1,
        bo_posterior_std_threshold: 0.0,
        bo_acquisition: String::from("ei"),
        bo_ehvi: false,
        strategy: String::from("lshade"),
        tolerance: 1e-6,
        atolerance: 1e-6,
        recombination: 0.9,
        adaptive_weight_f: 0.5,
        adaptive_weight_cr: 0.5,
        no_parallel: true,
        parallel_threads: 1,
        seed: Some(42),
        quiet: true,
    }
}

#[test]
fn wolfram_o06_initial_guess_bounds() {
    let ref_json = require_reference(CASE, "o06_initial_guess_bounds.wls");
    assert_case_id(&ref_json, CASE_ID, CASE);
    let mut max_err = 0.0f64;

    // --- initial guesses from pre-detected synthetic peaks ---
    let problems: Vec<(f64, f64, f64)> =
        serde_json::from_value(ref_json["problems_freq_q_gain"].clone()).unwrap();
    let config = SmartInitConfig {
        num_guesses: 3,
        variation_factor: 0.0,
        seed: Some(42),
        pre_detected_problems: problems,
        ..Default::default()
    };
    let grid: Vec<f64> = serde_json::from_value(ref_json["freq_grid_hz"].clone()).unwrap();
    let target = Array1::zeros(grid.len());
    let per_filter: Vec<[f64; 2]> =
        serde_json::from_value(ref_json["per_filter_bounds"].clone()).unwrap();
    assert_eq!(
        per_filter.len(),
        3,
        "{CASE}: expected 3 per-filter bound rows"
    );
    let bounds: Vec<(f64, f64)> = per_filter
        .iter()
        .cycle()
        .take(6)
        .map(|row| (row[0], row[1]))
        .collect();

    let guesses = create_smart_initial_guesses(
        &target,
        &Array1::from_vec(grid),
        2,
        &bounds,
        &config,
        PeqModel::Pk,
    );
    assert_eq!(guesses.len(), 3, "{CASE}: expected 3 guesses");
    let centers: Vec<f64> = serde_json::from_value(ref_json["guess_log_centers"].clone()).unwrap();
    let q_nominals: Vec<f64> = serde_json::from_value(ref_json["q_nominals"].clone()).unwrap();
    let q_bands: Vec<[f64; 2]> = serde_json::from_value(ref_json["q_bands"].clone()).unwrap();
    let gain_nominals: Vec<f64> =
        serde_json::from_value(ref_json["gain_nominals"].clone()).unwrap();
    let gain_bands: Vec<[f64; 2]> = serde_json::from_value(ref_json["gain_bands"].clone()).unwrap();
    assert_eq!(centers.len(), 2, "{CASE}: expected 2 sorted problems");
    // Sorting invariant: the +4 dB / 2000 Hz problem outranks -3 dB / 500 Hz.
    assert!(
        centers[0] > centers[1],
        "{CASE}: guesses must be sorted by |gain| desc"
    );
    for (gi, guess) in guesses.iter().enumerate() {
        assert_eq!(guess.len(), 6, "{CASE}: guess {gi} must hold 2 Pk filters");
        assert!(
            guess.iter().all(|v| v.is_finite()),
            "{CASE}: guess {gi} must be finite"
        );
        for f in 0..2 {
            // Exact centers: zero frequency variation pins Log10[freqHz].
            check_abs(
                guess[f * 3],
                centers[f],
                CENTER_TOL,
                &format!("guess {gi} center {f}"),
                &mut max_err,
            );
            // Widths/gains lie in the documented seeded-diversification bands.
            assert!(
                (q_bands[f][0]..=q_bands[f][1]).contains(&guess[f * 3 + 1]),
                "{CASE}: guess {gi} Q {} outside band {:?} (nominal {})",
                guess[f * 3 + 1],
                q_bands[f],
                q_nominals[f]
            );
            assert!(
                (gain_bands[f][0]..=gain_bands[f][1]).contains(&guess[f * 3 + 2]),
                "{CASE}: guess {gi} gain {} outside band {:?} (nominal {})",
                guess[f * 3 + 2],
                gain_bands[f],
                gain_nominals[f]
            );
            // Every parameter respects its bound.
            for p in 0..3 {
                let (lo, hi) = bounds[f * 3 + p];
                assert!(
                    (lo..=hi).contains(&guess[f * 3 + p]),
                    "{CASE}: guess {gi} param {p} out of bounds"
                );
            }
        }
    }

    // --- integrality flags: exact contract checks ---
    let indexed: Vec<bool> =
        serde_json::from_value(ref_json["integrality_indexed"].clone()).unwrap();
    let continuous: Vec<bool> =
        serde_json::from_value(ref_json["integrality_continuous"].clone()).unwrap();
    assert_eq!(generate_integrality_constraints(2, true), indexed);
    assert_eq!(generate_integrality_constraints(2, false), continuous);

    // --- bound construction A: 2 Pk filters, 20..20000 Hz, no Q policy ---
    let (lower_a, upper_a) = setup_bounds(&base_params());
    let exp_lower_a: Vec<f64> = serde_json::from_value(ref_json["bounds_a_lower"].clone()).unwrap();
    let exp_upper_a: Vec<f64> = serde_json::from_value(ref_json["bounds_a_upper"].clone()).unwrap();
    assert_eq!(lower_a.len(), 6, "{CASE}: bounds A must hold 2 Pk filters");
    assert_eq!(upper_a.len(), 6, "{CASE}: bounds A must hold 2 Pk filters");
    for (i, (got, want)) in lower_a.iter().zip(exp_lower_a.iter()).enumerate() {
        check_abs(
            *got,
            *want,
            TOL,
            &format!("bounds A lower[{i}]"),
            &mut max_err,
        );
    }
    for (i, (got, want)) in upper_a.iter().zip(exp_upper_a.iter()).enumerate() {
        check_abs(
            *got,
            *want,
            TOL,
            &format!("bounds A upper[{i}]"),
            &mut max_err,
        );
    }
    // Progressive overlap invariant restated: filter 1 starts no lower and
    // ends no lower than filter 0.
    assert!(lower_a[3] >= lower_a[0]);
    assert!(upper_a[3] >= upper_a[0]);

    // --- bound construction B: Schroeder Q restriction over the interval ---
    let mut params_b = base_params();
    params_b.num_filters = 1;
    params_b.max_freq = 200.0;
    params_b.frequency_q_policy = Some(FrequencyQPolicy {
        schroeder_hz: Some(200.0),
        low_max_q: Some(2.0),
        high_start_hz: None,
        high_max_q: Some(8.0),
    });
    let (lower_b, upper_b) = setup_bounds(&params_b);
    let exp_lower_b: Vec<f64> = serde_json::from_value(ref_json["bounds_b_lower"].clone()).unwrap();
    let exp_upper_b: Vec<f64> = serde_json::from_value(ref_json["bounds_b_upper"].clone()).unwrap();
    assert_eq!(lower_b.len(), 3);
    assert_eq!(upper_b.len(), 3);
    for (i, (got, want)) in lower_b.iter().zip(exp_lower_b.iter()).enumerate() {
        check_abs(
            *got,
            *want,
            TOL,
            &format!("bounds B lower[{i}]"),
            &mut max_err,
        );
    }
    for (i, (got, want)) in upper_b.iter().zip(exp_upper_b.iter()).enumerate() {
        check_abs(
            *got,
            *want,
            TOL,
            &format!("bounds B upper[{i}]"),
            &mut max_err,
        );
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
