//! Wolfram cross-check: regularized 2x2 CTC solve and FIR delivery (RE23).
//!
//! Oracle: `wolfram/re23_ctc_regularized.wls` (independent closed-form
//! per-bin Tikhonov normal-equation solve with the frequency-dependent
//! beta branches, exact bulk-delay/mirror/IDFT FIR synthesis, forward-DFT
//! delivered-response metrics). Compares the four realized FIRs, the
//! ideal reconstruction error, filter/sum headroom gains and the
//! delivered residual/balance metrics (catalogue class L/I).

use autoeq_qa::require_reference;
use autoeq_qa::{QaResult, assert_case_id, emit_result, provenance};
use num_complex::Complex64;
use roomeq_engine::ctc::{
    amplitude_to_db, beta_for_frequency, build_matrix_spectrum, enforce_electrical_sum_headroom,
    solve_prepared_ctc,
};
use roomeq_model::{CtcConfig, CtcRegularizationConfig, CtcWindowConfig};

const CASE: &str = "re23_ctc_regularized";
const CASE_ID: &str = "autoeq-qa.re23-ctc-regularized.v1";
const TOL: f64 = 1e-9;

fn check_abs(actual: f64, expected: f64, what: &str, max_err: &mut f64) {
    assert!(
        actual.is_finite() && expected.is_finite(),
        "{CASE}: non-finite {what}: actual={actual} expected={expected}"
    );
    let err = (actual - expected).abs();
    assert!(
        err <= TOL,
        "{what}: rust={actual:.12e} expected={expected:.12e} abs_err={err:.3e}"
    );
    *max_err = (*max_err).max(err);
}

#[test]
fn wolfram_re23_ctc_regularized() {
    let ref_json = require_reference(CASE, "re23_ctc_regularized.wls");
    assert_case_id(&ref_json, CASE_ID, CASE);
    let mut max_err = 0.0f64;

    // Frequency-dependent beta branches, exactly as implemented.
    let config = CtcConfig {
        enabled: true,
        matrix_source: "synthetic".to_string(),
        measurements: None,
        hrtf: None,
        window: CtcWindowConfig::default(),
        regularization: CtcRegularizationConfig {
            beta_db: -40.0,
            beta_lf_db: -30.0,
            beta_hf_db: -20.0,
            max_gain_db: 24.0,
        },
        robustness: "ct".to_string(),
        include_room_eq_dsp: false,
        fir_taps: 16,
        reference_sweep: None,
        sweep_duration_s: None,
        sweep_start_hz: None,
        sweep_end_hz: None,
        harmonic_suppression_harmonics: 8,
        harmonic_suppression_window_ms: 10.0,
        minimax_iterations: 32,
    };
    let betas: Vec<f64> = serde_json::from_value(ref_json["bin_betas"].clone()).unwrap();
    assert_eq!(betas.len(), 9, "{CASE}: expected 9 bins");
    for (k, want) in betas.iter().enumerate() {
        let got = beta_for_frequency(&config, k as f64 * 48000.0 / 16.0);
        check_abs(got, *want, &format!("beta[bin {k}]"), &mut max_err);
    }

    // Flat 2x2 plant, row-major ears x speakers, identical in every bin.
    let plant: Vec<[f64; 2]> = serde_json::from_value(ref_json["plant_re_im"].clone()).unwrap();
    assert_eq!(plant.len(), 4);
    let h = |i: usize| Complex64::new(plant[i][0], plant[i][1]);
    let flat_bin = |c: Complex64| vec![c; 9];
    let spectra = vec![vec![
        [flat_bin(h(0)), flat_bin(h(2))],
        [flat_bin(h(1)), flat_bin(h(3))],
    ]];
    let spectrum = build_matrix_spectrum(
        "synthetic".to_string(),
        vec!["L".to_string(), "R".to_string()],
        vec!["L".to_string(), "R".to_string()],
        vec!["pos0".to_string()],
        spectra,
        9,
    );
    let solved = solve_prepared_ctc(&spectrum, &config, 48000.0).unwrap();

    // Four realized FIRs, indexed [speaker][target ear][tap].
    let want_taps: Vec<Vec<Vec<f64>>> =
        serde_json::from_value(ref_json["fir_taps_row_major"].clone()).unwrap();
    assert_eq!(solved.filters.len(), 4, "{CASE}: expected four FIR paths");
    assert_eq!(solved.latency_samples, 8);
    check_abs(
        solved.latency_ms,
        8.0 * 1000.0 / 48000.0,
        "latency_ms",
        &mut max_err,
    );
    for (k, filter) in solved.filters.iter().enumerate() {
        assert_eq!(filter.taps.len(), 16);
        for (n, tap) in filter.taps.iter().enumerate() {
            let want = want_taps[k / 2][k % 2][n];
            let err = (tap - want).abs();
            assert!(
                err <= TOL,
                "{CASE}: taps[{k}][{n}]: rust={tap:.12e} expected={want:.12e} abs_err={err:.3e}"
            );
            max_err = max_err.max(err);
        }
    }

    check_abs(
        solved.max_filter_gain_db,
        ref_json["max_filter_gain_db"].as_f64().unwrap(),
        "max_filter_gain_db",
        &mut max_err,
    );
    check_abs(
        solved.max_electrical_sum_gain_db,
        ref_json["max_electrical_sum_gain_db"].as_f64().unwrap(),
        "max_electrical_sum_gain_db",
        &mut max_err,
    );
    check_abs(
        solved.mean_reconstruction_error,
        ref_json["mean_reconstruction_error"].as_f64().unwrap(),
        "mean_reconstruction_error",
        &mut max_err,
    );
    assert!(
        (solved.worst_position_error
            - ref_json["recon_error_per_bin"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_f64().unwrap())
                .fold(0.0f64, f64::max))
        .abs()
            <= TOL,
        "{CASE}: worst_position_error must be the worst per-bin error"
    );
    check_abs(
        solved.mean_crosstalk_residual_db,
        ref_json["mean_crosstalk_db"].as_f64().unwrap(),
        "mean_crosstalk_db",
        &mut max_err,
    );
    check_abs(
        solved.delivered_response.mean_crosstalk_db,
        ref_json["mean_crosstalk_db"].as_f64().unwrap(),
        "delivered mean_crosstalk_db",
        &mut max_err,
    );
    check_abs(
        solved.delivered_response.worst_crosstalk_db,
        ref_json["worst_crosstalk_db"].as_f64().unwrap(),
        "delivered worst_crosstalk_db",
        &mut max_err,
    );
    check_abs(
        solved.delivered_response.mean_target_error,
        ref_json["mean_target_error"].as_f64().unwrap(),
        "delivered mean_target_error",
        &mut max_err,
    );
    check_abs(
        solved.delivered_response.worst_target_error,
        ref_json["worst_target_error"].as_f64().unwrap(),
        "delivered worst_target_error",
        &mut max_err,
    );
    check_abs(
        solved.delivered_response.mean_channel_balance_db,
        ref_json["mean_channel_balance_db"].as_f64().unwrap(),
        "delivered mean_channel_balance_db",
        &mut max_err,
    );
    // No headroom limiting engaged on this well-conditioned fixture.
    assert!(
        !solved.driver_headroom_limited,
        "{CASE}: fixture must not be headroom-limited"
    );

    // Helper behaviour: dB floors, exact beta values, headroom scaling.
    check_abs(
        amplitude_to_db(0.0),
        -240.0,
        "amplitude_to_db(0)",
        &mut max_err,
    );
    check_abs(
        amplitude_to_db(2.0),
        6.020599913279624,
        "amplitude_to_db(2)",
        &mut max_err,
    );
    let mut over = vec![Complex64::new(200.0, 0.0), Complex64::new(200.0, 0.0)];
    assert!(enforce_electrical_sum_headroom(&mut over, 1, 2, 6.0));
    let cap = 10.0f64.powf(6.0 / 20.0);
    check_abs(over[0].norm(), cap / 2.0, "headroom row cap", &mut max_err);
    let mut under = vec![Complex64::new(0.5, 0.0), Complex64::new(0.5, 0.0)];
    assert!(!enforce_electrical_sum_headroom(&mut under, 1, 2, 6.0));
    check_abs(under[0].re, 0.5, "headroom passthrough", &mut max_err);

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
