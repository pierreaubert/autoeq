//! Wolfram cross-check: hybrid grid, delay fit, phase restore (RW11).
//!
//! Oracle: `wolfram/rw11_hybrid_grid_delay.wls` (independently derived
//! hybrid grid + clipping, OLS bulk-delay fit from the linear-phase
//! definition, delay-restored phase). The Rust test calls the real
//! `roomeq_analysis::frequency_grid` and `autoeq_core::phase_utils`
//! kernels the measurement reduction uses. Grid 1e-12 relative
//! (class N); delay/residual/restored absolute in ms/degrees
//! (class N/A with Hilbert interpolation budget).

use autoeq_core::Curve;
use autoeq_core::phase_utils::{
    compute_excess_phase, estimate_delay_from_excess_phase, reconstruct_minimum_phase,
    unwrap_phase_degrees,
};
use autoeq_qa::require_reference;
use autoeq_qa::{QaResult, assert_case_id, emit_result, provenance};
use ndarray::Array1;
use roomeq_analysis::frequency_grid::{
    clipped_room_eq_frequency_grid, room_eq_hybrid_frequency_grid,
};

const CASE: &str = "rw11_hybrid_grid_delay";
const CASE_ID: &str = "autoeq-qa.rw11-hybrid-grid-delay.v1";
const TOL_GRID: f64 = 1e-12;
const TOL_DELAY_MS: f64 = 1e-9;
const TOL_DEG: f64 = 1e-9;

fn max_abs_diff(a: &Array1<f64>, b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len(), "{CASE}: length mismatch");
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f64, f64::max)
}

fn max_rel_diff(a: &Array1<f64>, b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len(), "{CASE}: length mismatch");
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| {
            assert!(y.is_finite(), "{CASE}: non-finite reference");
            if *x == *y { 0.0 } else { ((x - y) / y).abs() }
        })
        .fold(0.0f64, f64::max)
}

#[test]
fn wolfram_rw11_hybrid_grid_delay() {
    let ref_json = require_reference(CASE, "rw11_hybrid_grid_delay.wls");
    assert_case_id(&ref_json, CASE_ID, CASE);
    let samples: usize = serde_json::from_value(ref_json["frequency_samples"].clone()).unwrap();
    let full_grid: Vec<f64> = serde_json::from_value(ref_json["full_grid_hz"].clone()).unwrap();
    let clipped_grid: Vec<f64> =
        serde_json::from_value(ref_json["clipped_grid_hz"].clone()).unwrap();
    let span: Vec<f64> = serde_json::from_value(ref_json["source_span_hz"].clone()).unwrap();
    let d_freqs: Vec<f64> = serde_json::from_value(ref_json["delay_freqs_hz"].clone()).unwrap();
    let excess: Vec<f64> = serde_json::from_value(ref_json["excess_phase_deg"].clone()).unwrap();
    let expected_delay: f64 =
        serde_json::from_value(ref_json["expected_delay_ms"].clone()).unwrap();
    let expected_residual: Vec<f64> =
        serde_json::from_value(ref_json["expected_residual_deg"].clone()).unwrap();
    let expected_restored: Vec<f64> =
        serde_json::from_value(ref_json["expected_restored_deg"].clone()).unwrap();
    let tau_ms: f64 = serde_json::from_value(ref_json["delay_tau_ms"].clone()).unwrap();

    let rust_full = room_eq_hybrid_frequency_grid(samples);
    let grid_err = max_rel_diff(&rust_full, &full_grid);
    assert!(
        grid_err <= TOL_GRID,
        "{CASE}: hybrid grid rel_err={grid_err:.3e} tol={TOL_GRID:.1e}"
    );

    let dense: Vec<f64> = {
        let mut v = Vec::new();
        let mut f = span[0];
        while f < span[1] {
            v.push(f);
            f += 10.0;
        }
        v.push(span[1]);
        v
    };
    let curve = Curve {
        freq: Array1::from_vec(dense.clone()),
        spl: Array1::from_elem(dense.len(), 80.0),
        ..Default::default()
    };
    let rust_clipped =
        clipped_room_eq_frequency_grid(&curve, samples).expect("valid dense curve must clip");
    let clip_err = max_rel_diff(&rust_clipped, &clipped_grid);
    assert!(
        clip_err <= TOL_GRID,
        "{CASE}: clipped grid rel_err={clip_err:.3e} tol={TOL_GRID:.1e}"
    );

    let freq_arr = Array1::from_vec(d_freqs.clone());
    let flat = Array1::from_elem(d_freqs.len(), 80.0);
    let min_phase = reconstruct_minimum_phase(&freq_arr, &flat);
    let min_err = min_phase.iter().map(|v| v.abs()).fold(0.0f64, f64::max);
    assert!(
        min_err <= TOL_DEG,
        "{CASE}: flat-magnitude min phase max={min_err:.3e} deg tol={TOL_DEG:.1e}"
    );

    let wrapped: Vec<f64> = excess
        .iter()
        .map(|p| ((p + 180.0).rem_euclid(360.0)) - 180.0)
        .collect();
    let unwrapped = unwrap_phase_degrees(&Array1::from_vec(wrapped));
    let unwrap_err = max_abs_diff(&unwrapped, &excess);
    assert!(
        unwrap_err <= TOL_DEG,
        "{CASE}: unwrap abs_err={unwrap_err:.3e} deg"
    );

    let total = &unwrapped + &min_phase;
    let rust_excess = compute_excess_phase(&total, &min_phase);
    let (delay_ms, residual) = estimate_delay_from_excess_phase(&freq_arr, &rust_excess);
    let delay_err = (delay_ms - expected_delay).abs();
    assert!(
        delay_err <= TOL_DELAY_MS,
        "{CASE}: delay rust={delay_ms:.12e} expected={expected_delay:.12e}"
    );
    let resid_err = max_abs_diff(&residual, &expected_residual);
    assert!(
        resid_err <= TOL_DEG,
        "{CASE}: residual abs_err={resid_err:.3e} deg"
    );

    let restored = Array1::from_iter(
        freq_arr
            .iter()
            .zip(min_phase.iter())
            .zip(residual.iter())
            .map(|((f, m), e)| m + e - 360.0 * f * tau_ms / 1000.0),
    );
    let restore_err = max_abs_diff(&restored, &expected_restored);
    assert!(
        restore_err <= TOL_DEG,
        "{CASE}: restored-phase abs_err={restore_err:.3e} deg"
    );

    emit_result(&QaResult {
        case: CASE_ID.to_string(),
        pass: true,
        max_rel_error: grid_err.max(clip_err),
        max_abs_error: delay_err.max(resid_err).max(restore_err).max(min_err),
        tolerance: TOL_DEG,
        tolerance_kind: "abs".to_string(),
        provenance: provenance(),
    });
}
