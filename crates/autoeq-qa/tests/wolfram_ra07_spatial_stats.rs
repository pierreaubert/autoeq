//! Wolfram cross-check: RA07 spatial means, spread, masks, bootstrap (A/B).
//!
//! Oracle: `wolfram/ra07_spatial_stats.wls` — weighted power average,
//! unbiased weighted dB spread, sigmoid correction mask (no smoothing),
//! and a bootstrap band plus CVaR over B = 6 explicit resample index
//! lists (no RNG matching). The Rust side calls the real
//! `try_rms_average_weighted` / `try_spatial_std_dev_weighted` /
//! `correction_depth_mask`, resamples manually through
//! `try_rms_average_weighted` over the golden index lists, and applies
//! the documented percentile/CVaR rules to those implementation
//! means; it also checks the deterministic single-curve
//! `bootstrap_band` collapse. Tolerance class A/B: 1e-9 absolute in dB
//! (1e-12 on the unit mask).

use autoeq_qa::require_reference;
use autoeq_qa::{QaResult, assert_case_id, emit_result, provenance};
use ndarray::Array1;
use roomeq_analysis::Curve;
use roomeq_analysis::spatial_robustness::{
    BootstrapConfig, SpatialRobustnessConfig, bootstrap_band, correction_depth_mask,
    try_rms_average_weighted, try_spatial_std_dev_weighted,
};

const CASE: &str = "ra07_spatial_stats";
const CASE_ID: &str = "autoeq-qa.ra07-spatial-stats.v1";
const TOL_DB: f64 = 1e-9;
const TOL_MASK: f64 = 1e-12;

fn vec_f64(value: &serde_json::Value) -> Vec<f64> {
    serde_json::from_value(value.clone()).expect("golden numeric array")
}

fn curve(freq: &[f64], spl: &[f64]) -> Curve {
    Curve {
        freq: Array1::from_vec(freq.to_vec()),
        spl: Array1::from_vec(spl.to_vec()),
        ..Default::default()
    }
}

fn assert_abs_vec(actual: &[f64], expected: &[f64], tol: f64, what: &str) -> f64 {
    assert_eq!(actual.len(), expected.len(), "{CASE}: {what} length zip");
    let mut worst = 0.0f64;
    for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            a.is_finite() && e.is_finite(),
            "{CASE}: {what}[{i}] non-finite: {a} vs {e}"
        );
        let err = (a - e).abs();
        assert!(
            err <= tol,
            "{what}[{i}]: rust={a:.12e} expected={e:.12e} abs_err={err:.3e}"
        );
        worst = worst.max(err);
    }
    worst
}

/// Linear-interpolation percentile on sorted values (the documented rule).
fn percentile(sorted: &[f64], q: f64) -> f64 {
    let pos = q * (sorted.len() - 1) as f64;
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;
    if lo == hi {
        sorted[lo]
    } else {
        let t = pos - lo as f64;
        sorted[lo] * (1.0 - t) + sorted[hi] * t
    }
}

#[test]
fn wolfram_ra07_spatial_stats() {
    let ref_json = require_reference(CASE, "ra07_spatial_stats.wls");
    assert_case_id(&ref_json, CASE_ID, CASE);

    let grid = vec_f64(&ref_json["grid_hz"]);
    let seats: Vec<Vec<f64>> = serde_json::from_value(ref_json["seat_spl_db"].clone()).unwrap();
    let weights = vec_f64(&ref_json["weights"]);
    assert_eq!(seats.len(), 3, "{CASE}: three seats");
    // The unbiased denominator 1 - sum(w^2) = 0.62 pins the unbiased
    // branch (floor 1/3 inactive); fail loudly if the fixture changes.
    let denom: f64 = serde_json::from_value(ref_json["spread_denominator"].clone()).unwrap();
    assert!(
        (denom - 0.62).abs() <= 1e-12,
        "{CASE}: spread denominator must be 0.62, got {denom}"
    );

    let curves: Vec<Curve> = seats.iter().map(|spl| curve(&grid, spl)).collect();

    // --- Weighted power average [A]. ---
    let avg = try_rms_average_weighted(&curves, Some(&weights)).expect("rms average");
    assert_eq!(avg.spl.len(), grid.len(), "{CASE}: grid/average length zip");
    let exp_avg = vec_f64(&ref_json["power_average_db"]);
    let mut worst = assert_abs_vec(avg.spl.as_slice().unwrap(), &exp_avg, TOL_DB, "power avg");

    // --- Weighted dB spread [A]. ---
    let std = try_spatial_std_dev_weighted(&curves, Some(&weights)).expect("spatial std");
    let exp_spread = vec_f64(&ref_json["spread_db"]);
    worst = worst.max(assert_abs_vec(
        std.as_slice().unwrap(),
        &exp_spread,
        TOL_DB,
        "spread",
    ));

    // --- Correction-depth mask, smoothing disabled [A]. ---
    let thr: f64 = serde_json::from_value(ref_json["mask_threshold_db"].clone()).unwrap();
    let wid: f64 = serde_json::from_value(ref_json["mask_transition_db"].clone()).unwrap();
    let min_d: f64 = serde_json::from_value(ref_json["mask_min_depth"].clone()).unwrap();
    let smooth: f64 = serde_json::from_value(ref_json["mask_smoothing_octaves"].clone()).unwrap();
    assert!(
        (thr, wid, min_d, smooth) == (3.0, 2.0, 0.1, 0.0),
        "{CASE}: mask config"
    );
    let config = SpatialRobustnessConfig {
        variance_threshold_db: thr,
        transition_width_db: wid,
        min_correction_depth: min_d,
        mask_smoothing_octaves: smooth,
    };
    let mask = correction_depth_mask(&avg.freq, &std, &config);
    let exp_mask = vec_f64(&ref_json["correction_mask"]);
    worst = worst.max(assert_abs_vec(
        mask.as_slice().unwrap(),
        &exp_mask,
        TOL_MASK,
        "mask",
    ));

    // --- Bootstrap over explicit index lists [B]. ---
    let idx: Vec<Vec<usize>> =
        serde_json::from_value(ref_json["resample_index_lists_1based"].clone()).unwrap();
    assert_eq!(idx.len(), 6, "{CASE}: six resample lists");
    let exp_means: Vec<Vec<f64>> =
        serde_json::from_value(ref_json["resampled_means_db"].clone()).unwrap();
    let mut means: Vec<Vec<f64>> = Vec::with_capacity(idx.len());
    for (l, draw) in idx.iter().enumerate() {
        assert_eq!(draw.len(), 3, "{CASE}: resample {l} draws 3 seats");
        let resampled: Vec<Curve> = draw.iter().map(|i| curves[i - 1].clone()).collect();
        let resampled_w: Vec<f64> = draw.iter().map(|i| weights[i - 1]).collect();
        let mean =
            try_rms_average_weighted(&resampled, Some(&resampled_w)).expect("resampled mean");
        let row: Vec<f64> = mean.spl.to_vec();
        worst = worst.max(assert_abs_vec(
            &row,
            &exp_means[l],
            TOL_DB,
            "resampled mean",
        ));
        means.push(row);
    }
    // Band arithmetic on the implementation-produced means.
    let nb = grid.len();
    let mut lower = vec![0.0; nb];
    let mut median = vec![0.0; nb];
    let mut upper = vec![0.0; nb];
    let mut std_b = vec![0.0; nb];
    let mut cvar = vec![0.0; nb];
    for b in 0..nb {
        let mut col: Vec<f64> = means.iter().map(|row| row[b]).collect();
        col.sort_by(|a, b| a.partial_cmp(b).unwrap());
        lower[b] = percentile(&col, 0.05);
        median[b] = percentile(&col, 0.5);
        upper[b] = percentile(&col, 0.95);
        let m: f64 = col.iter().sum::<f64>() / col.len() as f64;
        std_b[b] =
            (col.iter().map(|v| (v - m).powi(2)).sum::<f64>() / (col.len() - 1) as f64).sqrt();
        cvar[b] = (col[0] + col[1]) / 2.0;
    }
    worst = worst.max(assert_abs_vec(
        &lower,
        &vec_f64(&ref_json["band_lower_db"]),
        TOL_DB,
        "band lower",
    ));
    worst = worst.max(assert_abs_vec(
        &median,
        &vec_f64(&ref_json["band_median_db"]),
        TOL_DB,
        "band median",
    ));
    worst = worst.max(assert_abs_vec(
        &upper,
        &vec_f64(&ref_json["band_upper_db"]),
        TOL_DB,
        "band upper",
    ));
    worst = worst.max(assert_abs_vec(
        &std_b,
        &vec_f64(&ref_json["band_std_db"]),
        TOL_DB,
        "band std",
    ));
    worst = worst.max(assert_abs_vec(
        &cvar,
        &vec_f64(&ref_json["cvar_db"]),
        TOL_DB,
        "CVaR",
    ));

    // --- Deterministic single-curve bootstrap collapse [B/X]. ---
    let single = vec![curves[0].clone()];
    let band = bootstrap_band(
        &single,
        &BootstrapConfig {
            effective_sample_size: None,
            num_resamples: 8,
            alpha: 0.10,
            seed: 42,
        },
        None,
    )
    .expect("single-curve bootstrap");
    let exp_single = vec_f64(&ref_json["single_seat_spl_db"]);
    for (name, curve) in [
        ("lower", &band.lower),
        ("median", &band.median),
        ("upper", &band.upper),
    ] {
        worst = worst.max(assert_abs_vec(
            curve.spl.as_slice().unwrap(),
            &exp_single,
            TOL_DB,
            &format!("single-curve {name}"),
        ));
    }
    let zero = band.per_bin_std.iter().fold(0.0f64, |m, v| m.max(v.abs()));
    assert!(
        zero <= 1e-12,
        "{CASE}: single-curve band std must vanish, got {zero:.3e}"
    );
    worst = worst.max(zero);

    emit_result(&QaResult {
        case: CASE_ID.to_string(),
        pass: true,
        max_rel_error: 0.0,
        max_abs_error: worst,
        tolerance: TOL_DB,
        tolerance_kind: "abs".to_string(),
        provenance: provenance(),
    });
}
