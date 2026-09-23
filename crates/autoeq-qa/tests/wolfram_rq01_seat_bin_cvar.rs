//! Wolfram cross-check: seat/bin aggregation, worst-tail mean, and
//! group-delay residual metrics (RQ01).
//!
//! Oracle: `wolfram/rq01_seat_bin_cvar.wls` (independent weighted
//! means with both documented aggregation orders, textbook CVaR
//! selection, and the constant group delay of a pure bulk delay;
//! never calls Rust code). The test calls
//! `roomeq_quality::metrics::{aggregate_seat_bin, worst_tail_mean,
//! group_delay_ms}` and checks the recorded order ids. The residual
//! group delay of the transfer is the measured quantity here;
//! induced group delay from quality bookkeeping is a separate
//! computation and is not conflated with it. Tolerance 1e-12
//! absolute on aggregates/CVaR (A/B), 1e-9 ms absolute on GD (X).

use autoeq_qa::require_reference;
use autoeq_qa::{QaResult, assert_case_id, emit_result, provenance};
use num_complex::Complex64;
use roomeq_quality::{AggregationOrder, aggregate_seat_bin, group_delay_ms, worst_tail_mean};
use std::f64::consts::PI;

const CASE: &str = "rq01_seat_bin_cvar";
const CASE_ID: &str = "autoeq-qa.rq01-seat-bin-cvar.v1";
const TOL: f64 = 1e-12;
const TOL_GD: f64 = 1e-9;

#[test]
fn wolfram_rq01_seat_bin_cvar() {
    let ref_json = require_reference(CASE, "rq01_seat_bin_cvar.wls");
    assert_case_id(&ref_json, CASE_ID, CASE);
    let values: Vec<Vec<f64>> = serde_json::from_value(ref_json["seat_values"].clone()).unwrap();
    let weights: Vec<f64> = serde_json::from_value(ref_json["bin_weights"].clone()).unwrap();
    let want_bts: f64 = serde_json::from_value(ref_json["bins_then_seats"].clone()).unwrap();
    let want_stb: f64 = serde_json::from_value(ref_json["seats_then_bins"].clone()).unwrap();

    let bts = aggregate_seat_bin(&values, &weights, AggregationOrder::BinsThenSeats).unwrap();
    let stb = aggregate_seat_bin(&values, &weights, AggregationOrder::SeatsThenBins).unwrap();
    assert!(
        (bts - want_bts).abs() <= TOL,
        "{CASE}: bins-then-seats {bts:.17e} vs {want_bts:.17e}"
    );
    assert!(
        (stb - want_stb).abs() <= TOL,
        "{CASE}: seats-then-bins {stb:.17e} vs {want_stb:.17e}"
    );
    assert_eq!(
        AggregationOrder::BinsThenSeats.as_str(),
        ref_json["bins_then_seats_id"].as_str().unwrap()
    );
    assert_eq!(
        AggregationOrder::SeatsThenBins.as_str(),
        ref_json["seats_then_bins_id"].as_str().unwrap()
    );
    // Degenerate support is None, never zero.
    assert!(aggregate_seat_bin(&[], &weights, AggregationOrder::BinsThenSeats).is_none());
    assert!(
        aggregate_seat_bin(&values, &[0.0, 0.0, 0.0], AggregationOrder::BinsThenSeats).is_none()
    );

    let cvar_values: Vec<f64> = serde_json::from_value(ref_json["cvar_values"].clone()).unwrap();
    let tail: f64 = serde_json::from_value(ref_json["tail_fraction"].clone()).unwrap();
    let want_cvar: f64 = serde_json::from_value(ref_json["cvar"].clone()).unwrap();
    let count: usize = serde_json::from_value(ref_json["cvar_count"].clone()).unwrap();
    assert_eq!(count, 2, "{CASE}: tail selection count");
    let cvar = worst_tail_mean(&cvar_values, tail);
    assert!(
        (cvar - want_cvar).abs() <= TOL,
        "{CASE}: CVaR {cvar:.17e} vs {want_cvar:.17e}"
    );

    // Pure bulk delay: residual group delay is the constant delay.
    let tau_ms: f64 = serde_json::from_value(ref_json["bulk_delay_ms"].clone()).unwrap();
    let gd_grid: Vec<f64> = serde_json::from_value(ref_json["gd_grid_hz"].clone()).unwrap();
    let want_gd: Vec<f64> = serde_json::from_value(ref_json["group_delay_ms"].clone()).unwrap();
    let transfer: Vec<Complex64> = gd_grid
        .iter()
        .map(|f| Complex64::from_polar(1.0, -2.0 * PI * f * (tau_ms / 1000.0)))
        .collect();
    let gd = group_delay_ms(&gd_grid, &transfer);
    assert_eq!(gd.len(), want_gd.len(), "{CASE}: interior GD intervals");
    let mut max_gd_err = 0.0f64;
    for (got, want) in gd.iter().zip(&want_gd) {
        let err = (got - want).abs();
        assert!(
            err <= TOL_GD,
            "{CASE}: group delay {got:.17e} vs {want:.17e}"
        );
        max_gd_err = max_gd_err.max(err);
    }

    emit_result(&QaResult {
        case: CASE_ID.to_string(),
        pass: true,
        max_rel_error: 0.0,
        max_abs_error: max_gd_err
            .max((bts - want_bts).abs())
            .max((cvar - want_cvar).abs()),
        tolerance: TOL_GD,
        tolerance_kind: "abs".to_string(),
        provenance: provenance(),
    });
}
