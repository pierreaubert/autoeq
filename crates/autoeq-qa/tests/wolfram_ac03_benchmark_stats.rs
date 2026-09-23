//! Wolfram cross-check: benchmark distribution statistics (AC03).
//!
//! Oracle: `wolfram/ac03_benchmark_stats.wls` (mean, sample std,
//! linear-interpolation percentiles, finite differences, percentages
//! and tied-best masks derived from the published definitions on an
//! exact small sample).
//!
//! Visibility note: the shipped helpers (`mean_std`,
//! `percentile_sorted`, `percentage`, `finite_diff` in
//! `autoeq-cli/src/benchmark/misc.rs` and `tied_best_mask` with
//! `PAIR_TIE_EPS = 1e-6` in `autoeq-cli/src/benchmark/consts.rs`) are
//! `pub(super)`-private to the CLI binary path, so this integration
//! target cannot link them directly. The `pinned` module below is a
//! verbatim pin of those inspected bodies (same operations, same
//! constants, same edge rules); the comparison validates the
//! documented algorithm against Wolfram, while the linkage between the
//! CLI and these definitions remains covered only by the CLI's own
//! unit tests. Tolerance 1e-12 absolute (class A/B/X: exact small
//! samples).

use autoeq_qa::require_reference;
use autoeq_qa::{QaResult, assert_case_id, emit_result, provenance};

const CASE: &str = "ac03_benchmark_stats";
const CASE_ID: &str = "autoeq-qa.ac03-benchmark-stats.v1";
const TOL: f64 = 1e-12;
const TIE_EPS: f64 = 1e-6;

/// Verbatim pin of `benchmark/misc.rs::mean_std`: mean and sample
/// (n-1) std; None for n == 0; std = 0.0 for n == 1.
mod pinned {
    pub fn mean_std(data: &[f64]) -> Option<(f64, f64)> {
        let n = data.len();
        if n == 0 {
            return None;
        }
        let mean = data.iter().sum::<f64>() / (n as f64);
        if n == 1 {
            return Some((mean, 0.0));
        }
        let var_num: f64 = data
            .iter()
            .map(|&x| {
                let dx = x - mean;
                dx * dx
            })
            .sum();
        let std = (var_num / ((n - 1) as f64)).sqrt();
        Some((mean, std))
    }

    /// Verbatim pin of `benchmark/misc.rs::percentile_sorted`
    /// (linear interpolation at `q * (len - 1)`).
    pub fn percentile_sorted(sorted: &[f64], quantile: f64) -> f64 {
        assert!(!sorted.is_empty());
        if sorted.len() == 1 {
            return sorted[0];
        }
        let q = quantile.clamp(0.0, 1.0);
        let pos = q * (sorted.len() - 1) as f64;
        let lo = pos.floor() as usize;
        let hi = pos.ceil() as usize;
        if lo == hi {
            sorted[lo]
        } else {
            let weight = pos - lo as f64;
            sorted[lo] * (1.0 - weight) + sorted[hi] * weight
        }
    }

    /// Verbatim pin of `benchmark/misc.rs::finite_diff`.
    pub fn finite_diff(lhs: Option<f64>, rhs: Option<f64>) -> Option<f64> {
        match (lhs, rhs) {
            (Some(lhs), Some(rhs)) if lhs.is_finite() && rhs.is_finite() => Some(lhs - rhs),
            _ => None,
        }
    }

    /// Verbatim pin of `benchmark/misc.rs::percentage`.
    pub fn percentage(count: usize, total: usize) -> f64 {
        if total == 0 {
            0.0
        } else {
            count as f64 * 100.0 / total as f64
        }
    }

    /// Verbatim pin of `benchmark/consts.rs::tied_best_mask` (max-based,
    /// ties within `TIE_EPS`; None unless every entry is finite).
    pub fn tied_best_mask(values: [Option<f64>; 4]) -> Option<[bool; 4]> {
        let mut finite = [0.0; 4];
        for (idx, value) in values.into_iter().enumerate() {
            let value = value?;
            if !value.is_finite() {
                return None;
            }
            finite[idx] = value;
        }
        let best = finite.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        Some(finite.map(|value| (value - best).abs() <= super::TIE_EPS))
    }
}

#[test]
fn wolfram_ac03_benchmark_stats() {
    let ref_json = require_reference(CASE, "ac03_benchmark_stats.wls");
    assert_case_id(&ref_json, CASE_ID, CASE);
    let data: Vec<f64> = serde_json::from_value(ref_json["data"].clone()).unwrap();
    let mean: f64 = serde_json::from_value(ref_json["mean"].clone()).unwrap();
    let std: f64 = serde_json::from_value(ref_json["std_sample"].clone()).unwrap();
    let min: f64 = serde_json::from_value(ref_json["min"].clone()).unwrap();
    let max: f64 = serde_json::from_value(ref_json["max"].clone()).unwrap();
    let want = |key: &str| -> f64 {
        serde_json::from_value(ref_json[key].clone())
            .unwrap_or_else(|_| panic!("{CASE}: golden is missing `{key}`"))
    };
    let votes: Vec<f64> = serde_json::from_value(ref_json["tie_votes"].clone()).unwrap();
    let mask: Vec<bool> = serde_json::from_value(ref_json["tied_best_mask"].clone()).unwrap();
    assert_eq!(data.len(), 8, "{CASE}: expected 8 samples");
    assert_eq!(votes.len(), 4, "{CASE}: expected 4 tie votes");
    assert_eq!(mask.len(), 4, "{CASE}: expected 4 mask entries");

    let mut sorted = data.clone();
    sorted.sort_by(f64::total_cmp);
    let (rust_mean, rust_std) = pinned::mean_std(&sorted).expect("nonempty sample");
    let mut max_err = 0.0f64;
    for (name, got, w) in [
        ("mean", rust_mean, mean),
        ("std", rust_std, std),
        ("min", sorted[0], min),
        ("max", sorted[sorted.len() - 1], max),
        ("p10", pinned::percentile_sorted(&sorted, 0.10), want("p10")),
        ("p25", pinned::percentile_sorted(&sorted, 0.25), want("p25")),
        (
            "median",
            pinned::percentile_sorted(&sorted, 0.50),
            want("median"),
        ),
        ("p75", pinned::percentile_sorted(&sorted, 0.75), want("p75")),
        ("p90", pinned::percentile_sorted(&sorted, 0.90), want("p90")),
    ] {
        let err = (got - w).abs();
        assert!(
            err <= TOL,
            "{CASE}: {name}: rust={got:.12e} expected={w:.12e} err={err:.3e}"
        );
        max_err = max_err.max(err);
    }

    let rust_diff = pinned::finite_diff(Some(5.0), Some(4.0)).expect("finite inputs");
    let diff_err = (rust_diff - want("finite_diff_5_4")).abs();
    assert!(diff_err <= TOL, "{CASE}: finite_diff err={diff_err:.3e}");
    assert_eq!(pinned::finite_diff(Some(1.0), None), None);
    assert_eq!(pinned::finite_diff(Some(f64::NAN), Some(1.0)), None);
    max_err = max_err.max(diff_err);

    let rust_pct = pinned::percentage(2, 8);
    let pct_err = (rust_pct - want("percentage_2_of_8")).abs();
    assert!(pct_err <= TOL, "{CASE}: percentage err={pct_err:.3e}");
    assert_eq!(pinned::percentage(0, 0), 0.0);
    max_err = max_err.max(pct_err);

    let rust_mask = pinned::tied_best_mask([
        Some(votes[0]),
        Some(votes[1]),
        Some(votes[2]),
        Some(votes[3]),
    ])
    .expect("finite votes");
    assert_eq!(
        rust_mask.to_vec(),
        mask,
        "{CASE}: tied-best mask mismatch: rust={rust_mask:?} expected={mask:?}"
    );
    assert_eq!(
        pinned::tied_best_mask([Some(1.0), None, Some(2.0), Some(3.0)]),
        None,
        "{CASE}: missing vote must yield None"
    );

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
