//! Aligned magnitude-domain monitor sums for report sidecars.
//!
//! These are upper-bound magnitude sums, not phase-coherent acoustic predictions.
//! Interpolation uses dB against log frequency only within shared measured support.

use roomeq_model::{CurveData, DspGraph};

fn valid(c: &CurveData) -> bool {
    c.freq.len() >= 2
        && c.freq.len() == c.spl.len()
        && c.freq.iter().all(|f| f.is_finite() && *f > 0.0)
        && c.spl.iter().all(|s| s.is_finite())
        && c.freq.windows(2).all(|w| w[0] < w[1])
}

fn interpolate(c: &CurveData, f: f64) -> f64 {
    let i = c.freq.partition_point(|x| *x < f);
    if i == 0 {
        return c.spl[0];
    }
    if i == c.freq.len() {
        return c.spl[i - 1];
    }
    let t = (f / c.freq[i - 1]).ln() / (c.freq[i] / c.freq[i - 1]).ln();
    c.spl[i - 1] + t * (c.spl[i] - c.spl[i - 1])
}

fn pair(a: &CurveData, b: &CurveData) -> Option<serde_json::Value> {
    if !valid(a) || !valid(b) {
        return None;
    }
    let lo = a.freq[0].max(b.freq[0]);
    let hi = a.freq.last()?.min(*b.freq.last()?);
    let mut freq: Vec<f64> = a
        .freq
        .iter()
        .chain(&b.freq)
        .copied()
        .filter(|f| *f >= lo && *f <= hi)
        .collect();
    freq.sort_by(f64::total_cmp);
    freq.dedup();
    if freq.len() < 2 {
        return None;
    }
    let mut sum = Vec::with_capacity(freq.len());
    let mut difference = Vec::with_capacity(freq.len());
    for f in &freq {
        let a = interpolate(a, *f);
        let b = interpolate(b, *f);
        // Scale relative to the larger level to avoid amplitude overflow.
        let reference = a.max(b);
        let ratio = 10.0_f64.powf((a.min(b) - reference) / 20.0);
        sum.push(reference + 20.0 * (1.0 + ratio).log10());
        difference.push(if ratio == 1.0 {
            None
        } else {
            Some(reference + 20.0 * (1.0 - ratio).log10())
        });
    }
    Some(
        serde_json::json!({"freq": freq, "sum_spl": sum, "diff_spl": difference,
        "method": "magnitude_sum_log_frequency_interpolation",
        "basis": "predicted_final_curves", "coherent": false}),
    )
}

/// Compute monitor pairs before the final curves are extracted into sidecars.
pub fn pairs(output: &DspGraph) -> serde_json::Value {
    let mut reports = serde_json::Map::new();
    for (left, right) in [
        ("L", "R"),
        ("SL", "SR"),
        ("TFL", "TFR"),
        ("TBL", "TBR"),
        ("SSL", "SSR"),
        ("LBL", "LBR"),
    ] {
        let find = |key: &str| {
            output
                .channels
                .iter()
                .find(|(name, _)| name.trim().eq_ignore_ascii_case(key))
                .and_then(|(_, chain)| chain.final_curve.as_ref())
        };
        if let (Some(a), Some(b)) = (find(left), find(right))
            && let Some(report) = pair(a, b)
        {
            reports.insert(format!("{left}+{right}"), report);
        }
    }
    serde_json::Value::Object(reports)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn different_grids_sum_on_shared_support_without_normalization() {
        let a = CurveData {
            freq: vec![20., 100., 1000.],
            spl: vec![80.; 3],
            ..Default::default()
        };
        let b = CurveData {
            freq: vec![40., 200., 2000.],
            spl: vec![80.; 3],
            ..Default::default()
        };
        let report = pair(&a, &b).unwrap();
        assert_eq!(report["freq"], serde_json::json!([40., 100., 200., 1000.]));
        for value in report["sum_spl"].as_array().unwrap() {
            assert!((value.as_f64().unwrap() - 86.020599913).abs() < 1e-8);
        }
        assert!(
            report["diff_spl"]
                .as_array()
                .unwrap()
                .iter()
                .all(|v| v.is_null())
        );
        let disjoint = CurveData {
            freq: vec![2000., 3000.],
            spl: vec![80.; 2],
            ..Default::default()
        };
        assert!(pair(&a, &disjoint).is_none());
    }
}
