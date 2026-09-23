//! Negative controls for the app cross-validation group (AW01, PL01,
//! PL02, AC01, AC02, AC03).
//!
//! Each control loads an engine-blessed golden, applies one classic
//! defect to a copy of the reference values, and asserts the resulting
//! error exceeds the case tolerance. A control that cannot fail is
//! worthless; these must keep failing.

use autoeq_qa::golden_dir;

fn load_golden(case: &str) -> serde_json::Value {
    let path = golden_dir().join(format!("{case}.json"));
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("missing golden {}: {error}", path.display()));
    serde_json::from_str(&text).expect("golden must be JSON")
}

fn vec_of(payload: &serde_json::Value, key: &str) -> Vec<f64> {
    serde_json::from_value(payload[key].clone())
        .unwrap_or_else(|_| panic!("golden is missing `{key}`"))
}

/// AW01: applying the correction with the wrong sign (subtracting the
/// EQ instead of adding it) must blow the 1e-6 dB trace tolerance.
#[test]
fn sign_flipped_correction_fails_aw01_tolerance() {
    let payload = load_golden("aw01_workflow_correction");
    let input = vec_of(&payload, "input_spl_db");
    let eq = vec_of(&payload, "eq_response_db");
    let corrected = vec_of(&payload, "corrected_spl_db");
    let worst = input
        .iter()
        .zip(eq.iter())
        .zip(corrected.iter())
        .map(|((a, b), want)| (a - b - want).abs())
        .fold(0.0f64, f64::max);
    assert!(
        worst > 1e-6,
        "sign-flipped AW01 correction must exceed the 1e-6 dB tolerance, got {worst:.3e}"
    );
}

/// PL01: fitting against log10(f) instead of log2(f) scales the slope
/// by ln(2)/ln(10); the recovered slope must miss the 1e-9 tolerance.
#[test]
fn log10_axis_slope_fails_pl01_tolerance() {
    let payload = load_golden("pl01_tonal_regression");
    let slope: f64 = serde_json::from_value(payload["slope_db_per_oct"].clone()).unwrap();
    let wrong = slope * std::f64::consts::LN_2 / std::f64::consts::LN_10;
    let err = (wrong - slope).abs();
    assert!(
        err > 1e-9,
        "log10-axis PL01 slope must exceed the 1e-9 tolerance, got {err:.3e}"
    );
}

/// PL02: a 10*log10 (power) vs 20*log10 (amplitude) mix-up halves every
/// dB trace value; the combined EQ trace must expose that factor.
#[test]
fn power_vs_amplitude_db_factor_fails_pl02_trace() {
    let payload = load_golden("pl02_plot_response_trace");
    let eq = vec_of(&payload, "eq_response_db");
    let worst = eq
        .iter()
        .map(|v| (v - 0.5 * v).abs())
        .fold(0.0f64, f64::max);
    assert!(
        worst > 1e-6,
        "10-vs-20 dB factor must be visible in PL02 traces, got {worst:.3e} dB"
    );
}

/// AC01: octave spacings computed with natural log instead of log2 are
/// off by a factor of ln(2); the spacing check must catch that.
#[test]
fn natural_log_spacing_fails_ac01_tolerance() {
    let payload = load_golden("ac01_cli_spacing_score");
    let spacings = vec_of(&payload, "adjacent_spacings_oct");
    let worst = spacings
        .iter()
        .map(|s| (s * std::f64::consts::LN_2 - s).abs())
        .fold(0.0f64, f64::max);
    assert!(
        worst > 1e-12,
        "natural-log AC01 spacings must exceed the 1e-12 oct tolerance, got {worst:.3e}"
    );
}

/// AC01 (score half): replacing the population SD (n denominator, as
/// published) with the sample SD (n-1) must move the headphone score
/// beyond its 1e-9-point tolerance.
#[test]
fn sample_sd_score_fails_ac01_tolerance() {
    let payload = load_golden("ac01_cli_spacing_score");
    let dev = vec_of(&payload, "headphone_deviation_db");
    let score: f64 = serde_json::from_value(payload["headphone_score"].clone()).unwrap();
    let slope: f64 = serde_json::from_value(payload["headphone_slope_db_per_oct"].clone()).unwrap();
    let n = dev.len() as f64;
    let mean = dev.iter().sum::<f64>() / n;
    let pop_var = dev.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n;
    let sample_sd = (pop_var * n / (n - 1.0)).sqrt();
    let pop_sd = pop_var.sqrt();
    let wrong = 114.49 - 12.62 * sample_sd - 15.52 * slope.abs();
    let err = (wrong - score).abs();
    assert!(
        pop_sd > 0.0,
        "fixture needs nonzero spread for this control"
    );
    assert!(
        err > 1e-9,
        "sample-SD AC01 score must exceed the 1e-9 tolerance, got {err:.3e}"
    );
}

/// AC02: rebuilding the delivered response with the unrounded center
/// (997.3 Hz instead of the serialized 997 Hz) must disagree with the
/// after-rounding reference beyond the trace tolerance — i.e. the
/// rounding step is load-bearing and cannot be skipped.
#[test]
fn unrounded_rebuild_fails_ac02_tolerance() {
    let payload = load_golden("ac02_apo_roundtrip_gap");
    let before = vec_of(&payload, "eq_before_db");
    let after = vec_of(&payload, "eq_after_db");
    let worst = before
        .iter()
        .zip(after.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f64, f64::max);
    assert!(
        worst > 1e-6,
        "unrounded AC02 rebuild must differ from the rounded reference beyond 1e-6 dB, got {worst:.3e}"
    );
}

/// AC02 (artifact half): a one-Hz typo in the delivered APO line must
/// fail the exact line identity.
#[test]
fn off_by_one_fc_fails_ac02_line_identity() {
    let payload = load_golden("ac02_apo_roundtrip_gap");
    let line: String = serde_json::from_value(payload["apo_filter_line"].clone()).unwrap();
    let wrong = line.replace("997 Hz", "998 Hz");
    assert_ne!(wrong, line, "control needs a distinct mutated line");
    assert!(
        !wrong.contains("997 Hz"),
        "mutated AC02 line must no longer carry the delivered center"
    );
}

/// AC03: population std (n denominator) instead of the documented
/// sample std (n-1) must exceed the 1e-12 arithmetic tolerance.
#[test]
fn population_std_fails_ac03_tolerance() {
    let payload = load_golden("ac03_benchmark_stats");
    let data: Vec<f64> = serde_json::from_value(payload["data"].clone()).unwrap();
    let std: f64 = serde_json::from_value(payload["std_sample"].clone()).unwrap();
    let n = data.len() as f64;
    let mean = data.iter().sum::<f64>() / n;
    let pop = (data.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n).sqrt();
    let err = (pop - std).abs();
    assert!(
        err > 1e-12,
        "population-std AC03 value must exceed the 1e-12 tolerance, got {err:.3e}"
    );
}

/// AC03 (percentile half): nearest-rank selection instead of linear
/// interpolation must move the median/p90 beyond tolerance.
#[test]
fn nearest_rank_percentile_fails_ac03_tolerance() {
    let payload = load_golden("ac03_benchmark_stats");
    let data: Vec<f64> = serde_json::from_value(payload["data"].clone()).unwrap();
    let mut sorted = data.clone();
    sorted.sort_by(f64::total_cmp);
    let median: f64 = serde_json::from_value(payload["median"].clone()).unwrap();
    let nearest = sorted[(0.5 * sorted.len() as f64).round() as usize];
    let err = (nearest - median).abs();
    assert!(
        err > 1e-12,
        "nearest-rank AC03 median must exceed the 1e-12 tolerance, got {err:.3e}"
    );
}

/// AC03 (tie half): a loose 0.25 tie epsilon would crown the 6.9 vote
/// alongside the true best; the pinned 1e-6 mask must exclude it.
#[test]
fn loose_tie_eps_fails_ac03_mask() {
    let payload = load_golden("ac03_benchmark_stats");
    let votes: Vec<f64> = serde_json::from_value(payload["tie_votes"].clone()).unwrap();
    let mask: Vec<bool> = serde_json::from_value(payload["tied_best_mask"].clone()).unwrap();
    let best = votes.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let loose: Vec<bool> = votes.iter().map(|v| (v - best).abs() <= 0.25).collect();
    assert_ne!(
        loose, mask,
        "loose-eps AC03 mask must differ from the pinned mask"
    );
    assert!(
        !mask[2],
        "pinned AC03 mask must exclude the 6.9 vote, got {mask:?}"
    );
}
