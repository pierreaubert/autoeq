//! Negative controls for the roomeq-model Wolfram cases (RM01-RM04).
//!
//! Each control loads an engine-blessed golden, applies one classic
//! defect, and asserts the resulting error exceeds the case tolerance.
//! A control that cannot fail is worthless; these must keep failing.

use autoeq_qa::golden_dir;

fn load_golden(case: &str) -> serde_json::Value {
    let path = golden_dir().join(format!("{case}.json"));
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("missing golden {}: {error}", path.display()));
    serde_json::from_str(&text).expect("golden must be JSON")
}

/// A 10*log10 (power) vs 20*log10 (amplitude) mix-up halves every SPL
/// level difference; the RM01 doubling invariant must expose it.
#[test]
fn power_vs_amplitude_db_factor_fails_spl_doubling() {
    let payload = load_golden("rm01_spl_level");
    let diffs: Vec<f64> = serde_json::from_value(payload["doubling_diffs_db"].clone()).unwrap();
    let worst = diffs
        .iter()
        .map(|d| (d / 2.0 - d).abs())
        .fold(0.0f64, f64::max);
    assert!(
        worst > 1e-4,
        "halved SPL doubling steps must exceed the 1e-4 case tolerance, got {worst:.3e}"
    );
}

/// A sign flip on the Harman tilt (rising instead of falling) must blow
/// the RM02 target tolerance, not hide inside a magnitude comparison.
#[test]
fn tilt_sign_flip_fails_target_shape() {
    let payload = load_golden("rm02_target_shape");
    let totals: Vec<f64> = serde_json::from_value(payload["total_db"].clone()).unwrap();
    let tilts: Vec<f64> = serde_json::from_value(payload["tilt_db"].clone()).unwrap();
    let worst = totals
        .iter()
        .zip(tilts.iter())
        .map(|(t, tilt)| (t - 2.0 * tilt - t).abs())
        .fold(0.0f64, f64::max);
    assert!(
        worst > 1e-9,
        "tilt sign flip must exceed the 1e-9 dB case tolerance, got {worst:.3e}"
    );
}

/// Conjugating the preference shelves (bass/treble swapped in effect)
/// must be visible: negating the treble shelf alone must move the
/// high-frequency total beyond tolerance.
#[test]
fn shelf_sign_flip_fails_target_shape() {
    let payload = load_golden("rm02_target_shape");
    let totals: Vec<f64> = serde_json::from_value(payload["total_db"].clone()).unwrap();
    let trebles: Vec<f64> = serde_json::from_value(payload["treble_db"].clone()).unwrap();
    let worst = totals
        .iter()
        .zip(trebles.iter())
        .map(|(t, treble)| (t - 2.0 * treble - t).abs())
        .fold(0.0f64, f64::max);
    assert!(
        worst > 1e-9,
        "treble-shelf sign flip must exceed the 1e-9 dB case tolerance, got {worst:.3e}"
    );
}

/// An off-by-one smoothing window (4 instead of 5) changes the smoothed
/// deviations and RMS; the RM03 auto-tune golden must expose it.
#[test]
fn smoothing_window_shift_fails_auto_tune() {
    let payload = load_golden("rm03_auto_tune");
    let smoothed: Vec<f64> =
        serde_json::from_value(payload["smoothed_deviations_db"].clone()).unwrap();
    // Dropping the first smoothed sample (window misaligned by one index)
    // must move at least one compared position beyond tolerance.
    let worst = smoothed
        .iter()
        .zip(smoothed.iter().skip(1))
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f64, f64::max);
    assert!(
        worst > 1e-9,
        "one-index smoothing shift must exceed the 1e-9 case tolerance, got {worst:.3e}"
    );
}

/// Zipping the correction band against a mismatched observation grid
/// (swapped band edges) must fail the RM03 band intersection.
#[test]
fn swapped_band_edges_fail_correction_band() {
    let payload = load_golden("rm03_band_gauss");
    let active: [f64; 2] = serde_json::from_value(payload["active_band_hz"].clone()).unwrap();
    let swapped = [active[1], active[0]];
    assert!(
        swapped[0] > swapped[1],
        "swapped band [{}, {}] must be degenerate",
        swapped[0],
        swapped[1]
    );
    let gap = (swapped[0] - active[0]).abs() + (swapped[1] - active[1]).abs();
    assert!(
        gap > 1e-9,
        "swapped band edges must exceed the 1e-9 Hz case tolerance, got {gap:.3e}"
    );
}

/// Using variance instead of std deviation in the Gaussian box (a
/// missing sqrt) must blow the truncation bounds far beyond tolerance.
#[test]
fn variance_for_sigma_fails_gauss_box() {
    let payload = load_golden("rm03_band_gauss");
    let expected: Vec<[f64; 2]> =
        serde_json::from_value(payload["gaussian"]["expected_bounds_m"].clone()).unwrap();
    let variance: Vec<f64> =
        serde_json::from_value(payload["gaussian"]["variance_m2"].clone()).unwrap();
    // Wrong box with variance in place of sigma on axis 1 (var 1.0 is a
    // fixed point, so check axis 0 where var 0.25 != sigma 0.5).
    let wrong_half_width = 3.0 * variance[0];
    let right_half_width = (expected[0][1] - expected[0][0]) / 2.0;
    let gap = (wrong_half_width - right_half_width).abs();
    assert!(
        gap > 1e-6,
        "variance-for-sigma box must exceed tolerance, got {gap:.3e}"
    );
}

/// An unwrapped phase (missing the mod 360 step) must fail the RM04
/// wrap comparison at the multi-rotation entries.
#[test]
fn unwrapped_phase_fails_wrap() {
    let payload = load_golden("rm04_phase_thresholds");
    let phases: Vec<f64> = serde_json::from_value(payload["phases_deg"].clone()).unwrap();
    let wrapped: Vec<f64> = serde_json::from_value(payload["wrapped_deg"].clone()).unwrap();
    let worst = phases
        .iter()
        .zip(wrapped.iter())
        .map(|(p, w)| (p - w).abs())
        .fold(0.0f64, f64::max);
    assert!(
        worst > 1e-9,
        "unwrapped phase must exceed the 1e-9 deg case tolerance, got {worst:.3e}"
    );
}

/// Ignoring the 0.05 dB interpolation tolerance (exact-limit
/// acceptance) flips the boundary verdicts the RM04 oracle pins.
#[test]
fn exact_limit_acceptance_fails_cancellation_boundary() {
    let payload = load_golden("rm04_phase_thresholds");
    let cases: Vec<[f64; 3]> = serde_json::from_value(payload["accept_cases"].clone()).unwrap();
    let expected: Vec<bool> = serde_json::from_value(payload["accept_expected"].clone()).unwrap();
    // Strict `candidate <= limit` wrongly rejects the exact-boundary
    // case (3.05 <= 3.0 + 0.05 is accepted by the real rule).
    let flips = cases
        .iter()
        .zip(expected.iter())
        .filter(|([c, _, l], want)| (*c <= *l) != **want)
        .count();
    assert!(
        flips > 0,
        "exact-limit acceptance must flip at least one pinned boundary verdict"
    );
}
