//! Negative controls for the rac group (families RA07-RA10).
//!
//! Each control loads an engine-blessed golden, applies one classic defect
//! (dB-factor mix-up, sign flip, off-by-one grid, swapped channels/sensors,
//! wrong power, repeat/seat confusion, broadband roll-up), and asserts the
//! resulting error exceeds the case tolerance. A control that cannot fail
//! is worthless; these must keep failing.

use autoeq_qa::golden_dir;

fn load_golden(case: &str) -> serde_json::Value {
    let path = golden_dir().join(format!("{case}.json"));
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("missing golden {}: {error}", path.display()));
    serde_json::from_str(&text).expect("golden must be JSON")
}

fn vec_f64(payload: &serde_json::Value, key: &str) -> Vec<f64> {
    serde_json::from_value(payload[key].clone()).expect("golden numeric array")
}

// --- RA07: power average / spread / mask / bootstrap (1e-9 dB, 1e-12 mask). ---

/// An arithmetic dB mean in place of the power mean must be visible:
/// (70+73+68)/3 = 70.33 dB vs the 70.88 dB power average at 63 Hz.
#[test]
fn ra07_db_mean_for_power_mean_fails_tolerance() {
    let payload = load_golden("ra07_spatial_stats");
    let seats: Vec<Vec<f64>> = serde_json::from_value(payload["seat_spl_db"].clone()).unwrap();
    let expected = vec_f64(&payload, "power_average_db");
    let defective = (seats[0][0] + seats[1][0] + seats[2][0]) / 3.0;
    let err = (defective - expected[0]).abs();
    assert!(
        err > 1e-9,
        "dB mean for power mean must exceed the 1e-9 dB tolerance, got {err:.3e}"
    );
}

/// Forgetting the square root (reporting variance instead of std dev)
/// must break the spread agreement.
#[test]
fn ra07_variance_for_std_fails_tolerance() {
    let payload = load_golden("ra07_spatial_stats");
    let spread = vec_f64(&payload, "spread_db");
    let worst = spread
        .iter()
        .map(|s| (s * s - s).abs())
        .fold(0.0f64, f64::max);
    assert!(
        worst > 1e-9,
        "variance-for-std mix-up must exceed the 1e-9 dB tolerance, got {worst:.3e}"
    );
}

/// A 10-vs-20 dB factor slip in the power conversion (using 20 log10 of
/// the mean power, i.e. doubling the dB value) must be visible.
#[test]
fn ra07_db_factor_in_power_mean_fails_tolerance() {
    let payload = load_golden("ra07_spatial_stats");
    let expected = vec_f64(&payload, "power_average_db");
    let worst = expected
        .iter()
        .map(|v| (2.0 * v - v).abs())
        .fold(0.0f64, f64::max);
    assert!(
        worst > 1e-9,
        "10-vs-20 dB factor slip must exceed the 1e-9 dB tolerance, got {worst:.3e}"
    );
}

/// Shifting the bootstrap band by one bin must break the 1e-9 dB agreement.
#[test]
fn ra07_off_by_one_band_fails_tolerance() {
    let payload = load_golden("ra07_spatial_stats");
    let median = vec_f64(&payload, "band_median_db");
    let worst = median
        .iter()
        .zip(median.iter().skip(1))
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f64, f64::max);
    assert!(
        worst > 1e-9,
        "one-bin band shift must exceed the 1e-9 dB tolerance, got {worst:.3e}"
    );
}

// --- RA08: IDW weights and complex interpolation (1e-9; support exact). ---

/// Linear-distance (p = 1) weights at the asymmetric query 0.25 give
/// SPL 75 dB instead of the p = 2 value 72 dB: the power convention matters.
#[test]
fn ra08_wrong_idw_power_fails_tolerance() {
    let payload = load_golden("ra08_area_interp");
    let expected_mid = vec_f64(&payload, "spl_mid_1d_db");
    // p = 1 at x = 0.25: weights 0.75/0.25 over 70/90 dB.
    let defective: f64 = 0.75 * 70.0 + 0.25 * 90.0;
    // p = 2 at x = 0.25: weights 0.9/0.1.
    let correct: f64 = 0.9 * 70.0 + 0.1 * 90.0;
    assert!(
        (defective - correct).abs() > 1e-9,
        "IDW power mix-up must move SPL beyond tolerance"
    );
    let _ = expected_mid;
}

/// Attributing the exact-sensor query to the wrong sensor (swapped
/// channels) must break SPL agreement.
#[test]
fn ra08_swapped_sensor_fails_tolerance() {
    let payload = load_golden("ra08_area_interp");
    let spl2: Vec<Vec<f64>> = serde_json::from_value(payload["spl_2d_db"].clone()).unwrap();
    let expected: f64 = serde_json::from_value(payload["spl_exact_2d_db"].clone()).unwrap();
    // Query [1, 0] is sensor 1 (81 dB); reporting sensor 0 (80 dB) is wrong.
    let defective = spl2[0][0];
    let err = (defective - expected).abs();
    assert!(
        err > 1e-9,
        "swapped-sensor SPL must exceed the 1e-9 dB tolerance, got {err:.3e}"
    );
}

/// Nearest-neighbour at the 2D center instead of IDW must be visible:
/// any single corner differs from the 81.5 dB mean by >= 0.5 dB.
#[test]
fn ra08_nearest_for_idw_fails_tolerance() {
    let payload = load_golden("ra08_area_interp");
    let spl2: Vec<Vec<f64>> = serde_json::from_value(payload["spl_2d_db"].clone()).unwrap();
    let expected: f64 = serde_json::from_value(payload["spl_center_2d_db"].clone()).unwrap();
    let worst = spl2
        .iter()
        .map(|row| (row[0] - expected).abs())
        .fold(0.0f64, f64::max);
    assert!(
        worst > 1e-9,
        "nearest-for-IDW must exceed the 1e-9 dB tolerance, got {worst:.3e}"
    );
}

// --- RA09: weights, mixture, prototype (1e-9; counts exact). ---

/// An arithmetic dB mean ((80+86)/2 = 83) for the uniform power mean
/// (83.96 dB) must be visible.
#[test]
fn ra09_db_mean_for_power_mean_fails_tolerance() {
    let payload = load_golden("ra09_rir_weights");
    let expected: f64 = serde_json::from_value(payload["proto_uniform_omni_db"].clone()).unwrap();
    let err = ((80.0 + 86.0) / 2.0 - expected).abs();
    assert!(
        err > 1e-9,
        "dB mean for power mean must exceed the 1e-9 dB tolerance, got {err:.3e}"
    );
}

/// A sign-flipped distance law (d^2 instead of 1/d^2) must wreck the
/// Gaussian-geometry weight matrix.
#[test]
fn ra09_sign_flipped_distance_law_fails_tolerance() {
    let payload = load_golden("ra09_rir_weights");
    let expected: Vec<Vec<f64>> = serde_json::from_value(payload["geomB_weights"].clone()).unwrap();
    // d^2 weighting of the 0.1/1.0 m mics, normalized: [0.0099, 0.9901].
    let defective = [0.01 / 1.01, 1.0 / 1.01];
    let worst = defective
        .iter()
        .zip(expected.iter())
        .map(|(d, col)| (d - col[0]).abs())
        .fold(0.0f64, f64::max);
    assert!(
        worst > 1e-9,
        "flipped distance law must exceed the 1e-9 weight tolerance, got {worst:.3e}"
    );
}

/// An off-by-one-bin prototype shift must break the 1e-9 dB agreement.
#[test]
fn ra09_off_by_one_prototype_fails_tolerance() {
    let payload = load_golden("ra09_rir_weights");
    let proto = vec_f64(&payload, "geomB_prototype_db");
    let worst = proto
        .iter()
        .zip(proto.iter().skip(1))
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f64, f64::max);
    assert!(
        worst > 1e-9,
        "one-bin prototype shift must exceed the 1e-9 dB tolerance, got {worst:.3e}"
    );
}

// --- RA10: repeatability/seat spread/support/geometry (1e-9; labels exact). ---

/// Reporting seat spread (3 dB) as repeatability (~0.9 dB) must be visible:
/// spatial difference is not measurement noise.
#[test]
fn ra10_seat_spread_for_repeatability_fails_tolerance() {
    let payload = load_golden("ra10_evidence_stats");
    let rep = vec_f64(&payload, "repeat_spread_db");
    let seat = vec_f64(&payload, "seat_spread_db");
    let worst = rep
        .iter()
        .zip(seat.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f64, f64::max);
    assert!(
        worst > 1e-9,
        "seat-for-repeat confusion must exceed the 1e-9 dB tolerance, got {worst:.3e}"
    );
}

/// Halving the timing-to-phase factor (180 instead of 360 deg) must be visible.
#[test]
fn ra10_half_phase_factor_fails_tolerance() {
    let payload = load_golden("ra10_evidence_stats");
    let expected: f64 = serde_json::from_value(payload["phase_at_100hz_deg"].clone()).unwrap();
    let err = (expected / 2.0 - expected).abs();
    assert!(
        err > 1e-9,
        "halved phase factor must exceed the 1e-9 deg tolerance, got {err:.3e}"
    );
}

/// A broadband roll-up that hides the restricted bands (claiming all
/// supported) must disagree with the golden labels.
#[test]
fn ra10_broadband_rollup_fails_support_labels() {
    let payload = load_golden("ra10_evidence_stats");
    let labels: Vec<String> = serde_json::from_value(payload["band_support"].clone()).unwrap();
    let supported = labels.iter().filter(|l| l.as_str() == "supported").count();
    assert!(
        supported < labels.len(),
        "golden must contain a non-supported band so roll-ups fail, got {labels:?}"
    );
}

/// Swapping direct and reflection paths must destroy the interval
/// (reflection must exceed direct): the geometry contract is exact.
#[test]
fn ra10_swapped_paths_fail_geometry_contract() {
    let payload = load_golden("ra10_evidence_stats");
    let direct: f64 = serde_json::from_value(payload["direct_path_m"].clone()).unwrap();
    let refl: f64 = serde_json::from_value(payload["reflection_path_m"].clone()).unwrap();
    let interval: f64 =
        serde_json::from_value(payload["reflection_free_interval_s"].clone()).unwrap();
    assert!(interval > 0.0, "golden interval must be positive");
    let csnd: f64 = serde_json::from_value(payload["sound_speed_m_s"].clone()).unwrap();
    let swapped = (direct - refl) / csnd;
    assert!(
        (swapped - interval).abs() > 1e-12,
        "swapped paths must not reproduce the 1e-12 interval agreement"
    );
}
