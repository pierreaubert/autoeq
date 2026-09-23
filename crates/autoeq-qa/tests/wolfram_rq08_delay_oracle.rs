//! Wolfram cross-check: oracle transfer-equation plants (RQ08).
//!
//! Oracle: `wolfram/rq08_delay_oracle.wls` (analytic pure-delay phase ramp
//! and inverted-polarity plant). Tolerance 1e-9 complex-relative; grid
//! points 1e-12 relative. Magnitude/phase agreement only: no time-domain
//! or spatial claim is made here.

use autoeq_qa::require_reference;
use autoeq_qa::{QaResult, assert_case_id, complex_rel_error, emit_result, provenance};
use num_complex::Complex64;
use roomeq_quality::{delay_oracle, log_frequency_grid, polarity_oracle};

const CASE: &str = "rq08_delay_oracle";
const CASE_ID: &str = "autoeq-qa.rq08-delay-oracle.v1";
const TOL: f64 = 1e-9;

#[test]
fn wolfram_rq08_delay_oracle() {
    let ref_json = require_reference(CASE, "rq08_delay_oracle.wls");
    assert_case_id(&ref_json, CASE_ID, CASE);
    let freqs: Vec<f64> = serde_json::from_value(ref_json["freqs_hz"].clone()).unwrap();
    let delay_pairs: Vec<[f64; 2]> =
        serde_json::from_value(ref_json["delay_re_im"].clone()).unwrap();
    let delay_mag: Vec<f64> = serde_json::from_value(ref_json["delay_mag"].clone()).unwrap();
    let pol_pairs: Vec<[f64; 2]> =
        serde_json::from_value(ref_json["polarity_re_im"].clone()).unwrap();
    let delay_ms: f64 = serde_json::from_value(ref_json["delay_ms"].clone()).unwrap();
    assert_eq!(freqs.len(), 8, "{CASE}: expected 8 grid points");
    assert_eq!(delay_pairs.len(), freqs.len());
    assert_eq!(delay_mag.len(), freqs.len());
    assert_eq!(pol_pairs.len(), freqs.len());

    let grid = log_frequency_grid(freqs.len(), freqs[0], freqs[freqs.len() - 1]);
    let mut max_err = 0.0f64;
    for (i, (g, w)) in grid.iter().zip(freqs.iter()).enumerate() {
        let err = (g - w).abs() / w.abs();
        assert!(
            err <= 1e-12,
            "{CASE}: grid[{i}] rust={g:.12e} expected={w:.12e} rel_err={err:.3e}"
        );
        max_err = max_err.max(err);
    }

    let delay = delay_oracle(grid.clone(), delay_ms);
    assert_eq!(delay.expected_transfer.len(), freqs.len());
    for ((pair, mag), value) in delay_pairs
        .iter()
        .zip(delay_mag.iter())
        .zip(delay.expected_transfer.iter())
    {
        let expected = Complex64::new(pair[0], pair[1]);
        assert!(expected.norm().is_finite());
        let err = complex_rel_error(*value, expected);
        assert!(
            err <= TOL,
            "{CASE}: delay rust={value:?} expected={expected:?} rel_err={err:.3e}"
        );
        max_err = max_err.max(err);
        let mag_err = (value.norm() - mag).abs();
        assert!(
            mag_err <= TOL,
            "{CASE}: delay unit magnitude drift {mag_err:.3e}"
        );
        max_err = max_err.max(mag_err);
    }
    assert_eq!(
        delay.valid_correction_region_hz,
        (freqs[0], freqs[freqs.len() - 1]),
        "{CASE}: delay region must span the grid"
    );

    let pol = polarity_oracle(grid, true);
    for (pair, value) in pol_pairs.iter().zip(pol.expected_transfer.iter()) {
        let expected = Complex64::new(pair[0], pair[1]);
        let err = complex_rel_error(*value, expected);
        assert!(
            err <= TOL,
            "{CASE}: polarity rust={value:?} expected={expected:?} rel_err={err:.3e}"
        );
        max_err = max_err.max(err);
    }

    emit_result(&QaResult {
        case: CASE_ID.to_string(),
        pass: true,
        max_rel_error: max_err,
        max_abs_error: 0.0,
        tolerance: TOL,
        tolerance_kind: "rel".to_string(),
        provenance: provenance(),
    });
}
