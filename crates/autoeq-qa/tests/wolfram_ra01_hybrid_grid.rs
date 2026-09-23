//! Wolfram cross-check: RoomEQ hybrid optimizer grid (RA01).
//!
//! Oracle: `wolfram/ra01_hybrid_grid.wls` (grid re-derived from the
//! declared constants: 4 Hz bass spacing 20..1000 Hz, log density from
//! `frequency_samples` over 20..20000 Hz). Tolerance 1e-9 relative per
//! grid point (A); length, junction, endpoint and monotonicity exact (X).

use autoeq_qa::require_reference;
use autoeq_qa::{QaResult, assert_case_id, emit_result, provenance, rel_error};
use roomeq_analysis::frequency_grid::room_eq_hybrid_frequency_grid;

const CASE: &str = "ra01_hybrid_grid";
const CASE_ID: &str = "autoeq-qa.ra01-hybrid-grid.v1";
const TOL: f64 = 1e-9;

#[test]
fn wolfram_ra01_hybrid_grid() {
    let ref_json = require_reference(CASE, "ra01_hybrid_grid.wls");
    assert_case_id(&ref_json, CASE_ID, CASE);

    let freqs: Vec<f64> = serde_json::from_value(ref_json["freqs_hz"].clone()).unwrap();
    let frequency_samples: usize =
        serde_json::from_value(ref_json["frequency_samples"].clone()).unwrap();
    let n_low: usize = serde_json::from_value(ref_json["n_low"].clone()).unwrap();
    assert_eq!(frequency_samples, 200);
    assert_eq!(n_low, 246);
    assert_eq!(freqs.len(), 246 + 87, "{CASE}: expected 333 grid points");
    assert!(freqs.iter().all(|f| f.is_finite()));

    let rust = room_eq_hybrid_frequency_grid(frequency_samples);
    assert_eq!(
        rust.len(),
        freqs.len(),
        "{CASE}: grid length must match the oracle grid"
    );

    let mut max_err = 0.0f64;
    for (i, (got, want)) in rust.iter().zip(freqs.iter()).enumerate() {
        let err = rel_error(*got, *want);
        assert!(
            err <= TOL,
            "grid[{i}]: rust={got:.12e} expected={want:.12e} rel_err={err:.3e}"
        );
        max_err = max_err.max(err);
    }

    // Structural contract: linear bass leg, log treble leg, exact junction.
    assert_eq!(rust[0], 20.0, "{CASE}: grid must start at 20 Hz");
    assert_eq!(
        rust[n_low - 1],
        1000.0,
        "{CASE}: junction must sit at 1000 Hz"
    );
    for i in 0..n_low - 1 {
        let step = rust[i + 1] - rust[i];
        assert!(
            (step - 4.0).abs() <= 1e-9,
            "{CASE}: bass step {i} is {step:.12e}, want 4 Hz"
        );
    }
    assert!(
        (rust[rust.len() - 1] - 20_000.0).abs() / 20_000.0 <= TOL,
        "{CASE}: grid must end at 20000 Hz"
    );
    assert!(
        rust.windows(2).into_iter().all(|w| w[1] > w[0]),
        "{CASE}: grid must be strictly increasing"
    );

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
