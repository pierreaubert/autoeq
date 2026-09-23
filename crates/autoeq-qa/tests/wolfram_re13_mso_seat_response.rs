//! Wolfram cross-check: MSO per-seat transfer matrix and shared EQ (RE13).
//!
//! Oracle: `wolfram/re13_mso_seat_response.wls` (independent direct
//! complex sums with gain/delay path factors, plus shared-EQ addition).
//! SPL compares absolute (dB, catalogue class A); phase compares through
//! the complex phasor so the implementation's phase unwrap cannot hide a
//! defect. Tolerance 1e-9 dB absolute, 1e-12 complex-relative.

use autoeq_qa::require_reference;
use autoeq_qa::{QaResult, assert_case_id, complex_rel_error, emit_result, provenance};
use ndarray::Array1;
use num_complex::Complex64;
use roomeq_engine::Curve;
use roomeq_engine::multisub::joint_objective::apply_shared_eq_to_residual;
use roomeq_engine::multisub::render_mso_seat_responses;

const CASE: &str = "re13_mso_seat_response";
const CASE_ID: &str = "autoeq-qa.re13-mso-seat-response.v1";
const TOL_DB: f64 = 1e-9;
const TOL_COMPLEX: f64 = 1e-12;

fn curve(spl: f64, phase: f64, freq: &Array1<f64>) -> Curve {
    Curve {
        freq: freq.clone(),
        spl: Array1::from_elem(freq.len(), spl),
        phase: Some(Array1::from_elem(freq.len(), phase)),
        ..Default::default()
    }
}

fn phasor(spl_db: f64, phase_deg: f64) -> Complex64 {
    Complex64::from_polar(10.0f64.powf(spl_db / 20.0), phase_deg.to_radians())
}

#[test]
fn wolfram_re13_mso_seat_response() {
    let ref_json = require_reference(CASE, "re13_mso_seat_response.wls");
    assert_case_id(&ref_json, CASE_ID, CASE);

    let freqs: Vec<f64> = serde_json::from_value(ref_json["freqs_hz"].clone()).unwrap();
    let seat_re_im: Vec<Vec<[f64; 2]>> =
        serde_json::from_value(ref_json["seat_response_re_im"].clone()).unwrap();
    let seat_spl: Vec<Vec<f64>> = serde_json::from_value(ref_json["seat_spl_db"].clone()).unwrap();
    let eq_db: Vec<f64> = serde_json::from_value(ref_json["shared_eq_db"].clone()).unwrap();
    let corrected: Vec<Vec<f64>> =
        serde_json::from_value(ref_json["corrected_seat_spl_db"].clone()).unwrap();
    assert_eq!(freqs.len(), 6, "{CASE}: expected 6 grid points");
    assert_eq!(seat_re_im.len(), 2);
    assert_eq!(eq_db.len(), freqs.len());

    // Explicit shared grid on every measurement: never zip unequal grids.
    let freq = Array1::from_vec(freqs.clone());
    let sub1_spl: Vec<f64> =
        serde_json::from_value(ref_json["sub1_spl_db_per_seat"].clone()).unwrap();
    let sub1_ph: Vec<f64> =
        serde_json::from_value(ref_json["sub1_phase_deg_per_seat"].clone()).unwrap();
    let measurements = vec![
        vec![curve(80.0, 0.0, &freq), curve(80.0, 0.0, &freq)],
        vec![
            curve(sub1_spl[0], sub1_ph[0], &freq),
            curve(sub1_spl[1], sub1_ph[1], &freq),
        ],
    ];
    let seats = render_mso_seat_responses(&measurements, &[0.0, -6.0], &[0.0, 2.0]).unwrap();
    assert_eq!(seats.len(), 2, "{CASE}: expected 2 seat responses");

    let mut max_db = 0.0f64;
    let mut max_complex = 0.0f64;
    for (seat_idx, seat) in seats.iter().enumerate() {
        assert_eq!(seat.freq.len(), freqs.len());
        for (i, f) in seat.freq.iter().enumerate() {
            assert!(
                (f - freq[i]).abs() == 0.0,
                "{CASE}: seat {seat_idx} grid must reproduce the shared grid"
            );
        }
        let phase = seat.phase.as_ref().expect("{CASE}: seat must carry phase");
        for i in 0..seat.freq.len() {
            let got = phasor(seat.spl[i], phase[i]);
            let want = Complex64::new(seat_re_im[seat_idx][i][0], seat_re_im[seat_idx][i][1]);
            assert!(want.norm().is_finite());
            let cerr = complex_rel_error(got, want);
            assert!(
                cerr <= TOL_COMPLEX,
                "{CASE}: seat {seat_idx} bin {i}: phasor rel_err={cerr:.3e}"
            );
            max_complex = max_complex.max(cerr);
            let derr = (seat.spl[i] - seat_spl[seat_idx][i]).abs();
            assert!(
                derr <= TOL_DB,
                "{CASE}: seat {seat_idx} bin {i}: SPL abs_err={derr:.3e}"
            );
            max_db = max_db.max(derr);
        }
    }

    // Shared EQ adds the same dB at every seat: relative seat differences
    // are unchanged, only the common residual moves.
    let fixed = apply_shared_eq_to_residual(&seats, &eq_db).unwrap();
    assert_eq!(fixed.len(), 2);
    for (seat_idx, seat) in fixed.iter().enumerate() {
        for (i, (spl, want)) in seat.spl.iter().zip(corrected[seat_idx].iter()).enumerate() {
            let derr = (spl - want).abs();
            assert!(
                derr <= TOL_DB,
                "{CASE}: corrected seat {seat_idx} bin {i}: abs_err={derr:.3e}"
            );
            max_db = max_db.max(derr);
            let rel_rust = seats[0].spl[i] - seats[1].spl[i];
            let rel_got = fixed[0].spl[i] - fixed[1].spl[i];
            assert!(
                (rel_rust - rel_got).abs() <= 1e-12,
                "{CASE}: shared EQ must preserve relative seat response"
            );
        }
    }

    // Contract checks: control/seat count mismatch and EQ grid mismatch
    // refuse loudly instead of passing.
    assert!(render_mso_seat_responses(&measurements, &[0.0], &[0.0, 2.0]).is_err());
    assert!(apply_shared_eq_to_residual(&seats, &eq_db[..5]).is_err());

    emit_result(&QaResult {
        case: CASE_ID.to_string(),
        pass: true,
        max_rel_error: max_complex,
        max_abs_error: max_db,
        tolerance: TOL_DB,
        tolerance_kind: "abs".to_string(),
        provenance: provenance(),
    });
}
