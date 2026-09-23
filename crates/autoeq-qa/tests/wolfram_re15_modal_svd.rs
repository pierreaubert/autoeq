//! Wolfram cross-check: analytic 2x2 complex SVD oracle (RE15).
//!
//! Oracle: `wolfram/re15_modal_svd.wls` (engine SingularValueList plus
//! the rank-1 subspace projector, which is invariant under eigenvector
//! sign/rotation). The Rust side solves the 2x2 Hermitian Gram
//! eigensystem in closed form: no public SVD entry point exists in the
//! covered scope, so this case pins the documented modal mathematics
//! (singular values, retained energy, projector, residual) that the
//! modal-basis objective consumes. Tolerance 1e-12 relative/abs
//! (catalogue class L).

use autoeq_qa::require_reference;
use autoeq_qa::{QaResult, assert_case_id, complex_rel_error, emit_result, provenance};
use num_complex::Complex64;

const CASE: &str = "re15_modal_svd";
const CASE_ID: &str = "autoeq-qa.re15-modal-svd.v1";
const TOL: f64 = 1e-12;

fn pair(v: &serde_json::Value) -> Complex64 {
    Complex64::new(v[0].as_f64().unwrap(), v[1].as_f64().unwrap())
}

#[test]
fn wolfram_re15_modal_svd() {
    let ref_json = require_reference(CASE, "re15_modal_svd.wls");
    assert_case_id(&ref_json, CASE_ID, CASE);

    let raw: Vec<[f64; 2]> = serde_json::from_value(ref_json["matrix_re_im"].clone()).unwrap();
    assert_eq!(raw.len(), 4, "{CASE}: expected a 2x2 matrix");
    let h = [
        [
            Complex64::new(raw[0][0], raw[0][1]),
            Complex64::new(raw[1][0], raw[1][1]),
        ],
        [
            Complex64::new(raw[2][0], raw[2][1]),
            Complex64::new(raw[3][0], raw[3][1]),
        ],
    ];
    let want_sv: Vec<f64> = serde_json::from_value(ref_json["singular_values"].clone()).unwrap();
    assert_eq!(want_sv.len(), 2);

    // Closed-form 2x2 Hermitian Gram eigensystem: G = H^H H has
    // eigenvalues (tr +/- sqrt(tr^2 - 4 det)) / 2; sigma = sqrt(lambda).
    let g11 = h[0][0].norm_sqr() + h[1][0].norm_sqr();
    let g22 = h[0][1].norm_sqr() + h[1][1].norm_sqr();
    let g12 = h[0][0].conj() * h[0][1] + h[1][0].conj() * h[1][1];
    let trace = g11 + g22;
    let det = (g11 * g22 - g12.norm_sqr()).max(0.0);
    let disc = (trace * trace - 4.0 * det).max(0.0).sqrt();
    let lam1 = 0.5 * (trace + disc);
    let lam2 = (0.5 * (trace - disc)).max(0.0);
    let (s1, s2) = (lam1.sqrt(), lam2.sqrt());

    let mut max_err = 0.0f64;
    for (got, want) in [s1, s2].iter().zip(want_sv.iter()) {
        let err = (got - want).abs() / want.max(1e-300);
        assert!(
            err <= TOL,
            "{CASE}: singular value: rust={got:.12e} expected={want:.12e} rel_err={err:.3e}"
        );
        max_err = max_err.max(err);
    }

    // Retained rank-1 energy.
    let energy = s1 * s1 / (s1 * s1 + s2 * s2);
    let want_energy = ref_json["rank1_retained_energy"].as_f64().unwrap();
    let eerr = (energy - want_energy).abs();
    assert!(
        eerr <= TOL,
        "{CASE}: retained energy: rust={energy:.12e} expected={want_energy:.12e}"
    );
    max_err = max_err.max(eerr);

    // Dominant Gram eigenvector v ~ (g12, lam1 - g11) is the RIGHT
    // singular vector; the modal projector needs the LEFT one,
    // u1 = H v1 / sigma1. P = u1 u1^H is sign-invariant.
    assert!(
        g12.norm() > 1e-9,
        "{CASE}: fixture needs nonzero off-diagonal Gram entry"
    );
    let v = [g12, Complex64::new(lam1 - g11, 0.0)];
    let nrm = (v[0].norm_sqr() + v[1].norm_sqr()).sqrt();
    let v1 = [v[0] / nrm, v[1] / nrm];
    let u1 = [
        (h[0][0] * v1[0] + h[0][1] * v1[1]) / s1,
        (h[1][0] * v1[0] + h[1][1] * v1[1]) / s1,
    ];
    let proj = [
        [u1[0] * u1[0].conj(), u1[0] * u1[1].conj()],
        [u1[1] * u1[0].conj(), u1[1] * u1[1].conj()],
    ];
    let want_proj: Vec<[f64; 2]> =
        serde_json::from_value(ref_json["rank1_projector_re_im"].clone()).unwrap();
    for (i, row) in proj.iter().enumerate() {
        for (j, got) in row.iter().enumerate() {
            let want = pair(&ref_json["rank1_projector_re_im"][i * 2 + j]);
            let _ = want_proj.len();
            let err = complex_rel_error(*got, want);
            assert!(err <= TOL, "{CASE}: projector[{i},{j}]: rel_err={err:.3e}");
            max_err = max_err.max(err);
        }
    }

    // Projection residual must equal the discarded singular value exactly.
    let apply = |a: [[Complex64; 2]; 2], x: [Complex64; 2]| {
        [
            a[0][0] * x[0] + a[0][1] * x[1],
            a[1][0] * x[0] + a[1][1] * x[1],
        ]
    };
    let mut resid2 = 0.0f64;
    for col in [[h[0][0], h[1][0]], [h[0][1], h[1][1]]] {
        let kept = apply(proj, col);
        resid2 += (col[0] - kept[0]).norm_sqr() + (col[1] - kept[1]).norm_sqr();
    }
    let resid = resid2.sqrt();
    let want_resid = ref_json["projection_residual_frobenius"].as_f64().unwrap();
    let rerr = (resid - want_resid).abs() / want_resid.max(1e-300);
    assert!(
        rerr <= TOL,
        "{CASE}: residual: rust={resid:.12e} expected={want_resid:.12e} rel_err={rerr:.3e}"
    );
    max_err = max_err.max(rerr);
    // The residual identity pins the truncation: residual^2 == sigma2^2.
    let id_err = (resid2 - s2 * s2).abs() / s1.max(1e-300).powi(2);
    assert!(
        id_err <= TOL,
        "{CASE}: residual^2 must equal sigma2^2: err={id_err:.3e}"
    );
    max_err = max_err.max(id_err);

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
