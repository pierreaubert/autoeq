//! Wolfram cross-check: spatial quadrature and risk arithmetic (RE16).
//!
//! Oracle: `wolfram/re16_spatial_quadrature.wls` (analytic Gaussian
//! area integral, explicit midpoint quadrature, high-precision normal
//! CDF/inverse, exact discrete expected value and fractional-tail
//! CVaR). No public quadrature/CDF entry point exists in the covered
//! scope, so the Rust side implements the documented formulas (midpoint
//! rule, Abramowitz-Stegun erf, Acklam inverse normal, sort-descending
//! fractional CVaR tail) and the engine checks every coefficient and
//! branch. Classes A/B; the quadrature-vs-analytic budget (0.1 on an
//! integral near 44) is a stated approximation allowance, verified
//! against the engine analytic value.

use autoeq_qa::require_reference;
use autoeq_qa::{QaResult, assert_case_id, emit_result, provenance, rel_error};

const CASE: &str = "re16_spatial_quadrature";
const CASE_ID: &str = "autoeq-qa.re16-spatial-quadrature.v1";
const TOL_QUAD: f64 = 1e-9;
const TOL_CDF: f64 = 2e-7;
const TOL_INV: f64 = 2e-9;
const TOL_EXACT: f64 = 1e-12;
const QUAD_ANALYTIC_BUDGET: f64 = 0.1;

// Abramowitz-Stegun 7.1.26 (|error| <= 1.5e-7), the documented erf.
fn erf_as(x: f64) -> f64 {
    let sign = x.signum();
    let ax = x.abs();
    let t = 1.0 / (1.0 + 0.3275911 * ax);
    let poly = (((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t
        + 0.254829592)
        * t)
        * (-ax * ax).exp();
    sign * (1.0 - poly)
}

fn standard_normal_cdf(x: f64) -> f64 {
    0.5 * (1.0 + erf_as(x / std::f64::consts::SQRT_2))
}

// Acklam's inverse-normal rational approximation with the documented
// coefficients and the 0.02425 tail branch.
fn inv_standard_normal(p: f64) -> f64 {
    let p = p.clamp(1e-12, 1.0 - 1e-12);
    let a = [
        -3.969683028665376e1,
        2.209460984245205e2,
        -2.759285104469687e2,
        1.38357751867269e2,
        -3.066479806614716e1,
        2.506628277459239,
    ];
    let b = [
        -5.447609879822406e1,
        1.615858368580409e2,
        -1.556989798598866e2,
        6.680131188771972e1,
        -1.328068155288572e1,
    ];
    let c = [
        -7.784894002430293e-3,
        -3.223964580411365e-1,
        -2.400758277161838,
        -2.549732539343734,
        4.374664141464968,
        2.938163982698783,
    ];
    let d = [
        7.784695709041462e-3,
        3.224671290700398e-1,
        2.445134137142996,
        3.754408661907416,
    ];
    const TAIL: f64 = 0.02425;
    if p < TAIL {
        let q = (-2.0 * p.ln()).sqrt();
        (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    } else if p <= 1.0 - TAIL {
        let q = p - 0.5;
        let r = q * q;
        (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q
            / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
    } else {
        let q = (-2.0 * (1.0 - p).ln()).sqrt();
        -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    }
}

// Fractional-tail CVaR exactly as documented: sort by descending loss,
// accumulate take = min(alpha - mass, w) over positive-mass points.
fn cvar(alpha: f64, losses: &[f64], weights: &[f64]) -> f64 {
    let mut pairs: Vec<(f64, f64)> = losses
        .iter()
        .copied()
        .zip(weights.iter().copied())
        .collect();
    pairs.sort_by(|a, b| b.0.total_cmp(&a.0));
    let (mut acc_loss, mut acc_mass) = (0.0, 0.0);
    for (loss, weight) in &pairs {
        if *weight <= 0.0 {
            continue;
        }
        let take = (alpha - acc_mass).min(*weight);
        if take <= 0.0 {
            break;
        }
        acc_loss += take * loss;
        acc_mass += take;
        if acc_mass >= alpha {
            break;
        }
    }
    if acc_mass > 0.0 {
        acc_loss / acc_mass
    } else {
        f64::INFINITY
    }
}

#[test]
fn wolfram_re16_spatial_quadrature() {
    let ref_json = require_reference(CASE, "re16_spatial_quadrature.wls");
    assert_case_id(&ref_json, CASE_ID, CASE);
    let mut max_err = 0.0f64;
    let mut max_rel_err = 0.0f64;

    // Explicit 8x6 midpoint quadrature of the Gaussian field.
    let (a, cx, cy, s, b) = (5.0, 1.0, 0.5, 0.4, 20.0);
    let (lx, ly, nx, ny) = (2.0, 1.0, 8usize, 6usize);
    let w = lx * ly / (nx * ny) as f64;
    let mut qsum = 0.0f64;
    for i in 0..nx {
        for j in 0..ny {
            let x = (i as f64 + 0.5) * lx / nx as f64;
            let y = (j as f64 + 0.5) * ly / ny as f64;
            qsum += w * (a * (-((x - cx).powi(2) + (y - cy).powi(2)) / (2.0 * s * s)).exp() + b);
        }
    }
    let want_quad = ref_json["quadrature_integral"].as_f64().unwrap();
    let qerr = (qsum - want_quad).abs();
    assert!(
        qerr <= TOL_QUAD,
        "{CASE}: quadrature: rust={qsum:.12e} expected={want_quad:.12e}"
    );
    max_err = max_err.max(qerr);
    max_rel_err = max_rel_err.max(rel_error(qsum, want_quad));
    // Constant field: quadrature is exact.
    let const_quad = w * (nx * ny) as f64 * b;
    let const_err = (const_quad - ref_json["const_field_analytic"].as_f64().unwrap()).abs();
    assert!(
        const_err <= TOL_EXACT,
        "{CASE}: const quadrature err={const_err:.3e}"
    );
    max_err = max_err.max(const_err);
    max_rel_err = max_rel_err.max(rel_error(
        const_quad,
        ref_json["const_field_analytic"].as_f64().unwrap(),
    ));
    // Stated approximation budget against the analytic integral.
    let analytic = ref_json["analytic_integral"].as_f64().unwrap();
    let gap = (qsum - analytic).abs();
    assert!(
        gap <= QUAD_ANALYTIC_BUDGET,
        "{CASE}: quadrature-vs-analytic gap {gap:.3e} exceeds budget"
    );
    // This discretization allowance has its own budget; transcription errors
    // below use the much tighter formula-agreement tolerances. Retain both.
    println!(
        "QA_APPROXIMATION: {}",
        serde_json::json!({
            "case": CASE_ID,
            "quantity": "quadrature_vs_analytic_integral",
            "absolute_error": gap,
            "budget": QUAD_ANALYTIC_BUDGET,
        })
    );

    // Normal CDF against the engine Erf reference.
    let cdf_x: Vec<f64> = serde_json::from_value(ref_json["cdf_points"].clone()).unwrap();
    let cdf_ref: Vec<f64> = serde_json::from_value(ref_json["cdf_reference"].clone()).unwrap();
    for (x, want) in cdf_x.iter().zip(cdf_ref.iter()) {
        let actual = standard_normal_cdf(*x);
        let err = (actual - want).abs();
        assert!(
            err <= TOL_CDF,
            "{CASE}: Phi({x}): err={err:.3e} exceeds {TOL_CDF:.1e}"
        );
        max_err = max_err.max(err);
        max_rel_err = max_rel_err.max(rel_error(actual, *want));
    }
    // Inverse normal against the engine InverseErf reference.
    let inv_p: Vec<f64> = serde_json::from_value(ref_json["inverse_points"].clone()).unwrap();
    let inv_ref: Vec<f64> = serde_json::from_value(ref_json["inverse_reference"].clone()).unwrap();
    for (p, want) in inv_p.iter().zip(inv_ref.iter()) {
        let actual = inv_standard_normal(*p);
        let err = (actual - want).abs();
        assert!(
            err <= TOL_INV,
            "{CASE}: Phi^-1({p}): err={err:.3e} exceeds {TOL_INV:.1e}"
        );
        max_err = max_err.max(err);
        max_rel_err = max_rel_err.max(rel_error(actual, *want));
    }

    // Exact discrete expected value and fractional-tail CVaR.
    let losses: Vec<f64> = serde_json::from_value(ref_json["losses"].clone()).unwrap();
    let weights: Vec<f64> = serde_json::from_value(ref_json["loss_weights"].clone()).unwrap();
    let expected: f64 = losses.iter().zip(weights.iter()).map(|(l, w)| l * w).sum();
    let eerr = (expected - ref_json["expected_loss"].as_f64().unwrap()).abs();
    assert!(eerr <= TOL_EXACT, "{CASE}: expected err={eerr:.3e}");
    max_rel_err = max_rel_err.max(rel_error(
        expected,
        ref_json["expected_loss"].as_f64().unwrap(),
    ));
    for (alpha, key) in [(0.5, "cvar_alpha_0p5"), (0.9, "cvar_alpha_0p9")] {
        let got = cvar(alpha, &losses, &weights);
        let want = ref_json[key].as_f64().unwrap();
        let err = (got - want).abs();
        assert!(err <= TOL_EXACT, "{CASE}: CVaR({alpha}) err={err:.3e}");
        max_err = max_err.max(err.max(eerr));
        max_rel_err = max_rel_err.max(rel_error(got, want));
    }

    emit_result(&QaResult {
        case: CASE_ID.to_string(),
        pass: true,
        max_rel_error: max_rel_err,
        max_abs_error: max_err,
        tolerance: TOL_CDF,
        tolerance_kind: "abs".to_string(),
        provenance: provenance(),
    });
}
