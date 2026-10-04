//! Wolfram cross-check: bass routing transfer and correlated power (RE19).
//!
//! Oracle: `wolfram/re19_bass_power.wls` (independent per-route gain /
//! delay / polarity phasors summed coherently; programme power via
//! H R H^H for the declared coherent, uncorrelated and rho = 0.5
//! correlation matrices). Tolerance 1e-9 (dB absolute on SPL/phase,
//! relative on powers; catalogue class A/I).

use autoeq_qa::require_reference;
use autoeq_qa::{QaResult, assert_case_id, complex_rel_error, emit_result, provenance};
use ndarray::Array1;
use num_complex::Complex64;
use roomeq_engine::Curve;
use roomeq_engine::bass_management::predict_bass_output_curve_from_routes;
use roomeq_model::home_cinema::{BassManagementRoute, BassManagementRoutingGraph};
use std::collections::HashMap;

const CASE: &str = "re19_bass_power";
const CASE_ID: &str = "autoeq-qa.re19-bass-power.v1";
const TOL: f64 = 1e-9;

fn route(
    source: &str,
    kind: &str,
    gain_db: f64,
    delay_ms: f64,
    inverted: bool,
) -> BassManagementRoute {
    BassManagementRoute {
        group_id: None,
        source_channel: source.to_string(),
        source_index: 0,
        destination: "SUB".to_string(),
        destination_index: 0,
        pre_chain_channel: None,
        post_chain_channel: None,
        route_kind: kind.to_string(),
        crossover_type: "LR24".to_string(),
        high_pass_hz: None,
        low_pass_hz: None,
        gain_db,
        gain_linear: 1.0,
        matrix_gain: 1.0,
        delay_ms,
        polarity_inverted: inverted,
    }
}

fn graph(
    routes: Vec<BassManagementRoute>,
    trims: HashMap<String, f64>,
) -> BassManagementRoutingGraph {
    BassManagementRoutingGraph {
        physical_sub_output: "SUB".to_string(),
        physical_sub_outputs: vec!["SUB".to_string()],
        input_channels: vec!["C".to_string(), "LFE".to_string()],
        output_channels: vec!["SUB".to_string()],
        routes,
        matrix: None,
        input_trim_db: trims,
        post_dsp_main_alignment_band_hz: None,
        stereo_routing: None,
        advisories: vec![],
    }
}

fn phasor(curve: &Curve, i: usize) -> Complex64 {
    Complex64::from_polar(
        10.0f64.powf(curve.spl[i] / 20.0),
        curve.phase.as_ref().unwrap()[i].to_radians(),
    )
}

#[test]
fn wolfram_re19_bass_power() {
    let ref_json = require_reference(CASE, "re19_bass_power.wls");
    assert_case_id(&ref_json, CASE_ID, CASE);

    let freqs: Vec<f64> = serde_json::from_value(ref_json["freqs_hz"].clone()).unwrap();
    assert_eq!(freqs.len(), 5, "{CASE}: expected 5 grid points");
    let freq = Array1::from_vec(freqs);
    let sub = Curve {
        freq: freq.clone(),
        spl: Array1::from_elem(freq.len(), 85.0),
        phase: Some(Array1::from_elem(freq.len(), 0.0)),
        ..Default::default()
    };
    let routes = vec![
        route("C", "redirected_bass_lowpass_to_sub", -2.0, 1.5, false),
        route("LFE", "lfe_lowpass_to_sub", 3.0, 0.0, true),
    ];
    let full = graph(routes, HashMap::from([("C".to_string(), 1.0)]));
    let combined = predict_bass_output_curve_from_routes(&sub, None, &full, "SUB", 48000.0)
        .expect("{CASE}: routed prediction must succeed");
    assert_eq!(combined.freq.len(), freq.len());

    let mut max_err = 0.0f64;
    let want_spl: Vec<f64> = serde_json::from_value(ref_json["combined_spl_db"].clone()).unwrap();
    let want_ph: Vec<f64> = serde_json::from_value(ref_json["combined_phase_deg"].clone()).unwrap();
    let phase = combined
        .phase
        .as_ref()
        .expect("{CASE}: output must carry phase");
    for i in 0..freq.len() {
        for (got, want, what) in [
            (combined.spl[i], want_spl[i], "SPL"),
            (phase[i], want_ph[i], "phase"),
        ] {
            let err = (got - want).abs();
            assert!(
                err <= TOL,
                "{CASE}: {what}[{i}]: rust={got:.12e} expected={want:.12e} abs_err={err:.3e}"
            );
            max_err = max_err.max(err);
        }
    }

    // Per-route transfers from single-route graphs feed the H R H^H
    // power identities: coherent power is the combined magnitude
    // squared, uncorrelated is the sum of squares, partial adds the
    // cross term with rho = 0.5.
    let singles: Vec<Curve> = ["C", "LFE"]
        .iter()
        .enumerate()
        .map(|(k, _)| {
            let one = vec![
                route("C", "redirected_bass_lowpass_to_sub", -2.0, 1.5, false),
                route("LFE", "lfe_lowpass_to_sub", 3.0, 0.0, true),
            ]
            .into_iter()
            .nth(k)
            .unwrap();
            let g = graph(vec![one], HashMap::from([("C".to_string(), 1.0)]));
            predict_bass_output_curve_from_routes(&sub, None, &g, "SUB", 48000.0).unwrap()
        })
        .collect();
    let want_h: Vec<Vec<[f64; 2]>> =
        serde_json::from_value(ref_json["route_transfer_re_im"].clone()).unwrap();
    let (mut h1, mut h2) = (Vec::new(), Vec::new());
    for (k, single) in singles.iter().enumerate() {
        for (i, want_pair) in want_h[k].iter().enumerate() {
            let got = phasor(single, i);
            let want = Complex64::new(want_pair[0], want_pair[1]);
            let err = complex_rel_error(got, want);
            assert!(
                err <= 1e-12,
                "{CASE}: route {k} bin {i}: transfer rel_err={err:.3e}"
            );
            max_err = max_err.max(err);
            if k == 0 {
                h1.push(got);
            } else {
                h2.push(got);
            }
        }
    }
    let want_coh: Vec<f64> = serde_json::from_value(ref_json["power_coherent"].clone()).unwrap();
    let want_unc: Vec<f64> =
        serde_json::from_value(ref_json["power_uncorrelated"].clone()).unwrap();
    let want_par: Vec<f64> =
        serde_json::from_value(ref_json["power_partial_rho_0p5"].clone()).unwrap();
    for i in 0..freq.len() {
        let z = phasor(&combined, i);
        let coh = (h1[i] + h2[i]).norm_sqr();
        let unc = h1[i].norm_sqr() + h2[i].norm_sqr();
        let par = unc + 2.0 * 0.5 * (h1[i] * h2[i].conj()).re;
        // Routing linearity: the full-graph sum is the sum of routes.
        let lin = complex_rel_error(h1[i] + h2[i], z);
        assert!(lin <= 1e-12, "{CASE}: routing linearity bin {i}: {lin:.3e}");
        for (got, want, what) in [
            (coh, want_coh[i], "coherent"),
            (unc, want_unc[i], "uncorrelated"),
            (par, want_par[i], "partial"),
        ] {
            let err = (got - want).abs() / want.max(1e-300);
            assert!(err <= TOL, "{CASE}: {what} power[{i}]: rel_err={err:.3e}");
            max_err = max_err.max(err);
        }
        // Coherent power is the delivered magnitude squared.
        assert!((coh - z.norm_sqr()).abs() / coh.max(1e-300) <= 1e-12);
    }

    // Contract checks: unknown destinations and phaseless plants refuse.
    assert!(predict_bass_output_curve_from_routes(&sub, None, &full, "NOPE", 48000.0).is_none());
    let phaseless = Curve {
        phase: None,
        ..sub.clone()
    };
    assert!(
        predict_bass_output_curve_from_routes(&phaseless, None, &full, "SUB", 48000.0).is_none()
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
