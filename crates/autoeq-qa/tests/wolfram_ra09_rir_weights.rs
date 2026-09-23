//! Wolfram cross-check: RA09 RIR prototype weighting (A/N).
//!
//! Oracle: `wolfram/ra09_rir_weights.wls` — Euclidean distances,
//! source-axis angles, Uniform/InverseSquare/Gaussian distance weights,
//! rigid-sphere directivity (`ka = 2 pi f r/c`, c = 343 m/s),
//! per-frequency column normalization, the magnitude-only power
//! prototype `10 log10(sum w P)`, and the measured arrival-energy
//! mixture for the in-band bin (out-of-band bins keep geometric
//! weights). The Rust side builds prototypes through
//! `build_weighted_prototype` / `build_weighted_prototype_with_capture`
//! on identical grids and compares weight matrices, prototype SPL,
//! measured-direction counts, and the magnitude-only contract
//! (phase `None`). Tolerance class A: 1e-9 absolute in dB / unit
//! weights; decision counts exact. Prototype magnitude is never read
//! as measured coherent phase.

use autoeq_core::capture_provenance::{
    CaptureArrival, CaptureCorrection, CaptureGeometry, CaptureProvenance, CaptureReflectionReport,
    CaptureTakeProvenance,
};
use autoeq_qa::require_reference;
use autoeq_qa::{QaResult, assert_case_id, emit_result, provenance};
use ndarray::Array1;
use roomeq_analysis::Curve;
use roomeq_analysis::rir_prototype::{
    DirectivityModel, DistanceWeightMode, RirPrototypeConfig, build_weighted_prototype,
    build_weighted_prototype_with_capture,
};

const CASE: &str = "ra09_rir_weights";
const CASE_ID: &str = "autoeq-qa.ra09-rir-weights.v1";
const TOL: f64 = 1e-9;

fn vec_f64(value: &serde_json::Value) -> Vec<f64> {
    serde_json::from_value(value.clone()).expect("golden numeric array")
}

fn flat_curve(freq: &[f64], level: f64) -> Curve {
    Curve {
        freq: Array1::from_vec(freq.to_vec()),
        spl: Array1::from_vec(vec![level; freq.len()]),
        ..Default::default()
    }
}

fn assert_abs_vec(actual: &[f64], expected: &[f64], tol: f64, what: &str) -> f64 {
    assert_eq!(actual.len(), expected.len(), "{CASE}: {what} length zip");
    let mut worst = 0.0f64;
    for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            a.is_finite() && e.is_finite(),
            "{CASE}: {what}[{i}] non-finite: {a} vs {e}"
        );
        let err = (a - e).abs();
        assert!(
            err <= tol,
            "{what}[{i}]: rust={a:.12e} expected={e:.12e} abs_err={err:.3e}"
        );
        worst = worst.max(err);
    }
    worst
}

fn config(
    reference: [f64; 3],
    source: [f64; 3],
    mics: Vec<[f64; 3]>,
    distance_mode: DistanceWeightMode,
    directivity: DirectivityModel,
    freq_dep: bool,
) -> RirPrototypeConfig {
    RirPrototypeConfig {
        reference_position: reference,
        source_position: source,
        microphone_positions: mics,
        distance_mode,
        directivity,
        frequency_dependent_directivity: freq_dep,
    }
}

#[test]
fn wolfram_ra09_rir_weights() {
    let ref_json = require_reference(CASE, "ra09_rir_weights.wls");
    assert_case_id(&ref_json, CASE_ID, CASE);
    let mut worst = 0.0f64;

    // --- Part A: uniform/omnidirectional power mean from 80/86 dB. ---
    let grid_a = vec![100.0, 1000.0, 10000.0];
    let cfg_a = config(
        [0.0, 0.0, 0.0],
        [0.0, 2.0, 0.0],
        vec![[0.0, 0.0, 0.0], [0.3, 0.0, 0.0]],
        DistanceWeightMode::Uniform,
        DirectivityModel::Omnidirectional,
        false,
    );
    let proto_a = build_weighted_prototype(
        &[flat_curve(&grid_a, 80.0), flat_curve(&grid_a, 86.0)],
        &cfg_a,
    )
    .expect("part A builds");
    let exp_a: f64 = serde_json::from_value(ref_json["proto_uniform_omni_db"].clone()).unwrap();
    worst = worst.max(assert_abs_vec(
        proto_a.curve.spl.as_slice().unwrap(),
        &[exp_a, exp_a, exp_a],
        TOL,
        "uniform/omni prototype",
    ));
    for (i, mic) in proto_a.weights.rows().into_iter().enumerate() {
        worst = worst.max(assert_abs_vec(
            mic.as_slice().unwrap(),
            &[0.5, 0.5, 0.5],
            1e-12,
            &format!("uniform/omni weights mic {i}"),
        ));
    }
    assert!(
        proto_a.curve.phase.is_none(),
        "{CASE}: magnitude-only prototype must carry no phase"
    );

    // --- Part B: Gaussian distance + spherical-head directivity. ---
    let freqs_b = vec_f64(&ref_json["geomB_freqs_hz"]);
    let mics_b: Vec<[f64; 3]> = serde_json::from_value(ref_json["geomB_mics_m"].clone()).unwrap();
    let ref_b: [f64; 3] = serde_json::from_value(ref_json["geomB_reference_m"].clone()).unwrap();
    let src_b: [f64; 3] = serde_json::from_value(ref_json["geomB_source_m"].clone()).unwrap();
    let sigma: f64 = serde_json::from_value(ref_json["geomB_gauss_sigma_m"].clone()).unwrap();
    let radius: f64 = serde_json::from_value(ref_json["geomB_head_radius_m"].clone()).unwrap();
    let spl_b: Vec<Vec<f64>> = serde_json::from_value(ref_json["geomB_spl_db"].clone()).unwrap();
    let cfg_b = config(
        ref_b,
        src_b,
        mics_b,
        DistanceWeightMode::Gaussian { sigma_m: sigma },
        DirectivityModel::SphericalHead { radius_m: radius },
        true,
    );
    let curves_b: Vec<Curve> = spl_b
        .iter()
        .map(|spl| Curve {
            freq: Array1::from_vec(freqs_b.clone()),
            spl: Array1::from_vec(spl.clone()),
            ..Default::default()
        })
        .collect();
    let proto_b = build_weighted_prototype(&curves_b, &cfg_b).expect("part B builds");
    // Golden weights are [mic][freq], matching the (n, m) matrix layout.
    let exp_wb: Vec<Vec<f64>> = serde_json::from_value(ref_json["geomB_weights"].clone()).unwrap();
    for (i, row) in proto_b.weights.rows().into_iter().enumerate() {
        worst = worst.max(assert_abs_vec(
            row.as_slice().unwrap(),
            &exp_wb[i],
            TOL,
            &format!("gauss/head weights mic {i}"),
        ));
    }
    for j in 0..freqs_b.len() {
        let col: f64 = (0..2).map(|i| proto_b.weights[[i, j]]).sum();
        assert!(
            (col - 1.0).abs() <= 1e-12,
            "{CASE}: part B column {j} must normalize, sums to {col}"
        );
    }
    worst = worst.max(assert_abs_vec(
        proto_b.curve.spl.as_slice().unwrap(),
        &vec_f64(&ref_json["geomB_prototype_db"]),
        TOL,
        "gauss/head prototype",
    ));

    // --- Part C: arrival-energy mixture on the compact capture. ---
    let freqs_c = vec_f64(&ref_json["mix_freqs_hz"]);
    let ref_c: [f64; 3] = serde_json::from_value(ref_json["mix_reference_m"].clone()).unwrap();
    let src_c: [f64; 3] = serde_json::from_value(ref_json["mix_source_m"].clone()).unwrap();
    let mics_c: Vec<[f64; 3]> = serde_json::from_value(ref_json["mix_mics_m"].clone()).unwrap();
    let levels: Vec<f64> = serde_json::from_value(ref_json["mix_levels_db"].clone()).unwrap();
    let cfg_c = config(
        ref_c,
        src_c,
        mics_c.clone(),
        DistanceWeightMode::Uniform,
        DirectivityModel::SphericalHead { radius_m: 0.0875 },
        true,
    );
    let curves_c: Vec<Curve> = levels.iter().map(|l| flat_curve(&freqs_c, *l)).collect();
    let baseline = build_weighted_prototype(&curves_c, &cfg_c).expect("baseline builds");
    // Golden geometric weights are [bin][mic].
    let exp_geo: Vec<Vec<f64>> =
        serde_json::from_value(ref_json["mix_geometric_weights"].clone()).unwrap();
    for (j, exp_col) in exp_geo.iter().enumerate() {
        let col: Vec<f64> = (0..mics_c.len())
            .map(|i| baseline.weights[[i, j]])
            .collect();
        worst = worst.max(assert_abs_vec(
            &col,
            exp_col,
            TOL,
            &format!("geometric bin {j}"),
        ));
    }

    let capture = compact_capture(&mics_c, &ref_json);
    let measured = build_weighted_prototype_with_capture(&curves_c, &cfg_c, Some(&capture))
        .expect("mixture builds");
    assert_eq!(
        measured.measured_direction_bins,
        vec![1; 4],
        "{CASE}: only the in-band bin claims a measured direction"
    );
    let exp_mix: Vec<Vec<f64>> =
        serde_json::from_value(ref_json["mix_weights_by_bin"].clone()).unwrap();
    for (j, exp_col) in exp_mix.iter().enumerate() {
        let col: Vec<f64> = (0..mics_c.len())
            .map(|i| measured.weights[[i, j]])
            .collect();
        worst = worst.max(assert_abs_vec(
            &col,
            exp_col,
            TOL,
            &format!("mixture bin {j}"),
        ));
    }
    worst = worst.max(assert_abs_vec(
        measured.curve.spl.as_slice().unwrap(),
        &vec_f64(&ref_json["mix_prototype_db"]),
        TOL,
        "mixture prototype",
    ));
    assert!(
        measured.curve.phase.is_none(),
        "{CASE}: mixture prototype must carry no phase"
    );

    emit_result(&QaResult {
        case: CASE_ID.to_string(),
        pass: true,
        max_rel_error: 0.0,
        max_abs_error: worst,
        tolerance: TOL,
        tolerance_kind: "abs".to_string(),
        provenance: provenance(),
    });
}

/// Compact 4-mic capture with one direct + one reflection arrival, mirroring
/// the construction exercised by the prototype's own unit tests.
fn compact_capture(mics: &[[f64; 3]], ref_json: &serde_json::Value) -> CaptureProvenance {
    let band: [f64; 2] = serde_json::from_value(ref_json["mix_band_hz"].clone()).unwrap();
    let e_direct: Vec<f64> =
        serde_json::from_value(ref_json["mix_direct_energy_db"].clone()).unwrap();
    let e_refl: Vec<f64> =
        serde_json::from_value(ref_json["mix_reflection_energy_db"].clone()).unwrap();
    let takes = mics
        .iter()
        .enumerate()
        .map(|(index, position)| CaptureTakeProvenance {
            microphone_id: format!("mic-{index}"),
            device_id: "aggregate".into(),
            offset_samples: Some(100.0),
            skew_ppm: Some(0.0),
            residual_uncertainty_us: Some(10.0),
            correction_applied: CaptureCorrection::Resampled,
            timing_reference_id: Some("fixed".into()),
            calibration_id: "frozen".into(),
            gain_db: 0.0,
            calibration_orientation: "on_axis".into(),
            position_m: *position,
            position_uncertainty_mm: 0.5,
            preserves_acoustic_delay: true,
            quality_passed: true,
        })
        .collect();
    let event = |direction: [f64; 3], energies: Vec<f64>| CaptureArrival {
        arrival_ms: 10.0,
        relative_ms: 5.0,
        level_db: -6.0,
        microphone_energy_db: energies,
        direction: Some(direction),
        mirror_ambiguous: false,
        residual_samples: Some(0.1),
        band_hz: Some(band),
        issues: Vec::new(),
    };
    CaptureProvenance {
        geometry: CaptureGeometry::Compact,
        takes,
        reflection_report: Some(CaptureReflectionReport {
            source_id: "left".into(),
            issues: Vec::new(),
            direct_sound: Some(event([1.0, 0.0, 0.0], e_direct)),
            early_reflections: vec![event([-1.0, 0.0, 0.0], e_refl)],
        }),
    }
}
