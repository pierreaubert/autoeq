//! Wolfram cross-check: RA10 evidence statistics (A/X).
//!
//! Oracle: `wolfram/ra10_evidence_stats.wls` — worst-bin max-min band
//! spread for repeats vs seats (same mechanics, distinct types),
//! timing-to-phase `360 f dt` with `None` for non-physical inputs,
//! per-band coherence/SNR minima against the v1 policy with exact
//! Supported/Restricted/Unknown labels, support fractions as exact
//! counts, reflection-free interval `(rR-rD)/c`, gate lower bound
//! `cycles/gate`, per-frequency gate labels, and the quasi-anechoic
//! verdict for a gated direct-sound capture. The Rust side exercises
//! `compute_band_repeatability` / `compute_band_seat_spread` /
//! `phase_uncertainty_deg` / `assess_band_support` /
//! `reflection_free_interval_s` / `valid_lower_bound_hz` /
//! `gate_label_for` / `validate_quasi_anechoic` on identical grids.
//! Tolerance class A: 1e-9 absolute in native units (dB, s, Hz,
//! degrees); every label, count, and reason code exact.

use autoeq_core::direct_sound::{
    QuasiAnechoicPolicy, reflection_free_interval_s, valid_lower_bound_hz,
};
use autoeq_core::evidence::CaptureKind;
use autoeq_qa::require_reference;
use autoeq_qa::{QaResult, assert_case_id, emit_result, provenance};
use ndarray::Array1;
use roomeq_analysis::Curve;
use roomeq_analysis::evidence::{
    BandSupport, ConfidencePolicy, EstimatorKind, assess_band_support, compute_band_repeatability,
    compute_band_seat_spread, phase_uncertainty_deg,
};
use roomeq_analysis::quasi_anechoic::{
    DetailVerdict, GateLabel, PhaseSourceVerdict, QuasiAnechoicInput, gate_label_for,
    validate_quasi_anechoic,
};

const CASE: &str = "ra10_evidence_stats";
const CASE_ID: &str = "autoeq-qa.ra10-evidence-stats.v1";
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

#[test]
fn wolfram_ra10_evidence_stats() {
    let ref_json = require_reference(CASE, "ra10_evidence_stats.wls");
    assert_case_id(&ref_json, CASE_ID, CASE);

    let grid = vec_f64(&ref_json["grid_hz"]);
    let bands: Vec<(f64, f64)> = serde_json::from_value(ref_json["bands_hz"].clone()).unwrap();
    assert_eq!(bands.len(), 3, "{CASE}: three bands");
    let mut worst = 0.0f64;

    // --- Repeatability vs seat spread: same mechanics, distinct types. ---
    let reps: Vec<Vec<f64>> = serde_json::from_value(ref_json["repeat_curves_db"].clone()).unwrap();
    let repeats: Vec<Curve> = reps
        .iter()
        .map(|spl| Curve {
            freq: Array1::from_vec(grid.clone()),
            spl: Array1::from_vec(spl.clone()),
            ..Default::default()
        })
        .collect();
    let rep_reports =
        compute_band_repeatability(&repeats, &bands, "ev-rep").expect("repeatability");
    let rep_spread: Vec<f64> = rep_reports.iter().map(|r| r.spread_db.unwrap()).collect();
    worst = worst.max(assert_abs_vec(
        &rep_spread,
        &vec_f64(&ref_json["repeat_spread_db"]),
        TOL,
        "repeat spread",
    ));
    for report in &rep_reports {
        assert_eq!(report.repeat_count, 3, "{CASE}: repeat count");
        assert_eq!(
            report.estimator,
            EstimatorKind::MaxMinRange,
            "{CASE}: estimator"
        );
    }
    let levels: Vec<f64> = serde_json::from_value(ref_json["seat_levels_db"].clone()).unwrap();
    let seats: Vec<Curve> = levels.iter().map(|l| flat_curve(&grid, *l)).collect();
    let seat_reports = compute_band_seat_spread(&seats, &bands, "ev-seat").expect("seat spread");
    let seat_spread: Vec<f64> = seat_reports.iter().map(|r| r.spread_db.unwrap()).collect();
    worst = worst.max(assert_abs_vec(
        &seat_spread,
        &vec_f64(&ref_json["seat_spread_db"]),
        1e-12,
        "seat spread",
    ));
    for report in &seat_reports {
        assert_eq!(report.seat_count, 3, "{CASE}: seat count");
        assert_eq!(
            report.estimator,
            EstimatorKind::MaxMinRange,
            "{CASE}: estimator"
        );
    }

    // --- Timing-to-phase conversion and its contract refusals [X]. ---
    let dt: f64 = serde_json::from_value(ref_json["timing_s"].clone()).unwrap();
    let ph100: f64 = serde_json::from_value(ref_json["phase_at_100hz_deg"].clone()).unwrap();
    let ph1000: f64 = serde_json::from_value(ref_json["phase_at_1000hz_deg"].clone()).unwrap();
    for (freq, expected) in [(100.0, ph100), (1000.0, ph1000)] {
        let actual = phase_uncertainty_deg(dt, freq).expect("physical conversion");
        let err = (actual - expected).abs();
        assert!(
            err <= TOL,
            "phase({freq} Hz): {actual} vs {expected}, err {err:.3e}"
        );
        worst = worst.max(err);
    }
    assert!(
        ref_json["phase_invalid_freq"].is_null(),
        "{CASE}: golden marks bad freq"
    );
    assert!(
        ref_json["phase_invalid_timing"].is_null(),
        "{CASE}: golden marks bad timing"
    );
    assert!(
        phase_uncertainty_deg(dt, 0.0).is_none(),
        "{CASE}: non-positive frequency must give no value"
    );
    assert!(
        phase_uncertainty_deg(-0.5, 100.0).is_none(),
        "{CASE}: negative timing must give no value"
    );

    // --- Band support minima, labels, and fractions [X]. ---
    let policy = ConfidencePolicy::v1();
    assert!(
        (policy.min_coherence, policy.min_snr_db)
            == (
                serde_json::from_value(ref_json["policy_min_coherence"].clone()).unwrap(),
                serde_json::from_value(ref_json["policy_min_snr_db"].clone()).unwrap()
            ),
        "{CASE}: v1 policy thresholds"
    );
    let support_curve = Curve {
        freq: Array1::from_vec(grid.clone()),
        spl: Array1::from_vec(vec_f64(&ref_json["support_spl_db"])),
        coherence: Some(Array1::from_vec(vec_f64(&ref_json["support_coherence"]))),
        noise_floor_db: Some(Array1::from_vec(vec_f64(&ref_json["support_noise_db"]))),
        ..Default::default()
    };
    let support = assess_band_support(&support_curve, &bands, &policy).expect("support");
    let min_coh: Vec<f64> = support.iter().map(|r| r.min_coherence.unwrap()).collect();
    let min_snr: Vec<f64> = support.iter().map(|r| r.min_snr_db.unwrap()).collect();
    worst = worst.max(assert_abs_vec(
        &min_coh,
        &vec_f64(&ref_json["band_min_coherence"]),
        1e-12,
        "min coherence",
    ));
    worst = worst.max(assert_abs_vec(
        &min_snr,
        &vec_f64(&ref_json["band_min_snr_db"]),
        1e-12,
        "min snr",
    ));
    let labels: Vec<&str> = support
        .iter()
        .map(|r| match r.support {
            BandSupport::Supported => "supported",
            BandSupport::Restricted => "restricted",
            BandSupport::Unknown => "unknown",
        })
        .collect();
    let exp_labels: Vec<String> = serde_json::from_value(ref_json["band_support"].clone()).unwrap();
    assert_eq!(labels, exp_labels, "{CASE}: support labels");
    let n_supported = support
        .iter()
        .filter(|r| r.support == BandSupport::Supported)
        .count();
    let n_restricted = support
        .iter()
        .filter(|r| r.support == BandSupport::Restricted)
        .count();
    let frac_supported = n_supported as f64 / support.len() as f64;
    let frac_restricted = n_restricted as f64 / support.len() as f64;
    let exp_fs: f64 = serde_json::from_value(ref_json["fraction_supported"].clone()).unwrap();
    let exp_fr: f64 = serde_json::from_value(ref_json["fraction_restricted"].clone()).unwrap();
    assert!(
        (frac_supported - exp_fs).abs() <= 1e-12 && (frac_restricted - exp_fr).abs() <= 1e-12,
        "{CASE}: support fractions {frac_supported}/{frac_restricted} vs {exp_fs}/{exp_fr}"
    );
    // Coverage gap and missing evidence are Unknown, never good quality.
    let gap: (f64, f64) = serde_json::from_value(ref_json["gap_band_hz"].clone()).unwrap();
    let gap_report = assess_band_support(&support_curve, &[gap], &policy).expect("gap");
    assert_eq!(
        gap_report[0].support,
        BandSupport::Unknown,
        "{CASE}: gap band"
    );
    let bare = flat_curve(&grid, 80.0);
    let bare_report = assess_band_support(&bare, &bands[1..2], &policy).expect("bare");
    assert_eq!(
        bare_report[0].support,
        BandSupport::Unknown,
        "{CASE}: missing evidence"
    );

    // --- Direct/reflected geometry and gate labels. ---
    let direct: f64 = serde_json::from_value(ref_json["direct_path_m"].clone()).unwrap();
    let refl: f64 = serde_json::from_value(ref_json["reflection_path_m"].clone()).unwrap();
    let csnd: f64 = serde_json::from_value(ref_json["sound_speed_m_s"].clone()).unwrap();
    let interval =
        reflection_free_interval_s(direct, refl, csnd).expect("reflection-free interval");
    let exp_interval: f64 =
        serde_json::from_value(ref_json["reflection_free_interval_s"].clone()).unwrap();
    let err = (interval - exp_interval).abs();
    assert!(
        err <= 1e-12,
        "{CASE}: interval {interval} vs {exp_interval}"
    );
    worst = worst.max(err);
    let gate: f64 = serde_json::from_value(ref_json["gate_s"].clone()).unwrap();
    let cycles: f64 = serde_json::from_value(ref_json["cycles_for_valid_band"].clone()).unwrap();
    let lower = valid_lower_bound_hz(gate, cycles).expect("valid lower bound");
    let exp_lower: f64 = serde_json::from_value(ref_json["valid_lower_hz"].clone()).unwrap();
    let err = (lower - exp_lower).abs();
    assert!(err <= TOL, "{CASE}: lower bound {lower} vs {exp_lower}");
    worst = worst.max(err);
    assert_eq!(
        gate_label_for(Some(lower), 2000.0),
        GateLabel::ReflectionFree
    );
    assert_eq!(
        gate_label_for(Some(lower), 100.0),
        GateLabel::ReflectionContaminated
    );
    assert_eq!(
        gate_label_for(None, 2000.0),
        GateLabel::ReflectionContaminated
    );

    // --- Quasi-anechoic verdict for the gated direct-sound capture. ---
    let band: [f64; 2] =
        serde_json::from_value(ref_json["quasi_requested_band_hz"].clone()).unwrap();
    let input = QuasiAnechoicInput {
        record_id: serde_json::from_value(ref_json["quasi_record_id"].clone()).unwrap(),
        gate_s: Some(gate),
        direct_path_m: Some(direct),
        first_reflection_path_m: Some(refl),
        sound_speed_m_s: csnd,
        capture_kind: CaptureKind::DirectSound,
        moving_microphone_average: false,
        angles_deg: serde_json::from_value(ref_json["quasi_angles_deg"].clone()).unwrap(),
        requested_band_hz: Some(band),
        seat_ids: vec!["mlp".to_string()],
        evidence_refs: vec!["ev-1".to_string()],
    };
    let report = validate_quasi_anechoic(&input, &QuasiAnechoicPolicy::v1()).expect("verdict");
    let err = (report.reflection_free_interval_s.unwrap() - exp_interval).abs();
    assert!(err <= 1e-12, "{CASE}: report interval");
    worst = worst.max(err);
    let err = (report.valid_lower_hz.unwrap() - exp_lower).abs();
    assert!(err <= TOL, "{CASE}: report lower bound");
    worst = worst.max(err);
    let exp_upper: f64 = serde_json::from_value(ref_json["valid_upper_hz"].clone()).unwrap();
    assert_eq!(
        report.valid_upper_hz,
        Some(exp_upper),
        "{CASE}: upper edge carried"
    );
    assert_eq!(
        report.detail,
        DetailVerdict::DetailEligible,
        "{CASE}: detail verdict"
    );
    assert_eq!(
        report.phase_source,
        PhaseSourceVerdict::Supported,
        "{CASE}: phase verdict"
    );
    assert!(
        report.tonal_shaping_permitted,
        "{CASE}: tonal shaping permitted"
    );
    let exp_reasons: Vec<String> =
        serde_json::from_value(ref_json["quasi_reasons"].clone()).unwrap();
    assert_eq!(report.reason_codes, exp_reasons, "{CASE}: reason codes");

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
