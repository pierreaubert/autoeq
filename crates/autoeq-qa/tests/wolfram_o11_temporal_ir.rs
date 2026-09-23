//! Wolfram cross-check: temporal IR masking + aggregation paths (O11).
//!
//! Oracle: `wolfram/o11_temporal_ir.wls` (pre/post IR metrics with
//! masking windows, modal temporal penalty, channel energy weights,
//! normalized-vs-absolute SPL alignment). The IR under test is the
//! engine's own vector, so input alignment is exact. Tolerances: 1e-9
//! absolute in dB/ms.

use autoeq_optim::loss::epa::score::{
    EpaChannelRole, EpaConfig, TemporalMaskingConfig, TemporalMaskingMode, TemporalMaskingProfile,
    compute_epa, compute_epa_normalized, epa_channel_energy_weight, temporal_ir_masking_metrics,
    temporal_masking_penalty,
};
use autoeq_qa::{QaResult, assert_case_id, emit_result, provenance, require_reference};

const CASE: &str = "o11_temporal_ir";
const CASE_ID: &str = "autoeq-qa.o11-temporal-ir.v1";
const TOL: f64 = 1e-9;

#[test]
fn wolfram_o11_temporal_ir() {
    let ref_json = require_reference(CASE, "o11_temporal_ir.wls");
    assert_case_id(&ref_json, CASE_ID, CASE);
    let sr = ref_json["sample_rate_hz"].as_f64().unwrap();
    let ir: Vec<f64> = serde_json::from_value(ref_json["ir"].clone()).unwrap();
    assert_eq!(ir.len(), 256, "{CASE}: expected 256-sample IR");

    let config = TemporalMaskingConfig {
        enabled: true,
        weight: 0.15,
        profile: TemporalMaskingProfile::Mixed,
        ir_enabled: true,
        ir_weight: 0.05,
        pre_mask_ms: 3.0,
        post_mask_ms: 120.0,
        pre_ringing_weight: 2.0,
        post_ringing_weight: 1.0,
        ir_audibility_threshold_db: -45.0,
    };
    let m = temporal_ir_masking_metrics(&ir, sr, &config).expect("fixture IR must score");
    assert_eq!(m.main_index, 48, "{CASE}: main peak index");
    let mut max_abs: f64 = 0.0;
    for (label, actual, key) in [
        ("main_time_ms", m.main_time_ms, "main_time_ms"),
        ("pre_peak", m.pre_ringing_peak_db, "pre_ringing_peak_db"),
        ("post_peak", m.post_ringing_peak_db, "post_ringing_peak_db"),
        (
            "pre_audible",
            m.pre_ringing_audible_db,
            "pre_ringing_audible_db",
        ),
        (
            "post_audible",
            m.post_ringing_audible_db,
            "post_ringing_audible_db",
        ),
        ("penalty", m.penalty, "penalty"),
    ] {
        let expected = ref_json[key].as_f64().unwrap();
        assert!(actual.is_finite(), "{CASE}: non-finite {label}");
        let err = (actual - expected).abs();
        assert!(
            err <= TOL,
            "{CASE}: {label} rust={actual:.12e} expected={expected:.12e} abs_err={err:.3e}"
        );
        max_abs = max_abs.max(err);
    }
    // Both pre-ringing above threshold and post-ringing below it are
    // exercised: the excess branches are not degenerate.
    assert!(
        ref_json["pre_ringing_audible_db"].as_f64().unwrap() > -45.0
            && ref_json["post_ringing_audible_db"].as_f64().unwrap() < -45.0,
        "{CASE}: fixture must straddle the audibility threshold"
    );

    let modal = &ref_json["modal"];
    let modes = [TemporalMaskingMode {
        frequency: modal["frequency"].as_f64().unwrap(),
        q: modal["q"].as_f64().unwrap(),
        prominence_db: modal["prominence_db"].as_f64().unwrap(),
        temporal_severity_db: modal["severity_db"].as_f64().unwrap(),
    }];
    let eq_freqs: Vec<f64> = serde_json::from_value(modal["eq_freqs_hz"].clone()).unwrap();
    let eq_spl: Vec<f64> = serde_json::from_value(modal["eq_spl_db"].clone()).unwrap();
    let modal_config = TemporalMaskingConfig {
        weight: modal["weight"].as_f64().unwrap(),
        ..Default::default()
    };
    let got_modal = temporal_masking_penalty(&eq_freqs, &eq_spl, &modes, &modal_config);
    let want_modal = modal["penalty"].as_f64().unwrap();
    let err = (got_modal - want_modal).abs();
    assert!(
        err <= TOL,
        "{CASE}: modal penalty rust={got_modal:.12e} expected={want_modal:.12e}"
    );
    max_abs = max_abs.max(err);

    let weights = &ref_json["channel_weights"];
    assert_eq!(
        epa_channel_energy_weight(EpaChannelRole::Main),
        weights["main"].as_f64().unwrap()
    );
    assert_eq!(
        epa_channel_energy_weight(EpaChannelRole::Surround),
        weights["surround"].as_f64().unwrap()
    );
    assert_eq!(
        epa_channel_energy_weight(EpaChannelRole::Lfe),
        weights["lfe"].as_f64().unwrap()
    );

    // Normalized vs absolute-SPL paths: the oracle blesses the alignment
    // vector; Rust must evaluate both paths identically on it.
    let phon = ref_json["listening_level_phon"].as_f64().unwrap();
    let rel: Vec<f64> = serde_json::from_value(ref_json["spl_rel_db"].clone()).unwrap();
    let abs_levels: Vec<f64> = serde_json::from_value(ref_json["spl_abs_db"].clone()).unwrap();
    for (i, (&r, &a)) in rel.iter().zip(abs_levels.iter()).enumerate() {
        let err = ((r + phon) - a).abs();
        assert!(err <= 1e-12, "{CASE}: denormalize[{i}] mismatch");
        max_abs = max_abs.max(err);
    }
    let freqs = vec![125.0, 1000.0, 8000.0, 16000.0];
    let epa_config = EpaConfig::default();
    let via_normalized = compute_epa_normalized(&freqs, &rel, &epa_config);
    let via_absolute = compute_epa(&freqs, &abs_levels, &epa_config);
    for (label, a, b) in [
        (
            "evaluation",
            via_normalized.evaluation,
            via_absolute.evaluation,
        ),
        ("potency", via_normalized.potency, via_absolute.potency),
        ("activity", via_normalized.activity, via_absolute.activity),
        (
            "preference",
            via_normalized.preference,
            via_absolute.preference,
        ),
    ] {
        assert!(
            a == b,
            "{CASE}: normalized vs absolute {label} differ: {a:.12e} vs {b:.12e}"
        );
    }

    emit_result(&QaResult {
        case: CASE_ID.to_string(),
        pass: true,
        max_rel_error: 0.0,
        max_abs_error: max_abs,
        tolerance: TOL,
        tolerance_kind: "abs".to_string(),
        provenance: provenance(),
    });
}
