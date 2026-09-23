//! Wolfram cross-check: RA04 decay thresholds, Q-decay relation, severity.
//!
//! Oracle: `wolfram/ra04_decay_severity.wls` — log-frequency interpolation
//! of the published artificial/music threshold tables, Q = f T pi/ln(1000),
//! 20 log10 severity, the 32-250 Hz validated domain, and the
//! damped-sinusoid plant (x(t) = exp(-b t) sin(2 pi f t),
//! b = 3 ln(10)/RT60) for the regression check. Tolerance class A for the
//! closed forms (1e-12 s absolute on thresholds, 1e-12 relative on Q,
//! 1e-9 dB absolute on severity); the band-limited regression carries a
//! documented 15% approximation budget plus exact confidence/domain gates.

use autoeq_qa::{QaResult, assert_case_id, emit_result, provenance, rel_error, require_reference};
use roomeq_analysis::impulse_analysis::{
    MIN_MODE_DECAY_CONFIDENCE, ModeDecayEstimate, RoomMode, estimate_mode_decays,
    measured_temporal_severity_db,
};
use roomeq_analysis::temporal_targets::{
    DECAY_THRESHOLD_MAX_HZ, DECAY_THRESHOLD_MIN_HZ, decay_threshold_domain_contains,
    max_acceptable_decay_time, max_acceptable_decay_time_checked, max_acceptable_q,
    q_for_decay_time, temporal_severity,
};

const CASE: &str = "ra04_decay_severity";
const CASE_ID: &str = "autoeq-qa.ra04-decay-severity.v1";
const TOL_TIME_ABS: f64 = 1e-12;
const TOL_Q_REL: f64 = 1e-12;
const TOL_SEV_ABS: f64 = 1e-9;
const TOL_REG_REL: f64 = 0.15;

fn vec_f64(value: &serde_json::Value, key: &str) -> Vec<f64> {
    serde_json::from_value(value[key].clone())
        .unwrap_or_else(|_| panic!("{CASE}: golden is missing numeric array `{key}`"))
}

fn num(value: &serde_json::Value, key: &str) -> f64 {
    value[key]
        .as_f64()
        .unwrap_or_else(|| panic!("{CASE}: golden lacks `{key}`"))
}

#[test]
fn wolfram_ra04_decay_severity() {
    let ref_json = require_reference(CASE, "ra04_decay_severity.wls");
    assert_case_id(&ref_json, CASE_ID, CASE);
    assert_eq!(num(&ref_json, "domain_lo_hz"), DECAY_THRESHOLD_MIN_HZ);
    assert_eq!(num(&ref_json, "domain_hi_hz"), DECAY_THRESHOLD_MAX_HZ);

    let probe: Vec<f64> = vec_f64(&ref_json, "probe_freqs_hz");
    let thr_art = vec_f64(&ref_json, "threshold_artificial_s");
    let thr_mus = vec_f64(&ref_json, "threshold_music_s");
    let max_q = vec_f64(&ref_json, "max_q_artificial");
    assert_eq!(probe.len(), 7, "{CASE}: expected 7 threshold probes");

    // --- Threshold curves: log-frequency interpolation of both tables. ---
    for (i, &f) in probe.iter().enumerate() {
        for (music, expected) in [(false, thr_art[i]), (true, thr_mus[i])] {
            let actual = max_acceptable_decay_time(f, music);
            let err = (actual - expected).abs();
            assert!(
                err <= TOL_TIME_ABS,
                "threshold({f} Hz, music={music}): rust={actual:.12e} expected={expected:.12e} abs_err={err:.3e}"
            );
        }
        let actual_q = max_acceptable_q(f, false);
        let err = rel_error(actual_q, max_q[i]);
        assert!(
            err <= TOL_Q_REL,
            "max_q({f} Hz): rust={actual_q:.12e} expected={:.12e} rel_err={err:.3e}",
            max_q[i]
        );
        let direct = q_for_decay_time(f, thr_art[i]);
        assert!(
            rel_error(direct, max_q[i]) <= TOL_Q_REL,
            "{CASE}: Q-decay relation at {f} Hz"
        );
    }

    // --- Severity: silent below threshold, 20 log10 above; music differs. ---
    for (f, q, key) in [
        (100.0, 3.0, "severity_below_db"),
        (100.0, 30.0, "severity_above_db"),
    ] {
        let actual = temporal_severity(f, q, false);
        let expected = num(&ref_json, key);
        assert!(
            (actual - expected).abs() <= TOL_SEV_ABS,
            "severity({f} Hz, Q={q}): rust={actual:.12e} expected={expected:.12e}"
        );
    }
    let sev_music = temporal_severity(63.0, 30.0, true);
    assert!(
        (sev_music - num(&ref_json, "severity_music_db")).abs() <= TOL_SEV_ABS,
        "{CASE}: music severity"
    );

    // --- Domain: unknown outside 32-250 Hz, never a clamped limit. ---
    for f in [20.0, 300.0, f64::NAN] {
        assert!(
            !decay_threshold_domain_contains(f),
            "{CASE}: {f} Hz must be out of domain"
        );
        assert!(
            max_acceptable_decay_time_checked(f, false).is_none(),
            "{CASE}: {f} Hz must report unknown"
        );
        assert_eq!(
            temporal_severity(f, 30.0, false),
            0.0,
            "{CASE}: {f} Hz severity"
        );
    }
    assert!(decay_threshold_domain_contains(100.0));
    assert!(max_acceptable_decay_time_checked(100.0, false).is_some());

    // --- Measured severity: confident estimate judges the threshold. ---
    let estimate = ModeDecayEstimate {
        frequency_hz: 80.0,
        rt60_seconds: num(&ref_json, "measured_rt60_s"),
        rt60_lower_seconds: 0.85,
        rt60_upper_seconds: 0.95,
        confidence: num(&ref_json, "measured_confidence"),
    };
    assert!(estimate.confidence >= num(&ref_json, "confidence_gate"));
    for (music, key) in [
        (false, "measured_severity_artificial_db"),
        (true, "measured_severity_music_db"),
    ] {
        let actual =
            measured_temporal_severity_db(num(&ref_json, "measured_freq_hz"), &estimate, music)
                .expect("confident in-domain decay must judge");
        assert!(
            (actual - num(&ref_json, key)).abs() <= TOL_SEV_ABS,
            "measured severity music={music}: rust={actual:.12e} expected={} ",
            num(&ref_json, key)
        );
    }
    let weak = ModeDecayEstimate {
        confidence: 0.4,
        ..estimate
    };
    assert!(weak.confidence < MIN_MODE_DECAY_CONFIDENCE);
    assert!(
        measured_temporal_severity_db(80.0, &weak, false).is_none(),
        "{CASE}: low confidence must keep the fallback"
    );
    assert!(
        measured_temporal_severity_db(300.0, &estimate, false).is_none(),
        "{CASE}: out-of-domain measured decay stays unknown"
    );

    // --- Regression: damped sinusoids recover their planted RT60. ---
    let sr = 48000.0;
    let mode = RoomMode {
        frequency: 80.0,
        q: 8.0,
        temporal_severity_db: 0.0,
        prominence_db: 10.0,
        index: 0,
    };
    let mut rts = Vec::new();
    let mut max_reg: f64 = 0.0;
    let plant: Vec<(f64, f64)> = ref_json["regression_plant"]
        .as_array()
        .unwrap_or_else(|| panic!("{CASE}: golden lacks regression_plant"))
        .iter()
        .map(|entry| {
            (
                entry["frequency_hz"].as_f64().expect("plant frequency"),
                entry["rt60_s"].as_f64().expect("plant RT60"),
            )
        })
        .collect();
    assert_eq!(plant.len(), 2, "{CASE}: expected two regression plants");
    for (plant_f, rt) in plant {
        assert_eq!(plant_f, 80.0, "{CASE}: plant frequency");
        let beta = 3.0 * std::f64::consts::LN_10 / rt;
        let ir: Vec<f32> = (0..96_000)
            .map(|i| {
                let t = i as f64 / sr;
                ((-beta * t).exp() * (2.0 * std::f64::consts::PI * 80.0 * t).sin()) as f32
            })
            .collect();
        let est = estimate_mode_decays(std::slice::from_ref(&mode), &ir, sr)[0]
            .unwrap_or_else(|| panic!("{CASE}: planted {rt} s decay must fit"));
        assert!(
            est.confidence >= MIN_MODE_DECAY_CONFIDENCE,
            "{CASE}: planted {rt} s decay confidence {}",
            est.confidence
        );
        let err = rel_error(est.rt60_seconds, rt);
        assert!(
            err <= TOL_REG_REL,
            "RT60 regression: planted={rt} measured={:.6} rel_err={err:.3e}",
            est.rt60_seconds
        );
        assert!(
            est.rt60_lower_seconds <= est.rt60_seconds
                && est.rt60_seconds <= est.rt60_upper_seconds,
            "{CASE}: confidence interval must bracket the estimate"
        );
        rts.push(est.rt60_seconds);
        max_reg = max_reg.max(err);
    }
    assert!(
        rts[1] > 2.0 * rts[0],
        "{CASE}: long vs short plants must be distinguished ({} vs {})",
        rts[1],
        rts[0]
    );

    emit_result(&QaResult {
        case: CASE_ID.to_string(),
        pass: true,
        max_rel_error: max_reg,
        max_abs_error: TOL_SEV_ABS,
        tolerance: TOL_REG_REL,
        tolerance_kind: "rel".to_string(),
        provenance: provenance(),
    });
}
