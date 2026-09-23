//! Wolfram cross-check: GD target + coherence + confidence gates (RE12).
//!
//! Oracle: `wolfram/re12_gd_coherence.wls` (closed forms: a pure-delay
//! coherent sum has GD exactly equal to the delay; the coherence
//! summary is the arithmetic mean; gate reasons, advisories, and the
//! band derivation follow their documented rules). Absolute tolerance
//! 1e-9 on GD (ms) and coherence; verdict strings match exactly.
//! Weighted-median targeting and bootstrap enumeration have no
//! independently evaluable public contract at this fixture scale and
//! are not claimed here.

use autoeq_qa::{QaResult, assert_case_id, emit_result, provenance, require_reference};
use ndarray::Array1;
use roomeq_engine::Curve;
use roomeq_engine::bass_phase_confidence::{bass_phase_confidence, crossover_phase_advisories};
use roomeq_engine::gd_opt::{
    ChannelGdResult, ChannelMeasurementInput, GdOptConfig, GroupDelayOptResult,
    build_gd_alignment_target, derive_band,
};

const CASE: &str = "re12_gd_coherence";
const CASE_ID: &str = "autoeq-qa.re12-gd-coherence.v1";
const TOL: f64 = 1e-9;

fn vec_f64(value: &serde_json::Value, key: &str) -> Vec<f64> {
    serde_json::from_value(value[key].clone())
        .unwrap_or_else(|_| panic!("{CASE}: golden is missing `{key}`"))
}

fn reason_of(verdict: &roomeq_engine::bass_phase_confidence::BassPhaseConfidence) -> String {
    match verdict {
        roomeq_engine::bass_phase_confidence::BassPhaseConfidence::Trustworthy { .. } => {
            "trustworthy".to_string()
        }
        roomeq_engine::bass_phase_confidence::BassPhaseConfidence::Degraded { reason } => {
            reason.to_string()
        }
    }
}

#[test]
fn wolfram_re12_gd_coherence() {
    let ref_json = require_reference(CASE, "re12_gd_coherence.wls");
    assert_case_id(&ref_json, CASE_ID, CASE);
    let freqs = vec_f64(&ref_json, "freqs_hz");
    let band = vec_f64(&ref_json, "band_hz");
    let want_gd = vec_f64(&ref_json, "sum_gd_reference_ms");
    let want_coh: f64 = serde_json::from_value(ref_json["mean_coherence"].clone()).unwrap();
    assert_eq!(freqs.len(), 6, "{CASE}: expected 6 grid points");
    assert!(
        freqs.iter().all(|v| v.is_finite()),
        "{CASE}: non-finite grid"
    );
    let mut max_abs = 0.0f64;

    // Controlled channels: flat levels, zero intrinsic phase (radians),
    // pure 2 ms delays, second channel inverted.
    let channels = vec![
        ChannelMeasurementInput {
            freq: Array1::from_vec(freqs.clone()),
            spl: Array1::from_elem(freqs.len(), 80.0),
            phase: Array1::zeros(freqs.len()),
            coherence: Array1::from_elem(freqs.len(), 0.95),
        },
        ChannelMeasurementInput {
            freq: Array1::from_vec(freqs.clone()),
            spl: Array1::from_elem(freqs.len(), 76.0),
            phase: Array1::zeros(freqs.len()),
            coherence: Array1::from_elem(freqs.len(), 0.92),
        },
    ];
    let config = GdOptConfig {
        sample_rate: 48000.0,
        ap_per_channel: 0,
        optimize_polarity: true,
        ..GdOptConfig::default()
    };
    let result = GroupDelayOptResult {
        band: (band[0], band[1]),
        per_channel: vec![
            ChannelGdResult {
                delay_ms: 2.0,
                polarity_inverted: false,
                ap_filters: vec![],
                channel_gd_pre_rms_ms: 0.0,
                channel_gd_post_rms_ms: 0.0,
            },
            ChannelGdResult {
                delay_ms: 2.0,
                polarity_inverted: true,
                ap_filters: vec![],
                channel_gd_pre_rms_ms: 0.0,
                channel_gd_post_rms_ms: 0.0,
            },
        ],
        sum_gd_pre_rms_ms: 0.0,
        sum_gd_post_rms_ms: 0.0,
        mean_coherence: 0.0,
        improvement_db: 0.0,
    };
    let target = build_gd_alignment_target(&channels, &result, &config);
    assert_eq!(
        target.per_channel_delay_ms,
        vec![2.0, 2.0],
        "{CASE}: per-channel delays"
    );
    assert_eq!(
        target.per_channel_polarity_inverted,
        vec![false, true],
        "{CASE}: per-channel polarity"
    );
    assert_eq!(target.freq.len(), want_gd.len(), "{CASE}: GD grid length");
    for (i, (&got, &want)) in target
        .sum_gd_reference_ms
        .iter()
        .zip(want_gd.iter())
        .enumerate()
    {
        assert!(got.is_finite(), "{CASE}: non-finite GD at bin {i}");
        assert!(
            (got - want).abs() <= TOL,
            "{CASE}: GD[{i}] rust={got:.12e} oracle={want:.12e}"
        );
        max_abs = max_abs.max((got - want).abs());
    }

    // Coherence summary + confidence gate (phase in degrees here).
    let curves = vec![
        Curve {
            freq: Array1::from_vec(freqs.clone()),
            spl: Array1::from_elem(freqs.len(), 80.0),
            phase: Some(Array1::zeros(freqs.len())),
            coherence: Some(Array1::from_elem(freqs.len(), 0.95)),
            ..Curve::default()
        },
        Curve {
            freq: Array1::from_vec(freqs.clone()),
            spl: Array1::from_elem(freqs.len(), 76.0),
            phase: Some(Array1::zeros(freqs.len())),
            coherence: Some(Array1::from_elem(freqs.len(), 0.92)),
            ..Curve::default()
        },
    ];
    let verdict = bass_phase_confidence(&curves, (band[0], band[1]), None);
    match &verdict {
        roomeq_engine::bass_phase_confidence::BassPhaseConfidence::Trustworthy {
            mean_coherence,
        } => {
            assert!(
                (mean_coherence - want_coh).abs() <= 1e-12,
                "{CASE}: mean coherence {mean_coherence} vs oracle {want_coh}"
            );
            max_abs = max_abs.max((mean_coherence - want_coh).abs());
        }
        other => panic!("{CASE}: expected Trustworthy, got {other:?}"),
    }

    // Degraded reasons follow the documented gate order.
    let empty = bass_phase_confidence(&[], (band[0], band[1]), None);
    assert_eq!(reason_of(&empty), "no_curves", "{CASE}: empty gate");
    let mut no_phase = curves.clone();
    no_phase[0].phase = None;
    let degraded = bass_phase_confidence(&no_phase, (band[0], band[1]), None);
    assert_eq!(reason_of(&degraded), "no_phase_data", "{CASE}: phase gate");
    let low: Vec<Curve> = curves
        .iter()
        .map(|c| Curve {
            coherence: Some(Array1::from_elem(c.freq.len(), 0.5)),
            ..c.clone()
        })
        .collect();
    let degraded = bass_phase_confidence(&low, (band[0], band[1]), None);
    assert_eq!(
        reason_of(&degraded),
        "coherence_below_threshold",
        "{CASE}: coherence gate"
    );
    for (name, key) in [
        ("empty", "degraded_empty"),
        ("no_phase", "degraded_no_phase"),
        ("low_coherence", "degraded_low_coherence"),
    ] {
        let want: String = serde_json::from_value(ref_json[key].clone()).unwrap();
        let got = match name {
            "empty" => reason_of(&empty),
            "no_phase" => reason_of(&bass_phase_confidence(&no_phase, (band[0], band[1]), None)),
            _ => reason_of(&bass_phase_confidence(&low, (band[0], band[1]), None)),
        };
        assert_eq!(got, want, "{CASE}: degraded reason for {name}");
    }

    // Crossover advisories: unknown evidence vs bad evidence.
    let probe = Curve {
        freq: Array1::from_vec(vec![20.0, 40.0, 80.0, 160.0, 1000.0]),
        spl: Array1::from_elem(5, 80.0),
        phase: Some(Array1::zeros(5)),
        ..Curve::default()
    };
    let unknown = crossover_phase_advisories(&probe, (40.0, 160.0), None).unwrap();
    let want_unknown: Vec<String> =
        serde_json::from_value(ref_json["advisories_unknown"].clone()).unwrap();
    let unknown: Vec<String> = unknown.iter().map(|s| s.to_string()).collect();
    assert_eq!(unknown, want_unknown, "{CASE}: unknown-evidence advisories");
    let good = Curve {
        coherence: Some(Array1::from_elem(5, 1.0)),
        noise_floor_db: Some(Array1::from_elem(5, 60.0)),
        ..probe.clone()
    };
    assert!(
        crossover_phase_advisories(&good, (40.0, 160.0), None)
            .unwrap()
            .is_empty(),
        "{CASE}: good evidence must be advisory-free"
    );
    let bad = Curve {
        coherence: Some(Array1::from_vec(vec![1.0, 1.0, 0.0, 1.0, 1.0])),
        noise_floor_db: Some(Array1::from_elem(5, 60.0)),
        ..probe.clone()
    };
    assert!(
        crossover_phase_advisories(&bad, (40.0, 160.0), None).is_err(),
        "{CASE}: bad coherence must fail, not average out"
    );

    // Band derivation from the crossover frequency.
    let derive_in = vec_f64(&ref_json, "derive_band_in");
    let derive_out = vec_f64(&ref_json, "derive_band_out");
    let (lo, hi) = derive_band(derive_in[0], derive_in[1]);
    assert!(
        (lo - derive_out[0]).abs() == 0.0 && (hi - derive_out[1]).abs() == 0.0,
        "{CASE}: derive_band ({lo}, {hi}) vs oracle {derive_out:?}"
    );

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
