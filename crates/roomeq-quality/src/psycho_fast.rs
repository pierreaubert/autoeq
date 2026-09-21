//! Fast psychoacoustic-principle coverage, sections C/E/F (quality half).
//!
//! Production entry points under test:
//! - [`crate::protocol::abx_p_value`], [`crate::protocol::abx_min_correct`],
//!   [`crate::protocol::score_abx_condition`] (exact statistics; the oracle
//!   is independent integer combinatorics: 10/10 at p=0.5 is 1/1024).
//! - [`crate::protocol::ComparisonSpec::validate`] (intent/claim guards).
//! - [`crate::acceptance::evaluate_multi_seat_acceptance`] (worst-seat
//!   visibility with nonconstant residuals).
//! - [`crate::acceptance::evaluate_correction_acceptance`] (raw evidence
//!   retains output loss that display normalization hides).
//! - [`crate::listening::ChainStimulusBinding::verify_unchanged`] (binding
//!   identity for F05).

use autoeq_core::Curve;
use ndarray::Array1;
use roomeq_model::CorrectionAcceptancePolicy;

use crate::acceptance::{evaluate_correction_acceptance, evaluate_multi_seat_acceptance};
use crate::listening::ChainStimulusBinding;
use crate::protocol::{
    ComparisonIntent, ComparisonSpec, ReferenceKind, abx_min_correct, abx_p_value,
    score_abx_condition,
};

fn log_grid(count: usize) -> Array1<f64> {
    Array1::from(
        (0..count)
            .map(|i| 20.0 * 1_000.0_f64.powf(i as f64 / (count - 1) as f64))
            .collect::<Vec<_>>(),
    )
}

fn flat_curve(level_db: f64, count: usize) -> Curve {
    Curve {
        freq: log_grid(count),
        spl: Array1::from(vec![level_db; count]),
        ..Default::default()
    }
}

fn shaped_curve(levels: &[f64]) -> Curve {
    Curve {
        freq: log_grid(levels.len()),
        spl: Array1::from(levels.to_vec()),
        ..Default::default()
    }
}

/// F03: ten correct independent binary trials at chance 0.5 have one-sided
/// exact probability 1/1024. The production calculation must match this
/// independent scalar, and the 5% rule for ten trials is 9 correct.
#[test]
fn psycho_fast_f03_abx_ten_for_ten_is_one_in_1024() {
    let p = abx_p_value(10, 10).expect("valid trial count");
    assert!((p - 1.0 / 1_024.0).abs() <= 1e-12, "p={p}");
    assert_eq!(abx_min_correct(10, 0.05).expect("attainable rule"), 9);
    let outcome =
        score_abx_condition("mono-timbre", 10, 10, 9, 10).expect("matching rule counts");
    assert_eq!(outcome.decision, "pass");
    assert!((outcome.p_value - 1.0 / 1_024.0).abs() <= 1e-12);
}

/// F03/E03: a nonsignificant result never establishes equivalence, and
/// preference never claims inaudibility. Equivalence needs its bound up
/// front; forbidden attribute words fail validation.
#[test]
fn psycho_fast_f03_nonsignificant_is_not_equivalence() {
    let weak = score_abx_condition("mono-timbre", 5, 10, 9, 10).expect("scored");
    assert_eq!(weak.decision, "fail");
    assert!(weak.p_value > 0.05, "5/10 must stay nonsignificant");

    let unbounded = ComparisonSpec {
        name: "pruned-vs-full".to_string(),
        intent: ComparisonIntent::Equivalence,
        attributes: vec!["detectability".to_string()],
        equivalence_bound: None,
        reference: ReferenceKind::PublishedCases {
            citation: "Toole ch.3".to_string(),
        },
        validated_domain: "unvalidated".to_string(),
    };
    assert!(
        unbounded.validate().is_err(),
        "equivalence without a prespecified bound must be rejected"
    );

    let preference = ComparisonSpec {
        name: "corrected-vs-baseline".to_string(),
        intent: ComparisonIntent::Preference,
        attributes: vec!["transparent".to_string()],
        equivalence_bound: None,
        reference: ReferenceKind::PublishedCases {
            citation: "Toole ch.3".to_string(),
        },
        validated_domain: "unvalidated".to_string(),
    };
    assert!(
        preference.validate().is_err(),
        "preference must not claim transparency"
    );
}

/// C02: nonconstant residuals whose mean improves while one guarded seat
/// worsens. The gate names the degraded seat and withholds acceptance; the
/// mean improvement must not hide it.
#[test]
fn psycho_fast_c02_worst_seat_named_not_averaged() {
    let target = flat_curve(70.0, 64);
    // Seat 0 improves, seat 1 improves less, seat 2 regresses across the
    // band with a shaped (nonconstant) residual so no normalization can
    // erase the planted regression.
    let pre = vec![
        shaped_curve(&[72.0; 64]),
        shaped_curve(&[71.0; 64]),
        shaped_curve(&{
            let mut levels = vec![70.5; 64];
            for (i, level) in levels.iter_mut().enumerate() {
                *level += 0.4 * ((i as f64) / 8.0).sin();
            }
            levels
        }),
    ];
    let post = vec![
        shaped_curve(&[70.2; 64]),
        shaped_curve(&[70.4; 64]),
        shaped_curve(&{
            let mut levels = vec![71.6; 64];
            for (i, level) in levels.iter_mut().enumerate() {
                *level += 0.6 * ((i as f64) / 6.0).cos();
            }
            levels
        }),
    ];
    let acceptance = evaluate_multi_seat_acceptance(&pre, &post, &[], &[], &target)
        .expect("aligned grids");
    assert_eq!(acceptance.training.seats.len(), 3);
    assert_eq!(acceptance.training.worst_seat_index, Some(2));
    let worst = acceptance
        .training
        .worst_position_improvement_db
        .expect("worst improvement reported");
    assert!(worst < 0.0, "degraded seat must read negative, got {worst}");
    let mean: f64 = acceptance
        .training
        .seats
        .iter()
        .map(|seat| seat.improvement_db.unwrap_or(0.0))
        .sum::<f64>()
        / 3.0;
    assert!(
        mean > worst,
        "mean {mean} must not stand in for worst {worst}"
    );
    assert!(
        !acceptance.training.accepted(),
        "a regressed seat withholds partition acceptance"
    );
}

/// C03: a candidate that loses 6 dB of useful output is rejected on raw
/// evidence, while the display-normalized copy looks clean. Normalization
/// must never launder output loss through acceptance.
#[test]
fn psycho_fast_c03_normalization_cannot_hide_output_loss() {
    let target = flat_curve(70.0, 64);
    let pre = flat_curve(70.0, 64);
    let post_raw = flat_curve(64.0, 64);
    let raw = evaluate_correction_acceptance(
        &pre,
        &post_raw,
        &target,
        None,
        CorrectionAcceptancePolicy::RuntimeSafety,
    )
    .expect("aligned grids");
    assert!(!raw.accepted, "6 dB loss must not pass");
    assert!(
        raw.violations
            .iter()
            .any(|violation| violation.contains("regressed")),
        "violation must name the regression: {:?}",
        raw.violations
    );

    let mut display_levels = post_raw.spl.to_vec();
    for level in &mut display_levels {
        *level += 6.0;
    }
    let post_display = Curve {
        freq: log_grid(64),
        spl: Array1::from(display_levels),
        ..Default::default()
    };
    let display = evaluate_correction_acceptance(
        &pre,
        &post_display,
        &target,
        None,
        CorrectionAcceptancePolicy::RuntimeSafety,
    )
    .expect("aligned grids");
    assert!(
        display.accepted,
        "normalized view passes shape checks, proving the raw loss is what the gate caught"
    );
    assert!(
        (raw.metrics.pre_target_weighted_rms_db - display.metrics.pre_target_weighted_rms_db)
            .abs()
            <= 1e-9
    );
    assert!(
        raw.metrics.post_target_weighted_rms_db - display.metrics.post_target_weighted_rms_db
            > 5.0,
        "raw and display post metrics must diverge by the hidden loss"
    );
}

/// E03: equal total energy with different spectral envelopes is not
/// timbral equivalence. The oracle is analytic (equal RMS, large local
/// divergence); the claim guard refuses to bless the comparison without a
/// bound and a validated domain.
#[test]
fn psycho_fast_e03_single_scalar_never_equivalence() {
    let raw: Vec<f64> = (0..32)
        .map(|i| 4.0 * ((i as f64) / 3.0).sin() + 2.0 * ((i as f64) / 7.0).cos())
        .collect();
    // Demean so the mirror pair carries exactly equal energy: sums of
    // (65±u)^2 agree only when u sums to zero.
    let mean = raw.iter().sum::<f64>() / raw.len() as f64;
    let tilted_up: Vec<f64> = raw.iter().map(|d| 65.0 + (d - mean)).collect();
    let tilted_down: Vec<f64> = raw.iter().map(|d| 65.0 - (d - mean)).collect();
    let rms = |levels: &[f64]| {
        (levels.iter().map(|level| level * level).sum::<f64>() / levels.len() as f64).sqrt()
    };
    assert!((rms(&tilted_up) - rms(&tilted_down)).abs() <= 1e-9);
    let local: f64 = tilted_up
        .iter()
        .zip(&tilted_down)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f64::max);
    assert!(local > 5.0, "envelopes must differ locally, got {local}");
    let unbounded = ComparisonSpec {
        name: "equal-energy-timbre".to_string(),
        intent: ComparisonIntent::Equivalence,
        attributes: vec!["timbre".to_string()],
        equivalence_bound: None,
        reference: ReferenceKind::PublishedCases {
            citation: "none".to_string(),
        },
        validated_domain: "unvalidated".to_string(),
    };
    assert!(
        unbounded.validate().is_err(),
        "equal energy alone authorizes no equivalence comparison"
    );
}

fn binding() -> ChainStimulusBinding {
    ChainStimulusBinding {
        baseline_graph_id: "graph-baseline".to_string(),
        candidate_graph_id: "graph-candidate".to_string(),
        full_graph_id: "graph-full".to_string(),
        pruned_graph_id: None,
        stimulus_hash: "stimulus-1".to_string(),
        sample_rate_hz: 48_000.0,
        calibration_id: "cal-1".to_string(),
        processing_state: "final-delivered".to_string(),
    }
}

/// F05: single-speaker mono, coherent L+R, and stereo stay distinct
/// bindings, and any changed chain or stimulus voids the comparison.
/// A mismatched record cannot validate a result it did not produce.
#[test]
fn psycho_fast_f05_changed_binding_invalidates() {
    let baseline = binding();
    baseline
        .verify_unchanged(&binding())
        .expect("identical binding verifies");

    let mut changed_stimulus = binding();
    changed_stimulus.stimulus_hash = "stimulus-2".to_string();
    let error = baseline
        .verify_unchanged(&changed_stimulus)
        .expect_err("changed stimulus must void binding");
    assert!(error.contains("stimulus"), "unexpected error: {error}");

    let mut changed_chain = binding();
    changed_chain.candidate_graph_id = "graph-other".to_string();
    let error = baseline
        .verify_unchanged(&changed_chain)
        .expect_err("changed chain must void binding");
    assert!(error.contains("candidate"), "unexpected error: {error}");
}
