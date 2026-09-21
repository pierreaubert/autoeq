//! Fast psychoacoustic-principle coverage, section A (matrix half).
//!
//! Production entry point under test:
//! [`crate::matrix::TakeMatrix::average`] with explicit
//! [`crate::matrix::AverageKind`]. Oracles are independent analytic
//! power/dB sums computed in-test from first principles.

use ndarray::Array1;

use crate::Curve;
use crate::matrix::{AverageKind, Take, TakeDecision, TakeMatrix};

fn grid() -> Array1<f64> {
    Array1::from(vec![100.0, 200.0, 500.0, 1_000.0, 2_000.0])
}

fn magnitude_curve(grid: &Array1<f64>, level_db: f64, phase_deg: f64) -> Curve {
    Curve {
        freq: grid.clone(),
        spl: Array1::from(vec![level_db; grid.len()]),
        phase: Some(Array1::from(vec![phase_deg; grid.len()])),
        ..Default::default()
    }
}

fn take(id: &str, seat: &str, curve: Curve, reference_id: Option<&str>) -> Take {
    Take {
        take_id: id.to_string(),
        source_id: "sub-1".to_string(),
        seat_id: seat.to_string(),
        weight: 1.0,
        decision: TakeDecision::Accepted,
        curve: Some(curve),
        reference_id: reference_id.map(str::to_string),
    }
}

/// A06: coherent averaging without a shared timing/gain reference is
/// refused, not silently performed. The first take lacking a reference and
/// a later take with a different reference both fail closed.
#[test]
fn psycho_fast_a06_coherent_average_requires_common_reference() {
    let grid = grid();
    let takes = vec![
        take("t1", "seat-1", magnitude_curve(&grid, 70.0, 0.0), None),
        take("t2", "seat-2", magnitude_curve(&grid, 70.0, 0.0), None),
    ];
    let matrix = TakeMatrix {
        takes,
        average_kind: AverageKind::Coherent,
    };
    let error = matrix
        .average(None)
        .expect_err("missing reference must not average");
    assert!(
        format!("{error:?}").contains("reference"),
        "error must name the missing reference: {error:?}"
    );

    let takes = vec![
        take(
            "t1",
            "seat-1",
            magnitude_curve(&grid, 70.0, 0.0),
            Some("ref-a"),
        ),
        take(
            "t2",
            "seat-2",
            magnitude_curve(&grid, 70.0, 0.0),
            Some("ref-b"),
        ),
    ];
    let matrix = TakeMatrix {
        takes,
        average_kind: AverageKind::Coherent,
    };
    let error = matrix
        .average(None)
        .expect_err("split references must not average");
    assert!(
        format!("{error:?}").contains("ref-a"),
        "error must name the divergent reference: {error:?}"
    );
}

/// A06: a coherent request with a shared reference but no explicit contract
/// is still refused: phase gating belongs to the caller, never to a guess.
#[test]
fn psycho_fast_a06_coherent_average_requires_explicit_contract() {
    let grid = grid();
    let takes = vec![
        take(
            "t1",
            "seat-1",
            magnitude_curve(&grid, 70.0, 0.0),
            Some("ref-a"),
        ),
        take(
            "t2",
            "seat-2",
            magnitude_curve(&grid, 70.0, 180.0),
            Some("ref-a"),
        ),
    ];
    let matrix = TakeMatrix {
        takes,
        average_kind: AverageKind::Coherent,
    };
    assert!(
        matrix.average(None).is_err(),
        "shared reference without a contract must not average"
    );
}

/// A06: two seats with opposing phase do not cancel in a power average, and
/// the spatial average carries no measured phase. Analytic oracle: RMS of
/// 70 dB and 70 dB is 70 dB, while a coherent sum would cancel to silence.
#[test]
fn psycho_fast_a06_spatial_averages_carry_no_phase() {
    let grid = grid();
    for kind in [AverageKind::Power, AverageKind::Magnitude] {
        let takes = vec![
            take("t1", "seat-1", magnitude_curve(&grid, 70.0, 0.0), None),
            take("t2", "seat-2", magnitude_curve(&grid, 70.0, 180.0), None),
        ];
        let matrix = TakeMatrix {
            takes,
            average_kind: kind,
        };
        let average = matrix.average(None).expect("spatial average");
        assert!(
            average.phase.is_none(),
            "{kind:?} average must not manufacture phase"
        );
        for level in &average.spl {
            assert!(
                (*level - 70.0).abs() <= 1e-9,
                "{kind:?} average distorted opposing-phase seats: {level}"
            );
        }
    }
}

/// A06: rejected takes never enter the average, so a bad take cannot make
/// the result look worse (or better) than the accepted evidence.
#[test]
fn psycho_fast_a06_rejected_takes_excluded_from_average() {
    let grid = grid();
    let mut bad = take(
        "t-bad",
        "seat-2",
        magnitude_curve(&grid, 40.0, 0.0),
        None,
    );
    bad.decision = TakeDecision::Rejected {
        reason: "clipped".to_string(),
    };
    let matrix = TakeMatrix {
        takes: vec![
            take("t1", "seat-1", magnitude_curve(&grid, 70.0, 0.0), None),
            bad,
        ],
        average_kind: AverageKind::Power,
    };
    assert_eq!(matrix.averaging_set().len(), 1);
    let average = matrix.average(None).expect("accepted-only average");
    for level in &average.spl {
        assert!((*level - 70.0).abs() <= 1e-9, "rejected take leaked: {level}");
    }
}
