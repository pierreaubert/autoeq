//! Wolfram cross-check: correction-band intersection (RM03).
//!
//! Oracle: `wolfram/rm03_band_gauss.wls` (max/min band intersection
//! for the active band and per-curve data support; Gaussian
//! truncation box mean +/- k*sqrt(variance) with the configured
//! boundary tolerance). Tolerance 1e-9 absolute on bands; the
//! Gaussian accept/reject boundary is an exact decision check.

use autoeq_qa::require_reference;
use autoeq_qa::{QaResult, assert_case_id, emit_result, provenance};
use roomeq_model::validation_rules::validate_room_config;
use roomeq_model::{
    AreaPriorKind, ContinuousListeningAreaConfig, CorrectionBandPolicy, MultiSeatConfig,
    MultiSeatStrategy, OptimizerConfig, RoomConfig,
};

const CASE: &str = "rm03_band_gauss";
const CASE_ID: &str = "autoeq-qa.rm03-band-gauss.v1";
const TOL: f64 = 1e-9;

fn area_config(bounds: Vec<(f64, f64)>) -> RoomConfig {
    let mut config = RoomConfig::default();
    config.optimizer.min_freq = 20.0;
    config.optimizer.max_freq = 20_000.0;
    config.optimizer.multi_seat = Some(MultiSeatConfig {
        enabled: true,
        strategy: MultiSeatStrategy::ContinuousArea,
        continuous_area: Some(ContinuousListeningAreaConfig {
            dimensions: 2,
            bounds,
            seat_positions: vec![vec![0.0, 0.0]],
            prior: AreaPriorKind::Gaussian {
                mean: vec![0.0, 0.0],
                cov_diag: vec![0.25, 1.0],
                truncation_sigmas: 3.0,
            },
            quadrature: roomeq_model::AreaQuadratureKind::Sobol {
                num_points: 16,
                seed: 0,
            },
            scalarisation: Default::default(),
            idw_power: 2.0,
        }),
        ..Default::default()
    });
    config
}

fn gauss_errors(config: &RoomConfig) -> Vec<String> {
    validate_room_config(config)
        .errors
        .into_iter()
        .filter(|e| e.contains("Gaussian truncation box"))
        .collect()
}

#[test]
fn wolfram_rm03_band_gauss() {
    let ref_json = require_reference(CASE, "rm03_band_gauss.wls");
    assert_case_id(&ref_json, CASE_ID, CASE);
    let want_active: [f64; 2] = serde_json::from_value(ref_json["active_band_hz"].clone()).unwrap();
    let want_legacy: [f64; 2] = serde_json::from_value(ref_json["legacy_band_hz"].clone()).unwrap();
    let supports: Vec<[f64; 2]> =
        serde_json::from_value(ref_json["data_supports_hz"].clone()).unwrap();
    let want_for_data: Vec<[f64; 2]> =
        serde_json::from_value(ref_json["active_for_data_hz"].clone()).unwrap();
    let matching: Vec<(f64, f64)> =
        serde_json::from_value(ref_json["gaussian"]["matching_bounds_m"].clone()).unwrap();
    let mismatched: Vec<(f64, f64)> =
        serde_json::from_value(ref_json["gaussian"]["mismatched_bounds_m"].clone()).unwrap();
    let tolerances: Vec<f64> =
        serde_json::from_value(ref_json["gaussian"]["boundary_tolerances_m"].clone()).unwrap();
    let gap: f64 =
        serde_json::from_value(ref_json["gaussian"]["mismatch_gap_axis1_m"].clone()).unwrap();
    assert_eq!(supports.len(), want_for_data.len());

    // Active band intersects the policy with the observation band.
    let mut opt = OptimizerConfig {
        min_freq: 20.0,
        max_freq: 20_000.0,
        ..Default::default()
    };
    assert_eq!(opt.active_correction_band(), want_legacy);
    opt.correction_band = Some(CorrectionBandPolicy {
        min_hz: 40.0,
        max_hz: 16_000.0,
        allow_natural_rolloff: true,
    });
    let active = opt.active_correction_band();
    assert!(
        (active[0] - want_active[0]).abs() <= TOL && (active[1] - want_active[1]).abs() <= TOL,
        "active band: rust={active:?} expected={want_active:?}"
    );
    // Data support intersects again without moving the observation band.
    for ([lo, hi], [w_lo, w_hi]) in supports.iter().zip(want_for_data.iter()) {
        let got = opt.active_correction_band_for_data(*lo, *hi);
        assert!(
            (got[0] - w_lo).abs() <= TOL && (got[1] - w_hi).abs() <= TOL,
            "for_data([{lo}, {hi}]): rust={got:?} expected=[{w_lo}, {w_hi}]"
        );
    }
    // A narrowed band without the explicit rolloff opt-in must refuse.
    assert!(
        CorrectionBandPolicy {
            min_hz: 40.0,
            max_hz: 16_000.0,
            allow_natural_rolloff: false,
        }
        .validate_against(20.0, 20_000.0)
        .is_err()
    );
    assert!(
        opt.correction_band
            .unwrap()
            .validate_against(20.0, 20_000.0)
            .is_ok()
    );

    // Gaussian truncation box: matching bounds validate cleanly.
    assert!(
        gap > tolerances[1],
        "oracle mismatch gap {gap} must exceed the boundary tolerance"
    );
    assert!(
        gauss_errors(&area_config(matching)).is_empty(),
        "matching truncation box must validate: {:?}",
        gauss_errors(&area_config(vec![(-1.5, 1.5), (-3.0, 3.0)]))
    );
    // A 1 mm breach on axis 1 (333x the tolerance) must fail exactly.
    let errors = gauss_errors(&area_config(mismatched));
    assert!(
        !errors.is_empty(),
        "mismatched truncation box must be rejected"
    );
    emit_result(&QaResult {
        case: CASE_ID.to_string(),
        pass: true,
        max_rel_error: 0.0,
        max_abs_error: 0.0,
        tolerance: TOL,
        tolerance_kind: "abs".to_string(),
        provenance: provenance(),
    });
}
