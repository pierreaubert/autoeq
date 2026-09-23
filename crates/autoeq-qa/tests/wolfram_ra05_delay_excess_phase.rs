//! Wolfram cross-check: RA05 bulk-delay removal + bounded phase correction.
//!
//! Oracle: `wolfram/ra05_delay_excess_phase.wls` — closed-form
//! least-squares bulk delay of a pure 2.3 ms delay on flat magnitude, and
//! the analytic excess group delay 1000/(pi f0 (1+(f/f0)^2)) ms of a
//! first-order all-pass (corner 200 Hz). Magnitude here is flat so the
//! minimum-phase baseline is separated from the excess-phase claims.
//! Tolerances: bulk delay absolute 1e-6 s [A]; residual/correction phase
//! absolute 1e-9 [A]; all-pass GD absolute 0.35 ms [N, smoothing budget];
//! correction bounds exact [X].

use autoeq_qa::{QaResult, assert_case_id, emit_result, provenance, require_reference};
use roomeq_analysis::excess_phase::{
    Assessment, CorrectionConfig, ExcessPhaseConfig, ExcessPhaseInput, assess_excess_phase,
    propose_correction,
};

const CASE: &str = "ra05_delay_excess_phase";
const CASE_ID: &str = "autoeq-qa.ra05-delay-excess-phase.v1";
const TOL_BULK_ABS: f64 = 1e-6;
const TOL_PHASE_ABS: f64 = 1e-9;
const TOL_GD_ABS: f64 = 0.35;

fn vec_f64(value: &serde_json::Value, key: &str) -> Vec<f64> {
    serde_json::from_value(value[key].clone())
        .unwrap_or_else(|_| panic!("{CASE}: golden is missing numeric array `{key}`"))
}

fn num(value: &serde_json::Value, key: &str) -> f64 {
    value[key]
        .as_f64()
        .unwrap_or_else(|| panic!("{CASE}: golden lacks `{key}`"))
}

fn config_of(ref_json: &serde_json::Value) -> ExcessPhaseConfig {
    let band: Vec<f64> = serde_json::from_value(ref_json["analysis_band_hz"].clone()).unwrap();
    ExcessPhaseConfig {
        taper_oct: num(ref_json, "taper_oct"),
        snr_floor_db: num(ref_json, "snr_floor_db"),
        min_valid_fraction: num(ref_json, "min_valid_fraction"),
        smooth_narrow_oct: num(ref_json, "smooth_narrow_oct"),
        smooth_wide_oct: num(ref_json, "smooth_wide_oct"),
        consistency_tol_ms: num(ref_json, "consistency_tol_ms"),
        strict_dips: false,
        dip_depth_db: num(ref_json, "dip_depth_db"),
        analysis_band_hz: (band[0], band[1]),
    }
}

#[test]
fn wolfram_ra05_delay_excess_phase() {
    let ref_json = require_reference(CASE, "ra05_delay_excess_phase.wls");
    assert_case_id(&ref_json, CASE_ID, CASE);

    let freqs = vec_f64(&ref_json, "grid_hz");
    let mag = vec_f64(&ref_json, "magnitude_db");
    let snr = vec_f64(&ref_json, "snr_db");
    let sr = num(&ref_json, "sample_rate_hz");
    assert_eq!(freqs.len(), 256, "{CASE}: expected 256 grid points");
    assert_eq!(mag.len(), freqs.len(), "{CASE}: grid/magnitude length zip");
    assert_eq!(snr.len(), freqs.len(), "{CASE}: grid/SNR length zip");
    assert_eq!(sr, 48000.0, "{CASE}: sample rate");
    let cfg = config_of(&ref_json);

    // --- Pure delay: bulk removal leaves zero excess, Supported. ---
    let delay_input = ExcessPhaseInput {
        freqs_hz: freqs.clone(),
        magnitude_db: mag.clone(),
        phase_deg: Some(vec_f64(&ref_json, "phase_delay_deg")),
        snr_db: snr.clone(),
        sample_rate_hz: sr,
    };
    let Assessment::Supported(report) = assess_excess_phase(&delay_input, &cfg) else {
        panic!("{CASE}: pure delay must assess Supported");
    };
    assert!(report.gates_passed, "{CASE}: gates must pass");
    let bulk_err = (report.bulk_delay_s - num(&ref_json, "expected_bulk_delay_s")).abs();
    assert!(
        bulk_err <= TOL_BULK_ABS,
        "bulk delay: rust={:.12e} expected={:.12e} abs_err={bulk_err:.3e}",
        report.bulk_delay_s,
        num(&ref_json, "expected_bulk_delay_s")
    );
    assert!(
        (report.bulk_delay_samples - num(&ref_json, "delay_tau_s") * sr).abs() <= 1.0,
        "{CASE}: bulk delay samples"
    );
    assert_eq!(report.valid_fraction, 1.0, "{CASE}: full SNR coverage");
    let worst_gd = report
        .excess_gd_ms
        .iter()
        .zip(report.valid.iter())
        .filter(|&(_, v)| *v)
        .map(|(&g, _)| g.abs())
        .fold(0.0f64, f64::max);
    assert!(
        worst_gd <= 0.2,
        "{CASE}: residual excess group delay {worst_gd:.3e} ms"
    );
    let worst_corr = report
        .correction_phase_rad
        .iter()
        .map(|c| c.abs())
        .fold(0.0f64, f64::max);
    assert!(
        worst_corr <= TOL_PHASE_ABS,
        "{CASE}: correction phase {worst_corr:.3e} rad"
    );

    // --- All-pass: smoothed excess GD matches the analytic curve. ---
    let ap_input = ExcessPhaseInput {
        freqs_hz: freqs.clone(),
        magnitude_db: mag.clone(),
        phase_deg: Some(vec_f64(&ref_json, "phase_allpass_deg")),
        snr_db: snr.clone(),
        sample_rate_hz: sr,
    };
    let Assessment::Supported(ap_report) = assess_excess_phase(&ap_input, &cfg) else {
        panic!("{CASE}: all-pass must assess Supported");
    };
    let probe = num(&ref_json, "allpass_probe_hz");
    let idx = freqs
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| (*a - probe).abs().partial_cmp(&(*b - probe).abs()).unwrap())
        .map(|(i, _)| i)
        .unwrap();
    let gd_err = (ap_report.excess_gd_ms[idx] - num(&ref_json, "expected_allpass_gd_ms")).abs();
    assert!(
        gd_err <= TOL_GD_ABS,
        "all-pass GD @ {probe:.3} Hz: rust={:.6} expected={:.6} abs_err={gd_err:.3e} ms",
        ap_report.excess_gd_ms[idx],
        num(&ref_json, "expected_allpass_gd_ms")
    );

    // --- Bounded correction: causal FIR within the latency budget. ---
    let ccfg = CorrectionConfig {
        max_taps: num(&ref_json, "correction_max_taps") as usize,
        max_added_latency_ms: num(&ref_json, "correction_max_latency_ms"),
    };
    let proposal = propose_correction(&ap_report, &freqs, sr, &ccfg).expect("correction");
    assert_eq!(proposal.fir_taps.len(), 512, "{CASE}: tap count");
    assert!(
        proposal.added_latency_ms <= 4.0 + 1e-9 && proposal.added_latency_ms > 0.0,
        "{CASE}: added latency {}",
        proposal.added_latency_ms
    );
    assert!(
        (0.0..=1.0).contains(&proposal.pre_ring_energy_ratio),
        "{CASE}: pre-ring energy ratio"
    );
    assert!(
        proposal.max_magnitude_deviation_db < 3.0,
        "{CASE}: unity-magnitude deviation {}",
        proposal.max_magnitude_deviation_db
    );
    assert!(
        proposal.fir_taps.iter().all(|t| t.is_finite()),
        "{CASE}: finite taps"
    );

    emit_result(&QaResult {
        case: CASE_ID.to_string(),
        pass: true,
        max_rel_error: 0.0,
        max_abs_error: bulk_err.max(gd_err),
        tolerance: TOL_GD_ABS,
        tolerance_kind: "abs".to_string(),
        provenance: provenance(),
    });
}
