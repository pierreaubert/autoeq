//! Measured-room early/late energy for the optimization DSP JSON.
//!
//! When a channel declares a measured impulse response at optimization time,
//! this module decomposes it into third-octave early/late band energies with
//! the published math-audio shared-reference primitive and shapes the result
//! for the optional [`roomeq_model::ChannelEarlyLateCurves`] field. The
//! viewer renders that field only when its method, reference, smoothing,
//! split, and 1–8 kHz grid contract holds; otherwise the report cell stays
//! pending. Anything the IR cannot support returns `None` — never a
//! fabricated curve.
//!
//! The twin mapping for bound operator-capture verification lives in
//! `roomeq-cli/src/verification/ir_views/early_late.rs` (pre/post capture
//! pairs). This module covers the single measured IR available at
//! optimization time (pre-correction only).

use math_audio_dsp::rir_early_late::{
    EARLY_LATE_SPLIT_MS, envelope_peak, split_early_late, sub_lowpass_envelope_peak,
    third_octave_contributions,
};
use roomeq_model::{ChannelEarlyLateCurves, CurveData};

/// Third-octave band edge ratio (half a third octave each side).
const BAND_EDGE: f64 = 1.0717734625362931; // 2^(1/6)

/// Lowest and highest third-octave centers the summary ratio needs.
const RATIO_BAND_HZ: [f64; 2] = [1000.0, 8000.0];

/// Build the channel early/late report from a measured room IR.
///
/// Returns `None` when the IR is missing, non-finite, out of range, has no
/// supported direct reference or complete 20 ms split, carries no energy in
/// either segment, yields non-finite band energies, or cannot cover the full
/// 1–8 kHz viewer band below Nyquist.
pub(crate) fn measured_early_late_curves(
    samples: &[f32],
    sample_rate: f64,
    sub_or_lfe: bool,
) -> Option<ChannelEarlyLateCurves> {
    if samples.is_empty() || samples.iter().any(|v| !v.is_finite() || v.abs() > f32::MAX) {
        return None;
    }
    let (direct_sample, peak) = if sub_or_lfe {
        sub_lowpass_envelope_peak(samples, sample_rate)
    } else {
        envelope_peak(samples, sample_rate)
    };
    if peak <= 0.0 {
        return None;
    }
    let (early, late) = split_early_late(samples, direct_sample, sample_rate, EARLY_LATE_SPLIT_MS);
    if early.is_empty() || late.is_empty() {
        return None;
    }
    if early.iter().all(|v| *v == 0.0) || late.iter().all(|v| *v == 0.0) {
        return None;
    }
    // Keep every third-octave band fully below Nyquist; the viewer ratio
    // uses the 1–8 kHz subset, and coverage below is enforced on the grid.
    let bands: Vec<_> = third_octave_contributions(&early, &late, sample_rate)
        .into_iter()
        .filter(|band| band.centre_hz * BAND_EDGE < sample_rate / 2.0)
        .collect();
    if bands.len() < 2
        || bands.iter().any(|band| {
            ![band.full_db, band.early_db, band.late_db]
                .iter()
                .all(|v| v.is_finite())
        })
    {
        return None;
    }
    let freq: Vec<f64> = bands.iter().map(|band| band.centre_hz).collect();
    if freq[0] > RATIO_BAND_HZ[0] || freq[freq.len() - 1] < RATIO_BAND_HZ[1] {
        return None;
    }
    let curve = |values: Vec<f64>| CurveData {
        freq: freq.clone(),
        spl: values,
        ..CurveData::default()
    };
    Some(ChannelEarlyLateCurves {
        method: String::from("incoherent_band_energy"),
        reference: String::from("full_peak_band"),
        smoothing: String::from("third_octave"),
        split_ms: EARLY_LATE_SPLIT_MS,
        basis: String::from("measured_room_ir"),
        direct_reference: String::from(if sub_or_lfe {
            "120 Hz lowpass envelope peak"
        } else {
            "broadband envelope peak"
        }),
        valid_band_hz: [freq[0] / BAND_EDGE, freq[freq.len() - 1] * BAND_EDGE],
        full: curve(bands.iter().map(|band| band.full_db).collect()),
        early: curve(bands.iter().map(|band| band.early_db).collect()),
        late: curve(bands.iter().map(|band| band.late_db).collect()),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Dirac plus a half-amplitude tap 30 ms later: the late tap sits fully
    /// past the 20 ms split, so early exceeds late by 20·log10(2) ≈ 6.02 dB
    /// in every band.
    fn two_tap_ir() -> Vec<f32> {
        let mut ir = vec![0.0; 4_800];
        ir[100] = 1.0;
        ir[100 + 1_440] = 0.5;
        ir
    }

    #[test]
    fn two_tap_ir_reports_viewer_contract_and_six_db_split() {
        let report = measured_early_late_curves(&two_tap_ir(), 48_000.0, false)
            .expect("complete measured IR reports");
        assert_eq!(report.method, "incoherent_band_energy");
        assert_eq!(report.reference, "full_peak_band");
        assert_eq!(report.smoothing, "third_octave");
        assert_eq!(report.split_ms, 20.0);
        assert_eq!(report.basis, "measured_room_ir");
        assert_eq!(report.direct_reference, "broadband envelope peak");
        for curve in [&report.full, &report.early, &report.late] {
            assert_eq!(curve.freq, report.full.freq);
            assert!(curve.spl.iter().all(|v| v.is_finite()));
        }
        assert!(report.full.freq[0] <= 1000.0);
        assert!(report.full.freq[report.full.freq.len() - 1] >= 8000.0);
        assert!(report.valid_band_hz[0] <= 1000.0);
        assert!(report.valid_band_hz[1] >= 8000.0);
        let index = report
            .full
            .freq
            .iter()
            .position(|&f| f == 1000.0)
            .expect("1 kHz third-octave center");
        let split = report.early.spl[index] - report.late.spl[index];
        assert!((split - 6.0206).abs() < 0.05, "early/late split {split}");
        // JSON shape matches the viewer contract keys exactly.
        let json = serde_json::to_value(&report).unwrap();
        for key in [
            "method",
            "reference",
            "smoothing",
            "split_ms",
            "full",
            "early",
            "late",
        ] {
            assert!(json.get(key).is_some(), "missing viewer key {key}");
        }
    }

    #[test]
    fn sub_path_uses_lowpass_reference() {
        // Denser taps: the 120 Hz lowpass reference shifts the direct index
        // past a lone onset, so both sides of the 20 ms split need energy.
        let mut ir = vec![0.0; 4_800];
        ir[100] = 1.0;
        ir[600] = 0.5;
        ir[2_000] = 0.25;
        let report = measured_early_late_curves(&ir, 48_000.0, true).expect("sub IR reports");
        assert_eq!(report.direct_reference, "120 Hz lowpass envelope peak");
        assert_eq!(report.split_ms, 20.0);
    }

    #[test]
    fn incomplete_ir_stays_pending() {
        assert!(measured_early_late_curves(&[], 48_000.0, false).is_none());
        assert!(measured_early_late_curves(&[0.0; 64], 48_000.0, false).is_none());
        // Late window past the IR end: no complete split.
        assert!(
            measured_early_late_curves(&two_tap_ir()[..200].to_vec(), 48_000.0, false).is_none()
        );
        let mut non_finite = two_tap_ir();
        non_finite[100] = f32::NAN;
        assert!(measured_early_late_curves(&non_finite, 48_000.0, false).is_none());
        // 8 kHz sample rate cannot cover the viewer 1–8 kHz band.
        assert!(measured_early_late_curves(&two_tap_ir(), 8_000.0, false).is_none());
    }
}
