//! Third-octave early/late energy from declared, matched room IR captures.

use math_audio_dsp::rir_early_late::{
    EARLY_LATE_SPLIT_MS, envelope_peak, split_early_late, sub_lowpass_envelope_peak,
    third_octave_contributions,
};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReportCurve {
    pub freq: Vec<f64>,
    pub spl: Vec<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EarlyLateSide {
    pub direct_sample: usize,
    pub full: ReportCurve,
    pub early: ReportCurve,
    pub late: ReportCurve,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CapturedEarlyLate {
    pub method: String,
    pub reference: String,
    pub smoothing: String,
    pub split_ms: f64,
    pub direct_reference: String,
    pub valid_band_hz: [f64; 2],
    pub pre: EarlyLateSide,
    pub post: EarlyLateSide,
    pub scope: String,
}

fn one_side(
    ir: &[f64],
    sample_rate: f64,
    valid_band_hz: [f64; 2],
    sub_or_lfe: bool,
) -> Result<EarlyLateSide, String> {
    if ir.is_empty()
        || ir
            .iter()
            .any(|v| !v.is_finite() || v.abs() > f32::MAX as f64)
    {
        return Err("non-finite, out-of-range, or empty captured IR".into());
    }
    let samples: Vec<f32> = ir.iter().map(|&v| v as f32).collect();
    let (direct_sample, peak) = if sub_or_lfe {
        sub_lowpass_envelope_peak(&samples, sample_rate)
    } else {
        envelope_peak(&samples, sample_rate)
    };
    if peak <= 0.0 {
        return Err("captured IR has no supported direct reference".into());
    }
    let (early, late) = split_early_late(&samples, direct_sample, sample_rate, EARLY_LATE_SPLIT_MS);
    if early.is_empty() || late.is_empty() {
        return Err("captured IR has no complete 20 ms early split and late window".into());
    }
    if early.iter().all(|v| *v == 0.0) || late.iter().all(|v| *v == 0.0) {
        return Err("early or late captured IR segment has no energy".into());
    }
    let edge = 2.0_f64.powf(1.0 / 6.0);
    let bands: Vec<_> = third_octave_contributions(&early, &late, sample_rate)
        .into_iter()
        .filter(|band| {
            band.centre_hz / edge >= valid_band_hz[0]
                && band.centre_hz * edge <= valid_band_hz[1]
                && band.centre_hz * edge < sample_rate / 2.0
        })
        .collect();
    if bands.len() < 2 {
        return Err(
            "declared capture band contains fewer than two complete third-octave bands".into(),
        );
    }
    if bands.iter().any(|band| {
        !band.full_db.is_finite() || !band.early_db.is_finite() || !band.late_db.is_finite()
    }) {
        return Err("non-finite third-octave energy contribution".into());
    }
    let freq: Vec<f64> = bands.iter().map(|band| band.centre_hz).collect();
    Ok(EarlyLateSide {
        direct_sample,
        full: ReportCurve {
            freq: freq.clone(),
            spl: bands.iter().map(|band| band.full_db).collect(),
        },
        early: ReportCurve {
            freq: freq.clone(),
            spl: bands.iter().map(|band| band.early_db).collect(),
        },
        late: ReportCurve {
            freq,
            spl: bands.iter().map(|band| band.late_db).collect(),
        },
    })
}

pub(super) fn build_pair(
    pre: &[f64],
    post: &[f64],
    sample_rate: f64,
    valid_band_hz: [f64; 2],
    sub_or_lfe: bool,
) -> Result<CapturedEarlyLate, String> {
    if !sample_rate.is_finite()
        || sample_rate <= 0.0
        || !valid_band_hz[0].is_finite()
        || !valid_band_hz[1].is_finite()
        || valid_band_hz[0] <= 0.0
        || valid_band_hz[1] <= valid_band_hz[0]
    {
        return Err("invalid sample rate or declared capture band".into());
    }
    let pre = one_side(pre, sample_rate, valid_band_hz, sub_or_lfe)?;
    let post = one_side(post, sample_rate, valid_band_hz, sub_or_lfe)?;
    Ok(CapturedEarlyLate {
        method: "incoherent_band_energy".into(),
        reference: "full_peak_band".into(),
        smoothing: "third_octave".into(),
        split_ms: EARLY_LATE_SPLIT_MS,
        direct_reference: if sub_or_lfe {
            "120 Hz lowpass envelope peak"
        } else {
            "broadband envelope peak"
        }
        .into(),
        valid_band_hz,
        pre,
        post,
        scope: "declared matched room IRs; each capture's early and late third-octave energies share that capture's full peak-band reference; full is the incoherent sum of early and late band energies, not a complex pressure sum; 20 ms from the declared direct reference through the end of each recorded IR; levels do not compare absolute output across captures or establish passive-room damping".into(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matched_impulses_keep_early_late_level_difference() {
        let mut ir = vec![0.0; 4_800];
        ir[100] = 1.0;
        ir[1_540] = 0.5;
        let result = build_pair(&ir, &ir, 48_000.0, [80.0, 12_000.0], false).unwrap();
        assert_eq!(result.pre.direct_sample, 100);
        assert_eq!(result.pre.full.freq, result.pre.early.freq);
        assert_eq!(result.pre.full.freq, result.pre.late.freq);
        let index = result
            .pre
            .full
            .freq
            .iter()
            .position(|&f| f == 1_000.0)
            .unwrap();
        assert!((result.pre.early.spl[index] - result.pre.late.spl[index] - 6.0206).abs() < 0.02);
        assert_eq!(result.pre.early.spl, result.post.early.spl);
    }

    #[test]
    fn short_or_unsupported_ir_stays_unavailable() {
        let mut ir = vec![0.0; 1_000];
        ir[100] = 1.0;
        ir[900] = 0.5;
        assert!(build_pair(&ir, &ir, 48_000.0, [80.0, 12_000.0], false).is_err());
        let mut long = vec![0.0; 4_800];
        long[100] = 1.0;
        long[1_540] = 0.5;
        assert!(build_pair(&long, &long, 48_000.0, [7_900.0, 8_000.0], false).is_err());
    }
}
