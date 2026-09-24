//! Capture-bound time-frequency diagnostics for the report.

use math_audio_dsp::rir_waterfall::{
    WaterfallConfig, WaterfallGrid, detect_resonances, waterfall_grid_at,
};
use math_audio_dsp::rir_wavelet::{WaveletConfig, wavelet_heatmap_at};
use serde::{Deserialize, Serialize};

const POST_MS: f64 = 500.0;
const HALF_WINDOW_MS: f64 = 16.0;
const WAVELET_HALF_SUPPORT_MS: f64 = 20.0;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReportGrid {
    pub times_ms: Vec<f64>,
    pub freqs_hz: Vec<f64>,
    /// `mags_db[time][frequency]`, relative to the full input grid peak.
    pub mags_db: Vec<Vec<f32>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReportResonance {
    pub freq_hz: f64,
    pub level_db: f64,
    pub decay_time_s: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WaterfallSide {
    pub grid: ReportGrid,
    pub resonances: Vec<ReportResonance>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CapturedWaterfalls {
    pub method: String,
    pub reference: String,
    pub valid_band_hz: [f64; 2],
    pub window_ms: f64,
    pub hop_ms: f64,
    pub post_ms: f64,
    pub pre: WaterfallSide,
    pub post: WaterfallSide,
    pub scope: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CapturedWavelets {
    pub method: String,
    pub reference: String,
    pub valid_band_hz: [f64; 2],
    pub cycles: f64,
    pub freqs_per_octave: f64,
    pub hop_ms: f64,
    pub display_range_db: [f64; 2],
    pub pre: WaveletSide,
    pub post: WaveletSide,
    pub scope: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WaveletSide {
    pub freqs_hz: Vec<f64>,
    pub times_ms: Vec<f64>,
    /// `mags_db[frequency][time]`, relative to the input grid peak.
    pub mags_db: Vec<Vec<f32>>,
}

fn wavelet_side(ir: &[f64], sample_rate: f64, band: [f64; 2]) -> Result<WaveletSide, String> {
    if ir.is_empty()
        || ir
            .iter()
            .any(|v| !v.is_finite() || v.abs() > f32::MAX as f64)
    {
        return Err("non-finite or empty captured IR".into());
    }
    if ir.iter().all(|v| *v == 0.0) {
        return Err("silent captured IR has no wavelet level reference".into());
    }
    let direct = ir
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.abs().total_cmp(&b.1.abs()))
        .map_or(0, |(index, _)| index);
    let required = ((POST_MS + WAVELET_HALF_SUPPORT_MS) * sample_rate / 1_000.0).ceil() as usize;
    if ir.len().saturating_sub(direct) <= required {
        return Err("IR does not contain the complete 500 ms post-peak wavelet window and low-frequency filter support".into());
    }
    let samples: Vec<f32> = ir.iter().map(|&sample| sample as f32).collect();
    let config = WaveletConfig {
        freqs_per_octave: 6.0,
        max_freqs: 64,
        max_frames: 100,
    };
    let full = wavelet_heatmap_at(&samples, sample_rate, direct, &config);
    if full.times_ms.len() < 2 || full.freqs_hz.is_empty() {
        return Err("wavelet analysis yielded no supported frames".into());
    }
    let indices: Vec<_> = full
        .freqs_hz
        .iter()
        .enumerate()
        .filter(|(_, freq)| **freq >= band[0] && **freq <= band[1])
        .map(|(index, _)| index)
        .collect();
    if indices.len() < 2 {
        return Err("declared capture band contains fewer than two wavelet centers".into());
    }
    Ok(WaveletSide {
        times_ms: full.times_ms,
        freqs_hz: indices.iter().map(|&i| full.freqs_hz[i]).collect(),
        mags_db: indices.iter().map(|&i| full.mags_db[i].clone()).collect(),
    })
}

pub(super) fn wavelet_pair(
    pre: &[f64],
    post: &[f64],
    sample_rate: f64,
    band: [f64; 2],
) -> Result<CapturedWavelets, String> {
    if !sample_rate.is_finite()
        || sample_rate <= 0.0
        || !band[0].is_finite()
        || !band[1].is_finite()
        || band[0] <= 0.0
        || band[1] <= band[0]
    {
        return Err("invalid sample rate or declared capture band".into());
    }
    Ok(CapturedWavelets {
        method: "complex_morlet_three_cycle_v1".into(),
        reference: "each_full_grid_peak".into(),
        valid_band_hz: band,
        cycles: 3.0,
        freqs_per_octave: 6.0,
        hop_ms: 1.0,
        display_range_db: [-30.0, 0.0],
        pre: wavelet_side(pre, sample_rate, band)?,
        post: wavelet_side(post, sample_rate, band)?,
        scope: "declared matched IRs; complex Morlet CWT with three cycles and six log centers per octave, 1 ms raw frame hop over −5…500 ms relative to each broadband absolute peak; max-pooled to at most 64 frequency × 100 time cells; −30…0 dB relative to each full input grid peak; displayed colors do not compare absolute output between captures or establish a reflection path".into(),
    })
}

fn one_side(
    ir: &[f64],
    sample_rate: f64,
    valid_band_hz: [f64; 2],
) -> Result<WaterfallSide, String> {
    if ir.is_empty()
        || ir
            .iter()
            .any(|v| !v.is_finite() || v.abs() > f32::MAX as f64)
    {
        return Err("non-finite or empty captured IR".into());
    }
    if ir.iter().all(|v| *v == 0.0) {
        return Err("silent captured IR has no waterfall level reference".into());
    }
    let direct = ir
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.abs().total_cmp(&b.1.abs()))
        .map_or(0, |(index, _)| index);
    let required = ((POST_MS + HALF_WINDOW_MS) * sample_rate / 1_000.0).ceil() as usize;
    if ir.len().saturating_sub(direct) <= required {
        return Err(
            "IR does not contain the complete 500 ms post-peak window and analysis half-window"
                .into(),
        );
    }
    let samples: Vec<f32> = ir.iter().map(|&sample| sample as f32).collect();
    let config = WaterfallConfig {
        max_frames: 100,
        max_bins: 64,
        ..WaterfallConfig::default()
    };
    let full = waterfall_grid_at(&samples, sample_rate, direct, &config);
    if full.times_ms.is_empty() || full.freqs_hz.is_empty() {
        return Err("waterfall analysis yielded no supported frames".into());
    }
    let indices: Vec<_> = full
        .freqs_hz
        .iter()
        .enumerate()
        .filter(|(_, freq)| **freq >= valid_band_hz[0] && **freq <= valid_band_hz[1])
        .map(|(index, _)| index)
        .collect();
    if indices.len() < 2 {
        return Err("declared capture band contains fewer than two waterfall bins".into());
    }
    let grid = WaterfallGrid {
        times_ms: full.times_ms,
        freqs_hz: indices.iter().map(|&i| full.freqs_hz[i]).collect(),
        mags_db: full
            .mags_db
            .iter()
            .map(|row| indices.iter().map(|&i| row[i]).collect())
            .collect(),
    };
    let resonances = detect_resonances(&grid, &config)
        .into_iter()
        .map(|item| ReportResonance {
            freq_hz: item.freq_hz,
            level_db: item.level_db,
            decay_time_s: (item.decay_time_s.is_finite() && item.decay_time_s > 0.0)
                .then_some(item.decay_time_s),
        })
        .collect();
    Ok(WaterfallSide {
        grid: ReportGrid {
            times_ms: grid.times_ms,
            freqs_hz: grid.freqs_hz,
            mags_db: grid.mags_db,
        },
        resonances,
    })
}

pub(super) fn waterfall_pair(
    pre: &[f64],
    post: &[f64],
    sample_rate: f64,
    valid_band_hz: [f64; 2],
) -> Result<CapturedWaterfalls, String> {
    if !sample_rate.is_finite()
        || sample_rate <= 0.0
        || !valid_band_hz[0].is_finite()
        || !valid_band_hz[1].is_finite()
        || valid_band_hz[0] <= 0.0
        || valid_band_hz[1] <= valid_band_hz[0]
    {
        return Err("invalid sample rate or declared capture band".into());
    }
    Ok(CapturedWaterfalls {
        method: "hann_stft_waterfall_v1".into(),
        reference: "each_full_grid_peak".into(),
        valid_band_hz,
        window_ms: 32.0,
        hop_ms: 2.0,
        post_ms: POST_MS,
        pre: one_side(pre, sample_rate, valid_band_hz)?,
        post: one_side(post, sample_rate, valid_band_hz)?,
        scope: "declared matched IRs; Hann STFT, 32 ms window and 2 ms hop; −5…500 ms relative to each broadband absolute peak; grid max-pooled to at most 100 time frames × 64 frequency bins; levels relative to each full input grid peak and therefore not a between-capture output comparison; resonance peaks at 60 ms with 20–200 ms fitted decay; no passive-room damping or audibility verdict".into(),
    })
}

#[cfg(test)]
mod tests {
    use super::{waterfall_pair, wavelet_pair};

    #[test]
    fn matched_capture_grid_is_bounded_and_band_limited() {
        let sample_rate = 8_000.0;
        let mut ir = vec![0.0; 8_000];
        ir[80] = 1.0;
        for (index, sample) in ir.iter_mut().enumerate().skip(81) {
            let time = (index - 80) as f64 / sample_rate;
            *sample =
                0.2 * (-time / 0.12).exp() * (2.0 * std::f64::consts::PI * 110.0 * time).sin();
        }
        let result = waterfall_pair(&ir, &ir, sample_rate, [20.0, 1_000.0]).unwrap();
        assert_eq!(result.method, "hann_stft_waterfall_v1");
        assert_eq!(result.reference, "each_full_grid_peak");
        assert!(result.pre.grid.times_ms.len() <= 100);
        assert!((2..=64).contains(&result.pre.grid.freqs_hz.len()));
        assert!(
            result
                .pre
                .grid
                .freqs_hz
                .iter()
                .all(|f| (20.0..=1_000.0).contains(f))
        );
        assert_eq!(
            result.pre.grid.mags_db.len(),
            result.pre.grid.times_ms.len()
        );
        assert!(
            result
                .pre
                .grid
                .mags_db
                .iter()
                .all(|row| row.len() == result.pre.grid.freqs_hz.len())
        );
    }

    #[test]
    fn short_capture_cannot_fake_a_complete_decay_window() {
        let mut ir = vec![0.0; 4_000];
        ir[80] = 1.0;
        assert!(
            waterfall_pair(&ir, &ir, 8_000.0, [20.0, 1_000.0])
                .unwrap_err()
                .contains("complete 500 ms")
        );
    }

    #[test]
    fn silent_capture_has_no_relative_level_reference() {
        let ir = vec![0.0; 8_000];
        assert!(
            waterfall_pair(&ir, &ir, 8_000.0, [100.0, 1_000.0])
                .unwrap_err()
                .contains("silent captured IR")
        );
        assert!(
            wavelet_pair(&ir, &ir, 8_000.0, [100.0, 1_000.0])
                .unwrap_err()
                .contains("silent captured IR")
        );
    }

    #[test]
    fn wavelet_pair_keeps_frequency_time_orientation_and_declared_band() {
        let mut ir = vec![0.0; 8_000];
        ir[80] = 1.0;
        let result = wavelet_pair(&ir, &ir, 8_000.0, [100.0, 1_000.0]).unwrap();
        assert_eq!(result.reference, "each_full_grid_peak");
        assert!((2..=64).contains(&result.pre.freqs_hz.len()));
        assert!((2..=100).contains(&result.pre.times_ms.len()));
        assert!(
            result
                .pre
                .freqs_hz
                .iter()
                .all(|f| (100.0..=1_000.0).contains(f))
        );
        assert!(result.pre.mags_db.iter().all(|row| {
            row.len() == result.pre.times_ms.len()
                && row
                    .iter()
                    .all(|value| value.is_finite() && (-30.0..=0.0).contains(value))
        }));
    }
}
