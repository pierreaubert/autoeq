//! Measured-room acoustics for the optimization DSP JSON.
//!
//! When a channel declares a measured impulse response at optimization time,
//! this module derives the two math-audio report primitives the viewer
//! renders per channel:
//!
//! - [`measured_early_reflections`]: band-limited 1–8 kHz early-reflection
//!   table (`bandlimited_early_reflection_table_v1`, −15 dBFS threshold,
//!   15 ms window). Only the pre-correction side exists at optimization
//!   time; `post` stays empty until a post-correction IR is measured.
//! - [`measured_octave_t60`]: nine-band Schroeder octave T60 with the
//!   fit-range + `min_r2` policy shared with bound capture verification.
//!
//! The twin mapping for pre/post operator-capture pairs lives in
//! `roomeq-cli/src/verification/ir_views.rs`. Anything the IR cannot
//! support returns `None` — never a fabricated table.

use math_audio_dsp::rir_waterfall::{
    WaterfallConfig, WaterfallGrid, detect_resonances, waterfall_grid_at,
};
use math_audio_dsp::rir_wavelet::{WaveletConfig, wavelet_heatmap_at};
use math_rir::report::reflection_table::{ReflectionTableConfig, early_reflection_table};
use math_rir::report::t60_batch::{T60BatchConfig, T60FitRange, analyze_t60_octaves};
use roomeq_model::{
    ChannelEarlyReflections, ChannelOctaveT60, ChannelReflectionEvent, ChannelResonanceDecay,
    ChannelResonanceDecays, ChannelT60Band, ChannelWaterfall, ChannelWavelet,
};

/// Detection threshold in dBFS below the filtered direct peak.
///
/// Matches the viewer contract exactly: gains arrive in `[-15, 0]` dBFS.
const REFLECTION_THRESHOLD_DB: f64 = 15.0;

/// Reflection search window in milliseconds after the direct peak.
const REFLECTION_WINDOW_MS: f64 = 15.0;

/// Viewer event cap; longer tails stay pending rather than silently partial.
const MAX_REFLECTION_EVENTS: usize = 64;

/// Direct-sound reference vocabulary shared with capture verification.
const DIRECT_REFERENCE: &str = "0 dB = 1–8 kHz filtered direct peak; times are post-direct";

/// Build the channel early-reflection table from a measured room IR.
///
/// Returns `None` for missing, non-finite, or silent IRs, and when the
/// detected table would exceed the viewer event cap.
pub fn measured_early_reflections(
    samples: &[f32],
    sample_rate: f64,
) -> Option<ChannelEarlyReflections> {
    if samples.is_empty() || samples.iter().any(|sample| !sample.is_finite()) {
        return None;
    }
    let config = ReflectionTableConfig {
        threshold_db: REFLECTION_THRESHOLD_DB,
        window_ms: REFLECTION_WINDOW_MS,
        ..ReflectionTableConfig::default()
    };
    let table = early_reflection_table(samples, sample_rate, &config);
    if table.direct_peak <= 0.0 {
        return None;
    }
    if table.reflections.len() > MAX_REFLECTION_EVENTS {
        return None;
    }
    let mut pre = Vec::with_capacity(table.reflections.len());
    for event in table.reflections {
        if !event.gain_db.is_finite()
            || !event.delay_ms.is_finite()
            || !event.path_difference_m.is_finite()
            || !event.first_dip_hz.is_finite()
            || !event.comb_ripple_db.is_finite()
        {
            return None;
        }
        let gain = 10.0_f64.powf(event.gain_db / 20.0);
        let denominator = (1.0 - gain).abs();
        pre.push(ChannelReflectionEvent {
            gain_dbfs: event.gain_db,
            time_ms: event.delay_ms,
            distance_cm: event.path_difference_m * 100.0,
            first_dip_hz: event.first_dip_hz,
            ripple_db: (denominator > 1e-12).then(|| 20.0 * ((1.0 + gain) / denominator).log10()),
        });
    }
    Some(ChannelEarlyReflections {
        basis: String::from("measured_room_ir"),
        method: String::from("bandlimited_early_reflection_table_v1"),
        band_hz: [1000.0, 8000.0],
        threshold_dbfs: -15.0,
        direct_reference: String::from(DIRECT_REFERENCE),
        pre,
        post: Vec::new(),
    })
}

/// Build the nine-band octave T60 report from a measured room IR.
///
/// Returns `None` for missing or non-finite IRs. Bands the batched analysis
/// cannot support arrive invalid with machine-readable reasons; EDT-only
/// fits stay invalid because EDT alone is not late-decay T60.
pub fn measured_octave_t60(samples: &[f32], sample_rate: f64) -> Option<ChannelOctaveT60> {
    if samples.is_empty() || samples.iter().any(|sample| !sample.is_finite()) {
        return None;
    }
    let config = T60BatchConfig::default();
    let bands = analyze_t60_octaves(samples, sample_rate, &config)
        .into_iter()
        .map(|band| {
            let late_fit = matches!(band.fit_range, T60FitRange::T30 | T60FitRange::T20);
            let valid = band.valid && late_fit;
            ChannelT60Band {
                centre_hz: band.centre_hz,
                t60_s: valid
                    .then_some(band.t60_s)
                    .filter(|value| value.is_finite()),
                fit_range: match band.fit_range {
                    T60FitRange::T30 => Some(String::from("T30")),
                    T60FitRange::T20 => Some(String::from("T20")),
                    T60FitRange::Edt => Some(String::from("EDT")),
                    T60FitRange::None => None,
                },
                r2: if band.r2.is_finite() { band.r2 } else { 0.0 },
                valid,
                reason: if !late_fit && band.valid {
                    String::from("EDT alone is not late-decay T60")
                } else {
                    band.reason
                },
            }
        })
        .collect::<Vec<_>>();
    if bands.len() != 9 {
        return None;
    }
    Some(ChannelOctaveT60 {
        basis: String::from("measured_room_ir"),
        min_r2: config.min_r2,
        bands,
    })
}

/// Post-peak window the waterfall/wavelet grids require, in milliseconds.
const POST_PEAK_MS: f64 = 500.0;

/// Waterfall STFT half-window support past the window end, in milliseconds.
const WATERFALL_HALF_WINDOW_MS: f64 = 16.0;

/// Wavelet low-frequency filter half-support past the window end, in ms.
const WAVELET_HALF_SUPPORT_MS: f64 = 20.0;

/// Viewer grid budget caps shared with bound capture verification.
const GRID_MAX_FRAMES: usize = 100;
const GRID_MAX_BINS: usize = 64;

/// Resonance slice offset after the direct peak, in milliseconds.
const RESONANCE_SLICE_MS: f64 = 60.0;

/// Shared validation for the time-frequency producers: finite non-silent
/// samples, a finite positive sample rate, and a complete 500 ms
/// post-peak window (plus analysis support) after the broadband peak.
/// Returns the direct-peak sample index.
fn time_frequency_direct(samples: &[f32], sample_rate: f64, support_ms: f64) -> Option<usize> {
    if samples.is_empty()
        || !sample_rate.is_finite()
        || sample_rate <= 0.0
        || samples.iter().any(|v| !v.is_finite())
    {
        return None;
    }
    let direct = samples
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.abs().total_cmp(&b.1.abs()))
        .map_or(0, |(index, _)| index);
    if samples[direct] == 0.0 {
        return None;
    }
    let required = ((POST_PEAK_MS + support_ms) * sample_rate / 1000.0).ceil() as usize;
    if samples.len().saturating_sub(direct) <= required {
        return None;
    }
    Some(direct)
}

fn waterfall_grid(samples: &[f32], sample_rate: f64) -> Option<WaterfallGrid> {
    let direct = time_frequency_direct(samples, sample_rate, WATERFALL_HALF_WINDOW_MS)?;
    let config = WaterfallConfig {
        max_frames: GRID_MAX_FRAMES,
        max_bins: GRID_MAX_BINS,
        ..WaterfallConfig::default()
    };
    let grid = waterfall_grid_at(samples, sample_rate, direct, &config);
    if grid.times_ms.len() < 2 || grid.freqs_hz.len() < 2 {
        return None;
    }
    if grid.times_ms.iter().any(|v| !v.is_finite())
        || grid.freqs_hz.iter().any(|v| !v.is_finite())
        || grid.mags_db.len() != grid.times_ms.len()
        || grid
            .mags_db
            .iter()
            .any(|row| row.len() != grid.freqs_hz.len() || row.iter().any(|v| !v.is_finite()))
    {
        return None;
    }
    Some(grid)
}

/// Build the STFT waterfall decay grid from a measured room IR.
///
/// Returns `None` for missing, non-finite, or silent IRs, without a
/// complete 500 ms post-peak window, or when the analysis yields no
/// plottable grid. Levels are relative to the grid's own peak.
pub fn measured_waterfall(
    samples: &[f32],
    sample_rate: f64,
) -> Option<(ChannelWaterfall, ChannelResonanceDecays)> {
    let grid = waterfall_grid(samples, sample_rate)?;
    let config = WaterfallConfig {
        max_frames: GRID_MAX_FRAMES,
        max_bins: GRID_MAX_BINS,
        ..WaterfallConfig::default()
    };
    let decays = detect_resonances(&grid, &config)
        .into_iter()
        .map(|item| ChannelResonanceDecay {
            freq_hz: item.freq_hz,
            level_db: item.level_db,
            decay_time_s: (item.decay_time_s.is_finite() && item.decay_time_s > 0.0)
                .then_some(item.decay_time_s),
        })
        .collect();
    let band = [grid.freqs_hz[0], grid.freqs_hz[grid.freqs_hz.len() - 1]];
    Some((
        ChannelWaterfall {
            basis: String::from("measured_room_ir"),
            method: String::from("hann_stft_waterfall_v1"),
            reference: String::from("full_grid_peak"),
            valid_band_hz: band,
            window_ms: 32.0,
            hop_ms: 2.0,
            post_ms: POST_PEAK_MS,
            times_ms: grid.times_ms,
            freqs_hz: grid.freqs_hz,
            mags_db: grid.mags_db,
            scope: String::from(
                "single measured room IR (pre-correction); Hann STFT, 32 ms window and 2 ms hop; \
                 -5...500 ms relative to the broadband absolute peak; grid max-pooled to at most \
                 100 time frames x 64 frequency bins; levels relative to the full grid peak and \
                 therefore not a between-channel output comparison; resonance peaks at 60 ms \
                 with 20-200 ms fitted decay; no passive-room damping or audibility verdict",
            ),
        },
        ChannelResonanceDecays {
            basis: String::from("measured_room_ir"),
            method: String::from("hann_stft_waterfall_v1"),
            reference: String::from("full_grid_peak"),
            slice_ms: RESONANCE_SLICE_MS,
            decays,
        },
    ))
}

/// Build the three-cycle wavelet heatmap from a measured room IR.
///
/// Returns `None` under the same conditions as [`measured_waterfall`].
/// Levels are relative to the heatmap's own peak, on the −30…0 dB
/// display range.
pub fn measured_wavelet(samples: &[f32], sample_rate: f64) -> Option<ChannelWavelet> {
    let direct = time_frequency_direct(samples, sample_rate, WAVELET_HALF_SUPPORT_MS)?;
    let config = WaveletConfig {
        freqs_per_octave: 6.0,
        max_freqs: GRID_MAX_BINS,
        max_frames: GRID_MAX_FRAMES,
    };
    let heat = wavelet_heatmap_at(samples, sample_rate, direct, &config);
    if heat.times_ms.len() < 2 || heat.freqs_hz.len() < 2 {
        return None;
    }
    if heat.times_ms.iter().any(|v| !v.is_finite())
        || heat.freqs_hz.iter().any(|v| !v.is_finite())
        || heat.mags_db.len() != heat.freqs_hz.len()
        || heat
            .mags_db
            .iter()
            .any(|row| row.len() != heat.times_ms.len() || row.iter().any(|v| !v.is_finite()))
    {
        return None;
    }
    let band = [heat.freqs_hz[0], heat.freqs_hz[heat.freqs_hz.len() - 1]];
    Some(ChannelWavelet {
        basis: String::from("measured_room_ir"),
        method: String::from("complex_morlet_three_cycle_v1"),
        reference: String::from("full_grid_peak"),
        valid_band_hz: band,
        cycles: 3.0,
        freqs_per_octave: 6.0,
        hop_ms: 1.0,
        display_range_db: [-30.0, 0.0],
        freqs_hz: heat.freqs_hz,
        times_ms: heat.times_ms,
        mags_db: heat.mags_db,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Band-shaped bursts (2 kHz carrier, ~1 ms decay): a bare dirac leaves
    /// no resolvable envelope maxima by design, so each onset rings through
    /// the 1–8 kHz picking band. Direct burst amplitude 1.0 with a
    /// half-amplitude reflection 10 ms later: delay 10 ms, gain −6.02 dB,
    /// first dip 50 Hz, path difference 3.43 m, ripple 9.54 dB.
    fn tap_ir() -> Vec<f32> {
        let rate = 48_000.0;
        let n = 100 + (20.0 * rate / 1000.0) as usize;
        let mut out = vec![0.0f32; n];
        let omega = 2.0 * std::f64::consts::PI * 2000.0 / rate;
        let decay = (-1.0 / (0.001 * rate)).exp();
        for (off, amp) in [(0usize, 1.0f32), (480, 0.5f32)] {
            let start = 100 + off;
            let mut env = f64::from(amp);
            let mut i = start;
            while i < n && env > 1e-4 {
                out[i] += (env * ((i - start) as f64 * omega).cos()) as f32;
                env *= decay;
                i += 1;
            }
        }
        out
    }

    #[test]
    fn tap_ir_reports_viewer_contract_with_known_geometry() {
        let report = measured_early_reflections(&tap_ir(), 48_000.0).expect("tap IR reports");
        assert_eq!(report.basis, "measured_room_ir");
        assert_eq!(report.method, "bandlimited_early_reflection_table_v1");
        assert_eq!(report.band_hz, [1000.0, 8000.0]);
        assert_eq!(report.threshold_dbfs, -15.0);
        assert!(!report.direct_reference.is_empty());
        assert!(report.post.is_empty());
        assert!(!report.pre.is_empty());
        assert!(report.pre.len() <= MAX_REFLECTION_EVENTS);
        let event = &report.pre[0];
        assert!((event.time_ms - 10.0).abs() < 0.6, "{event:?}");
        assert!((event.gain_dbfs + 6.0206).abs() < 1.0, "{event:?}");
        assert!((event.distance_cm - 343.0).abs() < 21.0, "{event:?}");
        assert!((event.first_dip_hz - 50.0).abs() < 3.0, "{event:?}");
        let ripple = event.ripple_db.expect("bounded ripple");
        assert!((ripple - 9.54).abs() < 0.8, "{event:?}");
        for event in &report.pre {
            assert!((-15.0..=0.0).contains(&event.gain_dbfs), "{event:?}");
            assert!(event.time_ms > 0.0 && event.time_ms <= 15.0, "{event:?}");
        }
        let json = serde_json::to_value(&report).unwrap();
        for key in [
            "basis",
            "method",
            "band_hz",
            "threshold_dbfs",
            "direct_reference",
            "pre",
            "post",
        ] {
            assert!(json.get(key).is_some(), "missing viewer key {key}");
        }
    }

    #[test]
    fn silent_or_broken_ir_stays_pending() {
        assert!(measured_early_reflections(&[], 48_000.0).is_none());
        assert!(measured_early_reflections(&[0.0; 64], 48_000.0).is_none());
        let mut broken = tap_ir();
        broken[100] = f32::NAN;
        assert!(measured_early_reflections(&broken, 48_000.0).is_none());
        assert!(measured_octave_t60(&[], 48_000.0).is_none());
        assert!(measured_octave_t60(&broken, 48_000.0).is_none());
        assert!(measured_waterfall(&[], 48_000.0).is_none());
        assert!(measured_waterfall(&[0.0; 48_000], 48_000.0).is_none());
        assert!(measured_waterfall(&broken, 48_000.0).is_none());
        assert!(measured_wavelet(&[], 48_000.0).is_none());
        assert!(measured_wavelet(&[0.0; 48_000], 48_000.0).is_none());
        assert!(measured_wavelet(&broken, 48_000.0).is_none());
    }

    /// One-second 200 Hz decaying tone (τ = 0.2 s): the grids peak at the
    /// tone bin/row near 0 dB (own-peak reference) with matching finite
    /// dimensions, and the 60 ms slice sees the tone as a resonance.
    fn decaying_tone_ir() -> Vec<f32> {
        let rate = 48_000.0;
        (0..48_000)
            .map(|i| {
                let t = i as f64 / rate;
                (2.0 * std::f64::consts::PI * 200.0 * t).cos() as f32 * (-t / 0.2).exp() as f32
            })
            .collect()
    }

    #[test]
    fn decaying_tone_reports_waterfall_contract() {
        let (waterfall, decays) =
            measured_waterfall(&decaying_tone_ir(), 48_000.0).expect("tone IR reports");
        assert_eq!(waterfall.basis, "measured_room_ir");
        assert_eq!(waterfall.method, "hann_stft_waterfall_v1");
        assert_eq!(waterfall.reference, "full_grid_peak");
        assert_eq!(waterfall.window_ms, 32.0);
        assert_eq!(waterfall.hop_ms, 2.0);
        assert_eq!(waterfall.post_ms, 500.0);
        assert!(!waterfall.scope.is_empty());
        assert!(waterfall.times_ms.len() >= 2 && waterfall.times_ms.len() <= 100);
        assert!(waterfall.freqs_hz.len() >= 2 && waterfall.freqs_hz.len() <= 64);
        assert_eq!(waterfall.mags_db.len(), waterfall.times_ms.len());
        let peak = waterfall
            .mags_db
            .iter()
            .flat_map(|row| row.iter())
            .fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        assert!(peak <= 0.0 && peak > -1.0, "own-peak reference {peak}");
        let peak_bin = waterfall
            .freqs_hz
            .iter()
            .enumerate()
            .max_by(|a, b| waterfall.mags_db[0][a.0].total_cmp(&waterfall.mags_db[0][b.0]))
            .map(|(_, &f)| f)
            .unwrap();
        assert!((peak_bin - 200.0).abs() < 60.0, "peak bin {peak_bin}");
        assert_eq!(decays.basis, "measured_room_ir");
        assert_eq!(decays.method, "hann_stft_waterfall_v1");
        assert_eq!(decays.reference, "full_grid_peak");
        assert_eq!(decays.slice_ms, 60.0);
        assert!(!decays.decays.is_empty(), "tone is a resonance");
        for decay in &decays.decays {
            assert!(decay.freq_hz.is_finite() && decay.freq_hz > 0.0);
            assert!(decay.level_db.is_finite() && decay.level_db <= 0.0);
            if let Some(tau) = decay.decay_time_s {
                assert!(tau > 0.0 && tau < 10.0, "decay {tau}");
            }
        }
        let json = serde_json::to_value(&waterfall).unwrap();
        for key in [
            "basis",
            "method",
            "reference",
            "valid_band_hz",
            "times_ms",
            "freqs_hz",
            "mags_db",
        ] {
            assert!(json.get(key).is_some(), "missing viewer key {key}");
        }
    }

    #[test]
    fn decaying_tone_reports_wavelet_contract() {
        let heat = measured_wavelet(&decaying_tone_ir(), 48_000.0).expect("tone IR reports");
        assert_eq!(heat.basis, "measured_room_ir");
        assert_eq!(heat.method, "complex_morlet_three_cycle_v1");
        assert_eq!(heat.reference, "full_grid_peak");
        assert_eq!(heat.cycles, 3.0);
        assert_eq!(heat.freqs_per_octave, 6.0);
        assert_eq!(heat.display_range_db, [-30.0, 0.0]);
        assert!(heat.times_ms.len() >= 2 && heat.times_ms.len() <= 100);
        assert!(heat.freqs_hz.len() >= 2 && heat.freqs_hz.len() <= 64);
        assert_eq!(heat.mags_db.len(), heat.freqs_hz.len());
        let peak = heat
            .mags_db
            .iter()
            .flat_map(|row| row.iter())
            .fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        assert!(peak <= 0.0 && peak > -1.0, "own-peak reference {peak}");
        let json = serde_json::to_value(&heat).unwrap();
        for key in [
            "basis",
            "method",
            "reference",
            "freqs_hz",
            "times_ms",
            "mags_db",
        ] {
            assert!(json.get(key).is_some(), "missing viewer key {key}");
        }
    }

    #[test]
    fn short_ir_without_post_peak_window_stays_pending() {
        // 100 ms cannot hold the 500 ms post-peak window plus support.
        let short: Vec<f32> = (0..4800).map(|i| (i as f32) / 4800.0).collect();
        assert!(measured_waterfall(&short, 48_000.0).is_none());
        assert!(measured_wavelet(&short, 48_000.0).is_none());
    }

    #[test]
    fn unfittable_ir_reports_invalid_rows_with_reasons() {
        // Silence has no decay to fit: all nine rows arrive invalid with
        // nonempty reasons and no plotted values, never fabricated T60.
        let report = measured_octave_t60(&[0.0; 64], 48_000.0).expect("rows report");
        assert_eq!(report.bands.len(), 9);
        for band in &report.bands {
            assert!(!band.valid, "{band:?}");
            assert!(band.t60_s.is_none(), "{band:?}");
            assert!(!band.reason.is_empty(), "{band:?}");
            assert!((0.0..=1.0).contains(&band.r2), "{band:?}");
        }
    }

    /// Exponentially decaying 1 kHz tone (τ = 0.1 s ⇒ T60 ≈ 0.69 s):
    /// the 1 kHz octave fit must be valid with a T20/T30 range near it.
    #[test]
    fn decaying_tone_recovers_known_t60() {
        let rate = 48_000.0;
        let ir: Vec<f32> = (0..96_000)
            .map(|i| {
                let t = i as f64 / rate;
                (2.0 * std::f64::consts::PI * 1000.0 * t).cos() as f32 * (-t / 0.1).exp() as f32
            })
            .collect();
        let report = measured_octave_t60(&ir, rate).expect("decaying tone reports");
        assert_eq!(report.basis, "measured_room_ir");
        assert_eq!(report.min_r2, 0.90);
        assert_eq!(report.bands.len(), 9);
        let centers: Vec<f64> = report.bands.iter().map(|band| band.centre_hz).collect();
        assert_eq!(
            centers,
            vec![
                63.0, 125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0, 16000.0
            ]
        );
        let kilo = report
            .bands
            .iter()
            .find(|band| band.centre_hz == 1000.0)
            .unwrap();
        assert!(kilo.valid, "{kilo:?}");
        assert!(matches!(
            kilo.fit_range.as_deref(),
            Some("T20") | Some("T30")
        ));
        let t60 = kilo.t60_s.expect("valid fit plots");
        assert!((t60 - 0.69).abs() < 0.12, "T60 {t60}");
        for band in &report.bands {
            assert!((0.0..=1.0).contains(&band.r2), "{band:?}");
            if band.valid {
                assert!(band.t60_s.is_some_and(|value| value > 0.0), "{band:?}");
            } else {
                assert!(!band.reason.is_empty(), "{band:?}");
            }
        }
    }
}
