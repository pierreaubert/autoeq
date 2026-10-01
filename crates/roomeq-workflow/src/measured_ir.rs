//! Measured room-IR ingestion for the R1–R5 acoustic report.
//!
//! Channels declaring `RoomConfig::measured_impulse_responses` load a
//! `time_ms,amplitude` CSV captured in the room (swept-sine deconvolution
//! or equivalent) and attach it as the chain's `pre_ir`, replacing the
//! synthesized prediction. The R-analysis then runs on measured data only:
//! anything the IR cannot support returns `None` and the viewer cell stays
//! pending — never a fabricated curve. Runs without declarations behave
//! exactly as before.

use math_audio_dsp::rir_waterfall::{WaterfallConfig, detect_resonances, waterfall_grid};
use math_audio_dsp::rir_wavelet::{WaveletConfig, wavelet_heatmap_detailed};
use math_rir::report::{
    DEFAULT_MIN_R2, ReflectionTableConfig, T60BatchConfig, T60FitRange, analyze_t60_octaves,
    early_reflection_table,
};
use roomeq_model::config::{MeasuredIrSource, validate_measured_ir_map};
use roomeq_model::{
    ChannelEarlyReflections, ChannelOctaveT60, ChannelReflectionEvent, ChannelResonanceDecay,
    ChannelResonanceDecays, ChannelT60Band, ChannelWaterfall, ChannelWavelet, DspGraph, IrWaveform,
};
use std::collections::BTreeMap;
use std::path::Path;

/// Minimum samples for a measured IR to ingest.
const MIN_SAMPLES: usize = 64;

/// Maximum relative deviation of the time grid step for the grid to count
/// as uniform (sample rate is derived from the grid).
const MAX_GRID_DEVIATION: f64 = 0.01;

/// R1 picking threshold in dB below the direct peak. The viewer contract
/// admits event gains in `[-15, 0]` dBFS, so the picker keeps exactly that
/// window; anything fainter stays out rather than rendering as invalid.
const REFLECTION_THRESHOLD_DB: f64 = 15.0;

/// R1 picking window after the direct peak in ms (viewer maximum).
const REFLECTION_WINDOW_MS: f64 = 15.0;

/// R1 minimum separation between picked reflections in ms.
const REFLECTION_MIN_SEPARATION_MS: f64 = 0.5;

/// Retain all 2 ms waterfall frames and dense early wavelet samples.
const GRID_MAX_FRAMES: usize = 800;
/// Retain the positive FFT bins at 48 kHz without losing the bass in pooling.
const GRID_MAX_BINS: usize = 2049;

/// R5 log-grid density in frequencies per octave (viewer contract).
const WAVELET_FREQS_PER_OCTAVE: f64 = 48.0;

/// Contract strings shared with the viewer (single source; the model
/// doc-comments pin the same values).
const BASIS_MEASURED_ROOM_IR: &str = "measured_room_ir";
const METHOD_REFLECTION_TABLE: &str = "bandlimited_early_reflection_table_v1";
const METHOD_WATERFALL: &str = "hann_stft_waterfall_v1";
const METHOD_WAVELET: &str = "complex_morlet_three_cycle_v1";
const REFERENCE_FULL_GRID_PEAK: &str = "full_grid_peak";
const DIRECT_REFERENCE_REFLECTION_TABLE: &str =
    "broadband SSIR direct TOA with bandpassed peak search +/-1 ms";

/// A validated measured IR with its derived sample rate.
#[derive(Debug, Clone)]
pub struct MeasuredIr {
    /// Channel name as declared in the configuration map.
    pub channel: String,
    /// Waveform attached as the chain's `pre_ir`.
    pub waveform: IrWaveform,
    /// Sample rate derived from the file's time grid, in Hz.
    pub sample_rate_hz: f64,
}

/// Name heuristic for subwoofer/LFE channels, mirroring the fallback in
/// `room_optimization::misc::is_subwoofer_channel`.
fn is_sub_channel(name: &str) -> bool {
    let lower = name.to_lowercase();
    lower == "lfe" || lower == "sub" || lower.starts_with("sub") || lower.ends_with("_sub")
}

/// Derive the sample rate from a millisecond time grid.
///
/// # Errors
///
/// Returns a reason for a too-short, non-finite, non-increasing, or
/// non-uniform grid.
fn grid_sample_rate_hz(time_ms: &[f64]) -> Result<f64, String> {
    if time_ms.len() < MIN_SAMPLES {
        return Err(format!(
            "measured IR needs at least {MIN_SAMPLES} samples, got {}",
            time_ms.len()
        ));
    }
    if time_ms.iter().any(|t| !t.is_finite()) {
        return Err(String::from("measured IR time grid must be finite"));
    }
    let mut steps: Vec<f64> = time_ms.windows(2).map(|w| w[1] - w[0]).collect();
    if steps.iter().any(|dt| !dt.is_finite() || *dt <= 0.0) {
        return Err(String::from(
            "measured IR time grid must increase monotonically",
        ));
    }
    steps.sort_by(|a, b| a.total_cmp(b));
    let median = steps[steps.len() / 2];
    let max_deviation = steps
        .iter()
        .map(|dt| (dt - median).abs() / median)
        .fold(0.0_f64, f64::max);
    if max_deviation > MAX_GRID_DEVIATION {
        return Err(format!(
            "measured IR time grid is not uniform (deviation {max_deviation:.4})"
        ));
    }
    Ok(1000.0 / median)
}

/// Load and validate one measured-IR CSV.
///
/// The header must read `time_ms,amplitude`; amplitudes must be finite with
/// a nonzero peak (the file's own peak is the dBFS reference, matching the
/// R1 convention).
///
/// # Errors
///
/// Returns a reason for an unreadable file, a bad header, or a failed
/// validation gate.
pub fn load_measured_ir(
    channel: &str,
    source: &MeasuredIrSource,
    config_dir: &Path,
) -> Result<MeasuredIr, String> {
    source.validate()?;
    let path = if source.path.is_absolute() {
        source.path.clone()
    } else {
        config_dir.join(&source.path)
    };
    let text = std::fs::read_to_string(&path).map_err(|error| {
        format!(
            "channel '{channel}': cannot read IR '{}': {error}",
            path.display()
        )
    })?;
    let mut lines = text.lines();
    let header = lines
        .next()
        .ok_or_else(|| format!("channel '{channel}': IR '{}' is empty", path.display()))?;
    if header.trim() != "time_ms,amplitude" {
        return Err(format!(
            "channel '{channel}': IR '{}' needs a 'time_ms,amplitude' header",
            path.display()
        ));
    }
    let mut time_ms = Vec::new();
    let mut amplitude = Vec::new();
    for (index, line) in lines.enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let mut cells = line.split(',');
        let time: f64 = cells.next().unwrap_or("").trim().parse().map_err(|_| {
            format!(
                "channel '{channel}': IR line {} has a bad time cell",
                index + 2
            )
        })?;
        let amp: f64 = cells.next().unwrap_or("").trim().parse().map_err(|_| {
            format!(
                "channel '{channel}': IR line {} has a bad amplitude cell",
                index + 2
            )
        })?;
        time_ms.push(time);
        amplitude.push(amp);
    }
    if amplitude.iter().any(|v| !v.is_finite()) {
        return Err(format!("channel '{channel}': IR amplitudes must be finite"));
    }
    let peak = amplitude.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
    if peak <= 0.0 {
        return Err(format!("channel '{channel}': IR peak must be nonzero"));
    }
    let grid_rate = grid_sample_rate_hz(&time_ms)?;
    if !(1_000.0..=192_000.0).contains(&grid_rate) {
        return Err(format!(
            "channel '{channel}': measured IR grid must be within 1–192 kHz"
        ));
    }
    if let Some(declared) = source.sample_rate_hz
        && (declared - grid_rate).abs() / grid_rate > 0.001
    {
        return Err(format!(
            "channel '{channel}': declared rate {declared} Hz disagrees with the file grid ({grid_rate:.1} Hz)"
        ));
    }
    Ok(MeasuredIr {
        channel: channel.to_string(),
        waveform: IrWaveform { time_ms, amplitude },
        sample_rate_hz: source.sample_rate_hz.unwrap_or(grid_rate),
    })
}

/// R1 band-limited early-reflection table from a measured IR.
///
/// Gains are clamped into the viewer-admitted `[-15, 0]` window the picker
/// threshold already enforces (float rounding at the boundary); events are
/// strictly post-direct, ascending, and capped by the picker's separation
/// gate far below the viewer maximum. Returns `None` (cell stays pending)
/// when any mapped value would violate the viewer contract.
fn reflection_report(samples: &[f32], sample_rate_hz: f64) -> Option<ChannelEarlyReflections> {
    if sample_rate_hz <= 16_000.0 {
        return None;
    }
    let table = early_reflection_table(
        samples,
        sample_rate_hz,
        &ReflectionTableConfig {
            threshold_db: REFLECTION_THRESHOLD_DB,
            min_separation_ms: REFLECTION_MIN_SEPARATION_MS,
            window_ms: REFLECTION_WINDOW_MS,
        },
    );
    if table.reflections.len() > 64 {
        return None;
    }
    let mut pre = Vec::with_capacity(table.reflections.len());
    let mut last_time_ms = 0.0;
    for reflection in &table.reflections {
        let gain_dbfs = reflection.gain_db.clamp(-15.0, 0.0);
        let time_ms = reflection.delay_ms;
        let distance_cm = reflection.path_difference_m * 100.0;
        if !(time_ms > last_time_ms
            && time_ms <= REFLECTION_WINDOW_MS
            && distance_cm > 0.0
            && reflection.first_dip_hz > 0.0
            && reflection.comb_ripple_db >= 0.0
            && gain_dbfs.is_finite()
            && time_ms.is_finite()
            && distance_cm.is_finite()
            && reflection.first_dip_hz.is_finite()
            && reflection.comb_ripple_db.is_finite())
        {
            return None;
        }
        last_time_ms = time_ms;
        pre.push(ChannelReflectionEvent {
            gain_dbfs,
            time_ms,
            distance_cm,
            first_dip_hz: reflection.first_dip_hz,
            ripple_db: Some(reflection.comb_ripple_db),
        });
    }
    Some(ChannelEarlyReflections {
        basis: BASIS_MEASURED_ROOM_IR.to_string(),
        method: METHOD_REFLECTION_TABLE.to_string(),
        band_hz: [1000.0, 8000.0],
        threshold_dbfs: -15.0,
        direct_reference: DIRECT_REFERENCE_REFLECTION_TABLE.to_string(),
        pre,
        post: Vec::new(),
    })
}

/// R3 nine-band octave T60 from a measured IR.
///
/// EDT-only selections stay invalid (EDT alone is not late-decay T60) with
/// the primitive's reason; the viewer renders invalid rows with reasons.
/// Returns `None` only when the nine-band shape itself cannot be honored.
fn t60_report(samples: &[f32], sample_rate_hz: f64) -> Option<ChannelOctaveT60> {
    let config = T60BatchConfig {
        min_r2: DEFAULT_MIN_R2,
        ..T60BatchConfig::default()
    };
    let rows = analyze_t60_octaves(samples, sample_rate_hz, &config);
    if rows.len() != 9 {
        return None;
    }
    let mut bands = Vec::with_capacity(rows.len());
    for row in &rows {
        if row.centre_hz * std::f64::consts::SQRT_2 >= sample_rate_hz / 2.0 {
            bands.push(ChannelT60Band {
                centre_hz: row.centre_hz,
                t60_s: None,
                fit_range: None,
                r2: 0.0,
                valid: false,
                reason: "octave band exceeds native capture Nyquist frequency".into(),
            });
            continue;
        }
        let fit = match row.fit_range {
            T60FitRange::T30 => Some("T30"),
            T60FitRange::T20 => Some("T20"),
            T60FitRange::Edt | T60FitRange::None => None,
        };
        let r2 = if row.r2.is_finite() {
            row.r2.clamp(0.0, 1.0)
        } else {
            return None;
        };
        let value = if row.t60_s.is_finite() && row.t60_s > 0.0 {
            Some(row.t60_s)
        } else {
            None
        };
        let valid = row.valid && fit.is_some() && value.is_some() && r2 >= DEFAULT_MIN_R2;
        if valid {
            bands.push(ChannelT60Band {
                centre_hz: row.centre_hz,
                t60_s: value,
                fit_range: fit.map(str::to_string),
                r2,
                valid: true,
                reason: String::new(),
            });
        } else {
            let reason = if row.reason.trim().is_empty() {
                if fit.is_none() {
                    "edt-only"
                } else {
                    "below-threshold"
                }
            } else {
                row.reason.trim()
            };
            bands.push(ChannelT60Band {
                centre_hz: row.centre_hz,
                t60_s: None,
                fit_range: None,
                r2,
                valid: false,
                reason: reason.to_string(),
            });
        }
    }
    Some(ChannelOctaveT60 {
        basis: BASIS_MEASURED_ROOM_IR.to_string(),
        min_r2: DEFAULT_MIN_R2,
        bands,
    })
}

/// Shared grid gate for the R4 heatmap: bounded, strictly increasing axes
/// with a matching finite magnitude matrix (`mags_db[frame][bin]`).
fn grid_shape_ok(
    times_ms: &[f64],
    freqs_hz: &[f64],
    mags_db: &[Vec<f32>],
    mag_floor_db: f32,
) -> bool {
    if !(2..=GRID_MAX_FRAMES).contains(&times_ms.len())
        || !(2..=GRID_MAX_BINS).contains(&freqs_hz.len())
        || mags_db.len() != times_ms.len()
    {
        return false;
    }
    if times_ms.windows(2).any(|w| w[0] >= w[1]) || freqs_hz.windows(2).any(|w| w[0] >= w[1]) {
        return false;
    }
    if times_ms.iter().any(|v| !v.is_finite()) || freqs_hz.iter().any(|v| !v.is_finite()) {
        return false;
    }
    mags_db.iter().all(|row| {
        row.len() == freqs_hz.len()
            && row
                .iter()
                .all(|v| v.is_finite() && *v >= mag_floor_db && *v <= 0.0)
    })
}

/// R4 STFT waterfall grid plus R4 resonance decays from a measured IR.
///
/// Returns `None` (both cells stay pending) when the grid shape, the
/// emitted band, or any resonance row would violate the viewer contract.
fn waterfall_report(
    samples: &[f32],
    sample_rate_hz: f64,
) -> Option<(ChannelWaterfall, ChannelResonanceDecays)> {
    let config = WaterfallConfig {
        max_frames: GRID_MAX_FRAMES,
        max_bins: GRID_MAX_BINS,
        ..WaterfallConfig::default()
    };
    let mut grid = waterfall_grid(samples, sample_rate_hz, &config);
    // DC cannot be represented on a log-frequency axis. Preserve every positive bin.
    if grid.freqs_hz.first() == Some(&0.0) {
        grid.freqs_hz.remove(0);
        for row in &mut grid.mags_db {
            row.remove(0);
        }
    }
    if grid.times_ms.is_empty()
        || grid.freqs_hz.is_empty()
        || !grid_shape_ok(&grid.times_ms, &grid.freqs_hz, &grid.mags_db, -100.0)
    {
        return None;
    }
    let first_hz = grid.freqs_hz[0];
    let last_hz = grid.freqs_hz[grid.freqs_hz.len() - 1];
    // Finiteness is established by grid_shape_ok above.
    if first_hz >= last_hz {
        return None;
    }
    let mut decays = Vec::new();
    for resonance in detect_resonances(&grid, &config) {
        if !(resonance.freq_hz.is_finite()
            && first_hz <= resonance.freq_hz
            && resonance.freq_hz <= last_hz
            && resonance.level_db.is_finite()
            && (-100.0..=0.0).contains(&resonance.level_db))
        {
            return None;
        }
        let decay_time_s = if resonance.decay_time_s.is_finite() && resonance.decay_time_s > 0.0 {
            Some(resonance.decay_time_s)
        } else {
            None
        };
        // NaN decay is unfittable (→ null); a finite non-positive decay is
        // a broken fit, not an unfittable one.
        if resonance.decay_time_s.is_finite() && decay_time_s.is_none() {
            return None;
        }
        if decays.len() >= 64 {
            return None;
        }
        decays.push(ChannelResonanceDecay {
            freq_hz: resonance.freq_hz,
            level_db: resonance.level_db,
            decay_time_s,
        });
    }
    let waterfall = ChannelWaterfall {
        basis: BASIS_MEASURED_ROOM_IR.to_string(),
        method: METHOD_WATERFALL.to_string(),
        reference: REFERENCE_FULL_GRID_PEAK.to_string(),
        valid_band_hz: [first_hz, last_hz],
        window_ms: 32.0,
        hop_ms: 2.0,
        post_ms: 500.0,
        times_ms: grid.times_ms.clone(),
        freqs_hz: grid.freqs_hz.clone(),
        mags_db: grid.mags_db.clone(),
        scope: String::from(
            "Pre-correction STFT waterfall of the measured room IR, dB relative \
             to the grid peak; levels do not compare between channels.",
        ),
    };
    let resonance_decays = ChannelResonanceDecays {
        basis: BASIS_MEASURED_ROOM_IR.to_string(),
        method: METHOD_WATERFALL.to_string(),
        reference: REFERENCE_FULL_GRID_PEAK.to_string(),
        slice_ms: 60.0,
        decays,
    };
    Some((waterfall, resonance_decays))
}

/// R5 three-cycle wavelet heatmap from a measured IR.
///
/// Returns `None` (cell stays pending) when the heatmap shape or any value
/// would violate the viewer contract.
fn wavelet_report(samples: &[f32], sample_rate_hz: f64) -> Option<ChannelWavelet> {
    let config = WaveletConfig {
        freqs_per_octave: WAVELET_FREQS_PER_OCTAVE,
        max_freqs: GRID_MAX_BINS,
        max_frames: GRID_MAX_FRAMES,
    };
    let heatmap = wavelet_heatmap_detailed(samples, sample_rate_hz, &config);
    if heatmap.freqs_hz.is_empty()
        || heatmap.times_ms.is_empty()
        || heatmap.mags_db.len() != heatmap.freqs_hz.len()
        || !(2..=GRID_MAX_BINS).contains(&heatmap.freqs_hz.len())
        || !(2..=GRID_MAX_FRAMES).contains(&heatmap.times_ms.len())
    {
        return None;
    }
    if heatmap.times_ms.windows(2).any(|w| w[0] >= w[1])
        || heatmap.freqs_hz.windows(2).any(|w| w[0] >= w[1])
        || heatmap.times_ms.iter().any(|v| !v.is_finite())
        || heatmap.freqs_hz.iter().any(|v| !v.is_finite())
    {
        return None;
    }
    for row in &heatmap.mags_db {
        if row.len() != heatmap.times_ms.len()
            || row.iter().any(|v| !v.is_finite() || *v < -30.0 || *v > 0.0)
        {
            return None;
        }
    }
    let first_hz = heatmap.freqs_hz[0];
    let last_hz = heatmap.freqs_hz[heatmap.freqs_hz.len() - 1];
    // Finiteness is established by the ascending-order gate above.
    if first_hz >= last_hz {
        return None;
    }
    Some(ChannelWavelet {
        basis: BASIS_MEASURED_ROOM_IR.to_string(),
        method: METHOD_WAVELET.to_string(),
        reference: REFERENCE_FULL_GRID_PEAK.to_string(),
        valid_band_hz: [first_hz, last_hz],
        cycles: 3.0,
        freqs_per_octave: WAVELET_FREQS_PER_OCTAVE,
        hop_ms: 0.1,
        display_range_db: [-30.0, 0.0],
        freqs_hz: heatmap.freqs_hz.clone(),
        times_ms: heatmap.times_ms.clone(),
        mags_db: heatmap.mags_db.clone(),
    })
}

/// Attach declared measured IRs to an optimized output.
///
/// For every declared channel present in the output, the measured waveform
/// replaces the synthesized `pre_ir` and the R1–R5 acoustic analyses run on
/// it. Channels without a declaration are untouched. A declaration for an
/// unknown channel, an unreadable file, or a failed validation gate fails
/// closed with a reason.
///
/// Returns per-channel warning lines (an analysis that declines on a valid
/// file, never a failure).
///
/// # Errors
///
/// Returns a reason for an invalid declaration map or a failed attachment.
pub fn attach_measured_acoustics(
    output: &mut DspGraph,
    declared: &BTreeMap<String, MeasuredIrSource>,
    config_dir: &Path,
) -> Result<Vec<String>, String> {
    validate_measured_ir_map(declared)?;
    let mut warnings = Vec::new();
    for (channel, source) in declared {
        let output_channel = source.output_channel.as_deref().unwrap_or(channel);
        let chain = output.channels.get_mut(output_channel).ok_or_else(|| {
            format!("channel '{channel}' declares a measured IR but has no output chain")
        })?;
        let measured = load_measured_ir(channel, source, config_dir)?;
        let samples: Vec<f32> = measured
            .waveform
            .amplitude
            .iter()
            .map(|v| *v as f32)
            .collect();
        if let Some(driver_name) = &source.driver {
            let driver = chain.drivers.as_mut().and_then(|drivers| {
                drivers.iter_mut().find(|driver| driver.name == *driver_name)
            }).ok_or_else(|| format!(
                "capture '{channel}' targets missing driver '{driver_name}' in '{output_channel}'"
            ))?;
            let waterfall = waterfall_report(&samples, measured.sample_rate_hz);
            driver.measured_acoustics = Some(roomeq_model::MeasuredRoomAcoustics {
                sample_rate_hz: measured.sample_rate_hz,
                timing_reference_id: source.timing_reference_id.clone(),
                early_late_curves: crate::channel_acoustics::measured_early_late_curves(
                    &samples,
                    measured.sample_rate_hz,
                    is_sub_channel(driver_name),
                ),
                early_reflections: reflection_report(&samples, measured.sample_rate_hz),
                t60_octaves: t60_report(&samples, measured.sample_rate_hz),
                waterfall: waterfall.as_ref().map(|(grid, _)| grid.clone()),
                resonance_decays: waterfall.map(|(_, decays)| decays),
                wavelet: wavelet_report(&samples, measured.sample_rate_hz),
                pre_ir: measured.waveform,
            });
            continue;
        }
        chain.pre_ir = Some(measured.waveform);
        match crate::channel_acoustics::measured_early_late_curves(
            &samples,
            measured.sample_rate_hz,
            is_sub_channel(channel),
        ) {
            Some(curves) => chain.early_late_curves = Some(curves),
            None => warnings.push(format!(
                "channel '{channel}': early/late analysis declined on the measured IR \
                 (too short or too narrow for the 1–8 kHz contract); cell stays pending"
            )),
        }
        match reflection_report(&samples, measured.sample_rate_hz) {
            Some(report) => chain.early_reflections = Some(report),
            None => warnings.push(format!(
                "channel '{channel}': reflection-table analysis declined on the measured IR; \
                 cell stays pending"
            )),
        }
        match t60_report(&samples, measured.sample_rate_hz) {
            Some(report) => chain.t60_octaves = Some(report),
            None => warnings.push(format!(
                "channel '{channel}': octave-T60 analysis declined on the measured IR; \
                 cell stays pending"
            )),
        }
        match waterfall_report(&samples, measured.sample_rate_hz) {
            Some((waterfall, decays)) => {
                chain.waterfall = Some(waterfall);
                chain.resonance_decays = Some(decays);
            }
            None => warnings.push(format!(
                "channel '{channel}': waterfall analysis declined on the measured IR; \
                 cells stay pending"
            )),
        }
        match wavelet_report(&samples, measured.sample_rate_hz) {
            Some(report) => chain.wavelet = Some(report),
            None => warnings.push(format!(
                "channel '{channel}': wavelet analysis declined on the measured IR; \
                 cell stays pending"
            )),
        }
    }
    Ok(warnings)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn native_sub_driver_capture_is_not_an_aggregate_measurement() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = write_ir(dir.path(), "sub.csv", 3_000.0, 1.0);
        let mut source = source_for(path);
        source.sample_rate_hz = Some(3_000.0);
        source.output_channel = Some("L".into());
        source.driver = Some("left_sub".into());
        let mut output = DspGraph::new("1");
        output.add_channel("L", Vec::new());
        output.channels.get_mut("L").expect("channel").drivers =
            Some(vec![roomeq_model::DriverDspChain {
                measured_acoustics: None,
                name: "left_sub".into(),
                index: 0,
                plugins: Vec::new(),
                initial_curve: None,
                measured_band_hz: None,
            }]);
        let declared = BTreeMap::from([("sub_capture".into(), source)]);
        attach_measured_acoustics(&mut output, &declared, dir.path()).expect("attach");
        let chain = &output.channels["L"];
        assert!(
            chain.pre_ir.is_none(),
            "driver IR must not replace aggregate IR"
        );
        let capture = chain.drivers.as_ref().expect("drivers")[0]
            .measured_acoustics
            .as_ref()
            .expect("capture");
        assert!((capture.sample_rate_hz - 3_000.0).abs() < 0.1);
        assert_eq!(capture.pre_ir.amplitude.len(), 3_000);
        assert!(capture.timing_reference_id.is_none());
        assert!(capture.early_reflections.is_none());
        assert!(
            capture
                .t60_octaves
                .as_ref()
                .expect("T60")
                .bands
                .iter()
                .filter(|band| band.centre_hz >= 2_000.0)
                .all(|band| !band.valid)
        );
        if let Some(grid) = &capture.waterfall {
            assert!(grid.valid_band_hz[1] <= 1_500.0);
        }
        if let Some(grid) = &capture.wavelet {
            assert!(grid.valid_band_hz[1] <= 1_500.0);
        }
        let mut bad = declared.clone();
        bad.get_mut("sub_capture").expect("source").driver = Some("missing".into());
        assert!(attach_measured_acoustics(&mut output, &bad, dir.path()).is_err());
        let bundle_path = dir.path().join("out.json");
        crate::output_bundle::save_output_bundle(&mut output, &bundle_path).expect("save");
        let restored = crate::output_bundle::load_output_bundle(&bundle_path).expect("reload");
        let capture = restored.channels["L"].drivers.as_ref().expect("drivers")[0]
            .measured_acoustics
            .as_ref()
            .expect("restored native capture");
        assert_eq!(capture.sample_rate_hz, 3_000.0);
        assert_eq!(capture.pre_ir.amplitude.len(), 3_000);
    }

    fn write_ir(dir: &Path, name: &str, rate_hz: f64, seconds: f64) -> std::path::PathBuf {
        // Exponentially decaying sweep-like IR with a real reflection tap.
        let n = (seconds * rate_hz) as usize;
        let path = dir.join(name);
        let mut file = std::fs::File::create(&path).expect("write fixture IR");
        file.write_all(b"time_ms,amplitude\n").expect("header");
        for i in 0..n {
            let t = i as f64 / rate_hz;
            let mut amp = (-t * 30.0).exp() * (2.0 * std::f64::consts::PI * 440.0 * t).sin();
            let tap = (0.010 * rate_hz) as usize;
            if i == tap {
                amp += 0.5;
            }
            if i == 0 {
                amp = 1.0;
            }
            writeln!(file, "{:.6},{:.9}", t * 1000.0, amp).expect("row");
        }
        path
    }

    fn source_for(path: std::path::PathBuf) -> MeasuredIrSource {
        MeasuredIrSource {
            output_channel: None,
            driver: None,
            path,
            sample_rate_hz: None,
            timing_reference_id: None,
        }
    }

    #[test]
    fn uniform_grid_derives_rate() {
        let time: Vec<f64> = (0..128).map(|i| i as f64 * 1000.0 / 48_000.0).collect();
        assert!((grid_sample_rate_hz(&time).expect("rate") - 48_000.0).abs() < 1.0);
    }

    #[test]
    fn ragged_grid_rejected() {
        let mut time: Vec<f64> = (0..128).map(|i| i as f64 * 1000.0 / 48_000.0).collect();
        time[64] += 5.0;
        assert!(grid_sample_rate_hz(&time).is_err());
    }

    #[test]
    fn short_or_silent_files_rejected() {
        let dir = tempfile::tempdir().expect("tempdir");
        let short = dir.path().join("short.csv");
        std::fs::write(&short, "time_ms,amplitude\n0.0,1.0\n").expect("write");
        assert!(load_measured_ir("L", &source_for(short), dir.path()).is_err());
        let silent = dir.path().join("silent.csv");
        let mut text = String::from("time_ms,amplitude\n");
        for i in 0..128 {
            text.push_str(&format!("{:.6},0.0\n", i as f64 * 1000.0 / 48_000.0));
        }
        std::fs::write(&silent, text).expect("write");
        assert!(load_measured_ir("L", &source_for(silent), dir.path()).is_err());
    }

    #[test]
    fn declared_rate_mismatch_fails() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = write_ir(dir.path(), "ir.csv", 48_000.0, 0.5);
        let mut source = source_for(path);
        source.sample_rate_hz = Some(44_100.0);
        assert!(load_measured_ir("L", &source, dir.path()).is_err());
    }

    #[test]
    fn attach_replaces_pre_ir_and_reports_early_late() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = write_ir(dir.path(), "L__ir.csv", 48_000.0, 1.0);
        let declared = BTreeMap::from([("L".to_string(), source_for(path))]);
        let mut output = DspGraph::new("1");
        output.add_channel("L", Vec::new());
        let warnings =
            attach_measured_acoustics(&mut output, &declared, dir.path()).expect("attach");
        assert!(warnings.is_empty(), "unexpected warnings: {warnings:?}");
        let chain = &output.channels["L"];
        assert!(chain.pre_ir.is_some());
        let curves = chain
            .early_late_curves
            .as_ref()
            .expect("early/late attaches on a 1 s decaying IR");
        assert_eq!(curves.method, "incoherent_band_energy");
        assert_eq!(curves.basis, "measured_room_ir");
        // Late energy sits below early energy on a decaying room IR.
        for (early, late) in curves.early.spl.iter().zip(curves.late.spl.iter()) {
            assert!(early >= late, "early {early} < late {late}");
        }
    }

    #[test]
    fn attach_fails_closed_on_unknown_channel() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = write_ir(dir.path(), "X__ir.csv", 48_000.0, 0.5);
        let declared = BTreeMap::from([("X".to_string(), source_for(path))]);
        let mut output = DspGraph::new("1");
        output.add_channel("L", Vec::new());
        assert!(attach_measured_acoustics(&mut output, &declared, dir.path()).is_err());
    }

    /// Decaying 1 s room IR with a 10 ms reflection bounce, as f32 samples.
    ///
    /// The bounce is a 2 ms 2 kHz burst (a realistic band-limited
    /// reflection, not a single-sample delta, which the 1 ms envelope
    /// would dilute below the picking threshold).
    fn room_samples() -> (Vec<f32>, f64) {
        let rate = 48_000.0;
        let n = rate as usize;
        let tap_start = (0.010 * rate) as usize;
        let tap_len = (0.002 * rate) as usize;
        let mut samples = Vec::with_capacity(n);
        for i in 0..n {
            let t = i as f64 / rate;
            let mut amp = (-t * 8.0).exp() * (2.0 * std::f64::consts::PI * 440.0 * t).sin();
            if i >= tap_start && i < tap_start + tap_len {
                let local = (i - tap_start) as f64 / rate;
                amp += 0.6
                    * (std::f64::consts::PI * local / 0.002).sin()
                    * (2.0 * std::f64::consts::PI * 2000.0 * local).sin();
            }
            if i == 0 {
                amp = 1.0;
            }
            samples.push(amp as f32);
        }
        (samples, rate)
    }

    #[test]
    fn reflection_table_picks_the_tap_inside_viewer_contract() {
        let (samples, rate) = room_samples();
        let report = reflection_report(&samples, rate).expect("reflection table");
        assert_eq!(report.basis, "measured_room_ir");
        assert_eq!(report.method, "bandlimited_early_reflection_table_v1");
        assert_eq!(report.band_hz, [1000.0, 8000.0]);
        assert_eq!(report.threshold_dbfs, -15.0);
        assert!(report.post.is_empty());
        assert!(!report.pre.is_empty(), "the 10 ms tap must be picked");
        let tap = report
            .pre
            .iter()
            .find(|event| (event.time_ms - 10.0).abs() < 1.0)
            .expect("tap near 10 ms");
        assert!((-15.0..=0.0).contains(&tap.gain_dbfs));
        assert!((tap.distance_cm - 34.3 * tap.time_ms).abs() <= 0.02 * 34.3 * tap.time_ms + 0.1);
        assert!((tap.first_dip_hz - 500.0 / tap.time_ms).abs() <= 0.02 * 500.0 / tap.time_ms + 0.5);
        assert!(tap.ripple_db.is_some());
    }

    #[test]
    fn t60_report_keeps_nine_bands_with_reasons() {
        let (samples, rate) = room_samples();
        let report = t60_report(&samples, rate).expect("t60 report");
        assert_eq!(report.basis, "measured_room_ir");
        assert_eq!(report.min_r2, 0.90);
        assert_eq!(report.bands.len(), 9);
        let mut valid_count = 0;
        for band in &report.bands {
            if band.valid {
                valid_count += 1;
                assert!(band.t60_s.is_some());
                assert!(matches!(
                    band.fit_range.as_deref(),
                    Some("T20") | Some("T30")
                ));
                assert!(band.r2 >= 0.90);
            } else {
                assert!(band.t60_s.is_none());
                assert!(!band.reason.trim().is_empty());
            }
        }
        assert!(valid_count > 0, "a 1 s decaying IR must fit some bands");
    }

    #[test]
    fn waterfall_and_resonances_hold_viewer_shape() {
        let (samples, rate) = room_samples();
        let (waterfall, decays) = waterfall_report(&samples, rate).expect("waterfall report");
        assert_eq!(waterfall.method, "hann_stft_waterfall_v1");
        assert_eq!(
            (waterfall.window_ms, waterfall.hop_ms, waterfall.post_ms),
            (32.0, 2.0, 500.0)
        );
        assert!((200..=GRID_MAX_FRAMES).contains(&waterfall.times_ms.len()));
        assert!((1000..=GRID_MAX_BINS).contains(&waterfall.freqs_hz.len()));
        assert!(waterfall.freqs_hz[0] > 0.0 && waterfall.freqs_hz[0] < 30.0);
        assert_eq!(waterfall.mags_db.len(), waterfall.times_ms.len());
        for row in &waterfall.mags_db {
            assert_eq!(row.len(), waterfall.freqs_hz.len());
        }
        assert_eq!(decays.slice_ms, 60.0);
        for decay in &decays.decays {
            assert!(decay.decay_time_s.is_none_or(|v| v > 0.0));
        }
    }

    #[test]
    fn wavelet_holds_viewer_shape() {
        let (samples, rate) = room_samples();
        let report = wavelet_report(&samples, rate).expect("wavelet report");
        assert_eq!(report.method, "complex_morlet_three_cycle_v1");
        assert_eq!(
            (report.cycles, report.freqs_per_octave, report.hop_ms),
            (3.0, 48.0, 0.1)
        );
        assert_eq!(report.display_range_db, [-30.0, 0.0]);
        assert!((390..=GRID_MAX_BINS).contains(&report.freqs_hz.len()));
        assert!((500..=GRID_MAX_FRAMES).contains(&report.times_ms.len()));
        assert!(
            report
                .times_ms
                .iter()
                .filter(|t| **t >= 0.0 && **t <= 15.0)
                .count()
                >= 150
        );
        assert_eq!(report.mags_db.len(), report.freqs_hz.len());
    }

    #[test]
    fn attach_reports_all_five_analyses() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = write_ir(dir.path(), "L__ir.csv", 48_000.0, 1.0);
        let declared = BTreeMap::from([("L".to_string(), source_for(path))]);
        let mut output = DspGraph::new("1");
        output.add_channel("L", Vec::new());
        let warnings =
            attach_measured_acoustics(&mut output, &declared, dir.path()).expect("attach");
        assert!(warnings.is_empty(), "unexpected warnings: {warnings:?}");
        let chain = &output.channels["L"];
        assert!(chain.early_late_curves.is_some());
        assert!(chain.early_reflections.is_some());
        assert!(chain.t60_octaves.is_some());
        assert!(chain.waterfall.is_some());
        assert!(chain.resonance_decays.is_some());
        assert!(chain.wavelet.is_some());
    }
}
