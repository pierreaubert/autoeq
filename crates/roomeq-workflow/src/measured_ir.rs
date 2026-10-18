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
    ChannelDspChain, ChannelEarlyReflections, ChannelOctaveT60, ChannelReflectionEvent,
    ChannelResonanceDecay, ChannelResonanceDecays, ChannelT60Band, ChannelWaterfall,
    ChannelWavelet, DspGraph, IrWaveform, MeasurementSource, RoomConfig, SpeakerConfig,
};
use std::collections::BTreeMap;
use std::path::Path;

/// Approximately one period of the lowest STI modulation (0.63 Hz).
/// Shorter captures require an explicit check that the decay was not truncated.
const STI_SHORT_CAPTURE_S: f64 = 1.6;

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
    // Dense ringing (e.g. flutter echo) can yield more envelope crests than
    // the table holds. Keep the loudest candidates instead of declining
    // the whole channel; the cap itself is unchanged.
    let mut picked = table.reflections;
    if picked.len() > 64 {
        picked.sort_by(|a, b| b.gain_db.total_cmp(&a.gain_db));
        picked.truncate(64);
        picked.sort_by(|a, b| a.delay_ms.total_cmp(&b.delay_ms));
    }
    let mut pre = Vec::with_capacity(picked.len());
    let mut last_time_ms = 0.0;
    for reflection in &picked {
        let gain_dbfs = reflection.gain_db.clamp(-15.0, 0.0);
        let time_ms = reflection.delay_ms;
        let distance_cm = reflection.path_difference_m * 100.0;
        // Drop individual out-of-contract events (a refined peak can land
        // just past the window) instead of declining the whole table.
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
            continue;
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

fn speech_transmission_report(
    samples: &[f32],
    sample_rate_hz: f64,
) -> Result<roomeq_model::ChannelSpeechTransmission, String> {
    use math_rir::sti::{STI_MODULATION_FREQUENCIES_HZ, STI_OCTAVE_CENTERS_HZ, analyze_sti};
    let result = analyze_sti(samples, sample_rate_hz).map_err(|error| error.to_string())?;
    let warnings = if result.duration_s < STI_SHORT_CAPTURE_S {
        vec!["Capture is shorter than 1.6 s; verify the full room decay was captured. Truncation can overestimate STI.".into()]
    } else {
        Vec::new()
    };
    Ok(roomeq_model::ChannelSpeechTransmission {
        method: "iec_60268_16_2020_indirect_ir_only_v1".into(),
        basis: BASIS_MEASURED_ROOM_IR.into(),
        sample_rate_hz,
        duration_s: result.duration_s,
        sti: result.sti,
        octave_centers_hz: STI_OCTAVE_CENTERS_HZ,
        modulation_frequencies_hz: STI_MODULATION_FREQUENCIES_HZ,
        modulation_transfer: result.modulation_transfer,
        mti: result.mti,
        warnings,
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
        let speech_transmission =
            match speech_transmission_report(&samples, measured.sample_rate_hz) {
                Ok(report) => Some(report),
                Err(reason) => {
                    warnings.push(format!(
                        "capture '{channel}': speech transmission unavailable: {reason}"
                    ));
                    None
                }
            };
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
                speech_transmission,
                waterfall: waterfall.as_ref().map(|(grid, _)| grid.clone()),
                resonance_decays: waterfall.map(|(_, decays)| decays),
                wavelet: wavelet_report(&samples, measured.sample_rate_hz),
                pre_ir: measured.waveform,
            });
            continue;
        }
        chain.pre_ir = Some(measured.waveform);
        chain.speech_transmission = speech_transmission;
        apply_chain_acoustics(chain, &samples, measured.sample_rate_hz, channel, &mut warnings);
    }
    Ok(warnings)
}

/// Run the R1–R5 chain-level acoustic analyses on measured IR samples.
///
/// Every analysis is quality-gated: a decline leaves the field unset (the
/// report cell stays pending) and records a warning instead of fabricating
/// a value.
fn apply_chain_acoustics(
    chain: &mut ChannelDspChain,
    samples: &[f32],
    sample_rate_hz: f64,
    channel: &str,
    warnings: &mut Vec<String>,
) {
    match crate::channel_acoustics::measured_early_late_curves(
        samples,
        sample_rate_hz,
        is_sub_channel(channel),
    ) {
        Some(curves) => chain.early_late_curves = Some(curves),
        None => warnings.push(format!(
            "channel '{channel}': early/late analysis declined on the measured IR \
             (too short or too narrow for the 1–8 kHz contract); cell stays pending"
        )),
    }
    match reflection_report(samples, sample_rate_hz) {
        Some(report) => chain.early_reflections = Some(report),
        None => warnings.push(format!(
            "channel '{channel}': reflection-table analysis declined on the measured IR; \
             cell stays pending"
        )),
    }
    // The predicted after-EQ candidates come from running the same table
    // on the chain's predicted post-correction IR, when one is saved.
    if let Some(report) = chain.early_reflections.as_mut()
        && let Some(post_ir) = chain.post_ir.as_ref()
        && let Some((post_samples, post_rate_hz)) = waveform_samples(post_ir)
        && let Some(post_report) = reflection_report(&post_samples, post_rate_hz)
    {
        report.post = post_report.pre;
    }
    match t60_report(samples, sample_rate_hz) {
        Some(report) => chain.t60_octaves = Some(report),
        None => warnings.push(format!(
            "channel '{channel}': octave-T60 analysis declined on the measured IR; \
             cell stays pending"
        )),
    }
    match waterfall_report(samples, sample_rate_hz) {
        Some((waterfall, decays)) => {
            chain.waterfall = Some(waterfall);
            chain.resonance_decays = Some(decays);
        }
        None => warnings.push(format!(
            "channel '{channel}': waterfall analysis declined on the measured IR; \
             cells stay pending"
        )),
    }
    match wavelet_report(samples, sample_rate_hz) {
        Some(report) => chain.wavelet = Some(report),
        None => warnings.push(format!(
            "channel '{channel}': wavelet analysis declined on the measured IR; \
             cell stays pending"
        )),
    }
}

/// Extract native-rate samples from a saved IR waveform, when regular.
fn waveform_samples(waveform: &IrWaveform) -> Option<(Vec<f32>, f64)> {
    if waveform.time_ms.len() != waveform.amplitude.len() || waveform.time_ms.len() < MIN_SAMPLES {
        return None;
    }
    let step_ms = waveform.time_ms[1] - waveform.time_ms[0];
    if !step_ms.is_finite() || step_ms <= 0.0 {
        return None;
    }
    let samples: Vec<f32> = waveform
        .amplitude
        .iter()
        .map(|&amplitude| amplitude as f32)
        .collect();
    if samples.iter().any(|sample| !sample.is_finite()) {
        return None;
    }
    Some((samples, 1000.0 / step_ms))
}

/// Maximum capture-derived IR length fed to the acoustic analyses.
///
/// Five seconds covers domestic-room decays and the STI duration check
/// while bounding analysis cost; later deconvolution noise is excluded.
const MAX_CAPTURE_IR_SECONDS: f64 = 5.0;

/// Deconvolve a processed take against the stimulus, preserving delay.
///
/// Unlike a lag-stripped analysis IR, the direct sound keeps its measured
/// arrival offset, so the reflection picker's onset search sees the same
/// pre-direct silence as in a declared measured IR.
fn capture_impulse_response(
    recording: &[f32],
    stimulus: &[f32],
    sample_rate_hz: u32,
) -> Option<Vec<f32>> {
    use rustfft::{FftPlanner, num_complex::Complex};
    let spectrum =
        math_audio_dsp::analysis::deconvolve_sweep(recording, stimulus, sample_rate_hz).ok()?;
    let fft_size = (spectrum.len().checked_sub(1)?).checked_mul(2)?;
    if fft_size == 0 {
        return None;
    }
    let mut full_spectrum = vec![Complex::new(0.0_f32, 0.0_f32); fft_size];
    for (index, value) in spectrum.iter().enumerate() {
        full_spectrum[index] = *value;
    }
    for index in 1..fft_size / 2 {
        full_spectrum[fft_size - index] = spectrum[index].conj();
    }
    let mut planner = FftPlanner::<f32>::new();
    planner.plan_fft_inverse(fft_size).process(&mut full_spectrum);
    let scale = 1.0 / fft_size as f32;
    Some(
        full_spectrum
            .iter()
            .map(|value| value.re * scale)
            .collect(),
    )
}

/// Attach measured acoustics derived from a capture handoff's sweep takes.
///
/// Sessions captured as spatial magnitude sweeps already contain a full
/// measured impulse response per speaker and microphone: the selected
/// primary (first) microphone take is deconvolved against the handoff
/// stimulus at the handoff sample rate, and the R1–R5 analyses run on the
/// result, exactly as for a declared measured IR. The synthesized `pre_ir`
/// is left untouched. Any gap — no handoff, no stimulus, changed audio,
/// failed deconvolution, or a declined analysis — leaves the affected
/// fields pending with a warning; this path never fails the output.
///
/// Returns per-channel warning lines.
///
/// # Errors
///
/// This function does not intentionally return an error; failures are
/// reported as warnings so a capture gap cannot refuse a valid output.
pub fn attach_capture_measured_acoustics(
    output: &mut DspGraph,
    config: &RoomConfig,
    config_dir: &Path,
) -> Vec<String> {
    let mut warnings = Vec::new();
    let Some(handoff_name) = config
        .recording_config
        .as_ref()
        .and_then(|recording| recording.capture_handoff_file.as_deref())
    else {
        return warnings;
    };
    if !crate::capture_handoff::sweep_capture_declared(config) {
        return warnings;
    }
    let Some(handoff) = std::fs::read(config_dir.join(handoff_name))
        .ok()
        .and_then(|bytes| {
            serde_json::from_slice::<autoeq_core::capture_handoff::CaptureHandoff>(&bytes).ok()
        })
        .filter(|handoff| handoff.validate().is_ok())
    else {
        warnings.push("capture measured acoustics unavailable: handoff unreadable".into());
        return warnings;
    };
    let Some(stimulus) = crate::capture_handoff::stimulus_artifact(&handoff)
        .and_then(|artifact| {
            crate::capture_handoff::verified_audio_samples(config_dir, &handoff, &artifact.file)
        })
    else {
        warnings.push("capture measured acoustics unavailable: no usable stimulus".into());
        return warnings;
    };
    let Some(_sweep_range_hz) = config
        .recording_config
        .as_ref()
        .and_then(|recording| {
            Some((recording.sweep_start_freq?, recording.sweep_end_freq?))
        })
        .filter(|(start, end)| start.is_finite() && end.is_finite() && *start > 0.0 && *end > *start)
    else {
        warnings.push("capture measured acoustics unavailable: no sweep range".into());
        return warnings;
    };
    let sample_rate_hz = f64::from(handoff.sample_rate_hz);
    let max_samples = (MAX_CAPTURE_IR_SECONDS * sample_rate_hz) as usize;
    let primary_microphone = handoff.microphone_ids.first().cloned();
    let channel_names: Vec<String> = output.channels.keys().cloned().collect();
    for channel in &channel_names {
        let Some(source_id) = capture_source_id(config, channel) else {
            continue;
        };
        let Some(SpeakerConfig::Single(MeasurementSource::Multiple(_))) =
            config.speakers.get(&source_id)
        else {
            continue;
        };
        let Some(microphone_id) = primary_microphone.as_deref() else {
            continue;
        };
        let Some(take) = handoff.takes.iter().find(|take| {
            take.source_id == source_id
                && take.provenance.microphone_id == microphone_id
                && handoff
                    .selected_take_ids
                    .as_ref()
                    .map_or(take.repeat_index == 0, |selected| {
                        selected.iter().any(|id| id == &take.take_id)
                    })
        }) else {
            warnings.push(format!(
                "channel '{channel}': capture measured acoustics unavailable: primary take missing"
            ));
            continue;
        };
        let Some(recording) = crate::capture_handoff::verified_audio_samples(
            config_dir,
            &handoff,
            &take.processed_audio_file,
        ) else {
            warnings.push(format!(
                "channel '{channel}': capture measured acoustics unavailable: processed audio unusable"
            ));
            continue;
        };
        let Some(samples) = capture_impulse_response(&recording, &stimulus, handoff.sample_rate_hz)
        else {
            warnings.push(format!(
                "channel '{channel}': capture measured acoustics unavailable: deconvolution failed"
            ));
            continue;
        };
        // Window to 5 ms before the direct peak: deconvolution pre-ringing
        // at the buffer start must not be mistaken for the direct sound.
        // The deconvolved IR is circular, so a take whose direct arrival
        // precedes the stimulus reference wraps to the buffer end; rotate
        // the window around the peak instead of slicing.
        let peak = samples
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.abs().total_cmp(&b.1.abs()))
            .map_or(0, |(index, _)| index);
        let pre_roll = (0.005 * sample_rate_hz) as usize;
        let window_start = if samples.is_empty() || peak >= pre_roll {
            peak.saturating_sub(pre_roll)
        } else {
            samples.len() - (pre_roll - peak)
        };
        let count = max_samples.min(samples.len());
        let samples: Vec<f32> = (0..count)
            .map(|offset| samples[(window_start + offset) % samples.len()])
            .collect();
        if samples.len() < MIN_SAMPLES {
            warnings.push(format!(
                "channel '{channel}': capture measured acoustics unavailable: deconvolved IR too short"
            ));
            continue;
        }
        let Some(chain) = output.channels.get_mut(channel) else {
            continue;
        };
        match speech_transmission_report(&samples, sample_rate_hz) {
            Ok(report) => chain.speech_transmission = Some(report),
            Err(reason) => warnings.push(format!(
                "capture '{channel}': speech transmission unavailable: {reason}"
            )),
        }
        apply_chain_acoustics(chain, &samples, sample_rate_hz, channel, &mut warnings);
    }
    warnings
}

/// Resolve an output channel name to its capture source identifier.
fn capture_source_id(config: &RoomConfig, channel: &str) -> Option<String> {
    if config.speakers.contains_key(channel) {
        return Some(channel.to_owned());
    }
    config
        .system
        .as_ref()?
        .speakers
        .iter()
        .find(|(role, _)| role == &channel)
        .map(|(_, measurement_key)| measurement_key.clone())
        .filter(|key| config.speakers.contains_key(key))
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
        assert!(capture.speech_transmission.is_none());
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

    fn capture_wav_bytes(samples: &[f32]) -> Vec<u8> {
        let mut audio = std::io::Cursor::new(Vec::new());
        {
            let spec = hound::WavSpec {
                channels: 1,
                sample_rate: 48000,
                bits_per_sample: 32,
                sample_format: hound::SampleFormat::Float,
            };
            let mut writer = hound::WavWriter::new(&mut audio, spec).unwrap();
            for sample in samples {
                writer.write_sample(*sample).unwrap();
            }
            writer.finalize().unwrap();
        }
        audio.into_inner()
    }

    #[test]
    fn capture_handoff_takes_populate_measured_acoustics() {
        use autoeq_core::capture_handoff::{
            CaptureArtifactIdentity, CaptureArtifactRole, CaptureCompletion, CaptureHandoff,
            CaptureTakeIdentity,
        };
        use autoeq_core::capture_provenance::{
            CaptureCorrection, CaptureGeometry, CaptureProvenance, CaptureTakeProvenance,
        };
        use autoeq_core::{MeasurementProvenance, ProvenanceCaptureKind};
        use roomeq_model::{MeasurementMultiple, MeasurementRef, RecordingConfiguration};

        let dir = tempfile::tempdir().expect("tempdir");
        let digest = |bytes: &[u8]| autoeq_artifacts::sha256_hex(bytes);
        // ESS stimulus (100 ms silence head, as in real captures) and a
        // recording through a two-tap room (direct plus a 10 ms reflection).
        let ratio = (20000.0_f64 / 20.0).ln();
        let stimulus: Vec<f32> = (0..48000)
            .map(|i| {
                if i < 4800 {
                    return 0.0;
                }
                let t = (i - 4800) as f64 / 48000.0;
                let phase = 2.0 * std::f64::consts::PI * 20.0 / ratio * ((ratio * t).exp() - 1.0);
                (0.1 * phase.sin()) as f32
            })
            .collect();
        // The sweep reaches the microphone 960 samples (20 ms) after the
        // stimulus starts; the room adds a reflection 480 samples (10 ms)
        // later over an exponentially decaying noise tail (T60 ~ 0.3 s).
        let room: Vec<f32> = (0..48000)
            .map(|i| {
                if i < 960 {
                    return 0.0;
                }
                let t = (i - 960) as f64 / 48000.0;
                let noise = ((i as f64 * 12.9898).sin() * 43758.5453).fract() - 0.5;
                let mut amp = (0.3 * noise * (-t * 23.0).exp()) as f32;
                if i == 960 {
                    amp += 1.0;
                }
                if i == 1440 {
                    amp += 0.8;
                }
                amp
            })
            .collect();
        let recording: Vec<f32> = {
            use rustfft::{FftPlanner, num_complex::Complex};
            let size = 131072_usize;
            let mut planner = FftPlanner::<f32>::new();
            let fft = planner.plan_fft_forward(size);
            let mut a: Vec<Complex<f32>> = stimulus
                .iter()
                .map(|&s| Complex::new(s, 0.0))
                .collect();
            a.resize(size, Complex::default());
            let mut b: Vec<Complex<f32>> = room
                .iter()
                .map(|&s| Complex::new(s, 0.0))
                .collect();
            b.resize(size, Complex::default());
            fft.process(&mut a);
            fft.process(&mut b);
            let mut spectrum: Vec<Complex<f32>> =
                a.iter().zip(&b).map(|(x, y)| x * y).collect();
            planner.plan_fft_inverse(size).process(&mut spectrum);
            spectrum[..48000]
                .iter()
                .map(|value| value.re / size as f32)
                .collect()
        };
        let calibration = b"20 0\n20000 0\n";
        let provenance = CaptureTakeProvenance {
            seat_id: None,
            microphone_id: "room-reference".into(),
            device_id: "declared-usb-device".into(),
            offset_samples: None,
            skew_ppm: None,
            residual_uncertainty_us: None,
            correction_applied: CaptureCorrection::None,
            timing_reference_id: None,
            calibration_id: digest(calibration),
            gain_db: 0.0,
            calibration_orientation: "on_axis".into(),
            position_m: [0.0; 3],
            position_uncertainty_mm: 2.0,
            preserves_acoustic_delay: true,
            quality_passed: true,
        };
        let config = RoomConfig {
            speakers: std::collections::HashMap::from([(
                "L".into(),
                SpeakerConfig::Single(MeasurementSource::Multiple(MeasurementMultiple {
                    measurements: vec![MeasurementRef::Named {
                        path: "response.csv".into(),
                        name: Some("room-reference".into()),
                    }],
                    speaker_name: None,
                    provenance: MeasurementProvenance {
                        capture_kind: ProvenanceCaptureKind::SpatialMagnitude,
                        capture: Some(CaptureProvenance {
                            geometry: CaptureGeometry::Spread,
                            takes: vec![provenance.clone()],
                            reflection_report: None,
                        }),
                        ..Default::default()
                    },
                    estimated_position: None,
                })),
            )]),
            recording_config: Some(RecordingConfiguration {
                recording_sample_rate: Some(48000),
                capture_handoff_file: Some(
                    autoeq_core::capture_handoff::CAPTURE_HANDOFF_FILENAME.into(),
                ),
                signal_type: Some("Sweep".into()),
                sweep_start_freq: Some(20.0),
                sweep_end_freq: Some(20000.0),
                ..Default::default()
            }),
            ..Default::default()
        };
        let config_bytes = serde_json::to_vec_pretty(&config).unwrap();
        let stimulus_bytes = capture_wav_bytes(&stimulus);
        let recording_bytes = capture_wav_bytes(&recording);
        let payloads: Vec<(&str, CaptureArtifactRole, &[u8])> = vec![
            ("recordings.json", CaptureArtifactRole::Configuration, &config_bytes),
            (
                "response.csv",
                CaptureArtifactRole::MagnitudeResponse,
                b"frequency_hz,spl_db\n100,80\n1000,80\n",
            ),
            ("calibration.txt", CaptureArtifactRole::Calibration, calibration),
            ("raw.wav", CaptureArtifactRole::RawAudio, &recording_bytes),
            ("processed.wav", CaptureArtifactRole::ProcessedAudio, &recording_bytes),
            ("stimulus.wav", CaptureArtifactRole::SupportingEvidence, &stimulus_bytes),
        ];
        let artifacts = payloads
            .iter()
            .map(|(file, role, bytes)| {
                std::fs::write(dir.path().join(file), bytes).unwrap();
                CaptureArtifactIdentity {
                    file: (*file).into(),
                    role: *role,
                    bytes: bytes.len() as u64,
                    sha256: digest(bytes),
                }
            })
            .collect();
        let handoff = CaptureHandoff {
            version: 1,
            producer: "fixture-producer".into(),
            producer_version: "1".into(),
            session_id: "session-1".into(),
            completion: CaptureCompletion::Complete,
            sample_rate_hz: 48000,
            source_ids: vec!["L".into()],
            microphone_ids: vec!["room-reference".into()],
            repeat_count: 1,
            selected_take_ids: None,
            parent_inventory_file: None,
            configuration_file: "recordings.json".into(),
            artifacts,
            takes: vec![CaptureTakeIdentity {
                take_id: "take-1".into(),
                source_id: "L".into(),
                repeat_index: 0,
                raw_audio_file: "raw.wav".into(),
                processed_audio_file: "processed.wav".into(),
                response_file: Some("response.csv".into()),
                calibration_file: "calibration.txt".into(),
                provenance,
            }],
        };
        std::fs::write(
            dir.path().join(autoeq_core::capture_handoff::CAPTURE_HANDOFF_FILENAME),
            serde_json::to_vec_pretty(&handoff).unwrap(),
        )
        .unwrap();

        let mut output = DspGraph::new("1");
        output.add_channel("L", Vec::new());
        let warnings = attach_capture_measured_acoustics(&mut output, &config, dir.path());
        assert!(
            !warnings.iter().any(|warning| warning.contains("unavailable")),
            "{warnings:?}"
        );
        let chain = &output.channels["L"];
        assert!(
            chain.early_reflections.is_some(),
            "reflection table attached from deconvolved capture IR"
        );
        assert!(chain.t60_octaves.is_some(), "octave T60 report attached");
        assert!(chain.pre_ir.is_none(), "synthesized pre_ir left untouched");
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
        let sti = chain
            .speech_transmission
            .as_ref()
            .expect("full-band native STI");
        assert_eq!(sti.basis, "measured_room_ir");
        assert_eq!(sti.octave_centers_hz[6], 8000.0);
        assert!((0.0..=1.0).contains(&sti.sti));
        assert!(
            !sti.warnings.is_empty(),
            "1 s capture carries truncation warning"
        );
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

    #[test]
    fn sti_report_roundtrips_and_legacy_graphs_remain_readable() {
        let mut output = DspGraph::new("1");
        output.add_channel("L", Vec::new());
        assert!(output.channels["L"].speech_transmission.is_none());
        let mut samples = vec![0.0; 96_000];
        samples[2400] = 1.0;
        let report = speech_transmission_report(&samples, 48_000.0).expect("STI");
        assert!(report.warnings.is_empty());
        output.channels.get_mut("L").unwrap().speech_transmission = Some(report);
        let json = serde_json::to_value(&output).unwrap();
        let decoded: DspGraph = serde_json::from_value(json.clone()).unwrap();
        assert!(
            decoded.channels["L"]
                .speech_transmission
                .as_ref()
                .unwrap()
                .sti
                > 0.99
        );
        let mut legacy = json;
        legacy["channels"]["L"]
            .as_object_mut()
            .unwrap()
            .remove("speech_transmission");
        let decoded: DspGraph = serde_json::from_value(legacy).unwrap();
        assert!(decoded.channels["L"].speech_transmission.is_none());
        assert!(speech_transmission_report(&samples, 3_000.0).is_err());
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("room.json");
        crate::output_bundle::save_output_bundle(&mut output, &path).expect("save");
        let restored = crate::output_bundle::load_output_bundle(&path).expect("load");
        assert!(
            restored.channels["L"]
                .speech_transmission
                .as_ref()
                .unwrap()
                .sti
                > 0.99
        );
    }

    #[test]
    fn sti_survives_native_bundle_and_stays_on_its_physical_driver() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = write_ir(dir.path(), "driver.csv", 48_000.0, 1.0);
        let mut source = source_for(path);
        source.output_channel = Some("L".into());
        source.driver = Some("mid".into());
        let declared = BTreeMap::from([("mid_capture".into(), source)]);
        let mut output = DspGraph::new("1");
        output.add_channel("L", Vec::new());
        output.channels.get_mut("L").unwrap().drivers = Some(vec![roomeq_model::DriverDspChain {
            measured_acoustics: None,
            name: "mid".into(),
            index: 0,
            plugins: Vec::new(),
            initial_curve: None,
            measured_band_hz: None,
        }]);
        attach_measured_acoustics(&mut output, &declared, dir.path()).expect("attach");
        assert!(output.channels["L"].speech_transmission.is_none());
        let get_sti = |graph: &DspGraph| {
            graph.channels["L"].drivers.as_ref().unwrap()[0]
                .measured_acoustics
                .as_ref()
                .unwrap()
                .speech_transmission
                .as_ref()
                .unwrap()
                .sti
        };
        let expected = get_sti(&output);
        let output_path = dir.path().join("room.json");
        crate::output_bundle::save_output_bundle(&mut output, &output_path).expect("save");
        let restored = crate::output_bundle::load_output_bundle(&output_path).expect("load");
        assert_eq!(get_sti(&restored), expected);
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
        // A saved predicted post-correction IR yields after-EQ candidates.
        let post_time_ms: Vec<f64> = (0..4800).map(|index| index as f64 / 48.0).collect();
        let mut post_amplitude = vec![0.0_f64; 4800];
        post_amplitude[240] = 1.0;
        post_amplitude[480] = 0.5;
        output.channels.get_mut("L").unwrap().post_ir = Some(IrWaveform {
            time_ms: post_time_ms,
            amplitude: post_amplitude,
        });
        let warnings =
            attach_measured_acoustics(&mut output, &declared, dir.path()).expect("attach");
        assert!(warnings.is_empty(), "unexpected warnings: {warnings:?}");
        let chain = &output.channels["L"];
        assert!(chain.early_late_curves.is_some());
        let reflections = chain.early_reflections.as_ref().expect("reflections");
        assert!(!reflections.post.is_empty(), "post-EQ candidates attached");
        assert!(chain.t60_octaves.is_some());
        assert!(chain.waterfall.is_some());
        assert!(chain.resonance_decays.is_some());
        assert!(chain.wavelet.is_some());
    }
}
