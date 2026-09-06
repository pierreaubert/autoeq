//! Seeded listening-stimulus synthesis for Stage 2 validation staging.
//!
//! Renders actual comparison audio (tones, bursts, sweeps, transients,
//! shaped and band-limited noise, harmonic complexes, masking probes)
//! plus level-calibrated programme material, with a content
//! manifest recording every parameter, seed, SPL mapping, and file hash.
//! JSON descriptors alone are not stimuli: only rendered WAV bytes plus
//! the manifest count as evidence inputs.
//!
//! Determinism: all randomness comes from [`SeededRng`]; rendering the
//! same spec twice yields identical bytes and an identical manifest (no
//! timestamps, relative file names only). Float synthesis may differ in
//! the last ulp across libm implementations; the manifest records the
//! platform so cross-machine comparisons stay honest.

use std::path::{Path, PathBuf};

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use super::SeededRng;

/// Renderer identifier recorded in every manifest.
pub const STIMULUS_RENDERER: &str = "roomeq-quality-stimuli-v1";
/// Named SPL conversion: affine full-scale-to-SPL map.
pub const CONVERSION_AFFINE_FS_SPL_V1: &str = "affine-fs-spl-v1";
/// Longest renderable stimulus (accident guard).
pub const MAX_STIMULUS_SECONDS: f64 = 60.0;
/// Synthesis headroom: normalized peaks sit 1 dB below full scale.
pub const SYNTHESIS_PEAK: f32 = 0.89125094;

/// One listenable stimulus kind.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum StimulusKind {
    /// Steady sine at `freq_hz` for `duration_s`.
    Tone {
        /// Tone frequency in Hz (must clear Nyquist with margin).
        freq_hz: f64,
        /// Duration in seconds.
        duration_s: f64,
    },
    /// Sine with raised-cosine onset/offset ramps of `ramp_ms`.
    ToneBurst {
        /// Tone frequency in Hz.
        freq_hz: f64,
        /// Duration in seconds including ramps.
        duration_s: f64,
        /// Ramp length in milliseconds each end.
        ramp_ms: f64,
    },
    /// Constant-amplitude frequency sweep with raised-cosine ends.
    Sweep {
        /// Start frequency in Hz.
        f0_hz: f64,
        /// End frequency in Hz.
        f1_hz: f64,
        /// Duration in seconds.
        duration_s: f64,
        /// `true` for logarithmic sweep, `false` for linear.
        log: bool,
    },
    /// Seeded-polarity impulse with a 2 ms exponential tail.
    TransientClick {
        /// Total duration in seconds (impulse at 10%).
        duration_s: f64,
    },
    /// Stationary noise, white or pink (Paul Kellet filter), seeded.
    ShapedNoise {
        /// `true` for pink, `false` for white.
        pink: bool,
        /// Duration in seconds.
        duration_s: f64,
    },
    /// Band-limited seeded noise for bandwidth-change comparisons: white
    /// noise through a windowed-sinc FIR bandpass.
    BandLimitedNoise {
        /// Lower band edge in Hz (must be positive).
        low_hz: f64,
        /// Upper band edge in Hz (must clear `low_hz` and Nyquist).
        high_hz: f64,
        /// Duration in seconds.
        duration_s: f64,
    },
    /// Harmonic complex for equal-level different-timbre comparisons:
    /// `n_harmonics` partials of `f0_hz` with per-octave tilt and seeded
    /// start phases. Equal loudness is a listening property; staging
    /// renders equal-SPL variants and the protocol tests them.
    HarmonicComplex {
        /// Fundamental in Hz.
        f0_hz: f64,
        /// Partial count (must be at least 1; highest partial must clear
        /// Nyquist with margin).
        n_harmonics: u32,
        /// Level fall per octave in dB (`0.0` keeps all partials equal).
        tilt_db_per_octave: f64,
        /// Duration in seconds.
        duration_s: f64,
    },
    /// Forward-masking probe: masker burst, silent gap, probe burst, each
    /// with 5 ms raised-cosine ends.
    MaskerProbe {
        /// Masker frequency in Hz.
        masker_freq_hz: f64,
        /// Probe frequency in Hz.
        probe_freq_hz: f64,
        /// Masker duration in seconds.
        masker_duration_s: f64,
        /// Silent gap in milliseconds (must be finite and non-negative).
        gap_ms: f64,
        /// Probe duration in seconds.
        probe_duration_s: f64,
    },
    /// External programme (speech/music) loaded from disk and verified by
    /// hash. Never synthesized: the manifest pins the exact bytes heard.
    ExternalProgramme {
        /// Source audio path (WAV, mono or multi-channel).
        path: PathBuf,
        /// Expected SHA-256 hex of the source file bytes.
        sha256: String,
        /// SPL the file's RMS is assumed to represent, when known. With
        /// `None`, level sweeps are file-relative and the manifest marks
        /// them non-absolute.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        assumed_rms_spl_db: Option<f64>,
    },
}

impl StimulusKind {
    /// Short stable filename stem for this kind.
    pub fn stem(&self) -> String {
        match self {
            Self::Tone { freq_hz, .. } => format!("tone-{freq_hz}hz"),
            Self::ToneBurst { freq_hz, .. } => format!("burst-{freq_hz}hz"),
            Self::Sweep { f0_hz, f1_hz, .. } => format!("sweep-{f0_hz}-{f1_hz}hz"),
            Self::TransientClick { .. } => String::from("click"),
            Self::ShapedNoise { pink, .. } => {
                String::from(if *pink { "noise-pink" } else { "noise-white" })
            }
            Self::BandLimitedNoise { low_hz, high_hz, .. } => {
                format!("noise-band-{low_hz}-{high_hz}hz")
            }
            Self::HarmonicComplex { f0_hz, n_harmonics, .. } => {
                format!("complex-{f0_hz}hz-x{n_harmonics}")
            }
            Self::MaskerProbe { masker_freq_hz, probe_freq_hz, gap_ms, .. } => {
                format!("masker-{masker_freq_hz}-{probe_freq_hz}hz-gap{gap_ms}ms")
            }
            Self::ExternalProgramme { path, .. } => format!(
                "programme-{}",
                path.file_stem()
                    .and_then(|stem| stem.to_str())
                    .unwrap_or("audio")
            ),
        }
    }

    /// Duration in seconds (`None` for external programme: read the file).
    pub fn duration_s(&self) -> Option<f64> {
        match self {
            Self::Tone { duration_s, .. }
            | Self::ToneBurst { duration_s, .. }
            | Self::TransientClick { duration_s, .. }
            | Self::ShapedNoise { duration_s, .. }
            | Self::BandLimitedNoise { duration_s, .. }
            | Self::HarmonicComplex { duration_s, .. } => Some(*duration_s),
            Self::Sweep { duration_s, .. } => Some(*duration_s),
            Self::MaskerProbe {
                masker_duration_s,
                gap_ms,
                probe_duration_s,
                ..
            } => Some(masker_duration_s + gap_ms / 1000.0 + probe_duration_s),
            Self::ExternalProgramme { .. } => None,
        }
    }
}

/// Absolute-SPL handling with a named conversion.
///
/// [`CONVERSION_AFFINE_FS_SPL_V1`] maps RMS to SPL affinely:
/// `spl = 20·log10(rms) + db_spl_at_0dbfs_rms`. A nominal setting (e.g.
/// 75) is a mapping assumption, not a room calibration — the manifest
/// records which.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct SplCalibration {
    /// Conversion name; only [`CONVERSION_AFFINE_FS_SPL_V1`] is supported.
    pub conversion: String,
    /// dB SPL produced by a 0 dBFS RMS sine under this mapping.
    pub db_spl_at_0dbfs_rms: f64,
    /// Sweep levels in dB SPL (absolute) or dB relative to file level
    /// (programme without `assumed_rms_spl_db`).
    #[serde(default)]
    pub levels_db: Vec<f64>,
}

impl SplCalibration {
    /// Validate the conversion name and level list.
    pub fn validate(&self) -> Result<(), String> {
        if self.conversion != CONVERSION_AFFINE_FS_SPL_V1 {
            return Err(format!("unknown SPL conversion '{}'", self.conversion));
        }
        if !self.db_spl_at_0dbfs_rms.is_finite() {
            return Err(String::from("db_spl_at_0dbfs_rms must be finite"));
        }
        if self.levels_db.is_empty() {
            return Err(String::from("levels_db must list at least one level"));
        }
        for level in &self.levels_db {
            if !level.is_finite() {
                return Err(String::from("levels_db must be finite"));
            }
        }
        Ok(())
    }
}

/// One line of a stimulus render request.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct StimulusRequest {
    /// What to render.
    pub kind: StimulusKind,
    /// Seed for all randomness in this stimulus.
    pub seed: u64,
}

/// One rendered file in a stimulus set.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct StimulusFile {
    /// Relative file name inside the set directory.
    pub file: String,
    /// SHA-256 hex of the WAV bytes.
    pub sha256: String,
    /// Rendered kind (programme entries keep source path + hash).
    pub kind: StimulusKind,
    /// Seed used.
    pub seed: u64,
    /// Sample rate in Hz.
    pub sample_rate_hz: f64,
    /// Frame count.
    pub frames: usize,
    /// Sweep level in dB SPL. `None` for file-relative programme sweeps.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub spl_db: Option<f64>,
    /// Gain applied to the normalized render, in dB.
    pub gain_db: f64,
    /// Whether `spl_db` is absolute (`true`) or file-relative (`false`,
    /// programme without an assumed reference level only).
    pub spl_absolute: bool,
}

/// Content manifest for a rendered stimulus set. Byte-identical across
/// runs of the same spec: relative names only, no timestamps.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct StimulusManifest {
    /// [`STIMULUS_RENDERER`].
    pub renderer: String,
    /// Crate version that rendered the set.
    pub renderer_version: String,
    /// `arch-os` of the rendering machine.
    pub platform: String,
    /// SPL calibration used for the sweeps.
    pub calibration: SplCalibration,
    /// Sample rate in Hz shared by the set.
    pub sample_rate_hz: f64,
    /// Rendered files in deterministic order.
    pub files: Vec<StimulusFile>,
}

/// SHA-256 hex of bytes.
pub fn sha256_hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

/// Validate shared render parameters.
fn validate_render(sample_rate_hz: f64, duration_s: f64) -> Result<usize, String> {
    if !(8_000.0..=192_000.0).contains(&sample_rate_hz) {
        return Err(format!("sample rate {sample_rate_hz} out of 8–192 kHz"));
    }
    if !(0.001..=MAX_STIMULUS_SECONDS).contains(&duration_s) {
        return Err(format!(
            "duration {duration_s}s outside 1 ms–{MAX_STIMULUS_SECONDS}s"
        ));
    }
    Ok((duration_s * sample_rate_hz).round() as usize)
}

/// Raised-cosine ramp multiplier for sample `index` of `frames`.
fn ramp_gain(index: usize, frames: usize, ramp_frames: usize) -> f64 {
    if ramp_frames == 0 {
        return 1.0;
    }
    let ramp = ramp_frames.min(frames / 2);
    if index < ramp {
        0.5 - 0.5 * (std::f64::consts::PI * index as f64 / ramp as f64).cos()
    } else if index >= frames - ramp {
        let reverse = frames - 1 - index;
        0.5 - 0.5 * (std::f64::consts::PI * reverse as f64 / ramp as f64).cos()
    } else {
        1.0
    }
}

/// Peak-normalize to [`SYNTHESIS_PEAK`]; errors on silence.
fn normalize_peak(samples: &mut [f32]) -> Result<(), String> {
    let peak = samples.iter().map(|v| v.abs()).fold(0.0_f32, f32::max);
    if !matches!(peak.partial_cmp(&0.0), Some(std::cmp::Ordering::Greater)) {
        return Err(String::from("synthesis produced silence"));
    }
    let gain = SYNTHESIS_PEAK / peak;
    for sample in samples.iter_mut() {
        *sample *= gain;
    }
    Ok(())
}

/// Synthesize one stimulus at unit headroom (peak [`SYNTHESIS_PEAK`]).
/// Pure function of kind, sample rate, and seed.
pub fn render_samples(
    kind: &StimulusKind,
    sample_rate_hz: f64,
    seed: u64,
) -> Result<Vec<f32>, String> {
    let mut rng = SeededRng::new(seed);
    match kind {
        StimulusKind::Tone { freq_hz, duration_s } => {
            let frames = validate_render(sample_rate_hz, *duration_s)?;
            check_tone_freq(*freq_hz, sample_rate_hz)?;
            let mut samples = Vec::with_capacity(frames);
            for index in 0..frames {
                let time = index as f64 / sample_rate_hz;
                samples.push((2.0 * std::f64::consts::PI * freq_hz * time).sin() as f32);
            }
            normalize_peak(&mut samples)?;
            Ok(samples)
        }
        StimulusKind::ToneBurst { freq_hz, duration_s, ramp_ms } => {
            let frames = validate_render(sample_rate_hz, *duration_s)?;
            check_tone_freq(*freq_hz, sample_rate_hz)?;
            if !ramp_ms.is_finite() || *ramp_ms < 0.0 {
                return Err(String::from("ramp_ms must be finite and non-negative"));
            }
            let ramp_frames = (ramp_ms / 1000.0 * sample_rate_hz).round() as usize;
            let mut samples = Vec::with_capacity(frames);
            for index in 0..frames {
                let time = index as f64 / sample_rate_hz;
                let envelope = ramp_gain(index, frames, ramp_frames);
                samples.push(
                    (envelope * (2.0 * std::f64::consts::PI * freq_hz * time).sin()) as f32,
                );
            }
            normalize_peak(&mut samples)?;
            Ok(samples)
        }
        StimulusKind::Sweep { f0_hz, f1_hz, duration_s, log } => {
            let frames = validate_render(sample_rate_hz, *duration_s)?;
            check_tone_freq(*f0_hz, sample_rate_hz)?;
            check_tone_freq(*f1_hz, sample_rate_hz)?;
            if f0_hz == f1_hz {
                return Err(String::from("sweep endpoints must differ"));
            }
            let mut phase = 0.0;
            let mut samples = Vec::with_capacity(frames);
            for index in 0..frames {
                let progress = index as f64 / frames as f64;
                let freq = if *log {
                    f0_hz * (f1_hz / f0_hz).powf(progress)
                } else {
                    f0_hz + (f1_hz - f0_hz) * progress
                };
                phase += 2.0 * std::f64::consts::PI * freq / sample_rate_hz;
                samples.push((phase.sin() * ramp_gain(index, frames, frames / 32)) as f32);
            }
            normalize_peak(&mut samples)?;
            Ok(samples)
        }
        StimulusKind::TransientClick { duration_s } => {
            let frames = validate_render(sample_rate_hz, *duration_s)?;
            let polarity = if rng.next_f64() < 0.5 { -1.0 } else { 1.0 };
            let decay = (-1.0 / (0.002 * sample_rate_hz)) as f32;
            let mut samples = vec![0.0_f32; frames];
            let start = frames / 10;
            for (offset, sample) in samples.iter_mut().enumerate().skip(start) {
                let age = (offset - start) as f32;
                *sample = polarity * (decay * age).exp();
                if sample.abs() < 1e-6 {
                    break;
                }
            }
            normalize_peak(&mut samples)?;
            Ok(samples)
        }
        StimulusKind::ShapedNoise { pink, duration_s } => {
            let frames = validate_render(sample_rate_hz, *duration_s)?;
            // Paul Kellet pink filter; white path bypasses it.
            let mut stages = [0.0_f64; 6];
            let mut samples = Vec::with_capacity(frames);
            for _ in 0..frames {
                let white = rng.next_gaussian();
                if !pink {
                    samples.push(white as f32);
                    continue;
                }
                stages[0] = 0.99886 * stages[0] + white * 0.0555179;
                stages[1] = 0.99332 * stages[1] + white * 0.0750759;
                stages[2] = 0.96900 * stages[2] + white * 0.1538520;
                stages[3] = 0.86650 * stages[3] + white * 0.3104856;
                stages[4] = 0.55000 * stages[4] + white * 0.5329522;
                stages[5] = -0.7616 * stages[5] - white * 0.0168980;
                let pink_sample =
                    stages[0] + stages[1] + stages[2] + stages[3] + stages[4] + stages[5] + white * 0.5362;
                samples.push((pink_sample * 0.11) as f32);
            }
            normalize_peak(&mut samples)?;
            Ok(samples)
        }
        StimulusKind::BandLimitedNoise { low_hz, high_hz, duration_s } => {
            let frames = validate_render(sample_rate_hz, *duration_s)?;
            if !low_hz.is_finite() || *low_hz <= 0.0 {
                return Err(format!("band low edge {low_hz} must be positive"));
            }
            if !high_hz.is_finite() || *high_hz <= *low_hz {
                return Err(format!(
                    "band high edge {high_hz} must clear low edge {low_hz}"
                ));
            }
            if *high_hz > 0.45 * sample_rate_hz {
                return Err(format!(
                    "band high edge {high_hz} Hz too close to Nyquist at {sample_rate_hz} Hz"
                ));
            }
            let taps = bandpass_fir(*low_hz, *high_hz, sample_rate_hz);
            let delay = (taps.len() - 1) / 2;
            let mut samples = Vec::with_capacity(frames);
            let mut history = vec![0.0_f64; taps.len()];
            for _ in 0..frames + delay {
                history.rotate_right(1);
                history[0] = rng.next_gaussian();
                let value: f64 = history
                    .iter()
                    .zip(taps.iter())
                    .map(|(sample, tap)| sample * tap)
                    .sum();
                samples.push(value as f32);
            }
            let mut samples = samples[delay..].to_vec();
            samples.truncate(frames);
            normalize_peak(&mut samples)?;
            Ok(samples)
        }
        StimulusKind::HarmonicComplex { f0_hz, n_harmonics, tilt_db_per_octave, duration_s } => {
            let frames = validate_render(sample_rate_hz, *duration_s)?;
            check_tone_freq(*f0_hz, sample_rate_hz)?;
            if *n_harmonics == 0 {
                return Err(String::from("harmonic complex needs at least one partial"));
            }
            let top_hz = *f0_hz * f64::from(*n_harmonics);
            check_tone_freq(top_hz, sample_rate_hz).map_err(|_| {
                format!("highest partial {top_hz} Hz too close to Nyquist at {sample_rate_hz} Hz")
            })?;
            if !tilt_db_per_octave.is_finite() {
                return Err(String::from("tilt_db_per_octave must be finite"));
            }
            let partials: Vec<(f64, f64, f64)> = (1..=*n_harmonics)
                .map(|order| {
                    let amplitude = 10.0_f64.powf(
                        -tilt_db_per_octave * f64::from(order).log2() / 20.0,
                    );
                    let phase = 2.0 * std::f64::consts::PI * rng.next_f64();
                    (f64::from(order) * *f0_hz, amplitude, phase)
                })
                .collect();
            let mut samples = Vec::with_capacity(frames);
            for index in 0..frames {
                let time = index as f64 / sample_rate_hz;
                let value: f64 = partials
                    .iter()
                    .map(|(freq, amplitude, phase)| {
                        amplitude * (2.0 * std::f64::consts::PI * freq * time + phase).sin()
                    })
                    .sum();
                samples.push(value as f32);
            }
            normalize_peak(&mut samples)?;
            Ok(samples)
        }
        StimulusKind::MaskerProbe {
            masker_freq_hz,
            probe_freq_hz,
            masker_duration_s,
            gap_ms,
            probe_duration_s,
        } => {
            check_tone_freq(*masker_freq_hz, sample_rate_hz)?;
            check_tone_freq(*probe_freq_hz, sample_rate_hz)?;
            if !gap_ms.is_finite() || *gap_ms < 0.0 {
                return Err(String::from("gap_ms must be finite and non-negative"));
            }
            let masker_frames =
                validate_render(sample_rate_hz, *masker_duration_s).map_err(|_| {
                    format!("masker duration {masker_duration_s}s outside 1 ms–{MAX_STIMULUS_SECONDS}s")
                })?;
            let probe_frames =
                validate_render(sample_rate_hz, *probe_duration_s).map_err(|_| {
                    format!("probe duration {probe_duration_s}s outside 1 ms–{MAX_STIMULUS_SECONDS}s")
                })?;
            let gap_frames = (gap_ms / 1000.0 * sample_rate_hz).round() as usize;
            let ramp_frames = (0.005 * sample_rate_hz).round() as usize;
            let total = masker_frames + gap_frames + probe_frames;
            if total == 0 || total as f64 > MAX_STIMULUS_SECONDS * sample_rate_hz {
                return Err(String::from("masker-probe total duration out of range"));
            }
            let mut samples = Vec::with_capacity(total);
            let burst = |freq_hz: f64, frames: usize| {
                (0..frames)
                    .map(|index| {
                        let time = index as f64 / sample_rate_hz;
                        (ramp_gain(index, frames, ramp_frames)
                            * (2.0 * std::f64::consts::PI * freq_hz * time).sin())
                            as f32
                    })
                    .collect::<Vec<f32>>()
            };
            samples.extend(burst(*masker_freq_hz, masker_frames));
            samples.extend(std::iter::repeat_n(0.0_f32, gap_frames));
            samples.extend(burst(*probe_freq_hz, probe_frames));
            normalize_peak(&mut samples)?;
            Ok(samples)
        }
        StimulusKind::ExternalProgramme { path, sha256, .. } => load_programme(path, sha256),
    }
}

/// 127-tap Hamming-windowed sinc FIR bandpass (lowpass difference).
fn bandpass_fir(low_hz: f64, high_hz: f64, sample_rate_hz: f64) -> Vec<f64> {
    const TAPS: usize = 127;
    let middle = (TAPS - 1) as f64 / 2.0;
    let sinc = |x: f64| {
        if x == 0.0 {
            1.0
        } else {
            (std::f64::consts::PI * x).sin() / (std::f64::consts::PI * x)
        }
    };
    (0..TAPS)
        .map(|tap| {
            let offset = tap as f64 - middle;
            let lowpass = |edge_hz: f64| {
                2.0 * edge_hz / sample_rate_hz * sinc(2.0 * edge_hz * offset / sample_rate_hz)
            };
            let window = 0.54
                - 0.46
                    * (2.0 * std::f64::consts::PI * tap as f64 / (TAPS - 1) as f64).cos();
            (lowpass(high_hz) - lowpass(low_hz)) * window
        })
        .collect()
}

/// Tones must clear Nyquist with 5% margin (anti-imaging guard).
fn check_tone_freq(freq_hz: f64, sample_rate_hz: f64) -> Result<(), String> {
    if !freq_hz.is_finite() || freq_hz <= 0.0 {
        return Err(format!("tone frequency {freq_hz} must be positive"));
    }
    if freq_hz > 0.45 * sample_rate_hz {
        return Err(format!(
            "tone frequency {freq_hz} Hz too close to Nyquist at {sample_rate_hz} Hz"
        ));
    }
    Ok(())
}

/// Load external programme audio: verify existence, hash, and mono-ize by
/// channel mean (recorded in the manifest entry by the caller).
fn load_programme(path: &Path, sha256: &str) -> Result<Vec<f32>, String> {
    let bytes =
        std::fs::read(path).map_err(|error| format!("programme file unreadable: {error}"))?;
    if sha256_hex(&bytes) != sha256.to_lowercase() {
        return Err(String::from(
            "programme file hash mismatch: bytes heard would differ from manifest",
        ));
    }
    let cursor = std::io::Cursor::new(bytes);
    let mut reader =
        hound::WavReader::new(cursor).map_err(|error| format!("programme WAV error: {error}"))?;
    let channels = reader.spec().channels as usize;
    if channels == 0 {
        return Err(String::from("programme WAV has no channels"));
    }
    let read_frame = |frame: &[f32]| frame.iter().sum::<f32>() / channels as f32;
    let mut mono = Vec::new();
    let mut frame = Vec::with_capacity(channels);
    match reader.spec().sample_format {
        hound::SampleFormat::Float => {
            for sample in reader.samples::<f32>() {
                let sample =
                    sample.map_err(|error| format!("programme sample error: {error}"))?;
                frame.push(sample);
                if frame.len() == channels {
                    mono.push(read_frame(&frame));
                    frame.clear();
                }
            }
        }
        hound::SampleFormat::Int => {
            for sample in reader.samples::<i32>() {
                let sample =
                    sample.map_err(|error| format!("programme sample error: {error}"))?;
                frame.push(sample as f32 / i32::MAX as f32);
                if frame.len() == channels {
                    mono.push(read_frame(&frame));
                    frame.clear();
                }
            }
        }
    }
    if mono.is_empty() {
        return Err(String::from("programme file has no frames"));
    }
    normalize_peak(&mut mono)?;
    Ok(mono)
}

/// RMS of a sample buffer (0.0 for empty).
fn rms(samples: &[f32]) -> f64 {
    if samples.is_empty() {
        return 0.0;
    }
    (samples.iter().map(|v| f64::from(*v).powi(2)).sum::<f64>() / samples.len() as f64).sqrt()
}

/// How one sweep level interprets `level_db`.
#[derive(Debug, Clone, Copy, PartialEq)]
enum LevelMode {
    /// `level_db` is an absolute dB SPL target; `reference_spl_db` is the
    /// SPL the render RMS represents (conversion-mapped, or the declared
    /// programme assumption).
    Absolute { reference_spl_db: f64 },
    /// `level_db` is a gain in dB relative to the file as rendered.
    /// Programme without a declared reference level only.
    Relative,
}

/// Apply one sweep level to a normalized render.
///
/// Returns `(scaled, gain_db, spl_db, spl_absolute)`. Errors instead of
/// clipping: peaks past full scale are never silently truncated.
fn apply_level(
    samples: &[f32],
    level_db: f64,
    mode: LevelMode,
) -> Result<(Vec<f32>, f64, Option<f64>, bool), String> {
    let file_rms = rms(samples);
    if !matches!(
        file_rms.partial_cmp(&0.0),
        Some(std::cmp::Ordering::Greater)
    ) {
        return Err(String::from("cannot level empty render"));
    }
    let (gain_db, spl_db, spl_absolute) = match mode {
        LevelMode::Absolute { reference_spl_db } => {
            let gain_db = level_db - reference_spl_db;
            (gain_db, Some(level_db), true)
        }
        LevelMode::Relative => (level_db, None, false),
    };
    let gain = 10.0_f64.powf(gain_db / 20.0);
    let peak = samples.iter().map(|v| v.abs()).fold(0.0_f32, f32::max) as f64;
    if peak * gain > 1.0 {
        return Err(format!(
            "level {level_db} dB would clip (peak {peak:.3} x gain {gain_db:.1} dB); lower the level or the 0dBFS reference",
        ));
    }
    let scaled: Vec<f32> = samples.iter().map(|v| (*v as f64 * gain) as f32).collect();
    Ok((scaled, gain_db, spl_db, spl_absolute))
}

/// Write mono 32-bit float WAV bytes.
fn write_wav_bytes(samples: &[f32], sample_rate_hz: f64) -> Result<Vec<u8>, String> {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: sample_rate_hz.round() as u32,
        bits_per_sample: 32,
        sample_format: hound::SampleFormat::Float,
    };
    let mut cursor = std::io::Cursor::new(Vec::new());
    {
        let mut writer = hound::WavWriter::new(&mut cursor, spec)
            .map_err(|error| format!("WAV writer error: {error}"))?;
        for sample in samples {
            writer
                .write_sample(*sample)
                .map_err(|error| format!("WAV sample error: {error}"))?;
        }
        writer
            .finalize()
            .map_err(|error| format!("WAV finalize error: {error}"))?;
    }
    Ok(cursor.into_inner())
}

/// Render a full stimulus set: every request at every sweep level.
///
/// Writes `{index:02}-{stem}-{seed}__L{level}.wav` files plus
/// `stimulus-manifest.json` into `dir` (created when missing). File names
/// use relative paths only, so re-rendering the same spec reproduces the
/// manifest byte-for-byte.
pub fn render_stimulus_set(
    dir: &Path,
    requests: &[StimulusRequest],
    calibration: &SplCalibration,
    sample_rate_hz: f64,
) -> Result<StimulusManifest, String> {
    calibration.validate()?;
    if requests.is_empty() {
        return Err(String::from("stimulus set must list at least one request"));
    }
    std::fs::create_dir_all(dir)
        .map_err(|error| format!("cannot create stimulus dir: {error}"))?;
    let mut files = Vec::new();
    for (index, request) in requests.iter().enumerate() {
        let samples = render_samples(&request.kind, sample_rate_hz, request.seed)?;
        let (assumed, is_programme) = match &request.kind {
            StimulusKind::ExternalProgramme { assumed_rms_spl_db, .. } => {
                (*assumed_rms_spl_db, true)
            }
            _ => (None, false),
        };
        for level_db in &calibration.levels_db {
            // Synthesized renders map RMS through the conversion; declared
            // programme uses its assumption; undeclared programme sweeps
            // file-relative gains.
            let file_rms = rms(&samples);
            let mode = if is_programme && assumed.is_none() {
                LevelMode::Relative
            } else {
                let reference = assumed.unwrap_or_else(|| {
                    20.0 * file_rms.log10() + calibration.db_spl_at_0dbfs_rms
                });
                LevelMode::Absolute {
                    reference_spl_db: reference,
                }
            };
            let (scaled, gain_db, spl_db, spl_absolute) =
                apply_level(&samples, *level_db, mode)?;
            let file_name = format!(
                "{index:02}-{}-{}__L{level_db}.wav",
                request.kind.stem(),
                request.seed
            );
            let wav = write_wav_bytes(&scaled, sample_rate_hz)?;
            std::fs::write(dir.join(&file_name), &wav)
                .map_err(|error| format!("cannot write stimulus file: {error}"))?;
            files.push(StimulusFile {
                file: file_name,
                sha256: sha256_hex(&wav),
                kind: request.kind.clone(),
                seed: request.seed,
                sample_rate_hz,
                frames: scaled.len(),
                spl_db,
                gain_db,
                spl_absolute,
            });
        }
    }
    let manifest = StimulusManifest {
        renderer: String::from(STIMULUS_RENDERER),
        renderer_version: String::from(env!("CARGO_PKG_VERSION")),
        platform: format!("{}-{}", std::env::consts::ARCH, std::env::consts::OS),
        calibration: calibration.clone(),
        sample_rate_hz,
        files,
    };
    let manifest_json = serde_json::to_string_pretty(&manifest)
        .map_err(|error| format!("manifest serialize error: {error}"))?;
    std::fs::write(dir.join("stimulus-manifest.json"), manifest_json)
        .map_err(|error| format!("cannot write manifest: {error}"))?;
    Ok(manifest)
}

#[cfg(test)]
mod stimuli_tests {
    use super::*;

    const SAMPLE_RATE: f64 = 48_000.0;

    fn calibration(levels: &[f64]) -> SplCalibration {
        SplCalibration {
            conversion: String::from(CONVERSION_AFFINE_FS_SPL_V1),
            db_spl_at_0dbfs_rms: 90.0,
            levels_db: levels.iter().copied().collect(),
        }
    }

    fn test_dir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "roomeq-stimuli-test-{}-{name}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        dir
    }

    #[test]
    fn all_kinds_render_finite_bounded_audio() {
        let kinds = [
            StimulusKind::Tone { freq_hz: 1000.0, duration_s: 0.1 },
            StimulusKind::ToneBurst { freq_hz: 440.0, duration_s: 0.1, ramp_ms: 5.0 },
            StimulusKind::Sweep { f0_hz: 100.0, f1_hz: 8000.0, duration_s: 0.2, log: true },
            StimulusKind::Sweep { f0_hz: 100.0, f1_hz: 8000.0, duration_s: 0.2, log: false },
            StimulusKind::TransientClick { duration_s: 0.1 },
            StimulusKind::ShapedNoise { pink: false, duration_s: 0.2 },
            StimulusKind::ShapedNoise { pink: true, duration_s: 0.2 },
            StimulusKind::BandLimitedNoise { low_hz: 500.0, high_hz: 1500.0, duration_s: 0.2 },
            StimulusKind::HarmonicComplex { f0_hz: 220.0, n_harmonics: 8, tilt_db_per_octave: 6.0, duration_s: 0.2 },
            StimulusKind::MaskerProbe { masker_freq_hz: 1000.0, probe_freq_hz: 1000.0, masker_duration_s: 0.1, gap_ms: 20.0, probe_duration_s: 0.05 },
        ];
        for kind in kinds {
            let duration = kind.duration_s().unwrap();
            let samples = render_samples(&kind, SAMPLE_RATE, 11).unwrap();
            assert_eq!(samples.len(), (duration * SAMPLE_RATE).round() as usize);
            assert!(samples.iter().all(|v| v.is_finite()), "{kind:?}");
            let peak = samples.iter().map(|v| v.abs()).fold(0.0_f32, f32::max);
            assert!(peak <= 1.0, "{kind:?} peak {peak}");
            assert!(peak >= 0.5, "{kind:?} suspiciously quiet ({peak})");
        }
    }

    #[test]
    fn render_is_deterministic() {
        let kind = StimulusKind::ShapedNoise { pink: true, duration_s: 0.3 };
        let first = render_samples(&kind, SAMPLE_RATE, 20260905).unwrap();
        let second = render_samples(&kind, SAMPLE_RATE, 20260905).unwrap();
        assert_eq!(first, second);
        let other = render_samples(&kind, SAMPLE_RATE, 20260906).unwrap();
        assert_ne!(first, other);
    }

    #[test]
    fn invalid_params_fail_closed() {
        let tone = StimulusKind::Tone { freq_hz: 1000.0, duration_s: 0.1 };
        assert!(render_samples(&tone, 1000.0, 1).is_err());
        let long = StimulusKind::Tone { freq_hz: 1000.0, duration_s: 120.0 };
        assert!(render_samples(&long, SAMPLE_RATE, 1).is_err());
        let alias = StimulusKind::Tone { freq_hz: 30_000.0, duration_s: 0.1 };
        assert!(render_samples(&alias, SAMPLE_RATE, 1).is_err());
        let flat_sweep = StimulusKind::Sweep { f0_hz: 500.0, f1_hz: 500.0, duration_s: 0.2, log: true };
        assert!(render_samples(&flat_sweep, SAMPLE_RATE, 1).is_err());
        assert!(render_stimulus_set(&test_dir("empty"), &[], &calibration(&[75.0]), SAMPLE_RATE).is_err());
        let bad_conversion = SplCalibration {
            conversion: String::from("nope"),
            db_spl_at_0dbfs_rms: 90.0,
            levels_db: vec![75.0],
        };
        let request = StimulusRequest { kind: tone, seed: 1 };
        assert!(render_stimulus_set(&test_dir("badconv"), &[request], &bad_conversion, SAMPLE_RATE).is_err());
    }

    #[test]
    fn spl_mapping_hits_the_declared_level() {
        // 1 kHz sine at 75 dB SPL under a 90 dBFS reference: file RMS must
        // equal 10^((75-90)/20) exactly (gain arithmetic, not hearing).
        let kind = StimulusKind::Tone { freq_hz: 1000.0, duration_s: 0.5 };
        let samples = render_samples(&kind, SAMPLE_RATE, 3).unwrap();
        let file_rms = rms(&samples);
        let reference = 20.0 * file_rms.log10() + 90.0;
        let (scaled, gain_db, spl_db, absolute) =
            apply_level(&samples, 75.0, LevelMode::Absolute { reference_spl_db: reference })
                .unwrap();
        assert!(absolute);
        assert_eq!(spl_db, Some(75.0));
        let expected_gain = 75.0 - reference;
        assert!((gain_db - expected_gain).abs() < 1e-12);
        assert!((rms(&scaled) - 10.0_f64.powf((75.0 - 90.0) / 20.0)).abs() < 1e-6);
    }

    #[test]
    fn clipping_errors_instead_of_truncating() {
        let kind = StimulusKind::Tone { freq_hz: 1000.0, duration_s: 0.1 };
        let samples = render_samples(&kind, SAMPLE_RATE, 3).unwrap();
        let file_rms = rms(&samples);
        let reference = 20.0 * file_rms.log10() + 90.0;
        let loud = apply_level(&samples, 130.0, LevelMode::Absolute { reference_spl_db: reference });
        assert!(loud.is_err(), "130 dB SPL from a 90 dBFS map must refuse");
    }

    /// Goertzel energy at `freq_hz` (exact-bin test tones stay orthogonal).
    fn goertzel_power(samples: &[f32], freq_hz: f64, sample_rate_hz: f64) -> f64 {
        let omega = 2.0 * std::f64::consts::PI * freq_hz / sample_rate_hz;
        let cosine = omega.cos();
        let (mut s0, mut s1, mut s2) = (0.0_f64, 0.0_f64, 0.0_f64);
        for sample in samples {
            s0 = f64::from(*sample) + 2.0 * cosine * s1 - s2;
            s2 = s1;
            s1 = s0;
        }
        s1 * s1 + s2 * s2 - 2.0 * cosine * s1 * s2
    }

    #[test]
    fn band_limited_noise_holds_its_band() {
        // 8 kHz rate, 500–1500 Hz band, exact-bin probe tones: in-band
        // energy must dominate out-of-band by 20 dB.
        let kind = StimulusKind::BandLimitedNoise {
            low_hz: 500.0,
            high_hz: 1500.0,
            duration_s: 0.5,
        };
        let samples = render_samples(&kind, 8_000.0, 21).unwrap();
        let in_band = goertzel_power(&samples, 1000.0, 8_000.0);
        let out_band = goertzel_power(&samples, 3000.0, 8_000.0);
        assert!(in_band > 100.0 * out_band, "in {in_band} vs out {out_band}");
    }

    #[test]
    fn harmonic_complex_weights_partials_by_tilt() {
        // f0 200 Hz at 8 kHz, 0.5 s: exact bins, orthogonal partials.
        // 12 dB/oct tilt puts 800 Hz (2 octaves up) 24 dB down.
        let kind = StimulusKind::HarmonicComplex {
            f0_hz: 200.0,
            n_harmonics: 4,
            tilt_db_per_octave: 12.0,
            duration_s: 0.5,
        };
        let samples = render_samples(&kind, 8_000.0, 22).unwrap();
        let fundamental = goertzel_power(&samples, 200.0, 8_000.0);
        let fourth = goertzel_power(&samples, 800.0, 8_000.0);
        let ratio = fundamental / fourth;
        assert!(ratio > 50.0, "tilt ratio {ratio}");
        // Flat tilt keeps partials at comparable energy (within 6 dB).
        let flat = StimulusKind::HarmonicComplex {
            f0_hz: 200.0,
            n_harmonics: 4,
            tilt_db_per_octave: 0.0,
            duration_s: 0.5,
        };
        let flat_samples = render_samples(&flat, 8_000.0, 22).unwrap();
        let flat_ratio = goertzel_power(&flat_samples, 200.0, 8_000.0)
            / goertzel_power(&flat_samples, 800.0, 8_000.0);
        assert!(flat_ratio < 4.0, "flat ratio {flat_ratio}");
    }

    #[test]
    fn masker_probe_gap_is_silent_between_bursts() {
        let masker_s = 0.1;
        let gap_ms = 20.0;
        let kind = StimulusKind::MaskerProbe {
            masker_freq_hz: 1000.0,
            probe_freq_hz: 1500.0,
            masker_duration_s: masker_s,
            gap_ms,
            probe_duration_s: 0.05,
        };
        let samples = render_samples(&kind, SAMPLE_RATE, 23).unwrap();
        let masker_frames = (masker_s * SAMPLE_RATE).round() as usize;
        let gap_frames = (gap_ms / 1000.0 * SAMPLE_RATE).round() as usize;
        assert_eq!(
            samples.len(),
            masker_frames + gap_frames + (0.05 * SAMPLE_RATE).round() as usize
        );
        let masker_rms = rms(&samples[..masker_frames]);
        assert!(masker_rms > 0.1, "masker rms {masker_rms}");
        let gap_rms = rms(&samples[masker_frames..masker_frames + gap_frames]);
        assert!(gap_rms < 0.01 * masker_rms, "gap rms {gap_rms}");
        let probe_rms = rms(&samples[masker_frames + gap_frames..]);
        assert!(probe_rms > 0.1, "probe rms {probe_rms}");
    }

    #[test]
    fn new_kinds_fail_closed() {
        let inverted = StimulusKind::BandLimitedNoise {
            low_hz: 2000.0,
            high_hz: 1000.0,
            duration_s: 0.2,
        };
        assert!(render_samples(&inverted, SAMPLE_RATE, 1).is_err());
        let too_wide = StimulusKind::BandLimitedNoise {
            low_hz: 100.0,
            high_hz: 30_000.0,
            duration_s: 0.2,
        };
        assert!(render_samples(&too_wide, SAMPLE_RATE, 1).is_err());
        let no_partials = StimulusKind::HarmonicComplex {
            f0_hz: 220.0,
            n_harmonics: 0,
            tilt_db_per_octave: 6.0,
            duration_s: 0.2,
        };
        assert!(render_samples(&no_partials, SAMPLE_RATE, 1).is_err());
        let aliasing_top = StimulusKind::HarmonicComplex {
            f0_hz: 10_000.0,
            n_harmonics: 4,
            tilt_db_per_octave: 6.0,
            duration_s: 0.2,
        };
        assert!(render_samples(&aliasing_top, SAMPLE_RATE, 1).is_err());
        let bad_tilt = StimulusKind::HarmonicComplex {
            f0_hz: 220.0,
            n_harmonics: 4,
            tilt_db_per_octave: f64::NAN,
            duration_s: 0.2,
        };
        assert!(render_samples(&bad_tilt, SAMPLE_RATE, 1).is_err());
        let negative_gap = StimulusKind::MaskerProbe {
            masker_freq_hz: 1000.0,
            probe_freq_hz: 1000.0,
            masker_duration_s: 0.1,
            gap_ms: -5.0,
            probe_duration_s: 0.05,
        };
        assert!(render_samples(&negative_gap, SAMPLE_RATE, 1).is_err());
    }

    #[test]
    fn set_render_is_manifest_deterministic_and_hashed() {
        let dir_a = test_dir("set-a");
        let dir_b = test_dir("set-b");
        // Levels stay clear of the click's crest-factor clip point: an
        // impulse RMS cannot reach high absolute SPLs without truncating.
        let manifest_a = render_stimulus_set(
            &dir_a,
            &make_set_requests(),
            &calibration(&[55.0, 65.0]),
            SAMPLE_RATE,
        )
        .unwrap();
        let manifest_b = render_stimulus_set(
            &dir_b,
            &make_set_requests(),
            &calibration(&[55.0, 65.0]),
            SAMPLE_RATE,
        )
        .unwrap();
        assert_eq!(manifest_a, manifest_b);
        assert_eq!(manifest_a.files.len(), 4);
        // Manifest JSON itself is byte-stable (no timestamps/paths).
        let json_a = std::fs::read(dir_a.join("stimulus-manifest.json")).unwrap();
        let json_b = std::fs::read(dir_b.join("stimulus-manifest.json")).unwrap();
        assert_eq!(json_a, json_b);
        // Every file hash verifies against bytes on disk.
        for entry in &manifest_a.files {
            let bytes = std::fs::read(dir_a.join(&entry.file)).unwrap();
            assert_eq!(sha256_hex(&bytes), entry.sha256);
            // And the WAV parses with the declared frame count.
            let reader = hound::WavReader::new(std::io::Cursor::new(bytes)).unwrap();
            assert_eq!(reader.duration() as usize, entry.frames);
            assert_eq!(reader.spec().sample_rate as f64, SAMPLE_RATE);
        }
        let _ = std::fs::remove_dir_all(&dir_a);
        let _ = std::fs::remove_dir_all(&dir_b);
    }

    fn make_set_requests() -> Vec<StimulusRequest> {
        vec![
            StimulusRequest {
                kind: StimulusKind::Tone { freq_hz: 1000.0, duration_s: 0.1 },
                seed: 5,
            },
            StimulusRequest {
                kind: StimulusKind::TransientClick { duration_s: 0.1 },
                seed: 6,
            },
        ]
    }

    #[test]
    fn external_programme_validates_hash_and_presence() {
        let dir = test_dir("programme");
        std::fs::create_dir_all(&dir).unwrap();
        // Write a small known WAV to stand in for programme material.
        let source = dir.join("speech.wav");
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: 48_000,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        {
            let mut writer = hound::WavWriter::create(&source, spec).unwrap();
            for index in 0..4800 {
                let sample = ((index as f32 * 0.1).sin() * 10_000.0) as i16;
                writer.write_sample(sample).unwrap();
            }
            writer.finalize().unwrap();
        }
        let bytes = std::fs::read(&source).unwrap();
        let hash = sha256_hex(&bytes);
        let good = StimulusKind::ExternalProgramme {
            path: source.clone(),
            sha256: hash.clone(),
            assumed_rms_spl_db: Some(70.0),
        };
        let samples = render_samples(&good, SAMPLE_RATE, 1).unwrap();
        assert!(!samples.is_empty());
        // Wrong hash refuses (bytes heard would differ from manifest).
        let tampered = StimulusKind::ExternalProgramme {
            path: source.clone(),
            sha256: String::from("00").repeat(32),
            assumed_rms_spl_db: Some(70.0),
        };
        assert!(render_samples(&tampered, SAMPLE_RATE, 1).is_err());
        // Missing file refuses.
        let missing = StimulusKind::ExternalProgramme {
            path: dir.join("absent.wav"),
            sha256: hash,
            assumed_rms_spl_db: None,
        };
        assert!(render_samples(&missing, SAMPLE_RATE, 1).is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn uncalibrated_programme_sweeps_relative_gains() {
        // Without an assumed reference, sweep levels are relative gains
        // and the manifest says so (spl_absolute false, no spl_db).
        let kind = StimulusKind::Tone { freq_hz: 500.0, duration_s: 0.1 };
        let samples = render_samples(&kind, SAMPLE_RATE, 9).unwrap();
        let (scaled, gain_db, spl_db, absolute) =
            apply_level(&samples, -6.0, LevelMode::Relative).unwrap();
        assert!(!absolute);
        assert!(spl_db.is_none());
        assert!((gain_db + 6.0).abs() < 1e-12);
        let expected = rms(&samples) * 10.0_f64.powf(-6.0 / 20.0);
        assert!(((rms(&scaled) / expected) - 1.0).abs() < 1e-6);
    }
}
