//! Rendered/stimulus controls (plan tasks S3, gate G3).
//!
//! Physical-signal fixtures only: equal-energy spectra, tilt, sustained
//! resonance excitation, transients, AM-rate sweeps, beating tones and a
//! missing-fundamental complex, plus a constant output-loss pair whose raw
//! loss survives display normalization (F11). The fixtures expose the signal
//! samples, generating settings, sample rates and seeds. They carry no
//! sone/asper/preference truth and make no equal-loudness or equivalence
//! claims; QA must not read them as perceptual verdicts.

use crate::Curve;
use crate::error::{AutoeqError, Result};

/// Documented sample rate for all stimulus signals.
pub const STIMULUS_SAMPLE_RATE_HZ: f64 = 48_000.0;

/// Which physical control a stimulus signal implements.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StimulusKind {
    EqualEnergyA,
    EqualEnergyB,
    Tilt,
    Resonance,
    Transient,
    AmSweep,
    Beats,
    MissingFundamental,
}

impl StimulusKind {
    pub fn stable_id(&self) -> &'static str {
        match self {
            StimulusKind::EqualEnergyA => "S3-equal-energy-a",
            StimulusKind::EqualEnergyB => "S3-equal-energy-b",
            StimulusKind::Tilt => "S3-tilt",
            StimulusKind::Resonance => "S3-resonance",
            StimulusKind::Transient => "S3-transient",
            StimulusKind::AmSweep => "S3-am-sweep",
            StimulusKind::Beats => "S3-beats",
            StimulusKind::MissingFundamental => "S3-missing-fundamental",
        }
    }
}

/// A physical stimulus signal with its generating settings.
///
/// `energy` is the plain sum of squares (physical, not perceptual);
/// `peak` is the maximum absolute sample. Neither implies loudness.
#[derive(Debug, Clone)]
pub struct StimulusSignal {
    pub id: String,
    pub kind: StimulusKind,
    pub sample_rate_hz: f64,
    pub seed: u64,
    /// Component frequencies in Hz (empty for non-tonal signals).
    pub frequencies_hz: Vec<f64>,
    /// Component amplitudes, parallel to `frequencies_hz`.
    pub amplitudes: Vec<f64>,
    pub duration_s: f64,
    pub samples: Vec<f64>,
    pub energy: f64,
    pub peak: f64,
}

fn finish(
    kind: StimulusKind,
    sample_rate_hz: f64,
    seed: u64,
    frequencies_hz: Vec<f64>,
    amplitudes: Vec<f64>,
    duration_s: f64,
    samples: Vec<f64>,
) -> StimulusSignal {
    let energy: f64 = samples.iter().map(|v| v * v).sum();
    let peak: f64 = samples.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
    StimulusSignal {
        id: kind.stable_id().to_string(),
        kind,
        sample_rate_hz,
        seed,
        frequencies_hz,
        amplitudes,
        duration_s,
        samples,
        energy,
        peak,
    }
}

fn check_rate_duration(sample_rate_hz: f64, duration_s: f64) -> Result<usize> {
    if !sample_rate_hz.is_finite()
        || sample_rate_hz <= 0.0
        || !duration_s.is_finite()
        || duration_s <= 0.0
    {
        return Err(AutoeqError::InvalidConfiguration {
            message: "stimulus signals require a finite positive sample rate and duration"
                .to_string(),
        });
    }
    Ok((sample_rate_hz * duration_s) as usize)
}

/// Scale samples and their component amplitudes to a target peak.
///
/// Zero-phase tone banks sum constructively at t = 0; normalization keeps
/// every control inside the declared amplitude bound without changing the
/// relative spectrum. Fully deterministic.
fn normalize_peak(samples: Vec<f64>, amplitudes: &mut [f64], target: f64) -> Vec<f64> {
    let peak = samples.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
    if peak > 0.0 {
        let gain = target / peak;
        for a in amplitudes.iter_mut() {
            *a *= gain;
        }
        samples.iter().map(|v| v * gain).collect()
    } else {
        samples
    }
}

/// Sum of cosines over integer-cycle components (zero starting phase).
fn tone_bank(n: usize, sample_rate_hz: f64, components: &[(f64, f64)]) -> Vec<f64> {
    let mut out = vec![0.0; n];
    for (i, sample) in out.iter_mut().enumerate() {
        let t = i as f64 / sample_rate_hz;
        *sample = components
            .iter()
            .map(|(f, a)| a * (2.0 * std::f64::consts::PI * f * t).cos())
            .sum();
    }
    out
}

/// Equal-energy pair, first voice: harmonic complex on 110 Hz.
pub fn equal_energy_signal_a(
    sample_rate_hz: f64,
    duration_s: f64,
    seed: u64,
) -> Result<StimulusSignal> {
    let n = check_rate_duration(sample_rate_hz, duration_s)?;
    let components = vec![(110.0, 0.5), (165.0, 0.3), (220.0, 0.2)];
    let samples = tone_bank(n, sample_rate_hz, &components);
    let (frequencies_hz, amplitudes): (Vec<f64>, Vec<f64>) = components.iter().cloned().unzip();
    Ok(finish(
        StimulusKind::EqualEnergyA,
        sample_rate_hz,
        seed,
        frequencies_hz,
        amplitudes,
        duration_s,
        samples,
    ))
}

/// Equal-energy pair, second voice: different spectrum, matched energy.
///
/// Amplitudes are scaled at construction so the analytic tone energy
/// (integer-cycle components) equals voice A's; the test verifies the match
/// from the samples with plain arithmetic.
pub fn equal_energy_signal_b(
    sample_rate_hz: f64,
    duration_s: f64,
    seed: u64,
) -> Result<StimulusSignal> {
    let n = check_rate_duration(sample_rate_hz, duration_s)?;
    let base = [(90.0, 0.5), (180.0, 0.3), (270.0, 0.2)];
    let energy_a_like: f64 = [(0.5_f64, 110.0), (0.3, 165.0), (0.2, 220.0)]
        .iter()
        .map(|(a, _)| a * a)
        .sum();
    let energy_base: f64 = base.iter().map(|(_, a)| a * a).sum();
    let scale = (energy_a_like / energy_base).sqrt();
    let components: Vec<(f64, f64)> = base.iter().map(|(f, a)| (*f, a * scale)).collect();
    let samples = tone_bank(n, sample_rate_hz, &components);
    let (frequencies_hz, amplitudes): (Vec<f64>, Vec<f64>) = components.iter().cloned().unzip();
    Ok(finish(
        StimulusKind::EqualEnergyB,
        sample_rate_hz,
        seed,
        frequencies_hz,
        amplitudes,
        duration_s,
        samples,
    ))
}

/// Spectral tilt: harmonic complex on 55 Hz with 1/n amplitude falloff.
pub fn tilt_signal(sample_rate_hz: f64, duration_s: f64, seed: u64) -> Result<StimulusSignal> {
    let n = check_rate_duration(sample_rate_hz, duration_s)?;
    let components: Vec<(f64, f64)> = (1..=8).map(|k| (55.0 * k as f64, 0.8 / k as f64)).collect();
    let (frequencies_hz, mut amplitudes): (Vec<f64>, Vec<f64>) = components.iter().cloned().unzip();
    let samples = normalize_peak(
        tone_bank(n, sample_rate_hz, &components),
        &mut amplitudes,
        0.9,
    );
    Ok(finish(
        StimulusKind::Tilt,
        sample_rate_hz,
        seed,
        frequencies_hz,
        amplitudes,
        duration_s,
        samples,
    ))
}

/// Sustained resonance excitation: long 75 Hz tone with raised-cosine edges.
pub fn resonance_signal(sample_rate_hz: f64, duration_s: f64, seed: u64) -> Result<StimulusSignal> {
    let n = check_rate_duration(sample_rate_hz, duration_s)?;
    let edge = (0.1 * sample_rate_hz) as usize;
    let mut samples = tone_bank(n, sample_rate_hz, &[(75.0, 0.8)]);
    for (i, sample) in samples.iter_mut().enumerate() {
        let gain = if i < edge {
            0.5 - 0.5 * (std::f64::consts::PI * i as f64 / edge as f64).cos()
        } else if i >= n - edge {
            0.5 - 0.5 * (std::f64::consts::PI * (n - 1 - i) as f64 / edge as f64).cos()
        } else {
            1.0
        };
        *sample *= gain;
    }
    Ok(finish(
        StimulusKind::Resonance,
        sample_rate_hz,
        seed,
        vec![75.0],
        vec![0.8],
        duration_s,
        samples,
    ))
}

/// Transient: unit impulse plus a short decaying 2 kHz burst.
pub fn transient_signal(sample_rate_hz: f64, duration_s: f64, seed: u64) -> Result<StimulusSignal> {
    let n = check_rate_duration(sample_rate_hz, duration_s)?;
    let mut samples = vec![0.0; n];
    let at = (0.01 * sample_rate_hz) as usize;
    if at >= n {
        return Err(AutoeqError::InvalidConfiguration {
            message: "transient duration must exceed the 10 ms impulse offset".to_string(),
        });
    }
    samples[at] = 1.0;
    let burst_end = n.min(at + (0.05 * sample_rate_hz) as usize);
    for (i, sample) in samples.iter_mut().enumerate().take(burst_end).skip(at + 1) {
        let t = (i - at) as f64 / sample_rate_hz;
        *sample = (-t / 0.008).exp() * (2.0 * std::f64::consts::PI * 2000.0 * t).sin() * 0.5;
    }
    Ok(finish(
        StimulusKind::Transient,
        sample_rate_hz,
        seed,
        Vec::new(),
        Vec::new(),
        duration_s,
        samples,
    ))
}

/// AM-rate sweep: 440 Hz carrier, modulation rate sweeping 2 → 20 Hz.
pub fn am_sweep_signal(sample_rate_hz: f64, duration_s: f64, seed: u64) -> Result<StimulusSignal> {
    let n = check_rate_duration(sample_rate_hz, duration_s)?;
    let mut samples = Vec::with_capacity(n);
    for i in 0..n {
        let t = i as f64 / sample_rate_hz;
        let progress = t / duration_s;
        let rate = 2.0 + 18.0 * progress;
        // Instantaneous AM phase for a linear rate sweep.
        let am_phase = 2.0 * std::f64::consts::PI * (2.0 * t + 9.0 * t * t / duration_s);
        let _ = rate;
        let modulator = 1.0 - 0.8 * (0.5 - 0.5 * am_phase.cos());
        samples.push(0.7 * modulator * (2.0 * std::f64::consts::PI * 440.0 * t).cos());
    }
    Ok(finish(
        StimulusKind::AmSweep,
        sample_rate_hz,
        seed,
        vec![440.0],
        vec![0.7],
        duration_s,
        samples,
    ))
}

/// Beating tones: 440 Hz + 443 Hz at equal amplitude (3 Hz beat).
pub fn beats_signal(sample_rate_hz: f64, duration_s: f64, seed: u64) -> Result<StimulusSignal> {
    let n = check_rate_duration(sample_rate_hz, duration_s)?;
    let samples = tone_bank(n, sample_rate_hz, &[(440.0, 0.5), (443.0, 0.5)]);
    Ok(finish(
        StimulusKind::Beats,
        sample_rate_hz,
        seed,
        vec![440.0, 443.0],
        vec![0.5, 0.5],
        duration_s,
        samples,
    ))
}

/// Missing fundamental: 200–500 Hz harmonics of f0 = 100 Hz, f0 absent.
pub fn missing_fundamental_signal(
    sample_rate_hz: f64,
    duration_s: f64,
    seed: u64,
) -> Result<StimulusSignal> {
    let n = check_rate_duration(sample_rate_hz, duration_s)?;
    let components: Vec<(f64, f64)> = vec![(200.0, 0.5), (300.0, 0.4), (400.0, 0.3), (500.0, 0.2)];
    let (frequencies_hz, mut amplitudes): (Vec<f64>, Vec<f64>) = components.iter().cloned().unzip();
    let samples = normalize_peak(
        tone_bank(n, sample_rate_hz, &components),
        &mut amplitudes,
        0.9,
    );
    Ok(finish(
        StimulusKind::MissingFundamental,
        sample_rate_hz,
        seed,
        frequencies_hz,
        amplitudes,
        duration_s,
        samples,
    ))
}

/// Constant output-loss pair (F11): raw loss plus a display-normalized view.
///
/// `raw_after` is the physical truth (a flat `loss_db` below `raw_before`);
/// `display_after` is renormalized to the original peak, hiding the loss from
/// any scorer that only looks at displayed data.
#[derive(Debug, Clone)]
pub struct OutputLossFixture {
    pub loss_db: f64,
    pub raw_before: Curve,
    pub raw_after: Curve,
    pub display_after: Curve,
}

/// Build the F11 plant with a 6 dB constant output loss.
pub fn output_loss_pair() -> OutputLossFixture {
    let loss_db = 6.0;
    let raw_before = crate::generate_harman_tilt_curve(20.0, 20_000.0, 400);
    let raw_after = Curve {
        freq: raw_before.freq.clone(),
        spl: &raw_before.spl - loss_db,
        ..Default::default()
    };
    let peak_before = raw_before
        .spl
        .iter()
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max);
    let peak_after = raw_after
        .spl
        .iter()
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max);
    let display_after = Curve {
        freq: raw_after.freq.clone(),
        spl: &raw_after.spl + (peak_before - peak_after),
        ..Default::default()
    };
    OutputLossFixture {
        loss_db,
        raw_before,
        raw_after,
        display_after,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const RATE: f64 = STIMULUS_SAMPLE_RATE_HZ;

    #[test]
    fn synthetic_stimulus_signals_valid() {
        // Lengths, rates, amplitudes and seeds are exactly as declared.
        let cases: Vec<StimulusSignal> = vec![
            equal_energy_signal_a(RATE, 1.0, 1).expect("energy A"),
            equal_energy_signal_b(RATE, 1.0, 1).expect("energy B"),
            tilt_signal(RATE, 1.0, 2).expect("tilt"),
            resonance_signal(RATE, 2.0, 3).expect("resonance"),
            transient_signal(RATE, 0.25, 4).expect("transient"),
            am_sweep_signal(RATE, 2.0, 5).expect("AM sweep"),
            beats_signal(RATE, 2.0, 6).expect("beats"),
            missing_fundamental_signal(RATE, 1.0, 7).expect("missing fundamental"),
        ];
        let expected_len = [48000, 48000, 48000, 96000, 12000, 96000, 96000, 48000];
        for (signal, len) in cases.iter().zip(expected_len) {
            assert_eq!(
                signal.samples.len(),
                len,
                "signal {} must fill rate*duration samples",
                signal.id
            );
            assert_eq!(signal.sample_rate_hz, RATE);
            assert!(
                signal.samples.iter().all(|v| v.is_finite()),
                "signal {} must be finite",
                signal.id
            );
            assert!(
                signal.peak <= 1.0 + 1e-12,
                "signal {} must respect the declared amplitude bound, got {}",
                signal.id,
                signal.peak
            );
            assert!(
                signal.peak > 0.1,
                "signal {} must be non-trivial",
                signal.id
            );
            // Stored energy matches a plain recomputation from the samples.
            let recomputed: f64 = signal.samples.iter().map(|v| v * v).sum();
            assert!(
                (recomputed - signal.energy).abs() < 1e-6 * signal.energy.max(1.0),
                "stored energy of {} must match its samples",
                signal.id
            );
        }
        // Determinism: same seed, same bytes.
        let again = beats_signal(RATE, 2.0, 6).expect("beats");
        assert_eq!(again.samples, cases[6].samples);
        // Distinct stable identifiers per control.
        let mut ids: Vec<&str> = cases.iter().map(|s| s.id.as_str()).collect();
        ids.sort_unstable();
        ids.dedup();
        assert_eq!(ids.len(), 8, "every control needs a distinct stable id");
        // Invalid construction is rejected, not panicked or zero-filled.
        assert!(beats_signal(0.0, 1.0, 1).is_err());
        assert!(beats_signal(RATE, -1.0, 1).is_err());
        assert!(transient_signal(RATE, 0.001, 1).is_err());
    }

    #[test]
    fn synthetic_equal_energy_no_loudness_claim() {
        // Equal total energy, different spectra — and no perceptual verdict.
        let a = equal_energy_signal_a(RATE, 1.0, 1).expect("energy A");
        let b = equal_energy_signal_b(RATE, 1.0, 1).expect("energy B");
        let rel = ((a.energy - b.energy) / a.energy).abs();
        assert!(
            rel < 1e-9,
            "equal-energy controls must match within 1e-9 relative, got {rel:.3e}"
        );
        assert_ne!(
            a.frequencies_hz, b.frequencies_hz,
            "the pair must differ in spectral envelope"
        );
        assert!(
            a.frequencies_hz
                .iter()
                .all(|f| !b.frequencies_hz.contains(f)),
            "the pair must use disjoint component sets"
        );
    }

    #[test]
    fn synthetic_beats_and_missing_fundamental_behave() {
        // Beats: constructive interference reaches ~the amplitude sum, and the
        // envelope dips near silence each 1/3 s.
        let beats = beats_signal(RATE, 2.0, 6).expect("beats");
        assert!(
            (beats.peak - 1.0).abs() < 0.05,
            "equal 0.5+0.5 tones must peak near 1.0, got {:.3}",
            beats.peak
        );
        // The 3 Hz beat places a null every 1/3 s; each beat period must
        // contain a sample dipping near silence.
        let window = (RATE / 3.0) as usize;
        let mut worst_dip = 0.0_f64;
        for chunk in beats.samples.chunks(window) {
            let m = chunk.iter().map(|v| v.abs()).fold(f64::INFINITY, f64::min);
            worst_dip = worst_dip.max(m);
        }
        assert!(
            worst_dip < 0.05,
            "beat envelope must dip toward cancellation every period, worst dip {worst_dip:.3}"
        );
        // Missing fundamental: orthogonal to f0 over the 1 s window, strong
        // on the first present harmonic. (Physical correlation, not pitch.)
        let missing = missing_fundamental_signal(RATE, 1.0, 7).expect("missing f0");
        assert!(!missing.frequencies_hz.contains(&100.0));
        let dot = |freq: f64| {
            missing
                .samples
                .iter()
                .enumerate()
                .map(|(i, v)| v * (2.0 * std::f64::consts::PI * freq * i as f64 / RATE).cos())
                .sum::<f64>()
                / missing.samples.len() as f64
        };
        assert!(
            dot(100.0).abs() < 1e-3,
            "no energy may sit at the absent 100 Hz fundamental"
        );
        assert!(
            dot(200.0) > 0.1,
            "the present 200 Hz harmonic must correlate strongly"
        );
    }

    #[test]
    fn synthetic_output_loss_survives_normalization() {
        // F11: the raw loss is intact even though the display view hides it.
        let fix = output_loss_pair();
        assert!((fix.loss_db - 6.0).abs() < 1e-12);
        let raw_gap: Vec<f64> = fix
            .raw_before
            .spl
            .iter()
            .zip(fix.raw_after.spl.iter())
            .map(|(b, a)| b - a)
            .collect();
        assert!(
            raw_gap.iter().all(|g| (g - 6.0).abs() < 1e-9),
            "raw curves must differ by exactly the planted loss on every bin"
        );
        let peak = |c: &Curve| c.spl.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        assert!(
            (peak(&fix.display_after) - peak(&fix.raw_before)).abs() < 1e-9,
            "display normalization must hide the loss at matched peaks"
        );
        assert!(
            (peak(&fix.raw_before) - peak(&fix.raw_after) - 6.0).abs() < 1e-9,
            "raw output loss must survive as a 6 dB peak drop"
        );
    }
}
