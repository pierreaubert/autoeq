//! Finite-window analytic octave analysis of matched captures without circular wraparound.

use super::{EtcBand, ViewProvenance};
use rustfft::{FftPlanner, num_complex::Complex};
use serde::{Deserialize, Serialize};

/// Matched octave envelopes sharing baseline references, not left/right similarity.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MatchedEtcView {
    pub provenance: ViewProvenance,
    pub settings_hash: String,
    pub method: String,
    pub window_ms: [f64; 2],
    pub display_floor_db: f64,
    pub bands: Vec<EtcBand>,
    pub baseline_peak_amplitudes: Vec<f64>,
    /// Symmetric analysis support; initial samples depend on declared zero extension.
    pub filter_half_support_ms: Vec<f64>,
    pub scope: String,
}

pub(super) fn kernel(rate: f64, center: f64) -> (Vec<Complex<f64>>, usize) {
    let low = center / std::f64::consts::SQRT_2;
    let high = center * std::f64::consts::SQRT_2;
    let bandwidth = high - low;
    let midpoint = (high + low) / 2.0;
    // Four center-frequency periods on each side: a reproducible display
    // resolution, not an auditory integration window or room-decay limit.
    let half = (4.0 * rate / center).ceil() as usize;
    let taps = (0..=2 * half)
        .map(|index| {
            let n = index as f64 - half as f64;
            let ideal = if n == 0.0 {
                2.0 * bandwidth / rate
            } else {
                2.0 * (std::f64::consts::PI * bandwidth * n / rate).sin()
                    / (std::f64::consts::PI * n)
            };
            let hann = 0.5 + 0.5 * (std::f64::consts::PI * n / half as f64).cos();
            Complex::from_polar(ideal * hann, std::f64::consts::TAU * midpoint * n / rate)
        })
        .collect();
    (taps, half)
}

pub(super) fn envelope(
    samples: &[f64],
    taps: &[Complex<f64>],
    half: usize,
    count: usize,
) -> Vec<f64> {
    // At least Lx + Lh - 1 samples makes the FFT product linear convolution.
    let n = (samples.len() + taps.len() - 1).next_power_of_two();
    let mut planner = FftPlanner::<f64>::new();
    let forward = planner.plan_fft_forward(n);
    let inverse = planner.plan_fft_inverse(n);
    let mut input = vec![Complex::new(0.0, 0.0); n];
    let mut filter = input.clone();
    for (value, sample) in input.iter_mut().zip(samples) {
        value.re = *sample;
    }
    filter[..taps.len()].copy_from_slice(taps);
    forward.process(&mut input);
    forward.process(&mut filter);
    for (value, transfer) in input.iter_mut().zip(filter) {
        *value *= transfer;
    }
    inverse.process(&mut input);
    // Remove only the known analysis-filter indexing delay. Capture timing
    // remains untouched; the symmetric analysis still has temporal spreading.
    input[half..half + count]
        .iter()
        .map(|value| value.norm() / n as f64)
        .collect()
}

/// Compute matched 500–4000 Hz octave ETCs over the declared 0–40 ms display window.
///
/// Uses Hann-windowed positive-frequency sinc filters and zero-extended linear
/// convolution. Both traces share each band's baseline peak reference; values
/// below -160 dB are explicitly display-floored. This is not a decay or safety gate.
///
/// # Errors
/// Rejects unsupported nominal bands, insufficient filter lookahead, invalid
/// rates/samples, silent baseline bands, and nonfinite numerical output.
pub fn matched_capture_etc(
    pre: &[f64],
    post: &[f64],
    support: [f64; 2],
    provenance: ViewProvenance,
    settings_hash: String,
) -> Result<MatchedEtcView, String> {
    let rate = provenance.sample_rate_hz;
    if !rate.is_finite()
        || !(12_000.0..=384_000.0).contains(&rate)
        || pre.len() != post.len()
        || pre.is_empty()
        || pre.len() > 65_536
        || pre.iter().chain(post).any(|value| !value.is_finite())
        || support.iter().any(|value| !value.is_finite())
        || support[0] <= 0.0
        || support[0] > 500.0 / std::f64::consts::SQRT_2
        || support[1] < 4000.0 * std::f64::consts::SQRT_2
        || support[1] > rate / 2.0
    {
        return Err("ETC needs finite matched captures and nominal octave support from 500/sqrt(2) through 4000*sqrt(2) Hz below Nyquist".into());
    }
    let count = (0.040 * rate).floor() as usize + 1;
    let mut result = MatchedEtcView {
        provenance, settings_hash, method: "hann_analytic_octave_linear_v1".into(),
        window_ms: [0.0, 40.0], display_floor_db: -160.0,
        bands: Vec::new(), baseline_peak_amplitudes: Vec::new(), filter_half_support_ms: Vec::new(),
        scope: "diagnostic envelope; nominal octave bands have finite-filter transition skirts; symmetric analysis creates time spreading, not evidence of room pre-ringing; zero extension before record start affects the first half-support interval; common baseline-peak reference preserves relative level; no left/right similarity, passive-room decay, safety or audibility verdict".into(),
    };
    for center in [500.0, 1000.0, 2000.0, 4000.0] {
        let (taps, half) = kernel(rate, center);
        if pre.len() < count + half {
            return Err("ETC requires the complete 0–40 ms window plus analysis-filter lookahead; no tail was invented".into());
        }
        let before = envelope(pre, &taps, half, count);
        let after = envelope(post, &taps, half, count);
        if before.iter().chain(&after).any(|value| !value.is_finite()) {
            return Err("ETC envelope overflow".into());
        }
        let reference = before.iter().copied().fold(0.0_f64, f64::max);
        if reference <= 0.0 {
            return Err("ETC baseline band has no nonzero reference".into());
        }
        let db = |values: &[f64]| {
            values
                .iter()
                .map(|value| (20.0 * (value.log10() - reference.log10())).max(-160.0))
                .collect()
        };
        result.bands.push(EtcBand {
            center_hz: center,
            times_ms: (0..count)
                .map(|index| 1000.0 * index as f64 / rate)
                .collect(),
            pre_db: db(&before),
            post_db: db(&after),
        });
        result.baseline_peak_amplitudes.push(reference);
        result
            .filter_half_support_ms
            .push(1000.0 * half as f64 / rate);
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn captured_etc_analytic_filter_has_expected_tone_scale() {
        for center in [500.0, 1000.0, 2000.0, 4000.0] {
            let rate = 48_000.0;
            let (taps, half) = kernel(rate, center);
            let gain = |frequency: f64| {
                taps.iter()
                    .enumerate()
                    .map(|(index, tap)| {
                        tap * Complex::from_polar(
                            1.0,
                            -std::f64::consts::TAU * frequency * (index as f64 - half as f64)
                                / rate,
                        )
                    })
                    .sum::<Complex<f64>>()
                    .norm()
            };
            // Positive-frequency analytic gain is two; a real unit cosine
            // contributes half amplitude on each side and yields unit envelope.
            // Finite Hann truncation makes this approximate rather than ideal.
            assert!((gain(center) - 2.0).abs() < 0.02);
            assert!(gain(-center) < 0.002);
            assert!(gain(center * 2.0) < 0.002);
        }
    }

    #[test]
    fn captured_etc_fft_matches_direct_and_has_no_tail_wrap() {
        for rate in [44_100.0, 48_000.0, 96_000.0] {
            let mut samples = vec![0.0; (rate * 0.060) as usize];
            samples[10] = 0.25;
            samples[100] = -0.125;
            let (taps, half) = kernel(rate, 500.0);
            let fft = envelope(&samples, &taps, half, 200);
            for (time, actual) in fft.iter().enumerate() {
                let direct: Complex<f64> = samples
                    .iter()
                    .enumerate()
                    .filter_map(|(index, sample)| {
                        (time + half)
                            .checked_sub(index)
                            .and_then(|tap| taps.get(tap))
                            .map(|tap| tap * *sample)
                    })
                    .sum();
                assert!((actual - direct.norm()).abs() < 1e-12);
            }
            let last = samples.len() - 1;
            samples[last] = 100.0;
            let with_tail = envelope(&samples, &taps, half, 200);
            assert!(
                fft.iter()
                    .zip(with_tail)
                    .all(|(a, b)| (a - b).abs() < 1e-12)
            );
        }
    }

    #[test]
    fn captured_etc_shared_reference_preserves_gain_and_rejects_missing_support() {
        let provenance = ViewProvenance {
            measurement_ids: vec!["synthetic".into()],
            graph_identity: "graph".into(),
            sample_rate_hz: 48_000.0,
            calibration: "declared".into(),
            processing_chain: "test".into(),
        };
        let mut pre = vec![0.0; 4096];
        pre[480] = 0.25;
        let post: Vec<_> = pre.iter().map(|value| value * 2.0).collect();
        let view = matched_capture_etc(
            &pre,
            &post,
            [100.0, 6000.0],
            provenance.clone(),
            "settings".into(),
        )
        .unwrap();
        for band in &view.bands {
            assert!((band.pre_db[480]).abs() < 1e-10);
            assert!((band.post_db[480] - 20.0 * 2.0_f64.log10()).abs() < 1e-10);
            assert_eq!(band.times_ms.last(), Some(&40.0));
        }
        for (samples, support) in [(&pre[..64], [100.0, 6000.0]), (&pre[..], [100.0, 500.0])] {
            assert!(
                matched_capture_etc(
                    samples,
                    samples,
                    support,
                    provenance.clone(),
                    "settings".into()
                )
                .is_err()
            );
        }
        assert!(
            matched_capture_etc(
                &vec![0.0; 4096],
                &post,
                [100.0, 6000.0],
                provenance,
                "settings".into()
            )
            .is_err()
        );
    }
}
