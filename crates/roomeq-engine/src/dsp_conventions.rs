//! DSP convention and numerical-discipline audit.
//!
//! Declares the transform normalizations, window gains, convolution padding,
//! group-delay sign, FIR tap/latency math, delay preservation, SOS
//! preference, and pole/ROC discipline used across `roomeq-engine`, with one
//! executable check per convention. Kautz realization, multirate delay
//! handling, and headroom accounting each enter through a reproduction-first
//! test against the production code path; no threshold, baseline, or
//! allowance is relaxed here.

// Rust guideline compliant 2026-02-21

use roomeq_model::PluginConfigWrapper;

/// FFT convention: forward transform unnormalized, inverse scales by `1/N`.
///
/// A forward-inverse roundtrip must recover the input samples; this is the
/// executable statement of the normalization used by every in-repo FFT stage
/// (for example the Hilbert envelope in `roomeq-quality`).
pub fn fft_roundtrip(samples: &[f64]) -> Vec<f64> {
    use rustfft::{FftPlanner, num_complex::Complex};
    if samples.is_empty() {
        return Vec::new();
    }
    let n = samples.len();
    let mut planner = FftPlanner::<f64>::new();
    let forward = planner.plan_fft_forward(n);
    let inverse = planner.plan_fft_inverse(n);
    let mut spectrum: Vec<Complex<f64>> = samples.iter().map(|s| Complex::new(*s, 0.0)).collect();
    forward.process(&mut spectrum);
    inverse.process(&mut spectrum);
    spectrum.iter().map(|z| z.re / n as f64).collect()
}

/// Rectangular and Hann window gains from definition.
///
/// Returns `(coherent_gain, noise_bandwidth)` where coherent gain is
/// `sum(w)/N` and noise bandwidth is `N * sum(w^2)/sum(w)^2` in bins.
/// Rectangular: `(1.0, 1.0)`; Hann: `(0.5, 1.5)`.
pub fn window_gains(window: &[f64]) -> (f64, f64) {
    let n = window.len() as f64;
    if n == 0.0 {
        return (f64::NAN, f64::NAN);
    }
    let sum: f64 = window.iter().sum();
    let sum_sq: f64 = window.iter().map(|w| w * w).sum();
    if sum == 0.0 {
        return (f64::NAN, f64::NAN);
    }
    (sum / n, n * sum_sq / (sum * sum))
}

/// Hann window of length `n` (periodic form).
pub fn hann_window(n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| 0.5 * (1.0 - (2.0 * std::f64::consts::PI * i as f64 / n as f64).cos()))
        .collect()
}

/// Linear-phase FIR latency in ms: `(taps - 1) / (2 * Fs)`.
///
/// A symmetric FIR of `taps` coefficients delays every frequency by
/// `(taps - 1) / 2` samples; even tap counts therefore carry a half-sample
/// (non-integer) group delay, which resampling stages must preserve as
/// fractional time rather than rounding to a sample.
pub fn linear_phase_fir_latency_ms(taps: usize, sample_rate_hz: f64) -> f64 {
    if taps == 0 || !sample_rate_hz.is_finite() || sample_rate_hz <= 0.0 {
        return f64::NAN;
    }
    (taps.saturating_sub(1)) as f64 * 500.0 / sample_rate_hz
}

/// Pole radius of a normalized biquad denominator `1 + a1 z^-1 + a2 z^-2`.
///
/// Poles are the roots of `z^2 + a1 z + a2 = 0`. A causal realization is
/// BIBO-stable if and only if every pole lies strictly inside the unit
/// circle, so stability needs this radius *and* the causality statement, not
/// the radius alone.
pub fn biquad_pole_radius(a1: f64, a2: f64) -> f64 {
    let discriminant = a1 * a1 - 4.0 * a2;
    if discriminant >= 0.0 {
        let root = discriminant.sqrt();
        ((-a1 + root) / 2.0).abs().max(((-a1 - root) / 2.0).abs())
    } else {
        // Complex-conjugate pair with magnitude sqrt(a2).
        a2.max(0.0).sqrt()
    }
}

/// Whether a causal biquad with denominator `1 + a1 z^-1 + a2 z^-2` is stable.
pub fn is_causal_biquad_stable(a1: f64, a2: f64) -> bool {
    a1.is_finite() && a2.is_finite() && biquad_pole_radius(a1, a2) < 1.0
}

/// Build a one-filter Kautz EQ plugin matching the serialized schema.
pub fn kautz_eq_plugin(sections: &[(f64, f64, f64)]) -> PluginConfigWrapper {
    let kautz_sections: Vec<serde_json::Value> = sections
        .iter()
        .map(|(pole_freq, q, gain)| {
            serde_json::json!({ "pole_freq": pole_freq, "q": q, "gain": gain })
        })
        .collect();
    PluginConfigWrapper {
        plugin_type: "eq".to_string(),
        parameters: serde_json::json!({
            "filters": [{
                "topology": "kautz_filter",
                "filter_type": "peak",
                "freq": 100.0,
                "q": 1.0,
                "db_gain": 0.0,
                "kautz_sections": kautz_sections,
            }],
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dsp_realization::{NoConvolutionIr, RealizedDsp};
    use crate::output::{create_delay_plugin, create_gain_plugin};
    use crate::quality::group_delay_ms;
    use num_complex::Complex64;
    use roomeq_model::ChannelDspChain;

    fn chain(plugins: Vec<PluginConfigWrapper>) -> ChannelDspChain {
        ChannelDspChain {
            physical_correction_target: None,
            channel: "L".to_string(),
            plugins,
            drivers: None,
            initial_curve: None,
            final_curve: None,
            eq_response: None,
            pre_ir: None,
            post_ir: None,
            fir_temporal_masking: None,
            direct_early_late_correction: None,
            joint_sub: None,
            early_reflections: None,
            t60_octaves: None,
            waterfall: None,
            resonance_decays: None,
            wavelet: None,
            early_late_curves: None,
            target_curve: None,
        }
    }

    fn realized_response(
        chain: &ChannelDspChain,
        sample_rate: f64,
        frequency_hz: f64,
    ) -> Complex64 {
        let mut provider = NoConvolutionIr;
        RealizedDsp::new(chain, sample_rate, &mut provider)
            .expect("test chain realizes")
            .response_at(frequency_hz)
            .expect("test frequency evaluates")
    }

    /// Reproduction: Kautz realization evaluates deterministically and a
    /// modal correction with nonzero gains actually changes the response.
    #[test]
    fn kautz_realization_is_deterministic_and_active() {
        let chain = chain(vec![kautz_eq_plugin(&[(60.0, 4.0, 6.0)])]);
        let first = realized_response(&chain, 48_000.0, 60.0);
        let second = realized_response(&chain, 48_000.0, 60.0);
        assert!(first.is_finite());
        assert!((first - second).norm() < 1e-15);
        let flat = realized_response(&chain, 48_000.0, 5_000.0);
        assert!(
            (first.norm() - flat.norm()).abs() > 1e-6,
            "modal section must shape its pole region"
        );
    }

    /// Zero basis coefficients preserve the playback consumer's unity dry path.
    #[test]
    fn roadmap_correction_kautz_zero_weights_preserve_dry_playback() {
        for rate in [44_100.0, 48_000.0, 96_000.0] {
            let chain = chain(vec![kautz_eq_plugin(&[
                (60.0, 4.0, 0.0),
                (103.0, 8.0, 0.0),
            ])]);
            for frequency in [20.0, 60.0, 103.0, 1000.0, 0.49 * rate] {
                assert_eq!(
                    realized_response(&chain, rate, frequency),
                    Complex64::new(1.0, 0.0)
                );
            }
        }
    }

    #[test]
    fn roadmap_correction_kautz_realization_matches_dry_plus_streamed_bank() {
        use math_audio_iir_fir::KautzFilter;
        // The playback consumer's KautzRuntime::process is x + bank.process(x).
        // Compare the serialized evaluator with that actual bank recurrence,
        // not a second call to its complex_response formula. This is not a
        // complete plugin-host execution or evidence that gain fitting works.
        let sections = [(60.0, 4.0, -0.5), (103.0, 8.0, 0.25)];
        for rate in [44_100.0, 48_000.0, 96_000.0] {
            let mut bank = KautzFilter::from_room_modes(&[(60.0, 4.0), (103.0, 8.0)], rate);
            for (section, (_, _, gain)) in bank.sections.iter_mut().zip(sections) {
                section.gain = gain;
            }
            // More than fifty decay time constants at the slowest fixture pole,
            // keeping finite-tail error far below the complex comparison budget.
            let impulse: Vec<_> = (0..131_072)
                .map(|index| {
                    let input = if index == 0 { 1.0 } else { 0.0 };
                    input + bank.process(input)
                })
                .collect();
            let serialized = serde_json::to_vec(&chain(vec![kautz_eq_plugin(&sections)])).unwrap();
            let chain: ChannelDspChain = serde_json::from_slice(&serialized).unwrap();
            for frequency in [20.0, 60.0, 80.0, 103.0, 200.0, 1000.0, 5000.0] {
                let step = Complex64::from_polar(1.0, -std::f64::consts::TAU * frequency / rate);
                let mut phase = Complex64::new(1.0, 0.0);
                let mut measured = Complex64::new(0.0, 0.0);
                for sample in &impulse {
                    measured += phase * sample;
                    phase *= step;
                }
                let realized = realized_response(&chain, rate, frequency);
                assert!(
                    (realized - measured).norm() < 1e-8,
                    "{rate} Hz / {frequency} Hz: {realized} vs {measured}"
                );
            }
        }
    }

    #[test]
    fn roadmap_correction_kautz_legacy_single_section_matches_playback() {
        let reference = chain(vec![kautz_eq_plugin(&[(60.0, 4.0, -0.5)])]);
        for explicit_empty in [false, true] {
            let mut filter = serde_json::json!({"topology": "kautz_filter", "filter_type": "peak",
                "freq": 60.0, "q": 4.0, "db_gain": -0.5});
            if explicit_empty {
                filter["kautz_sections"] = serde_json::json!([]);
            }
            let legacy = chain(vec![PluginConfigWrapper {
                plugin_type: "eq".into(),
                parameters: serde_json::json!({"filters": [filter]}),
            }]);
            for frequency in [20.0, 60.0, 5000.0] {
                assert_eq!(
                    realized_response(&legacy, 48_000.0, frequency),
                    realized_response(&reference, 48_000.0, frequency)
                );
            }
        }
    }

    #[test]
    fn roadmap_correction_kautz_aliases_defaults_and_malformed_values() {
        let original = kautz_eq_plugin(&[(60.0, 4.0, -0.5)]);
        let expected = realized_response(&chain(vec![original.clone()]), 48_000.0, 80.0);
        for sections_key in ["kautz_sections", "sections"] {
            for frequency_key in ["pole_freq", "freq", "frequency", "pole_freq_hz"] {
                let mut plugin = original.clone();
                let filter = &mut plugin.parameters["filters"][0];
                let mut sections = filter
                    .as_object_mut()
                    .unwrap()
                    .remove("kautz_sections")
                    .unwrap();
                let frequency = sections[0]
                    .as_object_mut()
                    .unwrap()
                    .remove("pole_freq")
                    .unwrap();
                sections[0][frequency_key] = frequency;
                filter[sections_key] = sections;
                assert_eq!(
                    realized_response(&chain(vec![plugin.clone()]), 48_000.0, 80.0),
                    expected
                );
                plugin.parameters["filters"][0][sections_key][0]
                    .as_object_mut()
                    .unwrap()
                    .remove("gain");
                assert_eq!(
                    realized_response(&chain(vec![plugin]), 48_000.0, 80.0),
                    Complex64::new(1.0, 0.0)
                );
            }
        }
        for invalid in 0..4 {
            let mut plugin = original.clone();
            let filter = &mut plugin.parameters["filters"][0];
            match invalid {
                0 => filter["kautz_sections"] = serde_json::Value::Null,
                1 => filter["sections"] = filter["kautz_sections"].clone(),
                2 => filter["kautz_sections"][0]["freq"] = serde_json::json!(60.0),
                _ => filter["kautz_sections"][0]["gain"] = serde_json::Value::Null,
            }
            let chain = chain(vec![plugin]);
            let mut provider = NoConvolutionIr;
            assert!(
                RealizedDsp::new(&chain, 48_000.0, &mut provider)
                    .unwrap()
                    .response_at(80.0)
                    .is_err(),
                "case {invalid}"
            );
        }
    }

    #[test]
    fn kautz_pole_above_nyquist_rejected() {
        let chain = chain(vec![kautz_eq_plugin(&[(30_000.0, 4.0, 6.0)])]);
        let mut provider = NoConvolutionIr;
        assert!(
            RealizedDsp::new(&chain, 48_000.0, &mut provider)
                .unwrap()
                .response_at(100.0)
                .is_err()
        );
    }

    /// Reproduction: a serialized Kautz filter without sections is malformed,
    /// never silently unity.
    #[test]
    fn kautz_missing_sections_is_malformed() {
        let plugin = PluginConfigWrapper {
            plugin_type: "eq".to_string(),
            parameters: serde_json::json!({
                "filters": [{ "topology": "kautz_filter", "kautz_sections": [] }],
            }),
        };
        let chain = chain(vec![plugin]);
        let mut provider = NoConvolutionIr;
        let response = RealizedDsp::new(&chain, 48_000.0, &mut provider)
            .unwrap()
            .response_at(100.0);
        assert!(
            response.is_err(),
            "missing sections and fallback parameters are malformed"
        );
    }

    /// Delay is preserved in seconds across sample rates: the same delay
    /// chain realizes identically at 44.1, 48, and 96 kHz.
    #[test]
    fn delay_preserved_across_sample_rates() {
        let chain = chain(vec![create_gain_plugin(-3.0), create_delay_plugin(1.25)]);
        let reference = realized_response(&chain, 48_000.0, 1_000.0);
        for sample_rate in [44_100.0, 48_000.0, 96_000.0] {
            let actual = realized_response(&chain, sample_rate, 1_000.0);
            assert!(
                (actual - reference).norm() < 1e-12,
                "delay response changed at {sample_rate} Hz"
            );
        }
    }

    /// Group-delay sign: a pure delay has positive group delay equal to the
    /// delay (`GD = -dphi/domega`, unwrapped phase).
    #[test]
    fn group_delay_sign_matches_causal_delay() {
        let chain = chain(vec![create_delay_plugin(0.5)]);
        let freqs: Vec<f64> = (1..64).map(|i| i as f64 * 100.0).collect();
        let transfer: Vec<Complex64> = freqs
            .iter()
            .map(|f| realized_response(&chain, 48_000.0, *f))
            .collect();
        let delays = group_delay_ms(&freqs, &transfer);
        assert_eq!(delays.len(), freqs.len() - 1);
        for delay in &delays {
            assert!((delay - 0.5).abs() < 1e-6, "group delay was {delay} ms");
        }
    }

    /// Declared FFT normalization: a forward-inverse roundtrip recovers input.
    #[test]
    fn fft_normalization_roundtrip() {
        let samples = vec![0.0, 1.0, 0.5, -0.25, 0.125, -0.0625, 0.75, 0.2];
        let recovered = fft_roundtrip(&samples);
        assert_eq!(recovered.len(), samples.len());
        for (a, b) in recovered.iter().zip(samples.iter()) {
            assert!((a - b).abs() < 1e-12);
        }
        assert!(fft_roundtrip(&[]).is_empty());
    }

    /// Window gains from definition: rectangular (1, 1), Hann (0.5, 1.5).
    #[test]
    fn window_gains_match_definitions() {
        let rectangular = vec![1.0; 64];
        let (coherent, noise) = window_gains(&rectangular);
        assert!((coherent - 1.0).abs() < 1e-15);
        assert!((noise - 1.0).abs() < 1e-15);
        let (coherent, noise) = window_gains(&hann_window(1024));
        assert!((coherent - 0.5).abs() < 1e-3);
        assert!((noise - 1.5).abs() < 1e-2);
    }

    /// One-sided amplitude scaling: a full-scale tone at a bin center reads
    /// its amplitude back after `2|X[k]|/N` (rectangular window, coherent gain
    /// 1, non-DC bin).
    #[test]
    fn one_sided_scaling_recovers_tone_amplitude() {
        use rustfft::{FftPlanner, num_complex::Complex};
        let n = 256;
        let amplitude = 0.7;
        let bin = 32;
        let tone: Vec<f64> = (0..n)
            .map(|i| {
                amplitude * (2.0 * std::f64::consts::PI * bin as f64 * i as f64 / n as f64).cos()
            })
            .collect();
        let mut planner = FftPlanner::<f64>::new();
        let forward = planner.plan_fft_forward(n);
        let mut spectrum: Vec<Complex<f64>> = tone.iter().map(|s| Complex::new(*s, 0.0)).collect();
        forward.process(&mut spectrum);
        let recovered = 2.0 * spectrum[bin].norm() / n as f64;
        assert!((recovered - amplitude).abs() < 1e-9);
    }

    /// Linear convolution needs `Lx + Lh - 1` padding: padded FFT
    /// convolution matches direct convolution, unpadded circular does not.
    #[test]
    fn convolution_padding_linear_not_circular() {
        use rustfft::{FftPlanner, num_complex::Complex};
        let x = [1.0, 2.0, 3.0, 4.0];
        let h = [0.5, -0.25, 0.125];
        let mut linear = vec![0.0; x.len() + h.len() - 1];
        for (i, a) in x.iter().enumerate() {
            for (j, b) in h.iter().enumerate() {
                linear[i + j] += a * b;
            }
        }
        let convolve_fft = |n: usize| -> Vec<f64> {
            let mut planner = FftPlanner::<f64>::new();
            let forward = planner.plan_fft_forward(n);
            let inverse = planner.plan_fft_inverse(n);
            let mut xs: Vec<Complex<f64>> = x
                .iter()
                .chain(std::iter::repeat(&0.0))
                .take(n)
                .map(|s| Complex::new(*s, 0.0))
                .collect();
            let mut hs: Vec<Complex<f64>> = h
                .iter()
                .chain(std::iter::repeat(&0.0))
                .take(n)
                .map(|s| Complex::new(*s, 0.0))
                .collect();
            forward.process(&mut xs);
            forward.process(&mut hs);
            for (a, b) in xs.iter_mut().zip(hs.iter()) {
                *a *= *b;
            }
            inverse.process(&mut xs);
            xs.iter().map(|z| z.re / n as f64).collect()
        };
        let padded = convolve_fft(linear.len());
        for (a, b) in padded.iter().zip(linear.iter()) {
            assert!((a - b).abs() < 1e-9);
        }
        // Circular convolution without padding aliases the tail onto the
        // head; it must differ here so the padding requirement is load-bearing.
        let circular = convolve_fft(x.len());
        let aliased: Vec<f64> = linear
            .iter()
            .enumerate()
            .take(x.len())
            .map(|(i, _)| linear[i] + linear.get(i + x.len()).copied().unwrap_or(0.0))
            .collect();
        for (a, b) in circular.iter().zip(aliased.iter()) {
            assert!((a - b).abs() < 1e-9);
        }
        assert!((circular[0] - linear[0]).abs() > 1e-6);
    }

    /// FIR tap/latency math: `(N-1)/(2Fs)` in ms, including the half-sample
    /// non-integer delay of even tap counts.
    #[test]
    fn fir_tap_latency_math() {
        assert!((linear_phase_fir_latency_ms(65, 48_000.0) - 32.0 / 48.0).abs() < 1e-12);
        assert!((linear_phase_fir_latency_ms(64, 48_000.0) - 31.5 / 48.0).abs() < 1e-12);
        assert!(linear_phase_fir_latency_ms(0, 48_000.0).is_nan());
        assert!(linear_phase_fir_latency_ms(65, 0.0).is_nan());
    }

    /// Symmetric FIR group delay equals `(N-1)/2` samples at low frequency.
    #[test]
    fn symmetric_fir_group_delay_matches_tap_math() {
        let taps = vec![0.25, 0.5, 1.0, 0.5, 0.25];
        let freqs: Vec<f64> = (1..32).map(|i| i as f64 * 50.0).collect();
        let freq_array = ndarray::Array1::from_vec(freqs.clone());
        let transfer =
            autoeq_core::response::compute_fir_complex_response(&taps, &freq_array, 48_000.0);
        let delays = group_delay_ms(&freqs, &transfer);
        let expected_ms = 2.0 / 48_000.0 * 1000.0;
        for delay in delays.iter().take(8) {
            assert!(
                (delay - expected_ms).abs() < 1e-6,
                "FIR delay was {delay} ms"
            );
        }
    }

    /// SOS discipline: realized PEQ biquads are causal-stable, and cascaded
    /// sections compose exactly (no direct-form higher-order section).
    #[test]
    fn peq_biquads_causal_stable_and_cascade_exact() {
        use math_audio_iir_fir::{Biquad, BiquadFilterType};
        for filter_type in [
            BiquadFilterType::Peak,
            BiquadFilterType::Lowpass,
            BiquadFilterType::Highpass,
        ] {
            let filter = Biquad::new(filter_type, 1_000.0, 48_000.0, 1.0, 6.0);
            let (a1, a2, _, _, _) = filter.constants();
            assert!(
                is_causal_biquad_stable(a1, a2),
                "biquad pole radius was {}",
                biquad_pole_radius(a1, a2)
            );
        }
        // A known-unstable denominator is flagged, not trusted.
        assert!(!is_causal_biquad_stable(-0.1, -1.5));
        assert!(biquad_pole_radius(-0.1, -1.5) > 1.0);
        // Cascade composition: two identical peaks square one peak exactly.
        let single = Biquad::new(BiquadFilterType::Peak, 1_000.0, 48_000.0, 1.0, 6.0);
        let pair = vec![single.clone(), single.clone()];
        let freqs = ndarray::array![100.0, 1_000.0, 10_000.0];
        let one = autoeq_core::response::compute_peq_complex_response(
            std::slice::from_ref(&single),
            &freqs,
            48_000.0,
        );
        let two = autoeq_core::response::compute_peq_complex_response(&pair, &freqs, 48_000.0);
        for (a, b) in one.iter().zip(two.iter()) {
            assert!((a * a - b).norm() < 1e-12);
        }
    }
}
