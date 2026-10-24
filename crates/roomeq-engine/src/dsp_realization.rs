//! Canonical complex-response evaluation of serialized RoomEQ DSP chains.
//!
//! Optimizer reports, runtime safety checks, held-out QA, and export replay
//! must agree on the transfer represented by [`ChannelDspChain`]. File loading
//! remains owned by `roomeq-workflow` through [`ConvolutionIrProvider`].

use crate::error::{AutoeqError, Result};
use crate::response::{MIN_REALIZATION_RESPONSE_DB, apply_complex_response_with_min_db};
use ndarray::Array1;
use num_complex::Complex64;
use roomeq_model::ChannelDspChain;

pub mod driver;
mod fft_grid;

#[doc(inline)]
#[doc(inline)]
pub use autoeq_core::serialized_chain::{ConvolutionIrProvider, NoConvolutionIr};
use autoeq_core::serialized_chain::{checked_sample_rate, plugin_chain_response};

/// Canonical evaluator for one serialized channel chain.
pub struct RealizedDsp<'a, P: ConvolutionIrProvider> {
    chain: &'a ChannelDspChain,
    sample_rate: f64,
    sample_rate_u32: u32,
    convolution: &'a mut P,
}

impl<'a, P: ConvolutionIrProvider> RealizedDsp<'a, P> {
    pub fn new(
        chain: &'a ChannelDspChain,
        sample_rate: f64,
        convolution: &'a mut P,
    ) -> Result<Self> {
        let sample_rate_u32 = checked_sample_rate(sample_rate)?;
        Ok(Self {
            chain,
            sample_rate,
            sample_rate_u32,
            convolution,
        })
    }

    pub fn response_at(&mut self, frequency_hz: f64) -> Result<Complex64> {
        if !frequency_hz.is_finite() || frequency_hz < 0.0 {
            return Err(AutoeqError::InvalidConfiguration {
                message: format!("invalid DSP realization frequency {frequency_hz}"),
            });
        }
        channel_chain_response(
            self.chain,
            frequency_hz,
            self.sample_rate,
            self.sample_rate_u32,
            self.convolution,
        )
    }

    pub fn complex_response(&mut self, frequencies_hz: &Array1<f64>) -> Result<Vec<Complex64>> {
        frequencies_hz
            .iter()
            .map(|frequency| self.response_at(*frequency))
            .collect()
    }

    /// Evaluate a frequency grid, using FFTs for complete one-sided FFT grids.
    ///
    /// Added off-grid anchors and other grids retain scalar evaluation.
    /// FFT evaluation includes every FIR
    /// tap, including filters longer than the transform, without interpolation.
    /// The cache lives only for this call, so changed sidecars cannot be reused.
    ///
    /// # Errors
    /// Returns the same configuration and sidecar errors as scalar evaluation.
    pub fn response_grid(&mut self, frequencies_hz: &[f64]) -> Result<Vec<Complex64>> {
        if let Some(size) = fft_grid::fft_size(frequencies_hz, self.sample_rate) {
            let mut provider = fft_grid::FftGrid::new(&mut *self.convolution, size);
            let mut realized = RealizedDsp::new(self.chain, self.sample_rate, &mut provider)?;
            frequencies_hz
                .iter()
                .map(|frequency| realized.response_at(*frequency))
                .collect()
        } else {
            frequencies_hz
                .iter()
                .map(|frequency| self.response_at(*frequency))
                .collect()
        }
    }

    pub fn apply_to_curve(&mut self, curve: &crate::Curve) -> Result<crate::Curve> {
        curve.validate("channel DSP realization curve")?;
        let response = self.complex_response(&curve.freq)?;
        Ok(apply_complex_response_with_min_db(
            curve,
            &response,
            MIN_REALIZATION_RESPONSE_DB,
        ))
    }
}

fn channel_chain_response<P: ConvolutionIrProvider>(
    chain: &ChannelDspChain,
    frequency_hz: f64,
    sample_rate: f64,
    sample_rate_u32: u32,
    convolution: &mut P,
) -> Result<Complex64> {
    let branch_response = match chain.drivers.as_deref() {
        None | Some([]) => Complex64::new(1.0, 0.0),
        Some(drivers) => {
            let mut sum = Complex64::new(0.0, 0.0);
            for driver in drivers {
                sum += plugin_chain_response(
                    &driver.plugins,
                    frequency_hz,
                    sample_rate,
                    sample_rate_u32,
                    convolution,
                )?;
            }
            sum
        }
    };

    Ok(branch_response
        * plugin_chain_response(
            &chain.plugins,
            frequency_hz,
            sample_rate,
            sample_rate_u32,
            convolution,
        )?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::output::{
        create_band_merge_plugin, create_band_split_plugin, create_crossover_plugin,
        create_delay_plugin, create_gain_plugin, create_gain_plugin_with_invert,
        create_warped_eq_plugin,
    };
    use autoeq_core::fir_response::fir_response_at;
    use math_audio_iir_fir::{Biquad, BiquadFilterType};
    use roomeq_model::DriverDspChain;
    use roomeq_model::PluginConfigWrapper;
    use std::f64::consts::PI;

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
            speech_transmission: None,
            waterfall: None,
            resonance_decays: None,
            wavelet: None,
            early_late_curves: None,
            target_curve: None,
        }
    }

    #[test]
    fn realization_preserves_gain_delay_and_polarity() {
        let chain = chain(vec![
            create_gain_plugin_with_invert(-6.0, true),
            create_delay_plugin(0.25),
        ]);
        let mut provider = NoConvolutionIr;
        let actual = RealizedDsp::new(&chain, 48_000.0, &mut provider)
            .unwrap()
            .response_at(1_000.0)
            .unwrap();
        let expected = Complex64::new(-10.0_f64.powf(-6.0 / 20.0), 0.0)
            * Complex64::from_polar(1.0, -2.0 * PI * 1_000.0 * 0.25 / 1_000.0);
        assert!((actual - expected).norm() < 1e-12);
    }

    #[test]
    fn lr24_lowpass_rotates_phase_180_at_crossover() {
        // Two cascaded Butterworth-2 low-pass sections: at the corner each
        // contributes -90 degrees and 1/sqrt(2) magnitude, so the cascade
        // sits at |H| = 0.5 with a half-turn of phase. Routed-bass splice
        // accounting depends on this rotation being realized exactly.
        let chain = chain(vec![create_crossover_plugin("LR24", 80.0, "low")]);
        let mut provider = NoConvolutionIr;
        let resp = RealizedDsp::new(&chain, 48_000.0, &mut provider)
            .unwrap()
            .response_at(80.0)
            .unwrap();
        assert!((resp.norm() - 0.5).abs() < 0.01, "|H|={}", resp.norm());
        assert!(
            (resp.arg().to_degrees().abs() - 180.0).abs() < 1.0,
            "arg={}",
            resp.arg().to_degrees()
        );
        let curve = crate::Curve {
            freq: ndarray::array![40.0, 80.0],
            spl: ndarray::array![0.0, 0.0],
            phase: Some(ndarray::array![-181.0, -181.0]),
            ..Default::default()
        };
        let mut provider = NoConvolutionIr;
        let out = RealizedDsp::new(&chain, 48_000.0, &mut provider)
            .unwrap()
            .apply_to_curve(&curve)
            .unwrap();
        assert!(
            (out.spl[1] + 20.0 * 2.0_f64.log10()).abs() < 0.05,
            "spl={}",
            out.spl[1]
        );
        let phase = out.phase.as_ref().unwrap()[1];
        let wrapped = (phase + 1.0 + 180.0).rem_euclid(360.0) - 180.0;
        assert!(wrapped.abs() < 1.0, "phase={phase}");
    }

    #[test]
    fn realization_honors_configured_crossover_family() {
        let make = |kind| chain(vec![create_crossover_plugin(kind, 80.0, "high")]);
        let lr24 = make("LR24");
        let lr48 = make("LR48");
        let mut provider = NoConvolutionIr;
        let lr24_db = 20.0
            * RealizedDsp::new(&lr24, 48_000.0, &mut provider)
                .unwrap()
                .response_at(20.0)
                .unwrap()
                .norm()
                .log10();
        let mut provider = NoConvolutionIr;
        let lr48_db = 20.0
            * RealizedDsp::new(&lr48, 48_000.0, &mut provider)
                .unwrap()
                .response_at(20.0)
                .unwrap()
                .norm()
                .log10();
        assert!(lr48_db < lr24_db - 30.0);
    }

    #[test]
    fn realization_models_non_lr24_mixed_split() {
        let chain = chain(vec![
            create_band_split_plugin(1_000.0, "LR48"),
            PluginConfigWrapper {
                plugin_type: "gain".to_string(),
                parameters: serde_json::json!({"gain_db": -12.0, "channels": [0, 1]}),
            },
            create_band_merge_plugin(2),
        ]);
        let mut provider = NoConvolutionIr;
        let mut realized = RealizedDsp::new(&chain, 48_000.0, &mut provider).unwrap();
        assert!(realized.response_at(100.0).unwrap().norm() < 0.35);
        assert!(realized.response_at(10_000.0).unwrap().norm() > 0.8);
    }

    #[test]
    fn mixed_fir_iir_replay_matches_independent_complex_branch_sum() {
        struct Ir;
        impl ConvolutionIrProvider for Ir {
            fn taps(&mut self, file: &str, _sample_rate: u32) -> Result<&[f64]> {
                assert_eq!(file, "low.wav");
                Ok(&[1.0, 0.5])
            }
        }
        for rate in [44100.0, 48000.0, 96000.0] {
            let peak = Biquad::new(BiquadFilterType::Peak, 2000.0, rate, 1.2, 3.0);
            let mut eq = crate::output::create_eq_plugin(std::slice::from_ref(&peak));
            eq.parameters["channels"] = serde_json::json!([2, 3]);
            let mut fir = crate::output::create_convolution_plugin("low.wav");
            fir.parameters["channels"] = serde_json::json!([0, 1]);
            let chain = chain(vec![
                create_band_split_plugin(1000.0, "LR24"),
                fir,
                eq,
                create_band_merge_plugin(2),
                create_gain_plugin(-6.0),
            ]);
            let low = Biquad::new(
                BiquadFilterType::Lowpass,
                1000.0,
                rate,
                std::f64::consts::FRAC_1_SQRT_2,
                0.0,
            );
            let high = Biquad::new(
                BiquadFilterType::Highpass,
                1000.0,
                rate,
                std::f64::consts::FRAC_1_SQRT_2,
                0.0,
            );
            let mut provider = Ir;
            let mut realized = RealizedDsp::new(&chain, rate, &mut provider).unwrap();
            for frequency in [100.0, 800.0, 1000.0, 2000.0, 10000.0] {
                let z = Complex64::from_polar(1.0, -std::f64::consts::TAU * frequency / rate);
                let section = |filter: &Biquad| {
                    let (a1, a2, b0, b1, b2) = filter.constants();
                    (b0 + b1 * z + b2 * z * z) / (1.0 + a1 * z + a2 * z * z)
                };
                // Independent coefficient-polynomial/DTFT oracle: parallel
                // branches add complexly, then the full-band gain is applied.
                let expected = (section(&low).powu(2) * (1.0 + 0.5 * z)
                    + section(&high).powu(2) * section(&peak))
                    * 10.0_f64.powf(-6.0 / 20.0);
                let actual = realized.response_at(frequency).unwrap();
                assert!(
                    (actual - expected).norm() < 1e-10,
                    "mixed branch mismatch at {frequency} Hz / {rate}: {actual} vs {expected}"
                );
            }
            let grid: Vec<_> = (0..=128).map(|i| i as f64 * rate / 256.0).collect();
            let batched = realized.response_grid(&grid).unwrap();
            for (frequency, actual) in grid.iter().zip(batched) {
                let expected = realized.response_at(*frequency).unwrap();
                assert!((actual - expected).norm() < 1e-10);
            }
        }
    }

    #[test]
    fn realization_evaluates_warped_biquad_topology() {
        let filter = Biquad::new(BiquadFilterType::Peak, 100.0, 48_000.0, 2.0, -6.0);
        let chain = chain(vec![create_warped_eq_plugin(&[], &[filter], Some(0.8))]);
        let mut provider = NoConvolutionIr;
        let magnitude = RealizedDsp::new(&chain, 48_000.0, &mut provider)
            .unwrap()
            .response_at(100.0)
            .unwrap()
            .norm();
        assert!(magnitude.is_finite());
        assert!(magnitude < 1.0);
    }

    #[test]
    fn realization_rejects_unknown_plugins_instead_of_assuming_unity() {
        let chain = chain(vec![PluginConfigWrapper {
            plugin_type: "misspelled_eq".to_string(),
            parameters: serde_json::json!({}),
        }]);
        let mut provider = NoConvolutionIr;
        let error = RealizedDsp::new(&chain, 48_000.0, &mut provider)
            .unwrap()
            .response_at(1_000.0)
            .unwrap_err();
        assert!(error.to_string().contains("misspelled_eq"));
    }

    #[test]
    fn realization_rejects_malformed_known_plugins_instead_of_defaulting() {
        let malformed_plugins = [
            PluginConfigWrapper {
                plugin_type: "gain".to_string(),
                parameters: serde_json::json!({}),
            },
            PluginConfigWrapper {
                plugin_type: "delay".to_string(),
                parameters: serde_json::json!({"delay_ms": "late"}),
            },
            PluginConfigWrapper {
                plugin_type: "crossover".to_string(),
                parameters: serde_json::json!({
                    "type": "not-a-crossover",
                    "frequency": 80.0,
                    "output": "high"
                }),
            },
            PluginConfigWrapper {
                plugin_type: "eq".to_string(),
                parameters: serde_json::json!({}),
            },
        ];

        for plugin in malformed_plugins {
            let chain = chain(vec![plugin]);
            let mut provider = NoConvolutionIr;
            RealizedDsp::new(&chain, 48_000.0, &mut provider)
                .unwrap()
                .response_at(1_000.0)
                .expect_err("malformed known plugin must fail closed");
        }

        let chain = chain(vec![
            create_band_split_plugin(1_000.0, "LR24"),
            create_band_merge_plugin(3),
        ]);
        let mut provider = NoConvolutionIr;
        RealizedDsp::new(&chain, 48_000.0, &mut provider)
            .unwrap()
            .response_at(1_000.0)
            .expect_err("unsupported band count must fail closed");
    }

    #[test]
    fn realization_uses_injected_convolution_sidecar() {
        struct TestIr(Vec<f64>);

        impl ConvolutionIrProvider for TestIr {
            fn taps(&mut self, ir_file: &str, sample_rate: u32) -> Result<&[f64]> {
                assert_eq!(ir_file, "correction.wav");
                assert_eq!(sample_rate, 48_000);
                Ok(&self.0)
            }
        }

        let chain = chain(vec![crate::output::create_convolution_plugin(
            "correction.wav",
        )]);
        let mut provider = TestIr(vec![1.0, 1.0]);
        let actual = RealizedDsp::new(&chain, 48_000.0, &mut provider)
            .unwrap()
            .response_at(12_000.0)
            .unwrap();
        assert!((actual - Complex64::new(1.0, -1.0)).norm() < 1e-12);
    }

    #[test]
    fn fft_grid_matches_scalar_for_long_firs_and_parallel_chains() {
        struct TestIr {
            taps: Vec<f64>,
            reads: usize,
        }
        impl ConvolutionIrProvider for TestIr {
            fn taps(&mut self, _: &str, _: u32) -> Result<&[f64]> {
                self.reads += 1;
                Ok(&self.taps)
            }
        }
        for rate in [44100.0, 48000.0, 96000.0] {
            for length in [0, 1, 97, 4096] {
                let taps: Vec<_> = (0..length)
                    .map(|i| (i as f64 * 0.13).cos() * (-0.003 * i as f64).exp())
                    .collect();
                let mut fir = crate::output::create_convolution_plugin("test.wav");
                fir.parameters["mix"] = serde_json::json!(0.7);
                fir.parameters["gain_db"] = serde_json::json!(-3.0);
                let mut channel = chain(vec![fir.clone(), create_gain_plugin(-2.0)]);
                channel.drivers = Some(vec![
                    DriverDspChain {
                        measured_acoustics: None,
                        name: "low".into(),
                        index: 0,
                        plugins: vec![fir.clone()],
                        initial_curve: None,
                        measured_band_hz: None,
                    },
                    DriverDspChain {
                        measured_acoustics: None,
                        name: "high".into(),
                        index: 1,
                        plugins: vec![create_gain_plugin(-6.0)],
                        initial_curve: None,
                        measured_band_hz: None,
                    },
                ]);
                let mut grid: Vec<_> = (0..=128).map(|i| i as f64 * rate / 256.0).collect();
                let mut provider = TestIr { taps, reads: 0 };
                let fast = RealizedDsp::new(&channel, rate, &mut provider)
                    .unwrap()
                    .response_grid(&grid)
                    .unwrap();
                assert_eq!(
                    provider.reads, 1,
                    "reuse shared FIR spectrum across branches"
                );
                let mut scalar = RealizedDsp::new(&channel, rate, &mut provider).unwrap();
                for (frequency, actual) in grid.iter().zip(fast) {
                    let expected = scalar.response_at(*frequency).unwrap();
                    assert!(
                        (actual - expected).norm() < 1e-9 * (1.0 + expected.norm()),
                        "{rate} Hz, {length} taps, bin {frequency}: {actual} != {expected}"
                    );
                }
                grid.extend([20.123, 305.7, rate * 0.49999]);
                grid.sort_by(f64::total_cmp);
                let mut realized = RealizedDsp::new(&channel, rate, &mut provider).unwrap();
                let anchored = realized.response_grid(&grid).unwrap();
                for (frequency, actual) in grid.iter().zip(anchored) {
                    let expected = realized.response_at(*frequency).unwrap();
                    assert!((actual - expected).norm() < 1e-9 * (1.0 + expected.norm()));
                }
                // Changed taps must be observed by the next call, even with
                // the same resource identity. Arbitrary grids remain scalar.
                provider.taps = vec![0.25];
                let arbitrary = [0.0, 57.3, 900.0, rate / 2.0];
                let mut realized = RealizedDsp::new(&channel, rate, &mut provider).unwrap();
                let actual = realized.response_grid(&arbitrary).unwrap();
                for (f, h) in arbitrary.iter().zip(actual) {
                    assert_eq!(h, realized.response_at(*f).unwrap());
                }
                let refreshed = realized.response_grid(&grid).unwrap();
                assert_eq!(refreshed[0], realized.response_at(0.0).unwrap());
            }
        }
    }

    #[test]
    fn scalar_fir_polynomial_matches_direct_phasor_sum() {
        for rate in [44100.0, 48000.0, 96000.0] {
            for length in [0, 1, 4096, 65537] {
                let taps: Vec<_> = (0..length)
                    .map(|i| (i as f64 * 0.113).cos() * (-0.0003 * i as f64).exp())
                    .collect();
                for frequency in [
                    0.0,
                    0.001,
                    20.123,
                    300.1,
                    1000.0,
                    rate * 0.49999,
                    rate * 0.5,
                    rate,
                ] {
                    let expected = math_audio_dsp::fir_complex_response(&taps, frequency, rate);
                    let actual = fir_response_at(&taps, frequency, rate);
                    assert!(
                        (actual - expected).norm() < 1e-9 * (1.0 + expected.norm()),
                        "{rate} Hz, {length} taps, {frequency} Hz: {actual} vs {expected}"
                    );
                }
            }
            // Exact DC and Nyquist nulls must remain nulls.
            assert_eq!(
                fir_response_at(&[1.0, -1.0], 0.0, rate),
                Complex64::default()
            );
            assert!(fir_response_at(&[1.0, 1.0], rate / 2.0, rate).norm() < 1e-12);
        }
    }

    #[test]
    fn realization_sums_driver_branches_before_channel_plugins() {
        let mut chain = chain(vec![create_gain_plugin(-6.020_599_913_279_624)]);
        chain.drivers = Some(vec![
            DriverDspChain {
                measured_acoustics: None,
                name: "woofer".to_string(),
                index: 0,
                plugins: vec![create_gain_plugin(0.0)],
                initial_curve: None,
                measured_band_hz: None,
            },
            DriverDspChain {
                measured_acoustics: None,
                name: "tweeter".to_string(),
                index: 1,
                plugins: vec![create_gain_plugin(0.0)],
                initial_curve: None,
                measured_band_hz: None,
            },
        ]);
        let mut provider = NoConvolutionIr;
        let actual = RealizedDsp::new(&chain, 48_000.0, &mut provider)
            .unwrap()
            .response_at(1_000.0)
            .unwrap();
        assert!((actual - Complex64::new(1.0, 0.0)).norm() < 1e-12);
    }
}
