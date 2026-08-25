//! Canonical complex-response evaluation of serialized RoomEQ DSP chains.
//!
//! Optimizer reports, runtime safety checks, held-out QA, and export replay
//! must agree on the transfer represented by [`ChannelDspChain`]. File loading
//! remains owned by `roomeq-workflow` through [`ConvolutionIrProvider`].

use crate::error::{AutoeqError, Result};
use crate::response::{MIN_REALIZATION_RESPONSE_DB, apply_complex_response_with_min_db};
use math_audio_dsp::{biquad_complex_response, fir_complex_response};
use math_audio_iir_fir::{
    Biquad, BiquadFilterType, KautzFilter, KautzSection, bark_lambda, warp_frequency,
};
use ndarray::Array1;
use num_complex::Complex64;
use roomeq_model::{ChannelDspChain, PluginConfigWrapper};
use std::f64::consts::PI;

/// Resolves convolution sidecars without coupling the DSP engine to file I/O.
pub trait ConvolutionIrProvider {
    fn taps(&mut self, ir_file: &str, sample_rate: u32) -> Result<&[f64]>;
}

/// Provider for chains that are required not to contain convolution plugins.
#[derive(Debug, Default)]
pub struct NoConvolutionIr;

impl ConvolutionIrProvider for NoConvolutionIr {
    fn taps(&mut self, ir_file: &str, _sample_rate: u32) -> Result<&[f64]> {
        Err(AutoeqError::InvalidConfiguration {
            message: format!(
                "DSP realization requires convolution IR '{ir_file}', but no sidecar provider was supplied"
            ),
        })
    }
}

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

fn checked_sample_rate(sample_rate: f64) -> Result<u32> {
    if !sample_rate.is_finite() || sample_rate <= 0.0 || sample_rate > u32::MAX as f64 {
        return Err(AutoeqError::InvalidConfiguration {
            message: format!("invalid DSP realization sample rate {sample_rate}"),
        });
    }
    Ok(sample_rate.round() as u32)
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

fn plugin_chain_response<P: ConvolutionIrProvider>(
    plugins: &[PluginConfigWrapper],
    frequency_hz: f64,
    sample_rate: f64,
    sample_rate_u32: u32,
    convolution: &mut P,
) -> Result<Complex64> {
    let mut response = Complex64::new(1.0, 0.0);
    let mut index = 0usize;
    while index < plugins.len() {
        let plugin = &plugins[index];
        if plugin.plugin_type == "band_split" {
            let merge_offset = plugins[index + 1..]
                .iter()
                .position(|candidate| candidate.plugin_type == "band_merge")
                .ok_or_else(|| AutoeqError::InvalidConfiguration {
                    message: "serialized band_split has no matching band_merge".to_string(),
                })?;
            let merge_index = index + 1 + merge_offset;
            let merge = &plugins[merge_index];
            let bands = merge
                .parameters
                .get("bands")
                .and_then(serde_json::Value::as_u64)
                .ok_or_else(|| malformed("band_merge plugin", "bands", &merge.parameters))?;
            if bands != 2 {
                return Err(AutoeqError::InvalidConfiguration {
                    message: format!(
                        "unsupported band_merge plugin with {bands} bands; canonical realization supports 2"
                    ),
                });
            }
            response *= mixed_band_response(
                plugin,
                &plugins[index + 1..merge_index],
                frequency_hz,
                sample_rate,
                sample_rate_u32,
                convolution,
            )?;
            index = merge_index + 1;
            continue;
        }
        if plugin.plugin_type == "band_merge" {
            return Err(AutoeqError::InvalidConfiguration {
                message: "serialized band_merge has no preceding band_split".to_string(),
            });
        }
        response *= plugin_response(
            plugin,
            frequency_hz,
            sample_rate,
            sample_rate_u32,
            convolution,
        )?;
        index += 1;
    }
    Ok(response)
}

fn plugin_response<P: ConvolutionIrProvider>(
    plugin: &PluginConfigWrapper,
    frequency_hz: f64,
    sample_rate: f64,
    sample_rate_u32: u32,
    convolution: &mut P,
) -> Result<Complex64> {
    match plugin.plugin_type.as_str() {
        "gain" => {
            let gain_db = required_finite_number(&plugin.parameters, "gain_db", "gain plugin")?;
            let sign = if optional_bool(&plugin.parameters, "invert", false, "gain plugin")? {
                -1.0
            } else {
                1.0
            };
            Ok(Complex64::new(sign * 10.0_f64.powf(gain_db / 20.0), 0.0))
        }
        "convolution" => convolution_response(
            plugin,
            frequency_hz,
            sample_rate,
            sample_rate_u32,
            convolution,
        ),
        "crossover" => {
            let crossover_hz = required_frequency(
                &plugin.parameters,
                "frequency",
                "crossover plugin",
                sample_rate,
            )?;
            let output = required_string(&plugin.parameters, "output", "crossover plugin")?;
            let is_lowpass = match output.to_ascii_lowercase().as_str() {
                "low" | "lowpass" | "lp" => true,
                "high" | "highpass" | "hp" => false,
                "both" => return Ok(Complex64::new(1.0, 0.0)),
                other => {
                    return Err(AutoeqError::InvalidConfiguration {
                        message: format!("unsupported crossover output mode '{other}'"),
                    });
                }
            };
            let crossover_type = required_crossover_type(plugin, "crossover plugin")?;
            Ok(crate::topology::compute_crossover_complex_response(
                crossover_type,
                crossover_hz,
                sample_rate,
                is_lowpass,
                &ndarray::array![frequency_hz],
            )[0])
        }
        "delay" => {
            let delay_ms = required_finite_number(&plugin.parameters, "delay_ms", "delay plugin")?;
            Ok(Complex64::from_polar(
                1.0,
                -2.0 * PI * frequency_hz * delay_ms / 1_000.0,
            ))
        }
        "eq" => eq_response(plugin, frequency_hz, sample_rate),
        other => Err(AutoeqError::InvalidConfiguration {
            message: format!("unsupported serialized RoomEQ plugin '{other}' in DSP realization"),
        }),
    }
}

fn eq_response(
    plugin: &PluginConfigWrapper,
    frequency_hz: f64,
    sample_rate: f64,
) -> Result<Complex64> {
    let mut response = Complex64::new(1.0, 0.0);
    let filters = plugin
        .parameters
        .get("filters")
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| malformed("EQ plugin", "filters", &plugin.parameters))?;
    for filter in filters {
        response *= serialized_filter_response(filter, frequency_hz, sample_rate)?;
    }
    Ok(response)
}

fn serialized_filter_response(
    filter: &serde_json::Value,
    frequency_hz: f64,
    sample_rate: f64,
) -> Result<Complex64> {
    let topology = optional_string(filter, "topology", "serialized RoomEQ filter")?;
    match topology {
        Some("kautz_filter") => return kautz_response(filter, frequency_hz, sample_rate),
        None | Some("biquad" | "warped_biquad") => {}
        Some(other) => {
            return Err(AutoeqError::InvalidConfiguration {
                message: format!("unsupported serialized RoomEQ filter topology '{other}'"),
            });
        }
    }

    let filter_type = filter
        .get("filter_type")
        .and_then(serde_json::Value::as_str)
        .and_then(parse_biquad_filter_type)
        .ok_or_else(|| AutoeqError::InvalidConfiguration {
            message: format!("unsupported serialized RoomEQ biquad filter {filter}"),
        })?;
    let center_hz = required_frequency(filter, "freq", "serialized RoomEQ biquad", sample_rate)?;
    let q = required_positive_number(filter, "q", "serialized RoomEQ biquad")?;
    let gain_db = required_finite_number(filter, "db_gain", "serialized RoomEQ biquad")?;

    let (design_hz, evaluation_hz) = if topology == Some("warped_biquad") {
        let lambda = optional_finite_number(
            filter,
            "lambda",
            bark_lambda(sample_rate),
            "serialized warped biquad",
        )?;
        if !(-1.0..1.0).contains(&lambda) {
            return Err(AutoeqError::InvalidConfiguration {
                message: format!(
                    "serialized warped biquad lambda must be between -1 and 1, got {lambda}"
                ),
            });
        }
        (
            warp_frequency(center_hz, sample_rate, lambda),
            warp_frequency(frequency_hz, sample_rate, lambda),
        )
    } else {
        (center_hz, frequency_hz)
    };

    Ok(biquad_complex_response(
        &Biquad::new(filter_type, design_hz, sample_rate, q, gain_db),
        evaluation_hz,
    ))
}

fn kautz_response(
    filter: &serde_json::Value,
    frequency_hz: f64,
    sample_rate: f64,
) -> Result<Complex64> {
    let sections = filter
        .get("kautz_sections")
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| AutoeqError::InvalidConfiguration {
            message: "serialized Kautz filter has no kautz_sections".to_string(),
        })?
        .iter()
        .map(|section| {
            let pole_hz = required_frequency(
                section,
                "pole_freq",
                "serialized Kautz section",
                sample_rate,
            )?;
            let q = required_positive_number(section, "q", "serialized Kautz section")?;
            let gain = required_finite_number(section, "gain", "serialized Kautz section")?;
            Ok(KautzSection::new(pole_hz, q, gain, sample_rate))
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(KautzFilter {
        sections,
        srate: sample_rate,
    }
    .complex_response(frequency_hz))
}

fn mixed_band_response<P: ConvolutionIrProvider>(
    split: &PluginConfigWrapper,
    plugins: &[PluginConfigWrapper],
    frequency_hz: f64,
    sample_rate: f64,
    sample_rate_u32: u32,
    convolution: &mut P,
) -> Result<Complex64> {
    let crossover_hz = required_frequency(
        &split.parameters,
        "frequency",
        "band_split plugin",
        sample_rate,
    )?;
    let crossover_type = required_crossover_type(split, "band_split plugin")?;
    let grid = ndarray::array![frequency_hz];
    let mut low = crate::topology::compute_crossover_complex_response(
        crossover_type,
        crossover_hz,
        sample_rate,
        true,
        &grid,
    )[0];
    let mut high = crate::topology::compute_crossover_complex_response(
        crossover_type,
        crossover_hz,
        sample_rate,
        false,
        &grid,
    )[0];

    for plugin in plugins {
        let branch_response = plugin_response(
            plugin,
            frequency_hz,
            sample_rate,
            sample_rate_u32,
            convolution,
        )?;
        if plugin_affects_mixed_band(plugin, true)? {
            low *= branch_response;
        }
        if plugin_affects_mixed_band(plugin, false)? {
            high *= branch_response;
        }
    }
    Ok(low + high)
}

fn plugin_affects_mixed_band(plugin: &PluginConfigWrapper, low_band: bool) -> Result<bool> {
    let Some(channels_value) = plugin.parameters.get("channels") else {
        return Ok(true);
    };
    let channels = channels_value.as_array().ok_or_else(|| {
        malformed(
            &format!("{} plugin", plugin.plugin_type),
            "channels",
            channels_value,
        )
    })?;
    let mut affects = false;
    for value in channels {
        let channel = value
            .as_u64()
            .filter(|channel| *channel <= 3)
            .ok_or_else(|| {
                malformed(
                    &format!("{} plugin", plugin.plugin_type),
                    "channels entry (expected 0..=3)",
                    value,
                )
            })?;
        affects |= if low_band {
            channel == 0 || channel == 1
        } else {
            channel == 2 || channel == 3
        };
    }
    Ok(affects)
}

fn convolution_response<P: ConvolutionIrProvider>(
    plugin: &PluginConfigWrapper,
    frequency_hz: f64,
    sample_rate: f64,
    sample_rate_u32: u32,
    convolution: &mut P,
) -> Result<Complex64> {
    let ir_file = plugin
        .parameters
        .get("ir_file")
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| AutoeqError::InvalidConfiguration {
            message: "serialized convolution plugin has no ir_file".to_string(),
        })?;
    let wet = fir_complex_response(
        convolution.taps(ir_file, sample_rate_u32)?,
        frequency_hz,
        sample_rate,
    );
    if ir_file.is_empty() {
        return Err(AutoeqError::InvalidConfiguration {
            message: "serialized convolution plugin has an empty ir_file".to_string(),
        });
    }
    let mix = optional_finite_number(&plugin.parameters, "mix", 1.0, "convolution plugin")?
        .clamp(0.0, 1.0);
    let gain_db = optional_finite_number(&plugin.parameters, "gain_db", 0.0, "convolution plugin")?;
    let gain = 10.0_f64.powf(gain_db / 20.0);
    Ok(Complex64::new(1.0 - mix, 0.0) + wet * (mix * gain))
}

fn malformed(owner: &str, field: &str, value: &serde_json::Value) -> AutoeqError {
    AutoeqError::InvalidConfiguration {
        message: format!("malformed {owner} field '{field}': {value}"),
    }
}

fn required_finite_number(object: &serde_json::Value, field: &str, owner: &str) -> Result<f64> {
    object
        .get(field)
        .and_then(serde_json::Value::as_f64)
        .filter(|value| value.is_finite())
        .ok_or_else(|| {
            malformed(
                owner,
                field,
                object.get(field).unwrap_or(&serde_json::Value::Null),
            )
        })
}

fn optional_finite_number(
    object: &serde_json::Value,
    field: &str,
    default: f64,
    owner: &str,
) -> Result<f64> {
    match object.get(field) {
        None => Ok(default),
        Some(value) => value
            .as_f64()
            .filter(|number| number.is_finite())
            .ok_or_else(|| malformed(owner, field, value)),
    }
}

fn required_positive_number(object: &serde_json::Value, field: &str, owner: &str) -> Result<f64> {
    let value = required_finite_number(object, field, owner)?;
    if value <= 0.0 {
        return Err(malformed(owner, field, object.get(field).unwrap()));
    }
    Ok(value)
}

fn required_frequency(
    object: &serde_json::Value,
    field: &str,
    owner: &str,
    sample_rate: f64,
) -> Result<f64> {
    let value = required_positive_number(object, field, owner)?;
    if value >= sample_rate / 2.0 {
        return Err(AutoeqError::InvalidConfiguration {
            message: format!(
                "malformed {owner} field '{field}': {value} must be below Nyquist {}",
                sample_rate / 2.0
            ),
        });
    }
    Ok(value)
}

fn required_string<'a>(object: &'a serde_json::Value, field: &str, owner: &str) -> Result<&'a str> {
    object
        .get(field)
        .and_then(serde_json::Value::as_str)
        .filter(|value| !value.is_empty())
        .ok_or_else(|| {
            malformed(
                owner,
                field,
                object.get(field).unwrap_or(&serde_json::Value::Null),
            )
        })
}

fn optional_string<'a>(
    object: &'a serde_json::Value,
    field: &str,
    owner: &str,
) -> Result<Option<&'a str>> {
    match object.get(field) {
        None => Ok(None),
        Some(value) => value
            .as_str()
            .filter(|text| !text.is_empty())
            .map(Some)
            .ok_or_else(|| malformed(owner, field, value)),
    }
}

fn optional_bool(
    object: &serde_json::Value,
    field: &str,
    default: bool,
    owner: &str,
) -> Result<bool> {
    match object.get(field) {
        None => Ok(default),
        Some(value) => value
            .as_bool()
            .ok_or_else(|| malformed(owner, field, value)),
    }
}

fn required_crossover_type<'a>(plugin: &'a PluginConfigWrapper, owner: &str) -> Result<&'a str> {
    let crossover_type = required_string(&plugin.parameters, "type", owner)?;
    crossover_type
        .parse::<roomeq_model::CrossoverType>()
        .map_err(|message| AutoeqError::InvalidConfiguration { message })?;
    Ok(crossover_type)
}

fn parse_biquad_filter_type(value: &str) -> Option<BiquadFilterType> {
    match value {
        "lowpass" => Some(BiquadFilterType::Lowpass),
        "highpass" => Some(BiquadFilterType::Highpass),
        "highpassvariableq" => Some(BiquadFilterType::HighpassVariableQ),
        "bandpass" => Some(BiquadFilterType::Bandpass),
        "peak" => Some(BiquadFilterType::Peak),
        "notch" => Some(BiquadFilterType::Notch),
        "lowshelf" => Some(BiquadFilterType::Lowshelf),
        "highshelf" => Some(BiquadFilterType::Highshelf),
        "allpass" => Some(BiquadFilterType::AllPass),
        "lowshelforf" => Some(BiquadFilterType::LowshelfOrf),
        "highshelforf" => Some(BiquadFilterType::HighshelfOrf),
        "peakmatched" => Some(BiquadFilterType::PeakMatched),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::output::{
        create_band_merge_plugin, create_band_split_plugin, create_crossover_plugin,
        create_delay_plugin, create_gain_plugin, create_gain_plugin_with_invert,
        create_warped_eq_plugin,
    };
    use roomeq_model::DriverDspChain;

    fn chain(plugins: Vec<PluginConfigWrapper>) -> ChannelDspChain {
        ChannelDspChain {
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
    fn realization_sums_driver_branches_before_channel_plugins() {
        let mut chain = chain(vec![create_gain_plugin(-6.020_599_913_279_624)]);
        chain.drivers = Some(vec![
            DriverDspChain {
                name: "woofer".to_string(),
                index: 0,
                plugins: vec![create_gain_plugin(0.0)],
                initial_curve: None,
            },
            DriverDspChain {
                name: "tweeter".to_string(),
                index: 1,
                plugins: vec![create_gain_plugin(0.0)],
                initial_curve: None,
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
