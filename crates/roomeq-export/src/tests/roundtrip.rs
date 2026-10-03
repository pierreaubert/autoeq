//! Round-trip verification tests (Stage 5): exported artifacts read back
//! against the canonical graph.

use super::super::roundtrip::{
    encode_mono_f32_wav, verify_biquad_json_roundtrip, verify_convolution_wav_roundtrip,
    wav_resource,
};
use super::make::make_test_output;
use num_complex::Complex64;
use roomeq_model::PluginConfigWrapper;
use serde_json::json;

#[test]
fn biquad_json_roundtrip_preserves_graph() {
    let graph = make_test_output();
    let report = verify_biquad_json_roundtrip(&graph, 48_000.0, 1e-12).unwrap();
    // Fixture: two channels, 3 + 2 sections.
    assert_eq!(report.channels, 2);
    assert_eq!(report.sections, 5);
    assert_eq!(report.sample_rate_hz, 48_000.0);
    assert!(report.max_abs_coefficient_error <= 1e-12);
}

#[test]
fn biquad_json_roundtrip_preserves_pareto_frequency_bits() {
    let mut graph = make_test_output();
    // This NSGA-II result previously parsed one ULP away from the exported value.
    graph
        .channels
        .get_mut("left")
        .unwrap()
        .plugins
        .push(PluginConfigWrapper {
            plugin_type: "eq".to_string(),
            parameters: json!({
                "filters": [{"filter_type": "peak", "freq": 26.493561210188922,
                             "q": 1.0, "db_gain": -0.1}]
            }),
        });
    let report = verify_biquad_json_roundtrip(&graph, 48_000.0, 1e-12).unwrap();
    assert_eq!(report.sections, 6);
}

#[test]
fn biquad_json_roundtrip_rejects_bad_shapes() {
    let graph = make_test_output();
    assert!(verify_biquad_json_roundtrip(&graph, 48_000.0, 0.0).is_err());
    assert!(verify_biquad_json_roundtrip(&graph, f64::NAN, 1e-12).is_err());
    // Ultrasonic filters cannot export: the round trip fails instead of
    // emitting an aliasing chain.
    let mut ultrasonic = make_test_output();
    ultrasonic
        .channels
        .get_mut("left")
        .unwrap()
        .plugins
        .push(PluginConfigWrapper {
            plugin_type: "eq".to_string(),
            parameters: json!({
                "filters": [{"filter_type": "peak", "freq": 30_000.0, "q": 1.0, "db_gain": 3.0}]
            }),
        });
    assert!(verify_biquad_json_roundtrip(&ultrasonic, 48_000.0, 1e-12).is_err());
}

#[test]
fn convolution_wav_roundtrip_is_sample_exact() {
    let mut graph = make_test_output();
    graph
        .channels
        .get_mut("left")
        .unwrap()
        .plugins
        .push(PluginConfigWrapper {
            plugin_type: "convolution".to_string(),
            parameters: json!({"ir_file": "ir-left.wav"}),
        });
    // Exponential-decay impulse, 256 frames at 48 kHz.
    let samples: Vec<f32> = (0..256)
        .map(|index| (-(index as f32) / 64.0).exp())
        .collect();
    let wav = encode_mono_f32_wav(&samples, 48_000);
    let resources = vec![wav_resource("ir-left.wav", wav)];
    let (_, members) = super::super::package::package_convolution_sidecars(
        &graph,
        &resources,
        &std::collections::BTreeSet::new(),
        &std::collections::HashMap::new(),
    )
    .unwrap();
    let report = verify_convolution_wav_roundtrip(&members, &resources, 0.0).unwrap();
    assert_eq!(report.sidecars, 1);
    assert_eq!(report.frames, 256);
    assert_eq!(report.max_abs_sample_error, 0.0);
    // A tampered resource no longer matches the packaged sidecar.
    let mut tampered = resources[0].bytes.to_vec();
    let last = tampered.len() - 1;
    tampered[last] = tampered[last].wrapping_add(1);
    let bad = vec![wav_resource("ir-left.wav", tampered)];
    assert!(verify_convolution_wav_roundtrip(&members, &bad, 0.0).is_err());
    // Missing resources fail instead of verifying silence.
    let empty: Vec<super::super::package::ConvolutionResource> = Vec::new();
    assert!(verify_convolution_wav_roundtrip(&members, &empty, 0.0).is_err());
    assert!(verify_convolution_wav_roundtrip(&[], &resources, 0.0).is_err());
}

const PCM_REFERENCE_GOLDEN: &str =
    include_str!("../../../autoeq-qa/wolfram/goldens/ex01_apo_biquad_gain_delay.json");
const PCM_TONE_HZ: f64 = 1_000.0;
const PCM_INPUT_AMPLITUDE: f64 = 0.1;
const PCM_PHASE_TOLERANCE: f64 = 2.0e-8;

#[derive(Clone, Copy)]
struct EmittedBiquad {
    a1: f64,
    a2: f64,
    b0: f64,
    b1: f64,
    b2: f64,
}

#[derive(Clone)]
struct EmittedBiquadState {
    coefficients: EmittedBiquad,
    x1: f64,
    x2: f64,
    y1: f64,
    y2: f64,
}

/// Minimal downstream reader for the documented normalized-coefficient contract.
///
/// It consumes the serialized a0=1 coefficient values directly. The recurrence
/// and the preamp/delay composition follow the independent EX01 reference in
/// `autoeq-qa/wolfram/ex01_apo_biquad_gain_delay.wls`.
#[derive(Clone)]
struct EmittedBiquadPcmConsumer {
    sample_rate_hz: u32,
    preamp_linear: f64,
    sections: Vec<EmittedBiquadState>,
    delay_line: Vec<f64>,
    delay_offset: usize,
}

impl EmittedBiquadPcmConsumer {
    fn from_emitted_json(document: &serde_json::Value) -> anyhow::Result<Self> {
        anyhow::ensure!(
            document.get("format").and_then(serde_json::Value::as_str)
                == Some("roomeq_normalized_biquad_coefficients"),
            "unexpected emitted format"
        );
        anyhow::ensure!(
            document.get("version").and_then(serde_json::Value::as_u64) == Some(1),
            "unsupported emitted format version"
        );
        anyhow::ensure!(
            document
                .get("coefficient_convention")
                .and_then(serde_json::Value::as_str)
                == Some("H(z) = (b0 + b1*z^-1 + b2*z^-2) / (1 + a1*z^-1 + a2*z^-2)"),
            "unsupported emitted coefficient convention"
        );

        let sample_rate = document
            .get("sample_rate_hz")
            .and_then(serde_json::Value::as_f64)
            .ok_or_else(|| anyhow::anyhow!("sample_rate_hz is missing or not numeric"))?;
        anyhow::ensure!(
            sample_rate.is_finite()
                && sample_rate >= 1.0
                && sample_rate <= f64::from(u32::MAX)
                && sample_rate.fract() == 0.0,
            "sample_rate_hz must be a positive integer rate"
        );
        let sample_rate_hz = sample_rate as u32;

        let channels = document
            .get("channels")
            .and_then(serde_json::Value::as_array)
            .ok_or_else(|| anyhow::anyhow!("channels is missing or not an array"))?;
        anyhow::ensure!(channels.len() == 1, "expected one emitted channel");
        let channel = &channels[0];
        anyhow::ensure!(
            channel.get("channel").and_then(serde_json::Value::as_str) == Some("L"),
            "emitted channel routing changed"
        );

        let preamp_db = channel
            .get("preamp_gain_db")
            .and_then(serde_json::Value::as_f64)
            .ok_or_else(|| anyhow::anyhow!("preamp_gain_db is missing or not numeric"))?;
        anyhow::ensure!(preamp_db.is_finite(), "preamp_gain_db is not finite");
        let preamp_linear = 10.0_f64.powf(preamp_db / 20.0);
        anyhow::ensure!(
            preamp_linear.is_finite() && preamp_linear > 0.0,
            "preamp gain is not representable"
        );

        let delay_ms = channel
            .get("delay_ms")
            .and_then(serde_json::Value::as_f64)
            .ok_or_else(|| anyhow::anyhow!("delay_ms is missing or not numeric"))?;
        anyhow::ensure!(
            delay_ms.is_finite() && delay_ms >= 0.0,
            "delay_ms is invalid"
        );
        let delay_as_samples = delay_ms * sample_rate / 1_000.0;
        let rounded_delay = delay_as_samples.round();
        anyhow::ensure!(
            delay_as_samples.is_finite()
                && rounded_delay <= 1_000_000.0
                && (delay_as_samples - rounded_delay).abs() <= 1.0e-9,
            "this consumer contract requires an exact integer-sample delay"
        );
        let delay_samples = rounded_delay as usize;

        let rendered_sections = channel
            .get("sections")
            .and_then(serde_json::Value::as_array)
            .ok_or_else(|| anyhow::anyhow!("sections is missing or not an array"))?;
        let mut sections = Vec::with_capacity(rendered_sections.len());
        for (index, section) in rendered_sections.iter().enumerate() {
            anyhow::ensure!(
                section
                    .get("section_index")
                    .and_then(serde_json::Value::as_u64)
                    == Some(index as u64),
                "section ordering/index changed at {index}"
            );
            anyhow::ensure!(
                section.get("order").and_then(serde_json::Value::as_u64) == Some(2),
                "section {index} is not second order"
            );
            anyhow::ensure!(
                section.get("a0").and_then(serde_json::Value::as_f64) == Some(1.0),
                "section {index} denominator is not normalized to a0=1"
            );
            let coefficient = |name: &str| -> anyhow::Result<f64> {
                let value = section
                    .get(name)
                    .and_then(serde_json::Value::as_f64)
                    .ok_or_else(|| anyhow::anyhow!("section {index} is missing {name}"))?;
                anyhow::ensure!(value.is_finite(), "section {index} {name} is not finite");
                Ok(value)
            };
            sections.push(EmittedBiquadState {
                coefficients: EmittedBiquad {
                    a1: coefficient("a1")?,
                    a2: coefficient("a2")?,
                    b0: coefficient("b0")?,
                    b1: coefficient("b1")?,
                    b2: coefficient("b2")?,
                },
                x1: 0.0,
                x2: 0.0,
                y1: 0.0,
                y2: 0.0,
            });
        }

        Ok(Self {
            sample_rate_hz,
            preamp_linear,
            sections,
            delay_line: vec![0.0; delay_samples],
            delay_offset: 0,
        })
    }

    fn process_block(&mut self, input: &[f64]) -> Vec<f64> {
        input
            .iter()
            .map(|sample| {
                let mut output = *sample;
                for section in &mut self.sections {
                    let c = section.coefficients;
                    let next = c.b0 * output + c.b1 * section.x1 + c.b2 * section.x2
                        - c.a1 * section.y1
                        - c.a2 * section.y2;
                    section.x2 = section.x1;
                    section.x1 = output;
                    section.y2 = section.y1;
                    section.y1 = next;
                    output = next;
                }
                // EX01 defines H = preamp * delay * section transfer. There is
                // no clipping stage in the serialized coefficient contract.
                output *= self.preamp_linear;
                if self.delay_line.is_empty() {
                    output
                } else {
                    let delayed = self.delay_line[self.delay_offset];
                    self.delay_line[self.delay_offset] = output;
                    self.delay_offset = (self.delay_offset + 1) % self.delay_line.len();
                    delayed
                }
            })
            .collect()
    }
}

fn normalized_biquad_test_graph(sample_rate_hz: u32) -> roomeq_model::DspGraph {
    let mut graph = make_test_output();
    graph.channels.remove("right");
    let left = graph.channels.get_mut("left").unwrap();
    left.plugins = vec![
        PluginConfigWrapper {
            plugin_type: "gain".to_string(),
            parameters: json!({"gain_db": -3.0}),
        },
        PluginConfigWrapper {
            plugin_type: "delay".to_string(),
            parameters: json!({"delay_ms": 5.0 * 1_000.0 / f64::from(sample_rate_hz)}),
        },
        PluginConfigWrapper {
            plugin_type: "eq".to_string(),
            parameters: json!({"filters": [
                {"filter_type": "peak", "freq": 1_000.0, "q": 1.0, "db_gain": 6.0},
                {"filter_type": "peak", "freq": 4_000.0, "q": 0.8, "db_gain": 0.0}
            ]}),
        },
    ];
    graph
}

fn process_partitioned(
    consumer: &mut EmittedBiquadPcmConsumer,
    input: &[f64],
    partition_sizes: &[usize],
) -> Vec<f64> {
    let mut output = Vec::with_capacity(input.len());
    let mut offset = 0;
    let mut partition = 0;
    while offset < input.len() {
        let size = partition_sizes[partition % partition_sizes.len()];
        let end = (offset + size).min(input.len());
        output.extend(consumer.process_block(&input[offset..end]));
        offset = end;
        partition += 1;
    }
    output
}

fn measured_sine_transfer(
    output: &[f64],
    first_sample: usize,
    rate_hz: u32,
    frequency_hz: f64,
) -> Complex64 {
    let omega = std::f64::consts::TAU * frequency_hz / f64::from(rate_hz);
    let count = output.len() - first_sample;
    let (in_phase, quadrature) = output[first_sample..].iter().enumerate().fold(
        (0.0, 0.0),
        |(in_phase, quadrature), (offset, sample)| {
            let phase = omega * (first_sample + offset) as f64;
            (
                in_phase + sample * phase.sin(),
                quadrature + sample * phase.cos(),
            )
        },
    );
    Complex64::new(
        2.0 * in_phase / count as f64,
        2.0 * quadrature / count as f64,
    )
}

fn complex_relative_error(actual: Complex64, expected: Complex64) -> f64 {
    (actual - expected).norm() / expected.norm()
}

#[test]
fn normalized_biquad_export_drives_partitioned_pcm_at_supported_rates() {
    // The checked-in Wolfram EX01 golden independently evaluates the published
    // RBJ peaking equations in raw-a0 form and adds -3 dB preamp plus five samples
    // of delay. Its center-frequency point anchors the 48 kHz PCM comparison.
    let golden: serde_json::Value = serde_json::from_str(PCM_REFERENCE_GOLDEN).unwrap();
    let golden_center = Complex64::new(
        golden["response_re_im"][3][0].as_f64().unwrap(),
        golden["response_re_im"][3][1].as_f64().unwrap(),
    );

    for sample_rate_hz in [44_100_u32, 48_000, 96_000] {
        let graph = normalized_biquad_test_graph(sample_rate_hz);
        let emitted =
            super::super::export_normalized_biquad_coefficients(&graph, f64::from(sample_rate_hz))
                .unwrap();
        // Read the exact serialized bytes; the downstream contract test never
        // constructs or asks production code for a Biquad.
        let document: serde_json::Value = serde_json::from_slice(emitted.as_bytes()).unwrap();
        let consumer = EmittedBiquadPcmConsumer::from_emitted_json(&document).unwrap();
        assert_eq!(consumer.sample_rate_hz, sample_rate_hz);
        assert_eq!(consumer.sections.len(), 2);

        let channel = &document["channels"][0];
        let delay_ms = channel["delay_ms"].as_f64().unwrap();
        let delay_samples = delay_ms * f64::from(sample_rate_hz) / 1_000.0;
        assert!(
            (delay_samples - 5.0).abs() <= 1.0e-9,
            "rate {sample_rate_hz}: emitted delay {delay_ms} ms did not retain five samples"
        );

        // An impulse checks exact delay onset and that all filter state survives
        // boundaries even when blocks split inside the nonzero response.
        let mut impulse = vec![0.0; 256];
        impulse[0] = 0.25;
        let mut impulse_whole = consumer.clone();
        let whole_impulse = impulse_whole.process_block(&impulse);
        let mut impulse_partitioned = consumer.clone();
        let split_impulse =
            process_partitioned(&mut impulse_partitioned, &impulse, &[1, 3, 1, 7, 16, 2, 31]);
        assert_eq!(whole_impulse, split_impulse);
        assert!(whole_impulse[..5].iter().all(|sample| *sample == 0.0));
        assert_ne!(whole_impulse[5], 0.0);

        // A low-level sine stays below full scale after the exported +3 dB
        // net response. Thirty-two whole periods avoid leakage in the phasor
        // estimate; preceding sine samples let the IIR recurrence settle.
        let period = sample_rate_hz / gcd(sample_rate_hz, PCM_TONE_HZ as u32);
        let settle = 8_192;
        let measured_samples = period as usize * 32;
        let input: Vec<f64> = (0..settle + measured_samples)
            .map(|index| {
                PCM_INPUT_AMPLITUDE
                    * (std::f64::consts::TAU * PCM_TONE_HZ * index as f64
                        / f64::from(sample_rate_hz))
                    .sin()
            })
            .collect();
        let mut tone_whole = consumer.clone();
        let whole_tone = tone_whole.process_block(&input);
        let mut tone_partitioned = consumer.clone();
        let split_tone =
            process_partitioned(&mut tone_partitioned, &input, &[1, 13, 48, 257, 1_024, 71]);
        assert_eq!(whole_tone, split_tone);
        let actual = measured_sine_transfer(&split_tone, settle, sample_rate_hz, PCM_TONE_HZ);
        let expected = Complex64::from_polar(
            PCM_INPUT_AMPLITUDE * 10.0_f64.powf((6.0 - 3.0) / 20.0),
            -std::f64::consts::TAU * PCM_TONE_HZ * 5.0 / f64::from(sample_rate_hz),
        );
        if sample_rate_hz == 48_000 {
            assert!(
                complex_relative_error(golden_center * PCM_INPUT_AMPLITUDE, expected) < 1.0e-12
            );
        }
        let error = complex_relative_error(actual, expected);
        assert!(
            error <= PCM_PHASE_TOLERANCE,
            "rate {sample_rate_hz}: emitted PCM transfer {actual:?}, expected {expected:?}, relative error {error:.3e}"
        );

        // Altering an emitted coefficient must move the PCM transfer away from
        // the independent EX01/center-frequency prediction.
        let mut corrupted = document.clone();
        let original_b0 = corrupted["channels"][0]["sections"][0]["b0"]
            .as_f64()
            .unwrap();
        corrupted["channels"][0]["sections"][0]["b0"] = json!(original_b0 * 1.01);
        let mut corrupted_consumer =
            EmittedBiquadPcmConsumer::from_emitted_json(&corrupted).unwrap();
        let corrupted_tone = process_partitioned(
            &mut corrupted_consumer,
            &input,
            &[1, 13, 48, 257, 1_024, 71],
        );
        let corrupted_transfer =
            measured_sine_transfer(&corrupted_tone, settle, sample_rate_hz, PCM_TONE_HZ);
        assert!(
            complex_relative_error(corrupted_transfer, expected) > 1.0e-3,
            "rate {sample_rate_hz}: corrupted emitted coefficient escaped the independent PCM check"
        );

        let mut reordered = document.clone();
        reordered["channels"][0]["sections"]
            .as_array_mut()
            .unwrap()
            .swap(0, 1);
        assert!(EmittedBiquadPcmConsumer::from_emitted_json(&reordered).is_err());

        let mut wrong_convention = document;
        wrong_convention["coefficient_convention"] =
            json!("H(z) = (b0 + b1*z^-1 + b2*z^-2) / (1 - a1*z^-1 - a2*z^-2)");
        assert!(EmittedBiquadPcmConsumer::from_emitted_json(&wrong_convention).is_err());
    }
}

fn gcd(mut left: u32, mut right: u32) -> u32 {
    while right != 0 {
        let remainder = left % right;
        left = right;
        right = remainder;
    }
    left
}
