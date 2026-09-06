//! Export round-trip verification (Stage 5).
//!
//! Renders an artifact from the canonical graph, reads the artifact back,
//! and compares it against the graph: biquad coefficients, channel
//! routing and order, preamp normalization, delay (latency), and
//! convolution WAV bytes. A passing round trip means the exported bytes
//! realize the graph — not just that the exporter ran without errors.

use anyhow::Context;
use roomeq_model::DspGraph;
use std::sync::Arc;

/// Biquad JSON round-trip outcome.
#[derive(Debug, Clone, PartialEq)]
pub struct BiquadRoundtripReport {
    /// Channels compared.
    pub channels: usize,
    /// Biquad sections compared.
    pub sections: usize,
    /// Largest absolute coefficient deviation across a1/a2/b0/b1/b2.
    pub max_abs_coefficient_error: f64,
    /// Artifact sample rate echoed correctly.
    pub sample_rate_hz: f64,
}

/// Render the normalized-biquad JSON artifact from `graph`, parse it
/// back, and compare every section against freshly computed canonical
/// biquads plus the graph's preamp/delay metadata.
///
/// `tolerance` bounds the absolute coefficient deviation (covers the
/// exporter's decimal rendering). Metadata echoes (filter type,
/// frequency, Q, gain, preamp, delay, channel order) must match exactly:
/// a reordered or relabeled export is a different chain.
pub fn verify_biquad_json_roundtrip(
    graph: &DspGraph,
    sample_rate: f64,
    tolerance: f64,
) -> anyhow::Result<BiquadRoundtripReport> {
    if !(tolerance > 0.0) || !tolerance.is_finite() {
        anyhow::bail!("round-trip tolerance must be finite and positive");
    }
    let artifact = super::export_normalized_biquad_coefficients(graph, sample_rate)?;
    let parsed: serde_json::Value =
        serde_json::from_str(&artifact).context("exported biquads are not valid JSON")?;
    let format = parsed.get("format").and_then(serde_json::Value::as_str);
    anyhow::ensure!(
        format == Some("roomeq_normalized_biquad_coefficients"),
        "unexpected biquad artifact format: {format:?}"
    );
    anyhow::ensure!(
        parsed.get("version").and_then(serde_json::Value::as_u64) == Some(1),
        "unexpected biquad artifact version"
    );
    let rate = parsed
        .get("sample_rate_hz")
        .and_then(serde_json::Value::as_f64)
        .context("biquad artifact is missing sample_rate_hz")?;
    anyhow::ensure!(
        (rate - sample_rate).abs() <= f64::EPSILON,
        "artifact rate {rate} differs from export rate {sample_rate}"
    );
    let convention = parsed
        .get("coefficient_convention")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("");
    anyhow::ensure!(
        convention.contains("a1*z^-1") && convention.contains("b0"),
        "artifact coefficient convention changed: {convention:?}"
    );

    let expected_channels = super::channel::sorted_channels(graph);
    let rendered_channels = parsed
        .get("channels")
        .and_then(serde_json::Value::as_array)
        .context("biquad artifact is missing channels")?;
    anyhow::ensure!(
        rendered_channels.len() == expected_channels.len(),
        "channel count changed in export ({} vs {})",
        rendered_channels.len(),
        expected_channels.len()
    );
    let mut sections = 0usize;
    let mut max_error = 0.0_f64;
    for ((channel_name, chain), rendered) in expected_channels.iter().zip(rendered_channels.iter()) {
        let short = super::channel::channel_short_name(channel_name);
        let rendered_channel = rendered
            .get("channel")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("");
        anyhow::ensure!(
            rendered_channel == short,
            "channel routing changed in export ({rendered_channel} vs {short})"
        );
        let expected_gain = super::extract::extract_gain_db(&chain.plugins);
        let rendered_gain = rendered
            .get("preamp_gain_db")
            .and_then(serde_json::Value::as_f64)
            .context("channel is missing preamp_gain_db")?;
        anyhow::ensure!(
            (rendered_gain - expected_gain).abs() <= f64::EPSILON,
            "preamp normalization changed in export ({rendered_gain} vs {expected_gain})"
        );
        let expected_delay = super::extract::extract_delay_ms(&chain.plugins).unwrap_or(0.0);
        let rendered_delay = rendered
            .get("delay_ms")
            .and_then(serde_json::Value::as_f64)
            .context("channel is missing delay_ms")?;
        anyhow::ensure!(
            (rendered_delay - expected_delay).abs() <= f64::EPSILON,
            "delay changed in export ({rendered_delay} vs {expected_delay})"
        );
        let filters = super::extract::extract_eq_filters(&chain.plugins)?;
        let rendered_sections = rendered
            .get("sections")
            .and_then(serde_json::Value::as_array)
            .context("channel is missing sections")?;
        anyhow::ensure!(
            rendered_sections.len() == filters.len(),
            "section count changed in export"
        );
        for (section_index, (filter, rendered_section)) in
            filters.iter().zip(rendered_sections.iter()).enumerate()
        {
            let as_f64 = |key: &str| {
                rendered_section
                    .get(key)
                    .and_then(serde_json::Value::as_f64)
                    .with_context(|| format!("section {section_index} is missing '{key}'"))
            };
            for (key, expected) in [
                ("frequency_hz", filter.freq),
                ("q", filter.q),
                ("gain_db", filter.gain_db),
            ] {
                let rendered_value = as_f64(key)?;
                anyhow::ensure!(
                    (rendered_value - expected).abs() <= f64::EPSILON,
                    "section {section_index} '{key}' changed in export ({rendered_value} vs {expected})"
                );
            }
            let filter_type = super::misc::parse_biquad_filter_type(&filter.filter_type)?;
            let biquad = math_audio_iir_fir::Biquad::new(
                filter_type,
                filter.freq,
                sample_rate,
                filter.q,
                filter.gain_db,
            );
            let (a1, a2, b0, b1, b2) = biquad.constants();
            for (key, expected) in [
                ("a1", a1),
                ("a2", a2),
                ("b0", b0),
                ("b1", b1),
                ("b2", b2),
            ] {
                let rendered_value = as_f64(key)?;
                let error = (rendered_value - expected).abs();
                max_error = max_error.max(error);
                anyhow::ensure!(
                    error <= tolerance,
                    "section {section_index} '{key}' deviates {error} past tolerance {tolerance}"
                );
            }
            sections += 1;
        }
    }
    Ok(BiquadRoundtripReport {
        channels: expected_channels.len(),
        sections,
        max_abs_coefficient_error: max_error,
        sample_rate_hz: rate,
    })
}

/// Convolution WAV round-trip outcome.
#[derive(Debug, Clone, PartialEq)]
pub struct ConvolutionRoundtripReport {
    /// WAV sidecars compared.
    pub sidecars: usize,
    /// Total frames compared.
    pub frames: usize,
    /// Largest absolute sample deviation.
    pub max_abs_sample_error: f32,
}

/// Decode packaged convolution WAV `members` back and compare samples
/// against the resource bytes they were packaged from.
///
/// Byte-identical resources must decode byte-identical: any deviation
/// means the packaged convolution no longer realizes the graph. Member
/// hashes are re-verified on read, and every sidecar must match a
/// supplied resource — an unmatched sidecar is a packaging integrity
/// failure, not a skipped check.
pub fn verify_convolution_wav_roundtrip(
    members: &[super::package::ExportPackageMember],
    resources: &[super::package::ConvolutionResource],
    sample_tolerance: f32,
) -> anyhow::Result<ConvolutionRoundtripReport> {
    if !(sample_tolerance >= 0.0) || !sample_tolerance.is_finite() {
        anyhow::bail!("sample tolerance must be finite and non-negative");
    }
    anyhow::ensure!(!members.is_empty(), "no convolution sidecars packaged");
    let mut sidecars = 0usize;
    let mut frames = 0usize;
    let mut max_error = 0.0_f32;
    for member in members {
        let name = member.relative_path.to_string_lossy();
        if !name.ends_with(".wav") {
            continue;
        }
        let expected_hash = super::hash::sha256_hex(&member.bytes);
        anyhow::ensure!(
            expected_hash == member.sha256,
            "sidecar '{name}' hash mismatch on read"
        );
        let expected = decode_mono_f32(&member.bytes)
            .with_context(|| format!("packaged sidecar '{name}' does not decode as WAV"))?;
        // Find the resource this sidecar was packaged from by decoding it
        // too: the packaged bytes must equal some resource's bytes.
        let mut matched = false;
        for resource in resources {
            if resource.bytes.as_ref() == member.bytes.as_ref() {
                let resource_samples = decode_mono_f32(&resource.bytes).with_context(|| {
                    format!("resource '{}' does not decode as WAV", resource.reference)
                })?;
                anyhow::ensure!(
                    resource_samples.len() == expected.len(),
                    "sidecar '{name}' frame count changed"
                );
                for (index, (got, want)) in
                    expected.iter().zip(resource_samples.iter()).enumerate()
                {
                    let error = (got - want).abs();
                    max_error = max_error.max(error);
                    anyhow::ensure!(
                        error <= sample_tolerance,
                        "sidecar '{name}' sample {index} deviates {error} past tolerance {sample_tolerance}"
                    );
                }
                matched = true;
                break;
            }
        }
        anyhow::ensure!(matched, "sidecar '{name}' matches no supplied resource");
        sidecars += 1;
        frames += expected.len();
    }
    anyhow::ensure!(sidecars > 0, "no WAV sidecars found to verify");
    Ok(ConvolutionRoundtripReport { sidecars, frames, max_abs_sample_error: max_error })
}

/// Decode a mono f32 sample vector from WAV bytes (any channel layout is
/// rejected: packaged IRs are mono by construction).
fn decode_mono_f32(bytes: &[u8]) -> anyhow::Result<Vec<f32>> {
    let mut reader = hound::WavReader::new(std::io::Cursor::new(bytes))
        .context("WAV header unreadable")?;
    let spec = reader.spec();
    anyhow::ensure!(
        spec.channels == 1,
        "expected mono IR, found {} channels",
        spec.channels
    );
    match spec.sample_format {
        hound::SampleFormat::Float => reader
            .samples::<f32>()
            .collect::<Result<Vec<f32>, _>>()
            .context("float samples unreadable"),
        hound::SampleFormat::Int => {
            let bits = spec.bits_per_sample;
            anyhow::ensure!(bits == 16 || bits == 24 || bits == 32, "unsupported bit depth {bits}");
            let scale = (1u64 << (bits - 1)) as f32;
            reader
                .samples::<i32>()
                .map(|sample| sample.map(|value| value as f32 / scale))
                .collect::<Result<Vec<f32>, _>>()
                .context("int samples unreadable")
        }
    }
}

/// Encode mono f32 samples to WAV bytes (test helper, same layout the
/// packager accepts).
pub fn encode_mono_f32_wav(samples: &[f32], sample_rate: u32) -> Vec<u8> {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 32,
        sample_format: hound::SampleFormat::Float,
    };
    let mut cursor = std::io::Cursor::new(Vec::new());
    {
        let mut writer = hound::WavWriter::new(&mut cursor, spec).expect("WAV writer");
        for sample in samples {
            writer.write_sample(*sample).expect("WAV sample");
        }
        writer.finalize().expect("WAV finalize");
    }
    cursor.into_inner()
}

/// Reference bytes behind one packaged sidecar (test helper).
pub fn wav_resource(reference: &str, wav: Vec<u8>) -> super::package::ConvolutionResource {
    super::package::ConvolutionResource {
        reference: String::from(reference),
        bytes: Arc::from(wav.into_boxed_slice()),
    }
}
