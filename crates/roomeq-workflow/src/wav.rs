//! Workflow-owned WAV decoding helpers.

use std::path::Path;

pub(crate) struct DecodedMonoWav {
    pub(crate) samples: Vec<f32>,
    pub(crate) sample_rate: u32,
}

/// Decode the first channel of a WAV resource as normalized `f32` samples.
pub(crate) fn decode_first_channel(path: &Path) -> Result<DecodedMonoWav, String> {
    let reader = hound::WavReader::open(path)
        .map_err(|error| format!("Failed to open WAV '{}': {error}", path.display()))?;
    let spec = reader.spec();
    let channels = usize::from(spec.channels);
    if channels == 0 || spec.sample_rate == 0 || spec.bits_per_sample == 0 {
        return Err(format!("WAV '{}' has an invalid format", path.display()));
    }

    let interleaved: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => reader
            .into_samples::<f32>()
            .collect::<Result<_, _>>()
            .map_err(|error| format!("Failed to decode WAV '{}': {error}", path.display()))?,
        hound::SampleFormat::Int => {
            let full_scale = 1_i64
                .checked_shl(u32::from(spec.bits_per_sample - 1))
                .ok_or_else(|| format!("WAV '{}' has unsupported bit depth", path.display()))?;
            let scale = 1.0 / full_scale as f32;
            reader
                .into_samples::<i32>()
                .map(|sample| sample.map(|sample| sample as f32 * scale))
                .collect::<Result<_, _>>()
                .map_err(|error| format!("Failed to decode WAV '{}': {error}", path.display()))?
        }
    };
    let samples = interleaved
        .chunks_exact(channels)
        .map(|frame| frame[0])
        .collect::<Vec<_>>();
    if samples.is_empty() {
        return Err(format!("WAV '{}' contains no samples", path.display()));
    }

    Ok(DecodedMonoWav {
        samples,
        sample_rate: spec.sample_rate,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decodes_only_the_first_channel() {
        let file = tempfile::Builder::new().suffix(".wav").tempfile().unwrap();
        let spec = hound::WavSpec {
            channels: 2,
            sample_rate: 48_000,
            bits_per_sample: 32,
            sample_format: hound::SampleFormat::Float,
        };
        let mut writer = hound::WavWriter::create(file.path(), spec).unwrap();
        for sample in [0.25_f32, -0.25, 0.5, -0.5] {
            writer.write_sample(sample).unwrap();
        }
        writer.finalize().unwrap();

        let decoded = decode_first_channel(file.path()).unwrap();

        assert_eq!(decoded.sample_rate, 48_000);
        assert_eq!(decoded.samples, vec![0.25, 0.5]);
    }
}
