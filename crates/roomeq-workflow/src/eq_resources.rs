//! Resolve filesystem-backed EQ resources before engine execution.

use std::path::Path;

use roomeq_engine::eq::{EqResources, PreparedEqTarget, PreparedImpulseResponse};
use roomeq_model::{OptimizerConfig, TargetCurveConfig};

/// Resolve target curves and optional SSIR impulse responses for the EQ engine.
pub fn prepare_eq_resources(
    optimizer: &OptimizerConfig,
    target: Option<&TargetCurveConfig>,
) -> Result<EqResources, Box<dyn std::error::Error>> {
    let target = match target {
        Some(TargetCurveConfig::Path(path)) => Some(PreparedEqTarget::Curve(Box::new(
            autoeq_measurements::read::read_curve_from_csv(path)?,
        ))),
        Some(TargetCurveConfig::Predefined(name)) => {
            Some(PreparedEqTarget::Predefined(name.clone()))
        }
        None => None,
    };
    let impulse_response = optimizer.ssir_wav_path.as_deref().and_then(decode_mono_wav);

    Ok(EqResources {
        target,
        impulse_response,
    })
}

fn decode_mono_wav(path: &Path) -> Option<PreparedImpulseResponse> {
    let reader = hound::WavReader::open(path).ok()?;
    let spec = reader.spec();
    let channels = usize::from(spec.channels);
    if channels == 0 || spec.sample_rate == 0 || spec.bits_per_sample == 0 {
        return None;
    }
    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => reader
            .into_samples::<f32>()
            .collect::<Result<_, _>>()
            .ok()?,
        hound::SampleFormat::Int => {
            let full_scale = 1_i64.checked_shl(u32::from(spec.bits_per_sample - 1))?;
            let scale = 1.0 / full_scale as f32;
            reader
                .into_samples::<i32>()
                .map(|sample| sample.map(|sample| sample as f32 * scale))
                .collect::<Result<_, _>>()
                .ok()?
        }
    };
    if samples.is_empty() {
        return None;
    }

    let frames = samples.len() / channels;
    let samples = if channels == 1 {
        samples
    } else {
        (0..frames).map(|frame| samples[frame * channels]).collect()
    };
    Some(PreparedImpulseResponse {
        samples,
        sample_rate: f64::from(spec.sample_rate),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolves_target_csv_and_impulse_response() {
        let directory = tempfile::TempDir::new().expect("temp EQ resources");
        let target_path = directory.path().join("target.csv");
        std::fs::write(&target_path, "frequency,spl\n100,0\n1000,-1\n").expect("write target");
        let wav_path = directory.path().join("rir.wav");
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: 48_000,
            bits_per_sample: 32,
            sample_format: hound::SampleFormat::Float,
        };
        let mut writer = hound::WavWriter::create(&wav_path, spec).expect("create RIR");
        for sample in std::iter::once(1.0_f32).chain(std::iter::repeat_n(0.0, 511)) {
            writer.write_sample(sample).expect("write RIR sample");
        }
        writer.finalize().expect("finalize RIR");
        let optimizer = OptimizerConfig {
            ssir_wav_path: Some(wav_path),
            ..OptimizerConfig::default()
        };

        let resources =
            prepare_eq_resources(&optimizer, Some(&TargetCurveConfig::Path(target_path)))
                .expect("resolve resources");

        assert!(matches!(resources.target, Some(PreparedEqTarget::Curve(_))));
        assert_eq!(
            resources
                .impulse_response
                .as_ref()
                .expect("impulse")
                .samples
                .len(),
            512
        );
    }

    #[test]
    fn missing_optional_impulse_is_ignored() {
        let optimizer = OptimizerConfig {
            ssir_wav_path: Some("/missing/rir.wav".into()),
            ..OptimizerConfig::default()
        };
        let resources = prepare_eq_resources(&optimizer, None).expect("resolve optional resources");
        assert!(resources.impulse_response.is_none());
    }
}
