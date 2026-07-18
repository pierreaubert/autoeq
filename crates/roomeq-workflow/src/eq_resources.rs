//! Resolve filesystem-backed EQ resources before engine execution.

use roomeq_engine::eq::{EqResources, PreparedEqTarget, PreparedImpulseResponse};
use roomeq_model::{OptimizerConfig, TargetCurveConfig};

use crate::wav::decode_first_channel;

/// Resolve target curves and optional SSIR impulse responses for the EQ engine.
pub fn prepare_eq_resources(
    optimizer: &OptimizerConfig,
    target: Option<&TargetCurveConfig>,
) -> Result<EqResources, Box<dyn std::error::Error>> {
    let target = prepare_eq_target(target)?;
    let impulse_response = optimizer.ssir_wav_path.as_deref().and_then(decode_mono_wav);

    Ok(EqResources {
        target,
        impulse_response,
    })
}

/// Resolve only the configured target resource, leaving impulse preparation to
/// the channel-input adapter.
pub fn prepare_eq_target(
    target: Option<&TargetCurveConfig>,
) -> Result<Option<PreparedEqTarget>, Box<dyn std::error::Error>> {
    let target = match target {
        Some(TargetCurveConfig::Path(path)) => Some(PreparedEqTarget::Curve(Box::new(
            autoeq_measurements::read::read_curve_from_csv(path)?,
        ))),
        Some(TargetCurveConfig::Predefined(name)) => {
            Some(PreparedEqTarget::Predefined(name.clone()))
        }
        None => None,
    };
    Ok(target)
}

fn decode_mono_wav(path: &std::path::Path) -> Option<PreparedImpulseResponse> {
    let decoded = decode_first_channel(path).ok()?;
    Some(PreparedImpulseResponse {
        samples: decoded.samples,
        sample_rate: f64::from(decoded.sample_rate),
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
