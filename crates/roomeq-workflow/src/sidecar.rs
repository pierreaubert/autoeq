//! Workflow-owned reservation and persistence of convolution sidecars.

use std::path::{Path, PathBuf};

use autoeq_artifacts::roomeq::{ConvolutionArtifactKind, reserve_convolution_artifact_path};
use roomeq_engine::channel_result::{ConvolutionSidecarReference, GeneratedConvolutionSidecar};
use roomeq_model::ProcessingMode;

/// A logical engine reference paired with its workflow-only destination path.
pub struct ReservedConvolutionSidecar {
    reference: ConvolutionSidecarReference,
    path: PathBuf,
}

impl ReservedConvolutionSidecar {
    pub fn reference(&self) -> &ConvolutionSidecarReference {
        &self.reference
    }

    pub fn path(&self) -> &Path {
        &self.path
    }
}

/// Reserve the single sidecar kind used by a generic channel-processing mode.
pub fn reserve_channel_convolution_sidecar(
    output_dir: &Path,
    channel_name: &str,
    mode: ProcessingMode,
    sample_rate: f64,
) -> Result<Option<ReservedConvolutionSidecar>, Box<dyn std::error::Error>> {
    let kind = match mode {
        ProcessingMode::PhaseLinear => ConvolutionArtifactKind::Fir,
        ProcessingMode::Hybrid => ConvolutionArtifactKind::ResidualFir,
        ProcessingMode::MixedPhase => ConvolutionArtifactKind::ExcessPhaseFir,
        ProcessingMode::LowLatency | ProcessingMode::WarpedIir | ProcessingMode::KautzModal => {
            return Ok(None);
        }
    };
    let (filename, path) =
        reserve_convolution_artifact_path(output_dir, channel_name, kind, sample_rate);
    Ok(Some(ReservedConvolutionSidecar {
        reference: ConvolutionSidecarReference::new(filename)?,
        path,
    }))
}

/// Persist coefficients returned by the engine to the matching reservation.
pub fn persist_convolution_sidecar(
    reservation: &ReservedConvolutionSidecar,
    generated: &GeneratedConvolutionSidecar,
    coefficients: &[f64],
    sample_rate: u32,
) -> Result<(), Box<dyn std::error::Error>> {
    if generated.reference != reservation.reference {
        return Err(std::io::Error::other(format!(
            "generated convolution sidecar '{}' does not match reservation '{}'",
            generated.reference.filename(),
            reservation.reference.filename()
        ))
        .into());
    }
    math_audio_iir_fir::save_fir_to_wav(coefficients, sample_rate, &reservation.path)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reservation_keeps_paths_out_of_engine_reference_and_avoids_collisions() {
        let directory = tempfile::tempdir().unwrap();
        std::fs::write(directory.path().join("left_fir_48000hz.wav"), b"existing").unwrap();
        let reserved = reserve_channel_convolution_sidecar(
            directory.path(),
            "left",
            ProcessingMode::PhaseLinear,
            48_000.0,
        )
        .unwrap()
        .unwrap();

        assert_eq!(reserved.reference().filename(), "left_fir_48000hz_002.wav");
        assert_eq!(
            reserved.path(),
            directory.path().join("left_fir_48000hz_002.wav")
        );
    }

    #[test]
    fn persistence_writes_matching_float_wav_and_rejects_mismatch() {
        let directory = tempfile::tempdir().unwrap();
        let reserved = reserve_channel_convolution_sidecar(
            directory.path(),
            "left",
            ProcessingMode::Hybrid,
            48_000.0,
        )
        .unwrap()
        .unwrap();
        let generated = GeneratedConvolutionSidecar {
            reference: reserved.reference().clone(),
            required: true,
        };
        persist_convolution_sidecar(&reserved, &generated, &[0.25, -0.5], 48_000).unwrap();
        let reader = hound::WavReader::open(reserved.path()).unwrap();
        assert_eq!(reader.spec().sample_rate, 48_000);
        assert_eq!(reader.spec().sample_format, hound::SampleFormat::Float);

        let mismatched = GeneratedConvolutionSidecar {
            reference: ConvolutionSidecarReference::new("other.wav").unwrap(),
            required: true,
        };
        assert!(persist_convolution_sidecar(&reserved, &mismatched, &[1.0], 48_000).is_err());
    }

    #[test]
    fn artifact_free_modes_do_not_reserve_sidecars() {
        let directory = tempfile::tempdir().unwrap();
        for mode in [
            ProcessingMode::LowLatency,
            ProcessingMode::WarpedIir,
            ProcessingMode::KautzModal,
        ] {
            assert!(
                reserve_channel_convolution_sidecar(directory.path(), "left", mode, 48_000.0)
                    .unwrap()
                    .is_none()
            );
        }
    }
}
