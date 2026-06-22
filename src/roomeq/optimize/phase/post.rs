use super::super::*;
use super::types::GeneratedFir;

/// Post-generate FIR coefficients for a channel that only has IIR results.
///
/// For Hybrid mode, uses the IIR-corrected curve as FIR input;
/// for PhaseLinear (FIR-only) mode, uses the raw measurement.
pub(in super::super) fn post_generate_fir(
    name: &str,
    initial_curve: &Curve,
    final_curve: &Curve,
    config: &crate::roomeq::types::OptimizerConfig,
    target_curve: Option<&crate::roomeq::types::TargetCurveConfig>,
    sample_rate: f64,
    output_dir: Option<&Path>,
) -> Option<GeneratedFir> {
    let fir_input = match config.processing_mode {
        ProcessingMode::Hybrid => final_curve,
        _ => initial_curve,
    };
    match fir::generate_fir_correction(fir_input, config, target_curve, sample_rate) {
        Ok(coeffs) => {
            let mut filename = crate::roomeq::artifacts::convolution_artifact_filename(
                name,
                crate::roomeq::artifacts::ConvolutionArtifactKind::Fir,
                sample_rate,
            );
            if let Some(out_dir) = output_dir {
                let reserved = crate::roomeq::artifacts::reserve_convolution_artifact_path(
                    out_dir,
                    name,
                    crate::roomeq::artifacts::ConvolutionArtifactKind::Fir,
                    sample_rate,
                );
                filename = reserved.0;
                let wav_path = reserved.1;
                if let Err(e) = crate::fir::save_fir_to_wav(&coeffs, sample_rate as u32, &wav_path)
                {
                    warn!("Failed to save FIR WAV for {}: {}", name, e);
                } else {
                    info!("  Saved FIR filter to {}", wav_path.display());
                }
            }
            Some(GeneratedFir { coeffs, filename })
        }
        Err(e) => {
            warn!("FIR generation failed for {}: {}", name, e);
            None
        }
    }
}

/// Post-generate a short excess-phase FIR for MixedPhase mode.
///
/// The workflow path only runs IIR optimisation.  For MixedPhase we still need
/// the short FIR that corrects residual excess phase.  This mirrors the logic
/// in `optimize_speaker_eq` MixedPhase branch but runs after the workflow.
pub(in super::super) fn post_generate_mixed_phase_fir(
    name: &str,
    initial_curve: &Curve,
    config: &crate::roomeq::types::OptimizerConfig,
    sample_rate: f64,
    output_dir: Option<&Path>,
) -> Option<GeneratedFir> {
    let phase = initial_curve.phase.as_ref()?;
    if phase.is_empty() {
        return None;
    }

    let mp_config = match &config.mixed_phase {
        Some(sc) => crate::roomeq::mixed_phase::MixedPhaseConfig {
            max_fir_length_ms: sc.max_fir_length_ms,
            pre_ringing_threshold_db: sc.pre_ringing_threshold_db,
            min_spatial_depth: sc.min_spatial_depth,
            phase_smoothing_octaves: sc.phase_smoothing_octaves,
        },
        None => crate::roomeq::mixed_phase::MixedPhaseConfig::default(),
    };

    match crate::roomeq::mixed_phase::decompose_phase(initial_curve, &mp_config) {
        Ok((_min_phase, _excess, delay_ms, residual)) => {
            info!(
                "  Mixed-phase (post-workflow) '{}': delay={:.2} ms",
                name, delay_ms
            );
            let coeffs = crate::roomeq::mixed_phase::generate_excess_phase_fir(
                &initial_curve.freq,
                &residual,
                &mp_config,
                sample_rate,
            );

            let mut filename = crate::roomeq::artifacts::convolution_artifact_filename(
                name,
                crate::roomeq::artifacts::ConvolutionArtifactKind::ExcessPhaseFir,
                sample_rate,
            );
            if let Some(out_dir) = output_dir {
                let reserved = crate::roomeq::artifacts::reserve_convolution_artifact_path(
                    out_dir,
                    name,
                    crate::roomeq::artifacts::ConvolutionArtifactKind::ExcessPhaseFir,
                    sample_rate,
                );
                filename = reserved.0;
                let wav_path = reserved.1;
                if let Err(e) = crate::fir::save_fir_to_wav(&coeffs, sample_rate as u32, &wav_path)
                {
                    warn!("Failed to save excess phase FIR for {}: {}", name, e);
                } else {
                    info!("  Saved excess phase FIR to {}", wav_path.display());
                }
            }

            Some(GeneratedFir { coeffs, filename })
        }
        Err(e) => {
            warn!(
                "  Mixed-phase decomposition failed for '{}': {}. Using IIR only.",
                name, e
            );
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::roomeq::types::{MixedPhaseSerdeConfig, OptimizerConfig};
    use ndarray::Array1;

    fn small_curve_no_phase() -> crate::Curve {
        crate::Curve {
            freq: Array1::logspace(10.0, f64::log10(20.0), f64::log10(20_000.0), 16),
            spl: Array1::from_elem(16, 80.0),
            phase: None,
            ..Default::default()
        }
    }

    fn small_curve_empty_phase() -> crate::Curve {
        crate::Curve {
            freq: Array1::logspace(10.0, f64::log10(20.0), f64::log10(20_000.0), 16),
            spl: Array1::from_elem(16, 80.0),
            phase: Some(Array1::from_vec(vec![])),
            ..Default::default()
        }
    }

    fn default_mp_config() -> OptimizerConfig {
        OptimizerConfig {
            mixed_phase: Some(MixedPhaseSerdeConfig {
                max_fir_length_ms: 10.0,
                pre_ringing_threshold_db: -30.0,
                min_spatial_depth: 0.5,
                phase_smoothing_octaves: 1.0 / 6.0,
            }),
            ..OptimizerConfig::default()
        }
    }

    #[test]
    fn post_generate_mixed_phase_fir_returns_none_without_phase() {
        let curve = small_curve_no_phase();
        let result =
            post_generate_mixed_phase_fir("left", &curve, &default_mp_config(), 48_000.0, None);
        assert!(result.is_none());
    }

    #[test]
    fn post_generate_mixed_phase_fir_returns_none_with_empty_phase() {
        let curve = small_curve_empty_phase();
        let result =
            post_generate_mixed_phase_fir("left", &curve, &default_mp_config(), 48_000.0, None);
        assert!(result.is_none());
    }

    fn fir_config() -> crate::roomeq::types::FirConfig {
        crate::roomeq::types::FirConfig {
            taps: 64,
            phase: "linear".to_string(),
            correct_excess_phase: false,
            phase_smoothing: 1.0 / 6.0,
            pre_ringing: None,
        }
    }

    fn small_curve() -> crate::Curve {
        crate::Curve {
            freq: Array1::logspace(10.0, f64::log10(20.0), f64::log10(20_000.0), 32),
            spl: Array1::from_elem(32, 80.0),
            phase: None,
            ..Default::default()
        }
    }

    fn small_curve_with_phase() -> crate::Curve {
        crate::Curve {
            freq: Array1::logspace(10.0, f64::log10(20.0), f64::log10(20_000.0), 32),
            spl: Array1::from_elem(32, 80.0),
            phase: Some(Array1::from_elem(32, 0.0)),
            ..Default::default()
        }
    }

    #[test]
    fn post_generate_fir_phase_linear_succeeds() {
        let mut config = OptimizerConfig {
            processing_mode: crate::roomeq::types::ProcessingMode::PhaseLinear,
            fir: Some(fir_config()),
            ..OptimizerConfig::default()
        };
        config.max_freq = 10_000.0;

        let result = post_generate_fir(
            "left",
            &small_curve(),
            &small_curve(),
            &config,
            None,
            48_000.0,
            None,
        );
        assert!(
            result.is_some(),
            "phase-linear FIR generation should succeed"
        );
        let generated = result.unwrap();
        assert!(!generated.coeffs.is_empty());
    }

    #[test]
    fn post_generate_fir_hybrid_succeeds() {
        let mut config = OptimizerConfig {
            processing_mode: crate::roomeq::types::ProcessingMode::Hybrid,
            fir: Some(fir_config()),
            ..OptimizerConfig::default()
        };
        config.max_freq = 10_000.0;

        let result = post_generate_fir(
            "left",
            &small_curve(),
            &small_curve(),
            &config,
            None,
            48_000.0,
            None,
        );
        assert!(result.is_some(), "hybrid FIR generation should succeed");
    }

    #[test]
    fn post_generate_fir_returns_none_when_fir_config_missing() {
        let config = OptimizerConfig {
            processing_mode: crate::roomeq::types::ProcessingMode::PhaseLinear,
            fir: None,
            ..OptimizerConfig::default()
        };

        let result = post_generate_fir(
            "left",
            &small_curve(),
            &small_curve(),
            &config,
            None,
            48_000.0,
            None,
        );
        assert!(
            result.is_none(),
            "FIR generation should fail without FirConfig"
        );
    }

    #[test]
    fn post_generate_mixed_phase_fir_succeeds_with_flat_phase() {
        let result = post_generate_mixed_phase_fir(
            "left",
            &small_curve_with_phase(),
            &default_mp_config(),
            48_000.0,
            None,
        );
        assert!(
            result.is_some(),
            "mixed-phase FIR generation should succeed for flat phase"
        );
        let generated = result.unwrap();
        assert!(!generated.coeffs.is_empty());
    }
}
