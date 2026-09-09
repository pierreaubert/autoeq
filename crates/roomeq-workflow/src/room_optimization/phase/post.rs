use super::super::*;
use super::types::GeneratedFir;

/// Whether the post-workflow stage should generate a FIR for a channel.
///
/// The stage exists for channels that have no correction of their own yet
/// (e.g. plain IIR workflows that still want a Hybrid residual FIR). In
/// PhaseLinear mode the FIR is designed from the *raw* measurement, so
/// generating it on a channel whose chain already carries correction stages
/// (biquads or convolution) stacks a second, full correction on top of the
/// first one and wrecks the response.
pub(in super::super) fn should_post_generate_fir(
    mode: roomeq_model::ProcessingMode,
    has_fir_coeffs: bool,
    chain_has_correction: bool,
    chain_has_fir: bool,
) -> bool {
    if has_fir_coeffs {
        return false;
    }
    if mode == roomeq_model::ProcessingMode::Hybrid && chain_has_fir {
        return false;
    }
    if mode == roomeq_model::ProcessingMode::PhaseLinear && chain_has_correction {
        return false;
    }
    true
}

/// Post-generate FIR coefficients for a channel that only has IIR results.
///
/// For Hybrid mode, uses the IIR-corrected curve as FIR input;
/// for PhaseLinear (FIR-only) mode, uses the raw measurement. When the
/// channel's DSP `chain` is provided, Hybrid-mode input is evaluated on the
/// routing-removed curve so an intentional bass-management high-pass cannot
/// tilt the design (see `hybrid_fir_design_input`).
#[allow(clippy::too_many_arguments)]
pub(in super::super) fn post_generate_fir(
    name: &str,
    initial_curve: &Curve,
    final_curve: &Curve,
    config: &roomeq_model::OptimizerConfig,
    target_curve: Option<&roomeq_model::TargetCurveConfig>,
    sample_rate: f64,
    output_dir: Option<&Path>,
    chain: Option<&roomeq_model::ChannelDspChain>,
) -> Option<GeneratedFir> {
    let hybrid_input;
    let fir_input = match config.processing_mode {
        ProcessingMode::Hybrid => {
            hybrid_input = chain.map(|chain| {
                super::super::room_optimization_result::hybrid_fir_design_input(
                    chain,
                    initial_curve,
                    final_curve,
                    sample_rate,
                )
            });
            hybrid_input.as_ref().unwrap_or(final_curve)
        }
        // Groups have already applied driver EQ, crossover calibration and
        // global EQ. Their FIR must correct the remaining complex response,
        // not design a second full correction from the pre-global-EQ curve.
        _ if chain.is_some_and(|chain| chain.drivers.is_some()) => final_curve,
        _ => initial_curve,
    };
    // Fix the level reference before applying the IIR candidate, just as the
    // engine-owned Hybrid path does. The residual is not a new calibration.
    let prepared_target = match chain.and_then(|chain| chain.target_curve.clone()) {
        Some(target) => Ok(Curve::from(target)),
        None => fir::resolve_fir_target_curve(initial_curve, config, target_curve),
    };
    let coefficients =
        prepared_target.and_then(|target| {
            if chain.is_some_and(|chain| chain.drivers.is_some()) {
                roomeq_engine::fir::generate_group_residual_fir_prepared(fir_input, config, &target, sample_rate)
            } else {
                fir::generate_fir_correction_prepared(fir_input, config, &target, sample_rate)
            }
        });
    match coefficients {
        Ok(coeffs) => {
            let mut filename = autoeq_artifacts::roomeq::convolution_artifact_filename(
                name,
                autoeq_artifacts::roomeq::ConvolutionArtifactKind::Fir,
                sample_rate,
            );
            if let Some(out_dir) = output_dir {
                let reserved = autoeq_artifacts::roomeq::reserve_convolution_artifact_path(
                    out_dir,
                    name,
                    autoeq_artifacts::roomeq::ConvolutionArtifactKind::Fir,
                    sample_rate,
                );
                filename = reserved.0;
                let wav_path = reserved.1;
                if let Err(e) =
                    math_audio_iir_fir::save_fir_to_wav(&coeffs, sample_rate as u32, &wav_path)
                {
                    warn!("Failed to save FIR WAV for {}: {}", name, e);
                    return None;
                } else {
                    info!("  Saved FIR filter to {}", wav_path.display());
                }
            }
            Some(GeneratedFir {
                coeffs,
                filename,
                mixed_phase_report: None,
            })
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
#[derive(Debug)]
pub(in super::super) enum MixedPhasePostError {
    Candidate(String),
    Artifact(String),
}

pub(in super::super) fn post_generate_mixed_phase_fir(
    name: &str,
    initial_curve: &Curve,
    config: &roomeq_model::OptimizerConfig,
    sample_rate: f64,
    output_dir: Option<&Path>,
) -> std::result::Result<Option<GeneratedFir>, MixedPhasePostError> {
    let Some(phase) = initial_curve.phase.as_ref() else {
        return Ok(None);
    };
    if phase.is_empty() {
        return Ok(None);
    }

    let mp_config = match &config.mixed_phase {
        Some(sc) => roomeq_engine::mixed_phase::MixedPhaseConfig {
            max_fir_length_ms: sc.max_fir_length_ms,
            pre_ringing_threshold_db: sc.pre_ringing_threshold_db,
            min_spatial_depth: sc.min_spatial_depth,
            phase_smoothing_octaves: sc.phase_smoothing_octaves,
        },
        None => roomeq_engine::mixed_phase::MixedPhaseConfig::default(),
    };

    match roomeq_engine::mixed_phase::decompose_phase(initial_curve, &mp_config) {
        Ok((_min_phase, _excess, delay_ms, residual)) => {
            info!(
                "  Mixed-phase (post-workflow) '{}': delay={:.2} ms",
                name, delay_ms
            );
            let coeffs = roomeq_engine::mixed_phase::generate_excess_phase_fir(
                &initial_curve.freq,
                &residual,
                &mp_config,
                sample_rate,
            );
            let response = roomeq_engine::response::compute_fir_complex_response(
                &coeffs,
                &initial_curve.freq,
                sample_rate,
            );
            let passband_floor = initial_curve
                .spl
                .iter()
                .copied()
                .filter(|level| level.is_finite())
                .fold(f64::NEG_INFINITY, f64::max)
                - 30.0;
            let max_magnitude_deviation_db = response
                .iter()
                .zip(&initial_curve.spl)
                .filter(|(_, level)| level.is_finite() && **level >= passband_floor)
                .map(|(response, _)| {
                    let magnitude = response.norm();
                    if magnitude.is_finite() && magnitude > 0.0 {
                        (20.0 * magnitude.log10()).abs()
                    } else {
                        f64::INFINITY
                    }
                })
                .fold(0.0_f64, f64::max);
            debug!(
                "  Mixed-phase (post-workflow) '{}': max passband magnitude deviation {:.3} dB",
                name, max_magnitude_deviation_db
            );
            if max_magnitude_deviation_db > 0.5 {
                return Err(MixedPhasePostError::Candidate(format!(
                    "phase-only magnitude deviation {max_magnitude_deviation_db:.2} dB exceeds 0.50 dB"
                )));
            }
            let mixed_phase_report =
                roomeq_engine::mixed_phase::MixedPhaseCorrectionReport::from_residual(
                    delay_ms,
                    coeffs.len(),
                    &residual,
                );

            let mut filename = autoeq_artifacts::roomeq::convolution_artifact_filename(
                name,
                autoeq_artifacts::roomeq::ConvolutionArtifactKind::ExcessPhaseFir,
                sample_rate,
            );
            if let Some(out_dir) = output_dir {
                let reserved = autoeq_artifacts::roomeq::reserve_convolution_artifact_path(
                    out_dir,
                    name,
                    autoeq_artifacts::roomeq::ConvolutionArtifactKind::ExcessPhaseFir,
                    sample_rate,
                );
                filename = reserved.0;
                let wav_path = reserved.1;
                if let Err(e) =
                    math_audio_iir_fir::save_fir_to_wav(&coeffs, sample_rate as u32, &wav_path)
                {
                    warn!("Failed to save excess phase FIR for {}: {}", name, e);
                    return Err(MixedPhasePostError::Artifact(format!(
                        "failed to write excess-phase FIR for '{name}' to {}: {e}",
                        wav_path.display()
                    )));
                } else {
                    info!("  Saved excess phase FIR to {}", wav_path.display());
                }
            }

            Ok(Some(GeneratedFir {
                coeffs,
                filename,
                mixed_phase_report: Some(mixed_phase_report),
            }))
        }
        Err(error) => Err(MixedPhasePostError::Candidate(format!(
            "phase decomposition failed: {error}"
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;
    use roomeq_model::{MixedPhaseSerdeConfig, OptimizerConfig};

    fn small_curve_no_phase() -> roomeq_model::Curve {
        roomeq_model::Curve {
            freq: Array1::logspace(10.0, f64::log10(20.0), f64::log10(20_000.0), 16),
            spl: Array1::from_elem(16, 80.0),
            phase: None,
            ..Default::default()
        }
    }

    fn small_curve_empty_phase() -> roomeq_model::Curve {
        roomeq_model::Curve {
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
        assert!(result.unwrap().is_none());
    }

    #[test]
    fn post_generate_mixed_phase_fir_returns_none_with_empty_phase() {
        let curve = small_curve_empty_phase();
        let result =
            post_generate_mixed_phase_fir("left", &curve, &default_mp_config(), 48_000.0, None);
        assert!(result.unwrap().is_none());
    }

    fn fir_config() -> roomeq_model::FirConfig {
        roomeq_model::FirConfig {
            taps: 64,
            phase: "linear".to_string(),
            correct_excess_phase: false,
            phase_smoothing: 1.0 / 6.0,
            pre_ringing: None,
            max_boost_db: None,
        }
    }

    fn small_curve() -> roomeq_model::Curve {
        roomeq_model::Curve {
            freq: Array1::logspace(10.0, f64::log10(20.0), f64::log10(20_000.0), 32),
            spl: Array1::from_elem(32, 80.0),
            phase: None,
            ..Default::default()
        }
    }

    fn small_curve_with_phase() -> roomeq_model::Curve {
        roomeq_model::Curve {
            freq: Array1::logspace(10.0, f64::log10(20.0), f64::log10(20_000.0), 32),
            spl: Array1::from_elem(32, 80.0),
            phase: Some(Array1::from_elem(32, 0.0)),
            ..Default::default()
        }
    }

    #[test]
    fn group_fir_uses_residual_response_and_serialized_target() {
        let final_curve = small_curve_with_phase();
        let mut initial = final_curve.clone();
        for (&frequency, level) in initial.freq.iter().zip(initial.spl.iter_mut()) {
            if frequency <= 200.0 { *level += 8.0; }
        }
        let mut chain = crate::test_fixtures::single_channel_room_result("L").channels.remove("L").unwrap();
        chain.drivers = Some(Vec::new());
        chain.target_curve = Some((&final_curve).into());
        let config = OptimizerConfig {
            processing_mode: roomeq_model::ProcessingMode::PhaseLinear,
            min_freq: 20.0,
            max_freq: 200.0,
            fir: Some(fir_config()),
            ..OptimizerConfig::default()
        };
        let generated = post_generate_fir(
            "L", &initial, &final_curve, &config, None, 48_000.0, None, Some(&chain),
        ).unwrap();
        let response = roomeq_engine::response::compute_fir_complex_response(
            &generated.coeffs, &Array1::from_vec(vec![30.0, 50.0, 100.0, 160.0, 1000.0]), 48_000.0,
        );
        for value in response {
            assert!((20.0 * value.norm().log10()).abs() < 0.25,
                "already corrected group must not receive a second bass correction");
        }
    }

    #[test]
    fn post_generate_fir_phase_linear_succeeds() {
        let mut config = OptimizerConfig {
            processing_mode: roomeq_model::ProcessingMode::PhaseLinear,
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
            processing_mode: roomeq_model::ProcessingMode::Hybrid,
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
            None,
        );
        assert!(result.is_some(), "hybrid FIR generation should succeed");
    }

    #[test]
    fn hybrid_post_fir_preserves_original_level_reference_at_all_rates() {
        for sample_rate in [44_100.0, 48_000.0, 96_000.0] {
            for (candidate_gain, max_boost_db, expected_db) in
                [(-6.0, None, 0.0), (3.0, None, 0.0), (-6.0, Some(3.0), -3.0)]
            {
                let original = small_curve_no_phase();
                let mut residual = original.clone();
                residual.spl += candidate_gain;
                let config = OptimizerConfig {
                    processing_mode: ProcessingMode::Hybrid,
                    min_freq: 100.0,
                    max_freq: 10_000.0,
                    fir: Some(roomeq_model::FirConfig {
                        taps: 4096,
                        phase: "linear".into(),
                        max_boost_db,
                        ..Default::default()
                    }),
                    ..Default::default()
                };
                let generated = post_generate_fir(
                    "level_reference",
                    &original,
                    &residual,
                    &config,
                    None,
                    sample_rate,
                    None,
                    None,
                )
                .unwrap();
                // Independent DTFT: the residual's known constant gain and the
                // delivered FIR should sum to unity over the interior band.
                for bin in 0..257 {
                    let frequency = 500.0 * (10.0_f64).powf(bin as f64 / 256.0);
                    let (real, imaginary) = generated.coeffs.iter().enumerate().fold(
                        (0.0, 0.0),
                        |(real, imaginary), (tap, coefficient)| {
                            let phase =
                                -std::f64::consts::TAU * frequency * tap as f64 / sample_rate;
                            (
                                real + coefficient * phase.cos(),
                                imaginary + coefficient * phase.sin(),
                            )
                        },
                    );
                    let delivered_db = candidate_gain + 20.0 * real.hypot(imaginary).log10();
                    assert!(
                        (delivered_db - expected_db).abs() < 0.05,
                        "rate={sample_rate} gain={candidate_gain} f={frequency} delivered={delivered_db}"
                    );
                }
            }
        }
    }

    #[test]
    fn post_fir_write_failure_does_not_return_a_phantom_artifact() {
        let curve = small_curve_no_phase();
        let config = OptimizerConfig {
            processing_mode: ProcessingMode::PhaseLinear,
            fir: Some(roomeq_model::FirConfig {
                taps: 64,
                phase: "linear".into(),
                ..Default::default()
            }),
            ..Default::default()
        };
        let directory = tempfile::tempdir().unwrap();
        let not_a_directory = directory.path().join("regular_file");
        std::fs::write(&not_a_directory, b"preserve me").unwrap();
        assert!(
            post_generate_fir(
                "left",
                &curve,
                &curve,
                &config,
                None,
                48_000.0,
                Some(&not_a_directory),
                None
            )
            .is_none()
        );
        assert_eq!(std::fs::read(&not_a_directory).unwrap(), b"preserve me");
        let generated = post_generate_fir(
            "left",
            &curve,
            &curve,
            &config,
            None,
            48_000.0,
            Some(directory.path()),
            None,
        )
        .unwrap();
        assert!(directory.path().join(generated.filename).is_file());
    }

    #[test]
    fn post_generate_fir_returns_none_when_fir_config_missing() {
        let config = OptimizerConfig {
            processing_mode: roomeq_model::ProcessingMode::PhaseLinear,
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
            None,
        );
        assert!(
            result.is_none(),
            "FIR generation should fail without FirConfig"
        );
    }

    #[test]
    fn post_generate_fir_hybrid_on_routed_channel_has_no_level_tilt() {
        // Bass-managed channel: the reported final curve carries the
        // intentional crossover high-pass. The residual FIR must be designed
        // against the routing-removed curve; designing against the reported
        // curve bakes a large bogus broadband cut into the filter.
        let initial = small_curve();
        let final_curve = roomeq_engine::topology::apply_crossover_response_to_curve(
            &initial, "LR24", 80.0, 48_000.0, false,
        );
        let chain = roomeq_model::ChannelDspChain {
            channel: "left".to_string(),
            plugins: vec![roomeq_engine::output::create_crossover_plugin(
                "LR24", 80.0, "high",
            )],
            drivers: None,
            initial_curve: Some((&initial).into()),
            final_curve: Some((&final_curve).into()),
            eq_response: None,
            pre_ir: None,
            post_ir: None,
            fir_temporal_masking: None,
            direct_early_late_correction: None,
            target_curve: None,
        };
        let mut config = OptimizerConfig {
            processing_mode: roomeq_model::ProcessingMode::Hybrid,
            fir: Some(fir_config()),
            ..OptimizerConfig::default()
        };
        config.max_freq = 10_000.0;

        let generated = post_generate_fir(
            "left",
            &initial,
            &final_curve,
            &config,
            None,
            48_000.0,
            None,
            Some(&chain),
        )
        .expect("hybrid FIR generation should succeed");
        let freqs = Array1::from(vec![1_000.0]);
        let response = roomeq_engine::response::compute_fir_complex_response(
            &generated.coeffs,
            &freqs,
            48_000.0,
        );
        let magnitude_db = 20.0 * response[0].norm().log10();
        assert!(
            magnitude_db.abs() <= 2.0,
            "residual FIR on a routed channel should be near unity at 1 kHz, got {magnitude_db:.2} dB"
        );
    }

    #[test]
    fn post_generate_fir_skipped_when_phase_linear_channel_already_has_correction() {
        assert!(!super::should_post_generate_fir(
            roomeq_model::ProcessingMode::PhaseLinear,
            false,
            true,
            false,
        ));
        // Hybrid still completes its IIR correction with a residual FIR.
        assert!(super::should_post_generate_fir(
            roomeq_model::ProcessingMode::Hybrid,
            false,
            true,
            false,
        ));
        // A frequency-split Hybrid chain already owns its band FIR and must
        // not receive a second full-band convolution.
        assert!(!super::should_post_generate_fir(
            roomeq_model::ProcessingMode::Hybrid,
            false,
            true,
            true,
        ));
        // PhaseLinear on an uncorrected channel still generates the full FIR.
        assert!(super::should_post_generate_fir(
            roomeq_model::ProcessingMode::PhaseLinear,
            false,
            false,
            false,
        ));
        // Existing coefficients are never regenerated.
        assert!(!super::should_post_generate_fir(
            roomeq_model::ProcessingMode::Hybrid,
            true,
            false,
            false,
        ));
    }

    #[test]
    fn mixed_phase_write_failure_is_an_artifact_error_not_candidate_rejection() {
        let curve = small_curve_with_phase();
        let directory = tempfile::tempdir().unwrap();
        let obstruction = directory.path().join("regular_file");
        std::fs::write(&obstruction, b"preserve me").unwrap();
        let error = post_generate_mixed_phase_fir(
            "left",
            &curve,
            &OptimizerConfig::default(),
            48_000.0,
            Some(&obstruction),
        )
        .unwrap_err();
        assert!(
            matches!(error, MixedPhasePostError::Artifact(_)),
            "{error:?}"
        );
        assert_eq!(std::fs::read(&obstruction).unwrap(), b"preserve me");
        let generated = post_generate_mixed_phase_fir(
            "left",
            &curve,
            &OptimizerConfig::default(),
            48_000.0,
            Some(directory.path()),
        )
        .unwrap()
        .unwrap();
        assert!(directory.path().join(generated.filename).is_file());
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
        let generated = result
            .expect("mixed-phase FIR generation should not be rejected for flat phase")
            .expect("mixed-phase FIR generation should succeed for flat phase");
        assert!(!generated.coeffs.is_empty());
    }
}
