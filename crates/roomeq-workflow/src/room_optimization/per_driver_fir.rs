//! Artifact ownership for jointly designed physical-output FIRs.
use roomeq_engine::fir::PerDriverPhaseError;
use roomeq_model::decision_ledger::{
    DecisionAction, DecisionRecord, DecisionStatus, ObservedQuantity,
};
use roomeq_model::{AutoeqError, ChannelDspChain, Curve, Result, RoomConfig};
use std::path::Path;

/// Realize a routed common output FIR on each physical branch. Only post-route
/// filters commute with the branch sum: F * sum(B_i) = sum(F * B_i).
/// Pre-route filters belong to logical sources and must never be distributed.
pub(super) fn distribute_routed_firs(
    result: &mut super::RoomOptimizationResult,
    config: &RoomConfig,
    dir: &Path,
    store: &dyn autoeq_artifacts::ArtifactStore,
) -> Result<()> {
    if !config
        .optimizer
        .fir
        .as_ref()
        .is_some_and(|fir| fir.placement == roomeq_model::FirPlacement::PerDriver)
    {
        return Ok(());
    }
    for (owner, chain) in &mut result.channels {
        let has_routed_fir = chain.plugins.iter().any(|plugin| {
            plugin.plugin_type == "convolution"
                && plugin
                    .parameters
                    .get("room_eq_stage")
                    .and_then(|v| v.as_str())
                    == Some("post_route")
        });
        if chain
            .drivers
            .as_ref()
            .is_some_and(|drivers| !drivers.is_empty())
            && has_routed_fir
            && chain
                .plugins
                .iter()
                .any(|plugin| matches!(plugin.plugin_type.as_str(), "band_split" | "band_merge"))
        {
            return Err(invalid(
                "routed per_driver FIR placement cannot move a convolution out of its frequency-split hybrid block; whole-block physical realization is required",
            ));
        }
        let Some(drivers) = chain.drivers.as_mut().filter(|drivers| !drivers.is_empty()) else {
            continue;
        };
        let mut moved = Vec::new();
        for (plugin_index, plugin) in chain.plugins.iter().enumerate() {
            if plugin.plugin_type != "convolution"
                || plugin
                    .parameters
                    .get("room_eq_stage")
                    .and_then(|v| v.as_str())
                    != Some("post_route")
            {
                continue;
            }
            let filename = plugin
                .parameters
                .get("ir_file")
                .and_then(|v| v.as_str())
                .ok_or_else(|| invalid("routed convolution is missing its artifact path"))?;
            let source = dir.join(filename);
            let bytes = store.read(&source)?.ok_or_else(|| {
                invalid(format!(
                    "missing routed FIR artifact '{}'",
                    source.display()
                ))
            })?;
            for driver in drivers.iter_mut() {
                // Encode owner bytes to avoid sanitized-name collisions. Probe the
                // store too, so in-memory exports have the same no-overwrite contract.
                let owner_key: String = owner
                    .as_bytes()
                    .iter()
                    .map(|b| format!("{b:02x}"))
                    .collect();
                let stem = format!("physical_{owner_key}_{}_{}", driver.index, plugin_index);
                let mut suffix = 0usize;
                let (filename, path) = loop {
                    let filename = format!("{stem}_{suffix}.wav");
                    let path = dir.join(&filename);
                    if store.read(&path)?.is_none() {
                        break (filename, path);
                    }
                    suffix += 1;
                };
                store.write(&path, &bytes)?;
                let mut physical = plugin.clone();
                physical.parameters["ir_file"] = serde_json::json!(filename);
                physical.parameters["room_eq_fir_placement"] = serde_json::json!("per_driver");
                physical.parameters["room_eq_fir_design_scope"] =
                    serde_json::json!("shared_kernel_per_physical_output");
                driver.plugins.push(physical);
            }
            moved.push(plugin_index);
        }
        if !moved.is_empty() {
            for index in moved.into_iter().rev() {
                chain.plugins.remove(index);
            }
            if let Some(channel) = result.channel_results.get_mut(owner) {
                channel.fir_coeffs = None;
            }
        }
    }
    Ok(())
}

fn invalid(message: impl Into<String>) -> AutoeqError {
    AutoeqError::InvalidConfiguration {
        message: message.into(),
    }
}

#[cfg(test)]
fn generate(
    chain: &mut ChannelDspChain,
    config: &RoomConfig,
    fs: f64,
    dir: Option<&Path>,
) -> Result<Curve> {
    generate_recorded(chain, config, fs, dir).map(|(curve, _)| curve)
}

/// Generate physical FIRs and retain provisional decisions for final reconciliation.
pub(super) fn generate_recorded(
    chain: &mut ChannelDspChain,
    config: &RoomConfig,
    fs: f64,
    dir: Option<&Path>,
) -> Result<(Curve, Vec<DecisionRecord>)> {
    let dir = dir.ok_or_else(|| {
        invalid("per_driver FIR placement requires an output directory for physical FIR sidecars")
    })?;
    let target: Curve = if let Some(target) = chain.target_curve.clone() {
        target.into()
    } else {
        // Full-band group EQ has no untouched upper reference. Use the same
        // existing full-band FIR target convention once, for the group only.
        let initial: Curve = chain
            .initial_curve
            .clone()
            .ok_or_else(|| invalid("per_driver FIR requires a calibrated group capture"))?
            .into();
        crate::fir::resolve_fir_target_curve(
            &initial,
            &config.optimizer,
            config.target_curve.as_ref(),
        )
        .map_err(|e| invalid(e.to_string()))?
    };
    let sources = super::parallel_timing::validated_sources(chain, config, &target)
        .map_err(|reason| invalid(format!("per_driver FIR timing admission failed: {reason}")))?;
    let drivers = chain
        .drivers
        .as_ref()
        .ok_or_else(|| invalid("missing physical drivers"))?;
    let mut measurements = Vec::new();
    let mut branches = Vec::new();
    let mut optimizer = config.optimizer.clone();
    for driver in drivers {
        let raw: Curve = driver
            .initial_curve
            .clone()
            .ok_or_else(|| invalid(format!("missing calibrated capture for '{}'", driver.name)))?
            .into();
        raw.validate("per-driver capture")
            .map_err(|e| invalid(e.to_string()))?;
        if raw.phase.is_none() {
            return Err(invalid(
                "per_driver group FIR requires phase-referenced captures for every physical speaker",
            ));
        }
        optimizer.min_freq = optimizer.min_freq.max(raw.freq[0]);
        optimizer.max_freq = optimizer.max_freq.min(raw.freq[raw.freq.len() - 1]);
        // Interpolate BEFORE applying crossover, delay or FIR, preserving SPL
        // and unwrapping phase. Never normalize individual physical captures.
        let raw = autoeq_core::interpolate_log_space(&target.freq, &raw);
        let mut branch = chain.clone();
        branch.drivers = None;
        branch.plugins = driver.plugins.clone();
        branch.plugins.extend(chain.plugins.clone());
        branches.push(
            crate::ctc::apply_channel_dsp_chain_to_curve_with_sidecar_dir(&branch, &raw, fs, dir)?,
        );
        measurements.push(raw);
    }
    let excess_phase = optimizer.processing_mode == roomeq_model::ProcessingMode::MixedPhase
        || optimizer.fir.as_ref().is_some_and(|fir| {
            fir.phase.eq_ignore_ascii_case("kirkeby") && fir.correct_excess_phase
        });
    let mut decisions = if excess_phase {
        let assessment = optimizer
            .mixed_phase
            .as_ref()
            .map(|mixed| mixed.assessment.clone())
            .unwrap_or_default();
        let target_admission = validate_phase_target(
            &sources,
            config,
            &target,
            optimizer.active_correction_band(),
            &chain.channel,
        );
        // An evidence refusal must not hide a malformed later source or policy.
        let mut numerical_refusal = None;
        for (index, measurement) in measurements.iter().enumerate() {
            match roomeq_engine::fir::assess_per_driver_phase_target(
                measurement,
                &assessment,
                optimizer.active_correction_band(),
                fs,
            ) {
                Ok(_) => {}
                Err(PerDriverPhaseError::InvalidInput(reason)) => return Err(invalid(reason)),
                Err(PerDriverPhaseError::Unsupported(reason)) => {
                    numerical_refusal.get_or_insert_with(|| {
                        format!("per-driver {index} excess-phase assessment: {reason}")
                    });
                }
            }
        }
        let admission = target_admission.and_then(|records| match numerical_refusal {
            Some(reason) => Err(PerDriverPhaseError::Unsupported(reason)),
            None => Ok(records),
        });
        match admission {
            Ok(records) => records,
            Err(PerDriverPhaseError::Unsupported(reason))
                if assessment.retain_magnitude_on_refusal =>
            {
                // Keep the already realized magnitude chain, without inventing
                // a substitute FIR or accepting unsupported relative phase work.
                let retained = replay(chain, &target, fs, dir)?;
                let refusal = phase_refusal_record(
                    chain,
                    &sources,
                    optimizer.active_correction_band(),
                    reason,
                );
                chain.final_curve = Some((&retained).into());
                return Ok((retained, vec![refusal]));
            }
            Err(reason) => return Err(invalid(reason.to_string())),
        }
    } else {
        Vec::new()
    };
    let designed = roomeq_engine::fir::generate_per_driver_firs(
        &branches,
        &measurements,
        &target,
        &optimizer,
        fs,
    )
    .map_err(invalid)?;
    // Stage all artifact writes before mutating the deployed chain.
    let mut plugins = Vec::new();
    for (driver, taps) in drivers.iter().zip(&designed.coefficients) {
        let name = format!("{}_driver_{}_{}", chain.channel, driver.index, driver.name);
        let (filename, path) = autoeq_artifacts::roomeq::reserve_convolution_artifact_path(
            dir,
            &name,
            autoeq_artifacts::roomeq::ConvolutionArtifactKind::Fir,
            fs,
        );
        math_audio_iir_fir::save_fir_to_wav(taps, fs as u32, &path).map_err(|e| {
            invalid(format!(
                "failed to write physical FIR '{}': {e}",
                path.display()
            ))
        })?;
        let mut plugin = roomeq_engine::output::create_convolution_plugin(&filename);
        plugin.parameters["room_eq_fir_placement"] = serde_json::json!("per_driver");
        plugin.parameters["latency_samples"] = serde_json::json!(designed.latency_samples);
        plugin.parameters["correction_design_delay_ms"] =
            serde_json::json!(designed.latency_samples as f64 * 1000.0 / fs);
        plugin.parameters["fir_taps"] = serde_json::json!(taps.len());
        plugin.parameters["phase_mode"] = serde_json::json!(config.optimizer.processing_mode);
        plugin.parameters["protected_null_bins"] = serde_json::json!(designed.protected_bins);
        plugin.parameters["joint_target_rms_before_db"] = serde_json::json!(designed.before_rms_db);
        plugin.parameters["joint_target_rms_after_db"] = serde_json::json!(designed.after_rms_db);
        for decision in &mut decisions {
            decision.evidence_refs.push(format!("phase-fir:{filename}"));
            decision
                .evidence_refs
                .push(format!("phase-fir-driver:{}:{filename}", driver.index));
        }
        plugins.push(plugin);
    }
    for decision in &mut decisions {
        decision.status = if designed.selected_phase_strength > 0.0 {
            DecisionStatus::Applied
        } else {
            // An unsuccessful bounded search is not proof of physical impossibility
            // or of an already acceptable response.
            DecisionStatus::Unresolved
        };
        decision.reason_codes.push(
            if designed.selected_phase_strength > 0.0 {
                "joint_phase_candidate_realized"
            } else {
                "joint_search_selected_no_phase_correction"
            }
            .to_owned(),
        );
        for (name, value, unit) in [
            (
                "selected_phase_strength",
                designed.selected_phase_strength,
                "ratio",
            ),
            ("joint_target_rms_before", designed.before_rms_db, "db"),
            ("joint_target_rms_after", designed.after_rms_db, "db"),
            (
                "phase_fir_causal_center_delay",
                designed.latency_samples as f64 * 1000.0 / fs,
                "ms",
            ),
        ] {
            decision.observed.push(ObservedQuantity {
                name: name.into(),
                value,
                unit: unit.into(),
            });
        }
    }
    for (driver, plugin) in chain.drivers.as_mut().unwrap().iter_mut().zip(plugins) {
        driver.plugins.push(plugin);
    }
    chain.final_curve = Some((&designed.final_curve).into());
    chain.target_curve = Some((&target).into());
    Ok((designed.final_curve, decisions))
}

/// Check source-bound direct-sound support before designing detailed phase correction.
fn validate_phase_target(
    sources: &[roomeq_model::MeasurementSource],
    config: &RoomConfig,
    target: &Curve,
    band_hz: [f64; 2],
    channel: &str,
) -> std::result::Result<Vec<DecisionRecord>, PerDriverPhaseError> {
    use crate::evidence_intake::{ChannelGateInput, gate_all_channels, source_measurement_id};
    use roomeq_model::target_transition::{ProposedDetailBand, TargetChain};

    let ids: Vec<_> = sources.iter().map(source_measurement_id).collect();
    let grid = target.freq.to_vec();
    let inputs: Vec<_> = sources
        .iter()
        .zip(&ids)
        .map(|(source, id)| ChannelGateInput {
            channel: id,
            source: Some(source),
            freq_hz: &grid,
            // Only direct-sound records are consumed here. Do not claim individual
            // source phase from an aggregate branch; phase/timing have separate checks.
            has_phase_data: false,
        })
        .collect();
    let target_chain = TargetChain::from_target_config(
        config.optimizer.target_response.as_ref(),
        roomeq_model::auto_tune::resolved_schroeder_hz(&config.optimizer),
        ids.clone(),
    );
    let mut decisions = Vec::new();
    for (index, gate) in gate_all_channels(&inputs, None).iter().enumerate() {
        let (evidence, evidence_refs) = super::phase::direct_evidence_for_band(gate, band_hz);
        let proposals = [ProposedDetailBand {
            band_hz,
            evidence,
            evidence_refs,
        }];
        let report =
            roomeq_engine::target_enforcement::enforce_target_chain(&target_chain, &proposals)
                .map_err(PerDriverPhaseError::InvalidInput)?;
        if !report.resolution.limited_bands.is_empty() {
            return Err(PerDriverPhaseError::Unsupported(format!(
                "per_driver FIR target policy refused source {index} '{}', band {}-{} Hz: {:?}",
                gate.measurement_id, band_hz[0], band_hz[1], report.resolution.explanations
            )));
        }
        let mut records = crate::target_enforcement::reconcile_target_decisions(
            &report,
            &proposals,
            channel,
            channel,
            vec![gate.measurement_id.clone()],
            Vec::new(),
            None,
        );
        for record in &mut records {
            record.decision_id = format!("joint-phase-target-{channel}-{index}");
            record.action = DecisionAction::PhaseCorrect;
            // These are design bounds, not a promise of finite-FIR support.
            record.frequency_band_hz = None;
            for (name, value) in [
                ("requested_phase_band_low", band_hz[0]),
                ("requested_phase_band_high", band_hz[1]),
            ] {
                record.observed.push(ObservedQuantity {
                    name: name.into(),
                    value,
                    unit: "hz".into(),
                });
            }
            record
                .reason_codes
                .push("phase_requested_scope_not_realized_support".into());
            // The enforced user-target identity already rides in
            // `evidence_refs` from target reconciliation; never push it again.
        }
        decisions.extend(records);
    }
    Ok(decisions)
}

fn phase_refusal_record(
    chain: &ChannelDspChain,
    sources: &[roomeq_model::MeasurementSource],
    band: [f64; 2],
    reason: String,
) -> DecisionRecord {
    use roomeq_model::decision_ledger::{DECISION_LEDGER_VERSION, DecisionStage};
    DecisionRecord {
        decision_id: format!("joint-phase-refused-{}", chain.channel),
        ledger_version: DECISION_LEDGER_VERSION.into(),
        stage: DecisionStage::Provisional,
        logical_input: chain.channel.clone(),
        physical_output: chain.channel.clone(),
        measurement_refs: sources
            .iter()
            .map(crate::evidence_intake::source_measurement_id)
            .collect(),
        seat_refs: Vec::new(),
        // Evaluated request scope, not the support of an applied phase filter.
        frequency_band_hz: Some(band),
        filter_center_hz: None,
        action: DecisionAction::PhaseCorrect,
        status: DecisionStatus::InsufficientEvidence,
        reason_codes: vec![
            "joint_phase_evidence_refused".into(),
            "existing_magnitude_retained_by_policy".into(),
            reason,
        ],
        observed: Vec::new(),
        limits: Vec::new(),
        evidence_refs: Vec::new(),
        confidence: roomeq_model::AssessmentConfidence::Unknown,
        related_decision_ids: Vec::new(),
        supersedes_ids: Vec::new(),
        final_graph_identity: None,
    }
}

pub(super) fn has_physical_fir(chain: &ChannelDspChain) -> bool {
    chain.drivers.as_ref().is_some_and(|drivers| {
        drivers.iter().any(|d| {
            d.plugins.iter().any(|p| {
                p.parameters
                    .get("room_eq_fir_placement")
                    .and_then(|v| v.as_str())
                    == Some("per_driver")
            })
        })
    })
}

/// Replay the physical captures rather than multiplying their acoustic sum by
/// an electrical sum of filters. Those operations are not interchangeable.
///
/// # Errors
/// Returns an error for invalid grids, absent drivers, missing or invalid phase,
/// invalid sample rates, failed DSP realization, or non-finite complex sums.
/// Phase availability does not independently establish a common capture clock;
/// the caller must also establish compatible acquisition timing.
pub(super) fn replay(chain: &ChannelDspChain, grid: &Curve, fs: f64, dir: &Path) -> Result<Curve> {
    use num_complex::Complex64;
    Curve::validate_frequency_grid(&grid.freq, "physical replay grid")?;
    if !fs.is_finite() || fs <= 0.0 || grid.freq.iter().any(|f| *f > fs / 2.0) {
        return Err(invalid(
            "physical replay requires a valid sample rate and sub-Nyquist grid",
        ));
    }
    let drivers = chain
        .drivers
        .as_deref()
        .filter(|drivers| !drivers.is_empty())
        .ok_or_else(|| invalid("physical replay requires at least one driver"))?;
    let mut sum = vec![Complex64::new(0.0, 0.0); grid.freq.len()];
    for driver in drivers {
        let context = format!("physical replay driver '{}'", driver.name);
        let raw: Curve = driver
            .initial_curve
            .clone()
            .ok_or_else(|| invalid(format!("{context}: missing physical capture")))?
            .into();
        raw.validate(&context)?;
        if raw.phase.is_none() {
            return Err(invalid(format!(
                "{context}: measured phase is required for coherent replay"
            )));
        }
        let raw = autoeq_core::interpolate_log_space(&grid.freq, &raw);
        let mut branch = chain.clone();
        branch.drivers = None;
        branch.plugins = driver.plugins.clone();
        branch.plugins.extend(chain.plugins.clone());
        let realized =
            crate::ctc::apply_channel_dsp_chain_to_curve_with_sidecar_dir(&branch, &raw, fs, dir)?;
        realized.validate(&context)?;
        let phase = realized.phase.as_ref().ok_or_else(|| {
            invalid(format!(
                "{context}: realized phase is unavailable for coherent replay"
            ))
        })?;
        for (i, z) in sum.iter_mut().enumerate() {
            *z +=
                Complex64::from_polar(10.0_f64.powf(realized.spl[i] / 20.0), phase[i].to_radians());
            if !z.re.is_finite() || !z.im.is_finite() {
                return Err(invalid(format!(
                    "{context}: non-finite complex sum at bin {i}"
                )));
            }
        }
    }
    Ok(Curve {
        freq: grid.freq.clone(),
        spl: ndarray::Array1::from_iter(sum.iter().map(|z| 20.0 * z.norm().max(1e-15).log10())),
        phase: Some(ndarray::Array1::from_iter(
            sum.iter().map(|z| z.arg().to_degrees()),
        )),
        ..Curve::default()
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use roomeq_model::{DriverDspChain, FirConfig, FirPlacement, ProcessingMode};
    fn setup() -> (ChannelDspChain, RoomConfig, Curve) {
        let mut result = crate::test_fixtures::single_channel_room_result("L");
        let mut chain = result.channels.remove("L").unwrap();
        let freq = ndarray::Array1::logspace(10.0, 20.0_f64.log10(), 20000.0_f64.log10(), 160);
        let target = Curve {
            spl: ndarray::Array1::from_elem(freq.len(), 80.0),
            phase: Some(ndarray::Array1::zeros(freq.len())),
            freq,
            ..Curve::default()
        };
        let mut raw = target.clone();
        raw.spl -= 20.0 * 2.0_f64.log10();
        chain.plugins.clear();
        chain.target_curve = Some((&target).into());
        chain.drivers = Some(
            (0..2)
                .map(|index| DriverDspChain {
                    measured_acoustics: None,
                    name: format!("speaker/{index}"),
                    index,
                    plugins: vec![],
                    initial_curve: Some((&raw).into()),
                    measured_band_hz: match (raw.freq.first(), raw.freq.last()) {
                        (Some(&low), Some(&high)) => Some([low, high]),
                        _ => None,
                    },
                })
                .collect(),
        );
        let mut config = RoomConfig::default();
        // Explicit synthetic acquisition declarations for waveform admission.
        // They exercise the contract, not evidence of a physical recording.
        config.speakers.insert(
            "L".into(),
            roomeq_model::SpeakerConfig::Topology(roomeq_model::SpeakerTopology {
                name: "fixture".into(),
                speaker_name: None,
                crossover: None,
                parallel_groups: vec![],
                drivers: (0..2)
                    .map(|index| roomeq_model::SpeakerDriver {
                        id: format!("speaker/{index}"),
                        role: roomeq_model::SpeakerDriverRole::FullRange,
                        crossover_band: None,
                        measurement: roomeq_model::MeasurementSource::Single(
                            autoeq_core::MeasurementSingle {
                                measurement: autoeq_core::MeasurementRef::Inline(
                                    autoeq_core::InlineMeasurement {
                                        frequencies: raw.freq.to_vec(),
                                        magnitude_db: raw.spl.to_vec(),
                                        phase_deg: raw.phase.as_ref().map(|phase| phase.to_vec()),
                                        name: Some("fixture-seat".into()),
                                        wav_path: None,
                                        csv_path: None,
                                    },
                                ),
                                speaker_name: None,
                                provenance: autoeq_core::MeasurementProvenance {
                                    capture_kind: autoeq_core::ProvenanceCaptureKind::StationaryIr,
                                    timing_reference_id: Some("synthetic-fixture-clock".into()),
                                    ..Default::default()
                                },
                            },
                        ),
                    })
                    .collect(),
            }),
        );
        config.optimizer.processing_mode = ProcessingMode::PhaseLinear;
        config.optimizer.min_freq = 20.0;
        config.optimizer.max_freq = 200.0;
        config.optimizer.fir = Some(FirConfig {
            placement: FirPlacement::PerDriver,
            phase: "linear".into(),
            ..Default::default()
        });
        (chain, config, target)
    }
    #[test]
    fn physical_replay_requires_finite_phase_for_every_driver() {
        let (chain, _, grid) = setup();
        for phase in [
            None,
            Some(vec![]),
            Some(vec![0.0]),
            Some(vec![f64::NAN; grid.freq.len()]),
        ] {
            let mut invalid_chain = chain.clone();
            invalid_chain.drivers.as_mut().unwrap()[1]
                .initial_curve
                .as_mut()
                .unwrap()
                .phase = phase;
            let error = replay(&invalid_chain, &grid, 48000.0, Path::new("."))
                .expect_err("incomplete phase must not become a coherent sum");
            assert!(error.to_string().contains("speaker/1"), "{error}");
        }
    }

    #[test]
    fn roadmap_correction_physical_fir_refuses_unmapped_timing_before_artifacts() {
        let (mut chain, mut config, _) = setup();
        let dir = tempfile::tempdir().unwrap();
        config.speakers.clear();
        let before = serde_json::to_value(&chain).unwrap();
        let error = generate(&mut chain, &config, 48000.0, Some(dir.path()))
            .expect_err("phase arrays alone cannot authorize coherent FIR design");
        assert!(error.to_string().contains("timing"));
        assert_eq!(serde_json::to_value(&chain).unwrap(), before);
        assert_eq!(std::fs::read_dir(dir.path()).unwrap().count(), 0);
    }

    #[test]
    fn physical_fir_opt_in_refusal_retains_processing_but_not_invalid_inputs() {
        for mixed in [false, true] {
            let (mut chain, mut config, target) = setup();
            if mixed {
                config.optimizer.processing_mode = ProcessingMode::MixedPhase;
            } else {
                let fir = config.optimizer.fir.as_mut().unwrap();
                fir.phase = "kirkeby".into();
                fir.correct_excess_phase = true;
            }
            chain
                .plugins
                .push(roomeq_engine::output::create_gain_plugin(-1.0));
            let dir = tempfile::tempdir().unwrap();
            let before = chain.clone();
            assert!(generate_recorded(&mut chain, &config, 48000.0, Some(dir.path())).is_err());
            config
                .optimizer
                .mixed_phase
                .get_or_insert_with(|| serde_json::from_value(serde_json::json!({})).unwrap())
                .assessment
                .retain_magnitude_on_refusal = true;
            let expected = replay(&before, &target, 48000.0, dir.path()).unwrap();
            let (retained, records) =
                generate_recorded(&mut chain, &config, 48000.0, Some(dir.path())).unwrap();
            assert_eq!(
                serde_json::to_value(&chain.plugins).unwrap(),
                serde_json::to_value(&before.plugins).unwrap()
            );
            assert_eq!(
                serde_json::to_value(&chain.drivers).unwrap(),
                serde_json::to_value(&before.drivers).unwrap()
            );
            assert_eq!(retained.spl, expected.spl);
            assert_eq!(retained.phase, expected.phase);
            assert_eq!(std::fs::read_dir(dir.path()).unwrap().count(), 0);
            assert_eq!(records.len(), 1);
            records[0].validate().unwrap();
            assert_eq!(records[0].status, DecisionStatus::InsufficientEvidence);
            assert_eq!(records[0].measurement_refs.len(), 2);
            assert!(records[0].evidence_refs.is_empty());
            for fault in ["policy", "timing", "capture"] {
                let mut bad_config = config.clone();
                let mut bad_chain = before.clone();
                match fault {
                    "policy" => {
                        bad_config
                            .optimizer
                            .mixed_phase
                            .as_mut()
                            .unwrap()
                            .assessment
                            .min_valid_fraction = f64::NAN
                    }
                    "timing" => bad_config.speakers.clear(),
                    "capture" => {
                        bad_chain.drivers.as_mut().unwrap()[1]
                            .initial_curve
                            .as_mut()
                            .unwrap()
                            .noise_floor_db = Some(vec![0.0])
                    }
                    _ => unreachable!(),
                }
                assert!(
                    generate_recorded(&mut bad_chain, &bad_config, 48000.0, Some(dir.path()))
                        .is_err(),
                    "{fault}"
                );
                assert_eq!(std::fs::read_dir(dir.path()).unwrap().count(), 0);
            }
        }
    }

    #[test]
    fn roadmap_correction_physical_fir_retains_capture_quality_through_serialization() {
        for use_noise in [false, true] {
            let (mut chain, mut config, target) = setup();
            config.optimizer.fir.as_mut().unwrap().taps = 1024;
            // Request a real boost so loss of the quality evidence changes DSP.
            for level in &mut chain.target_curve.as_mut().unwrap().spl {
                *level += 6.0;
            }
            for driver in chain.drivers.as_mut().unwrap() {
                let mut capture: Curve = driver.initial_curve.clone().unwrap().into();
                // Explicit synthetic low-quality capture, not inferred evidence.
                if use_noise {
                    capture.noise_floor_db = Some(&capture.spl - 5.0);
                } else {
                    capture.coherence = Some(ndarray::Array1::from_elem(capture.freq.len(), 0.2));
                }
                driver.initial_curve = Some((&capture).into());
            }
            let json = serde_json::to_value(&chain).unwrap();
            let mut restored: ChannelDspChain = serde_json::from_value(json).unwrap();
            let dir = tempfile::tempdir().unwrap();
            let corrected = generate(&mut restored, &config, 48000.0, Some(dir.path())).unwrap();
            assert!(corrected.spl.iter().all(|level| *level <= 80.10001));
            for driver in restored.drivers.as_ref().unwrap() {
                let fir = driver.plugins.last().unwrap();
                assert_eq!(
                    fir.parameters["protected_null_bins"].as_u64(),
                    Some((2 * target.freq.len()) as u64),
                    "every noisy or incoherent capture bin must remain protected"
                );
            }
        }
    }

    #[test]
    fn roadmap_correction_physical_fir_refuses_unassessed_phase_before_artifacts() {
        for mode in [ProcessingMode::MixedPhase, ProcessingMode::PhaseLinear] {
            for missing in ["snr", "low_snr", "coherence", "window_sensitivity"] {
                let (mut chain, mut config, _) = setup();
                config.optimizer.processing_mode = mode.clone();
                let fir = config.optimizer.fir.as_mut().unwrap();
                fir.phase = "kirkeby".into();
                fir.correct_excess_phase = true;
                fir.taps = 1024;
                for driver in chain.drivers.as_mut().unwrap() {
                    let capture = driver.initial_curve.as_mut().unwrap();
                    // Synthetic controlled SNR isolates each refusal cause. The
                    // second driver alone is insufficient: all branches must pass.
                    capture.noise_floor_db = Some(capture.spl.iter().map(|v| v - 40.0).collect());
                    if driver.index == 1 {
                        match missing {
                            "snr" => capture.noise_floor_db = None,
                            "low_snr" => {
                                capture.noise_floor_db =
                                    Some(capture.spl.iter().map(|v| v - 5.0).collect())
                            }
                            "coherence" => capture.coherence = Some(vec![0.2; capture.freq.len()]),
                            "window_sensitivity" => {
                                for (i, phase) in
                                    capture.phase.as_mut().unwrap().iter_mut().enumerate()
                                {
                                    *phase = if i % 2 == 0 { 80.0 } else { -80.0 };
                                }
                            }
                            _ => unreachable!(),
                        }
                    }
                }
                let before = serde_json::to_value(&chain).unwrap();
                let dir = tempfile::tempdir().unwrap();
                let error = generate(&mut chain, &config, 48000.0, Some(dir.path()))
                    .expect_err("shared timing and phase arrays do not establish phase SNR");
                assert!(
                    error
                        .to_string()
                        .contains("per-driver 1 excess-phase assessment"),
                    "{error}"
                );
                assert_eq!(serde_json::to_value(&chain).unwrap(), before);
                assert_eq!(std::fs::read_dir(dir.path()).unwrap().count(), 0);
            }
        }
    }

    #[test]
    fn roadmap_correction_physical_fir_refuses_room_only_upper_phase() {
        for mode in [ProcessingMode::MixedPhase, ProcessingMode::PhaseLinear] {
            let (mut chain, mut config, _) = setup();
            config.optimizer.processing_mode = mode;
            config.optimizer.min_freq = 2000.0;
            config.optimizer.max_freq = 8000.0;
            let fir = config.optimizer.fir.as_mut().unwrap();
            fir.phase = "kirkeby".into();
            fir.correct_excess_phase = true;
            fir.taps = 1024;
            for driver in chain.drivers.as_mut().unwrap() {
                let capture = driver.initial_curve.as_mut().unwrap();
                capture.noise_floor_db = Some(capture.spl.iter().map(|v| v - 40.0).collect());
            }
            let before = serde_json::to_value(&chain).unwrap();
            let dir = tempfile::tempdir().unwrap();
            let error = generate(&mut chain, &config, 48000.0, Some(dir.path()))
                .expect_err("room-only phase and SNR cannot authorize upper-band detail");
            assert!(error.to_string().contains("target policy"), "{error}");
            assert_eq!(serde_json::to_value(&chain).unwrap(), before);
            assert_eq!(std::fs::read_dir(dir.path()).unwrap().count(), 0);
            // The phase refusal does not prohibit separately selected magnitude design.
            config.optimizer.processing_mode = ProcessingMode::PhaseLinear;
            let fir = config.optimizer.fir.as_mut().unwrap();
            fir.phase = "linear".into();
            fir.correct_excess_phase = false;
            generate(&mut chain, &config, 48000.0, Some(dir.path())).unwrap();
        }
    }

    #[test]
    fn roadmap_correction_physical_fir_checks_each_direct_capture() {
        use autoeq_core::direct_sound::{
            AngularCoverage, AveragingMethod, DirectSoundCaptureFacts, DirectSoundEvidence,
            QuasiAnechoicPolicy,
        };
        for missing in ["none", "facts", "angles", "array_angles", "gate", "policy"] {
            let (mut chain, mut config, target) = setup();
            config.optimizer.min_freq = 2000.0;
            config.optimizer.max_freq = 8000.0;
            let fir = config.optimizer.fir.as_mut().unwrap();
            fir.phase = "kirkeby".into();
            fir.correct_excess_phase = true;
            fir.taps = 1024;
            for driver in chain.drivers.as_mut().unwrap() {
                let capture = driver.initial_curve.as_mut().unwrap();
                capture.noise_floor_db = Some(capture.spl.iter().map(|v| v - 40.0).collect());
            }
            let roomeq_model::SpeakerConfig::Topology(topology) =
                config.speakers.get_mut("L").unwrap()
            else {
                unreachable!()
            };
            for (index, driver) in topology.drivers.iter_mut().enumerate() {
                let roomeq_model::MeasurementSource::Single(source) = &mut driver.measurement
                else {
                    unreachable!()
                };
                source.provenance.capture_kind = autoeq_core::ProvenanceCaptureKind::DirectSound;
                // Explicit synthetic acquisition facts, not real capture evidence.
                let mut evidence = DirectSoundEvidence {
                    facts: DirectSoundCaptureFacts {
                        gate_s: Some(0.1),
                        direct_path_m: Some(1.0),
                        first_reflection_path_m: Some(40.0),
                        angular: AngularCoverage {
                            angles_deg: vec![-30.0, 0.0, 30.0],
                        },
                        averaging: AveragingMethod::Stationary,
                        capture_kind: autoeq_core::evidence::CaptureKind::DirectSound,
                        sample_rate_hz: Some(48000.0),
                        ..Default::default()
                    },
                    policy: Some(QuasiAnechoicPolicy::v1()),
                };
                if index == 1 {
                    match missing {
                        "facts" => {
                            source.provenance.capture_kind =
                                autoeq_core::ProvenanceCaptureKind::StationaryIr;
                            continue;
                        }
                        "angles" | "array_angles" => evidence.facts.angular.angles_deg = vec![0.0],
                        "gate" => evidence.facts.gate_s = Some(0.0005),
                        "policy" => evidence.policy = None,
                        "none" => {}
                        _ => unreachable!(),
                    }
                }
                source.provenance.direct_sound = Some(evidence);
            }
            if missing == "array_angles" {
                let roomeq_model::SpeakerConfig::Topology(topology) =
                    config.speakers.remove("L").unwrap()
                else {
                    unreachable!()
                };
                let supported = topology.drivers[0].measurement.clone();
                let unsupported = topology.drivers[1].measurement.clone();
                config.speakers.insert(
                    "L".into(),
                    roomeq_model::SpeakerConfig::Dba(roomeq_model::DBAConfig {
                        name: "synthetic-array".into(),
                        speaker_name: None,
                        front: vec![supported.clone(), supported.clone()],
                        rear: vec![supported, unsupported],
                    }),
                );
                for (driver, name) in chain
                    .drivers
                    .as_mut()
                    .unwrap()
                    .iter_mut()
                    .zip(["Front Array", "Rear Array"])
                {
                    driver.name = name.into();
                }
            }
            let before = serde_json::to_value(&chain).unwrap();
            let dir = tempfile::tempdir().unwrap();
            let result = generate_recorded(&mut chain, &config, 48000.0, Some(dir.path()));
            if missing == "none" {
                let (designed, decisions) = result.unwrap();
                assert_eq!(decisions.len(), 2);
                assert!(
                    decisions
                        .iter()
                        .all(|decision| decision.status == DecisionStatus::Unresolved)
                );
                assert!(decisions.iter().all(|decision| {
                    decision
                        .observed
                        .iter()
                        .any(|value| value.name == "selected_phase_strength" && value.value == 0.0)
                }));
                let replayed = replay(&chain, &target, 48000.0, dir.path()).unwrap();
                assert!(
                    designed
                        .spl
                        .iter()
                        .zip(&replayed.spl)
                        .all(|(a, b)| (a - b).abs() < 1e-5)
                );
                assert_eq!(
                    serde_json::to_value(&chain).unwrap()["target_curve"],
                    before["target_curve"]
                );
                assert_eq!(std::fs::read_dir(dir.path()).unwrap().count(), 2);
            } else {
                let error =
                    result.expect_err("every contributing source needs supported direct evidence");
                let expected = if matches!(missing, "gate" | "policy") {
                    "timing admission failed"
                } else if missing == "array_angles" {
                    "target policy refused source 3"
                } else {
                    "target policy refused source 1"
                };
                assert!(error.to_string().contains(expected), "{missing}: {error}");
                assert_eq!(serde_json::to_value(&chain).unwrap(), before);
                assert_eq!(std::fs::read_dir(dir.path()).unwrap().count(), 0);
            }
        }
    }

    #[test]
    fn roadmap_correction_physical_fir_emits_assessed_phase_and_replays_sidecars() {
        let (mut chain, mut config, target) = setup();
        let fir = config.optimizer.fir.as_mut().unwrap();
        fir.phase = "kirkeby".into();
        fir.correct_excess_phase = true;
        fir.taps = 4096;
        fir.max_boost_db = Some(0.0);
        config.optimizer.max_db = 0.0;
        for driver in chain.drivers.as_mut().unwrap() {
            let capture = driver.initial_curve.as_mut().unwrap();
            capture.noise_floor_db = Some(capture.spl.iter().map(|v| v - 40.0).collect());
            if driver.index == 1 {
                for (phase, f) in capture
                    .phase
                    .as_mut()
                    .unwrap()
                    .iter_mut()
                    .zip(&capture.freq)
                {
                    *phase = 80.0 * (-((f / 85.0).log2() / 0.8).powi(2)).exp();
                }
            }
        }
        let dir = tempfile::tempdir().unwrap();
        let observer = std::sync::Arc::new(std::sync::Mutex::new(None));
        let collection = super::super::types::collect_generic_channel_results(
            vec![Ok((
                "L".into(),
                chain,
                0.0,
                0.0,
                target.clone(),
                target.clone(),
                vec![],
                80.0,
                None,
                None,
                vec![],
                vec![],
                None,
                None,
            ))],
            &config,
            48000.0,
            Some(dir.path()),
            1,
            &observer,
        )
        .unwrap();
        let chain = collection.channel_chains["L"].clone();
        let designed = &collection.curves["L"];
        let decisions = collection.provisional_decisions.clone();
        assert_eq!(decisions.len(), 2);
        assert!(
            decisions
                .iter()
                .all(|record| record.status == DecisionStatus::Applied)
        );
        for record in &decisions {
            record.validate().unwrap();
            assert!(record.frequency_band_hz.is_none());
            assert_eq!(
                record
                    .evidence_refs
                    .iter()
                    .filter(|value| value.starts_with("phase-fir-driver:"))
                    .count(),
                2
            );
            assert!(
                record
                    .observed
                    .iter()
                    .any(|value| value.name == "selected_phase_strength" && value.value > 0.0)
            );
        }
        let restored: ChannelDspChain =
            serde_json::from_value(serde_json::to_value(&chain).unwrap()).unwrap();
        let replayed = replay(&restored, &target, 48000.0, dir.path()).unwrap();
        assert!(
            designed
                .spl
                .iter()
                .zip(&replayed.spl)
                .all(|(a, b)| (a - b).abs() < 1e-5)
        );
        for driver in restored.drivers.as_ref().unwrap() {
            let plugin = driver.plugins.last().unwrap();
            let before = plugin.parameters["joint_target_rms_before_db"]
                .as_f64()
                .unwrap();
            let after = plugin.parameters["joint_target_rms_after_db"]
                .as_f64()
                .unwrap();
            assert!(after < before - 0.01, "{before} -> {after}");
            assert_eq!(plugin.parameters["latency_samples"], 2048);
        }
        for mutation in ["none", "missing_second", "swapped", "removed"] {
            let mut result = crate::test_fixtures::single_channel_room_result("L");
            result.channels.insert("L".into(), restored.clone());
            result.metadata.provisional_decisions = decisions.clone();
            let drivers = result
                .channels
                .get_mut("L")
                .unwrap()
                .drivers
                .as_mut()
                .unwrap();
            match mutation {
                "none" => {}
                "missing_second" => drivers[1].plugins.clear(),
                "swapped" => {
                    let first = drivers[0].plugins.clone();
                    drivers[0].plugins = drivers[1].plugins.clone();
                    drivers[1].plugins = first;
                }
                "removed" => drivers.iter_mut().for_each(|driver| driver.plugins.clear()),
                _ => unreachable!(),
            }
            // Mutation precedes the snapshot: resource/branch checks must catch it.
            let snapshot = crate::final_ledger::processing_snapshot(&result).unwrap();
            crate::final_ledger::finalize_result_ledger(&mut result, &snapshot, 48000.0).unwrap();
            let output = result.to_dsp_chain_output();
            let ledger = output.correction_decisions.unwrap();
            ledger.validate().unwrap();
            let applied = ledger
                .decisions
                .iter()
                .filter(|record| {
                    record.action == DecisionAction::PhaseCorrect
                        && record.status == DecisionStatus::Applied
                        && record.stage == roomeq_model::decision_ledger::DecisionStage::Final
                })
                .count();
            assert_eq!(
                applied,
                if mutation == "none" { 2 } else { 0 },
                "{mutation}"
            );
        }
        let assembled = super::super::assemble_generic_result_with_frequency_samples(
            collection,
            1,
            &config,
            48000.0,
            Some(dir.path()),
            &observer,
            &autoeq_artifacts::FsArtifactStore::new(),
            crate::DEFAULT_FREQUENCY_SAMPLES,
        )
        .unwrap();
        for decision in &decisions {
            assert!(
                assembled
                    .metadata
                    .provisional_decisions
                    .iter()
                    .any(|record| record.decision_id == decision.decision_id),
                "assembly dropped {}",
                decision.decision_id
            );
        }
    }

    #[test]
    fn roadmap_correction_collected_physical_fir_enforces_timing_admission() {
        let (chain, mut config, target) = setup();
        let roomeq_model::SpeakerConfig::Topology(topology) = config.speakers.get_mut("L").unwrap()
        else {
            unreachable!()
        };
        let roomeq_model::MeasurementSource::Single(source) = &mut topology.drivers[1].measurement
        else {
            unreachable!()
        };
        source.provenance.timing_reference_id = Some("different-clock".into());
        let dir = tempfile::tempdir().unwrap();
        let observer = std::sync::Arc::new(std::sync::Mutex::new(None));
        let collected = super::super::types::collect_generic_channel_results(
            vec![Ok((
                "L".into(),
                chain,
                0.0,
                0.0,
                target.clone(),
                target,
                vec![],
                80.0,
                None,
                None,
                vec![],
                vec![],
                None,
                None,
            ))],
            &config,
            48000.0,
            Some(dir.path()),
            1,
            &observer,
        );
        let error = match collected {
            Err(error) => error,
            Ok(_) => panic!("collection bypassed physical-FIR timing admission"),
        };
        assert!(
            error
                .to_string()
                .contains("per_driver FIR timing admission failed"),
            "{error}"
        );
        assert_eq!(std::fs::read_dir(dir.path()).unwrap().count(), 0);
    }

    #[test]
    fn parallel_non_fir_waveform_replays_branch_and_common_gains_once() {
        for rate in [44100.0, 48000.0, 96000.0] {
            let (mut chain, config, target) = setup();
            chain.drivers.as_mut().unwrap()[0].plugins.push(
                roomeq_engine::output::create_gain_plugin(20.0 * 2.0_f64.log10()),
            );
            chain
                .plugins
                .push(roomeq_engine::output::create_gain_plugin(
                    20.0 * 0.5_f64.log10(),
                ));
            let mut result = crate::test_fixtures::single_channel_room_result("L");
            let channel = result.channel_results.get_mut("L").unwrap();
            channel.initial_curve = target;
            channel.biquads.clear();
            result.channels.insert("L".into(), chain);
            super::super::reports::refresh_temporal_ir_evidence(
                &mut result,
                &config,
                rate,
                Path::new("."),
            );
            let post = result.channels["L"].post_ir.as_ref().unwrap();
            // Raw branch amplitudes are each half of the combined reference.
            // The first branch doubles, then the common chain halves the sum.
            assert!(
                (post.amplitude[0] - 0.75).abs() < 1e-9,
                "{rate}: {}",
                post.amplitude[0]
            );
            result
                .channels
                .get_mut("L")
                .unwrap()
                .drivers
                .as_mut()
                .unwrap()[1]
                .initial_curve
                .as_mut()
                .unwrap()
                .phase = None;
            super::super::reports::refresh_temporal_ir_evidence(
                &mut result,
                &config,
                rate,
                Path::new("."),
            );
            assert!(
                result.channels["L"].post_ir.is_none(),
                "missing phase used a PEQ fallback"
            );
            let status = result
                .metadata
                .stage_outcomes
                .iter()
                .find(|stage| stage.stage == "waveform_views")
                .unwrap();
            let missing = status
                .checks
                .iter()
                .find(|check| check.id == "post_ir:L")
                .unwrap();
            assert!(!missing.passed);
            assert!(missing.diagnostic.as_ref().unwrap().contains("speaker/1"));
            let encoded = serde_json::to_value(&result.metadata).unwrap();
            assert!(encoded.to_string().contains("measured phase is required"));
            let driver = &mut result
                .channels
                .get_mut("L")
                .unwrap()
                .drivers
                .as_mut()
                .unwrap()[1];
            let raw = driver.initial_curve.as_mut().unwrap();
            raw.phase = Some(vec![0.0; raw.freq.len()]);
            super::super::reports::refresh_temporal_ir_evidence(
                &mut result,
                &config,
                rate,
                Path::new("."),
            );
            let outcomes: Vec<_> = result
                .metadata
                .stage_outcomes
                .iter()
                .filter(|stage| stage.stage == "waveform_views")
                .collect();
            assert_eq!(outcomes.len(), 1);
            assert!(
                outcomes[0]
                    .checks
                    .iter()
                    .all(|check| check.passed && check.diagnostic.is_none())
            );
        }
    }

    #[test]
    fn physical_replay_waveforms_retain_shared_gain_changes() {
        let (mut chain, config, target) = setup();
        let dir = tempfile::tempdir().unwrap();
        generate(&mut chain, &config, 48000.0, Some(dir.path())).unwrap();
        let mut result = crate::test_fixtures::single_channel_room_result("L");
        result.channel_results.get_mut("L").unwrap().initial_curve = target;
        result.channels.insert("L".into(), chain);
        super::super::reports::refresh_temporal_ir_evidence(
            &mut result,
            &config,
            48000.0,
            dir.path(),
        );
        let baseline = result.channels["L"].post_ir.clone().unwrap();
        let pre = result.channels["L"].pre_ir.clone().unwrap();
        result
            .channels
            .get_mut("L")
            .unwrap()
            .plugins
            .push(roomeq_engine::output::create_gain_plugin(-6.0));
        super::super::reports::refresh_temporal_ir_evidence(
            &mut result,
            &config,
            48000.0,
            dir.path(),
        );
        let after = result.channels["L"].post_ir.as_ref().unwrap();
        let gain = 10.0_f64.powf(-6.0 / 20.0);
        let error = baseline
            .amplitude
            .iter()
            .zip(&after.amplitude)
            .map(|(before, after)| (after - gain * before).abs())
            .fold(0.0_f64, f64::max);
        assert!(
            error < 1e-9,
            "independent normalization hid shared gain: {error}"
        );
        assert_eq!(
            pre.amplitude,
            result.channels["L"].pre_ir.as_ref().unwrap().amplitude
        );
    }

    #[test]
    fn roadmap_correction_array_waveforms_check_all_source_timing() {
        for dba in [false, true] {
            let (chain, mut config, target) = setup();
            let roomeq_model::SpeakerConfig::Topology(topology) =
                config.speakers.remove("L").unwrap()
            else {
                unreachable!()
            };
            let sources: Vec<_> = topology
                .drivers
                .into_iter()
                .map(|driver| driver.measurement)
                .collect();
            let curves: Vec<Curve> = chain
                .drivers
                .unwrap()
                .into_iter()
                .map(|driver| driver.initial_curve.unwrap().into())
                .collect();
            let (speaker, chain) = if dba {
                (
                    roomeq_model::SpeakerConfig::Dba(roomeq_model::DBAConfig {
                        name: "fixture".into(),
                        speaker_name: None,
                        // Multiple source declarations feed each emitted aggregate.
                        front: vec![sources[0].clone(), sources[0].clone()],
                        rear: vec![sources[1].clone(), sources[1].clone()],
                    }),
                    roomeq_engine::output::build_dba_dsp_chain_with_curves(
                        "L",
                        &[0.0, -6.0],
                        &[0.0, 1.0],
                        &[],
                        None,
                        None,
                        Some(&curves),
                    ),
                )
            } else {
                (
                    roomeq_model::SpeakerConfig::Cardioid(Box::new(roomeq_model::CardioidConfig {
                        name: "fixture".into(),
                        speaker_name: None,
                        front: sources[0].clone(),
                        rear: sources[1].clone(),
                        separation_meters: 0.3,
                    })),
                    roomeq_engine::output::build_cardioid_dsp_chain_with_curves(
                        "L",
                        &[0.0, -6.0],
                        &[0.0, 1.0],
                        &[],
                        None,
                        None,
                        Some(&curves),
                    ),
                )
            };
            config.speakers.insert("L".into(), speaker);
            let mut result = crate::test_fixtures::single_channel_room_result("L");
            result.channel_results.get_mut("L").unwrap().initial_curve = target;
            result.channels.insert("L".into(), chain);
            super::super::reports::refresh_temporal_ir_evidence(
                &mut result,
                &config,
                48000.0,
                Path::new("."),
            );
            assert!(result.channels["L"].post_ir.is_some(), "dba={dba}");
            let mut invalid = config.clone();
            let source = match invalid.speakers.get_mut("L").unwrap() {
                roomeq_model::SpeakerConfig::Dba(group) => &mut group.rear[1],
                roomeq_model::SpeakerConfig::Cardioid(group) => &mut group.rear,
                _ => unreachable!(),
            };
            let roomeq_model::MeasurementSource::Single(source) = source else {
                unreachable!()
            };
            source.provenance.timing_reference_id = Some("different-clock".into());
            super::super::reports::refresh_temporal_ir_evidence(
                &mut result,
                &invalid,
                48000.0,
                Path::new("."),
            );
            assert!(
                result.channels["L"].post_ir.is_none(),
                "unchecked source, dba={dba}"
            );
            result
                .channels
                .get_mut("L")
                .unwrap()
                .drivers
                .as_mut()
                .unwrap()[1]
                .name = "wrong-array".into();
            super::super::reports::refresh_temporal_ir_evidence(
                &mut result,
                &config,
                48000.0,
                Path::new("."),
            );
            assert!(
                result.channels["L"].post_ir.is_none(),
                "unchecked branch, dba={dba}"
            );
        }
    }

    #[test]
    fn roadmap_correction_waveform_timing_resolves_system_output_aliases() {
        for sub_output in [false, true] {
            let (chain, mut config, target) = setup();
            let source = config.speakers.remove("L").unwrap();
            config.speakers.insert("source-group".into(), source);
            let mut system = roomeq_model::SystemConfig::default();
            if sub_output {
                system.subwoofers = Some(roomeq_model::SubwooferSystemConfig {
                    config: roomeq_model::SubwooferStrategy::Single,
                    crossover: None,
                    routing: Default::default(),
                    outputs: vec![roomeq_model::SubwooferOutput {
                        id: "L".into(),
                        speaker: "source-group".into(),
                    }],
                });
            } else {
                system.speakers.insert("L".into(), "source-group".into());
            }
            config.system = Some(system);
            let mut result = crate::test_fixtures::single_channel_room_result("L");
            result.channel_results.get_mut("L").unwrap().initial_curve = target;
            result.channels.insert("L".into(), chain);
            super::super::reports::refresh_temporal_ir_evidence(
                &mut result,
                &config,
                48000.0,
                Path::new("."),
            );
            assert!(
                result.channels["L"].post_ir.is_some(),
                "sub_output={sub_output}"
            );
            if sub_output {
                config
                    .system
                    .as_mut()
                    .unwrap()
                    .speakers
                    .insert("L".into(), "different-source".into());
                super::super::reports::refresh_temporal_ir_evidence(
                    &mut result,
                    &config,
                    48000.0,
                    Path::new("."),
                );
                assert!(result.channels["L"].post_ir.is_none());
                assert!(
                    result
                        .metadata
                        .stage_outcomes
                        .iter()
                        .flat_map(|stage| &stage.checks)
                        .any(|check| check
                            .diagnostic
                            .as_ref()
                            .is_some_and(|reason| reason.contains("ambiguous source mapping")))
                );
            }
        }
    }

    #[test]
    fn physical_sub_outputs_resolve_all_captures_for_parallel_waveforms() {
        let (mut chain, mut config, target) = setup();
        let roomeq_model::SpeakerConfig::Topology(topology) = config.speakers.remove("L").unwrap()
        else {
            unreachable!()
        };
        for (index, driver) in topology.drivers.into_iter().enumerate() {
            config.speakers.insert(
                format!("subs_{index}"),
                roomeq_model::SpeakerConfig::Single(driver.measurement),
            );
        }
        config.system = Some(roomeq_model::SystemConfig {
            model: roomeq_model::SystemModel::Stereo,
            subwoofers: Some(roomeq_model::SubwooferSystemConfig {
                config: roomeq_model::SubwooferStrategy::Mso,
                crossover: None,
                routing: Default::default(),
                outputs: (0..2)
                    .map(|index| roomeq_model::SubwooferOutput {
                        id: format!("Sub{}", index + 1),
                        speaker: format!("subs_{index}"),
                    })
                    .collect(),
            }),
            ..Default::default()
        });
        chain.channel = "Sub1".into();
        for driver in chain.drivers.as_mut().unwrap() {
            driver.name = format!("Sub{}", driver.index + 1);
        }
        let sources =
            super::super::parallel_timing::validated_sources(&chain, &config, &target).unwrap();
        assert_eq!(sources.len(), 2);
        let mut result = crate::test_fixtures::single_channel_room_result("Sub1");
        result
            .channel_results
            .get_mut("Sub1")
            .unwrap()
            .initial_curve = target.clone();
        result.channels.insert("Sub1".into(), chain.clone());
        super::super::reports::refresh_temporal_ir_evidence(
            &mut result,
            &config,
            48000.0,
            Path::new("."),
        );
        assert!(result.channels["Sub1"].pre_ir.is_some());
        assert!(result.channels["Sub1"].post_ir.is_some());

        for fault in [
            "identity",
            "index",
            "missing_driver",
            "duplicate_output",
            "missing_source",
            "timing",
        ] {
            let mut invalid_chain = chain.clone();
            let mut invalid_config = config.clone();
            match fault {
                "identity" => invalid_chain.drivers.as_mut().unwrap()[1].name = "wrong".into(),
                "index" => invalid_chain.drivers.as_mut().unwrap()[1].index = 0,
                "missing_driver" => {
                    invalid_chain.drivers.as_mut().unwrap().pop();
                }
                "duplicate_output" => {
                    invalid_config
                        .system
                        .as_mut()
                        .unwrap()
                        .subwoofers
                        .as_mut()
                        .unwrap()
                        .outputs[1]
                        .id = "Sub1".into();
                }
                "missing_source" => {
                    invalid_config.speakers.remove("subs_1");
                }
                "timing" => {
                    let roomeq_model::SpeakerConfig::Single(
                        roomeq_model::MeasurementSource::Single(source),
                    ) = invalid_config.speakers.get_mut("subs_1").unwrap()
                    else {
                        unreachable!()
                    };
                    source.provenance.timing_reference_id = Some("different-clock".into());
                }
                _ => unreachable!(),
            }
            assert!(
                super::super::parallel_timing::validate(&invalid_chain, &invalid_config, &target)
                    .is_err(),
                "{fault}"
            );
        }
    }

    #[test]
    fn roadmap_correction_dba_timing_uses_routed_output_ids() {
        let (mut chain, mut config, target) = setup();
        let roomeq_model::SpeakerConfig::Topology(topology) = config.speakers.remove("L").unwrap()
        else {
            unreachable!()
        };
        config.speakers.insert(
            "lfe".into(),
            roomeq_model::SpeakerConfig::Dba(roomeq_model::DBAConfig {
                name: "fixture".into(),
                speaker_name: None,
                front: vec![topology.drivers[0].measurement.clone()],
                rear: vec![topology.drivers[1].measurement.clone()],
            }),
        );
        config.system = Some(roomeq_model::SystemConfig {
            subwoofers: Some(roomeq_model::SubwooferSystemConfig {
                config: roomeq_model::SubwooferStrategy::Dba,
                crossover: None,
                routing: Default::default(),
                outputs: ["Sub1", "Sub2"]
                    .into_iter()
                    .map(|id| roomeq_model::SubwooferOutput {
                        id: id.into(),
                        speaker: "lfe".into(),
                    })
                    .collect(),
            }),
            ..Default::default()
        });
        chain.channel = "Sub1".into();
        for (driver, id) in chain
            .drivers
            .as_mut()
            .unwrap()
            .iter_mut()
            .zip(["Sub1", "Sub2"])
        {
            driver.name = id.into();
        }
        assert!(super::super::parallel_timing::validate(&chain, &config, &target).is_ok());
        chain.drivers.as_mut().unwrap()[1].name = "Rear Array".into();
        let error = super::super::parallel_timing::validate(&chain, &config, &target).unwrap_err();
        assert!(error.contains("array branch identity mismatch"), "{error}");
    }

    #[test]
    fn roadmap_correction_parallel_waveforms_require_common_capture_timing() {
        for physical_fir in [false, true] {
            let (mut chain, config, target) = setup();
            let dir = tempfile::tempdir().unwrap();
            if physical_fir {
                generate(&mut chain, &config, 48000.0, Some(dir.path())).unwrap();
            }
            let mut result = crate::test_fixtures::single_channel_room_result("L");
            result.channel_results.get_mut("L").unwrap().initial_curve = target;
            result.channels.insert("L".into(), chain);
            for fault in [
                "missing_mapping",
                "reference",
                "seat",
                "band",
                "kind",
                "driver_id",
            ] {
                super::super::reports::refresh_temporal_ir_evidence(
                    &mut result,
                    &config,
                    48000.0,
                    dir.path(),
                );
                assert!(
                    result.channels["L"].post_ir.is_some(),
                    "valid {physical_fir}"
                );
                let mut invalid = config.clone();
                if fault == "missing_mapping" {
                    invalid.speakers.clear();
                } else {
                    let roomeq_model::SpeakerConfig::Topology(topology) =
                        invalid.speakers.get_mut("L").unwrap()
                    else {
                        unreachable!()
                    };
                    if fault == "driver_id" {
                        topology.drivers[1].id = "wrong-driver".into();
                    } else {
                        let roomeq_model::MeasurementSource::Single(source) =
                            &mut topology.drivers[1].measurement
                        else {
                            unreachable!()
                        };
                        match fault {
                            "reference" => {
                                source.provenance.timing_reference_id =
                                    Some("different-clock".into())
                            }
                            "seat" => {
                                let autoeq_core::MeasurementRef::Inline(measurement) =
                                    &mut source.measurement
                                else {
                                    unreachable!()
                                };
                                measurement.name = Some("different-seat".into());
                            }
                            "band" => source.provenance.valid_band_hz = Some([20.0, 200.0]),
                            "kind" => {
                                source.provenance.capture_kind =
                                    autoeq_core::ProvenanceCaptureKind::SimulatedBackend
                            }
                            _ => unreachable!(),
                        }
                    }
                }
                super::super::reports::refresh_temporal_ir_evidence(
                    &mut result,
                    &invalid,
                    48000.0,
                    dir.path(),
                );
                assert!(
                    result.channels["L"].post_ir.is_none(),
                    "{physical_fir}: {fault}"
                );
                let check = result
                    .metadata
                    .stage_outcomes
                    .iter()
                    .find(|stage| stage.stage == "waveform_views")
                    .unwrap()
                    .checks
                    .iter()
                    .find(|check| check.id == "post_ir:L")
                    .unwrap();
                assert!(!check.passed);
                assert!(
                    check
                        .diagnostic
                        .as_ref()
                        .unwrap()
                        .contains("parallel waveform timing unavailable")
                );
                if physical_fir {
                    assert!(result.channels["L"].fir_temporal_masking.is_some());
                    assert!(result.channels["L"].pre_ir.is_none());
                }
            }
        }
    }

    #[test]
    fn physical_replay_report_clears_post_ir_when_branch_phase_is_removed() {
        let (mut chain, config, target) = setup();
        let dir = tempfile::tempdir().unwrap();
        generate(&mut chain, &config, 48000.0, Some(dir.path())).unwrap();
        let mut result = crate::test_fixtures::single_channel_room_result("L");
        result.channel_results.get_mut("L").unwrap().initial_curve = target;
        result.channels.insert("L".into(), chain);
        super::super::reports::refresh_temporal_ir_evidence(
            &mut result,
            &config,
            48000.0,
            dir.path(),
        );
        assert!(result.channels["L"].post_ir.is_some());
        result
            .channels
            .get_mut("L")
            .unwrap()
            .drivers
            .as_mut()
            .unwrap()[1]
            .initial_curve
            .as_mut()
            .unwrap()
            .phase = None;
        super::super::reports::refresh_temporal_ir_evidence(
            &mut result,
            &config,
            48000.0,
            dir.path(),
        );
        assert!(
            result.channels["L"].post_ir.is_none(),
            "stale coherent prediction survived"
        );
        // The independently supplied combined reference still has measured phase.
        assert!(result.channels["L"].pre_ir.is_some());
        // Kernel-only temporal metrics do not claim acoustic summation evidence.
        assert!(result.channels["L"].fir_temporal_masking.is_some());
    }

    #[test]
    fn physical_replay_rejects_missing_drivers_and_invalid_grid() {
        let (mut chain, _, mut grid) = setup();
        chain.drivers = None;
        assert!(replay(&chain, &grid, 48000.0, Path::new(".")).is_err());
        let (chain, _, _) = setup();
        grid.freq[1] = grid.freq[0];
        assert!(replay(&chain, &grid, 48000.0, Path::new(".")).is_err());
        let (_, _, grid) = setup();
        for rate in [f64::NAN, 0.0, 8000.0] {
            assert!(replay(&chain, &grid, rate, Path::new(".")).is_err());
        }
    }

    #[test]
    fn physical_replay_preserves_observed_relative_phase() {
        let (mut chain, _, grid) = setup();
        chain.drivers.as_mut().unwrap()[1]
            .initial_curve
            .as_mut()
            .unwrap()
            .phase
            .as_mut()
            .unwrap()
            .fill(120.0);
        for rate in [44100.0, 48000.0, 96000.0] {
            let result = replay(&chain, &grid, rate, Path::new(".")).unwrap();
            // Two equal sources separated by 120 degrees sum to one source's
            // magnitude and 60 degrees, not the +6 dB zero-phase prediction.
            let expected = 80.0 - 20.0 * 2.0_f64.log10();
            for (&spl, &phase) in result.spl.iter().zip(result.phase.as_ref().unwrap()) {
                assert!((spl - expected).abs() < 1e-9);
                assert!((phase - 60.0).abs() < 1e-9);
            }
        }
    }

    #[test]
    fn routed_split_fir_cannot_move_outside_its_owning_block() {
        let (mut chain, config, _) = setup();
        let mut convolution = roomeq_engine::output::create_convolution_plugin("common.wav");
        convolution.parameters["room_eq_stage"] = serde_json::json!("post_route");
        let mut split = convolution.clone();
        split.plugin_type = "band_split".into();
        chain.plugins = vec![split, convolution];
        let mut result = crate::test_fixtures::single_channel_room_result("L");
        result.channels.insert("L".into(), chain.clone());
        let store = autoeq_artifacts::MemoryArtifactStore::default();
        let error =
            distribute_routed_firs(&mut result, &config, Path::new("."), &store).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("whole-block physical realization")
        );
        assert_eq!(result.channels["L"].plugins.len(), chain.plugins.len());
        assert!(
            result.channels["L"]
                .drivers
                .as_ref()
                .unwrap()
                .iter()
                .all(|driver| driver.plugins.is_empty())
        );
    }

    #[test]
    fn routed_shared_kernel_preserves_each_source_and_physical_branch() {
        use autoeq_artifacts::{ArtifactStore, FsArtifactStore, MemoryArtifactStore};
        let (mut chain, config, grid) = setup();
        let dir = tempfile::tempdir().unwrap();
        let taps = [0.0, 0.25, 0.5, 0.25];
        math_audio_iir_fir::save_fir_to_wav(&taps, 48000, &dir.path().join("common.wav")).unwrap();
        let mut common = roomeq_engine::output::create_convolution_plugin("common.wav");
        common.parameters["room_eq_stage"] = serde_json::json!("post_route");
        common.parameters["latency_samples"] = serde_json::json!(2);
        chain.plugins.push(common.clone());
        for driver in chain.drivers.as_mut().unwrap() {
            driver
                .plugins
                .push(roomeq_engine::output::create_gain_plugin(
                    -3.0 * driver.index as f64,
                ));
        }
        let before = chain.clone();
        let mut result = crate::test_fixtures::single_channel_room_result("L");
        result.channels.insert("L".into(), chain);
        distribute_routed_firs(&mut result, &config, dir.path(), &FsArtifactStore::new()).unwrap();
        let after = &result.channels["L"];
        assert!(after.plugins.is_empty());
        let drivers = after.drivers.as_ref().unwrap();
        assert_ne!(
            drivers[0].plugins.last().unwrap().parameters["ir_file"],
            drivers[1].plugins.last().unwrap().parameters["ir_file"]
        );
        // Distinct logical source gains and phases exercise each route column.
        // Equality on every physical output also proves equality of any acoustic sum.
        for (gain_db, phase_deg) in [(0.0, 0.0), (-7.0, 117.0)] {
            let mut source = grid.clone();
            source.spl += gain_db;
            source.phase.as_mut().unwrap().fill(phase_deg);
            for i in 0..drivers.len() {
                let render = |parent: &ChannelDspChain| {
                    let mut branch = parent.clone();
                    branch.plugins = parent.drivers.as_ref().unwrap()[i].plugins.clone();
                    branch.plugins.extend(parent.plugins.clone());
                    branch.drivers = None;
                    crate::ctc::apply_channel_dsp_chain_to_curve_with_sidecar_dir(
                        &branch,
                        &source,
                        48000.0,
                        dir.path(),
                    )
                    .unwrap()
                };
                let a = render(&before);
                let b = render(after);
                for (a, b) in a.spl.iter().zip(b.spl.iter()) {
                    assert!((a - b).abs() < 1e-9);
                }
                for (a, b) in a.phase.unwrap().iter().zip(b.phase.unwrap().iter()) {
                    assert!((a - b).abs() < 1e-9);
                }
                assert_eq!(
                    drivers[i].plugins.last().unwrap().parameters["latency_samples"],
                    2
                );
            }
        }
        // Memory stores must receive actual bytes, and pre-route source filters
        // must remain on their owner instead of leaking into other input routes.
        let memory = MemoryArtifactStore::default();
        memory
            .write(
                &dir.path().join("common.wav"),
                &std::fs::read(dir.path().join("common.wav")).unwrap(),
            )
            .unwrap();
        let mut before = before;
        common.parameters["room_eq_stage"] = serde_json::json!("pre_route");
        before.plugins.insert(0, common);
        result.channels.insert("L".into(), before);
        distribute_routed_firs(&mut result, &config, dir.path(), &memory).unwrap();
        assert_eq!(result.channels["L"].plugins.len(), 1);
        assert_eq!(
            result.channels["L"].plugins[0].parameters["room_eq_stage"],
            "pre_route"
        );
        for driver in result.channels["L"].drivers.as_ref().unwrap() {
            let name = driver.plugins.last().unwrap().parameters["ir_file"]
                .as_str()
                .unwrap();
            assert!(memory.read(&dir.path().join(name)).unwrap().is_some());
        }
    }

    #[test]
    fn per_driver_sidecars_replay_the_physical_sum_and_have_unique_paths() {
        let (mut chain, config, target) = setup();
        let dir = tempfile::tempdir().unwrap();
        let rendered = generate(&mut chain, &config, 48000.0, Some(dir.path())).unwrap();
        assert!(chain.plugins.iter().all(|p| p.plugin_type != "convolution"));
        let paths: Vec<_> = chain
            .drivers
            .as_ref()
            .unwrap()
            .iter()
            .map(|driver| {
                let p = driver.plugins.last().unwrap();
                assert_eq!(p.plugin_type, "convolution");
                assert_eq!(p.parameters["latency_samples"], 2048);
                let path = p.parameters["ir_file"].as_str().unwrap().to_string();
                assert!(dir.path().join(&path).exists());
                path
            })
            .collect();
        assert_ne!(paths[0], paths[1]);
        let replayed = replay(&chain, &target, 48000.0, dir.path()).unwrap();
        for (a, b) in rendered.spl.iter().zip(&replayed.spl) {
            assert!((a - b).abs() < 1e-4);
        }
        // A physical FIR must not be silently replaced by a common filter.
        assert!(has_physical_fir(&chain));
    }
    #[test]
    fn per_driver_missing_artifact_directory_is_an_error_not_a_dangling_export() {
        let (mut chain, config, _) = setup();
        let before = serde_json::to_value(&chain).unwrap();
        assert!(generate(&mut chain, &config, 48000.0, None).is_err());
        assert_eq!(serde_json::to_value(&chain).unwrap(), before);
    }
    #[test]
    fn per_driver_missing_capture_fails_before_deploying_any_filter() {
        let (mut chain, config, _) = setup();
        chain.drivers.as_mut().unwrap()[1].initial_curve = None;
        let dir = tempfile::tempdir().unwrap();
        assert!(generate(&mut chain, &config, 48000.0, Some(dir.path())).is_err());
        assert!(chain.drivers.unwrap().iter().all(|d| d.plugins.is_empty()));
    }
}
