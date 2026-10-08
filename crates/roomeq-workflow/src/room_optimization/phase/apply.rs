use super::super::types::ChannelOptimizationResult;
use super::super::*;
use super::misc::compute_phase_alignment_delay_schedule;
use super::misc::convolve;
use super::sync::sync_reported_phase_adjustment;

/// Preserve a refused phase attempt without claiming that any processing was applied.
fn phase_refusal(
    name: &str,
    gate: &roomeq_model::eligibility::ChannelOperationGate,
    reason: &str,
) -> Vec<roomeq_model::decision_ledger::DecisionRecord> {
    use roomeq_model::decision_ledger::{
        DECISION_LEDGER_VERSION, DecisionAction, DecisionRecord, DecisionStage, DecisionStatus,
    };
    let records: Vec<_> = gate
        .records
        .iter()
        .filter(|record| {
            record.operation
                == roomeq_model::eligibility::CorrectionOperation::ExcessPhaseCorrection
        })
        .collect();
    // A missing gate record must still explain the refused request. Do not invent
    // a frequency interval, seat identity, or calibration for that case.
    let scopes: Vec<_> = if records.is_empty() {
        vec![None]
    } else {
        records.into_iter().map(Some).collect()
    };
    scopes
        .into_iter()
        .enumerate()
        .map(|(index, record)| DecisionRecord {
            decision_id: format!("phase-refused-{name}-{index}-{reason}"),
            ledger_version: DECISION_LEDGER_VERSION.to_owned(),
            stage: DecisionStage::Provisional,
            logical_input: name.to_owned(),
            physical_output: name.to_owned(),
            measurement_refs: vec![gate.measurement_id.clone()],
            seat_refs: record.map(|r| r.seat_ids.clone()).unwrap_or_default(),
            // This is the evaluated evidence scope, not the support of an applied FIR.
            frequency_band_hz: record.and_then(|r| r.band_hz),
            filter_center_hz: None,
            action: DecisionAction::PhaseCorrect,
            status: if matches!(
                reason,
                "phase_evidence_refused" | "phase_data_missing" | "phase_assessment_refused"
            ) {
                DecisionStatus::InsufficientEvidence
            } else {
                DecisionStatus::Unresolved
            },
            reason_codes: vec![reason.to_owned()],
            observed: Vec::new(),
            limits: Vec::new(),
            evidence_refs: record
                .map(|r| {
                    let mut refs = r.evidence_refs.clone();
                    refs.push(r.record_id.clone());
                    refs
                })
                .unwrap_or_default(),
            confidence: roomeq_model::AssessmentConfidence::Unknown,
            related_decision_ids: Vec::new(),
            supersedes_ids: Vec::new(),
            final_graph_identity: None,
        })
        .collect()
}

/// Attach finite observed demand and its configured limit to a refused attempt.
fn phase_budget_refusal(
    name: &str,
    gate: &roomeq_model::eligibility::ChannelOperationGate,
    reason: &str,
    quantity: &str,
    observed: f64,
    limit: f64,
    unit: &str,
) -> Vec<roomeq_model::decision_ledger::DecisionRecord> {
    use roomeq_model::decision_ledger::ObservedQuantity;
    let mut records = phase_refusal(name, gate, reason);
    for record in &mut records {
        if observed.is_finite() {
            record.observed.push(ObservedQuantity {
                name: quantity.to_owned(),
                value: observed,
                unit: unit.to_owned(),
            });
        } else {
            record
                .reason_codes
                .push("phase_nonfinite_observation".to_owned());
        }
        if limit.is_finite() && limit >= 0.0 {
            record.limits.push(ObservedQuantity {
                name: quantity.to_owned(),
                value: limit,
                unit: unit.to_owned(),
            });
        } else {
            record.reason_codes.push("phase_invalid_budget".to_owned());
        }
    }
    records
}

/// Apply standalone phase correction to a channel (rePhase-style).
///
/// Generates a phase-only FIR from the measurement's excess phase and appends it
/// to the channel's DSP chain. If the channel already has a magnitude FIR, the
/// two are convolved together so `fir_coeffs` remains a single filter for IR
/// computation.
///
/// The channel's intake `gate` must authorize
/// [`CorrectionOperation::ExcessPhaseCorrection`](roomeq_model::eligibility::CorrectionOperation::ExcessPhaseCorrection):
/// phase data alone never authorizes the operation. Past the gate, the
/// excess-phase assessment must support every authorized band on
/// trustworthy phase/SNR evidence, and the target chain must not limit
/// the proposed detail band. Refused channels keep their magnitude
/// correction and return refusal records for the workflow decision ledger.
/// Successful application returns provisional phase/target decisions with realized
/// magnitude and delay observations. Only final workflow reconciliation may bind them.
#[allow(clippy::too_many_arguments)]
pub(in super::super) fn apply_phase_correction(
    name: &str,
    ch: &mut ChannelOptimizationResult,
    chain: &mut roomeq_model::ChannelDspChain,
    config: &roomeq_model::MixedPhaseSerdeConfig,
    sample_rate: f64,
    output_dir: Option<&Path>,
    gate: &roomeq_model::eligibility::ChannelOperationGate,
    target_response: Option<&roomeq_model::TargetResponseConfig>,
    schroeder_hz: Option<f64>,
) -> Vec<roomeq_model::decision_ledger::DecisionRecord> {
    if !gate.authorizes(roomeq_model::eligibility::CorrectionOperation::ExcessPhaseCorrection) {
        warn!(
            " Phase correction refused '{}': evidence does not authorize excess-phase work ({:?})",
            name,
            gate.records
                .iter()
                .filter(|record| record.operation
                    == roomeq_model::eligibility::CorrectionOperation::ExcessPhaseCorrection)
                .flat_map(|record| record.observations.iter())
                .collect::<Vec<_>>()
        );
        return phase_refusal(name, gate, "phase_evidence_refused");
    }
    let phase = match ch.initial_curve.phase.as_ref() {
        Some(p) if !p.is_empty() => p,
        _ => return phase_refusal(name, gate, "phase_data_missing"),
    };
    let _ = phase; // used via initial_curve below
    // Assessment + target-chain enforcement on the actual proposal bands.
    // Unsupported or limited detail work stops the phase action while the
    // independently supported magnitude correction stays in place.
    let target_chain = roomeq_model::target_transition::TargetChain::from_target_config(
        target_response,
        schroeder_hz,
        vec![gate.measurement_id.clone()],
    );
    let supported = match super::assess_phase_support(
        name,
        &ch.initial_curve,
        gate,
        &config.assessment,
        &target_chain,
        sample_rate,
    ) {
        Ok(supported) => supported,
        Err(reason) => {
            warn!(" {reason}");
            let mut refusals = phase_refusal(name, gate, "phase_assessment_refused");
            for refusal in &mut refusals {
                // Preserve the assessor's exact explanation as well as the stable
                // category. No diagnosis is inferred from the final curve.
                refusal.reason_codes.push(reason.clone());
            }
            return refusals;
        }
    };
    debug!(
        " Phase correction '{}': assessment supports {} band(s): {:?}",
        name,
        supported.bands_hz.len(),
        supported.bands_hz
    );

    let mp_config = roomeq_engine::mixed_phase::MixedPhaseConfig {
        max_fir_length_ms: config.max_fir_length_ms,
        pre_ringing_threshold_db: config.pre_ringing_threshold_db,
        min_spatial_depth: config.min_spatial_depth,
        phase_smoothing_octaves: config.phase_smoothing_octaves,
    };

    // The assessment has already removed bulk delay, checked noise/window
    // sensitivity, and tapered the target to zero outside supported bands.
    // The FIR API accepts residual phase and negates it, so reverse the
    // correction target's sign here rather than recomputing full-band phase.
    let residual = supported.correction_phase_deg.mapv(|phase| -phase);
    let phase_fir = roomeq_engine::mixed_phase::generate_excess_phase_fir(
        &supported.frequencies_hz,
        &residual,
        &mp_config,
        sample_rate,
    );
    let mut mixed_phase_report =
        roomeq_engine::mixed_phase::MixedPhaseCorrectionReport::from_residual(
            supported.estimated_delay_ms,
            phase_fir.len(),
            &residual,
        );
    // generate_excess_phase_fir centers the zero-time sample at len / 2.
    // Acoustic bulk delay was removed before design; it is not this latency.
    let causal_center_delay_ms = (phase_fir.len() / 2) as f64 * 1000.0 / sample_rate;
    mixed_phase_report.causal_center_delay_ms = Some(causal_center_delay_ms);

    let phase_response = roomeq_engine::response::compute_fir_complex_response(
        &phase_fir,
        &ch.final_curve.freq,
        sample_rate,
    );
    let passband_floor = ch
        .initial_curve
        .spl
        .iter()
        .copied()
        .filter(|level| level.is_finite())
        .fold(f64::NEG_INFINITY, f64::max)
        - 30.0;
    let max_magnitude_deviation_db = phase_response
        .iter()
        .zip(&ch.initial_curve.spl)
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
        " Phase correction '{}': max passband magnitude deviation {:.3} dB",
        name, max_magnitude_deviation_db
    );
    if max_magnitude_deviation_db > 0.5 {
        warn!(
            " Phase correction skipped '{}': phase-only FIR magnitude deviation {:.2} dB exceeds 0.50 dB",
            name, max_magnitude_deviation_db
        );
        return phase_budget_refusal(
            name,
            gate,
            "phase_magnitude_budget_exceeded",
            "phase_fir_magnitude_deviation",
            max_magnitude_deviation_db,
            0.5,
            "db",
        );
    }
    // Bound the delay added by this realization, not the propagation delay
    // removed during assessment. Other stages and backend buffering are separate.
    if let Some(budget_ms) = config.max_correction_latency_ms
        && (!budget_ms.is_finite() || budget_ms < 0.0 || causal_center_delay_ms > budget_ms)
    {
        warn!(
            " Phase correction skipped '{}': causal FIR delay {:.2} ms violates the {:.2} ms latency budget",
            name, causal_center_delay_ms, budget_ms
        );
        return phase_budget_refusal(
            name,
            gate,
            "phase_latency_budget_exceeded",
            "phase_fir_causal_center_delay",
            causal_center_delay_ms,
            budget_ms,
            "ms",
        );
    }

    // Save phase FIR WAV and add convolution plugin
    let mut filename = autoeq_artifacts::roomeq::convolution_artifact_filename(
        name,
        autoeq_artifacts::roomeq::ConvolutionArtifactKind::PhaseCorrection,
        sample_rate,
    );
    if let Some(out_dir) = output_dir {
        let reserved = autoeq_artifacts::roomeq::reserve_convolution_artifact_path(
            out_dir,
            name,
            autoeq_artifacts::roomeq::ConvolutionArtifactKind::PhaseCorrection,
            sample_rate,
        );
        filename = reserved.0;
        let wav_path = reserved.1;
        if let Err(e) =
            math_audio_iir_fir::save_fir_to_wav(&phase_fir, sample_rate as u32, &wav_path)
        {
            warn!("Failed to save phase correction FIR for {}: {}", name, e);
            return phase_refusal(name, gate, "phase_artifact_write_failed");
        } else {
            info!("  Saved phase correction FIR to {}", wav_path.display());
        }
    }
    chain.plugins.push(
        roomeq_engine::output::create_mixed_phase_convolution_plugin(
            &filename,
            &mixed_phase_report,
        ),
    );

    ch.final_curve =
        roomeq_engine::response::apply_complex_response(&ch.final_curve, &phase_response);
    chain.final_curve = Some((&ch.final_curve).into());

    // Combine with existing FIR for IR computation (convolve the two)
    if let Some(ref existing) = ch.fir_coeffs {
        ch.fir_coeffs = Some(convolve(existing, &phase_fir));
    } else {
        ch.fir_coeffs = Some(phase_fir);
    }
    let mut decisions = supported.target_decisions;
    for decision in &mut decisions {
        use roomeq_model::decision_ledger::ObservedQuantity;
        decision.reason_codes.push("phase_fir_realized".to_owned());
        decision.evidence_refs.push(format!("phase-fir:{filename}"));
        for (quantity, value, unit) in [
            (
                "phase_fir_magnitude_deviation",
                max_magnitude_deviation_db,
                "db",
            ),
            (
                "phase_fir_causal_center_delay",
                causal_center_delay_ms,
                "ms",
            ),
        ] {
            decision.observed.push(ObservedQuantity {
                name: quantity.to_owned(),
                value,
                unit: unit.to_owned(),
            });
        }
        decision.limits.push(ObservedQuantity {
            name: "phase_fir_magnitude_deviation".to_owned(),
            value: 0.5,
            unit: "db".to_owned(),
        });
        if let Some(value) = config.max_correction_latency_ms {
            decision.limits.push(ObservedQuantity {
                name: "phase_fir_causal_center_delay".to_owned(),
                value,
                unit: "ms".to_owned(),
            });
        }
    }
    decisions
}

pub(in super::super) fn apply_phase_alignment_delay_schedule(
    phase_alignment_results: &HashMap<String, (f64, bool, String)>,
    channel_results: &mut HashMap<String, ChannelOptimizationResult>,
    channel_chains: &mut HashMap<String, ChannelDspChain>,
    sample_rate: f64,
) -> HashMap<String, f64> {
    let schedule = compute_phase_alignment_delay_schedule(phase_alignment_results);

    for (channel_name, delay_ms) in &schedule {
        let applied = if let Some(chain) = channel_chains.get_mut(channel_name.as_str()) {
            output::add_delay_plugin(chain, *delay_ms);
            // Tag the stage so timing diagnostics can tell intentional
            // crossover phase-alignment delays apart from the arrival-time
            // alignment they would otherwise masquerade as.
            if let Some(plugin) = chain.plugins.first_mut() {
                plugin.parameters["label"] =
                    serde_json::Value::String("room_eq_phase_alignment".to_string());
                plugin.parameters["room_eq_stage"] =
                    serde_json::Value::String("phase_alignment".to_string());
            }
            true
        } else {
            false
        };

        if applied {
            sync_reported_phase_adjustment(
                channel_name,
                channel_results,
                channel_chains,
                *delay_ms,
                false,
                sample_rate,
            );
            info!(
                "  Applied {:.3} ms phase alignment delay to '{}'",
                delay_ms, channel_name
            );
        }
    }

    schedule
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;
    use roomeq_engine::room_result::ChannelOptimizationResult;
    use roomeq_model::{ChannelDspChain, CurveData};
    use std::collections::HashMap;

    fn small_curve() -> roomeq_model::Curve {
        // 50 dB SNR everywhere: the assessment bar needs measured noise
        // evidence, never an invented floor.
        roomeq_model::Curve {
            freq: Array1::logspace(10.0, f64::log10(20.0), f64::log10(20_000.0), 16),
            spl: Array1::from_elem(16, 80.0),
            phase: Some(Array1::from_elem(16, 0.0)),
            noise_floor_db: Some(Array1::from_elem(16, 30.0)),
            ..Default::default()
        }
    }

    /// Flat magnitude with a pure acoustic delay: the assessment's known
    /// answer is a supported verdict with the fitted bulk delay. Phase is
    /// wrapped to ±180° like a real measurement.
    fn delay_curve(tau_ms: f64) -> roomeq_model::Curve {
        let freq = Array1::logspace(10.0, f64::log10(20.0), f64::log10(20_000.0), 64);
        let phase = freq.mapv(|f| {
            let raw = -360.0 * f * tau_ms / 1000.0;
            raw - 360.0 * (raw / 360.0).round()
        });
        roomeq_model::Curve {
            freq,
            spl: Array1::from_elem(64, 80.0),
            phase: Some(phase),
            noise_floor_db: Some(Array1::from_elem(64, 30.0)),
            ..Default::default()
        }
    }

    /// Dense flat curve for narrow-band assessment: the coarse bulk-delay
    /// fit degenerates on a handful of bins, so bass detail needs enough
    /// bins inside the band to be assessable.
    fn dense_flat_curve() -> roomeq_model::Curve {
        roomeq_model::Curve {
            freq: Array1::logspace(10.0, f64::log10(20.0), f64::log10(20_000.0), 64),
            spl: Array1::from_elem(64, 80.0),
            phase: Some(Array1::from_elem(64, 0.0)),
            noise_floor_db: Some(Array1::from_elem(64, 30.0)),
            ..Default::default()
        }
    }

    fn small_curve_no_phase() -> roomeq_model::Curve {
        roomeq_model::Curve {
            freq: Array1::logspace(10.0, f64::log10(20.0), f64::log10(20_000.0), 16),
            spl: Array1::from_elem(16, 80.0),
            phase: None,
            ..Default::default()
        }
    }

    fn curve_data(curve: &roomeq_model::Curve) -> CurveData {
        CurveData {
            freq: curve.freq.to_vec(),
            spl: curve.spl.to_vec(),
            phase: curve.phase.as_ref().map(|p| p.to_vec()),
            norm_range: None,
            ..Default::default()
        }
    }

    fn phase_gate(
        channel: &str,
        authorized: bool,
    ) -> roomeq_model::eligibility::ChannelOperationGate {
        phase_gate_band(channel, authorized, [20.0, 20_000.0])
    }

    fn phase_gate_band(
        channel: &str,
        authorized: bool,
        band: [f64; 2],
    ) -> roomeq_model::eligibility::ChannelOperationGate {
        use roomeq_model::eligibility::{
            CorrectionOperation, EligibilityRecord, EligibilityVerdict,
        };
        roomeq_model::eligibility::ChannelOperationGate {
            channel: channel.to_string(),
            measurement_id: channel.to_string(),
            records: vec![EligibilityRecord {
                record_id: format!("test-{channel}-excess-phase"),
                operation: CorrectionOperation::ExcessPhaseCorrection,
                verdict: if authorized {
                    EligibilityVerdict::Eligible
                } else {
                    EligibilityVerdict::Unsupported
                },
                measurement_id: channel.to_string(),
                seat_ids: Vec::new(),
                band_hz: Some(band),
                observations: vec![String::from("test gate")],
                policy_limits: Vec::new(),
                evidence_refs: vec![channel.to_string()],
                assessment: roomeq_model::AssessmentRecord::default(),
            }],
            rew_header_facts: None,
        }
    }

    /// Attach a direct-sound record to a gate: the damage guard's
    /// validated-direct-sound evidence axis, separate from the
    /// excess-phase authorization verdict.
    fn with_direct_sound(
        mut gate: roomeq_model::eligibility::ChannelOperationGate,
        verdict: roomeq_model::eligibility::EligibilityVerdict,
    ) -> roomeq_model::eligibility::ChannelOperationGate {
        use roomeq_model::eligibility::{CorrectionOperation, EligibilityRecord};
        gate.records.push(EligibilityRecord {
            record_id: format!("test-{}-direct-sound", gate.channel),
            operation: CorrectionOperation::DirectSoundSpeakerCorrection,
            verdict,
            measurement_id: gate.measurement_id.clone(),
            seat_ids: Vec::new(),
            band_hz: Some([20.0, 20_000.0]),
            observations: vec![String::from("test direct evidence")],
            policy_limits: Vec::new(),
            evidence_refs: vec![String::from("ev-direct")],
            assessment: roomeq_model::AssessmentRecord::default(),
        });
        gate
    }

    fn make_channel(
        name: &str,
        curve: roomeq_model::Curve,
    ) -> (ChannelOptimizationResult, ChannelDspChain) {
        let ch = ChannelOptimizationResult {
            measurement_conditioning: None,
            name: name.to_string(),
            pre_score: 0.0,
            post_score: 0.0,
            initial_curve: curve.clone(),
            final_curve: curve.clone(),
            biquads: Vec::new(),
            fir_coeffs: None,
            optimizer_evidence: Vec::new(),
            audibility_veto: Vec::new(),
            veto_adjudication: None,
        };
        let chain = ChannelDspChain {
            physical_correction_target: None,
            channel: name.to_string(),
            plugins: Vec::new(),
            drivers: None,
            initial_curve: None,
            final_curve: Some(curve_data(&curve)),
            eq_response: None,
            target_curve: None,
            pre_ir: None,
            post_ir: None,
            fir_temporal_masking: None,
            direct_early_late_correction: None,
            joint_sub: None,
            early_reflections: None,
            t60_octaves: None,
            speech_transmission: None,
            waterfall: None,
            resonance_decays: None,
            wavelet: None,
            early_late_curves: None,
        };
        (ch, chain)
    }

    #[test]
    fn apply_phase_correction_skips_without_phase() {
        let (mut ch, mut chain) = make_channel("left", small_curve_no_phase());
        let config = roomeq_model::MixedPhaseSerdeConfig {
            max_fir_length_ms: 10.0,
            pre_ringing_threshold_db: -30.0,
            min_spatial_depth: 0.5,
            phase_smoothing_octaves: 1.0 / 6.0,
            assessment: Default::default(),
            max_correction_latency_ms: None,
        };
        let gate = phase_gate("left", true);
        apply_phase_correction(
            "left", &mut ch, &mut chain, &config, 48_000.0, None, &gate, None, None,
        );
        assert!(chain.plugins.is_empty());
        assert!(ch.fir_coeffs.is_none());
    }

    /// A refused evidence gate blocks phase correction even with phase data.
    #[test]
    fn roadmap_correction_phase_refused_gate_skips_fir() {
        let (mut ch, mut chain) = make_channel("left", small_curve());
        let config = roomeq_model::MixedPhaseSerdeConfig {
            max_fir_length_ms: 10.0,
            pre_ringing_threshold_db: -30.0,
            min_spatial_depth: 0.5,
            phase_smoothing_octaves: 1.0 / 6.0,
            assessment: Default::default(),
            max_correction_latency_ms: None,
        };
        let gate = phase_gate("left", false);
        let refusals = apply_phase_correction(
            "left", &mut ch, &mut chain, &config, 48_000.0, None, &gate, None, None,
        );
        assert!(chain.plugins.is_empty());
        assert!(ch.fir_coeffs.is_none());
        assert_eq!(refusals.len(), 1);
        assert!(refusals[0].validate().is_ok());
        assert_eq!(refusals[0].reason_codes, ["phase_evidence_refused"]);
        assert_eq!(refusals[0].measurement_refs, [gate.measurement_id]);
        assert_eq!(refusals[0].frequency_band_hz, gate.records[0].band_hz);
        assert!(!refusals[0].is_final_claim());
    }

    #[test]
    fn roadmap_correction_phase_write_failure_keeps_original_chain() {
        use roomeq_model::eligibility::EligibilityVerdict;
        let temp = tempfile::tempdir().unwrap();
        // A regular file cannot be an artifact directory, on every platform.
        let not_a_directory = temp.path().join("regular-file");
        std::fs::write(&not_a_directory, b"fixture").unwrap();
        let (mut ch, mut chain) = make_channel("left", delay_curve(0.2));
        let before = serde_json::to_value(&chain).unwrap();
        let before_curve = ch.final_curve.clone();
        let gate = with_direct_sound(phase_gate("left", true), EligibilityVerdict::Eligible);
        let refusals = apply_phase_correction(
            "left",
            &mut ch,
            &mut chain,
            &assess_config(),
            48_000.0,
            Some(&not_a_directory),
            &gate,
            None,
            None,
        );
        assert_eq!(refusals.len(), 1);
        assert_eq!(refusals[0].reason_codes, ["phase_artifact_write_failed"]);
        assert!(refusals[0].validate().is_ok());
        assert_eq!(serde_json::to_value(&chain).unwrap(), before);
        assert_eq!(ch.final_curve.spl, before_curve.spl);
        assert_eq!(ch.final_curve.phase, before_curve.phase);
        assert!(ch.fir_coeffs.is_none());
    }

    #[test]
    fn apply_phase_alignment_delay_schedule_adds_delay_plugins() {
        let (l_ch, l_chain) = make_channel("L", small_curve());
        let (r_ch, r_chain) = make_channel("R", small_curve());
        let (sub_ch, sub_chain) = make_channel("Sub", small_curve());
        let mut results = HashMap::from([
            ("L".to_string(), l_ch),
            ("R".to_string(), r_ch),
            ("Sub".to_string(), sub_ch),
        ]);
        let mut chains = HashMap::from([
            ("L".to_string(), l_chain),
            ("R".to_string(), r_chain),
            ("Sub".to_string(), sub_chain),
        ]);
        let phase_results = HashMap::from([
            ("L".to_string(), (-2.0, false, "Sub".to_string())),
            ("R".to_string(), (1.0, false, "Sub".to_string())),
        ]);
        let schedule = apply_phase_alignment_delay_schedule(
            &phase_results,
            &mut results,
            &mut chains,
            48_000.0,
        );
        assert!(schedule.contains_key("Sub"));
        assert!(schedule.contains_key("R"));
        assert!(!schedule.contains_key("L"));
        assert!(
            chains["Sub"]
                .plugins
                .iter()
                .any(|p| p.plugin_type == "delay")
        );
        assert!(chains["R"].plugins.iter().any(|p| p.plugin_type == "delay"));
        assert!(!chains["L"].plugins.iter().any(|p| p.plugin_type == "delay"));
    }

    #[test]
    fn apply_phase_alignment_delay_schedule_empty_results_empty_schedule() {
        let mut results = HashMap::<String, ChannelOptimizationResult>::new();
        let mut chains = HashMap::<String, ChannelDspChain>::new();
        let schedule = apply_phase_alignment_delay_schedule(
            &HashMap::new(),
            &mut results,
            &mut chains,
            48_000.0,
        );
        assert!(schedule.is_empty());
    }

    #[test]
    fn apply_phase_correction_generates_fir_without_output_dir() {
        use roomeq_model::eligibility::EligibilityVerdict;
        let (mut ch, mut chain) = make_channel("left", small_curve());
        let config = roomeq_model::MixedPhaseSerdeConfig {
            max_fir_length_ms: 5.0,
            pre_ringing_threshold_db: -30.0,
            min_spatial_depth: 0.5,
            phase_smoothing_octaves: 1.0 / 6.0,
            assessment: Default::default(),
            max_correction_latency_ms: None,
        };
        let gate = with_direct_sound(phase_gate("left", true), EligibilityVerdict::Eligible);
        apply_phase_correction(
            "left", &mut ch, &mut chain, &config, 48_000.0, None, &gate, None, None,
        );
        assert!(ch.fir_coeffs.is_some());
        assert!(
            chain.plugins.iter().any(|p| p.plugin_type == "convolution"),
            "phase correction should add a convolution plugin"
        );
        let expected_taps = ch.fir_coeffs.as_ref().unwrap().len();
        let mut chains = HashMap::from([("left".to_string(), chain)]);
        let reports = roomeq_engine::output::take_mixed_phase_reports(&mut chains)
            .expect("phase correction should retain its decomposition report");
        assert_eq!(reports["left"].fir_taps, expected_taps);
    }

    #[test]
    fn apply_phase_correction_saves_wav_when_output_dir_provided() {
        use roomeq_model::eligibility::EligibilityVerdict;
        let tmp = tempfile::TempDir::new().unwrap();
        let (mut ch, mut chain) = make_channel("left", small_curve());
        let config = roomeq_model::MixedPhaseSerdeConfig {
            max_fir_length_ms: 5.0,
            pre_ringing_threshold_db: -30.0,
            min_spatial_depth: 0.5,
            phase_smoothing_octaves: 1.0 / 6.0,
            assessment: Default::default(),
            max_correction_latency_ms: None,
        };
        let gate = with_direct_sound(phase_gate("left", true), EligibilityVerdict::Eligible);
        apply_phase_correction(
            "left",
            &mut ch,
            &mut chain,
            &config,
            48_000.0,
            Some(tmp.path()),
            &gate,
            None,
            None,
        );
        assert!(ch.fir_coeffs.is_some());
        let wav_files: Vec<_> = std::fs::read_dir(tmp.path())
            .unwrap()
            .filter_map(|e| e.ok())
            .filter(|e| e.path().extension().is_some_and(|ext| ext == "wav"))
            .collect();
        assert!(!wav_files.is_empty(), "phase FIR WAV should be saved");
    }

    fn assess_config() -> roomeq_model::MixedPhaseSerdeConfig {
        roomeq_model::MixedPhaseSerdeConfig {
            max_fir_length_ms: 10.0,
            pre_ringing_threshold_db: -30.0,
            min_spatial_depth: 0.5,
            phase_smoothing_octaves: 1.0 / 6.0,
            assessment: Default::default(),
            max_correction_latency_ms: None,
        }
    }

    /// No noise floor means no SNR evidence: the assessment returns
    /// unknown and the phase action is refused while magnitude stays.
    #[test]
    fn roadmap_correction_phase_unknown_snr_refused() {
        use roomeq_model::eligibility::EligibilityVerdict;
        let mut curve = small_curve();
        curve.noise_floor_db = None;
        let (mut ch, mut chain) = make_channel("left", curve);
        let gate = with_direct_sound(phase_gate("left", true), EligibilityVerdict::Eligible);
        apply_phase_correction(
            "left",
            &mut ch,
            &mut chain,
            &assess_config(),
            48_000.0,
            None,
            &gate,
            None,
            None,
        );
        assert!(chain.plugins.is_empty(), "unknown SNR must refuse phase");
        assert!(ch.fir_coeffs.is_none());
    }

    /// Room-curve-only upper-band detail is refused by the damage guard
    /// even with a supported assessment and an authorizing gate.
    #[test]
    fn roadmap_correction_phase_room_only_detail_refused() {
        use roomeq_model::eligibility::EligibilityVerdict;
        let (mut ch, mut chain) = make_channel("left", small_curve());
        let gate = with_direct_sound(phase_gate("left", true), EligibilityVerdict::Unsupported);
        apply_phase_correction(
            "left",
            &mut ch,
            &mut chain,
            &assess_config(),
            48_000.0,
            None,
            &gate,
            None,
            None,
        );
        assert!(
            chain.plugins.is_empty(),
            "room-only upper-band detail must refuse phase"
        );
        assert!(ch.fir_coeffs.is_none());
    }

    /// Unknown direct evidence also refuses upper-band detail: only the
    /// in-situ bass region may proceed without validated direct sound.
    #[test]
    fn roadmap_correction_phase_unknown_evidence_detail_refused() {
        let (mut ch, mut chain) = make_channel("left", small_curve());
        let gate = phase_gate("left", true);
        apply_phase_correction(
            "left",
            &mut ch,
            &mut chain,
            &assess_config(),
            48_000.0,
            None,
            &gate,
            None,
            None,
        );
        assert!(
            chain.plugins.is_empty(),
            "unknown direct evidence must refuse upper-band detail"
        );
        assert!(ch.fir_coeffs.is_none());
    }

    /// Bass-region detail proceeds without direct-sound evidence: the
    /// damage guard never fires below the transition.
    #[test]
    fn roadmap_correction_phase_ignores_phase_outside_authorized_band() {
        let flat = dense_flat_curve();
        let mut altered = flat.clone();
        altered.phase = Some(altered.freq.mapv(|f| {
            if f > 1000.0 {
                20.0 * (f / 1000.0).ln().sin()
            } else {
                0.0
            }
        }));
        let render = |curve| {
            let (mut ch, mut chain) = make_channel("left", curve);
            apply_phase_correction(
                "left",
                &mut ch,
                &mut chain,
                &assess_config(),
                48_000.0,
                None,
                &phase_gate_band("left", true, [20.0, 120.0]),
                None,
                None,
            );
            ch.fir_coeffs
                .expect("supported bass correction is realized")
        };
        let reference = render(flat);
        let candidate = render(altered);
        assert_eq!(reference.len(), candidate.len());
        let largest_difference = reference
            .iter()
            .zip(&candidate)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        assert!(
            largest_difference < 1e-8,
            "unsupported upper-band phase must not change bass correction: {largest_difference}"
        );
    }

    #[test]
    fn roadmap_correction_phase_bass_detail_passes_without_direct() {
        let (mut ch, mut chain) = make_channel("left", dense_flat_curve());
        let gate = phase_gate_band("left", true, [20.0, 120.0]);
        apply_phase_correction(
            "left",
            &mut ch,
            &mut chain,
            &assess_config(),
            48_000.0,
            None,
            &gate,
            None,
            None,
        );
        assert!(
            ch.fir_coeffs.is_some(),
            "in-situ bass detail proceeds without direct evidence"
        );
    }

    #[test]
    fn roadmap_correction_phase_requires_full_direct_band_coverage() {
        use roomeq_model::eligibility::EligibilityVerdict;

        // Only evidence support changes, not the curve or requested action.
        // Touching intervals cover the band; even small gaps do not.
        for (bands, expected_applied) in [
            (vec![[20.0, 1_000.0]], false),
            (vec![[1_000.0, 20_000.0]], false),
            (vec![[20.0, 1_000.0], [1_001.0, 20_000.0]], false),
            (vec![[1_000.0, 20_000.0], [20.0, 1_000.0]], true),
            (vec![[20.0, 20_000.0]], true),
        ] {
            let (mut ch, mut chain) = make_channel("left", delay_curve(0.2));
            let mut gate =
                with_direct_sound(phase_gate("left", true), EligibilityVerdict::Eligible);
            let template = gate.records.pop().expect("direct-sound record");
            for (index, band) in bands.iter().enumerate() {
                let mut record = template.clone();
                record.record_id = format!("direct-band-{index}");
                record.band_hz = Some(*band);
                gate.records.push(record);
            }
            apply_phase_correction(
                "left",
                &mut ch,
                &mut chain,
                &assess_config(),
                48_000.0,
                None,
                &gate,
                None,
                None,
            );
            assert_eq!(
                ch.fir_coeffs.is_some(),
                expected_applied,
                "support: {bands:?}"
            );
            assert_eq!(
                chain
                    .plugins
                    .iter()
                    .any(|plugin| plugin.plugin_type == "convolution"),
                expected_applied,
                "delivered graph must match support: {bands:?}",
            );
        }
    }

    #[test]
    fn roadmap_correction_phase_rejects_unbound_permission_records() {
        use roomeq_model::decision_ledger::DecisionStatus;
        use roomeq_model::eligibility::EligibilityVerdict;

        // Exercise both the phase permission and the independent direct-sound
        // permission through application, not only the record validator.
        for record_index in [0, 1] {
            for mutation in 0..5 {
                let (mut ch, mut chain) = make_channel("left", delay_curve(0.2));
                let filter = math_audio_iir_fir::Biquad::new(
                    math_audio_iir_fir::BiquadFilterType::Peak,
                    80.0,
                    48_000.0,
                    1.0,
                    -1.0,
                );
                chain
                    .plugins
                    .push(roomeq_engine::output::create_eq_plugin(&[filter]));
                let before = serde_json::to_value(&chain).unwrap();
                let mut gate =
                    with_direct_sound(phase_gate("left", true), EligibilityVerdict::Eligible);
                let record = &mut gate.records[record_index];
                match mutation {
                    0 => record.evidence_refs.clear(),
                    1 => record.evidence_refs = vec!["  ".to_owned()],
                    2 => record.measurement_id = "different-measurement".to_owned(),
                    3 => record.record_id.clear(),
                    _ => record.band_hz = None,
                }
                let decisions = apply_phase_correction(
                    "left",
                    &mut ch,
                    &mut chain,
                    &assess_config(),
                    48_000.0,
                    None,
                    &gate,
                    None,
                    None,
                );
                assert!(
                    ch.fir_coeffs.is_none(),
                    "record={record_index}, mutation={mutation}"
                );
                assert_eq!(serde_json::to_value(&chain).unwrap(), before);
                assert!(!decisions.is_empty(), "refusal must be explained");
                assert!(
                    decisions
                        .iter()
                        .all(|decision| decision.status != DecisionStatus::Applied)
                );
            }
        }
    }

    /// Known pure-delay fixture: validated direct evidence authorizes the
    /// supported band, the correction applies, and the exported latency
    /// matches the fitted bulk delay.
    #[test]
    fn roadmap_correction_phase_supported_delay_applies_with_latency() {
        use roomeq_model::eligibility::EligibilityVerdict;
        let (mut ch, mut chain) = make_channel("left", delay_curve(0.2));
        let gate = with_direct_sound(phase_gate("left", true), EligibilityVerdict::Eligible);
        let decisions = apply_phase_correction(
            "left",
            &mut ch,
            &mut chain,
            &assess_config(),
            48_000.0,
            None,
            &gate,
            None,
            None,
        );
        assert!(ch.fir_coeffs.is_some(), "supported delay must apply");
        assert_eq!(decisions.len(), 1, "realized phase must produce a decision");
        let decision = &decisions[0];
        assert_eq!(
            decision.action,
            roomeq_model::decision_ledger::DecisionAction::PhaseCorrect
        );
        assert_eq!(
            decision.status,
            roomeq_model::decision_ledger::DecisionStatus::Applied
        );
        assert!(
            decision
                .reason_codes
                .iter()
                .any(|reason| reason == "target_ok")
        );
        assert!(
            decision
                .reason_codes
                .iter()
                .any(|reason| reason == "phase_fir_realized")
        );
        assert!(decision.validate().is_ok());
        assert!(
            !decision.is_final_claim(),
            "only final reconciliation binds delivery"
        );
        assert!(
            decision
                .observed
                .iter()
                .any(|value| value.name == "phase_fir_causal_center_delay")
        );
        assert!(
            chain.plugins.iter().any(|p| p.plugin_type == "convolution"),
            "supported delay adds a convolution plugin"
        );
        let mut chains = HashMap::from([("left".to_string(), chain)]);
        let reports = roomeq_engine::output::take_mixed_phase_reports(&mut chains)
            .expect("supported delay retains its decomposition report");
        let report = &reports["left"];
        assert!(
            (report.estimated_delay_ms - 0.2).abs() < 0.15,
            "exported latency matches the 0.2 ms fixture: {}",
            report.estimated_delay_ms
        );
    }

    /// An explicit latency budget refuses the phase action when the
    /// realized bulk delay exceeds it, instead of exporting surprise
    /// latency. The 0.2 ms fixture breaches a 0.05 ms budget.
    #[test]
    fn roadmap_correction_phase_latency_budget_refused() {
        use roomeq_model::eligibility::EligibilityVerdict;
        let (mut ch, mut chain) = make_channel("left", delay_curve(0.2));
        let mut config = assess_config();
        config.max_correction_latency_ms = Some(0.05);
        let gate = with_direct_sound(phase_gate("left", true), EligibilityVerdict::Eligible);
        let refusals = apply_phase_correction(
            "left", &mut ch, &mut chain, &config, 48_000.0, None, &gate, None, None,
        );
        assert!(
            chain.plugins.is_empty(),
            "latency over budget must refuse phase"
        );
        assert_eq!(refusals.len(), 1);
        assert_eq!(refusals[0].reason_codes, ["phase_latency_budget_exceeded"]);
        assert_eq!(refusals[0].limits[0].value, 0.05);
        assert_eq!(refusals[0].observed[0].unit, "ms");
        assert!(refusals[0].observed[0].value > 0.05);
        assert!(refusals[0].validate().is_ok());
        assert!(ch.fir_coeffs.is_none());
    }

    #[test]
    fn roadmap_correction_phase_latency_budget_uses_realized_fir_delay() {
        use roomeq_model::eligibility::EligibilityVerdict;
        for sample_rate in [44_100.0, 48_000.0, 96_000.0] {
            let (mut ch, mut chain) = make_channel("left", delay_curve(0.2));
            let mut config = assess_config();
            // The acoustic delay is 0.2 ms, but this 10 ms FIR adds a
            // roughly 5 ms causal centering delay at every sample rate.
            config.max_correction_latency_ms = Some(1.0);
            let gate = with_direct_sound(phase_gate("left", true), EligibilityVerdict::Eligible);
            apply_phase_correction(
                "left",
                &mut ch,
                &mut chain,
                &config,
                sample_rate,
                None,
                &gate,
                None,
                None,
            );
            assert!(
                ch.fir_coeffs.is_none(),
                "FIR exceeds latency budget at {sample_rate} Hz"
            );
            assert!(chain.plugins.is_empty());
            config.max_correction_latency_ms = Some(6.0);
            apply_phase_correction(
                "left",
                &mut ch,
                &mut chain,
                &config,
                sample_rate,
                None,
                &gate,
                None,
                None,
            );
            let taps = ch
                .fir_coeffs
                .as_ref()
                .expect("6 ms permits this realization");
            let peak = taps
                .iter()
                .enumerate()
                .max_by(|(_, left), (_, right)| left.abs().total_cmp(&right.abs()))
                .map(|(index, _)| index)
                .unwrap();
            assert_eq!(
                peak,
                taps.len() / 2,
                "pure-delay fixture yields a centered impulse"
            );
            let expected_ms = peak as f64 * 1000.0 / sample_rate;
            let mut chains = HashMap::from([("left".to_string(), chain)]);
            let reports = roomeq_engine::output::take_mixed_phase_reports(&mut chains).unwrap();
            let report = &reports["left"];
            assert_eq!(report.causal_center_delay_ms, Some(expected_ms));
            assert!((report.estimated_delay_ms - 0.2).abs() < 0.15);
        }
    }
}
