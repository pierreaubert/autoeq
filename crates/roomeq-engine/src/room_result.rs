//! Shared in-memory RoomEQ execution results.

use std::collections::HashMap;

use autoeq_core::Curve;
use autoeq_optim::optim::OptimizerRunEvidence;
use math_audio_iir_fir::Biquad;
use roomeq_model::{ChannelDspChain, DspChainOutput, OptimizationMetadata};

/// A validated decision snapshot bound to one serialized workflow result.
///
/// This binds the serialized payload, not external resource contents or playback.
/// Conversion after a payload change keeps history but invalidates delivery claims.
#[derive(Debug, Clone)]
pub struct FinalizedDecisions {
    fingerprint: String,
    ledger: roomeq_model::decision_ledger::CorrectionDecisionLedger,
}

impl FinalizedDecisions {
    /// Capture a reconciled output's decision ledger and exact payload identity.
    ///
    /// # Errors
    /// Returns an error for an absent/invalid ledger, stale final bindings, or
    /// a payload that cannot serialize.
    pub fn from_output(output: &DspChainOutput) -> Result<Self, String> {
        use roomeq_model::decision_ledger::{DecisionStage, canonical_value_identity};
        let ledger = output
            .correction_decisions
            .as_ref()
            .ok_or_else(|| "workflow output has no reconciled ledger".to_owned())?;
        ledger.validate()?;
        let mut value = serde_json::to_value(output).map_err(|error| error.to_string())?;
        value
            .as_object_mut()
            .ok_or("workflow payload is not an object")?
            .remove("correction_decisions");
        let fingerprint = canonical_value_identity(&value).fingerprint;
        if ledger
            .acceptance_evidence
            .as_ref()
            .is_some_and(|evidence| !evidence.matches(&fingerprint))
        {
            return Err("stale workflow acceptance evidence".to_owned());
        }
        if ledger
            .payload_binding
            .as_ref()
            .is_some_and(|binding| !binding.matches(&value, &fingerprint))
        {
            return Err("stale workflow payload binding".to_owned());
        }
        for record in &ledger.decisions {
            if record.stage == DecisionStage::Final
                && record.final_graph_identity.as_deref() != Some(fingerprint.as_str())
            {
                return Err(format!("stale workflow decision: {}", record.decision_id));
            }
        }
        Ok(Self {
            fingerprint,
            ledger: ledger.clone(),
        })
    }

    fn attach(&self, output: &mut DspChainOutput) {
        use roomeq_model::decision_ledger::{
            DecisionStage, DecisionStatus, canonical_value_identity,
        };
        let matches = serde_json::to_value(&*output)
            .ok()
            .is_some_and(|value| canonical_value_identity(&value).fingerprint == self.fingerprint);
        let mut ledger = self.ledger.clone();
        if !matches {
            for record in &mut ledger.decisions {
                record.stage = DecisionStage::Provisional;
                record.status = DecisionStatus::Unresolved;
                record.final_graph_identity = None;
                record
                    .reason_codes
                    .push("payload_changed_after_reconciliation".to_owned());
                record.confidence = roomeq_model::AssessmentConfidence::Unknown;
            }
        }
        output.correction_decisions = Some(ledger);
    }
}

/// Result for a single channel optimization.
#[derive(Debug, Clone)]
pub struct ChannelOptimizationResult {
    /// Recorded loading operations; absent means this path did not retain receipts.
    pub measurement_conditioning:
        Option<crate::channel_measurements::MeasurementConditioningReceipt>,
    pub name: String,
    pub pre_score: f64,
    pub post_score: f64,
    pub initial_curve: Curve,
    pub final_curve: Curve,
    /// Retained PEQ parameters, not an authoritative complete-chain transfer.
    /// Kautz bank weights are never encoded here as dB gains.
    pub biquads: Vec<Biquad>,
    pub fir_coeffs: Option<Vec<f64>>,
    pub optimizer_evidence: Vec<OptimizerRunEvidence>,
    pub audibility_veto: Vec<roomeq_model::FilterVetoVerdict>,
    pub veto_adjudication: Option<roomeq_model::VetoAdjudicationReport>,
}

/// Result for a single speaker optimization.
#[derive(Debug, Clone)]
pub struct SpeakerOptimizationResult {
    /// Recorded loading operations; absent is not proof of unchanged measurements.
    pub measurement_conditioning:
        Option<crate::channel_measurements::MeasurementConditioningReceipt>,
    pub chain: ChannelDspChain,
    pub pre_score: f64,
    pub post_score: f64,
    pub initial_curve: Curve,
    pub final_curve: Curve,
    pub biquads: Vec<Biquad>,
    pub fir_coeffs: Option<Vec<f64>>,
    pub optimizer_evidence: Vec<OptimizerRunEvidence>,
    pub audibility_veto: Vec<roomeq_model::FilterVetoVerdict>,
    pub veto_adjudication: Option<roomeq_model::VetoAdjudicationReport>,
}

/// Complete in-memory result of a RoomEQ optimization workflow.
#[derive(Debug, Clone)]
pub struct RoomOptimizationResult {
    /// Workflow-owned final snapshot; raw engine results leave this absent.
    pub finalized_decisions: Option<FinalizedDecisions>,
    pub channels: HashMap<String, ChannelDspChain>,
    pub channel_results: HashMap<String, ChannelOptimizationResult>,
    /// Coherent, per-logical-input deployed responses after bass-management routing.
    ///
    /// These are distinct from `channel_results`: the latter remains the raw
    /// serialized channel-chain response used for DSP realization and export
    /// replay.
    pub deployed_source_curves: HashMap<String, Curve>,
    pub combined_pre_score: f64,
    pub combined_post_score: f64,
    pub metadata: OptimizationMetadata,
}

impl RoomOptimizationResult {
    /// Convert the result into the serializable DSP-chain contract.
    pub fn to_dsp_chain_output(&self) -> DspChainOutput {
        let mut output = crate::output::create_dsp_chain_output(
            self.channels.clone(),
            Some(self.metadata.clone()),
        );
        output.deployed_source_curves = self
            .deployed_source_curves
            .iter()
            .map(|(channel, curve)| (channel.clone(), curve.into()))
            .collect();
        // Requested-vs-realized audit labeling: derive the shipped
        // correction family from serialized plugins and name any
        // divergence from the requested mode. Conversion is the last
        // point that sees both the final graph and the effective
        // configuration before ledger finalization binds the payload.
        if let Some(metadata) = output.metadata.as_mut()
            && let Some(report) = metadata.correction_acceptance.as_mut()
        {
            let realized =
                roomeq_model::report_contracts::assess_realized_processing(&output.channels);
            let requested = metadata
                .effective_config
                .as_ref()
                .map(|config| &config.optimizer.processing_mode);
            report.realized_processing = Some(realized);
            report.processing_fallback =
                roomeq_model::report_contracts::processing_fallback_reason(requested, &realized);
        }
        // WP6 latency split: the modeled playback total is the FIR design
        // delay plus the common alignment offset from causal delay
        // compilation. Host/block buffering stays unmodeled (see docs).
        let alignment = output
            .metadata
            .as_ref()
            .map(|metadata| {
                metadata
                    .stage_outcomes
                    .iter()
                    .find(|stage| stage.stage == "delay_compile_causal")
                    .and_then(|stage| {
                        stage
                            .checks
                            .iter()
                            .find(|check| check.id == "delay_compile:common_latency_ms")
                            .and_then(|check| check.observed)
                    })
                    .filter(|offset| offset.is_finite() && *offset >= 0.0)
            })
            .unwrap_or(None);
        if let Some(temporal) = output
            .metadata
            .as_mut()
            .and_then(|metadata| metadata.correction_acceptance.as_mut())
            .and_then(|report| report.acoustic_quality.as_mut())
            .map(|score| &mut score.temporal)
        {
            temporal.alignment_delay_ms = alignment;
            temporal.total_latency_ms = match (temporal.latency_ms, alignment) {
                (Some(design), Some(align)) => Some(design + align),
                _ => None,
            };
        }
        // WP7 claim-level reporting: EPA provenance plus the playback
        // summary. Both are pure projections of shipped evidence.
        let summary = output
            .metadata
            .as_ref()
            .and_then(|metadata| metadata.correction_acceptance.as_ref())
            .map(roomeq_model::report_contracts::playback_summary);
        if let Some(metadata) = output.metadata.as_mut() {
            metadata.playback_summary = summary;
            if metadata.epa_per_channel.is_some() || metadata.epa_multichannel.is_some() {
                let epa_config = metadata
                    .effective_config
                    .as_ref()
                    .and_then(|config| config.optimizer.epa_config.as_ref());
                metadata.epa_provenance =
                    Some(roomeq_model::report_contracts::EpaProvenance {
                        model: "epa_spectral_diagnostic".to_string(),
                        predicted_not_measured: true,
                        listening_level_phon: epa_config
                            .map(|config| config.listening_level_phon),
                        target_sharpness_acum: epa_config
                            .map(|config| config.target_sharpness),
                        note: "Predicted preference dimensions from frequency response; not measured audibility. Validate with listening.".to_string(),
                    });
            }
        }
        // The ledger binds the exact payload being shipped, so the attach
        // check runs after every derived field above is computed. Any
        // future post-attach mutation reopens the mismatch demotion.
        if let Some(decisions) = &self.finalized_decisions {
            decisions.attach(&mut output);
        }
        output
    }
}
