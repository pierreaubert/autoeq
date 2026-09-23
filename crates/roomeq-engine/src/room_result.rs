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
        if let Some(decisions) = &self.finalized_decisions {
            decisions.attach(&mut output);
        }
        output
    }
}
