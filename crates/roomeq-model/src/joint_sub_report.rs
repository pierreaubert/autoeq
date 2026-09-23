//! Stage-bound predictions from joint subwoofer control and shared equalization.
//!
//! Levels retain the measurement reference; they are not calibrated SPL unless
//! the original captures establish it. These predictions exclude subsequent
//! workflow trims, global routing, limiting, and acoustic playback verification.

use crate::{ChannelDspChain, CurveData};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// Objective components before or after joint array control.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct JointSubObjectiveReport {
    /// Mean seat-to-seat variance in squared decibels.
    pub variation_db2: f64,
    /// Unnormalized output-loss penalty; not a calibrated excursion estimate.
    pub output_drive_penalty: f64,
    /// Mean squared target error in squared decibels.
    pub target_error_db2: f64,
    /// Weighted scalar objective used by this search.
    pub total: f64,
}

/// One seat's unnormalized predictions through the joint processing stages.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct JointSubSeatReport {
    /// Seat index in the retained source-by-seat matrix.
    pub seat_index: usize,
    /// Shared timing-reference and seat scope carried by measurement intake.
    pub reference_scope: String,
    /// Combined response before array gain and delay changes.
    pub before: CurveData,
    /// Combined response after array gain and delay changes, before shared EQ.
    pub after_array: CurveData,
    /// Combined response after the same shared EQ is applied to every seat.
    pub after_shared_eq: CurveData,
    /// Mean input-reference level over the evaluated band, before control.
    pub before_level_db: f64,
    /// Mean input-reference level after array control.
    pub after_array_level_db: f64,
    /// Mean input-reference level after shared EQ; never independently normalized.
    pub after_shared_eq_level_db: f64,
}

/// One explicit logical gain application retained from array optimization.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct JointSubGainApplication {
    /// Logical gain stage: `lfe` or `redirected`.
    pub stage: String,
    /// Physical output identity used by the optimizer.
    pub physical_output: String,
    /// Gain in decibels; later routing must not apply this stage twice.
    pub gain_db: f64,
}

/// Joint-array and shared-EQ stage diagnostics, not a playback acceptance verdict.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct JointSubDiagnostics {
    /// Shared-EQ rejection reason before later trims and routing; not final-chain acceptance.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub shared_eq_rejection_reason: Option<String>,
    /// Array-stage rejection reason; absence does not establish final-chain acceptance.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub array_rejection_reason: Option<String>,
    /// Explicit scope of these predictions and their unavailable claims.
    pub scope: String,
    /// Frequency band used for the unnormalized level summaries, in hertz.
    pub level_band_hz: [f64; 2],
    /// Physical outputs in array-control order.
    pub physical_outputs: Vec<String>,
    /// Array gains in decibels, excluding shared EQ and later workflow trims.
    pub array_gains_db: Vec<f64>,
    /// Array delays in milliseconds, excluding subsequent processing.
    pub array_delays_ms: Vec<f64>,
    /// Whether the array optimizer reported convergence, independent of acceptance.
    pub converged: bool,
    /// Objective before array control.
    pub before_objective: JointSubObjectiveReport,
    /// Objective after array control and before shared EQ.
    pub after_array_objective: JointSubObjectiveReport,
    /// Every retained seat, not only its spatial mean or primary seat.
    pub seats: Vec<JointSubSeatReport>,
    /// Logical gain applications observed at the array stage.
    pub gain_applications: Vec<JointSubGainApplication>,
    /// Fingerprint of channel plugins and driver controls at assessment time.
    ///
    /// It does not bind external resource bytes, global routing, or hardware.
    pub assessed_channel_processing: String,
    /// Whether those channel controls still match at the last output conversion.
    ///
    /// False keeps the report as stage history after processing changes. True
    /// is not a full-graph, resource, output-safety, or acoustic validation.
    #[serde(default)]
    pub channel_processing_matches: bool,
}

/// Fingerprint the channel controls relevant to a joint-sub stage prediction.
///
/// # Errors
///
/// Returns a serialization error if channel plugin parameters cannot serialize.
pub fn joint_sub_processing_identity(chain: &ChannelDspChain) -> Result<String, serde_json::Error> {
    let drivers = chain.drivers.as_ref().map(|drivers| {
        drivers
            .iter()
            .map(|driver| (&driver.name, driver.index, &driver.plugins))
            .collect::<Vec<_>>()
    });
    let value = serde_json::to_value((&chain.plugins, drivers))?;
    Ok(crate::decision_ledger::canonical_value_identity(&value).fingerprint)
}

/// Refresh the channel-control binding without discarding historical stage predictions.
pub fn refresh_joint_sub_binding(chain: &mut ChannelDspChain) {
    if chain.joint_sub.is_none() {
        return;
    }
    let identity = joint_sub_processing_identity(chain).ok();
    if let Some(report) = &mut chain.joint_sub {
        report.channel_processing_matches =
            identity.as_ref() == Some(&report.assessed_channel_processing);
    }
}
