//! Retained parsed inputs used by final-seat replay, without acquisition claims.

use super::{Capture, invalid};
use crate::evidence_intake::{
    RawCaptureRef, WorkflowIntake, provenance_capture_kind, validate_intake_grids,
};
use autoeq_core::MeasurementProvenance;
use autoeq_measurements::{Take, TakeDecision};
use roomeq_model::decision_ledger::canonical_value_identity;
use roomeq_model::{Result, StageCheck, StageCheckKind, StageOutcome, StageStatus};

#[derive(Debug, Default)]
pub(super) struct Receipt {
    checks: Vec<StageCheck>,
}

impl Receipt {
    pub(super) fn record(
        &mut self,
        partition: &str,
        config_key: Option<&str>,
        capture: &Capture,
        provenance: MeasurementProvenance,
    ) -> Result<()> {
        let source = serde_json::json!([partition, capture.channel, config_key, capture.driver]);
        let source_id = format!(
            "loaded-source:{}",
            canonical_value_identity(&source).fingerprint
        );
        let bands = provenance.declared_support_bands().map_err(invalid)?;
        let mut intake = WorkflowIntake::default();
        for (index, curve) in capture.curves.iter().enumerate() {
            curve.validate("retained final-seat input")?;
            let value = serde_json::to_value(curve).map_err(|e| invalid(e.to_string()))?;
            let measurement_id = format!(
                "loaded-response:{}",
                canonical_value_identity(&value).fingerprint
            );
            // Source-scoped indices identify storage, not a proven common seat.
            // Measurement names may describe speakers rather than seat locations.
            let seat_id = format!("{source_id}:position-index:{index}");
            let take_id = format!("{seat_id}:loaded-response");
            intake
                .record_take(
                    RawCaptureRef {
                        measurement_id,
                        source_id: source_id.clone(),
                        seat_id: seat_id.clone(),
                        take_id: take_id.clone(),
                        capture_kind: provenance_capture_kind(provenance.capture_kind),
                        calibration_id: provenance.calibration_id.clone(),
                        artifact_hash: None,
                        grid_hz: curve.freq.to_vec(),
                        validity_mask: bands.as_ref().map(|bands| {
                            curve
                                .freq
                                .iter()
                                .map(|f| bands.iter().any(|[low, high]| *f >= *low && *f <= *high))
                                .collect()
                        }),
                    },
                    Take {
                        take_id,
                        source_id: source_id.clone(),
                        seat_id,
                        weight: 1.0,
                        decision: TakeDecision::Accepted,
                        curve: Some(curve.clone()),
                        // A declared timing reference does not prove shared gain.
                        reference_id: None,
                    },
                    None,
                )
                .map_err(invalid)?;
        }
        validate_intake_grids(&intake.raw_refs).map_err(invalid)?;
        let payload = serde_json::json!({
            "version": 1,
            "evidence_kind": "loaded_response_snapshot",
            "partition": partition,
            "source_id": source_id,
            "logical_channel_or_physical_role": capture.channel,
            "configuration_source_key": config_key,
            "driver": capture.driver,
            "declared_measurement_labels": capture.seat_labels,
            "seat_identity_scope": "source-scoped positional indices; labels do not authenticate physical seat identity",
            "declared_provenance": provenance,
            "raw_refs": intake.raw_refs,
            "takes": intake.takes.takes,
            "conditioning": intake.conditioning,
            "conditioning_scope": "no gain/alignment/calibration applied by snapshot retention; acquisition and other consumers' conditioning remain unknown",
            "take_decision_scope": "accepted for structural retention only; unit weights are not optimizer weights; no averaging performed",
            "consumer_scope": "native numerical response snapshot; public workflow supplies frozen training responses to optimization and final replay; held-out responses are validation-only; auxiliary WAV/CEA/target assets and later conditioning are outside this receipt",
        });
        self.checks.push(StageCheck {
            id: source_id,
            kind: StageCheckKind::Structural,
            passed: true,
            observed: None,
            limit: None,
            diagnostic: Some(serde_json::to_string(&payload).map_err(|e| invalid(e.to_string()))?),
        });
        Ok(())
    }

    pub(super) fn into_stage(self) -> StageOutcome {
        StageOutcome {
            stage: "final_seat_input_retention".into(),
            status: StageStatus::Applied,
            checks: self.checks,
            advisories: vec![
                "structural_snapshot_not_acoustic_acceptance_or_authenticated_raw_recording".into(),
                "declared_calibration_and_timing_not_independently_verified".into(),
                "training_and_held_out_partitions_retained_separately".into(),
            ],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use autoeq_core::{
        InlineMeasurement, MeasurementRef, MeasurementSingle, MeasurementSource,
        ProvenanceCaptureKind,
    };
    use roomeq_model::{RoomConfig, SpeakerConfig};
    use std::collections::HashMap;

    #[test]
    fn receipt_retains_disjoint_support_gap() {
        let mut receipt = Receipt::default();
        let curve = roomeq_model::Curve {
            freq: vec![100.0, 200.0, 300.0, 400.0, 500.0, 600.0].into(),
            spl: vec![80.0; 6].into(),
            ..Default::default()
        };
        receipt
            .record(
                "training",
                None,
                &Capture {
                    channel: "left".into(),
                    driver: None,
                    curves: vec![curve],
                    seat_labels: None,
                },
                MeasurementProvenance {
                    valid_bands_hz: vec![[100.0, 200.0], [500.0, 600.0]],
                    ..Default::default()
                },
            )
            .unwrap();
        let stage = receipt.into_stage();
        let payload: serde_json::Value =
            serde_json::from_str(stage.checks[0].diagnostic.as_ref().unwrap()).unwrap();
        assert_eq!(
            payload["raw_refs"][0]["validity_mask"],
            serde_json::json!([true, true, false, false, true, true])
        );
    }

    #[test]
    fn receipt_preserves_declarations_without_inventing_acquisition_or_gain_reference() {
        let provenance = MeasurementProvenance {
            capture_kind: ProvenanceCaptureKind::StationaryIr,
            calibration_id: Some("declared-calibration".into()),
            timing_reference_id: Some("declared-timing-only".into()),
            valid_band_hz: Some([40.0, 80.0]),
            ..Default::default()
        };
        let source = MeasurementSource::Single(MeasurementSingle {
            measurement: MeasurementRef::Inline(InlineMeasurement {
                frequencies: vec![40.0, 80.0, 160.0],
                magnitude_db: vec![81.0, 85.0, 80.0],
                phase_deg: Some(vec![0.0, -30.0, -60.0]),
                name: Some("speaker label, not seat evidence".into()),
                wav_path: None,
                csv_path: None,
            }),
            speaker_name: None,
            provenance: provenance.clone(),
        });
        let config = RoomConfig {
            speakers: HashMap::from([("left".into(), SpeakerConfig::Single(source))]),
            ..Default::default()
        };
        let (captures, stage) =
            super::super::capture_with_receipt(&config, &HashMap::new()).unwrap();
        let payload: serde_json::Value =
            serde_json::from_str(stage.checks[0].diagnostic.as_ref().unwrap()).unwrap();
        assert_eq!(
            payload["declared_provenance"],
            serde_json::to_value(provenance).unwrap()
        );
        assert_eq!(
            payload["takes"][0]["curve"],
            serde_json::to_value(&captures[0].curves[0]).unwrap()
        );
        assert!(payload["takes"][0]["reference_id"].is_null());
        assert!(payload["raw_refs"][0]["artifact_hash"].is_null());
        assert_eq!(
            payload["raw_refs"][0]["validity_mask"],
            serde_json::json!([true, true, false])
        );
        assert!(
            payload["raw_refs"][0]["measurement_id"]
                .as_str()
                .unwrap()
                .starts_with("loaded-response:")
        );
        assert!(
            payload["takes"][0]["seat_id"]
                .as_str()
                .unwrap()
                .contains("position-index:0")
        );
        assert_eq!(
            payload["conditioning"]["calibrations"],
            serde_json::json!([])
        );
        assert_eq!(stage.checks[0].kind, StageCheckKind::Structural);
        let (_, second) = super::super::capture_with_receipt(&config, &HashMap::new()).unwrap();
        assert_eq!(stage.checks[0].diagnostic, second.checks[0].diagnostic);
    }
}
