//! Freeze numerical measurement inputs before optimization and final replay.

use roomeq_model::{AutoeqError, MeasurementSource, Result, RoomConfig, SpeakerConfig};

pub(super) fn freeze(config: &RoomConfig) -> Result<RoomConfig> {
    let mut snapshot = config.clone();
    let mut keys: Vec<_> = snapshot.speakers.keys().cloned().collect();
    keys.sort();
    for key in keys {
        let speaker = snapshot
            .speakers
            .get_mut(&key)
            .expect("key came from this map");
        let sources: Vec<&mut MeasurementSource> = match speaker {
            SpeakerConfig::Single(source) => vec![source],
            SpeakerConfig::Topology(topology) => topology
                .drivers
                .iter_mut()
                .map(|driver| &mut driver.measurement)
                .collect(),
            SpeakerConfig::Group(group) => group.measurements.iter_mut().collect(),
            SpeakerConfig::MultiSub(group) => group.subwoofers.iter_mut().collect(),
            SpeakerConfig::Dba(group) => group.front.iter_mut().chain(&mut group.rear).collect(),
            SpeakerConfig::Cardioid(group) => vec![&mut group.front, &mut group.rear],
            SpeakerConfig::SupportingSource(group) => vec![&mut group.primary, &mut group.support],
        };
        for (index, source) in sources.into_iter().enumerate() {
            *source = autoeq_measurements::read::snapshot_source(source).map_err(|error| {
                AutoeqError::InvalidMeasurement {
                    message: format!("cannot snapshot source '{key}' branch {index}: {error}"),
                }
            })?;
        }
    }
    Ok(snapshot)
}

/// Canonicalize declared seat IDs before primary selection and calibration.
pub(crate) fn normalize_seat_identity(
    config: &RoomConfig,
) -> (RoomConfig, roomeq_model::StageOutcome) {
    use roomeq_model::{StageCheck, StageCheckKind, StageOutcome, StageStatus};
    let mut normalized = config.clone();
    let canonical = config
        .optimizer
        .multi_seat
        .as_ref()
        .and_then(|policy| policy.seat_identity.as_ref());
    let mut checks = Vec::new();
    let mut changed = false;
    let mut keys: Vec<_> = normalized.speakers.keys().cloned().collect();
    keys.sort();
    for key in keys {
        let speaker = normalized
            .speakers
            .get_mut(&key)
            .expect("key came from this map");
        let sources: Vec<&mut MeasurementSource> = match speaker {
            SpeakerConfig::Single(source) => vec![source],
            SpeakerConfig::Topology(topology) => topology
                .drivers
                .iter_mut()
                .map(|driver| &mut driver.measurement)
                .collect(),
            SpeakerConfig::Group(group) => group.measurements.iter_mut().collect(),
            SpeakerConfig::MultiSub(group) => group.subwoofers.iter_mut().collect(),
            SpeakerConfig::Dba(group) => group.front.iter_mut().chain(&mut group.rear).collect(),
            SpeakerConfig::Cardioid(group) => vec![&mut group.front, &mut group.rear],
            SpeakerConfig::SupportingSource(group) => vec![&mut group.primary, &mut group.support],
        };
        for (branch, source) in sources.into_iter().enumerate() {
            let before = serde_json::to_value(&*source).ok();
            let declared = crate::group_measurements::seat_labels(source);
            let plan = canonical.map(|map| {
                if !map.validate(map.ids.len()).is_empty() {
                    return Err("invalid_canonical_physical_seat_identity".to_string());
                }
                crate::group_measurements::validate_source_seat_bindings(source)?;
                let labels = declared
                    .as_ref()
                    .ok_or_else(|| "missing_physical_seat_identity".to_string())?;
                let permutation = crate::group_measurements::seat_permutation(&map.ids, labels)?;
                if let Some(capture) = source.provenance().capture.as_ref()
                    && capture.takes.len() != labels.len()
                {
                    return Err("physical_acquisition_identity_support_mismatch".into());
                }
                if let Some(capture) = source.provenance().capture.as_ref() {
                    capture.reordered_for_measurements(&permutation)?;
                }
                Ok(permutation)
            });
            if let Some(Ok(permutation)) = &plan
                && permutation
                    .iter()
                    .enumerate()
                    .any(|(index, actual)| index != *actual)
                && let MeasurementSource::Multiple(multiple) = source
            {
                let measurements = multiple.measurements.clone();
                multiple.measurements = permutation
                    .iter()
                    .map(|index| measurements[*index].clone())
                    .collect();
                if let Some(capture) = &mut multiple.provenance.capture {
                    *capture = capture.reordered_for_measurements(permutation).expect(
                        "capture permutation and parallel support validated before mutation",
                    );
                }
                changed = true;
            }
            let reason = plan.as_ref().and_then(|plan| plan.as_ref().err());
            checks.push(StageCheck {
                id: format!("physical_seat_identity_source_{key}_{branch}"),
                kind: StageCheckKind::Structural,
                passed: plan.as_ref().is_some_and(|plan| plan.is_ok()),
                observed: None,
                limit: None,
                diagnostic: Some(serde_json::json!({
                    "original_source_inventory": before,
                    "original_declared_order": declared,
                    "canonical_ids": canonical.map(|map| &map.ids),
                    "canonical_to_original_permutation": plan.as_ref().and_then(|plan| plan.as_ref().ok()),
                    "refusal_reason": reason,
                    "primary_physical_id": config.optimizer.multi_seat.as_ref().and_then(|policy| canonical.and_then(|map| map.ids.get(policy.primary_seat))),
                    "weight_order": "configured canonical IDs; no source weight invented or changed",
                    "scope": "declared identity mapping; not authentication of acoustic capture",
                }).to_string()),
            });
        }
    }
    let status = if canonical.is_none() {
        StageStatus::Skipped
    } else if checks.iter().any(|check| !check.passed) {
        StageStatus::Degraded
    } else if changed {
        StageStatus::Applied
    } else {
        StageStatus::Skipped
    };
    (
        normalized,
        StageOutcome {
            stage: "physical_seat_identity_normalization".into(),
            status,
            checks,
            advisories: vec![
                "original_source_inventory_retained_before_snapshot_and_primary_selection".into(),
                "ambiguous_coherent_correction_requires_refusal_individual_fits_remain_available"
                    .into(),
            ],
        },
    )
}

#[cfg(test)]
mod physical_seat_identity_tests {
    use super::*;
    use autoeq_core::{
        InlineMeasurement, MeasurementMultiple, MeasurementProvenance, MeasurementRef,
    };
    use roomeq_model::{MultiMeasurementConfig, MultiSeatConfig, SeatIdentityMap};

    fn source(labels: &[&str]) -> MeasurementSource {
        let takes = labels.iter().map(|label| serde_json::json!({
            "microphone_id": label, "seat_id": label, "device_id": "synthetic-device",
            "offset_samples": 0.0, "skew_ppm": 0.0, "residual_uncertainty_us": 1.0,
            "correction_applied": "resampled", "timing_reference_id": "fixed-reference",
            "calibration_id": "fixed-calibration", "gain_db": 0.0,
            "calibration_orientation": "on_axis", "position_m": [if *label == "A" {0.0} else {1.0}, 0.0, 0.0],
            "position_uncertainty_mm": 1.0, "preserves_acoustic_delay": true, "quality_passed": true,
        })).collect::<Vec<_>>();
        MeasurementSource::Multiple(MeasurementMultiple {
            measurements: labels
                .iter()
                .map(|label| {
                    MeasurementRef::Inline(InlineMeasurement {
                        frequencies: vec![40.0, 80.0, 160.0],
                        magnitude_db: vec![if *label == "A" { 80.0 } else { 90.0 }; 3],
                        phase_deg: Some(vec![if *label == "A" { 17.0 } else { -31.0 }; 3]),
                        name: Some((*label).into()),
                        wav_path: None,
                        csv_path: None,
                    })
                })
                .collect(),
            speaker_name: None,
            provenance: MeasurementProvenance {
                capture: Some(
                    serde_json::from_value(
                        serde_json::json!({"geometry": "spread", "takes": takes}),
                    )
                    .unwrap(),
                ),
                ..Default::default()
            },
        })
    }

    fn config(main: MeasurementSource, sub: MeasurementSource) -> RoomConfig {
        RoomConfig {
            speakers: std::collections::HashMap::from([
                ("main".into(), SpeakerConfig::Single(main)),
                ("sub".into(), SpeakerConfig::Single(sub)),
            ]),
            optimizer: roomeq_model::OptimizerConfig {
                multi_seat: Some(MultiSeatConfig {
                    primary_seat: 1,
                    seat_weights: Some(vec![2.0, 1.0]),
                    seat_identity: Some(SeatIdentityMap {
                        ids: vec!["A".into(), "B".into()],
                    }),
                    ..Default::default()
                }),
                ..Default::default()
            },
            ..Default::default()
        }
    }

    #[test]
    fn identity_join_reorders_refs_and_capture_takes_preserving_primary_and_weights() {
        let original = config(source(&["B", "A"]), source(&["A", "B"]));
        let original_inventory = serde_json::to_value(&original.speakers).unwrap();
        let (normalized, receipt) = normalize_seat_identity(&original);
        let SpeakerConfig::Single(main) = &normalized.speakers["main"] else {
            panic!()
        };
        let MeasurementSource::Multiple(main) = main else {
            panic!()
        };
        assert_eq!(main.measurements[0].name(), Some("A"));
        assert_eq!(
            main.measurements[0].inline_data().unwrap().magnitude_db,
            vec![80.0; 3]
        );
        assert_eq!(
            main.measurements[1].inline_data().unwrap().phase_deg,
            Some(vec![-31.0; 3])
        );
        assert_eq!(
            main.provenance.capture.as_ref().unwrap().takes[0].microphone_id,
            "A"
        );
        assert_eq!(
            main.provenance.capture.as_ref().unwrap().takes[1].position_m,
            [1.0, 0.0, 0.0]
        );
        let policy = normalized.optimizer.multi_seat.as_ref().unwrap();
        assert_eq!(policy.seat_weights, Some(vec![2.0, 1.0]));
        assert_eq!(
            policy.seat_identity.as_ref().unwrap().ids[policy.primary_seat],
            "B"
        );
        assert_eq!(
            serde_json::to_value(&original.speakers).unwrap(),
            original_inventory
        );
        assert!(receipt.checks.iter().all(|check| check.passed));
        let SpeakerConfig::Single(main) = &normalized.speakers["main"] else {
            panic!()
        };
        assert_eq!(
            crate::group_measurements::routed_seat_identity_order(
                &normalized,
                main,
                &normalized.speakers["sub"],
                2,
                2
            )
            .unwrap(),
            vec!["A", "B"]
        );
        let audit: serde_json::Value =
            serde_json::from_str(receipt.checks[0].diagnostic.as_ref().unwrap()).unwrap();
        assert_eq!(
            audit["canonical_to_original_permutation"],
            serde_json::json!([1, 0])
        );
        assert_eq!(audit["primary_physical_id"], "B");
    }

    #[test]
    fn identity_join_refuses_missing_duplicate_different_singleton_and_conflicting_weights() {
        for labels in [vec!["A", "A"], vec!["A", "C"], vec!["A"], vec!["", "B"]] {
            let configured = config(source(&["A", "B"]), source(&labels));
            let (normalized, _) = normalize_seat_identity(&configured);
            let SpeakerConfig::Single(main) = &normalized.speakers["main"] else {
                panic!()
            };
            assert!(
                crate::group_measurements::routed_seat_identity_order(
                    &normalized,
                    main,
                    &normalized.speakers["sub"],
                    2,
                    labels.len()
                )
                .is_err()
            );
        }
        let mut configured = config(source(&["A", "B"]), source(&["A", "B"]));
        configured.optimizer.multi_measurement = Some(MultiMeasurementConfig {
            weights: Some(vec![1.0, 2.0]),
            ..Default::default()
        });
        let SpeakerConfig::Single(main) = &configured.speakers["main"] else {
            panic!()
        };
        assert_eq!(
            crate::group_measurements::routed_seat_identity_order(
                &configured,
                main,
                &configured.speakers["sub"],
                2,
                2
            )
            .unwrap_err(),
            "conflicting_configured_physical_seat_weights"
        );
        configured
            .optimizer
            .multi_measurement
            .as_mut()
            .unwrap()
            .weights = Some(vec![4.0, 2.0]);
        assert!(
            crate::group_measurements::routed_seat_identity_order(
                &configured,
                main,
                &configured.speakers["sub"],
                2,
                2
            )
            .is_ok()
        );
    }

    #[test]
    fn identity_join_refuses_inline_capture_without_verified_projection() {
        let mut configured = config(source(&["A", "B"]), source(&["A", "B"]));
        configured
            .optimizer
            .multi_seat
            .as_mut()
            .unwrap()
            .seat_identity = None;
        let SpeakerConfig::Single(main) = &configured.speakers["main"] else {
            panic!()
        };
        assert!(
            crate::group_measurements::routed_seat_identity_order(
                &configured,
                main,
                &configured.speakers["sub"],
                2,
                2
            )
            .is_err()
        );
        let SpeakerConfig::Single(MeasurementSource::Multiple(sub)) =
            configured.speakers.get_mut("sub").unwrap()
        else {
            panic!()
        };
        sub.provenance.capture.as_mut().unwrap().takes[1].position_m[0] = 3.0;
        let SpeakerConfig::Single(main) = &configured.speakers["main"] else {
            panic!()
        };
        assert_eq!(
            crate::group_measurements::routed_seat_identity_order(
                &configured,
                main,
                &configured.speakers["sub"],
                2,
                2
            )
            .unwrap_err(),
            "fixed_projection_identity_requires_verified_ref_path_take_receipt"
        );
        let SpeakerConfig::Single(MeasurementSource::Multiple(sub)) =
            configured.speakers.get_mut("sub").unwrap()
        else {
            panic!()
        };
        sub.provenance.capture = None;
        let SpeakerConfig::Single(main) = &configured.speakers["main"] else {
            panic!()
        };
        assert!(
            crate::group_measurements::routed_seat_identity_order(
                &configured,
                main,
                &configured.speakers["sub"],
                2,
                2
            )
            .is_err()
        );
    }
    #[test]
    fn identity_join_refuses_failed_acquisition_and_conflicting_explicit_position() {
        let mut configured = config(source(&["A", "B"]), source(&["A", "B"]));
        let SpeakerConfig::Single(MeasurementSource::Multiple(sub)) =
            configured.speakers.get_mut("sub").unwrap()
        else {
            panic!()
        };
        sub.provenance.capture.as_mut().unwrap().takes[1].position_m[0] = 3.0;
        let SpeakerConfig::Single(main) = &configured.speakers["main"] else {
            panic!()
        };
        assert_eq!(
            crate::group_measurements::routed_seat_identity_order(
                &configured,
                main,
                &configured.speakers["sub"],
                2,
                2
            )
            .unwrap_err(),
            "explicit_identity_conflicts_with_physical_acquisition_positions"
        );
        configured
            .optimizer
            .multi_seat
            .as_mut()
            .unwrap()
            .seat_identity = None;
        let SpeakerConfig::Single(MeasurementSource::Multiple(sub)) =
            configured.speakers.get_mut("sub").unwrap()
        else {
            panic!()
        };
        sub.provenance.capture.as_mut().unwrap().takes[1].position_m[0] = 1.0;
        sub.provenance.capture.as_mut().unwrap().takes[1].quality_passed = false;
        let SpeakerConfig::Single(main) = &configured.speakers["main"] else {
            panic!()
        };
        assert!(
            crate::group_measurements::routed_seat_identity_order(
                &configured,
                main,
                &configured.speakers["sub"],
                2,
                2
            )
            .is_err()
        );
    }
    #[test]
    fn identity_join_refuses_finite_weights_whose_sum_overflows() {
        let mut configured = config(source(&["A", "B"]), source(&["A", "B"]));
        configured
            .optimizer
            .multi_seat
            .as_mut()
            .unwrap()
            .seat_weights = Some(vec![f64::MAX, f64::MAX]);
        configured.optimizer.multi_measurement = Some(MultiMeasurementConfig {
            weights: Some(vec![f64::MAX, f64::MAX / 2.0]),
            ..Default::default()
        });
        let SpeakerConfig::Single(main) = &configured.speakers["main"] else {
            panic!()
        };
        assert_eq!(
            crate::group_measurements::routed_seat_identity_order(
                &configured,
                main,
                &configured.speakers["sub"],
                2,
                2
            )
            .unwrap_err(),
            "nonfinite_or_zero_physical_seat_weight_sum"
        );
    }
    #[test]
    fn identity_join_preserves_positive_scaled_direct_weights() {
        let mut configured = config(source(&["A", "B"]), source(&["A", "B"]));
        configured
            .optimizer
            .multi_seat
            .as_mut()
            .unwrap()
            .seat_weights = Some(vec![1e-100, 2e-100]);
        configured.optimizer.multi_measurement = Some(MultiMeasurementConfig {
            weights: Some(vec![1.0, 2.0]),
            ..Default::default()
        });
        let SpeakerConfig::Single(main) = &configured.speakers["main"] else {
            panic!()
        };
        assert_eq!(
            crate::group_measurements::routed_seat_identity_order(
                &configured,
                main,
                &configured.speakers["sub"],
                2,
                2
            )
            .unwrap(),
            vec!["A", "B"]
        );
        // Direct policy is preserved. The separate all-channel derivation's
        // existing EPS advisory remains untouched and is not used here.
        assert!(
            crate::home_cinema::derive_all_channel_multiseat_config(&configured, "main", main)
                .is_none()
        );
    }

    #[test]
    fn explicit_seat_binding_supports_moving_hardware_without_relaxing_clock_gate() {
        let mut configured = config(source(&["B", "A"]), source(&["A", "B"]));
        for speaker in configured.speakers.values_mut() {
            let SpeakerConfig::Single(MeasurementSource::Multiple(multiple)) = speaker else {
                panic!()
            };
            for take in &mut multiple.provenance.capture.as_mut().unwrap().takes {
                take.microphone_id = "same-moving-physical-microphone".into();
            }
            assert!(
                multiple
                    .provenance
                    .capture
                    .as_ref()
                    .unwrap()
                    .coherent_reference(2)
                    .is_err(),
                "identity binding must not relax the separate unique-hardware clock eligibility contract"
            );
        }
        let (normalized, receipt) = normalize_seat_identity(&configured);
        assert!(receipt.checks.iter().all(|check| check.passed));
        let SpeakerConfig::Single(main) = &normalized.speakers["main"] else {
            panic!()
        };
        assert_eq!(
            crate::group_measurements::routed_seat_identity_order(
                &normalized,
                main,
                &normalized.speakers["sub"],
                2,
                2
            )
            .unwrap(),
            vec!["A", "B"]
        );
    }

    #[test]
    fn explicit_take_binding_contradiction_refuses_normalization_and_join() {
        let mut configured = config(source(&["B", "A"]), source(&["A", "B"]));
        let SpeakerConfig::Single(MeasurementSource::Multiple(main)) =
            configured.speakers.get_mut("main").unwrap()
        else {
            panic!()
        };
        main.provenance.capture.as_mut().unwrap().takes[0].seat_id = Some("A".into());
        let original = serde_json::to_value(&configured.speakers).unwrap();
        let (normalized, receipt) = normalize_seat_identity(&configured);
        assert_eq!(
            serde_json::to_value(&normalized.speakers).unwrap(),
            original
        );
        assert!(receipt.checks.iter().any(|check| !check.passed));
        let SpeakerConfig::Single(main) = &normalized.speakers["main"] else {
            panic!()
        };
        assert_eq!(
            crate::group_measurements::routed_seat_identity_order(
                &normalized,
                main,
                &normalized.speakers["sub"],
                2,
                2
            )
            .unwrap_err(),
            "explicit_seat_binding_contradicts_measurement_identity"
        );
    }
}
