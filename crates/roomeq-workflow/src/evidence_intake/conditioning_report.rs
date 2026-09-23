//! Expose recorded optimizer conditioning without inferring missing operations.

use super::ConditioningLedger;
use crate::RoomOptimizationResult;
use roomeq_model::{StageCheck, StageCheckKind, StageOutcome, StageStatus};

// Never reopen original paths while binding the numerical snapshot of this run.
fn frozen_source_identities(
    config: &roomeq_model::RoomConfig,
    channel: &str,
) -> Result<Option<Vec<String>>, String> {
    use autoeq_core::{Curve, MeasurementRef, MeasurementSource};
    fn frozen_curve(reference: &MeasurementRef) -> Option<&Curve> {
        match reference {
            MeasurementRef::Loaded {
                loaded_response, ..
            } => Some(loaded_response.as_ref()),
            _ => None,
        }
    }
    let key = config
        .system
        .as_ref()
        .and_then(|system| system.speakers.get(channel))
        .map_or(channel, String::as_str);
    let Some(roomeq_model::SpeakerConfig::Single(source)) = config.speakers.get(key) else {
        return Ok(None);
    };
    let curves: Option<Vec<&Curve>> = match source {
        MeasurementSource::Single(single) => {
            frozen_curve(&single.measurement).map(|curve| vec![curve])
        }
        MeasurementSource::Multiple(multiple) => {
            multiple.measurements.iter().map(frozen_curve).collect()
        }
        MeasurementSource::InMemory(curve) => Some(vec![curve]),
        MeasurementSource::InMemoryMultiple(curves) => Some(curves.iter().collect()),
    };
    curves
        .map(|curves| {
            curves
                .into_iter()
                .map(|curve| curve.content_hash().map_err(|error| error.to_string()))
                .collect()
        })
        .transpose()
}

// Interpret only the ordinary preparation operations whose attribution contract
// is defined here. This validates identity routing, not numerical DSP execution.
fn prepared_identity_issues(
    receipt: &roomeq_engine::channel_measurements::MeasurementConditioningReceipt,
) -> Vec<String> {
    let mut individual = receipt.native_identities.clone();
    let mut representative = (individual.len() == 1).then(|| individual[0].clone());
    let mut issues = Vec::new();
    for (index, entry) in receipt.entries.iter().enumerate() {
        let source_index = entry
            .parameters
            .get("source_index")
            .and_then(serde_json::Value::as_u64)
            .and_then(|index| usize::try_from(index).ok());
        let valid = entry.version == 1
            && match entry.operation.as_str() {
                "source_overlap_alignment" => {
                    if entry.input_hashes != receipt.native_identities {
                        false
                    } else if let Some(target) =
                        source_index.and_then(|index| individual.get_mut(index))
                    {
                        target.clone_from(&entry.output_hash);
                        if individual.len() == 1 {
                            representative = Some(entry.output_hash.clone());
                        }
                        true
                    } else {
                        false
                    }
                }
                "source_spatial_power_rms" => {
                    if individual.len() > 1 && entry.input_hashes == individual {
                        representative = Some(entry.output_hash.clone());
                        true
                    } else {
                        false
                    }
                }
                "roomeq_dense_grid_conditioning" => {
                    let target = match entry
                        .parameters
                        .get("role")
                        .and_then(serde_json::Value::as_str)
                    {
                        Some("individual") => {
                            source_index.and_then(|index| individual.get_mut(index))
                        }
                        Some("representative")
                            if entry
                                .parameters
                                .get("source_index")
                                .is_some_and(serde_json::Value::is_null) =>
                        {
                            representative.as_mut()
                        }
                        _ => None,
                    };
                    if let Some(target) = target {
                        if entry.input_hashes.len() == 1
                            && entry.input_hashes.first() == Some(&*target)
                        {
                            target.clone_from(&entry.output_hash);
                            true
                        } else {
                            false
                        }
                    } else {
                        false
                    }
                }
                _ => false,
            };
        if !valid {
            issues.push(format!(
                "entry_{index}_has_unsupported_or_inconsistent_source_attribution"
            ));
        }
    }
    if individual != receipt.individual_identities {
        issues.push("prepared_individual_content_count_or_order_mismatch".into());
    }
    if representative.as_ref() != Some(&receipt.representative_identity) {
        issues.push("prepared_representative_attribution_mismatch".into());
    }
    issues
}

#[cfg(test)]
mod attribution_tests {
    use super::prepared_identity_issues;

    #[test]
    fn dense_preparation_preserves_final_source_positions() {
        let curves = [80.0, 82.0]
            .into_iter()
            .map(|level| autoeq_core::Curve {
                freq: ndarray::Array1::from_iter((1..=1000).map(|index| f64::from(index) * 10.0)),
                spl: ndarray::Array1::from_elem(1000, level),
                ..Default::default()
            })
            .collect();
        let prepared =
            crate::channel_measurements::prepare_channel_measurements_with_frequency_samples(
                &autoeq_core::MeasurementSource::InMemoryMultiple(curves),
                64,
            )
            .unwrap();
        let receipt = prepared.conditioning_receipt().unwrap();
        assert!(prepared_identity_issues(&receipt).is_empty());
        let index = receipt
            .entries
            .iter()
            .position(|entry| {
                entry.operation == "roomeq_dense_grid_conditioning"
                    && entry.parameters["role"] == "representative"
            })
            .expect("fixture must execute representative dense conditioning");
        for mutation in 0..4 {
            let mut invalid = receipt.clone();
            let entry = &mut invalid.entries[index];
            match mutation {
                0 => entry.version = 2,
                1 => entry.input_hashes[0] = invalid.native_identities[0].clone(),
                2 => {
                    entry
                        .parameters
                        .insert("role".into(), serde_json::json!("unknown"));
                }
                _ => {
                    entry
                        .parameters
                        .insert("source_index".into(), serde_json::json!(0));
                }
            }
            assert!(
                !prepared_identity_issues(&invalid).is_empty(),
                "mutation {mutation}"
            );
        }
        let mut swapped = receipt;
        swapped.individual_identities.swap(0, 1);
        assert!(!prepared_identity_issues(&swapped).is_empty());
    }
}

pub(crate) fn attach_measurement_conditioning(
    result: &mut RoomOptimizationResult,
    snapshot: &roomeq_model::RoomConfig,
) -> Result<(), String> {
    let mut checks = Vec::new();
    let mut missing = Vec::new();
    let channels: std::collections::BTreeMap<_, _> = result.channel_results.iter().collect();
    let valid_hash = |hash: &str| hash.len() == 64 && hash.bytes().all(|b| b.is_ascii_hexdigit());
    for (channel, output) in channels {
        let Some(receipt) = &output.measurement_conditioning else {
            missing.push(channel.clone());
            continue;
        };
        let initial_identity = output
            .initial_curve
            .content_hash()
            .map_err(|error| error.to_string())?;
        let valid_shape = !receipt.native_identities.is_empty()
            && receipt
                .native_identities
                .iter()
                .all(|hash| valid_hash(hash))
            && valid_hash(&receipt.representative_identity)
            && !receipt.individual_identities.is_empty()
            && receipt
                .individual_identities
                .iter()
                .all(|hash| valid_hash(hash))
            && receipt.entries.iter().all(|entry| {
                entry.version > 0
                    && !entry.operation.trim().is_empty()
                    && !entry.input_hashes.is_empty()
                    && entry.input_hashes.iter().all(|hash| valid_hash(hash))
                    && valid_hash(&entry.output_hash)
            });
        let frozen_identities = frozen_source_identities(snapshot, channel)?;
        let snapshot_binding = match &frozen_identities {
            Some(identities)
                if !identities.is_empty() && *identities == receipt.native_identities =>
            {
                "verified_parsed_curve_snapshot"
            }
            Some(_) => "native_root_order_or_content_mismatch",
            None => "snapshot_or_source_attribution_unavailable",
        };
        // Only producer-recorded roots may start the chain. Inferring roots from
        // unknown ledger inputs would make a missing or corrupted link pass.
        let mut known: std::collections::HashSet<_> = receipt.native_identities.iter().collect();
        let mut chain_issues = prepared_identity_issues(receipt);
        for (index, entry) in receipt.entries.iter().enumerate() {
            if entry.input_hashes.iter().all(|hash| known.contains(hash)) {
                known.insert(&entry.output_hash);
            } else {
                chain_issues.push(format!("entry_{index}_has_unavailable_input"));
            }
        }
        if !known.contains(&receipt.representative_identity) {
            chain_issues.push("representative_not_reachable_from_native_roots".into());
        }
        for (index, identity) in receipt.individual_identities.iter().enumerate() {
            if !known.contains(identity) {
                chain_issues.push(format!(
                    "individual_{index}_not_reachable_from_native_roots"
                ));
            }
        }
        checks.push(StageCheck {
            id: format!("measurement-conditioning:{channel}"),
            kind: StageCheckKind::Structural,
            passed: snapshot_binding == "verified_parsed_curve_snapshot" && valid_shape && chain_issues.is_empty() && receipt.representative_identity == initial_identity,
            observed: Some(receipt.entries.len() as f64),
            limit: None,
            diagnostic: Some(serde_json::to_string(&serde_json::json!({
                "channel": channel,
                "receipt": receipt,
                "channel_initial_curve_identity": initial_identity,
                "chain_issues": chain_issues,
                "chain_scope": "ordered reachability and final source-position attribution for known preparation operations; not numerical operation replay or independent root authentication",
                "scope": "producer-recorded source loading and dense-grid preparation; not subsequent engine processing, playback gain, acquisition authentication, or complete lineage",
                "source_snapshot_binding": snapshot_binding,
                "frozen_native_identities": frozen_identities,
                "source_indices": "input order within this channel preparation, not independently authenticated physical seat identities"
            })).map_err(|error| error.to_string())?),
        });
    }
    let status = if !missing.is_empty() || checks.iter().any(|check| !check.passed) {
        StageStatus::Degraded
    } else if checks.is_empty() {
        StageStatus::Skipped
    } else {
        StageStatus::Applied
    };
    result
        .metadata
        .stage_outcomes
        .retain(|stage| stage.stage != "measurement_input_conditioning");
    result.metadata.stage_outcomes.push(StageOutcome {
        stage: "measurement_input_conditioning".into(),
        status,
        checks,
        advisories: vec![
            "partial_conditioning_coverage;_absence_is_not_evidence_of_unchanged_input".into(),
            format!(
                "channels_without_loading_receipts:{}",
                serde_json::to_string(&missing).map_err(|error| error.to_string())?
            ),
        ],
    });
    Ok(())
}

pub(crate) fn attach_optimizer_conditioning(
    result: &mut RoomOptimizationResult,
) -> Result<(), String> {
    let mut checks = Vec::new();
    let mut unrecorded_runs = Vec::new();
    let channels: std::collections::BTreeMap<_, _> = result.channel_results.iter().collect();
    for (channel, output) in channels {
        for (index, run) in output.optimizer_evidence.iter().enumerate() {
            let id = format!("optimizer-conditioning:{channel}:{index}");
            let mut records = Vec::new();
            if let Some(normalization) = &run.input_normalization {
                records.push((None, None, normalization));
            }
            if let Some(multi) = &run.multi_input_normalization {
                records.extend(multi.objectives.iter().enumerate().map(
                    |(index, normalization)| (Some(index), Some(multi.population), normalization),
                ));
            }
            if records.is_empty() {
                unrecorded_runs.push(id);
                continue;
            }
            for (objective_index, population, normalization) in records {
                let mut ledger = ConditioningLedger::default();
                ledger.record_gain(
                    format!(
                        "normalized-response:{}",
                        normalization.normalized_curve_identity
                    ),
                    normalization.applied_gain_db,
                    format!(
                        "EQ analysis normalization: {:?}",
                        normalization.reference_policy
                    ),
                )?;
                checks.push(StageCheck {
                id: objective_index.map_or_else(|| id.clone(), |index| format!("{id}:objective:{index}")),
                kind: StageCheckKind::Structural,
                passed: true,
                observed: Some(normalization.applied_gain_db),
                limit: None,
                diagnostic: Some(serde_json::to_string(&serde_json::json!({
                    "channel": channel,
                    "optimizer_run_index": index,
                    "objective_index": objective_index,
                    "objective_population": population,
                    "selected_for_output": run.selected_for_output,
                    "normalization": normalization,
                    "conditioning": ledger,
                    "scope": "historical analysis conditioning, not emitted playback gain or calibrated SPL; objective indices are not physical seat identities; repeated preparation identities across optimizer attempts are not cumulative gain steps",
                })).map_err(|error| error.to_string())?),
            });
            }
        }
    }
    result
        .metadata
        .stage_outcomes
        .retain(|stage| stage.stage != "optimizer_input_conditioning");
    result.metadata.stage_outcomes.push(StageOutcome {
        stage: "optimizer_input_conditioning".into(),
        status: if !unrecorded_runs.is_empty() { StageStatus::Degraded } else if checks.is_empty() { StageStatus::Skipped } else { StageStatus::Applied },
        checks,
        advisories: vec![
            "only_explicitly_recorded_optimizer_normalization;_source_loading_receipts_are_reported_separately;_other_engine_conditioning_remains_unreported".into(),
            format!("optimizer_runs_without_conditioning_receipts:{}", serde_json::to_string(&unrecorded_runs).map_err(|error| error.to_string())?),
        ],
    });
    Ok(())
}
