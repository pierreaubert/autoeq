//! Measurement-source preparation for multi-sub and group workflows.

use crate::measurement::load_source_individual_with_frequency_samples;
use roomeq_engine::Curve;
use roomeq_engine::error::{AutoeqError, Result};
use roomeq_model::{MeasurementSource, MultiSubGroup};

/// Seat labels declared by one subwoofer source, in seat order.
///
/// Returns `None` when any seat is unnamed (plain paths, in-memory curves):
/// without a complete label vector there is nothing to compare against, and
/// the legacy positional (index-order) correspondence applies. Callers must
/// keep unnamed inputs in identical seat order across subwoofers.
pub(crate) fn seat_labels(source: &MeasurementSource) -> Option<Vec<String>> {
    match source {
        MeasurementSource::Single(single) => {
            single.measurement.name().map(|name| vec![name.to_string()])
        }
        MeasurementSource::Multiple(multiple) => multiple
            .measurements
            .iter()
            .map(|measurement| measurement.name().map(str::to_string))
            .collect(),
        MeasurementSource::InMemory(_) | MeasurementSource::InMemoryMultiple(_) => None,
    }
}

/// Derive common timing scopes from labeled stationary acquisition records.
///
/// Returns no authorization for unknown, moving-microphone, mismatched, or
/// unlabeled captures. Matrix axes match the loader: `[sub][seat]`.
pub(crate) fn multisub_reference_scope(
    group: &MultiSubGroup,
    band_hz: [f64; 2],
) -> Option<Vec<Vec<String>>> {
    multisub_source_reference_scope(&group.subwoofers, band_hz)
}

pub(crate) fn multisub_source_reference_scope(
    sources: &[MeasurementSource],
    band_hz: [f64; 2],
) -> Option<Vec<Vec<String>>> {
    use autoeq_core::ProvenanceCaptureKind;
    let mut expected_labels: Option<Vec<String>> = None;
    let mut expected_reference: Option<String> = None;
    let mut scopes = Vec::with_capacity(sources.len());
    if !band_hz[0].is_finite()
        || !band_hz[1].is_finite()
        || band_hz[0] <= 0.0
        || band_hz[0] >= band_hz[1]
    {
        return None;
    }
    for source in sources {
        let provenance = source.provenance();
        if !matches!(
            provenance.capture_kind,
            ProvenanceCaptureKind::StationaryIr | ProvenanceCaptureKind::DirectSound
        ) {
            return None;
        }
        if let Some(bands) = provenance.declared_support_bands().ok()?
            && !bands
                .iter()
                .any(|[lo, hi]| *lo <= band_hz[0] && *hi >= band_hz[1])
        {
            return None;
        }
        if provenance.capture_kind == ProvenanceCaptureKind::DirectSound
            || provenance.direct_sound.is_some()
        {
            let report = crate::evidence_intake::assess_direct_capture(
                &provenance,
                band_hz,
                &crate::evidence_intake::source_measurement_id(source),
            )
            .ok()?;
            if report.phase_source
                != roomeq_engine::analysis::quasi_anechoic::PhaseSourceVerdict::Supported
                || report.valid_lower_hz.is_none_or(|lo| lo > band_hz[0])
                || report.valid_upper_hz.is_none_or(|hi| hi < band_hz[1])
            {
                return None;
            }
        }
        if let Some(capture) = &provenance.capture {
            capture
                .coherent_reference_at_frequency(capture.takes.len(), band_hz[1])
                .ok()?;
        }
        let reference = provenance.timing_reference_id?;
        if reference.trim().is_empty() || reference.trim().eq_ignore_ascii_case("unknown") {
            return None;
        }
        let labels = seat_labels(source)?;
        if labels.is_empty()
            || labels.iter().any(|label| label.trim().is_empty())
            || labels
                .iter()
                .collect::<std::collections::HashSet<_>>()
                .len()
                != labels.len()
            || expected_labels
                .as_ref()
                .is_some_and(|expected| expected != &labels)
            || expected_reference
                .as_ref()
                .is_some_and(|expected| expected != &reference)
        {
            return None;
        }
        scopes.push(vec![reference.clone(); labels.len()]);
        expected_labels = Some(labels);
        expected_reference = Some(reference);
    }
    (!scopes.is_empty()).then_some(scopes)
}

/// Load and validate per-subwoofer seat measurements before engine execution.
pub fn load_multisub_seat_measurements(group: &MultiSubGroup) -> Result<Option<Vec<Vec<Curve>>>> {
    load_multisub_seat_measurements_with_frequency_samples(group, crate::DEFAULT_FREQUENCY_SAMPLES)
}

/// Load multi-sub seat measurements using a configurable frequency grid.
pub fn load_multisub_seat_measurements_with_frequency_samples(
    group: &MultiSubGroup,
    frequency_samples: usize,
) -> Result<Option<Vec<Vec<Curve>>>> {
    let mut per_sub = Vec::with_capacity(group.subwoofers.len());
    let mut expected_seats = None;
    let mut expected_labels: Option<(usize, Vec<String>)> = None;
    let mut any_multi_seat = false;

    for (sub_index, source) in group.subwoofers.iter().enumerate() {
        let curves = load_source_individual_with_frequency_samples(source, frequency_samples)
            .map_err(|error| AutoeqError::InvalidMeasurement {
                message: format!(
                    "Failed to load seat measurements for sub {sub_index} in group '{}': {error}",
                    group.name
                ),
            })?;
        // A one-seat source can carry a descriptive subwoofer name; that name
        // is not a listening-seat identity.  Still establish the cardinality
        // up front so a mixed one-seat/multi-seat group is rejected rather
        // than silently pairing positions by accident.
        if let Some(expected) = expected_seats {
            if curves.len() != expected && (curves.len() > 1 || expected > 1) {
                return Err(AutoeqError::InvalidConfiguration {
                    message: format!(
                        "Multi-seat multi-sub group '{}' inconsistent seat counts: sub 0 {}, sub {} {}",
                        group.name,
                        expected,
                        sub_index,
                        curves.len()
                    ),
                });
            }
        } else {
            expected_seats = Some(curves.len());
        }

        if curves.len() > 1 {
            any_multi_seat = true;
        }
        match expected_seats {
            Some(expected) if curves.len() != expected => {
                return Err(AutoeqError::InvalidConfiguration {
                    message: format!(
                        "Multi-seat multi-sub group '{}' has inconsistent seat counts: sub 0 has {}, sub {} has {}",
                        group.name,
                        expected,
                        sub_index,
                        curves.len()
                    ),
                });
            }
            None => expected_seats = Some(curves.len()),
            _ => {}
        }
        // Seat-identity mapping: when every subwoofer labels every seat,
        // the label order must agree — index `i` of each sub is summed as
        // one physical seat, so a swapped order would silently combine
        // different listening positions. Unlabeled inputs keep the legacy
        // positional contract (identical order required, unchecked).
        if any_multi_seat {
            match (seat_labels(source), &expected_labels) {
                (Some(labels), Some((reference, expected))) if labels != *expected => {
                    return Err(AutoeqError::InvalidConfiguration {
                        message: format!(
                            "Multi-seat multi-sub group '{}' has inconsistent seat order: sub {reference} is [{}], sub {} is [{}]; \
                         reorder the measurements so each index is the same physical seat",
                            group.name,
                            expected.join(", "),
                            sub_index,
                            labels.join(", ")
                        ),
                    });
                }
                (Some(labels), None) => expected_labels = Some((sub_index, labels)),
                _ => {}
            }
        }
        per_sub.push(curves);
    }

    if any_multi_seat && expected_seats.unwrap_or(0) >= 2 {
        Ok(Some(per_sub))
    } else {
        Ok(None)
    }
}

/// Resolve a declared physical seat order without positional fallbacks.
pub(crate) fn seat_permutation(
    canonical: &[String],
    declared: &[String],
) -> std::result::Result<Vec<usize>, String> {
    use std::collections::HashSet;
    let valid = |ids: &[String]| {
        !ids.is_empty()
            && ids.iter().all(|id| !id.trim().is_empty())
            && ids.iter().collect::<HashSet<_>>().len() == ids.len()
    };
    if !valid(canonical) || !valid(declared) {
        return Err("missing_or_duplicate_physical_seat_identity".into());
    }
    if canonical.len() != declared.len() {
        return Err("physical_seat_identity_support_mismatch_no_singleton_broadcast".into());
    }
    canonical
        .iter()
        .map(|id| {
            declared
                .iter()
                .position(|candidate| candidate == id)
                .ok_or_else(|| "different_physical_seat_identity_sets".into())
        })
        .collect()
}

pub(crate) fn speaker_measurement_sources(
    speaker: &roomeq_model::SpeakerConfig,
) -> Vec<&MeasurementSource> {
    use roomeq_model::SpeakerConfig;
    match speaker {
        SpeakerConfig::Single(source) => vec![source],
        SpeakerConfig::Topology(topology) => topology
            .drivers
            .iter()
            .map(|driver| &driver.measurement)
            .collect(),
        SpeakerConfig::Group(group) => group.measurements.iter().collect(),
        SpeakerConfig::MultiSub(group) => group.subwoofers.iter().collect(),
        SpeakerConfig::Dba(group) => group.front.iter().chain(&group.rear).collect(),
        SpeakerConfig::Cardioid(group) => vec![&group.front, &group.rear],
        SpeakerConfig::SupportingSource(group) => vec![&group.primary, &group.support],
    }
}

/// Check that ordered take metadata explicitly binds each measurement's seat.
pub(crate) fn validate_source_seat_bindings(
    source: &MeasurementSource,
) -> std::result::Result<(), String> {
    let labels = seat_labels(source).ok_or("missing_physical_seat_identity")?;
    let provenance = source.provenance();
    let verified = match source {
        MeasurementSource::Multiple(multiple) => provenance
            .verified_fixed_projection
            .as_ref()
            .is_some_and(|receipt| receipt.matches_snapshot(multiple)),
        _ => false,
    };
    if provenance.verified_fixed_projection.is_some() && !verified {
        return Err("stale_fixed_capture_projection_receipt".into());
    }
    if let Some(capture) = provenance.capture.as_ref() {
        if capture.takes.len() != labels.len() {
            return Err("physical_acquisition_identity_support_mismatch".into());
        }
        for (take, label) in capture.takes.iter().zip(labels) {
            match take.seat_id.as_deref() {
                Some(seat) if seat == label => {}
                Some(_) => {
                    return Err("explicit_seat_binding_contradicts_measurement_identity".into());
                }
                None if verified && take.microphone_id == label => {}
                None => return Err("physical_seat_binding_missing_from_unverified_capture".into()),
            }
        }
    }
    Ok(())
}

/// Validate identity and weight ownership for a coherent Main/Sub seat join.
///
/// Display labels alone authorize no physical-seat join. Explicit configured
/// IDs or matching ordered acquisition identities and positions are required.
pub(crate) fn routed_seat_identity_order(
    config: &roomeq_model::RoomConfig,
    main: &MeasurementSource,
    sub: &roomeq_model::SpeakerConfig,
    main_count: usize,
    sub_count: usize,
) -> std::result::Result<Vec<String>, String> {
    let mut sources = vec![main];
    sources.extend(speaker_measurement_sources(sub));
    let declared: Vec<Vec<String>> = sources
        .iter()
        .map(|source| {
            seat_labels(source).ok_or_else(|| "missing_physical_seat_identity".to_string())
        })
        .collect::<std::result::Result<_, _>>()?;
    for source in &sources {
        validate_source_seat_bindings(source)?;
    }
    let policy = config.optimizer.multi_seat.as_ref();
    let explicit = policy.and_then(|policy| policy.seat_identity.as_ref());
    let canonical = explicit.map_or_else(|| declared[0].clone(), |map| map.ids.clone());
    if main_count != canonical.len() || sub_count != canonical.len() {
        return Err("physical_seat_identity_support_mismatch_no_singleton_broadcast".into());
    }
    for labels in &declared {
        let permutation = seat_permutation(&canonical, labels)?;
        if permutation
            .iter()
            .enumerate()
            .any(|(index, actual)| index != *actual)
        {
            return Err("physical_seat_identity_not_normalized_before_primary_selection".into());
        }
    }
    if explicit.is_none() {
        let mut reference_positions = None;
        let mut reference_clock = None;
        let mut reference_session = None;
        for (source, labels) in sources.iter().zip(&declared) {
            let MeasurementSource::Multiple(multiple) = source else {
                return Err("fixed_projection_identity_requires_verified_multiple_capture".into());
            };
            if !multiple
                .provenance
                .verified_fixed_projection
                .as_ref()
                .is_some_and(|receipt| receipt.matches_snapshot(multiple))
            {
                return Err(
                    "fixed_projection_identity_requires_verified_ref_path_take_receipt".into(),
                );
            }
            let session = multiple
                .provenance
                .verified_fixed_projection
                .as_ref()
                .unwrap()
                .session_id();
            if reference_session
                .as_ref()
                .is_some_and(|reference: &String| reference != session)
            {
                return Err("different_verified_capture_sessions".into());
            }
            reference_session = Some(session.to_string());
            let provenance = source.provenance();
            let capture = provenance
                .capture
                .as_ref()
                .ok_or_else(|| "labels_without_physical_acquisition_identity".to_string())?;
            if capture.takes.len() != labels.len() {
                return Err("physical_acquisition_identity_support_mismatch".into());
            }
            let clock = capture
                .coherent_reference(labels.len())
                .map_err(|reason| format!("unverified_physical_acquisition_contract: {reason}"))?;
            if reference_clock
                .as_ref()
                .is_some_and(|reference: &String| reference != clock)
            {
                return Err("different_physical_acquisition_reference_frames".into());
            }
            reference_clock = Some(clock.to_string());
            let mut positions = Vec::new();
            for (take, label) in capture.takes.iter().zip(labels) {
                if take.microphone_id != *label
                    || take.position_m.iter().any(|value| !value.is_finite())
                {
                    return Err("physical_acquisition_identity_label_or_position_mismatch".into());
                }
                positions.push(take.position_m);
            }
            if reference_positions
                .as_ref()
                .is_some_and(|reference| reference != &positions)
            {
                return Err("different_physical_acquisition_positions".into());
            }
            reference_positions = Some(positions);
        }
    }
    if explicit.is_some() {
        let mut reference_positions = None;
        for source in &sources {
            let provenance = source.provenance();
            if let Some(capture) = provenance.capture.as_ref() {
                if capture.takes.len() != canonical.len() {
                    return Err("physical_acquisition_identity_support_mismatch".into());
                }
                let positions: Vec<_> = capture.takes.iter().map(|take| take.position_m).collect();
                if positions.iter().flatten().any(|value| !value.is_finite()) {
                    return Err("invalid_physical_acquisition_position".into());
                }
                if reference_positions
                    .as_ref()
                    .is_some_and(|reference| reference != &positions)
                {
                    return Err(
                        "explicit_identity_conflicts_with_physical_acquisition_positions".into(),
                    );
                }
                reference_positions = Some(positions);
            }
        }
    }
    let count = canonical.len();
    if policy.is_some_and(|policy| policy.primary_seat >= count) {
        return Err("primary_physical_seat_identity_unavailable".into());
    }
    let main_weights = policy.and_then(|policy| policy.seat_weights.as_ref());
    let optimizer_weights = config
        .optimizer
        .multi_measurement
        .as_ref()
        .and_then(|policy| policy.weights.as_ref());
    let normalized = |weights: &Vec<f64>| -> std::result::Result<Vec<f64>, String> {
        if weights.len() != count
            || weights
                .iter()
                .any(|weight| !weight.is_finite() || *weight < 0.0)
        {
            return Err("invalid_physical_seat_weight_support".into());
        }
        let sum: f64 = weights.iter().sum();
        if !sum.is_finite() || sum <= 0.0 {
            return Err("nonfinite_or_zero_physical_seat_weight_sum".into());
        }
        Ok(weights.iter().map(|weight| weight / sum).collect())
    };
    let main_weights = main_weights.map(normalized).transpose()?;
    let optimizer_weights = optimizer_weights.map(normalized).transpose()?;
    if let (Some(main), Some(optimizer)) = (main_weights, optimizer_weights)
        && main
            .iter()
            .zip(optimizer)
            .any(|(a, b)| (*a - b).abs() > 1e-12)
    {
        return Err("conflicting_configured_physical_seat_weights".into());
    }
    Ok(canonical)
}

#[cfg(test)]
mod tests {
    use ndarray::array;
    use roomeq_model::{
        InlineMeasurement, MeasurementMultiple, MeasurementRef, MeasurementSource, SpeakerConfig,
    };

    use super::*;

    fn curve() -> Curve {
        Curve {
            freq: array![100.0, 200.0, 400.0],
            spl: array![80.0, 80.0, 80.0],
            ..Curve::default()
        }
    }

    fn named_inline(name: &str) -> MeasurementRef {
        MeasurementRef::Inline(InlineMeasurement {
            frequencies: vec![100.0, 200.0, 400.0],
            magnitude_db: vec![80.0, 80.0, 80.0],
            phase_deg: None,
            name: Some(name.to_string()),
            wav_path: None,
            csv_path: None,
        })
    }

    fn named_multiple(names: &[&str]) -> MeasurementSource {
        MeasurementSource::Multiple(MeasurementMultiple {
            measurements: names.iter().map(|name| named_inline(name)).collect(),
            speaker_name: None,
            provenance: Default::default(),
        })
    }

    fn group_of(sources: Vec<MeasurementSource>) -> MultiSubGroup {
        MultiSubGroup {
            name: "subs".to_string(),
            speaker_name: None,
            subwoofers: sources,
            allpass_optimization: false,
            joint_optimization: false,
        }
    }

    #[test]
    fn roadmap_correction_joint_scope_comes_from_stationary_capture_provenance() {
        let stationary = |reference: &str| {
            let mut source = named_multiple(&["seat-a", "seat-b", "seat-c"]);
            if let MeasurementSource::Multiple(multiple) = &mut source {
                multiple.provenance.capture_kind = autoeq_core::ProvenanceCaptureKind::StationaryIr;
                multiple.provenance.timing_reference_id = Some(reference.to_string());
            }
            source
        };
        let mut group = group_of(vec![stationary("clock-a"), stationary("clock-a")]);
        let scope = multisub_reference_scope(&group, [20.0, 200.0])
            .expect("declared common capture reference");
        assert_eq!(scope, vec![vec!["clock-a".to_string(); 3]; 2]);
        if let MeasurementSource::Multiple(multiple) = &mut group.subwoofers[1] {
            multiple.provenance.capture_kind = autoeq_core::ProvenanceCaptureKind::DirectSound;
        }
        assert!(
            multisub_reference_scope(&group, [20.0, 200.0]).is_none(),
            "direct capture needs facts, not only a timing ID"
        );
        if let MeasurementSource::Multiple(multiple) = &mut group.subwoofers[1] {
            multiple.provenance.direct_sound =
                Some(autoeq_core::direct_sound::DirectSoundEvidence {
                    facts: autoeq_core::direct_sound::DirectSoundCaptureFacts {
                        gate_s: Some(0.002),
                        direct_path_m: Some(1.0),
                        first_reflection_path_m: Some(40.0),
                        averaging: autoeq_core::direct_sound::AveragingMethod::Stationary,
                        capture_kind: autoeq_core::evidence::CaptureKind::DirectSound,
                        sample_rate_hz: Some(48_000.0),
                        ..Default::default()
                    },
                    policy: Some(autoeq_core::direct_sound::QuasiAnechoicPolicy::v1()),
                });
        }
        assert!(
            multisub_reference_scope(&group, [20.0, 200.0]).is_none(),
            "2 ms does not support coherent bass optimization"
        );
        assert!(
            multisub_reference_scope(&group, [1000.0, 2000.0]).is_some(),
            "same phase capture supports its actual band without angular detail claims"
        );
        group.subwoofers[1] = stationary("clock-b");
        assert!(multisub_reference_scope(&group, [20.0, 200.0]).is_none());
        group.subwoofers[1] = stationary("clock-a");
        if let MeasurementSource::Multiple(multiple) = &mut group.subwoofers[1] {
            multiple.provenance.capture_kind = autoeq_core::ProvenanceCaptureKind::SpatialMagnitude;
        }
        assert!(
            multisub_reference_scope(&group, [20.0, 200.0]).is_none(),
            "MMM cannot authorize timing"
        );
        group.subwoofers[1] = named_multiple(&["seat-a", "seat-b", "seat-c"]);
        assert!(
            multisub_reference_scope(&group, [20.0, 200.0]).is_none(),
            "unknown provenance stays unknown"
        );
    }

    #[test]
    fn rejects_inconsistent_seat_counts() {
        let group = MultiSubGroup {
            name: "subs".to_string(),
            speaker_name: None,
            subwoofers: vec![
                MeasurementSource::InMemoryMultiple(vec![curve(), curve()]),
                MeasurementSource::InMemoryMultiple(vec![curve()]),
            ],
            allpass_optimization: false,
            joint_optimization: false,
        };
        let error = load_multisub_seat_measurements(&group).unwrap_err();
        assert!(error.to_string().contains("inconsistent seat counts"));
    }

    #[test]
    fn returns_none_for_single_seat_sources() {
        let group = MultiSubGroup {
            name: "subs".to_string(),
            speaker_name: None,
            subwoofers: vec![
                MeasurementSource::InMemoryMultiple(vec![curve()]),
                MeasurementSource::InMemoryMultiple(vec![curve()]),
            ],
            allpass_optimization: false,
            joint_optimization: false,
        };
        assert!(load_multisub_seat_measurements(&group).unwrap().is_none());
    }

    #[test]
    fn measured_sigberg_two_sub_config_loads_as_independent_subs() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../data_tests/roomeq/measured/2.2_sigberg2/recordings.json");
        if !path.is_file() {
            return;
        }
        let (config, _, _) = crate::config_loader::load_config(&path, None).unwrap();
        let outputs = &config
            .system
            .as_ref()
            .unwrap()
            .subwoofers
            .as_ref()
            .unwrap()
            .outputs;
        assert_eq!(outputs.len(), 2);
        assert!(outputs.iter().all(|output| {
            matches!(
                config.speakers.get(&output.speaker),
                Some(SpeakerConfig::Single(_))
            )
        }));
    }

    #[test]
    fn rejects_permuted_named_seat_order() {
        // The release P1 risk: sub A=[MLP, left], sub B=[left, MLP] must
        // fail instead of silently summing different positions per index.
        let group = group_of(vec![
            named_multiple(&["MLP", "left"]),
            named_multiple(&["left", "MLP"]),
        ]);
        let error = load_multisub_seat_measurements(&group).unwrap_err();
        let message = error.to_string();
        assert!(
            message.contains("inconsistent seat order"),
            "unexpected error: {message}"
        );
        assert!(message.contains("MLP") && message.contains("left"));
    }

    #[test]
    fn accepts_matching_named_seat_order() {
        let group = group_of(vec![
            named_multiple(&["MLP", "left"]),
            named_multiple(&["MLP", "left"]),
        ]);
        let loaded = load_multisub_seat_measurements(&group).unwrap();
        assert_eq!(loaded.unwrap().len(), 2);
    }

    #[test]
    fn rejects_seat_count_mismatch_for_named_inputs() {
        let group = group_of(vec![
            named_multiple(&["seat-1", "seat-2"]),
            named_multiple(&["seat-1", "seat-2", "seat-3"]),
        ]);
        let error = load_multisub_seat_measurements(&group).unwrap_err();
        assert!(error.to_string().contains("inconsistent seat counts"));
    }

    #[test]
    fn unnamed_inputs_keep_positional_contract() {
        // In-memory curves carry no seat labels: identical order is
        // required by contract but unchecked here (documented fallback).
        let group = group_of(vec![
            MeasurementSource::InMemoryMultiple(vec![curve(), curve()]),
            MeasurementSource::InMemoryMultiple(vec![curve(), curve()]),
        ]);
        let loaded = load_multisub_seat_measurements(&group).unwrap();
        assert_eq!(loaded.unwrap().len(), 2);
    }
}
