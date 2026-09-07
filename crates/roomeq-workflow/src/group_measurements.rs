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
        per_sub.push(curves);
    }

    if any_multi_seat && expected_seats.unwrap_or(0) >= 2 {
        Ok(Some(per_sub))
    } else {
        Ok(None)
    }
}

#[cfg(test)]
mod tests {
    use ndarray::array;
    use roomeq_model::{InlineMeasurement, MeasurementMultiple, MeasurementRef, MeasurementSource};

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
        })
    }

    fn group_of(sources: Vec<MeasurementSource>) -> MultiSubGroup {
        MultiSubGroup {
            name: "subs".to_string(),
            speaker_name: None,
            subwoofers: sources,
            allpass_optimization: false,
        }
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
        };
        assert!(load_multisub_seat_measurements(&group).unwrap().is_none());
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
