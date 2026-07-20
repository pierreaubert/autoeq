//! Measurement-source preparation for multi-sub and group workflows.

use roomeq_engine::Curve;
use roomeq_engine::error::{AutoeqError, Result};
use roomeq_model::MultiSubGroup;

/// Load and validate per-subwoofer seat measurements before engine execution.
pub fn load_multisub_seat_measurements(group: &MultiSubGroup) -> Result<Option<Vec<Vec<Curve>>>> {
    let mut per_sub = Vec::with_capacity(group.subwoofers.len());
    let mut expected_seats = None;
    let mut any_multi_seat = false;

    for (sub_index, source) in group.subwoofers.iter().enumerate() {
        let curves = autoeq_measurements::load_source_individual(source).map_err(|error| {
            AutoeqError::InvalidMeasurement {
                message: format!(
                    "Failed to load seat measurements for sub {sub_index} in group '{}': {error}",
                    group.name
                ),
            }
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
    use roomeq_model::MeasurementSource;

    use super::*;

    fn curve() -> Curve {
        Curve {
            freq: array![100.0, 200.0, 400.0],
            spl: array![80.0, 80.0, 80.0],
            ..Curve::default()
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
}
