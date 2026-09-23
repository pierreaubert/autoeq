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
