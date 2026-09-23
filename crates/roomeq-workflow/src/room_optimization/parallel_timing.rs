//! Declared acquisition timing required by coherent parallel processing.

use roomeq_model::{ChannelDspChain, Curve, MeasurementSource, RoomConfig, SpeakerConfig};

pub(super) fn validate(
    chain: &ChannelDspChain,
    config: &RoomConfig,
    initial: &Curve,
) -> Result<(), String> {
    validated_sources(chain, config, initial).map(|_| ())
}

/// Resolve every contributing source and validate shared acquisition timing.
///
/// DBA arrays return all cabinets, not just their two aggregate branches.
///
/// # Errors
/// Rejects invalid grids, ambiguous mappings, incomplete branches, or unsupported timing.
pub(super) fn validated_sources(
    chain: &ChannelDspChain,
    config: &RoomConfig,
    initial: &Curve,
) -> Result<Vec<MeasurementSource>, String> {
    initial
        .validate("parallel waveform initial curve")
        .map_err(|error| error.to_string())?;
    let band_hz = [initial.freq[0], initial.freq[initial.freq.len() - 1]];
    let mut mapped_sources = Vec::new();
    if let Some(system) = &config.system {
        if let Some(source) = system.speakers.get(&chain.channel) {
            mapped_sources.push(source.as_str());
        }
        if let Some(subs) = &system.subwoofers {
            mapped_sources.extend(
                subs.outputs
                    .iter()
                    .filter(|output| output.id == chain.channel)
                    .map(|output| output.speaker.as_str()),
            );
        }
    }
    let source_name = mapped_sources.first().copied().unwrap_or(&chain.channel);
    if mapped_sources.iter().any(|source| *source != source_name) {
        return Err("parallel waveform timing unavailable: ambiguous source mapping".into());
    }
    let speaker = config
        .speakers
        .get(source_name)
        .ok_or("parallel waveform timing unavailable: missing source mapping")?;
    let drivers = chain.drivers.as_deref().unwrap_or_default();
    let mut array_names = None;
    let sources: Vec<MeasurementSource> = match speaker {
        SpeakerConfig::Group(group) => group.measurements.clone(),
        SpeakerConfig::MultiSub(group) => group.subwoofers.clone(),
        SpeakerConfig::Dba(group) => {
            if group.front.is_empty() || group.rear.is_empty() {
                return Err("parallel waveform timing unavailable: empty DBA array".into());
            }
            // Production emits two aggregate branches, not one branch per
            // cabinet. Timing must nevertheless cover every contributing source.
            array_names = Some(["Front Array", "Rear Array"]);
            group.front.iter().chain(&group.rear).cloned().collect()
        }
        SpeakerConfig::Cardioid(group) => {
            array_names = Some(["Front Sub", "Rear Sub"]);
            vec![group.front.clone(), group.rear.clone()]
        }
        SpeakerConfig::Topology(topology) => {
            for driver in drivers {
                if topology
                    .drivers
                    .get(driver.index)
                    .is_none_or(|source| source.id != driver.name)
                {
                    return Err(
                        "parallel waveform timing unavailable: driver/source identity mismatch"
                            .into(),
                    );
                }
            }
            topology
                .drivers
                .iter()
                .map(|driver| driver.measurement.clone())
                .collect()
        }
        _ => {
            return Err(
                "parallel waveform timing unavailable: unsupported source-to-driver mapping".into(),
            );
        }
    };
    let indices: std::collections::BTreeSet<_> =
        drivers.iter().map(|driver| driver.index).collect();
    let expected_count = if let Some(names) = array_names {
        if drivers.iter().any(|driver| {
            names
                .get(driver.index)
                .is_none_or(|name| *name != driver.name)
        }) {
            return Err(
                "parallel waveform timing unavailable: array branch identity mismatch".into(),
            );
        }
        names.len()
    } else {
        sources.len()
    };
    if drivers.is_empty()
        || drivers.len() != expected_count
        || indices != (0..expected_count).collect()
    {
        return Err(
            "parallel waveform timing unavailable: incomplete driver/source mapping".into(),
        );
    }
    crate::group_measurements::multisub_source_reference_scope(&sources, band_hz)
        .ok_or("parallel waveform timing unavailable: captures need matching stationary timing, seat labels, and supported band")?;
    Ok(sources)
}
