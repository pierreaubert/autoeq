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
    let mut array_names: Option<Vec<String>> = None;
    let sources: Vec<MeasurementSource> = match speaker {
        SpeakerConfig::Single(_) => {
            // Routed bass management synthesizes one MultiSub chain from
            // separate physical outputs. Its owner is the first output ID,
            // but acquisition timing must cover every contributing capture.
            let outputs = config
                .system
                .as_ref()
                .and_then(|system| system.subwoofers.as_ref())
                .filter(|subs| subs.outputs.iter().any(|output| output.id == chain.channel))
                .map(|subs| &subs.outputs)
                .ok_or(
                    "parallel waveform timing unavailable: unsupported source-to-driver mapping",
                )?;
            let mut sources = Vec::with_capacity(outputs.len());
            for output in outputs {
                let Some(SpeakerConfig::Single(source)) = config.speakers.get(&output.speaker)
                else {
                    return Err("parallel waveform timing unavailable: physical sub output requires a single capture".into());
                };
                sources.push(source.clone());
            }
            let names: std::collections::BTreeSet<_> =
                outputs.iter().map(|output| &output.id).collect();
            if names.len() != outputs.len()
                || drivers.iter().any(|driver| {
                    outputs
                        .get(driver.index)
                        .is_none_or(|output| output.id != driver.name)
                })
            {
                return Err("parallel waveform timing unavailable: physical sub driver/source identity mismatch".into());
            }
            sources
        }
        SpeakerConfig::Group(group) => group.measurements.clone(),
        SpeakerConfig::MultiSub(group) => group.subwoofers.clone(),
        SpeakerConfig::Dba(group) => {
            if group.front.is_empty() || group.rear.is_empty() {
                return Err("parallel waveform timing unavailable: empty DBA array".into());
            }
            // Production emits two aggregate branches, not one branch per
            // cabinet. Timing must nevertheless cover every contributing source.
            array_names = Some(vec![
                String::from("Front Array"),
                String::from("Rear Array"),
            ]);
            group.front.iter().chain(&group.rear).cloned().collect()
        }
        SpeakerConfig::Cardioid(group) => {
            array_names = Some(vec![String::from("Front Sub"), String::from("Rear Sub")]);
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
    let expected_count = if let Some(default_names) = array_names {
        let routed_names: Vec<_> = config
            .system
            .as_ref()
            .and_then(|system| system.subwoofers.as_ref())
            .into_iter()
            .flat_map(|subwoofers| &subwoofers.outputs)
            .filter(|output| output.speaker == source_name)
            .map(|output| output.id.clone())
            .collect();
        let names = if routed_names.is_empty() {
            default_names
        } else {
            if routed_names.len() != default_names.len() {
                return Err(format!(
                    "parallel waveform timing unavailable: array has {} aggregate branches but {} physical outputs are declared for '{source_name}'",
                    default_names.len(),
                    routed_names.len(),
                ));
            }
            routed_names
        };
        if drivers.iter().any(|driver| {
            names
                .get(driver.index)
                .is_none_or(|name| *name != driver.name)
        }) {
            let observed = drivers
                .iter()
                .map(|driver| format!("{}:{}", driver.index, driver.name))
                .collect::<Vec<_>>()
                .join(", ");
            return Err(format!(
                "parallel waveform timing unavailable: array branch identity mismatch for '{}': expected {:?}, observed [{observed}]",
                chain.channel, names
            ));
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
