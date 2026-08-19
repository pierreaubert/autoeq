//! Resource-owning adapters for prepared group/topology engine execution.

use crate::group_measurements::load_multisub_seat_measurements_with_frequency_samples;
use crate::measurement::load_source_with_frequency_samples;
use std::path::Path;

use roomeq_engine::eq::EqResources;
use roomeq_engine::error::{AutoeqError, Result};
use roomeq_engine::group_processing::{
    self as engine, GroupProcessingResult, PreparedCardioidInput, PreparedMultiSubGroup,
    PreparedSpeakerTopology,
};
use roomeq_model::{
    CardioidConfig, DBAConfig, MultiSubGroup, RoomConfig, SpeakerGroup, SpeakerTopology,
};

fn prepare_resources(room_config: &RoomConfig, include_target: bool) -> Result<EqResources> {
    crate::prepare_eq_resources(
        &room_config.optimizer,
        include_target
            .then_some(room_config.target_curve.as_ref())
            .flatten(),
    )
    .map_err(|error| AutoeqError::InvalidMeasurement {
        message: format!("Failed to prepare group EQ resources: {error}"),
    })
}

fn prepare_topology_with_frequency_samples(
    channel_name: &str,
    topology: &SpeakerTopology,
    frequency_samples: usize,
) -> Result<PreparedSpeakerTopology> {
    let drivers = topology
        .drivers
        .iter()
        .enumerate()
        .map(|(index, driver)| {
            load_source_with_frequency_samples(&driver.measurement, frequency_samples).map_err(|error| {
                AutoeqError::InvalidMeasurement {
                    message: format!(
                        "Failed to load driver '{}' ({index}) measurement for channel {channel_name}: {error}",
                        driver.id
                    ),
                }
            })
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(PreparedSpeakerTopology { drivers })
}

pub fn process_speaker_group(
    channel_name: &str,
    group: &SpeakerGroup,
    room_config: &RoomConfig,
    sample_rate: f64,
    _output_dir: &Path,
) -> Result<GroupProcessingResult> {
    process_speaker_group_with_callback(
        channel_name,
        group,
        room_config,
        sample_rate,
        _output_dir,
        None,
    )
}

pub fn process_speaker_group_with_callback(
    channel_name: &str,
    group: &SpeakerGroup,
    room_config: &RoomConfig,
    sample_rate: f64,
    _output_dir: &Path,
    callback: Option<roomeq_engine::OptimProgressCallback>,
) -> Result<GroupProcessingResult> {
    process_speaker_group_with_callback_and_frequency_samples(
        channel_name,
        group,
        room_config,
        sample_rate,
        _output_dir,
        callback,
        crate::DEFAULT_FREQUENCY_SAMPLES,
    )
}

pub fn process_speaker_group_with_callback_and_frequency_samples(
    channel_name: &str,
    group: &SpeakerGroup,
    room_config: &RoomConfig,
    sample_rate: f64,
    _output_dir: &Path,
    callback: Option<roomeq_engine::OptimProgressCallback>,
    frequency_samples: usize,
) -> Result<GroupProcessingResult> {
    let prepared = prepare_topology_with_frequency_samples(
        channel_name,
        &group.to_legacy_topology(),
        frequency_samples,
    )?;
    let resources = prepare_resources(room_config, true)?;
    engine::process_speaker_group_with_callback(
        channel_name,
        group,
        room_config,
        sample_rate,
        &prepared,
        &resources,
        callback,
    )
}

pub fn process_speaker_topology(
    channel_name: &str,
    topology: &SpeakerTopology,
    room_config: &RoomConfig,
    sample_rate: f64,
    _output_dir: &Path,
) -> Result<GroupProcessingResult> {
    process_speaker_topology_with_callback(
        channel_name,
        topology,
        room_config,
        sample_rate,
        _output_dir,
        None,
    )
}

pub fn process_speaker_topology_with_callback(
    channel_name: &str,
    topology: &SpeakerTopology,
    room_config: &RoomConfig,
    sample_rate: f64,
    _output_dir: &Path,
    callback: Option<roomeq_engine::OptimProgressCallback>,
) -> Result<GroupProcessingResult> {
    process_speaker_topology_with_callback_and_frequency_samples(
        channel_name,
        topology,
        room_config,
        sample_rate,
        _output_dir,
        callback,
        crate::DEFAULT_FREQUENCY_SAMPLES,
    )
}

pub fn process_speaker_topology_with_callback_and_frequency_samples(
    channel_name: &str,
    topology: &SpeakerTopology,
    room_config: &RoomConfig,
    sample_rate: f64,
    _output_dir: &Path,
    callback: Option<roomeq_engine::OptimProgressCallback>,
    frequency_samples: usize,
) -> Result<GroupProcessingResult> {
    let prepared =
        prepare_topology_with_frequency_samples(channel_name, topology, frequency_samples)?;
    let resources = prepare_resources(room_config, true)?;
    engine::process_speaker_topology_with_callback(
        channel_name,
        topology,
        room_config,
        sample_rate,
        &prepared,
        &resources,
        callback,
    )
}

pub fn process_multisub_group(
    channel_name: &str,
    group: &MultiSubGroup,
    room_config: &RoomConfig,
    sample_rate: f64,
    _output_dir: &Path,
) -> Result<GroupProcessingResult> {
    process_multisub_group_with_callback(
        channel_name,
        group,
        room_config,
        sample_rate,
        _output_dir,
        None,
    )
}

pub fn process_multisub_group_with_callback(
    channel_name: &str,
    group: &MultiSubGroup,
    room_config: &RoomConfig,
    sample_rate: f64,
    _output_dir: &Path,
    callback: Option<roomeq_engine::OptimProgressCallback>,
) -> Result<GroupProcessingResult> {
    process_multisub_group_with_callback_and_frequency_samples(
        channel_name,
        group,
        room_config,
        sample_rate,
        _output_dir,
        callback,
        crate::DEFAULT_FREQUENCY_SAMPLES,
    )
}

pub fn process_multisub_group_with_callback_and_frequency_samples(
    channel_name: &str,
    group: &MultiSubGroup,
    room_config: &RoomConfig,
    sample_rate: f64,
    _output_dir: &Path,
    callback: Option<roomeq_engine::OptimProgressCallback>,
    frequency_samples: usize,
) -> Result<GroupProcessingResult> {
    let subwoofers = group
        .subwoofers
        .iter()
        .enumerate()
        .map(|(index, source)| {
            load_source_with_frequency_samples(source, frequency_samples).map_err(|error| {
                AutoeqError::InvalidMeasurement {
                    message: format!(
                        "Failed to load subwoofer {index} measurement for group '{}': {error}",
                        group.name
                    ),
                }
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let prepared = PreparedMultiSubGroup {
        subwoofers,
        seat_measurements: load_multisub_seat_measurements_with_frequency_samples(
            group,
            frequency_samples,
        )?,
    };
    let resources = prepare_resources(room_config, true)?;
    let flat_resources = prepare_resources(room_config, false)?;
    engine::process_multisub_group_with_callback(
        channel_name,
        group,
        room_config,
        sample_rate,
        &prepared,
        &resources,
        &flat_resources,
        callback,
    )
}

pub fn process_dba(
    channel_name: &str,
    config: &DBAConfig,
    room_config: &RoomConfig,
    sample_rate: f64,
    _output_dir: &Path,
) -> Result<GroupProcessingResult> {
    process_dba_with_callback(
        channel_name,
        config,
        room_config,
        sample_rate,
        _output_dir,
        None,
    )
}

pub fn process_dba_with_callback(
    channel_name: &str,
    config: &DBAConfig,
    room_config: &RoomConfig,
    sample_rate: f64,
    _output_dir: &Path,
    callback: Option<roomeq_engine::OptimProgressCallback>,
) -> Result<GroupProcessingResult> {
    process_dba_with_callback_and_frequency_samples(
        channel_name,
        config,
        room_config,
        sample_rate,
        _output_dir,
        callback,
        crate::DEFAULT_FREQUENCY_SAMPLES,
    )
}

pub fn process_dba_with_callback_and_frequency_samples(
    channel_name: &str,
    config: &DBAConfig,
    room_config: &RoomConfig,
    sample_rate: f64,
    _output_dir: &Path,
    callback: Option<roomeq_engine::OptimProgressCallback>,
    frequency_samples: usize,
) -> Result<GroupProcessingResult> {
    let prepared = crate::dba::prepare_dba_with_frequency_samples(config, frequency_samples)
        .map_err(|error| AutoeqError::InvalidMeasurement {
            message: format!("Failed to prepare DBA measurements: {error}"),
        })?;
    let resources = prepare_resources(room_config, true)?;
    engine::process_dba_with_callback(
        channel_name,
        room_config,
        sample_rate,
        &prepared,
        &resources,
        callback,
    )
}

pub fn process_cardioid(
    channel_name: &str,
    config: &CardioidConfig,
    room_config: &RoomConfig,
    sample_rate: f64,
    _output_dir: &Path,
) -> Result<GroupProcessingResult> {
    process_cardioid_with_callback(
        channel_name,
        config,
        room_config,
        sample_rate,
        _output_dir,
        None,
    )
}

pub fn process_cardioid_with_callback(
    channel_name: &str,
    config: &CardioidConfig,
    room_config: &RoomConfig,
    sample_rate: f64,
    _output_dir: &Path,
    callback: Option<roomeq_engine::OptimProgressCallback>,
) -> Result<GroupProcessingResult> {
    process_cardioid_with_callback_and_frequency_samples(
        channel_name,
        config,
        room_config,
        sample_rate,
        _output_dir,
        callback,
        crate::DEFAULT_FREQUENCY_SAMPLES,
    )
}

pub fn process_cardioid_with_callback_and_frequency_samples(
    channel_name: &str,
    config: &CardioidConfig,
    room_config: &RoomConfig,
    sample_rate: f64,
    _output_dir: &Path,
    callback: Option<roomeq_engine::OptimProgressCallback>,
    frequency_samples: usize,
) -> Result<GroupProcessingResult> {
    let front =
        load_source_with_frequency_samples(&config.front, frequency_samples).map_err(|error| {
            AutoeqError::InvalidMeasurement {
                message: format!("Failed to load Front measurement: {error}"),
            }
        })?;
    let rear =
        load_source_with_frequency_samples(&config.rear, frequency_samples).map_err(|error| {
            AutoeqError::InvalidMeasurement {
                message: format!("Failed to load Rear measurement: {error}"),
            }
        })?;
    let prepared = PreparedCardioidInput { front, rear };
    let resources = prepare_resources(room_config, true)?;
    engine::process_cardioid_with_callback(
        channel_name,
        config,
        room_config,
        sample_rate,
        &prepared,
        &resources,
        callback,
    )
}

#[cfg(test)]
mod tests;
