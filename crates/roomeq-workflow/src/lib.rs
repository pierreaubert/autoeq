//! RoomEQ application workflows and resource adapters.

pub mod arrival;
pub mod cea2034;
pub mod channel;
pub mod channel_measurements;
pub mod config_loader;
pub mod ctc;
pub mod dba;
pub mod eq;
pub mod eq_resources;
pub mod executor;
pub mod export;
pub mod fir;
pub mod group_measurements;
pub mod group_processing;
pub mod home_cinema;
pub mod measurement;
pub mod multisub;
pub mod output;
pub mod pipeline;
pub mod room_optimization;
pub mod sidecar;
pub mod supporting_source;
pub mod topology;
mod wav;

pub use arrival::{prepare_channel_arrival_time, prepare_channel_input};
pub use channel::{ChannelWorkflowResult, process_single_channel};
pub use channel_measurements::prepare_channel_measurements;
pub use config_loader::{
    SHALLOW_MERGE_KEYS, deserialize_room_config_strict, load_config,
    load_config_with_frequency_samples, load_merged_config_strict, merge_json_objects,
};
pub use eq_resources::{prepare_eq_resources, prepare_eq_target};
pub use export::{
    export_dsp_chain, export_dsp_chain_with_convolution_sidecars, package_convolution_sidecars,
};
pub use group_measurements::load_multisub_seat_measurements;
pub use group_processing::{
    process_cardioid, process_dba, process_multisub_group, process_speaker_group,
    process_speaker_topology,
};
pub use measurement::{
    DEFAULT_FREQUENCY_SAMPLES, load_curve_from_csv, load_curve_from_csv_with_frequency_samples,
    load_measurement, load_measurement_with_frequency_samples, load_source, load_source_individual,
    load_source_individual_with_frequency_samples, load_source_with_frequency_samples,
    load_source_with_individual, load_source_with_individual_with_frequency_samples,
};
pub use output::save_dsp_chain;
pub use pipeline::{RoomPipeline, RoomPipelineRequest, WorkflowContext};
pub use room_optimization::{
    CallbackAction, ChannelOptimizationResult, RoomOptimizationCallback, RoomOptimizationProgress,
    RoomOptimizationResult, SpeakerOptimizationCallback, SpeakerOptimizationResult, optimize_room,
    optimize_room_with_probe_arrivals, optimize_speaker,
};
pub use roomeq_export::ExportFormat;
pub use sidecar::{
    ReservedConvolutionSidecar, persist_convolution_sidecar, reserve_channel_convolution_sidecar,
    reserve_mixed_crossover_sidecar,
};

#[cfg(test)]
mod test_fixtures;
