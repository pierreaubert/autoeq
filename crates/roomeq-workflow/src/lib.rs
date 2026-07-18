//! RoomEQ application workflows and resource adapters.

pub mod arrival;
pub mod channel;
pub mod channel_measurements;
pub mod config_loader;
pub mod dba;
pub mod eq;
pub mod eq_resources;
pub mod fir;
pub mod multisub;
pub mod output;
pub mod pipeline;
pub mod sidecar;
mod wav;

pub use arrival::{prepare_channel_arrival_time, prepare_channel_input};
pub use channel::{ChannelWorkflowResult, process_single_channel};
pub use channel_measurements::prepare_channel_measurements;
pub use config_loader::{SHALLOW_MERGE_KEYS, load_config, merge_json_objects};
pub use eq_resources::{prepare_eq_resources, prepare_eq_target};
pub use output::save_dsp_chain;
pub use pipeline::{RoomPipeline, RoomPipelineRequest, WorkflowContext};
pub use sidecar::{
    ReservedConvolutionSidecar, persist_convolution_sidecar, reserve_channel_convolution_sidecar,
    reserve_mixed_crossover_sidecar,
};
