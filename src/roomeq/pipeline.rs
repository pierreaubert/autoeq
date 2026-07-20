//! Backward-compatible RoomEQ pipeline facade.
//!
//! Application composition is owned by `roomeq-workflow`; observable pipeline
//! contracts are owned by `roomeq-engine`.

pub use roomeq_engine::{
    PipelineControl, PipelineEvent, PipelineObserver, PipelineStepId, PipelineStepStatus,
};
pub use roomeq_workflow::{RoomPipeline, RoomPipelineRequest};
