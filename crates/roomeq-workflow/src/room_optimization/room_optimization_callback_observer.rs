use super::room_optimization_progress::RoomOptimizationProgress;
use super::types::CallbackAction;
use super::types::RoomOptimizationCallback;
use roomeq_engine::pipeline::{PipelineControl, PipelineEvent, PipelineObserver};

pub(super) fn callback_pipeline_observer(
    callback: RoomOptimizationCallback,
) -> Box<dyn PipelineObserver> {
    Box::new(RoomOptimizationCallbackObserver { callback })
}

pub(super) struct RoomOptimizationCallbackObserver {
    pub(super) callback: RoomOptimizationCallback,
}

impl PipelineObserver for RoomOptimizationCallbackObserver {
    fn on_event(&mut self, event: &PipelineEvent) -> PipelineControl {
        let progress = RoomOptimizationProgress::from(event);
        match (self.callback)(&progress) {
            CallbackAction::Continue => PipelineControl::Continue,
            CallbackAction::Stop => PipelineControl::Stop,
        }
    }
}
