//! CEA-2034 compatibility surface over engine processing and workflow acquisition.

pub use roomeq_engine::cea2034::{
    SpeakerCorrectionResult, compute_speaker_correction, compute_speaker_correction_detailed,
    generate_preference_filters,
};
pub use roomeq_workflow::cea2034::{fetch_cea2034_blocking, pre_fetch_all_cea2034};
