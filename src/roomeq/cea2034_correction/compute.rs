//! Backward-compatible facade for engine-owned CEA-2034 correction.

pub use roomeq_engine::cea2034::{
    SpeakerCorrectionResult, compute_speaker_correction, compute_speaker_correction_detailed,
};
