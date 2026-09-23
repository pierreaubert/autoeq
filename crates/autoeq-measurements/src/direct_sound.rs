//! Compatibility exports for canonical direct-sound capture facts.
//!
//! Contracts live in core so measurement provenance can serialize them
//! without introducing a core-to-measurements dependency cycle.

#[doc(inline)]
pub use autoeq_core::direct_sound::{
    AngularCoverage, AveragingMethod, DirectSoundCaptureFacts, SPEED_OF_SOUND_M_S,
    reflection_free_interval_s, valid_lower_bound_hz,
};
