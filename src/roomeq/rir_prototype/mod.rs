//! Distance- and directivity-weighted RIR prototype builder.
//!
//! Implements the prototyping step described in
//! `docs/superpowers/specs/2026-07-10-roomeq-rir-prototype-design.md`.

pub mod config;
pub mod prototype;
pub mod weights;

pub use config::{DirectivityModel, DistanceWeightMode, RirPrototypeConfig};
pub use prototype::{WeightedPrototype, build_weighted_prototype};
pub use weights::{
    compute_angles, compute_distances, directivity_weight, distance_weight, normalized_weights,
};
