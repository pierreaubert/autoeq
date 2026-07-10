//! Distance- and directivity-weighted RIR prototype builder.
//!
//! Implements the prototyping step described in
//! `docs/superpowers/specs/2026-07-10-roomeq-rir-prototype-design.md`.

pub mod config;
pub mod prototype;
pub(crate) mod weights;

pub use config::{DirectivityModel, DistanceWeightMode, RirPrototypeConfig};
pub use prototype::{WeightedPrototype, build_weighted_prototype};
