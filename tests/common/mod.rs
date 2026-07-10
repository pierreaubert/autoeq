//! Shared test helpers for RoomEQ / AutoEQ integration tests.
//!
//! Use `mod common;` in an integration test to gain access to pre-built
//! configuration fixtures and the config builders.

#![allow(dead_code)]

pub mod apo;
pub mod binary_runner;

use autoeq::roomeq::types::{
    OptimizerConfig, OptimizerConfigBuilder, RoomConfig, RoomConfigBuilder, SpeakerConfig,
};
use autoeq::{Curve, MeasurementSource};

/// A fast, deterministic optimizer configuration for unit-style integration
/// tests. It uses tiny population/iteration limits and disables refinement so
/// the test suite stays fast.
pub fn fast_optimizer_config() -> OptimizerConfig {
    OptimizerConfigBuilder::new()
        .algorithm("autoeq:de".to_string())
        .max_iter(5)
        .population(8)
        .refine(false)
        .build()
}

/// A minimal stereo [`RoomConfig`] using in-memory (empty) measurements and the
/// fast optimizer fixture.
pub fn stereo_room_config() -> RoomConfig {
    RoomConfigBuilder::new()
        .speaker(
            "L",
            SpeakerConfig::Single(MeasurementSource::InMemory(Curve::default())),
        )
        .speaker(
            "R",
            SpeakerConfig::Single(MeasurementSource::InMemory(Curve::default())),
        )
        .optimizer(fast_optimizer_config())
        .build()
}
