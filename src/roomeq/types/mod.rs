//! Room EQ Types - Modular type definitions
//!
//! This module re-exports all types from submodules for backwards compatibility.
//! New code should import directly from submodules.

// Configuration types
pub mod config {
    pub use roomeq_model::config::*;
}
pub use roomeq_model::config::*;

// Simple wizard preset types
pub mod preset {
    pub use roomeq_model::preset::*;
}
pub use roomeq_model::preset::*;

// Output types
mod output;
pub use output::*;

// Measurement-related types (re-export from crate root)
pub use crate::MeasurementRef;
pub use crate::{MeasurementSingle, MeasurementSource};

// Re-export Curve since it's used everywhere
pub use crate::Curve;

// Helper function for default config version (used by multiple modules)
pub use config::default_config_version;
