//! HTML+WASM report renderer for AutoEQ.
//!
//! Plotly replacement: reports are a small HTML shell embedding a versioned
//! JSON payload plus WASM modules that create the plots client-side.
//! The default 2D renderer draws on Canvas 2D with geometry from
//! gpui-toolkit's `d3rs` (scales, axis layout, sankey layout, color schemes)
//! and needs no WebGPU. A full-GPUI enhanced view (sibling crate
//! `autoeq-report-gpui`) takes over when the shell detects WebGPU.

pub mod assemble;
pub mod draw;
pub mod schema;

#[cfg(target_arch = "wasm32")]
pub mod wasm;

pub use schema::*;
