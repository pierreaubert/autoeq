//! HTML+WASM report renderer for AutoEQ.
//!
//! Plotly replacement: reports are a small HTML shell embedding a versioned
//! JSON payload plus WASM modules that create the plots client-side.
//! The default 2D renderer draws on Canvas 2D with geometry from
//! gpui-toolkit's `d3rs` (scales, axis layout, sankey layout, color schemes)
//! and needs no WebGPU. When an adapter is available, the shell accelerates
//! projected surface triangles with WebGPU in the same plots. Layout, axes,
//! colors, and interaction remain shared with the Canvas fallback.

pub mod assemble;
pub mod draw;
pub mod grid;
pub mod schema;

pub use schema::*;
