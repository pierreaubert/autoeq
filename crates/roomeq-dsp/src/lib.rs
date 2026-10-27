//! Shared RoomEQ DSP realization helpers below engine and export.
//!
//! Extracted from `roomeq-engine` (refactor-20261025 Phase 3d) so
//! exporters can realize delays, resolve physical routing, and
//! build portable plugin configurations without depending on the
//! whole optimization engine. `roomeq-engine` re-exports every
//! module here, so existing `roomeq_engine::` paths keep resolving.

/// Band-limited fractional-delay FIR realization.
pub mod fir;
/// Portable serialized DSP plugin-config construction.
pub mod output;
/// Shared physical routing contract resolution.
pub mod physical_routing;
