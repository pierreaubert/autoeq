//! Home-cinema role and layout helpers for RoomEQ.
//!
//! This intentionally mirrors the channel-label vocabulary used by
//! `sotf-host` speaker configurations without making `autoeq` depend on the
//! host crate. RoomEQ needs the same semantic model for target bands, channel
//! matching, and multi-seat diagnostics.

pub use roomeq_engine::home_cinema::*;
pub use roomeq_workflow::home_cinema::*;
