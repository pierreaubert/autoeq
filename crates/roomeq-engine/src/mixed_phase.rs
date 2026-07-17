//! Mixed-phase correction for room EQ.
//!
//! Implements mixed-phase filter design:
//! 1. Decompose measurement into minimum-phase + excess phase
//! 2. IIR (parametric EQ) corrects the minimum-phase component
//! 3. Short FIR corrects the excess phase component (with pre-ringing constraint)
//!
//! This gives the best of both worlds: IIR latency for causal correction,
//! and a short FIR for the non-causal excess phase — without the long latency
//! of full FIR correction.
//!
//! Reference: Brännmark & Sternad, AES 124th Convention (2008)
//! Reference: Patent EP2104374B1

mod generate;
mod misc;
mod mixed_phase_config;
#[cfg(test)]
mod tests;
mod types;

pub use generate::*;
pub use mixed_phase_config::*;
pub use types::*;
