//! Integration tests for FIR filter generation
//!
//! Tests the round-trip accuracy of FIR correction filters and validates
//! that different phase types work correctly.

#[path = "fir_tests/compute.rs"]
mod compute;
#[path = "fir_tests/create.rs"]
mod create;
#[path = "fir_tests/misc.rs"]
mod misc;
#[cfg(test)]
#[path = "fir_tests/tests.rs"]
mod tests;
