//! Shared test fixtures for RoomEQ modules.
//!
//! These helpers are public only under `#[cfg(test)]` so they can be reused
//! across the library's `roomeq` unit tests without leaking into production
//! code.

use crate::Curve;
use crate::roomeq::types::{ChannelDspChain, OptimizationMetadata};
use crate::roomeq::{ChannelOptimizationResult, RoomOptimizationResult};
use ndarray::Array1;
use std::collections::HashMap;

/// An empty `OptimizationMetadata` for tests.
pub fn empty_metadata() -> OptimizationMetadata {
    OptimizationMetadata {
        pre_score: 0.0,
        post_score: 0.0,
        algorithm: "de".to_string(),
        loss_type: None,
        iterations: 0,
        timestamp: String::new(),
        inter_channel_deviation: None,
        epa_per_channel: None,
        epa_multichannel: None,
        group_delay: None,
        mixed_phase_per_channel: None,
        perceptual_metrics: None,
        home_cinema_layout: None,
        multi_seat_coverage: None,
        multi_seat_correction: None,
        bass_management: None,
        timing_diagnostics: None,
        ctc: None,
        perceptual_policy: None,
        bootstrap_uncertainty: None,
        validation_bundle: None,
        supporting_source: None,
        correction_acceptance: None,
        optimizer_evidence: None,
        stage_outcomes: Vec::new(),
    }
}

/// A flat 80 dB test curve.
pub fn flat_curve() -> Curve {
    Curve {
        freq: Array1::logspace(10.0, f64::log10(20.0), f64::log10(20000.0), 32),
        spl: Array1::from_elem(32, 80.0),
        phase: None,
        ..Default::default()
    }
}

/// A `RoomOptimizationResult` containing a single empty channel.
pub fn single_channel_room_result(channel_name: &str) -> RoomOptimizationResult {
    let curve = flat_curve();
    let mut channels = HashMap::new();
    channels.insert(
        channel_name.to_string(),
        ChannelDspChain {
            channel: channel_name.to_string(),
            plugins: Vec::new(),
            drivers: None,
            initial_curve: None,
            final_curve: None,
            eq_response: None,
            target_curve: None,
            pre_ir: None,
            post_ir: None,
            fir_temporal_masking: None,
            direct_early_late_correction: None,
        },
    );
    let mut channel_results = HashMap::new();
    channel_results.insert(
        channel_name.to_string(),
        ChannelOptimizationResult {
            name: channel_name.to_string(),
            pre_score: 0.5,
            post_score: 0.9,
            initial_curve: curve.clone(),
            final_curve: curve,
            biquads: Vec::new(),
            fir_coeffs: None,
            optimizer_evidence: Vec::new(),
        },
    );
    RoomOptimizationResult {
        channels,
        channel_results,
        combined_pre_score: 0.5,
        combined_post_score: 0.9,
        metadata: empty_metadata(),
    }
}
