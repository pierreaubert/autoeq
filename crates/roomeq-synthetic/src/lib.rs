//! Synthetic speaker curve generation for QA testing.
//!
//! Provides deterministic test scenarios with known ground truth for validating
//! optimization algorithms without relying on real measurement data.

pub use autoeq_core::{AutoeqError, Curve, Result};
pub mod error {
    pub use autoeq_core::error::{AutoeqError, Result};
}

pub mod catalog;
mod generate;
mod misc;
pub mod roundtrip;
pub mod spatial;
pub mod stimulus;
#[cfg(test)]
mod tests;
pub mod timing;
mod types;

pub use catalog::{FixtureRecord, fixture_catalog};
pub use generate::{
    add_noise, generate_cardioid_scenario, generate_channel_curve, generate_dba_scenario,
    generate_flat_curve, generate_harman_tilt_curve, generate_modal_room_scenario,
    generate_multisub_scenario, generate_scenario, generate_speaker_rolloff_curve,
    generate_sub_curve_with_phase, generate_subwoofer_rolloff_curve, try_generate_flat_curve,
    try_generate_harman_tilt_curve, try_generate_speaker_rolloff_curve,
    try_generate_subwoofer_rolloff_curve,
};
pub use misc::apply_known_eq;
pub use roundtrip::{
    PkSpec, ROUNDTRIP_SAMPLE_RATE, ROUNDTRIP_SEED, RoundtripCase, RoundtripChannel, band_rms,
    crossed_main_ideal, deform_curve, full_range_ideal, pk_bank, roundtrip_2_0, roundtrip_2_1,
    roundtrip_2_2_mso_subs, roundtrip_5_1, roundtrip_conflicting_seats, roundtrip_identity_2_0,
    sub_ideal, with_delay_phase,
};
pub use spatial::{
    BassOnlyCandidateFixture, BassPolarity, CoherentSum, CommonEqSeatPair, FeatureConstruction,
    ModalCutReference, OverlappingRemovalsFixture, RoomFeatureFixture, SPATIAL_SAMPLE_RATE_HZ,
    SeatTransfer, SharedBassFixture, SourceRole, WorstSeatCounterexample,
    bass_only_candidate_with_upper_fault, coherent_sum_db, common_eq_seat_pair,
    modal_cut_reference, moving_dip_fixture, narrow_peak_fixture, overlapping_removals_fixture,
    shared_bass_fixture, worse_seat_counterexample,
};
pub use stimulus::{
    OutputLossFixture, STIMULUS_SAMPLE_RATE_HZ, StimulusKind, StimulusSignal, am_sweep_signal,
    beats_signal, equal_energy_signal_a, equal_energy_signal_b, missing_fundamental_signal,
    output_loss_pair, resonance_signal, tilt_signal, transient_signal,
};
pub use timing::{
    CalibrationStatus, CalibrationUnknownSample, CaptureKind, ClockDriftFixture, ClockDriftParams,
    CoverageGapFixture, CoverageGapParams, MagnitudeOnlySample, SHARED_REFERENCE_ID,
    TIMING_SAMPLE_RATE_HZ, UncertaintyBandFixture, UncertaintyBandParams,
    calibration_unknown_sample, clock_drift_fixture, coverage_gap_fixture,
    spatial_magnitude_only_sample, uncertainty_band_fixture,
};
pub use types::{
    CardioidSyntheticScenario, DbaSyntheticScenario, ModalRoomScenario, MultiSubSyntheticScenario,
    SyntheticScenario,
};
