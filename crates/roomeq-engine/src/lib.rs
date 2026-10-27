//! In-memory RoomEQ execution and deterministic processing.

#![forbid(unsafe_code)]

pub use autoeq_core::{
    AutoeqError, Curve, PeqModel, curve_transforms::build_target_curve_by_name,
    smooth_one_over_n_octave,
};
pub use autoeq_optim::de::CallbackAction;
pub use autoeq_optim::optim::run_control::OptimizerRunControl;
pub use autoeq_optim::optim::{OptimProgressCallback, OptimizerConfidence, OptimizerRunEvidence};
/// Acoustic analysis used by engine and workflow orchestration without adding
/// parallel ownership of the underlying implementations.
pub mod analysis {
    pub use roomeq_analysis::{
        crossover_utils, frequency_grid, ir_waveform, quasi_anechoic, response_metrics, slope,
        time_align,
    };
}
pub mod error {
    pub use autoeq_core::error::{AutoeqError, Result};
}
pub mod loss {
    pub use autoeq_optim::loss::{
        AsymmetricLossConfig, CrossoverType, DriverMeasurement, DriversLossData, HeadphoneLossData,
        JointSubComponents, JointSubWeights, LossType, PreparedAsymmetricLoss, PreparedFlatLoss,
        SpeakerLossData, asymmetric, bass_boost, calculate_standard_deviation_in_range,
        compute_drivers_combined_response, compute_drivers_combined_response_complex,
        compute_per_driver_responses, curve_slope_per_octave_in_range, drivers, drivers_flat_loss,
        enhanced_weights, epa, flat, flat_loss, flat_loss_asymmetric, headphone, headphone_loss,
        headphone_loss_with_target, is_peq_layout_loss, joint_multisub, joint_multisub_loss,
        mixed_loss, multisub, multisub_flat_loss, phase_aware,
        regression_slope_per_octave_in_range, slope, speaker, speaker_score_loss, types,
        weighted_mse_asymmetric,
    };
}
/// Runtime correction quality and acceptance contracts used at the engine
/// execution boundary.
pub mod quality {
    pub use roomeq_quality::{
        AbxAnswer, AcceptanceBundle, AcceptanceReport, AcceptanceThresholds, AcceptanceViolation,
        AcousticBaselinePlatform, AcousticCorpusBaseline, AcousticCorpusBaselineEntry,
        AcousticCorpusManifest, AcousticCorpusScenario, AcousticEnvironment, AcousticMetrics,
        AcousticOracle, AcousticQualityScorecard, AggregationOrder, AmbientNoiseView,
        AssessedOutput, AssessedScorecardSupport, BLOCKED_EXTERNAL_PREFIX, BandSplitPolicy,
        BassDetail, BassPolicy, BatteryCell, BatteryVerdict, BlindedProtocol, BoostLimitAgreement,
        BoundedInversionPolicy, BundleGate, CONVERSION_AFFINE_FS_SPL_V1, CalibratedLevel,
        CandidateKind, CandidateRecommendationStatus, CandidateTransfer, CaptureComparisonReport,
        CaptureMetricOutcome, CaptureTolerances, CapturedNoiseSettings, CapturedNoiseView,
        ChainConstraints, ChainEvidence, ChainStimulusBinding, ChannelDisposition, ClaimVerdict,
        CoherentTimingEvidence, ComparisonDesign, ComparisonIntent, ComparisonSpec,
        ConditionOutcome, ControlFamily, ControlFamilySummary, ControlOutcome, CorpusProvenance,
        CorpusRobustnessConfig, CorrectionAcceptancePolicy, CorrectionAcceptanceReport,
        CorrectionDecision, CorrectionMetricSummary, DIGITAL_CEILING_DBFS, DecayFit, DecayView,
        DecisionRule, DeclaredAlignment, DistributionSummary, DriverDisposition,
        DspDispositionView, EnforcementReadiness, EtcBand, EtcView, ExcludedBand, FinalReferences,
        FinalSeatOutcome, FinalValidationReport, FrozenResidualSamples, GateOutcome, GridProfile,
        GroupOutputScore, HEADROOM_EPSILON_DB, HeadPositionPerturbationConfig, HeadroomTrial,
        HeadroomVerdict, HeldOutEvidenceClass, HeldOutEvidenceStatus, HeldOutMeasurement,
        HoldoutSplit, ImportedTrial, ImpulseEvidence, InversionDecision, InversionEvidence,
        InversionVerdict, IrStepView, JointSubScorecard, LevelMatch, ListeningArm,
        ListeningBattery, ListeningCondition, ListeningSetup, MAX_STIMULUS_SECONDS, MagnitudeView,
        MatchedDecayBand, MatchedDecaySettings, MatchedDecayView, MatchedEtcView, MaterialClass,
        MeasurementNoise, MissingDirectFallback, MultiSeatAcceptance, NO_FIR_PRE_RINGING_FLOOR_DB,
        NoiseAnalysisError, NoisePressureCalibration, NullClassification,
        ORACLE_FREQUENCY_MEASURE_VERSION, OptimizerBudget, OracleParameters, OutputHeadroomInput,
        ParallelSourceParameters, PeakOrEnergy, PercentileDomain, PhaseAvailability,
        PlaybackBinding, PlaybackEvidenceKind, PlaybackLevel, PreRingingDefinition, ProgrammeEntry,
        ProgrammeSet, ProhibitedBehavior, QaTier, QualityBaselineComparison,
        QualityBaselineMetrics, QualityBaselinePartition, QualityEvaluationConfig, QualityGateMode,
        QualityGatePolicy, QualityGateReport, QualityPartitionMetrics, QualityRegressionPolicy,
        RUNTIME_ACCEPTANCE_POLICY_VERSION, RealizationQualityEvidence, ReferenceKind,
        RoutedHeadroomView, RuntimeAcceptancePolicy, RuntimeOutputClass,
        SEAT_AGGREGATION_MEASURE_VERSION, STIMULUS_RENDERER, SYNTHESIS_PEAK, ScenarioSpec,
        SeatMetricDefinition, SeatPartitionAcceptance, SeatScore, SeatSpectralOutcome,
        SeatSubScore, SeededRng, SourcePresentation, SpeakerTopology, SplCalibration, StimulusFile,
        StimulusKind, StimulusManifest, StimulusRequest, SubMainSumEvidence,
        TemporalChannelEvidence, TemporalGate, TemporalGateBasis, TemporalQualityEvidence,
        TrialAssignment, TrialImport, TrialParticipantAssignment, ValidatedCondition,
        ValidatedImport, ValidationCaseKind, ValidationResult, ValidationStage, ViewProvenance,
        ViewSettings, WindowDiagnostics, WorstSeatReport, abx_min_correct, abx_p_value,
        abx_trials_needed, aggregate_seat_bin, allpass_oracle, analytic_oracle_suite,
        apply_boost_limit_agreement, assess_boost_limit_agreement, assess_inversion,
        assess_output_headroom, assess_routed_headroom, binomial_lower_tail, biquad_cascade_oracle,
        bootstrap_mean_ci95, calibrated_capture_noise, check_battery_equivalence,
        check_claim_matches_intent, check_loudness_claim, comb_null_oracle,
        compare_complex_transfers, compare_prediction_capture, compare_quality_to_baseline,
        correlated_bass_band_efficiency_regression, correlated_bass_efficiency_regression,
        delay_oracle, derive_dsp_dispositions, derive_headroom_verdict,
        derive_temporal_quality_evidence, describe_assessed_support, deterministic_split,
        electrical_headroom, enforce_runtime_acceptance_evidence, evaluate_acoustic_quality,
        evaluate_acoustic_quality_with_permitted_gain, evaluate_bundle_gate,
        evaluate_chain_constraints, evaluate_correction_acceptance, evaluate_final_validation,
        evaluate_joint_sub_scorecard, evaluate_multi_seat_acceptance, evaluate_oracle,
        evaluate_quality_gate, excess_phase_oracle, fit_t60, frozen_grid_residual_samples,
        graph_acceptance_evidence, group_delay_ms, hilbert_envelope, identity_oracle,
        linkwitz_riley4_oracle, log_frequency_grid, matched_capture_decay, matched_capture_etc,
        missing_required_kinds, normalized_timbre_spread_db, octave_etc_envelope,
        oracle_for_scenario, parallel_woofers_oracle, perturb_transfer, physical_drive,
        polarity_oracle, qualifies_for_listening_claim, render_samples, render_stimulus_set,
        required_case_kinds, room_mode_oracle, room_transition_oracle,
        runtime_acceptance_failure_checks, scenario_matrix, schroeder_decay_db,
        schroeder_frequency_hz, score_abx_condition, score_abx_condition_under_protocol,
        score_abx_condition_with_alpha, score_battery, score_battery_imports, sha256_hex,
        step_response, summarize_claims, summarize_controls, summarize_distribution,
        validate_trial_import, validate_trial_import_with_setup, wilson_ci95, worst_supported_seat,
        worst_tail_mean, write_results_sidecar, write_results_sidecar_under_protocol,
    };
}
pub mod response {
    pub use autoeq_core::response::{
        MIN_FILTER_RESPONSE_DB, MIN_REALIZATION_RESPONSE_DB, apply_complex_response,
        apply_complex_response_with_min_db, compute_fir_complex_response,
        compute_peq_complex_response, try_apply_complex_response, try_compute_fir_complex_response,
        try_compute_peq_complex_response,
    };
}

/// Deterministic bass-management planning, prediction, and joint optimization.
pub mod bass_management;
/// Measurement-confidence gate for bass-band phase correction.
pub mod bass_phase_confidence;
/// Pure CEA-2034 speaker and preference correction.
pub mod cea2034;
/// Complete path-free preparation and execution for one channel.
pub mod channel_execution;
/// Path-free phase-linear, hybrid, and mixed-phase channel processing.
pub mod channel_fir;
/// Path-free low-latency, warped-IIR, and Kautz-modal channel processing.
pub mod channel_iir;
/// Complete path-free input prepared for channel processing.
pub mod channel_input;
/// Prepared, path-free measurement inputs for channel processing.
pub mod channel_measurements;
mod channel_optimizer;
mod channel_preference;
/// Deterministic preprocessing for a prepared channel.
pub mod channel_preprocessing;
/// Shared results and logical sidecar references for channel processing.
pub mod channel_result;
/// Target preparation for one channel.
pub mod channel_target;
pub mod config_adapter;
/// Multi-driver crossover optimization and polarity search.
pub mod crossover;
/// Path-free cross-talk cancellation matrix solving and diagnostics.
pub mod ctc;
/// Double-bass-array optimization and phase-critical array summation.
pub mod dba;
/// DSP convention and numerical-discipline audit checks.
pub mod dsp_conventions;
/// Canonical complex-response evaluation of serialized DSP chains.
pub mod dsp_realization;
/// In-memory per-channel and multi-measurement EQ optimization.
pub mod eq;
/// Speaker-excursion protection analysis and high-pass realization.
pub mod excursion;
/// RoomEQ FIR correction design.
pub mod fir;
/// Group-delay optimization and IIR all-pass alignment.
pub mod gd_opt;
/// Pure group and topology execution helpers.
pub mod group;
/// Prepared group/topology execution and DSP graph construction.
pub mod group_processing;
/// Role-aware height-channel spectral, phase, and arrival-time alignment.
pub mod height_channel_alignment;
/// In-memory home-cinema policy, routing, reporting, and seat analysis.
pub mod home_cinema;
/// Inter-channel tonal matching using broadband spectral correction.
pub mod inter_channel_timbre_matching;
/// Measured-room acoustics (early reflections, octave T60) for reporting.
pub mod ir_acoustics;
/// Path-free frequency-split FIR/IIR channel processing.
pub mod mixed_crossover;
/// Mixed IIR/FIR phase decomposition and excess-phase correction.
pub mod mixed_phase;
/// Multi-seat continuous-listening-area subwoofer optimization.
pub mod multiseat;
/// Multi-subwoofer optimization and all-pass alignment.
pub mod multisub;
/// Deterministic DSP-chain and response assembly.
pub mod output;
pub mod phase_alignment;
/// Resolve canonical physical routing independently of playback backends.
pub use roomeq_dsp::physical_routing;
#[cfg(test)]
mod physical_routing_integration_tests;
/// Prepared pipeline requests, observable events, and the production execution port.
pub mod pipeline;
/// Progress reporting for long-running RoomEQ operations.
pub mod progress;
/// Filesystem-capable validation of configured measurement provenance.
pub mod provenance;
/// Provisional correction decision records emitted at the decision site (E2).
pub mod provisional_decisions;
/// Cumulative pruning audit and source-summation checks (E3).
pub mod pruning_audit;
pub mod report_adapter;
/// Shared in-memory results returned by RoomEQ execution workflows.
pub mod room_result;
pub mod runtime_limiter;
/// Post-realization authorization for multi-segment measurement support.
pub mod segment_support;
/// Broadband spectral inter-channel response alignment.
pub mod spectral_align;
pub mod summation_search;
pub mod target_enforcement;
pub use roomeq_analysis::spatial_robustness;
/// Evidence-aware operation gating and local constraint evaluation (E1).
pub mod evidence_gate;
/// Supporting-source room compensation filter design.
pub mod supporting_source;
/// Deterministic topology, crossover, and bass-routing primitives.
pub mod topology;
/// Path-free time-alignment analysis used by workflow preparation.
pub mod time_align {
    pub use roomeq_analysis::time_align::{
        ArrivalTimeResult, ProbeDelayResult, detect_delay_with_probe, find_arrival_time_samples,
    };
}

pub use channel_input::{PreparedCea2034, PreparedChannelInput};
pub use channel_measurements::PreparedChannelMeasurements;
pub use pipeline::{
    EngineRequest, PipelineControl, PipelineEvent, PipelineObserver, PipelineStepId,
    PipelineStepStatus, RoomEngine,
};

/// QA evidence directory for test artifact retention (test builds only).
///
/// Tests that retain JSON evidence for the QA harness write through this
/// directory: `ROOMEQ_QA_DIR` when set, otherwise the workspace `target/qa`
/// directory beside the crate. The override keeps runs hermetic where the
/// shared workspace target is read-only, without changing default
/// artifact locations.
#[cfg(test)]
pub(crate) fn qa_evidence_dir() -> std::path::PathBuf {
    std::env::var_os("ROOMEQ_QA_DIR")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/qa"))
}
