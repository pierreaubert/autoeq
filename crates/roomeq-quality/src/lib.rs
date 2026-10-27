//! Analytic acoustic ground truth and acceptance metrics for RoomEQ QA.
//!
//! This is the standalone RoomEQ acoustic-quality boundary.
//!
//! The fixtures in this module carry their generating parameters, expected
//! complex transfer function, valid correction region, and prohibited
//! behaviours. Candidate DSP is evaluated from its complex transfer function,
//! rather than inferred from plugin presence.
//!
//! CI entry points are intentionally ordinary Rust test filters so they do not
//! depend on a particular task runner:
//!
//! - PR: `cargo test -p autoeq acoustic_qa_pr_ --lib`
//! - Nightly: `cargo test -p autoeq acoustic_qa_nightly_ --lib -- --ignored`

mod acceptance;
mod acceptance_bundle;
mod band_policy;
mod battery;
mod capture;
mod chain_constraints;
mod corpus;
mod correlated_bass;
mod final_check;
mod fixtures;
mod inversion_support;
mod joint_sub_scorecard;
mod listening;
mod metrics;
mod promotion;
mod protocol;
mod quality;
mod scenario;
mod seeded;
mod stimuli;
mod trial_import;
mod types;
mod validation_corpus;

pub use acceptance::{
    BoostLimitAgreement, CorrectionAcceptancePolicy, CorrectionAcceptanceReport,
    CorrectionDecision, CorrectionMetricSummary, MultiSeatAcceptance,
    RUNTIME_ACCEPTANCE_POLICY_VERSION, RealizationQualityEvidence, RuntimeAcceptancePolicy,
    RuntimeOutputClass, SeatPartitionAcceptance, SeatSpectralOutcome, apply_boost_limit_agreement,
    assess_boost_limit_agreement, enforce_runtime_acceptance_evidence,
    evaluate_correction_acceptance, evaluate_multi_seat_acceptance,
    runtime_acceptance_failure_checks,
};
pub use acceptance_bundle::{
    AcceptanceBundle, AmbientNoiseView, AssessedOutput, BassDetail, BundleGate, CalibratedLevel,
    CapturedNoiseSettings, CapturedNoiseView, ChannelDisposition, DIGITAL_CEILING_DBFS, DecayFit,
    DecayView, DriverDisposition, DspDispositionView, EtcBand, EtcView, HEADROOM_EPSILON_DB,
    HeadroomTrial, HeadroomVerdict, IrStepView, MagnitudeView, MatchedDecayBand,
    MatchedDecaySettings, MatchedDecayView, MatchedEtcView, NoiseAnalysisError,
    NoisePressureCalibration, OutputHeadroomInput, RoutedHeadroomView, ViewProvenance,
    ViewSettings, assess_output_headroom, assess_routed_headroom, calibrated_capture_noise,
    derive_dsp_dispositions, derive_headroom_verdict, evaluate_bundle_gate, fit_t60,
    graph_acceptance_evidence, hilbert_envelope, matched_capture_decay, matched_capture_etc,
    octave_etc_envelope, schroeder_decay_db, step_response,
};
pub mod electrical_headroom;
pub mod physical_drive;
pub use band_policy::{BandSplitPolicy, BassPolicy, MissingDirectFallback, WindowDiagnostics};
pub use battery::{
    BatteryCell, BatteryVerdict, LevelMatch, ListeningBattery, MaterialClass,
    check_battery_equivalence, score_battery, score_battery_imports,
};
pub use capture::{
    CaptureComparisonReport, CaptureMetricOutcome, CaptureTolerances, DeclaredAlignment,
    ExcludedBand, PlaybackBinding, PlaybackEvidenceKind, PlaybackLevel, compare_prediction_capture,
};
pub use chain_constraints::{
    ChainConstraints, ChainEvidence, GateOutcome, PeakOrEnergy, PreRingingDefinition, TemporalGate,
    TemporalGateBasis, evaluate_chain_constraints,
};
pub use corpus::{
    AcousticBaselinePlatform, AcousticCorpusBaseline, AcousticCorpusBaselineEntry,
    AcousticCorpusManifest, AcousticCorpusScenario, CandidateRecommendationStatus,
    CorpusProvenance, CorpusRobustnessConfig, HeadPositionPerturbationConfig, HeldOutEvidenceClass,
    HeldOutEvidenceStatus, HeldOutMeasurement, QualityGateMode,
};
#[doc(inline)]
pub use correlated_bass::{
    correlated_bass_band_efficiency_regression, correlated_bass_efficiency_regression,
};
pub use final_check::{
    FinalReferences, FinalSeatOutcome, FinalValidationReport, PercentileDomain,
    SeatMetricDefinition, SubMainSumEvidence, evaluate_final_validation,
};
pub use fixtures::{
    allpass_oracle, analytic_oracle_suite, biquad_cascade_oracle, comb_null_oracle, delay_oracle,
    excess_phase_oracle, identity_oracle, linkwitz_riley4_oracle, log_frequency_grid,
    parallel_woofers_oracle, polarity_oracle, room_mode_oracle, room_transition_oracle,
    schroeder_frequency_hz,
};
pub use inversion_support::{
    BoundedInversionPolicy, InversionDecision, InversionEvidence, InversionVerdict,
    NullClassification, assess_inversion,
};
pub use joint_sub_scorecard::{
    GroupOutputScore, JointSubScorecard, SeatSubScore, evaluate_joint_sub_scorecard,
};
pub use listening::{
    ChainStimulusBinding, ListeningArm, ListeningCondition, ListeningSetup, SourcePresentation,
    TrialParticipantAssignment, check_claim_matches_intent,
};
pub use metrics::{
    AcceptanceReport, AcceptanceThresholds, AcceptanceViolation, AcousticMetrics, AggregationOrder,
    DistributionSummary, ORACLE_FREQUENCY_MEASURE_VERSION, SEAT_AGGREGATION_MEASURE_VERSION,
    SeatScore, aggregate_seat_bin, bootstrap_mean_ci95, compare_complex_transfers, evaluate_oracle,
    group_delay_ms, normalized_timbre_spread_db, summarize_distribution, worst_supported_seat,
    worst_tail_mean,
};
pub use promotion::{
    BLOCKED_EXTERNAL_PREFIX, ControlFamily, ControlFamilySummary, ControlOutcome,
    EnforcementReadiness, ProgrammeEntry, ProgrammeSet, check_loudness_claim, summarize_controls,
};
pub use protocol::{
    AbxAnswer, BlindedProtocol, ComparisonDesign, ComparisonIntent, ComparisonSpec,
    ConditionOutcome, DecisionRule, ReferenceKind, TrialAssignment, ValidationResult,
    WorstSeatReport, abx_min_correct, abx_p_value, abx_trials_needed, binomial_lower_tail,
    score_abx_condition, score_abx_condition_under_protocol, score_abx_condition_with_alpha,
    write_results_sidecar, write_results_sidecar_under_protocol,
};
pub use quality::{
    AcousticQualityScorecard, AssessedScorecardSupport, CoherentTimingEvidence,
    FrozenResidualSamples, NO_FIR_PRE_RINGING_FLOOR_DB, QualityBaselineComparison,
    QualityBaselineMetrics, QualityBaselinePartition, QualityEvaluationConfig, QualityGatePolicy,
    QualityGateReport, QualityPartitionMetrics, QualityRegressionPolicy, TemporalChannelEvidence,
    TemporalQualityEvidence, compare_quality_to_baseline, derive_temporal_quality_evidence,
    describe_assessed_support, evaluate_acoustic_quality,
    evaluate_acoustic_quality_with_permitted_gain, evaluate_quality_gate,
    frozen_grid_residual_samples,
};
pub use scenario::{
    AcousticEnvironment, CandidateKind, GridProfile, MeasurementNoise, OptimizerBudget,
    PhaseAvailability, QaTier, ScenarioSpec, SpeakerTopology, oracle_for_scenario,
    perturb_transfer, scenario_matrix,
};
pub use seeded::SeededRng;
pub use stimuli::{
    CONVERSION_AFFINE_FS_SPL_V1, MAX_STIMULUS_SECONDS, STIMULUS_RENDERER, SYNTHESIS_PEAK,
    SplCalibration, StimulusFile, StimulusKind, StimulusManifest, StimulusRequest, render_samples,
    render_stimulus_set, sha256_hex,
};
pub use trial_import::{
    ClaimVerdict, ImportedTrial, TrialImport, ValidatedCondition, ValidatedImport,
    qualifies_for_listening_claim, summarize_claims, validate_trial_import,
    validate_trial_import_with_setup, wilson_ci95,
};
pub use types::{
    AcousticOracle, CandidateTransfer, ImpulseEvidence, OracleParameters, ParallelSourceParameters,
    ProhibitedBehavior,
};
pub use validation_corpus::{
    HoldoutSplit, ValidationCaseKind, ValidationStage, deterministic_split, missing_required_kinds,
    required_case_kinds,
};

#[cfg(test)]
mod psycho_fast;
#[cfg(test)]
mod tests;
