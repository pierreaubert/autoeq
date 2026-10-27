//! Stable RoomEQ configuration and output contracts.

pub use autoeq_core::{
    AutoeqError, Curve, InlineMeasurement, MeasurementMultiple, MeasurementRef, MeasurementSingle,
    MeasurementSource, PeqModel, Result, SpinoramaBundle,
};
pub mod error {
    pub use autoeq_core::error::{AutoeqError, Result};
}

pub mod roomeq {
    pub mod types {
        pub use crate::config::{
            AppliedThreshold, AreaPriorKind, AreaQuadratureKind, AreaScalarisationKind,
            AssessmentConfidence, AssessmentProvenance, AssessmentRecord, AudibilityDeadbandConfig,
            AutoOptimizerConfig, BassAnchorChannelResultLegacy, BassAnchorResultsLegacy,
            BassHeadroomModelConfig, BassHeadroomModelKind, BassManagementConfig,
            BootstrapScalarisation, BootstrapUncertaintyConfig, BudgetAggregation,
            CURRENT_CONFIG_VERSION, CardioidConfig, Cea2034CorrectionConfig, Cea2034CorrectionMode,
            ChannelMatchingConfig, ConfigValidationReport, ContinuousListeningAreaConfig,
            CorrectionBandPolicy, CrossoverConfig, CtcConfig, CtcHeadPositionConfig, CtcHrtfConfig,
            CtcHrtfSpeakerConfig, CtcMeasurementConfig, CtcMeasurementFileConfig,
            CtcRegularizationConfig, CtcWindowConfig, DBAConfig, DEFAULT_LISTENING_ROOM_RT60_S,
            DecomposedCorrectionSerdeConfig, DriverCrossoverBand, EarlyLateCorrectionConfig,
            EnforcementState, ExcursionProtectionConfig, FilterAudibilityConfig, FilterVetoVerdict,
            FinalizationConfig, FirConfig, FirPlacement, GroupDelayOptimizationConfig,
            HeightChannelAlignmentConfig, HighFreqFilterConfig, HighFrequencyCorrectionConfig,
            HighpassType, InterChannelDeviation, InterChannelTimbreMatchingConfig,
            LEGACY_SPEAKER_GROUP_ADVISORY, LowFreqFilterConfig, MeasuredIrSource,
            MeasurementProvenanceReference, MixedModeConfig, MixedPhaseSerdeConfig,
            MultiMeasurementConfig, MultiMeasurementStrategy, MultiSeatConfig,
            MultiSeatMeasurement, MultiSeatSearchConfig, MultiSeatStrategy, MultiSubGroup,
            OptimizerConfig, OptimizerConfigBuilder, ParallelDriverGroup, PerceptualPolicyConfig,
            PerceptualPolicyPreset, PhaseAlignmentConfig, PhaseAssessmentConfig,
            PreRingingSerdeConfig, PrecedenceLimitBand, ProbeChannelResultLegacy,
            ProbeResultsLegacy, ProcessingMode, ProvenanceConfig, ProvenanceValidationMode,
            PruningBudget, PruningEvaluation, PruningEvaluationVersion, PruningProgramme,
            RecordingConfiguration, ReportOutcome, ReportingConfig, RoleTargetConfig, RoomConfig,
            RoomConfigBuilder, RoomDimensions, SchroederSplitConfig, SeatIdentityMap,
            SmoothnessPenaltyConfigSerde, SpatialRobustnessSerdeConfig, SpeakerConfig,
            SpeakerDriver, SpeakerDriverRole, SpeakerGroup, SpeakerTopology, SplCalibration,
            SubOptimizerConfig, SubwooferCrossoverRef, SubwooferOutput, SubwooferRouting,
            SubwooferStrategy, SubwooferSystemConfig, SupportingSourceConfig,
            SupportingSourceDecorrelation, SupportingSourceGroup, SupportingSourceOutputNaming,
            SystemConfig, SystemModel, TargetCurveConfig, TargetResponseConfig, TargetShape,
            TiltStageConfig, UserPreference, ValidationBundleConfig, ValidationStage,
            ValidationStageReport, ValidationStageStatus, VetoDecision, VetoReason,
            default_config_version, validate_config_version, validate_measured_ir_map,
        };
    }
    pub mod rir_prototype {
        pub use crate::rir_prototype_config::{
            DirectivityModel, DistanceWeightMode, RirPrototypeConfig,
        };
    }
}

pub mod auto_tune;
pub mod config;
pub mod contracts;
pub mod crossover_cancellation;
pub mod decision_ledger;
pub mod eligibility;
pub mod home_cinema;
pub use crossover_cancellation::{
    CROSSOVER_CANCELLATION_TOLERANCE_DB, CrossoverCancellationBaseline,
    CrossoverCancellationContext, CrossoverCancellationEvidence,
    DEFAULT_MAX_CROSSOVER_CANCELLATION_DB, crossover_cancellation_accepted,
};
pub mod headroom;
pub mod home_cinema_resolution;
pub mod ir_waveform;
pub mod joint_sub_report;
pub mod physical_drive;
#[doc(inline)]
pub use joint_sub_report::{
    JointSubDiagnostics, JointSubGainApplication, JointSubObjectiveReport, JointSubSeatReport,
};
pub mod acceptance_evidence;
pub mod optimizer_settings;
pub mod output;
pub mod payload_binding;
pub mod physical_routing;
pub mod preset;
pub mod reference_registry;
pub mod report_contracts;
pub mod rir_prototype_config;
pub mod target_tilt;
pub mod target_transition;
pub mod validation_rules;
pub use config::{
    AppliedThreshold, AreaPriorKind, AreaQuadratureKind, AreaScalarisationKind,
    AssessmentConfidence, AssessmentProvenance, AssessmentRecord, AudibilityDeadbandConfig,
    AutoOptimizerConfig, BassAnchorChannelResultLegacy, BassAnchorResultsLegacy,
    BassHeadroomModelConfig, BassHeadroomModelKind, BassManagementConfig, BootstrapScalarisation,
    BootstrapUncertaintyConfig, BudgetAggregation, CURRENT_CONFIG_VERSION, CardioidConfig,
    Cea2034CorrectionConfig, Cea2034CorrectionMode, ChannelMatchingConfig, ConfigValidationReport,
    ContinuousListeningAreaConfig, CorrectionBandPolicy, CrossoverConfig, CtcConfig,
    CtcHeadPositionConfig, CtcHrtfConfig, CtcHrtfSpeakerConfig, CtcMeasurementConfig,
    CtcMeasurementFileConfig, CtcRegularizationConfig, CtcWindowConfig, DBAConfig,
    DEFAULT_LISTENING_ROOM_RT60_S, DecomposedCorrectionSerdeConfig, DriverCrossoverBand,
    EarlyLateCorrectionConfig, EnforcementState, ExcursionProtectionConfig, FilterAudibilityConfig,
    FilterVetoVerdict, FinalizationConfig, FirConfig, FirPlacement, GroupDelayOptimizationConfig,
    HeightChannelAlignmentConfig, HighFreqFilterConfig, HighFrequencyCorrectionConfig,
    HighpassType, InterChannelDeviation, InterChannelTimbreMatchingConfig,
    LEGACY_SPEAKER_GROUP_ADVISORY, LowFreqFilterConfig, MeasuredIrSource,
    MeasurementProvenanceReference, MixedModeConfig, MixedPhaseSerdeConfig, MultiMeasurementConfig,
    MultiMeasurementStrategy, MultiSeatConfig, MultiSeatMeasurement, MultiSeatSearchConfig,
    MultiSeatStrategy, MultiSubGroup, OptimizerConfig, OptimizerConfigBuilder, ParallelDriverGroup,
    PerceptualPolicyConfig, PerceptualPolicyPreset, PhaseAlignmentConfig, PhaseAssessmentConfig,
    PreRingingSerdeConfig, PrecedenceLimitBand, ProbeChannelResultLegacy, ProbeResultsLegacy,
    ProcessingMode, ProvenanceConfig, ProvenanceValidationMode, PruningBudget, PruningEvaluation,
    PruningEvaluationVersion, PruningProgramme, RecordingConfiguration, ReportOutcome,
    ReportingConfig, RoleTargetConfig, RoomConfig, RoomConfigBuilder, RoomDimensions,
    SchroederSplitConfig, SeatIdentityMap, SmoothnessPenaltyConfigSerde,
    SpatialRobustnessSerdeConfig, SpeakerConfig, SpeakerDriver, SpeakerDriverRole, SpeakerGroup,
    SpeakerTopology, SplCalibration, SubOptimizerConfig, SubwooferCrossoverRef, SubwooferOutput,
    SubwooferRouting, SubwooferStrategy, SubwooferSystemConfig, SupportingSourceConfig,
    SupportingSourceDecorrelation, SupportingSourceGroup, SupportingSourceOutputNaming,
    SystemConfig, SystemModel, TargetCurveConfig, TargetResponseConfig, TargetShape,
    TiltStageConfig, UserPreference, ValidationBundleConfig, ValidationStage,
    ValidationStageReport, ValidationStageStatus, VetoDecision, VetoReason, default_config_version,
    validate_config_version, validate_measured_ir_map,
};
pub use contracts::{ChannelChain, DspGraph, Plugin};
pub use home_cinema::{
    AllChannelMultiSeatAcceptance, BassBusHeadroomSimulationReport, BassBusOutputHeadroomReport,
    BassManagementGroupReport, BassManagementMatrix, BassManagementOptimizationReport,
    BassManagementReport, BassManagementRoute, BassManagementRoutingGraph,
    BassManagementSignalFlowEntry, BassManagementSourceReport, BassManagementSubOutputReport,
    ChannelTimingReport, EffectiveBassManagement, HomeCinemaChannelReport, HomeCinemaLayoutReport,
    HomeCinemaRole, HomeCinemaRoleGroup, LfeBassManagementReport, MultiSeatChannelCorrectionReport,
    MultiSeatCorrectionReport, MultiSeatCoverageReport, MultiSeatPredictionReport,
    MultiSeatRoleGroupCorrectionReport, StereoBassRoutingCandidateReport, StereoBassRoutingReport,
    StereoBassTopology, TimingDiagnosticsReport, add_slope_offset, apply_role_target_adjustment,
    bass_management_crossover_frequency_hz, logical_channel_names, logical_speaker_configs,
    matching_group_key, matching_group_key_for_role, role_adjusted_target_response,
    role_for_channel, role_group_key, role_profile_base, role_score_band, role_slope_offset,
    role_target_curve_shape_active,
};
pub use ir_waveform::IrWaveform;
pub use optimizer_settings::{
    AsymmetricLossConfig, CrossoverType, EpaConfig, FrequencyBandWeights,
    PsychoacousticSmoothingConfig, TemporalMaskingConfig, TemporalMaskingProfile,
};
pub use output::{
    BootstrapUncertaintyReport, BootstrapUncertaintySource, ChannelDspChain,
    ChannelEarlyLateCurves, ChannelEarlyReflections, ChannelOctaveT60, ChannelReflectionEvent,
    ChannelResonanceDecay, ChannelResonanceDecays, ChannelSpeechTransmission, ChannelT60Band,
    ChannelWaterfall, ChannelWavelet, ContinuousAreaMetrics, CurveData,
    DirectEarlyLateCorrectionMetrics, DriverDspChain, DspChainOutput, EpaChannelMetrics,
    EpaMultichannelMetrics, LevelMatchedEpaMetrics, MeasuredRoomAcoustics, OptimizationMetadata,
    PerceptualMetrics, PerceptualPolicyReport, PhysicalCorrectionTarget, PluginConfigWrapper,
    PreferenceLayerReport, QaSeedDistribution, QaSeedOutcome, RoomOptimizerEvidence, StageCheck,
    StageCheckKind, StageOutcome, StageStatus, StatisticalSummary, SupportingSourceReport,
    ValidationBundleReport,
};
pub use physical_routing::{
    PhysicalRoute, PhysicalRouteCrossover, PhysicalRoutePass, PhysicalRoutingGraph,
    PhysicalRoutingPort,
};
pub use preset::{
    SimpleCrossoverChoice, SimpleLossChoice, SimplePresetConfig, SimpleProcessingChoice,
    SpeakerTier,
};
pub use report_contracts::{
    AcousticQualityScorecard, BandImprovement, CoherentTimingEvidence,
    ConfiguredBaselineComparison, ConfiguredBaselineResidual, CorrectionAcceptancePolicy,
    CorrectionAcceptanceReport, CorrectionDecision, CorrectionMetricSummary,
    CtcBinauralDiagnostics, CtcDeliveredResponseMetrics, CtcHrtfCandidateComparison, CtcReport,
    EpaProvenance, EpaScore, FinalSeatEvaluation, FrozenGridResidualEvidence, GroupDelayOptSummary,
    InputNormalizationEvidence, MixedPhaseCorrectionReport, MultiInputNormalizationEvidence,
    NormalizationPopulation, NormalizationReferencePolicy, OptimizerConfidence,
    OptimizerRestartEvidence, OptimizerRunEvidence, OptimizerTermination, OutputLossBand,
    ParetoCandidateEvidence, ParetoCrowdingDistance, ParetoDispatchReport, ParetoSelectionEvidence,
    PlaybackSummary, QualityPartitionMetrics, RUNTIME_ACCEPTANCE_POLICY_VERSION,
    RealizationQualityEvidence, RealizedProcessing, RelativeTargetReference, RoomEqOutcome,
    RuntimeAcceptancePolicy, RuntimeOutputClass, SummationSupportEvidence,
    TemporalIrMaskingMetrics, TemporalQualityEvidence, UpperBandAcousticBound,
    UsefulOutputEvidence, VetoAdjudicationReport, assess_realized_processing, playback_summary,
    processing_fallback_reason,
};
