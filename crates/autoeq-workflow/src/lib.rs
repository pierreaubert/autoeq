//! High-level speaker and headphone equalization workflows.

/// Conditional debug output used by historical AutoEQ CLI callers.
#[macro_export]
macro_rules! qa_println {
    ($fmt:literal) => {
        log::debug!($fmt);
    };
    ($fmt:literal, $($arg:expr),* $(,)?) => {
        log::debug!($fmt, $($arg),*);
    };
    ($args:expr, $fmt:literal) => {
        if $args.qa.is_none() {
            log::debug!($fmt);
        }
    };
    ($args:expr, $fmt:literal, $($arg:expr),* $(,)?) => {
        if $args.qa.is_none() {
            log::debug!($fmt, $($arg),*);
        }
    };
}

pub use autoeq_core::iir;
pub use autoeq_core::{AutoeqError, Curve, Result};
pub mod error {
    pub use autoeq_core::error::{AutoeqError, Result};
}
pub mod read {
    pub use autoeq_measurements::read::{
        AlignedCurves, ApiError, BackendResult, Cea2034Data, CoherentAverageContract,
        ContourPlotData, CurveSupport, DetailedLoad, EstimatedPosition, EstimatedRangeCell,
        FsMeasurementCache, InMemoryMeasurementCache, InlineMeasurement, MeasurementBackend,
        MeasurementCache, MeasurementMultiple, MeasurementRef, MeasurementSingle,
        MeasurementSource, MockMeasurementBackend, NORMALIZE_HIGH_FREQ, NORMALIZE_LOW_FREQ,
        PsychoacousticSmoothingConfig, ReqwestMeasurementBackend, RewHeaderFacts, SeatProvenance,
        build_cea2034_data, cache_root, clamp_positive_only, coherent_average_measurement,
        create_log_frequency_grid, curve_support, data_dir_for, data_dir_for_cache_root,
        extract_cea2034_curves, extract_cea2034_curves_original, extract_curve_by_name,
        fetch_available_headphones, fetch_available_speakers, fetch_contour_data,
        fetch_contour_data_at_cache_root, fetch_curve_from_api, fetch_curve_from_api_at_cache_root,
        fetch_directivity_data, fetch_directivity_data_at_cache_root,
        fetch_headphone_frequency_response, fetch_headphone_frequency_response_at_cache_root,
        fetch_headphone_frequency_response_with_backend,
        fetch_headphone_frequency_response_with_backend_at_cache_root, fetch_measurement_plot_data,
        fetch_measurement_plot_data_at_cache_root, fetch_measurement_plot_data_with_backend,
        fetch_measurement_plot_data_with_backend_at_cache_root, fetch_record_from_api,
        find_similar_speakers, format_speaker_not_found_error, get_available_angles,
        get_directivity_at_angle, get_on_axis, headphone_cache_dir,
        headphone_cache_dir_for_cache_root, interpolate, interpolate_log_space,
        interpolate_log_space_record, interpolate_response, load_driver_measurement,
        load_frequency_response, load_measurement, load_measurement_strict,
        load_measurement_with_policy, load_source, load_source_detailed, load_source_individual,
        load_source_individual_with_support, load_source_unaligned, load_source_with_individual,
        load_spinorama_with_spin, load_spinorama_with_spin_at_cache_root, measurement_filename,
        normalize_and_interpolate_record, normalize_and_interpolate_response,
        normalize_and_interpolate_response_with_range, parse_rew_header, read_curve_from_csv,
        read_record_from_csv, read_rew_header_facts, read_spinorama, read_spinorama_at_cache_root,
        sanitize_dir_name, smooth_gaussian, smooth_one_over_n_octave, smooth_psychoacoustic,
        snapshot_source,
    };
}
/// Measurement records and rig identities used by product-oriented workflows.
pub mod measurements {
    pub use autoeq_measurements::{MeasurementOrigin, MeasurementRecord, MeasurementRigIdentity};
}
pub use autoeq_measurements::read::Cea2034Data;
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
pub mod optim {
    pub use autoeq_optim::optim::{
        AlgorithmInfo, AlgorithmType, BudgetBreach, COMPOSITE_COMPARISON_EPS_DB, ClassifiedOutcome,
        CompositeBreach, ConstrainedCandidate, ConstraintCapabilities, ConstraintDiagnostic,
        ConstraintKind, ConstraintSpec, ControlledBackendUnsupported, ControlledOptimizerRun,
        FilterOptimizer, FilterOptimizerOutput, FinalizedCandidate, GainAdjustment,
        JudgedParetoFront, JudgedParetoMember, MockOptimizerBackend, MultiObjectiveData,
        OPTIMIZER_IMPLEMENTATION_VERSION, ObjectiveData, ObjectiveDataBuilder,
        OptimProgressCallback, OptimizationOutcomeKind, OptimizerBackend,
        OptimizerBackendCompletion, OptimizerBackendCompletionEvidence,
        OptimizerBackendDeCompletion, OptimizerBackendTypedCompletionEvidence,
        OptimizerBudgetPreflightRefusal, OptimizerConfidence, OptimizerDispatchBudgetProfile,
        OptimizerDispatchOutcome, OptimizerOutcome, OptimizerRestartEvidence, OptimizerRunEvidence,
        OptimizerTermination, OwnedConstraintSpec, ParetoFeasibility, ParetoValidationError,
        PeakResolutionGuard, PenaltyMode, QAdjustment, RealOptimizerBackend,
        SmoothnessPenaltyConfig, VALIDATED_SUBDIVISIONS_PER_BIN, ValidationScoreRefusal, backend,
        bo, callback, check_composite_gain_envelope, check_pareto_feasibility,
        clamp_cuts_to_envelope, clamp_envelopes_into, clamp_gains_to_envelope, classify_outcome,
        cmaes, cobra, cobyla, compute_base_fitness, compute_ceiling_violation_into,
        compute_fitness_penalties, compute_fitness_penalties_ref, compute_pareto_objectives,
        compute_penalty_total, compute_response_fitness, compute_smoothness_penalty,
        compute_sorted_freqs_and_adjacent_octave_spacings, constrain_candidate,
        constraint_envelope, constraints_install, de, enforce_local_q_at_centers,
        enforce_local_q_envelope_at_centers, envelope_bound_at, finalize_candidate,
        find_algorithm_info, get_all_algorithms, guarded_parameters, is_peq_layout_loss, isres,
        judge_pareto_members, judge_pareto_members_with_stop, loss, mh, nsga, optimize_filters,
        optimize_filters_detailed, optimize_filters_with_algo_override,
        optimize_filters_with_callback, optimize_filters_with_callback_detailed,
        optimize_filters_with_completion_evidence, optimize_filters_with_de_completion,
        optimize_filters_with_run_control,
        optimize_filters_with_run_control_and_algo_override_detailed,
        optimize_filters_with_run_control_detailed,
        optimize_filters_with_typed_completion_evidence, optimizer_backend, params, pareto,
        policy_local_q_knots, prepared_objective, project_gains_onto_envelopes, registry,
        run_control, run_descriptor, scalar, setup, try_compute_fitness_penalties_ref,
        try_compute_pareto_objectives, types, validate_envelope_knots, validated_composite_grid,
        verify_joint_budgets,
    };
}
pub mod constraints {
    pub use autoeq_optim::constraints::{
        CeilingConstraintData, CrossoverMonotonicityConstraintData, MinGainConstraintData,
        SpacingConstraintData, ceiling, constraint_ceiling, constraint_crossover_monotonicity,
        constraint_min_gain, constraint_spacing, crossover_monotonicity,
        is_definite_spacing_infeasibility, min_gain, min_spacing, project_min_spacing,
        spacing_projection, viol_ceiling_from_spl, viol_min_gain_from_xs, viol_spacing_from_xs,
    };
}
pub mod cea2034 {
    pub use autoeq_measurements::cea2034::{
        Curve, DirectivityCurve, DirectivityData, ScoreMetrics, SpinoramaBundle,
        SpinoramaBundleBuilder, cea2034, compute_cea2034_metrics, compute_pir_from_lw_er_sp, lfx,
        nbd, octave, octave_intervals, score, score_peq, score_peq_approx, sm,
    };
}
pub use autoeq_optim::LossType;
pub use autoeq_optim::{OptimParams, OptimizationRunDescriptor, PeqModel, de};
pub mod x2peq {
    pub use autoeq_core::x2peq::{
        build_sorted_filters, compute_peq_response_from_x, peq_print_from_x, peq2x,
        try_compute_peq_response_from_x, try_x2spl, x2peq, x2peq_into, x2spl, x2spl_into,
    };
}
pub use autoeq_core::x2peq::x2peq;

pub mod workflow;
pub use workflow::{
    DeviceProfile, DeviceRange, DriverOptimizationResult, DriverOptimizationTiming,
    ExactDECheckpointOptions, HeadphoneOptResult, InputConfig, OptimizationLineage,
    OptimizationOutput, PRODUCT_RENDERER_CAPABILITIES_SCHEMA_VERSION,
    PROFILED_APO_SHELF_COEFFICIENT_EPSILON_MULTIPLIER, PROFILED_APO_SHELF_MAX_TRANSFER_DELTA_DB,
    PROFILED_APO_SHELF_MINIMUM_VERSION_ASSUMPTION, PROFILED_APO_SHELF_SLOPE_DB_PER_OCTAVE,
    PROFILED_APO_SHELF_SOURCE_REVISION, PreparedMultisubObjective, PreparedProduct, ProductCurves,
    ProductMode, ProductProfileExportStatus, ProductRenderer, ProductRendererCapabilities,
    ProductRendererCapability, ProductRendererFormat, ProductRendererLimitation,
    ProductRendererRefusalCode, ProductRendererVerifiedFeature, ProductRequest, ProductSource,
    ProductSourceAdapters, ProductTarget, ProfiledApoShelfContract, ProgressCallbackConfig,
    ProgressUpdate, SpeakerOptResult, TargetCompatibility, TargetCompatibilityStatus, TargetConfig,
    TargetProfile, VisualizationCurves, VisualizationGridConfig, build_target_curve,
    build_target_curve_by_name, compute_visualization_curves, drivers_initial_guess,
    drivers_initial_guess_fixed_freqs, exact_resume, initial_guess,
    load_driver_measurements_from_files, load_input_curve, load_input_record,
    multisub_initial_guess, normalize_product_curves, optimize_drivers_crossover,
    optimize_drivers_crossover_with_smoothness, optimize_headphone, optimize_headphone_with_grid,
    optimize_multisub, optimize_multisub_prepared, optimize_speaker,
    optimize_speaker_with_checkpoint, optimize_speaker_with_grid, perform_optimization,
    perform_optimization_with_callback, perform_optimization_with_progress,
    perform_optimization_with_progress_and_candidate, perform_optimization_with_run_descriptor,
    perform_optimization_with_run_descriptor_and_candidate,
    perform_optimization_with_run_descriptor_and_exact_checkpoint, prepare_multisub_objective,
    product, product_bytes_sha256_hex, product_renderer_capabilities, product_target_path,
    profiled_apo_shelf_contract, publish_apo_preset_pair, q_max_for_frequency_range, resume,
    setup_bounds, setup_drivers_bounds, setup_drivers_bounds_fixed_freqs,
    setup_drivers_objective_data, setup_multisub_bounds, setup_multisub_objective_data,
    setup_objective_data, verify_apo_preset_binding,
};

pub use autoeq_optim_types::OptimizationConfig;

// Preserve the existing deprecated compatibility export.
#[allow(deprecated)]
pub use workflow::restrict_boost_above_schroeder;
