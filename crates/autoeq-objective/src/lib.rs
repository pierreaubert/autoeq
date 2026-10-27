//! Objective-function evaluation for AutoEQ, independent of optimizer backends.
//!
//! Extracted from `autoeq-optim` (refactor-20261025 Phase 3c) so
//! prepared objective evaluation, loss computations, and evaluation
//! budgets can be reused without compiling any optimizer backend.
//! `autoeq-optim` re-exports every module here, so existing
//! `autoeq_optim::` paths keep resolving.

pub use autoeq_core as core;
pub use autoeq_core::iir;
pub use autoeq_core::{AutoeqError, Curve, PeqModel, Result};
pub use autoeq_measurements as measurements;

pub mod error {
    pub use autoeq_core::error::{AutoeqError, Result};
}
pub mod read {
    pub use autoeq_measurements::read::{
        AlignedCurves, ApiError, BackendResult, Cea2034Data, CoherentAverageContract,
        ContourPlotData, CurveSupport, DetailedLoad, EstimatedPosition, EstimatedRangeCell,
        InMemoryMeasurementCache, InlineMeasurement, MeasurementBackend, MeasurementCache,
        MeasurementMultiple, MeasurementRef, MeasurementSingle, MeasurementSource,
        MockMeasurementBackend, NORMALIZE_HIGH_FREQ, NORMALIZE_LOW_FREQ,
        PsychoacousticSmoothingConfig, RewHeaderFacts, SeatProvenance, build_cea2034_data,
        cache_root, clamp_positive_only, coherent_average_measurement, create_log_frequency_grid,
        curve_support, data_dir_for, data_dir_for_cache_root, extract_cea2034_curves,
        extract_cea2034_curves_original, extract_curve_by_name,
        fetch_headphone_frequency_response_with_backend,
        fetch_headphone_frequency_response_with_backend_at_cache_root,
        fetch_measurement_plot_data_with_backend,
        fetch_measurement_plot_data_with_backend_at_cache_root, find_similar_speakers,
        format_speaker_not_found_error, get_available_angles, get_directivity_at_angle,
        get_on_axis, headphone_cache_dir, headphone_cache_dir_for_cache_root, interpolate,
        interpolate_log_space, interpolate_log_space_record, interpolate_response,
        load_driver_measurement, load_frequency_response, load_measurement,
        load_measurement_strict, load_measurement_with_policy, load_source, load_source_detailed,
        load_source_individual, load_source_individual_with_support, load_source_unaligned,
        load_source_with_individual, measurement_filename, normalize_and_interpolate_record,
        normalize_and_interpolate_response, normalize_and_interpolate_response_with_range,
        parse_rew_header, read_curve_from_csv, read_record_from_csv, read_rew_header_facts,
        sanitize_dir_name, smooth_gaussian, smooth_one_over_n_octave, smooth_psychoacoustic,
        snapshot_source,
    };
    #[cfg(feature = "api")]
    pub use autoeq_measurements::read::{
        FsMeasurementCache, ReqwestMeasurementBackend, fetch_available_headphones,
        fetch_available_speakers, fetch_contour_data, fetch_contour_data_at_cache_root,
        fetch_curve_from_api, fetch_curve_from_api_at_cache_root, fetch_directivity_data,
        fetch_directivity_data_at_cache_root, fetch_headphone_frequency_response,
        fetch_headphone_frequency_response_at_cache_root, fetch_measurement_plot_data,
        fetch_measurement_plot_data_at_cache_root, fetch_record_from_api, load_spinorama_with_spin,
        load_spinorama_with_spin_at_cache_root, read_spinorama, read_spinorama_at_cache_root,
    };
}
pub mod cea2034 {
    pub use autoeq_measurements::cea2034::{
        Curve, DirectivityCurve, DirectivityData, ScoreMetrics, SpinoramaBundle,
        SpinoramaBundleBuilder, cea2034, compute_cea2034_metrics, compute_pir_from_lw_er_sp, lfx,
        nbd, octave, octave_intervals, score, score_peq, score_peq_approx, sm,
    };
}
pub mod param_utils {
    pub use autoeq_core::param_utils::{
        FilterParams, ParamLayout, PeqLayout, decode_filter_type, determine_filter_type,
        encode_filter_type, filter_type_bounds, freq_from_log10, freq_from_log10_clamped,
        freq_to_log10, get_filter_params, num_filters, params_per_filter, set_filter_params,
    };
}
pub mod x2peq {
    pub use autoeq_core::x2peq::{
        build_sorted_filters, compute_peq_response_from_x, peq_print_from_x, peq2x,
        try_compute_peq_response_from_x, try_x2spl, x2peq, x2peq_into, x2spl, x2spl_into,
    };
}
pub mod roomeq {
    pub use crate::roomeq_types::{
        AudibilityDeadbandConfig, ModeProximityEvidence, MultiMeasurementStrategy,
    };
    pub mod phase_utils {
        pub use autoeq_core::phase_utils::{
            compute_excess_phase, estimate_delay_from_excess_phase, reconstruct_minimum_phase,
            unwrap_phase_degrees,
        };
    }
}

pub mod constraints;
pub mod loss;
pub mod optim;
pub mod penalty_mode;
pub mod roomeq_types;
pub mod smoothness_penalty_config;

pub use loss::{CrossoverType, HeadphoneLossData, LossType, SpeakerLossData};
pub use optim::{MultiObjectiveData, ObjectiveData, ObjectiveDataBuilder};
pub use penalty_mode::PenaltyMode;
pub use smoothness_penalty_config::SmoothnessPenaltyConfig;
