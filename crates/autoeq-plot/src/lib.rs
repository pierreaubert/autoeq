//! HTML+WASM visualizations and report generation for AutoEQ.
//!
//! Copyright (C) 2025-2026 Pierre Aubert pierre(at)spinorama(dot)org
//!
//! This program is free software: you can redistribute it and/or modify
//! it under the terms of the GNU General Public License as published by
//! the Free Software Foundation, either version 3 of the License, or
//! (at your option) any later version.
//!
//! This program is distributed in the hope that it will be useful,
//! but WITHOUT ANY WARRANTY; without even the implied warranty of
//! MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//! GNU General Public License for more details.
//!
//! You should have received a copy of the GNU General Public License
//! along with this program.  If not, see <https://www.gnu.org/licenses/>.

pub use autoeq_core::iir;
pub use autoeq_core::{Curve, PeqModel};

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
pub mod param_utils {
    pub use autoeq_core::param_utils::{
        FilterParams, ParamLayout, PeqLayout, decode_filter_type, determine_filter_type,
        encode_filter_type, filter_type_bounds, freq_from_log10, freq_from_log10_clamped,
        freq_to_log10, get_filter_params, num_filters, params_per_filter, set_filter_params,
    };
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
pub mod response {
    pub use autoeq_core::response::{
        MIN_FILTER_RESPONSE_DB, MIN_REALIZATION_RESPONSE_DB, apply_complex_response,
        apply_complex_response_with_min_db, compute_fir_complex_response,
        compute_peq_complex_response, try_apply_complex_response, try_compute_fir_complex_response,
        try_compute_peq_complex_response,
    };
}
pub mod x2peq {
    pub use autoeq_core::x2peq::{
        build_sorted_filters, compute_peq_response_from_x, peq_print_from_x, peq2x,
        try_compute_peq_response_from_x, try_x2spl, x2peq, x2peq_into, x2spl, x2spl_into,
    };
}

mod config;
mod filter_color;
mod plot_drivers;
mod plot_filters;
mod plot_results;
mod plot_spin;
mod ref_lines;
mod trend_lines;

pub use config::PlotConfig;
pub use filter_color::filter_color;
pub use plot_drivers::{plot_driver_figures, plot_drivers_results};
pub use plot_filters::plot_filter_figures;
pub use plot_results::{plot_compute, plot_results};
pub use plot_spin::{plot_spin, plot_spin_details, plot_spin_tonal};
pub use trend_lines::{calculate_tonal_balance, create_regression_trace, generate_regression_line};

pub use autoeq_optim_types::OptimizationConfig;
