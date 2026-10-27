//! Measurement sources, loading, preprocessing, and speaker metrics.

pub use autoeq_core::{AutoeqError, Curve, Result, build_target_curve_by_name};

pub mod error {
    pub use autoeq_core::error::{AutoeqError, Result};
}

pub mod cea2034;
pub mod direct_sound;
pub mod evidence;
pub mod matrix;
pub mod mic_phase_calibration;
pub mod provenance;
pub mod provenance_migration_cli;
pub mod quality;
pub mod read;
pub mod timing;

pub use cea2034::{
    DirectivityCurve, DirectivityData, ScoreMetrics, SpinoramaBundle, SpinoramaBundleBuilder,
    cea2034, compute_cea2034_metrics, compute_pir_from_lw_er_sp, lfx, nbd, octave,
    octave_intervals, score, score_peq, score_peq_approx, sm,
};
pub use direct_sound::{
    AngularCoverage, AveragingMethod, DirectSoundCaptureFacts, SPEED_OF_SOUND_M_S,
    reflection_free_interval_s, valid_lower_bound_hz,
};
pub use evidence::{
    LoadedMeasurement, MeasurementError, apply_display_shift, evidence_envelope_for_record,
    fixture_path, identified_record, load_record_with_evidence, validate_evidence_metadata,
};
pub use matrix::{
    AverageKind, CompletenessReport, CoverageMask, MatrixKey, RejectedTake, Take, TakeDecision,
    TakeMatrix, resample_curve_with_coverage,
};
pub use mic_phase_calibration::{MicPhaseCalibration, load_mic_phase_calibration};
pub use provenance::{
    AcousticMeasurementHardware, AcquisitionMetadata, CaptureKind, Determinism, EvidenceLevel,
    LedgerEntry, MAX_SIDECAR_BYTES, MeasurementEvidence, MeasurementOrigin, MeasurementProvenance,
    MeasurementRecord, MeasurementRigIdentity, MeasurementRigKind, MeasurementSet, MicOrientation,
    OperationContext, PROVENANCE_SCHEMA, PROVENANCE_SCHEMA_VERSION, ProvenanceError,
    RedactionProfile, ReferenceScope, SetProvenance, SourceArtifact, SplReference, ToolIdentity,
    UncertaintyMetadata, ValidationMode, ValidationReport, migrate_sidecar, read_sidecar,
    read_sidecar_file, sidecar_path, write_sidecar,
};
pub use quality::{
    MeasurementQuality, MeasurementQualityReport, assess_measurement_quality,
    assess_multiple_measurement_quality,
};
pub use read::{
    AlignedCurves, ApiError, BackendResult, Cea2034Data, CoherentAverageContract, ContourPlotData,
    CurveSupport, DetailedLoad, EstimatedPosition, EstimatedRangeCell, InMemoryMeasurementCache,
    InlineMeasurement, MeasurementBackend, MeasurementCache, MeasurementMultiple, MeasurementRef,
    MeasurementSingle, MeasurementSource, MockMeasurementBackend, NORMALIZE_HIGH_FREQ,
    NORMALIZE_LOW_FREQ, PsychoacousticSmoothingConfig, RewHeaderFacts, SeatProvenance,
    build_cea2034_data, cache_root, clamp_positive_only, coherent_average_measurement,
    create_log_frequency_grid, curve_support, data_dir_for, data_dir_for_cache_root,
    extract_cea2034_curves, extract_cea2034_curves_original, extract_curve_by_name,
    fetch_headphone_frequency_response_with_backend,
    fetch_headphone_frequency_response_with_backend_at_cache_root,
    fetch_measurement_plot_data_with_backend,
    fetch_measurement_plot_data_with_backend_at_cache_root, find_similar_speakers,
    format_speaker_not_found_error, get_available_angles, get_directivity_at_angle, get_on_axis,
    headphone_cache_dir, headphone_cache_dir_for_cache_root, interpolate, interpolate_log_space,
    interpolate_log_space_record, interpolate_response, load_driver_measurement,
    load_frequency_response, load_measurement, load_measurement_strict,
    load_measurement_with_policy, load_source, load_source_detailed, load_source_individual,
    load_source_individual_with_support, load_source_unaligned, load_source_with_individual,
    measurement_filename, normalize_and_interpolate_record, normalize_and_interpolate_response,
    normalize_and_interpolate_response_with_range, parse_rew_header, read_curve_from_csv,
    read_record_from_csv, read_rew_header_facts, sanitize_dir_name, smooth_gaussian,
    smooth_one_over_n_octave, smooth_psychoacoustic, snapshot_source,
};
#[cfg(feature = "api")]
pub use read::{
    FsMeasurementCache, ReqwestMeasurementBackend, fetch_available_headphones,
    fetch_available_speakers, fetch_contour_data, fetch_contour_data_at_cache_root,
    fetch_curve_from_api, fetch_curve_from_api_at_cache_root, fetch_directivity_data,
    fetch_directivity_data_at_cache_root, fetch_headphone_frequency_response,
    fetch_headphone_frequency_response_at_cache_root, fetch_measurement_plot_data,
    fetch_measurement_plot_data_at_cache_root, fetch_record_from_api, load_spinorama_with_spin,
    load_spinorama_with_spin_at_cache_root, read_spinorama, read_spinorama_at_cache_root,
};
pub use timing::{
    CALIBRATION_APPLY_OP, CLOCK_CORRECTION_OP, ClockCorrection, ClockCorrectionMethod,
    ClockCorrectionReport, ClockFit, CorrectedArtifact, RECENTER_OP, ReferenceMarker,
    apply_calibration_once, apply_clock_correction, fit_clock_markers, recenter_ir,
    record_clock_correction, recover_original_offset, report_unsupported_adapter,
};

#[cfg(test)]
mod psycho_fast;
