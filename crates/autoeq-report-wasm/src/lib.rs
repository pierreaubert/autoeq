//! HTML+WASM report renderer for AutoEQ.
//!
//! Plotly replacement: reports are a small HTML shell embedding a versioned
//! JSON payload plus WASM modules that create the plots client-side.
//! The default 2D renderer draws on Canvas 2D with geometry from
//! gpui-toolkit's `d3rs` (scales, axis layout, sankey layout, color schemes)
//! and needs no WebGPU. When an adapter is available, the shell accelerates
//! projected surface triangles with WebGPU in the same plots. Layout, axes,
//! colors, and interaction remain shared with the Canvas fallback.

pub mod acoustic_curves;
pub mod acoustic_reflections;
pub mod acoustic_summary;
pub mod acoustic_t60;
pub mod assemble;
pub mod bass_figures;
pub mod bass_management;
pub mod capture_quality;
pub mod channel_evidence;
pub mod combined_figures;
pub mod comparison_dataset;
pub mod comparison_figures;
pub mod comparison_tables;
pub mod crossover_config;
pub mod dashboard_payload;
pub mod draw;
pub mod eq_figures;
pub mod eq_sections;
pub mod evidence_sections;
pub mod fir_response;
pub mod gain_summary;
pub mod grid;
pub mod ir_group_delay;
mod plot_sampling;
pub mod prepared_report;
pub mod producer;
pub mod report_assembly;
pub mod report_data;
pub mod report_dataset;
pub mod report_dataset_channel;
pub mod report_dataset_global;
pub mod report_dataset_payload;
pub(crate) mod report_helpers;
pub mod report_provenance;
pub mod report_sections;
pub mod response_figures;
pub mod schema;
pub mod serialized_chain;
pub mod serialized_dsp;
pub mod signal_flow;
pub mod symmetric_pairs;
pub mod target_csv;
pub mod target_overlay;
pub mod temporal_figures;
pub mod time_of_flight;

pub use bass_management::{
    BassManagementSections, bass_management_groups_table_html, bass_management_sections,
    bass_management_sub_outputs_table_html, bass_management_summary_html,
};
pub use capture_quality::{
    CAPTURE_QUALITY_INPUT_VERSION, CaptureQualityBand, CaptureQualityFinding, CaptureQualityMetric,
    CaptureQualityReport, CaptureQualityReportError, CaptureQualitySource, CaptureQualityStatus,
    CaptureQualityThreshold, CaptureStimulusMetadata, CaptureThresholdComparison,
    append_capture_quality_section, append_capture_quality_section_to_value,
};
#[doc(inline)]
pub use crossover_config::crossover_config_html;
#[doc(inline)]
pub use gain_summary::gain_summary_html;
#[doc(inline)]
pub use prepared_report::{PREPARED_REPORT_SCHEMA_VERSION, PreparedReplay, PreparedReportData};
#[doc(inline)]
pub use report_sections::build_shared_sections;
pub use schema::{
    Annotation, AxisSpec, BarChart, BarGroup, DashOption, Figure, GridData, LineMark, RangeMark,
    ReportPayload, ReportProvenance, SCHEMA_VERSION, SankeyChart, SankeyLink, Section, Series,
    XScale,
};
#[doc(inline)]
pub use signal_flow::signal_flow_sections;
#[doc(inline)]
pub use time_of_flight::{time_of_flight_html, time_of_flight_rows};
