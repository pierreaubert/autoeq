use super::conformance::{
    validate_camilladsp_input, validate_pipewire_input, validate_serial_external_input,
};
use roomeq_model::DspGraph;
use serde::Serialize;
use std::path::{Path, PathBuf};

#[cfg(test)]
mod limiter_tests {
    use super::*;

    #[test]
    fn capability_query_reports_limiter_refusal_for_every_backend() {
        let output = DspGraph {
            version: "1.3.0".into(),
            artifact_bundle_schema_version: None,
            global_plugins: vec![],
            metadata: None,
            correction_decisions: None,
            deployed_source_curves: Default::default(),
            channels: std::collections::HashMap::from([(
                "Sub1".into(),
                serde_json::from_value(serde_json::json!({"channel": "Sub1", "plugins": [
                    roomeq_engine::runtime_limiter::plugin(0.0)
                ]}))
                .unwrap(),
            )]),
        };

        let capabilities = query_export_capabilities(&output);
        assert_eq!(capabilities.len(), 8);
        assert!(capabilities.iter().all(|capability| !capability.supported));
        assert!(capabilities.iter().all(|capability| {
            capability
                .reasons
                .iter()
                .any(|reason| reason.code == ExportCapabilityCode::MandatoryRuntimeLimiter)
        }));
        let json = serde_json::to_value(capabilities).unwrap();
        assert_eq!(json[0]["reasons"][0]["code"], "mandatory_runtime_limiter");
    }

    #[test]
    fn capability_query_reports_rate_and_unverified_convolution_requirements() {
        let graph = DspGraph::new("1.3.0");
        let invalid_rate =
            query_export_capability_at_sample_rate(&graph, ExportFormat::CamillaDsp, f64::NAN);
        assert!(!invalid_rate.supported);
        assert_eq!(
            invalid_rate.reasons[0].code,
            ExportCapabilityCode::InvalidSampleRate
        );

        let mut graph = DspGraph::new("1.3.0");
        graph.channels.insert(
            "L".into(),
            serde_json::from_value(serde_json::json!({
                "channel": "L",
                "plugins": [{
                    "plugin_type": "convolution",
                    "parameters": {"ir_file": "L_fir_48000hz.wav"}
                }]
            }))
            .unwrap(),
        );
        let capability = query_export_capability(&graph, ExportFormat::CamillaDsp);
        assert!(!capability.resources_verified);
        assert!(!capability.ready_to_export);
        assert_eq!(
            capability.required_convolution_resources,
            vec!["L_fir_48000hz.wav"]
        );
        assert!(
            capability
                .reasons
                .iter()
                .any(|reason| reason.code == ExportCapabilityCode::ConvolutionResourcesUnverified)
        );
    }

    #[test]
    fn capability_query_matches_existing_format_checks() {
        let graph = DspGraph::new("1.3.0");
        for capability in query_export_capabilities(&graph) {
            assert_eq!(
                capability.supported,
                external_export_supported(&graph, capability.format).is_ok(),
                "{:?}",
                capability.format
            );
        }
    }

    #[test]
    fn external_formats_cannot_drop_runtime_sub_protection() {
        let output = DspGraph {
            version: "1.3.0".into(),
            artifact_bundle_schema_version: None,
            global_plugins: vec![],
            metadata: None,
            correction_decisions: None,
            deployed_source_curves: Default::default(),
            channels: std::collections::HashMap::from([(
                "Sub1".into(),
                serde_json::from_value(serde_json::json!({"channel": "Sub1", "plugins": [
                    roomeq_engine::runtime_limiter::plugin(0.0)
                ]}))
                .unwrap(),
            )]),
        };
        for format in [
            ExportFormat::CamillaDsp,
            ExportFormat::EqualizerApo,
            ExportFormat::EasyEffects,
            ExportFormat::Wavelet,
            ExportFormat::PipeWire,
            ExportFormat::RoonDsp,
            ExportFormat::Rew,
            ExportFormat::BiquadCoefficients,
        ] {
            let error = ensure_external_export_supported(&output, format).unwrap_err();
            assert!(
                error
                    .to_string()
                    .contains("mandatory runtime sub-output limiter"),
                "{error}"
            );
        }
    }
}

/// Supported export formats for DSP chain output
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, clap::ValueEnum)]
pub enum ExportFormat {
    /// CamillaDSP YAML configuration
    #[value(name = "camilladsp")]
    #[serde(rename = "camilladsp")]
    CamillaDsp,
    /// Equalizer APO / Peace GUI text format (also works with PipeWire parametric-equalizer module)
    #[value(name = "apo")]
    #[serde(rename = "apo")]
    EqualizerApo,
    /// EasyEffects JSON preset
    #[value(name = "easyeffects")]
    #[serde(rename = "easyeffects")]
    EasyEffects,
    /// Wavelet GraphicEQ text format
    #[value(name = "wavelet")]
    #[serde(rename = "wavelet")]
    Wavelet,
    /// PipeWire filter-chain SPA-JSON configuration
    #[value(name = "pipewire")]
    #[serde(rename = "pipewire")]
    PipeWire,
    /// Roon DSP Engine preset (JSON)
    #[value(name = "roon")]
    #[serde(rename = "roon")]
    RoonDsp,
    /// REW Generic EQ reference text for manual entry; REW cannot reload this text.
    #[value(name = "rew")]
    #[serde(rename = "rew")]
    Rew,
    /// Raw normalized biquad coefficients with channel/order metadata
    #[value(name = "coefficients", alias = "biquad-coefficients")]
    #[serde(rename = "coefficients")]
    BiquadCoefficients,
}

/// Stable reason codes explaining why one backend cannot render a graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ExportCapabilityCode {
    /// Graph validation failed before backend rendering.
    InvalidGraph,
    /// The requested sample rate is not finite and positive.
    InvalidSampleRate,
    /// The external backend cannot preserve the mandatory runtime limiter.
    MandatoryRuntimeLimiter,
    /// The graph contains routing or global stages the backend cannot represent.
    RoutedGraphUnsupported,
    /// Backend-specific validation or rendering rejected the graph.
    BackendValidation,
    /// Convolution resources exist but this graph-only query did not verify their bytes.
    ConvolutionResourcesUnverified,
}

/// One machine-readable limitation returned by an export capability query.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ExportCapabilityReason {
    /// Stable machine-readable cause.
    pub code: ExportCapabilityCode,
    /// Human-readable detail from the graph or renderer validation.
    pub message: String,
}

/// Current graph compatibility for one external export backend.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ExportCapability {
    /// The external backend that was checked.
    pub format: ExportFormat,
    /// Whether the backend renderer accepted the graph at the checked rate.
    pub supported: bool,
    /// Whether rendering can proceed without resource resolution.
    ///
    /// This does not verify import or playback in an external application.
    /// In particular, REW Generic EQ text is a reference for manual entry.
    pub ready_to_export: bool,
    /// Sample rate used for the renderer dry run.
    pub sample_rate_hz: f64,
    /// Convolution members the caller must resolve and verify before packaging.
    pub required_convolution_resources: Vec<String>,
    /// Whether all required convolution resources were available to this query.
    pub resources_verified: bool,
    /// Machine-readable reasons and requirements.
    pub reasons: Vec<ExportCapabilityReason>,
}

fn push_capability_reason(
    capability: &mut ExportCapability,
    code: ExportCapabilityCode,
    message: String,
) {
    capability
        .reasons
        .push(ExportCapabilityReason { code, message });
}

fn capability_default_sample_rate(graph: &DspGraph) -> f64 {
    graph
        .correction_decisions
        .as_ref()
        .and_then(|ledger| ledger.acceptance_evidence.as_ref())
        .and_then(|evidence| evidence.payload.get("sample_rate_hz"))
        .and_then(serde_json::Value::as_f64)
        .filter(|rate| rate.is_finite() && *rate > 0.0)
        .unwrap_or(48_000.0)
}

/// Return the current capability for every external format.
///
/// The query uses the sample rate recorded in graph evidence, or 48 kHz when
/// the graph has no sample-rate evidence. It runs the actual
/// path-free renderer for each backend. Convolution resources are listed as
/// unverified because this query has no resource directory or byte inputs.
/// Renderer acceptance does not establish external application import or playback.
pub fn query_export_capabilities(graph: &DspGraph) -> Vec<ExportCapability> {
    query_export_capabilities_at_sample_rate(graph, capability_default_sample_rate(graph))
}

/// Return the current capability for every backend at an explicit sample rate.
pub fn query_export_capabilities_at_sample_rate(
    graph: &DspGraph,
    sample_rate_hz: f64,
) -> Vec<ExportCapability> {
    [
        ExportFormat::CamillaDsp,
        ExportFormat::EqualizerApo,
        ExportFormat::EasyEffects,
        ExportFormat::Wavelet,
        ExportFormat::PipeWire,
        ExportFormat::RoonDsp,
        ExportFormat::Rew,
        ExportFormat::BiquadCoefficients,
    ]
    .into_iter()
    .map(|format| query_export_capability_at_sample_rate(graph, format, sample_rate_hz))
    .collect()
}

/// Return whether one external format can render and preserve the graph.
///
/// Convolution paths are surfaced as unverified requirements. The query does
/// not claim the export is ready until a caller resolves and verifies them.
pub fn query_export_capability(graph: &DspGraph, format: ExportFormat) -> ExportCapability {
    query_export_capability_at_sample_rate(graph, format, capability_default_sample_rate(graph))
}

/// Return one backend capability at an explicit sample rate.
pub fn query_export_capability_at_sample_rate(
    graph: &DspGraph,
    format: ExportFormat,
    sample_rate_hz: f64,
) -> ExportCapability {
    let mut capability = ExportCapability {
        format,
        supported: false,
        ready_to_export: false,
        sample_rate_hz,
        required_convolution_resources: Vec::new(),
        resources_verified: false,
        reasons: Vec::new(),
    };
    if !sample_rate_hz.is_finite() || sample_rate_hz <= 0.0 {
        push_capability_reason(
            &mut capability,
            ExportCapabilityCode::InvalidSampleRate,
            format!("sample rate must be finite and positive, got {sample_rate_hz}"),
        );
        return capability;
    }
    if let Err(error) = graph.validate() {
        push_capability_reason(
            &mut capability,
            ExportCapabilityCode::InvalidGraph,
            error.to_string(),
        );
        return capability;
    }
    if graph
        .global_plugins
        .iter()
        .chain(graph.channels.values().flat_map(|chain| {
            chain.plugins.iter().chain(
                chain
                    .drivers
                    .iter()
                    .flatten()
                    .flat_map(|driver| driver.plugins.iter()),
            )
        }))
        .any(|plugin| plugin.plugin_type == "limiter")
    {
        push_capability_reason(
            &mut capability,
            ExportCapabilityCode::MandatoryRuntimeLimiter,
            format!(
                "{format:?} export cannot preserve the mandatory runtime sub-output limiter; use native playback"
            ),
        );
        return capability;
    }

    let has_extended_graph = has_routed_bass_management(graph) || !graph.global_plugins.is_empty();
    let extended_graph_unsupported = has_extended_graph
        && match format {
            ExportFormat::CamillaDsp => !has_only_bass_management_matrix(graph),
            ExportFormat::EqualizerApo => false,
            _ => true,
        };
    if extended_graph_unsupported {
        push_capability_reason(
            &mut capability,
            ExportCapabilityCode::RoutedGraphUnsupported,
            format!(
                "{format:?} export cannot represent routed home-cinema bass management or global graph stages safely"
            ),
        );
        return capability;
    }
    if let Err(error) = ensure_external_export_supported(graph, format) {
        push_capability_reason(
            &mut capability,
            ExportCapabilityCode::BackendValidation,
            error.to_string(),
        );
        return capability;
    }

    // This dry run is important for Equalizer APO: its renderer accepts some
    // static routings, but rejects arbitrary graphs that the shared precheck
    // cannot classify without rendering.
    if let Err(error) = super::render_dsp_graph(graph, format, sample_rate_hz) {
        push_capability_reason(
            &mut capability,
            ExportCapabilityCode::BackendValidation,
            error.to_string(),
        );
        return capability;
    }

    let resources = match super::checked_convolution_resource_references(graph) {
        Ok(resources) => resources,
        Err(error) => {
            push_capability_reason(
                &mut capability,
                ExportCapabilityCode::BackendValidation,
                error.to_string(),
            );
            return capability;
        }
    };
    capability.supported = true;
    capability.resources_verified = resources.is_empty();
    capability.required_convolution_resources = resources;
    capability.ready_to_export = capability.resources_verified;
    if !capability.resources_verified {
        push_capability_reason(
            &mut capability,
            ExportCapabilityCode::ConvolutionResourcesUnverified,
            "the graph references convolution resources; resolve and verify them before export"
                .to_string(),
        );
    }
    capability
}

impl ExportFormat {
    pub fn default_extension(&self) -> &'static str {
        match self {
            ExportFormat::CamillaDsp => "yaml",
            ExportFormat::EqualizerApo => "txt",
            ExportFormat::EasyEffects => "json",
            ExportFormat::Wavelet => "txt",
            ExportFormat::PipeWire => "conf",
            ExportFormat::RoonDsp => "json",
            ExportFormat::Rew => "txt",
            ExportFormat::BiquadCoefficients => "json",
        }
    }

    pub fn default_file_name(&self) -> &'static str {
        match self {
            ExportFormat::CamillaDsp => "room_eq_cdsp.yaml",
            ExportFormat::EqualizerApo => "room_eq.txt",
            ExportFormat::EasyEffects => "room_eq.json",
            ExportFormat::Wavelet => "room_eq.txt",
            ExportFormat::PipeWire => "room_eq.conf",
            ExportFormat::RoonDsp => "room_eq.json",
            ExportFormat::Rew => "room_eq_rew.txt",
            ExportFormat::BiquadCoefficients => "room_eq_biquads.json",
        }
    }

    pub fn default_export_path(&self, output_path: &Path) -> PathBuf {
        if matches!(self, ExportFormat::CamillaDsp)
            && let Some(stem) = output_path.file_stem().and_then(|stem| stem.to_str())
        {
            let mut path = output_path.to_path_buf();
            path.set_file_name(format!("{stem}_cdsp.{}", self.default_extension()));
            return path;
        }

        output_path.with_extension(self.default_extension())
    }
}

pub fn external_export_supported(output: &DspGraph, format: ExportFormat) -> anyhow::Result<()> {
    ensure_external_export_supported(output, format)
}

pub(super) fn ensure_external_export_supported(
    output: &DspGraph,
    format: ExportFormat,
) -> anyhow::Result<()> {
    if output
        .global_plugins
        .iter()
        .chain(output.channels.values().flat_map(|chain| {
            chain.plugins.iter().chain(
                chain
                    .drivers
                    .iter()
                    .flatten()
                    .flat_map(|driver| driver.plugins.iter()),
            )
        }))
        .any(|plugin| plugin.plugin_type == "limiter")
    {
        anyhow::bail!(
            "{format:?} export cannot preserve the mandatory runtime sub-output limiter; use native playback"
        );
    }
    let has_routed_bass_management = has_routed_bass_management(output);
    let has_global_plugins = !output.global_plugins.is_empty();

    if matches!(format, ExportFormat::CamillaDsp) {
        if (has_routed_bass_management || has_global_plugins)
            && !has_only_bass_management_matrix(output)
        {
            return unsupported_graph_error(format);
        }
        return validate_camilladsp_input(output, None);
    }

    // Equalizer APO has its own graph renderer. It performs stricter
    // capability checks at render time because some static Channel/Copy
    // routings are representable while arbitrary graphs are not.
    if matches!(format, ExportFormat::EqualizerApo) {
        if !has_routed_bass_management && !has_global_plugins {
            return validate_serial_external_input(output, format);
        }
        return Ok(());
    }

    if !has_routed_bass_management && !has_global_plugins {
        if matches!(format, ExportFormat::PipeWire) {
            return validate_pipewire_input(output, None);
        }
        return validate_serial_external_input(output, format);
    }

    unsupported_graph_error(format)
}

fn unsupported_graph_error(format: ExportFormat) -> anyhow::Result<()> {
    anyhow::bail!(
        "{format:?} export cannot represent routed home-cinema bass management safely. \
         Use SotF JSON or Apply as Graph so global_plugins and route-level bass-management DSP are preserved."
    );
}

fn has_routed_bass_management(output: &DspGraph) -> bool {
    output
        .metadata
        .as_ref()
        .and_then(|metadata| metadata.bass_management.as_ref())
        .and_then(|report| report.routing_graph.as_ref())
        .is_some_and(|graph| !graph.routes.is_empty())
        || output.global_plugins.iter().any(|plugin| {
            plugin.plugin_type == "matrix"
                && plugin
                    .parameters
                    .get("metadata")
                    .and_then(|metadata| metadata.get("routes"))
                    .and_then(|routes| routes.as_array())
                    .is_some_and(|routes| !routes.is_empty())
        })
}

fn has_only_bass_management_matrix(output: &DspGraph) -> bool {
    has_routed_bass_management(output)
        && output.global_plugins.iter().all(|plugin| {
            plugin.plugin_type == "matrix"
                && plugin
                    .parameters
                    .get("label")
                    .and_then(|label| label.as_str())
                    == Some("home_cinema_bass_management")
        })
}
