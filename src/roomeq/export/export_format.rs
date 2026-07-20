use super::super::types::DspChainOutput;
use super::conformance::{
    validate_camilladsp_input, validate_pipewire_input, validate_serial_external_input,
};
pub use roomeq_export::ExternalExportFormat as ExportFormat;

pub fn external_export_supported(
    output: &DspChainOutput,
    format: ExportFormat,
) -> anyhow::Result<()> {
    ensure_external_export_supported(output, format)
}

pub(super) fn ensure_external_export_supported(
    output: &DspChainOutput,
    format: ExportFormat,
) -> anyhow::Result<()> {
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

fn has_routed_bass_management(output: &DspChainOutput) -> bool {
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

fn has_only_bass_management_matrix(output: &DspChainOutput) -> bool {
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
