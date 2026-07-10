use super::super::types::DspChainOutput;
use super::conformance::validate_camilladsp_input;
use std::path::{Path, PathBuf};

/// Supported export formats for DSP chain output
#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum ExportFormat {
    /// CamillaDSP YAML configuration
    #[value(name = "camilladsp")]
    CamillaDsp,
    /// Equalizer APO / Peace GUI text format (also works with PipeWire parametric-equalizer module)
    #[value(name = "apo")]
    EqualizerApo,
    /// EasyEffects JSON preset
    #[value(name = "easyeffects")]
    EasyEffects,
    /// Wavelet GraphicEQ text format
    #[value(name = "wavelet")]
    Wavelet,
    /// PipeWire filter-chain SPA-JSON configuration
    #[value(name = "pipewire")]
    PipeWire,
    /// Roon DSP Engine preset (JSON)
    #[value(name = "roon")]
    RoonDsp,
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

    if !has_routed_bass_management && !has_global_plugins {
        return Ok(());
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
