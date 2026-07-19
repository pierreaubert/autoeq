//! Filesystem adapter for crate-owned external export package generation.

use anyhow::Context;
use roomeq_export::{
    ConvolutionResource, ExportFormat, ExportPackage, build_export_package,
    convolution_resource_references, render_dsp_graph,
};
use roomeq_model::DspGraph;
use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

/// Render and persist one external artifact without packaging sidecars.
pub fn export_dsp_chain(
    graph: &DspGraph,
    format: ExportFormat,
    path: &Path,
    sample_rate: f64,
) -> anyhow::Result<()> {
    let content = render_dsp_graph(graph, format, sample_rate)?;
    std::fs::write(path, content)
        .with_context(|| format!("failed to write external export '{}'", path.display()))?;
    Ok(())
}

/// Load exactly the graph-declared convolution resources, build an explicit
/// package in `roomeq-export`, then persist every returned member.
pub fn export_dsp_chain_with_convolution_sidecars(
    graph: &DspGraph,
    format: ExportFormat,
    path: &Path,
    sample_rate: f64,
    source_dir: &Path,
) -> anyhow::Result<()> {
    let destination_dir = path.parent().unwrap_or_else(|| Path::new("."));
    let main_file_name = path
        .file_name()
        .map(PathBuf::from)
        .context("external export path must include a file name")?;
    let resources = load_convolution_resources(graph, source_dir)?;
    let occupied_names = occupied_member_names(destination_dir)?;
    let package = build_export_package(
        graph,
        format,
        &main_file_name,
        sample_rate,
        &resources,
        &occupied_names,
    )?;
    persist_export_package(&package, destination_dir)
}

/// Compatibility adapter for callers that only want convolution sidecars and
/// a graph rewritten to package-local references.
pub fn package_convolution_sidecars(
    graph: &DspGraph,
    source_dir: &Path,
    destination_dir: &Path,
) -> anyhow::Result<DspGraph> {
    let resources = load_convolution_resources(graph, source_dir)?;
    let occupied_names = occupied_member_names(destination_dir)?;
    let (graph, members) =
        roomeq_export::package_convolution_sidecars(graph, &resources, &occupied_names)?;
    persist_export_package(&ExportPackage::new(members)?, destination_dir)?;
    Ok(graph)
}

fn load_convolution_resources(
    graph: &DspGraph,
    source_dir: &Path,
) -> anyhow::Result<Vec<ConvolutionResource>> {
    convolution_resource_references(graph)
        .into_iter()
        .map(|reference| {
            let path = Path::new(&reference);
            let path = if path.is_absolute() {
                path.to_path_buf()
            } else {
                source_dir.join(path)
            };
            let bytes = std::fs::read(&path).with_context(|| {
                format!(
                    "convolution resource '{}' was not found at '{}'",
                    reference,
                    path.display()
                )
            })?;
            Ok(ConvolutionResource { reference, bytes })
        })
        .collect()
}

fn occupied_member_names(directory: &Path) -> anyhow::Result<BTreeSet<String>> {
    if !directory.exists() {
        return Ok(BTreeSet::new());
    }
    let entries = std::fs::read_dir(directory).with_context(|| {
        format!(
            "failed to inspect export directory '{}'",
            directory.display()
        )
    })?;
    let mut names = BTreeSet::new();
    for entry in entries {
        let entry = entry.with_context(|| {
            format!(
                "failed to read export directory entry in '{}'",
                directory.display()
            )
        })?;
        if let Ok(name) = entry.file_name().into_string() {
            names.insert(name);
        }
    }
    Ok(names)
}

fn persist_export_package(package: &ExportPackage, destination_dir: &Path) -> anyhow::Result<()> {
    std::fs::create_dir_all(destination_dir).with_context(|| {
        format!(
            "failed to create export directory '{}'",
            destination_dir.display()
        )
    })?;
    for member in &package.members {
        let path = destination_dir.join(&member.relative_path);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(&path, &member.bytes).with_context(|| {
            format!(
                "failed to persist export package member '{}' (sha256 {})",
                path.display(),
                member.sha256
            )
        })?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use roomeq_model::{ChannelDspChain, PluginConfigWrapper, default_config_version};
    use serde_json::json;
    use std::collections::HashMap;

    fn graph_with_plugins(plugins: Vec<PluginConfigWrapper>) -> DspGraph {
        DspGraph {
            version: default_config_version(),
            global_plugins: Vec::new(),
            channels: HashMap::from([(
                "left".to_string(),
                ChannelDspChain {
                    channel: "left".to_string(),
                    plugins,
                    drivers: None,
                    initial_curve: None,
                    final_curve: None,
                    eq_response: None,
                    target_curve: None,
                    pre_ir: None,
                    post_ir: None,
                    fir_temporal_masking: None,
                    direct_early_late_correction: None,
                },
            )]),
            metadata: None,
        }
    }

    fn convolution_graph(reference: &str) -> DspGraph {
        graph_with_plugins(vec![PluginConfigWrapper {
            plugin_type: "convolution".to_string(),
            parameters: json!({"ir_file": reference}),
        }])
    }

    fn convolution_reference(graph: &DspGraph) -> &str {
        graph.channels["left"].plugins[0].parameters["ir_file"]
            .as_str()
            .unwrap()
    }

    #[test]
    fn export_dsp_chain_persists_rendered_content() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("room.yml");
        let graph = graph_with_plugins(vec![PluginConfigWrapper {
            plugin_type: "gain".to_string(),
            parameters: json!({"gain_db": -1.5}),
        }]);

        export_dsp_chain(&graph, ExportFormat::CamillaDsp, &path, 48_000.0).unwrap();

        let rendered = std::fs::read_to_string(path).unwrap();
        assert!(rendered.contains("samplerate: 48000"));
    }

    #[test]
    fn package_convolution_sidecars_loads_rewrites_and_persists_resource() {
        let source = tempfile::tempdir().unwrap();
        let destination = tempfile::tempdir().unwrap();
        std::fs::write(source.path().join("impulse.wav"), b"new impulse").unwrap();
        std::fs::write(destination.path().join("impulse.wav"), b"occupied").unwrap();

        let packaged = package_convolution_sidecars(
            &convolution_graph("impulse.wav"),
            source.path(),
            destination.path(),
        )
        .unwrap();

        assert_eq!(convolution_reference(&packaged), "impulse_002.wav");
        assert_eq!(
            std::fs::read(destination.path().join("impulse_002.wav")).unwrap(),
            b"new impulse"
        );
    }

    #[test]
    fn export_with_convolution_sidecars_persists_complete_package() {
        let source = tempfile::tempdir().unwrap();
        let destination = tempfile::tempdir().unwrap();
        std::fs::write(source.path().join("impulse.wav"), b"impulse").unwrap();
        let path = destination.path().join("room.yml");

        export_dsp_chain_with_convolution_sidecars(
            &convolution_graph("impulse.wav"),
            ExportFormat::CamillaDsp,
            &path,
            48_000.0,
            source.path(),
        )
        .unwrap();

        assert!(
            std::fs::read_to_string(path)
                .unwrap()
                .contains("impulse.wav")
        );
        assert_eq!(
            std::fs::read(destination.path().join("impulse.wav")).unwrap(),
            b"impulse"
        );
    }
}
