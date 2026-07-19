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
