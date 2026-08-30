//! Filesystem adapter for crate-owned external export package generation.

use anyhow::Context;
use roomeq_export::{
    ConvolutionResource, ExportFormat, ExportPackage, build_export_package,
    convolution_resource_references, render_dsp_graph,
};
use roomeq_model::DspGraph;
use std::collections::{BTreeSet, HashMap};
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::Arc;

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
    let reusable_names = reusable_member_names(destination_dir, &resources, &occupied_names)?;
    let package = build_export_package(
        graph,
        format,
        &main_file_name,
        sample_rate,
        &resources,
        &occupied_names,
        &reusable_names,
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
    let reusable_names = reusable_member_names(destination_dir, &resources, &occupied_names)?;
    let (graph, members) = roomeq_export::package_convolution_sidecars(
        graph,
        &resources,
        &occupied_names,
        &reusable_names,
    )?;
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
            Ok(ConvolutionResource {
                reference,
                bytes: Arc::from(bytes),
            })
        })
        .collect()
}

fn reusable_member_names(
    directory: &Path,
    resources: &[ConvolutionResource],
    occupied_names: &BTreeSet<String>,
) -> anyhow::Result<HashMap<String, String>> {
    let mut reusable = HashMap::new();
    for resource in resources {
        let preferred = Path::new(&resource.reference)
            .file_name()
            .and_then(|name| name.to_str())
            .filter(|name| !name.is_empty())
            .unwrap_or("room_eq_ir.wav");
        for candidate in occupied_names {
            if member_name_is_variant(preferred, candidate)
                && same_existing_content(&directory.join(candidate), &resource.bytes)?
            {
                reusable.insert(resource.reference.clone(), candidate.clone());
                break;
            }
        }
    }
    Ok(reusable)
}

fn member_name_is_variant(preferred: &str, candidate: &str) -> bool {
    if candidate == preferred {
        return true;
    }
    let preferred = Path::new(preferred);
    let stem = preferred
        .file_stem()
        .and_then(|stem| stem.to_str())
        .filter(|stem| !stem.is_empty())
        .unwrap_or("room_eq_ir");
    let extension = preferred
        .extension()
        .and_then(|extension| extension.to_str())
        .filter(|extension| !extension.is_empty())
        .map(|extension| format!(".{extension}"))
        .unwrap_or_default();
    let Some(suffix) = candidate
        .strip_prefix(&format!("{stem}_"))
        .and_then(|candidate| candidate.strip_suffix(&extension))
    else {
        return false;
    };
    suffix.len() >= 3 && suffix.bytes().all(|byte| byte.is_ascii_digit())
}

fn same_existing_content(path: &Path, expected: &[u8]) -> anyhow::Result<bool> {
    let metadata = match std::fs::metadata(path) {
        Ok(metadata) if metadata.is_file() => metadata,
        Ok(_) => return Ok(false),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(false),
        Err(error) => {
            return Err(error).with_context(|| {
                format!(
                    "failed to inspect existing export member '{}'",
                    path.display()
                )
            });
        }
    };
    if metadata.len() != expected.len() as u64 {
        return Ok(false);
    }
    let mut file = std::fs::File::open(path)?;
    let mut offset = 0;
    let mut buffer = [0_u8; 64 * 1024];
    loop {
        let count = file.read(&mut buffer)?;
        if count == 0 {
            return Ok(offset == expected.len());
        }
        if expected.get(offset..offset + count) != Some(&buffer[..count]) {
            return Ok(false);
        }
        offset += count;
    }
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
            deployed_source_curves: Default::default(),
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
            std::fs::read_to_string(&path)
                .unwrap()
                .contains("impulse.wav")
        );
        assert_eq!(
            std::fs::read(destination.path().join("impulse.wav")).unwrap(),
            b"impulse"
        );

        export_dsp_chain_with_convolution_sidecars(
            &convolution_graph("impulse.wav"),
            ExportFormat::CamillaDsp,
            &path,
            48_000.0,
            source.path(),
        )
        .unwrap();

        let mut names = std::fs::read_dir(destination.path())
            .unwrap()
            .map(|entry| entry.unwrap().file_name())
            .collect::<Vec<_>>();
        names.sort();
        assert_eq!(names, ["impulse.wav", "room.yml"]);
        assert!(
            std::fs::read_to_string(&path)
                .unwrap()
                .contains("impulse.wav")
        );
    }
}
