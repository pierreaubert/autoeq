//! Filesystem adapter for crate-owned external export package generation.

use anyhow::Context;
use roomeq_export::{
    ConvolutionResource, ExportFormat, ExportPackage, build_export_package,
    checked_convolution_resource_references, render_dsp_graph,
};
use roomeq_model::DspGraph;
use std::collections::{BTreeSet, HashMap};
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// Bind the final graph to available artifact-store bytes. Missing resources
/// remain explicitly unbound, so later export cannot silently bless new bytes.
pub(crate) fn bind_final_convolution_artifacts(
    result: &mut roomeq_engine::room_result::RoomOptimizationResult,
    source_dir: &Path,
    store: &dyn autoeq_artifacts::ArtifactStore,
    sample_rate: f64,
) -> autoeq_core::Result<()> {
    validate_final_routed_stage_ownership(result)?;
    let graph = result.to_dsp_chain_output();
    let mut inventory = std::collections::BTreeMap::new();
    let references = checked_convolution_resource_references(&graph).map_err(|error| {
        autoeq_core::AutoeqError::InvalidMeasurement { message: error.to_string() }
    })?;
    for reference in references {
        let path = Path::new(&reference);
        let path = if path.is_absolute() {
            path.to_path_buf()
        } else {
            source_dir.join(path)
        };
        let bytes = store.read(&path)?;
        if let Some(bytes) = bytes.as_ref() {
            // Validate every delivered resource, including driver/multi-FIR
            // resources without a retained-tap owner. A hash alone says nothing
            // about whether these bytes can be played at the workflow rate.
            let channels = crate::ctc::read_wav_bytes_channels_f64(
                bytes,
                &path,
                crate::ctc::checked_sample_rate(sample_rate)?,
                "final artifact-store FIR",
            )?;
            if channels.is_empty()
                || channels.iter().any(|channel| {
                    channel.is_empty()
                        || channel.len() != channels[0].len()
                        || channel.iter().any(|tap| !tap.is_finite())
                })
            {
                return Err(autoeq_core::AutoeqError::InvalidMeasurement {
                    message: format!(
                        "final artifact-store FIR '{}' has empty, unequal, or nonfinite channels",
                        path.display()
                    ),
                });
            }
            // Only a single channel-level convolution has unambiguous retained
            // FIR ownership. Do not infer ownership for driver or multi-FIR chains.
            for (owner, chain) in &result.channels {
                let convolutions: Vec<_> = chain
                    .plugins
                    .iter()
                    .filter(|plugin| plugin.plugin_type == "convolution")
                    .collect();
                if convolutions.len() != 1
                    || convolutions[0]
                        .parameters
                        .get("ir_file")
                        .and_then(|v| v.as_str())
                        != Some(reference.as_str())
                {
                    continue;
                }
                if let Some(taps) = result
                    .channel_results
                    .get(owner)
                    .and_then(|c| c.fir_coeffs.as_ref())
                {
                    if channels.len() != 1
                        || taps.is_empty()
                        || channels[0].len() != taps.len()
                        || channels[0].iter().zip(taps).any(|(stored, retained)| {
                            !stored.is_finite()
                                || !retained.is_finite()
                                || (*stored as f32) != (*retained as f32)
                        })
                    {
                        return Err(autoeq_core::AutoeqError::InvalidMeasurement {
                            message: format!(
                                "final artifact-store FIR '{}' conflicts with retained FIR for '{owner}'",
                                path.display()
                            ),
                        });
                    }
                }
            }
        }
        let hash = bytes.map(|bytes| {
            ConvolutionResource {
                reference: reference.clone(),
                bytes: bytes.into(),
            }
            .sha256()
        });
        inventory.insert(reference, hash);
    }
    result.metadata.final_convolution_sha256 = Some(inventory);
    Ok(())
}

/// Routed channel plugins are selected by stage in both replay and export.
/// Reject unowned operations rather than certifying a silently shortened chain.
/// Driver plugins have explicit branch ownership and are replayed as a whole.
pub(crate) fn validate_final_routed_stage_ownership(
    result: &roomeq_engine::room_result::RoomOptimizationResult,
) -> autoeq_core::Result<()> {
    if result.metadata.bass_management.as_ref()
        .and_then(|report| report.routing_graph.as_ref()).is_none() {
        return Ok(());
    }
    let mut channels: Vec<_> = result.channels.iter().collect();
    channels.sort_by_key(|(name, _)| *name);
    for (name, chain) in channels {
        for (index, plugin) in chain.plugins.iter().enumerate() {
            let stage = plugin.parameters.get("room_eq_stage").and_then(|value| value.as_str());
            if !matches!(stage, Some("pre_route" | "post_route" | "route_owned")) {
                return Err(autoeq_core::AutoeqError::InvalidMeasurement {
                    message: format!("routed channel '{name}' plugin #{index} ('{}') requires explicit pre_route, post_route or route_owned ownership; got {stage:?}", plugin.plugin_type),
                });
            }
        }
    }
    Ok(())
}

/// Render and persist one external artifact without packaging sidecars.
pub fn export_dsp_chain(
    graph: &DspGraph,
    format: ExportFormat,
    path: &Path,
    sample_rate: f64,
) -> anyhow::Result<()> {
    anyhow::ensure!(!graph.metadata.as_ref().and_then(|metadata| metadata.final_convolution_sha256.as_ref()).is_some_and(|inventory| !inventory.is_empty()),
        "bound convolution artifacts require export with verified sidecars");
    let content = render_dsp_graph(graph, format, sample_rate)?;
    if format == ExportFormat::CamillaDsp {
        log_backend_delay(&roomeq_export::camilladsp_delay_realization(graph, sample_rate)?);
    }
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
    checked_convolution_resource_references(graph)?
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

fn log_backend_delay(report: &roomeq_export::CamillaDspDelayRealization) {
    if report.fractional_delay_present {
        log::info!("CamillaDSP fractional-delay realization adds {} common samples ({:.6} ms); delay usable band 0..{} Hz. This excludes requested delays, existing FIR latency and device buffers.",
            report.common_padding_samples, report.additional_latency_ms, report.usable_band_upper_hz);
    }
}

fn persist_export_package(package: &ExportPackage, destination_dir: &Path) -> anyhow::Result<()> {
    // Validate the entire set before opening even its first destination file.
    package.validate_integrity()?;
    if let Some(report) = package.camilladsp_delay_realization()? {
        log_backend_delay(&report);
    }
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

    fn mono_fir_bytes(taps: &[f32], sample_rate: u32) -> Vec<u8> {
        let mut bytes = std::io::Cursor::new(Vec::new());
        {
            let mut writer = hound::WavWriter::new(
                &mut bytes,
                hound::WavSpec {
                    channels: 1,
                    sample_rate,
                    bits_per_sample: 32,
                    sample_format: hound::SampleFormat::Float,
                },
            )
            .unwrap();
            for &tap in taps {
                writer.write_sample(tap).unwrap();
            }
            writer.finalize().unwrap();
        }
        bytes.into_inner()
    }

    #[test]
    fn final_store_fir_binding_checks_rate_length_finiteness_and_float32_identity() {
        use autoeq_artifacts::{ArtifactStore, MemoryArtifactStore};
        for rate in [44_100, 48_000, 96_000] {
            let source = tempfile::tempdir().unwrap();
            let store = MemoryArtifactStore::default();
            let mut result = crate::test_fixtures::single_channel_room_result("left");
            result.channels = convolution_graph("impulse.wav").channels;
            let taps = vec![0.123456789, -0.234567891];
            result.channel_results.get_mut("left").unwrap().fir_coeffs = Some(taps.clone());
            let rounded: Vec<f32> = taps.iter().map(|&tap| tap as f32).collect();
            for bytes in [
                mono_fir_bytes(&rounded, rate + 1),
                mono_fir_bytes(&rounded[..1], rate),
                mono_fir_bytes(&[f32::NAN, rounded[1]], rate),
                mono_fir_bytes(&[f32::INFINITY, rounded[1]], rate),
                b"not a WAV".to_vec(),
            ] {
                store
                    .write(&source.path().join("impulse.wav"), &bytes)
                    .unwrap();
                assert!(
                    bind_final_convolution_artifacts(
                        &mut result,
                        source.path(),
                        &store,
                        rate as f64
                    )
                    .is_err()
                );
                assert!(result.metadata.final_convolution_sha256.is_none());
            }
            let bytes = mono_fir_bytes(&rounded, rate);
            store
                .write(&source.path().join("impulse.wav"), &bytes)
                .unwrap();
            bind_final_convolution_artifacts(&mut result, source.path(), &store, rate as f64)
                .unwrap();
            let expected = ConvolutionResource {
                reference: "impulse.wav".into(),
                bytes: bytes.into(),
            }
            .sha256();
            assert_eq!(
                result.metadata.final_convolution_sha256.as_ref().unwrap()["impulse.wav"],
                Some(expected)
            );
        }
    }

    #[test]
    fn final_store_binding_rejects_missing_or_invalid_convolution_reference() {
        let source = tempfile::tempdir().unwrap();
        let store = autoeq_artifacts::MemoryArtifactStore::default();
        for parameters in [
            serde_json::json!({}),
            serde_json::json!({"ir_file": null}),
            serde_json::json!({"ir_file": 42}),
            serde_json::json!({"ir_file": ""}),
            serde_json::json!({"ir_file": "  "}),
        ] {
            let mut result = crate::test_fixtures::single_channel_room_result("left");
            result.channels = convolution_graph("impulse.wav").channels;
            result.channels.get_mut("left").unwrap().plugins[0].parameters = parameters;
            assert!(bind_final_convolution_artifacts(
                &mut result, source.path(), &store, 48_000.0
            ).is_err(), "malformed convolution declaration was omitted from final evidence");
            assert!(result.metadata.final_convolution_sha256.is_none());
        }
    }

    #[test]
    fn final_store_binding_validates_resources_without_retained_taps() {
        use autoeq_artifacts::{ArtifactStore, MemoryArtifactStore};
        let source = tempfile::tempdir().unwrap();
        let store = MemoryArtifactStore::default();
        for rate in [44_100, 48_000, 96_000] {
            for bytes in [
                b"not a WAV".to_vec(),
                mono_fir_bytes(&[0.5], rate + 1),
                mono_fir_bytes(&[], rate),
                mono_fir_bytes(&[f32::NAN], rate),
                mono_fir_bytes(&[f32::INFINITY], rate),
            ] {
                let mut result = crate::test_fixtures::single_channel_room_result("left");
                result.channels = convolution_graph("impulse.wav").channels;
                result.channel_results.get_mut("left").unwrap().fir_coeffs = None;
                store.write(&source.path().join("impulse.wav"), &bytes).unwrap();
                assert!(bind_final_convolution_artifacts(
                    &mut result, source.path(), &store, rate as f64
                ).is_err(), "invalid unretained resource was bound at {rate} Hz");
                assert!(result.metadata.final_convolution_sha256.is_none());
            }
            let mut result = crate::test_fixtures::single_channel_room_result("left");
            result.channels = convolution_graph("impulse.wav").channels;
            result.channel_results.get_mut("left").unwrap().fir_coeffs = None;
            let bytes = mono_fir_bytes(&[0.5, -0.25], rate);
            store.write(&source.path().join("impulse.wav"), &bytes).unwrap();
            bind_final_convolution_artifacts(
                &mut result, source.path(), &store, rate as f64
            ).unwrap();
            let expected = ConvolutionResource {
                reference: "impulse.wav".into(), bytes: bytes.into(),
            }.sha256();
            assert_eq!(result.metadata.final_convolution_sha256.as_ref().unwrap()["impulse.wav"], Some(expected));
        }
    }

    #[test]
    fn final_store_binding_rejects_fir_different_from_retained_acceptance_taps() {
        use autoeq_artifacts::{ArtifactStore, MemoryArtifactStore};
        let source = tempfile::tempdir().unwrap();
        let store = MemoryArtifactStore::default();
        store
            .write(
                &source.path().join("impulse.wav"),
                &mono_fir_bytes(&[0.5], 48_000),
            )
            .unwrap();
        let mut result = crate::test_fixtures::single_channel_room_result("left");
        result.channels = convolution_graph("impulse.wav").channels;
        result.channel_results.get_mut("left").unwrap().fir_coeffs = Some(vec![0.123456789]);
        let error = bind_final_convolution_artifacts(&mut result, source.path(), &store, 48_000.0)
            .unwrap_err();
        assert!(error.to_string().contains("retained FIR"), "{error}");
        assert!(result.metadata.final_convolution_sha256.is_none());
    }

    #[test]
    fn final_store_binding_rejects_replaced_source_and_survives_packaging() {
        use autoeq_artifacts::{ArtifactStore, MemoryArtifactStore};
        let store = MemoryArtifactStore::default();
        let source = tempfile::tempdir().unwrap();
        let destination = tempfile::tempdir().unwrap();
        let reference = "impulse.wav";
        let original = mono_fir_bytes(&[0.5], 48_000);
        store.write(&source.path().join(reference), &original).unwrap();
        let mut result = crate::test_fixtures::single_channel_room_result("left");
        result.channels = convolution_graph(reference).channels;
        bind_final_convolution_artifacts(&mut result, source.path(), &store, 48_000.0).unwrap();
        let graph: DspGraph = serde_json::from_slice(&serde_json::to_vec(&result.to_dsp_chain_output()).unwrap()).unwrap();
        std::fs::write(source.path().join(reference), b"replacement").unwrap();
        let target = destination.path().join("room.yml");
        let error = export_dsp_chain_with_convolution_sidecars(&graph, ExportFormat::CamillaDsp,
            &target, 48_000.0, source.path()).unwrap_err();
        assert!(error.to_string().contains("changed since workflow completion"), "{error}");
        assert!(!target.exists());
        std::fs::write(source.path().join(reference), original).unwrap();
        std::fs::write(destination.path().join(reference), b"occupied").unwrap();
        let packaged = package_convolution_sidecars(&graph, source.path(), destination.path()).unwrap();
        assert_eq!(convolution_reference(&packaged), "impulse_002.wav");
        assert!(packaged.metadata.as_ref().unwrap().final_convolution_sha256.as_ref().unwrap().contains_key("impulse_002.wav"));
        export_dsp_chain_with_convolution_sidecars(&packaged, ExportFormat::CamillaDsp,
            &target, 48_000.0, destination.path()).unwrap();
        assert!(export_dsp_chain(&graph, ExportFormat::CamillaDsp, &target, 48_000.0).is_err());
    }

    #[test]
    fn missing_final_store_resource_cannot_be_bound_later_by_export() {
        let store = autoeq_artifacts::MemoryArtifactStore::default();
        let source = tempfile::tempdir().unwrap();
        let destination = tempfile::tempdir().unwrap();
        let mut result = crate::test_fixtures::single_channel_room_result("left");
        result.channels = convolution_graph("missing.wav").channels;
        bind_final_convolution_artifacts(&mut result, source.path(), &store, 48_000.0).unwrap();
        assert_eq!(result.metadata.final_convolution_sha256.as_ref().unwrap()["missing.wav"], None);
        std::fs::write(source.path().join("missing.wav"), b"newly appeared").unwrap();
        let error = export_dsp_chain_with_convolution_sidecars(&result.to_dsp_chain_output(),
            ExportFormat::CamillaDsp, &destination.path().join("room.yml"), 48_000.0, source.path()).unwrap_err();
        assert!(error.to_string().contains("unbound"), "{error}");
    }

    #[test]
    fn stale_package_hash_cannot_overwrite_existing_output() {
        use roomeq_export::ExportPackageMember;
        let destination = tempfile::tempdir().unwrap();
        let output = destination.path().join("config.json");
        std::fs::write(&output, b"existing accepted config").unwrap();
        let mut package = ExportPackage::new(vec![
            ExportPackageMember::new("config.json", b"replacement config".to_vec()).unwrap(),
            ExportPackageMember::new("impulse.wav", b"accepted impulse".to_vec()).unwrap(),
        ]).unwrap();
        // Keep the recorded identity, but replace a later member's actual bytes.
        package.members[1].bytes = b"stale or replaced impulse".to_vec().into();
        let error = persist_export_package(&package, destination.path()).unwrap_err();
        assert!(error.to_string().contains("content hash mismatch"), "{error}");
        assert_eq!(std::fs::read(&output).unwrap(), b"existing accepted config");
        assert!(!destination.path().join("impulse.wav").exists());
    }

    #[test]
    fn mutated_package_paths_are_rejected_before_creating_destination() {
        use roomeq_export::ExportPackageMember;
        let directory = tempfile::tempdir().unwrap();
        let destination = directory.path().join("not_created");
        let mut member = ExportPackageMember::new("impulse.wav", vec![0_u8]).unwrap();
        member.relative_path = "../escaped.wav".into();
        let package = ExportPackage { members: vec![member] };
        assert!(persist_export_package(&package, &destination).is_err());
        assert!(!destination.exists());
        assert!(!directory.path().join("escaped.wav").exists());
    }
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
