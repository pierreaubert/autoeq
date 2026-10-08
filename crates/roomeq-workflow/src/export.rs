//! Filesystem adapter for crate-owned external export package generation.

use anyhow::Context;
use roomeq_export::{
    ConvolutionResource, ExportFormat, ExportPackage, ExportPackageMember, build_export_package,
    checked_convolution_resource_references, render_dsp_graph,
};
use roomeq_model::DspGraph;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeSet, HashMap};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;

const MAX_EXPORT_MEMBER_BYTES: u64 = 256 * 1024 * 1024;
const MAX_EXPORT_PACKAGE_BYTES: u64 = 512 * 1024 * 1024;
const MAX_EXPORT_PACKAGE_FILES: usize = 50_000;
const MAX_EXPORT_MEMBER_PATH_BYTES: usize = 1_024;
const MAX_EXTERNAL_DIRECTORY_PATH_BYTES: usize = 16 * 1024;
const MAX_EXTERNAL_EXPORT_JOURNAL_BYTES: u64 = 16 * 1024 * 1024;
const EXTERNAL_EXPORT_JOURNAL_SCHEMA_VERSION: u32 = 1;

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
        autoeq_core::AutoeqError::InvalidMeasurement {
            message: error.to_string(),
        }
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
                    && (channels.len() != 1
                        || taps.is_empty()
                        || channels[0].len() != taps.len()
                        || channels[0].iter().zip(taps).any(|(stored, retained)| {
                            !stored.is_finite()
                                || !retained.is_finite()
                                || (*stored as f32) != (*retained as f32)
                        }))
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
    if result
        .metadata
        .bass_management
        .as_ref()
        .and_then(|report| report.routing_graph.as_ref())
        .is_none()
    {
        return Ok(());
    }
    let mut channels: Vec<_> = result.channels.iter().collect();
    channels.sort_by_key(|(name, _)| *name);
    for (name, chain) in channels {
        for (index, plugin) in chain.plugins.iter().enumerate() {
            let stage = plugin
                .parameters
                .get("room_eq_stage")
                .and_then(|value| value.as_str());
            if !matches!(stage, Some("pre_route" | "post_route" | "route_owned")) {
                return Err(autoeq_core::AutoeqError::InvalidMeasurement {
                    message: format!(
                        "routed channel '{name}' plugin #{index} ('{}') requires explicit pre_route, post_route or route_owned ownership; got {stage:?}",
                        plugin.plugin_type
                    ),
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
    anyhow::ensure!(
        !graph
            .metadata
            .as_ref()
            .and_then(|metadata| metadata.final_convolution_sha256.as_ref())
            .is_some_and(|inventory| !inventory.is_empty()),
        "bound convolution artifacts require export with verified sidecars"
    );
    let content = render_dsp_graph(graph, format, sample_rate)?;
    if format == ExportFormat::CamillaDsp {
        log_backend_delay(&roomeq_export::camilladsp_delay_realization(
            graph,
            sample_rate,
        )?);
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
    let file_name = path
        .file_name()
        .context("external export path must include a file name")?;
    let parent = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    let staging = tempfile::Builder::new()
        .prefix(".roomeq-export-")
        .tempdir_in(parent)
        .with_context(|| {
            format!(
                "failed to stage external export beside '{}'",
                path.display()
            )
        })?;
    let staged_path = staging.path().join(file_name);
    export_dsp_chain_with_convolution_sidecars_to_staging(
        graph,
        format,
        &staged_path,
        sample_rate,
        source_dir,
        path,
    )?;
    publish_staged_export_package_with(&staged_path, path, || Ok(()))
}

/// Render an export into a staging directory using the final directory's names.
///
/// This lets callers prepare every package member without changing a prior
/// external export. `staging_path` and `destination_path` must use the same
/// file name so relative package references remain valid after publication.
///
/// # Errors
/// Returns an error when graph rendering, resource loading, or staging fails.
pub fn export_dsp_chain_with_convolution_sidecars_to_staging(
    graph: &DspGraph,
    format: ExportFormat,
    staging_path: &Path,
    sample_rate: f64,
    source_dir: &Path,
    destination_path: &Path,
) -> anyhow::Result<()> {
    let destination_dir = staging_path.parent().unwrap_or_else(|| Path::new("."));
    let naming_dir = destination_path.parent().unwrap_or_else(|| Path::new("."));
    let main_file_name = destination_path
        .file_name()
        .map(PathBuf::from)
        .context("external export path must include a file name")?;
    anyhow::ensure!(
        staging_path.file_name() == Some(main_file_name.as_os_str()),
        "staged and destination export paths must use the same file name"
    );
    let resources = load_convolution_resources(graph, source_dir)?;
    let occupied_names = occupied_member_names(naming_dir)?;
    let reusable_names = reusable_member_names(naming_dir, &resources, &occupied_names)?;
    let package = build_export_package(
        graph,
        format,
        &main_file_name,
        sample_rate,
        &resources,
        &occupied_names,
        &reusable_names,
    )?;
    let _rollback = persist_export_package(&package, destination_dir, Some(&main_file_name))?;
    Ok(())
}

/// Publish a staged external package, then run the native-bundle commit.
///
/// If the native commit fails, this restores the previous main export and
/// removes sidecars created by this publication. Existing identical sidecars
/// remain in place. This synchronous rollback is not crash-atomic across the
/// external export and native bundle: interruption between their commits can
/// leave a newer external package beside the prior native graph.
///
/// # Errors
/// Returns an error when the staged package is invalid, installation fails,
/// or the native commit fails.
pub fn publish_staged_export_package_with(
    staged_main_path: &Path,
    destination_main_path: &Path,
    publish_native: impl FnOnce() -> anyhow::Result<()>,
) -> anyhow::Result<()> {
    let staged_package = read_staged_export_package(staged_main_path)?;
    let destination_dir = destination_main_path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    let main_name = destination_main_path
        .file_name()
        .context("external export path must include a file name")?;
    anyhow::ensure!(
        staged_main_path.file_name() == Some(main_name),
        "staged and destination export paths must use the same file name"
    );

    let rollback =
        persist_export_package(&staged_package, destination_dir, Some(Path::new(main_name)))?;
    if let Err(native_error) = publish_native() {
        return match rollback.restore() {
            Ok(()) => Err(native_error
                .context("native bundle publication failed; external export was restored")),
            Err(rollback_error) => Err(anyhow::anyhow!(
                "native bundle publication failed ({native_error:#}); restoring the external export also failed ({rollback_error:#})"
            )),
        };
    }
    Ok(())
}

/// Publish an external package and native bundle under one durable recovery
/// intent. Recovery always resolves the native bundle first, then rolls the
/// external package back when the old native root remains or completes it
/// when the new native root was committed.
///
/// The journal makes restart recovery deterministic. External readers can
/// still observe the package before the native root changes; this is not a
/// cross-filesystem visibility-atomic transaction. If a crash occurs after an
/// external file changes but before its ownership marker is durable, recovery
/// leaves the file and journal in place and fails closed rather than guessing
/// whether another writer created those bytes. On platforms where `std` cannot
/// sync directory entries, directory-entry durability is best-effort.
pub fn publish_staged_export_package_with_native_bundle(
    staged_main_path: &Path,
    destination_main_path: &Path,
    native_output_path: &Path,
    native_source_path: &Path,
    publish_native: impl FnOnce() -> anyhow::Result<()>,
) -> anyhow::Result<()> {
    crate::output_bundle::recover_output_bundle_transactions(native_output_path)
        .context("failed to recover an earlier RoomEQ publication")?;
    let previous_native_sha256 = crate::output_bundle::native_output_sha256(native_output_path)?;
    let next_native_sha256 = crate::output_bundle::native_output_sha256(native_source_path)?
        .context("staged native output does not exist")?;

    let staged_package = read_staged_export_package(staged_main_path)?;
    let destination_dir = destination_main_path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    let main_name = destination_main_path
        .file_name()
        .context("external export path must include a file name")?;
    anyhow::ensure!(
        staged_main_path.file_name() == Some(main_name),
        "staged and destination export paths must use the same file name"
    );

    let mut journal_created = false;
    let persist = persist_export_package_with_hook(
        &staged_package,
        destination_dir,
        Some(Path::new(main_name)),
        |plan| {
            write_external_export_journal(
                native_output_path,
                destination_dir,
                &previous_native_sha256,
                &next_native_sha256,
                plan,
            )?;
            journal_created = true;
            Ok(())
        },
        |member, is_main| {
            mark_external_member_installed(native_output_path, &member.relative_path, is_main)
        },
    );
    let _rollback = match persist {
        Ok(rollback) => rollback,
        Err(error) => {
            return match recover_pending_external_export_transaction(native_output_path) {
                Ok(()) if journal_created => Err(error.context(
                    "external export installation failed; its durable transaction was recovered",
                )),
                Ok(()) => Err(error),
                Err(recovery_error) => Err(anyhow::anyhow!(
                    "external export installation failed ({error:#}); journal recovery failed ({recovery_error:#})"
                )),
            };
        }
    };

    let native_result = publish_native();
    let recovery_result = recover_pending_external_export_transaction(native_output_path);
    let resulting_native_sha256 = crate::output_bundle::native_output_sha256(native_output_path)?;
    match (native_result, recovery_result) {
        (Ok(()), Ok(()))
            if resulting_native_sha256.as_deref() == Some(next_native_sha256.as_str()) =>
        {
            Ok(())
        }
        (Ok(()), Ok(())) => Err(anyhow::anyhow!(
            "native publication returned successfully without installing the expected graph; the external export was rolled back"
        )),
        (Err(native_error), Ok(())) => Err(native_error.context(
            "native bundle publication failed; the external package was recovered against the resulting native root",
        )),
        (Ok(()), Err(recovery_error)) => Err(recovery_error.context(
            "native bundle was published, but external transaction recovery remains pending",
        )),
        (Err(native_error), Err(recovery_error)) => Err(anyhow::anyhow!(
            "native bundle publication failed ({native_error:#}); external transaction recovery also failed ({recovery_error:#})"
        )),
    }
}

pub(crate) fn recover_pending_external_export_transaction(
    native_output_path: &Path,
) -> anyhow::Result<()> {
    recover_external_export_transaction(native_output_path)
}

pub(crate) fn validate_pending_external_export_source(
    native_output_path: &Path,
    native_source_path: &Path,
) -> anyhow::Result<()> {
    let journal_path = external_export_journal_path(native_output_path)?;
    let bytes = read_regular_file_bounded(&journal_path, MAX_EXTERNAL_EXPORT_JOURNAL_BYTES)?
        .context("native publication requires a durable external export journal")?;
    let journal: ExternalExportTransactionJournal =
        serde_json::from_slice(&bytes).context("external export transaction journal is invalid")?;
    validate_external_export_journal(native_output_path, &journal)?;
    let expected_previous = crate::output_bundle::native_output_sha256(native_output_path)?;
    anyhow::ensure!(
        expected_previous == journal.native_previous_sha256,
        "native output changed before its coupled publication began"
    );
    let source_sha256 = crate::output_bundle::native_output_sha256(native_source_path)?
        .context("staged native output does not exist")?;
    anyhow::ensure!(
        source_sha256 == journal.native_next_sha256,
        "staged native output does not match the external transaction journal"
    );
    Ok(())
}

fn read_staged_export_package(staged_main_path: &Path) -> anyhow::Result<ExportPackage> {
    const MAX_EXPORT_MEMBER_BYTES: u64 = 256 * 1024 * 1024;
    const MAX_EXPORT_PACKAGE_BYTES: u64 = 512 * 1024 * 1024;
    const MAX_EXPORT_PACKAGE_FILES: usize = 50_000;

    let directory = staged_main_path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    let main_name = staged_main_path
        .file_name()
        .context("staged external export path must include a file name")?;
    let mut members = Vec::new();
    let mut total_bytes = 0_u64;
    for entry in std::fs::read_dir(directory).with_context(|| {
        format!(
            "failed to read staged export directory '{}'",
            directory.display()
        )
    })? {
        let entry = entry?;
        let file_type = entry.file_type()?;
        anyhow::ensure!(
            file_type.is_file(),
            "staged export package may contain files only"
        );
        let name = entry.file_name();
        let name = name
            .to_str()
            .context("staged export member name is not valid UTF-8")?;
        let metadata = entry.metadata()?;
        anyhow::ensure!(
            metadata.len() <= MAX_EXPORT_MEMBER_BYTES,
            "staged export member '{}' exceeds the size limit",
            name
        );
        let file = std::fs::File::open(entry.path())?;
        let opened_metadata = file.metadata()?;
        anyhow::ensure!(
            opened_metadata.is_file() && opened_metadata.len() == metadata.len(),
            "staged export member '{}' changed while being opened",
            name
        );
        let mut bytes = Vec::with_capacity(metadata.len() as usize);
        file.take(MAX_EXPORT_MEMBER_BYTES + 1)
            .read_to_end(&mut bytes)?;
        anyhow::ensure!(
            bytes.len() as u64 == metadata.len() && bytes.len() as u64 <= MAX_EXPORT_MEMBER_BYTES,
            "staged export member '{}' changed size while being read",
            name
        );
        total_bytes = total_bytes.saturating_add(bytes.len() as u64);
        anyhow::ensure!(
            total_bytes <= MAX_EXPORT_PACKAGE_BYTES,
            "staged export package exceeds the total size limit"
        );
        members.push(ExportPackageMember::new(name, bytes)?);
        anyhow::ensure!(
            members.len() <= MAX_EXPORT_PACKAGE_FILES,
            "staged export package contains too many files"
        );
    }
    anyhow::ensure!(
        members
            .iter()
            .any(|member| member.relative_path == Path::new(main_name)),
        "staged external export '{}' is missing",
        staged_main_path.display()
    );
    ExportPackage::new(members)
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
    let _rollback = persist_export_package(&ExportPackage::new(members)?, destination_dir, None)?;
    Ok(graph)
}

fn load_convolution_resources(
    graph: &DspGraph,
    source_dir: &Path,
) -> anyhow::Result<Vec<ConvolutionResource>> {
    const MAX_RESOURCE_BYTES: u64 = 256 * 1024 * 1024;
    const MAX_RESOURCE_SET_BYTES: u64 = 512 * 1024 * 1024;

    let mut resources = Vec::new();
    let mut total_bytes = 0_u64;
    for reference in checked_convolution_resource_references(graph)? {
        let path = Path::new(&reference);
        let path = if path.is_absolute() {
            path.to_path_buf()
        } else {
            source_dir.join(path)
        };
        let metadata = std::fs::symlink_metadata(&path).with_context(|| {
            format!(
                "convolution resource '{}' was not found at '{}'",
                reference,
                path.display()
            )
        })?;
        anyhow::ensure!(
            metadata.is_file() && !metadata.file_type().is_symlink(),
            "convolution resource '{}' is not a regular file",
            path.display()
        );
        anyhow::ensure!(
            metadata.len() <= MAX_RESOURCE_BYTES,
            "convolution resource '{}' exceeds the size limit",
            path.display()
        );
        let file = std::fs::File::open(&path)?;
        let opened_metadata = file.metadata()?;
        anyhow::ensure!(
            opened_metadata.is_file() && opened_metadata.len() == metadata.len(),
            "convolution resource '{}' changed while being opened",
            path.display()
        );
        let mut bytes = Vec::with_capacity(metadata.len() as usize);
        file.take(MAX_RESOURCE_BYTES + 1).read_to_end(&mut bytes)?;
        anyhow::ensure!(
            bytes.len() as u64 == metadata.len() && bytes.len() as u64 <= MAX_RESOURCE_BYTES,
            "convolution resource '{}' changed size while being read",
            path.display()
        );
        total_bytes = total_bytes.saturating_add(bytes.len() as u64);
        anyhow::ensure!(
            total_bytes <= MAX_RESOURCE_SET_BYTES,
            "convolution resource set exceeds the total size limit"
        );
        resources.push(ConvolutionResource {
            reference,
            bytes: Arc::from(bytes),
        });
    }
    Ok(resources)
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
        log::info!(
            "CamillaDSP fractional-delay realization adds {} common samples ({:.6} ms); delay usable band 0..{} Hz. This excludes requested delays, existing FIR latency and device buffers.",
            report.common_padding_samples,
            report.additional_latency_ms,
            report.usable_band_upper_hz
        );
    }
}

#[derive(Debug)]
struct ExportPackageRollback {
    main_path: Option<PathBuf>,
    previous_main: Option<Vec<u8>>,
    published_main: Option<Vec<u8>>,
    created_sidecars: Vec<(PathBuf, Vec<u8>)>,
}

struct PlannedExportSidecar<'a> {
    member: &'a ExportPackageMember,
    path: PathBuf,
    existed_identical: bool,
}

struct ExportPublicationPlan<'a> {
    main_member: Option<(&'a ExportPackageMember, PathBuf, Option<Vec<u8>>)>,
    sidecars: Vec<PlannedExportSidecar<'a>>,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ExternalExportTransactionJournal {
    schema_version: u32,
    native_output_name: String,
    native_previous_sha256: Option<String>,
    native_next_sha256: String,
    transaction_directory: String,
    path_platform: String,
    external_directory: Vec<u8>,
    main_member_path: String,
    previous_main_sha256: Option<String>,
    previous_main_file: Option<String>,
    members: Vec<ExternalExportJournalMember>,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ExternalExportJournalMember {
    relative_path: String,
    sha256: String,
    size_bytes: u64,
    staged_file: String,
    is_main: bool,
    existed_identical: bool,
    installed_by_transaction: bool,
}

fn external_export_journal_path(native_output_path: &Path) -> anyhow::Result<PathBuf> {
    let name = native_output_path
        .file_name()
        .context("native output path must include a file name")?;
    let mut journal_name = std::ffi::OsString::from(".");
    journal_name.push(name);
    journal_name.push(".external-export-journal.json");
    Ok(native_output_path.with_file_name(journal_name))
}

fn write_external_export_journal(
    native_output_path: &Path,
    external_directory: &Path,
    native_previous_sha256: &Option<String>,
    native_next_sha256: &str,
    plan: &ExportPublicationPlan<'_>,
) -> anyhow::Result<()> {
    anyhow::ensure!(
        native_previous_sha256.as_deref().is_none_or(is_sha256_hex)
            && is_sha256_hex(native_next_sha256),
        "native transaction identities must be lowercase SHA-256 digests"
    );
    let journal_path = external_export_journal_path(native_output_path)?;
    match std::fs::symlink_metadata(&journal_path) {
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => return Err(error.into()),
        Ok(_) => anyhow::bail!(
            "an unresolved external export journal already exists at '{}'",
            journal_path.display()
        ),
    }
    let external_directory = std::fs::canonicalize(external_directory).with_context(|| {
        format!(
            "failed to resolve external export directory '{}'",
            external_directory.display()
        )
    })?;
    let native_parent = native_output_path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    std::fs::create_dir_all(native_parent)?;
    let transaction = tempfile::Builder::new()
        .prefix(".roomeq-external-export-")
        .tempdir_in(native_parent)?;
    let transaction_path = transaction.path().to_path_buf();
    let next_directory = transaction_path.join("next");
    std::fs::create_dir(&next_directory)?;

    let (main_member_path, previous_main_sha256, previous_main_file) =
        if let Some((member, _, previous)) = plan.main_member.as_ref() {
            let digest = previous
                .as_ref()
                .map(|bytes| autoeq_artifacts::sha256_hex(bytes));
            let backup_name = if let Some(bytes) = previous {
                let backup_path = transaction_path.join("previous-main.bin");
                autoeq_artifacts::write_file_atomically(&backup_path, bytes)?;
                Some("previous-main.bin".to_string())
            } else {
                None
            };
            (
                member.relative_path.to_string_lossy().into_owned(),
                digest,
                backup_name,
            )
        } else {
            anyhow::bail!("external transaction requires exactly one main export member")
        };

    let mut members =
        Vec::with_capacity(plan.sidecars.len() + usize::from(plan.main_member.is_some()));
    let mut index = 0_usize;
    if let Some((member, _, _)) = &plan.main_member {
        let staged_file = format!("next/{index:05}.member");
        autoeq_artifacts::write_file_atomically(
            &transaction_path.join(&staged_file),
            &member.bytes,
        )?;
        members.push(ExternalExportJournalMember {
            relative_path: member.relative_path.to_string_lossy().into_owned(),
            sha256: member.sha256.clone(),
            size_bytes: member.bytes.len() as u64,
            staged_file,
            is_main: true,
            existed_identical: false,
            installed_by_transaction: false,
        });
        index += 1;
    }
    for sidecar in &plan.sidecars {
        let staged_file = format!("next/{index:05}.member");
        autoeq_artifacts::write_file_atomically(
            &transaction_path.join(&staged_file),
            &sidecar.member.bytes,
        )?;
        members.push(ExternalExportJournalMember {
            relative_path: sidecar.member.relative_path.to_string_lossy().into_owned(),
            sha256: sidecar.member.sha256.clone(),
            size_bytes: sidecar.member.bytes.len() as u64,
            staged_file,
            is_main: false,
            existed_identical: sidecar.existed_identical,
            installed_by_transaction: false,
        });
        index += 1;
    }
    let native_output_name = native_output_path
        .file_name()
        .context("native output path must include a file name")?
        .to_str()
        .context("native output file name is not valid UTF-8")?
        .to_string();
    let transaction_directory = transaction_path
        .file_name()
        .context("external transaction directory has no name")?
        .to_str()
        .context("external transaction directory name is not valid UTF-8")?
        .to_string();
    let external_directory = encode_native_path(&external_directory);
    anyhow::ensure!(
        external_directory.len() <= MAX_EXTERNAL_DIRECTORY_PATH_BYTES,
        "external export directory path exceeds the size limit"
    );
    let journal = ExternalExportTransactionJournal {
        schema_version: EXTERNAL_EXPORT_JOURNAL_SCHEMA_VERSION,
        native_output_name,
        native_previous_sha256: native_previous_sha256.clone(),
        native_next_sha256: native_next_sha256.to_string(),
        transaction_directory,
        path_platform: current_path_platform().to_string(),
        external_directory,
        main_member_path,
        previous_main_sha256,
        previous_main_file,
        members,
    };
    let journal_bytes = serde_json::to_vec(&journal)?;
    anyhow::ensure!(
        journal_bytes.len() as u64 <= MAX_EXTERNAL_EXPORT_JOURNAL_BYTES,
        "external export journal exceeds its size limit"
    );
    sync_directory(&next_directory)?;
    sync_directory(&transaction_path)?;
    let transaction_path = transaction.keep();
    if let Err(error) = autoeq_artifacts::write_file_atomically(&journal_path, &journal_bytes) {
        let _ = std::fs::remove_dir_all(&transaction_path);
        return Err(error).context("failed to persist external export journal");
    }
    sync_directory(native_parent).context("failed to flush external export journal directory")?;
    Ok(())
}

fn recover_external_export_transaction(native_output_path: &Path) -> anyhow::Result<()> {
    let journal_path = external_export_journal_path(native_output_path)?;
    let bytes = match read_regular_file_bounded(&journal_path, MAX_EXTERNAL_EXPORT_JOURNAL_BYTES)? {
        Some(bytes) => bytes,
        None => return Ok(()),
    };
    let journal: ExternalExportTransactionJournal =
        serde_json::from_slice(&bytes).context("external export transaction journal is invalid")?;
    validate_external_export_journal(native_output_path, &journal)?;
    let current_native = crate::output_bundle::native_output_sha256(native_output_path)?;
    anyhow::ensure!(
        current_native == journal.native_previous_sha256
            || current_native.as_deref() == Some(journal.native_next_sha256.as_str()),
        "native output changed independently of its external export transaction"
    );
    let forward =
        if journal.native_previous_sha256.as_deref() == Some(journal.native_next_sha256.as_str()) {
            journal
                .members
                .iter()
                .all(|member| member.existed_identical || member.installed_by_transaction)
        } else {
            current_native.as_deref() == Some(journal.native_next_sha256.as_str())
        };
    let native_parent = native_output_path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    let transaction_path = native_parent.join(&journal.transaction_directory);
    let transaction_metadata = std::fs::symlink_metadata(&transaction_path)
        .context("external export transaction staging directory is missing")?;
    anyhow::ensure!(
        transaction_metadata.is_dir() && !transaction_metadata.file_type().is_symlink(),
        "external export transaction staging path is not a real directory"
    );
    let external_directory = decode_native_path(&journal.external_directory)?;
    let canonical_external_directory = std::fs::canonicalize(&external_directory)
        .context("external export destination directory is unavailable")?;
    anyhow::ensure!(
        canonical_external_directory == external_directory,
        "external export destination directory changed since publication"
    );

    let previous_main = match (&journal.previous_main_file, &journal.previous_main_sha256) {
        (Some(name), Some(expected)) => {
            let relative = validate_journal_relative_path(name)?;
            let backup_path = autoeq_artifacts::safe_artifact_path(&transaction_path, &relative)?;
            let bytes = read_regular_file_bounded(&backup_path, MAX_EXPORT_MEMBER_BYTES)?
                .context("external export main backup is missing")?;
            anyhow::ensure!(
                autoeq_artifacts::sha256_hex(&bytes) == *expected,
                "external export main backup failed integrity validation"
            );
            Some(bytes)
        }
        (None, None) => None,
        _ => anyhow::bail!("external export journal has inconsistent main backup metadata"),
    };
    let mut seen = BTreeSet::new();
    let mut seen_staged = BTreeSet::new();
    let mut main_count = 0;
    let mut total_bytes = 0_u64;
    let mut pending = Vec::with_capacity(journal.members.len());
    for member in &journal.members {
        validate_external_member_record(member, &mut seen)?;
        anyhow::ensure!(
            seen_staged.insert(member.staged_file.clone()),
            "external export journal reuses a staged member path"
        );
        total_bytes = total_bytes
            .checked_add(member.size_bytes)
            .context("external export journal size overflow")?;
        anyhow::ensure!(
            total_bytes <= MAX_EXPORT_PACKAGE_BYTES,
            "external export journal exceeds the total member size limit"
        );
        if member.is_main {
            main_count += 1;
            anyhow::ensure!(
                member.relative_path == journal.main_member_path,
                "external export journal main path does not match its member list"
            );
        }
        let staged_relative = validate_journal_relative_path(&member.staged_file)?;
        let staged_path =
            autoeq_artifacts::safe_artifact_path(&transaction_path, &staged_relative)?;
        let staged = read_regular_file_bounded(&staged_path, MAX_EXPORT_MEMBER_BYTES)?
            .context("external export staged member is missing")?;
        anyhow::ensure!(
            staged.len() as u64 == member.size_bytes
                && autoeq_artifacts::sha256_hex(&staged) == member.sha256,
            "external export staged member failed integrity validation"
        );
        let destination_relative = validate_journal_relative_path(&member.relative_path)?;
        let destination =
            autoeq_artifacts::safe_artifact_path(&external_directory, &destination_relative)?;
        ensure_no_symlink_parent(&external_directory, &destination_relative)?;
        pending.push((member, staged, destination));
    }
    anyhow::ensure!(
        main_count == 1,
        "external export journal must contain exactly one main member"
    );
    for (member, staged, destination) in pending {
        recover_external_member(
            &destination,
            &staged,
            member,
            previous_main.as_deref(),
            journal.previous_main_sha256.as_deref(),
            member.installed_by_transaction,
            forward,
        )?;
    }
    remove_external_export_journal_after_recovery(&journal_path, &transaction_path, native_parent)
}

fn mark_external_member_installed(
    native_output_path: &Path,
    relative_path: &Path,
    is_main: bool,
) -> anyhow::Result<()> {
    let journal_path = external_export_journal_path(native_output_path)?;
    let bytes = read_regular_file_bounded(&journal_path, MAX_EXTERNAL_EXPORT_JOURNAL_BYTES)?
        .context("external export journal disappeared before installation was recorded")?;
    let mut journal: ExternalExportTransactionJournal =
        serde_json::from_slice(&bytes).context("external export transaction journal is invalid")?;
    validate_external_export_journal(native_output_path, &journal)?;
    let relative = relative_path.to_string_lossy().into_owned();
    let matching = journal
        .members
        .iter_mut()
        .filter(|member| member.relative_path == relative && member.is_main == is_main)
        .collect::<Vec<_>>();
    anyhow::ensure!(
        matching.len() == 1,
        "external export journal does not contain exactly one matching member"
    );
    let member = matching
        .into_iter()
        .next()
        .expect("checked one matching member");
    anyhow::ensure!(
        !member.existed_identical && !member.installed_by_transaction,
        "external export member installation state is inconsistent"
    );
    member.installed_by_transaction = true;
    let updated = serde_json::to_vec(&journal)?;
    anyhow::ensure!(
        updated.len() as u64 <= MAX_EXTERNAL_EXPORT_JOURNAL_BYTES,
        "external export journal exceeds its size limit"
    );
    autoeq_artifacts::write_file_atomically(&journal_path, &updated)?;
    let native_parent = native_output_path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    sync_directory(native_parent)?;
    Ok(())
}

fn validate_external_export_journal(
    native_output_path: &Path,
    journal: &ExternalExportTransactionJournal,
) -> anyhow::Result<()> {
    anyhow::ensure!(
        journal.schema_version == EXTERNAL_EXPORT_JOURNAL_SCHEMA_VERSION,
        "unsupported external export journal schema {}",
        journal.schema_version
    );
    anyhow::ensure!(
        journal.path_platform == current_path_platform(),
        "external export journal was created on an incompatible path platform"
    );
    let expected_name = native_output_path
        .file_name()
        .and_then(|name| name.to_str())
        .context("native output file name is not valid UTF-8")?;
    anyhow::ensure!(
        journal.native_output_name == expected_name,
        "external export journal belongs to a different native output"
    );
    anyhow::ensure!(
        is_sha256_hex(&journal.native_next_sha256)
            && journal
                .native_previous_sha256
                .as_deref()
                .is_none_or(is_sha256_hex),
        "external export journal has invalid native identities"
    );
    validate_single_component(&journal.transaction_directory)?;
    anyhow::ensure!(
        journal
            .transaction_directory
            .starts_with(".roomeq-external-export-"),
        "external export transaction directory name is invalid"
    );
    anyhow::ensure!(
        journal.members.len() <= MAX_EXPORT_PACKAGE_FILES,
        "external export journal contains too many members"
    );
    anyhow::ensure!(
        !journal.main_member_path.is_empty(),
        "external export journal is missing its main member path"
    );
    let main_path = validate_journal_relative_path(&journal.main_member_path)?;
    anyhow::ensure!(
        main_path.components().count() == 1,
        "external export main member must be a file name"
    );
    match (
        journal.previous_main_file.as_deref(),
        journal.previous_main_sha256.as_deref(),
    ) {
        (Some(name), Some(digest)) => {
            let backup = validate_journal_relative_path(name)?;
            anyhow::ensure!(
                backup.components().count() == 1 && name == "previous-main.bin",
                "external export main backup path is invalid"
            );
            anyhow::ensure!(
                is_sha256_hex(digest),
                "external export main backup digest is invalid"
            );
        }
        (None, None) => {}
        _ => anyhow::bail!("external export journal has inconsistent main backup metadata"),
    }
    Ok(())
}

fn validate_external_member_record(
    member: &ExternalExportJournalMember,
    seen: &mut BTreeSet<String>,
) -> anyhow::Result<()> {
    let relative = validate_journal_relative_path(&member.relative_path)?;
    anyhow::ensure!(
        is_sha256_hex(&member.sha256) && member.size_bytes <= MAX_EXPORT_MEMBER_BYTES,
        "external export journal member has invalid content metadata"
    );
    let staged = validate_journal_relative_path(&member.staged_file)?;
    anyhow::ensure!(
        staged.components().count() == 2
            && staged.parent() == Some(Path::new("next"))
            && staged
                .file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| name.ends_with(".member")),
        "external export journal staged path is invalid"
    );
    anyhow::ensure!(
        !member.existed_identical || !member.installed_by_transaction,
        "pre-existing export sidecar cannot be marked transaction-installed"
    );
    anyhow::ensure!(
        !member.is_main || !member.existed_identical,
        "external export main has invalid prior-member state"
    );
    let folded = relative.to_string_lossy().to_lowercase();
    anyhow::ensure!(
        seen.insert(folded),
        "external export journal has duplicate paths"
    );
    Ok(())
}

fn validate_journal_relative_path(path: &str) -> anyhow::Result<PathBuf> {
    anyhow::ensure!(
        !path.is_empty() && path.len() <= MAX_EXPORT_MEMBER_PATH_BYTES,
        "external export journal path has an invalid length"
    );
    let relative = PathBuf::from(path);
    autoeq_artifacts::validate_relative_artifact_path(&relative)?;
    Ok(relative)
}

fn is_sha256_hex(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn validate_single_component(path: &str) -> anyhow::Result<()> {
    let relative = validate_journal_relative_path(path)?;
    anyhow::ensure!(
        relative.components().count() == 1,
        "external export transaction directory must be one path component"
    );
    Ok(())
}

#[cfg(unix)]
fn current_path_platform() -> &'static str {
    "unix"
}

#[cfg(windows)]
fn current_path_platform() -> &'static str {
    "windows"
}

#[cfg(not(any(unix, windows)))]
fn current_path_platform() -> &'static str {
    "utf8"
}

#[cfg(unix)]
fn encode_native_path(path: &Path) -> Vec<u8> {
    use std::os::unix::ffi::OsStrExt;
    path.as_os_str().as_bytes().to_vec()
}

#[cfg(windows)]
fn encode_native_path(path: &Path) -> Vec<u8> {
    use std::os::windows::ffi::OsStrExt;
    path.as_os_str()
        .encode_wide()
        .flat_map(u16::to_le_bytes)
        .collect()
}

#[cfg(not(any(unix, windows)))]
fn encode_native_path(path: &Path) -> Vec<u8> {
    path.to_string_lossy().as_bytes().to_vec()
}

#[cfg(unix)]
fn decode_native_path(bytes: &[u8]) -> anyhow::Result<PathBuf> {
    use std::os::unix::ffi::OsStringExt;
    anyhow::ensure!(
        bytes.len() <= MAX_EXTERNAL_DIRECTORY_PATH_BYTES,
        "external export directory path exceeds the size limit"
    );
    Ok(PathBuf::from(std::ffi::OsString::from_vec(bytes.to_vec())))
}

#[cfg(windows)]
fn decode_native_path(bytes: &[u8]) -> anyhow::Result<PathBuf> {
    use std::os::windows::ffi::OsStringExt;
    anyhow::ensure!(
        bytes.len() <= MAX_EXTERNAL_DIRECTORY_PATH_BYTES && bytes.len() % 2 == 0,
        "external export directory path has an invalid size"
    );
    let wide = bytes
        .chunks_exact(2)
        .map(|pair| u16::from_le_bytes([pair[0], pair[1]]))
        .collect::<Vec<_>>();
    Ok(PathBuf::from(std::ffi::OsString::from_wide(&wide)))
}

#[cfg(not(any(unix, windows)))]
fn decode_native_path(bytes: &[u8]) -> anyhow::Result<PathBuf> {
    anyhow::ensure!(
        bytes.len() <= MAX_EXTERNAL_DIRECTORY_PATH_BYTES,
        "external export directory path exceeds the size limit"
    );
    let path = String::from_utf8(bytes.to_vec())
        .context("external export directory path is not valid UTF-8")?;
    Ok(PathBuf::from(path))
}

fn recover_external_member(
    destination: &Path,
    staged: &[u8],
    member: &ExternalExportJournalMember,
    previous_main: Option<&[u8]>,
    previous_main_sha256: Option<&str>,
    installed_by_transaction: bool,
    forward: bool,
) -> anyhow::Result<()> {
    let current = read_regular_file_bounded(destination, MAX_EXPORT_MEMBER_BYTES)?;
    let current_digest = current.as_deref().map(autoeq_artifacts::sha256_hex);
    let current_is_new = current_digest.as_deref() == Some(member.sha256.as_str())
        && current
            .as_ref()
            .is_some_and(|bytes| bytes.len() as u64 == member.size_bytes);
    if forward {
        if member.is_main {
            anyhow::ensure!(
                installed_by_transaction,
                "external main installation ownership is ambiguous"
            );
            if current_is_new {
                return Ok(());
            }
            anyhow::ensure!(
                current_digest.as_deref() == previous_main_sha256
                    || (current.is_none() && previous_main_sha256.is_none()),
                "external main changed independently during forward recovery"
            );
            if current.is_none() {
                install_external_member_noclobber(destination, staged)
            } else {
                install_external_member(destination, staged)
            }
        } else if member.existed_identical {
            anyhow::ensure!(
                current_is_new,
                "pre-existing external sidecar changed during forward recovery"
            );
            Ok(())
        } else {
            anyhow::ensure!(
                installed_by_transaction,
                "external sidecar installation ownership is ambiguous"
            );
            if current_is_new {
                return Ok(());
            }
            anyhow::ensure!(
                current.is_none(),
                "external sidecar changed independently during forward recovery"
            );
            install_external_member_noclobber(destination, staged)
        }
    } else if member.is_main {
        if current_digest.as_deref() == previous_main_sha256
            && current.is_some() == previous_main.is_some()
        {
            return Ok(());
        }
        anyhow::ensure!(
            installed_by_transaction && current_is_new,
            "external main changed independently or has ambiguous installation ownership during rollback"
        );
        match previous_main {
            Some(bytes) => install_external_member(destination, bytes),
            None => {
                std::fs::remove_file(destination)?;
                sync_external_parent(destination)
            }
        }
    } else if member.existed_identical {
        anyhow::ensure!(
            current_is_new,
            "pre-existing external sidecar changed independently during rollback"
        );
        Ok(())
    } else if current.is_none() {
        Ok(())
    } else {
        anyhow::ensure!(
            installed_by_transaction && current_is_new,
            "new external sidecar changed independently or has ambiguous installation ownership during rollback"
        );
        std::fs::remove_file(destination)?;
        sync_external_parent(destination)
    }
}

fn install_external_member(destination: &Path, bytes: &[u8]) -> anyhow::Result<()> {
    if let Some(parent) = destination.parent() {
        std::fs::create_dir_all(parent)?;
    }
    autoeq_artifacts::write_file_atomically(destination, bytes)?;
    sync_external_parent(destination)
}

fn install_external_member_noclobber(destination: &Path, bytes: &[u8]) -> anyhow::Result<()> {
    if let Some(parent) = destination.parent() {
        std::fs::create_dir_all(parent)?;
    }
    write_file_noclobber_atomically(destination, bytes)?;
    sync_external_parent(destination)
}

fn sync_external_parent(path: &Path) -> anyhow::Result<()> {
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    sync_directory(parent)?;
    Ok(())
}

fn remove_external_export_journal_after_recovery(
    journal_path: &Path,
    transaction_path: &Path,
    native_parent: &Path,
) -> anyhow::Result<()> {
    std::fs::remove_file(journal_path)?;
    sync_directory(native_parent)?;
    std::fs::remove_dir_all(transaction_path)?;
    sync_directory(native_parent)?;
    Ok(())
}

#[cfg(unix)]
fn sync_directory(path: &Path) -> std::io::Result<()> {
    std::fs::File::open(path)?.sync_all()
}

// std does not expose a portable directory sync handle on Windows. The
// journal and member contents are still flushed; directory-entry durability
// remains best-effort on non-Unix platforms.
#[cfg(not(unix))]
fn sync_directory(_path: &Path) -> std::io::Result<()> {
    Ok(())
}

impl ExportPackageRollback {
    fn restore(self) -> anyhow::Result<()> {
        let mut errors = Vec::new();
        if let Some(main_path) = self.main_path {
            match (self.previous_main, self.published_main) {
                (Some(previous), Some(published)) => match file_matches(&main_path, &published) {
                    Ok(true) => {
                        if let Err(error) =
                            autoeq_artifacts::write_file_atomically(&main_path, &previous)
                        {
                            errors.push(format!(
                                "could not restore '{}': {error}",
                                main_path.display()
                            ));
                        }
                    }
                    Ok(false) => errors.push(format!(
                        "external export '{}' changed before rollback",
                        main_path.display()
                    )),
                    Err(error) => errors.push(error.to_string()),
                },
                (None, Some(published)) => match file_matches(&main_path, &published) {
                    Ok(true) => {
                        if let Err(error) = std::fs::remove_file(&main_path) {
                            errors.push(format!(
                                "could not remove new export '{}': {error}",
                                main_path.display()
                            ));
                        }
                    }
                    Ok(false) => errors.push(format!(
                        "new external export '{}' changed before rollback",
                        main_path.display()
                    )),
                    Err(error) => errors.push(error.to_string()),
                },
                _ => {}
            }
        }
        for (path, bytes) in self.created_sidecars.into_iter().rev() {
            match file_matches(&path, &bytes) {
                Ok(true) => {
                    if let Err(error) = std::fs::remove_file(&path) {
                        errors.push(format!(
                            "could not remove new export sidecar '{}': {error}",
                            path.display()
                        ));
                    }
                }
                Ok(false) => errors.push(format!(
                    "new export sidecar '{}' changed before rollback",
                    path.display()
                )),
                Err(error) => errors.push(error.to_string()),
            }
        }
        if errors.is_empty() {
            Ok(())
        } else {
            anyhow::bail!("{}", errors.join("; "))
        }
    }
}

fn persist_export_package(
    package: &ExportPackage,
    destination_dir: &Path,
    main_member_name: Option<&Path>,
) -> anyhow::Result<ExportPackageRollback> {
    persist_export_package_with_hook(
        package,
        destination_dir,
        main_member_name,
        |_| Ok(()),
        |_, _| Ok(()),
    )
}

fn persist_export_package_with_hook(
    package: &ExportPackage,
    destination_dir: &Path,
    main_member_name: Option<&Path>,
    before_write: impl FnOnce(&ExportPublicationPlan<'_>) -> anyhow::Result<()>,
    mut after_write: impl FnMut(&ExportPackageMember, bool) -> anyhow::Result<()>,
) -> anyhow::Result<ExportPackageRollback> {
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
    let mut main_member = None;
    let mut sidecars = Vec::new();
    let mut total_bytes = 0_u64;
    anyhow::ensure!(
        package.members.len() <= MAX_EXPORT_PACKAGE_FILES,
        "export package contains too many files"
    );
    for member in &package.members {
        let relative = Path::new(&member.relative_path);
        anyhow::ensure!(
            member.relative_path.to_string_lossy().len() <= MAX_EXPORT_MEMBER_PATH_BYTES,
            "export package member path exceeds the size limit"
        );
        let path = autoeq_artifacts::safe_artifact_path(destination_dir, relative)?;
        ensure_no_symlink_parent(destination_dir, relative)?;
        anyhow::ensure!(
            member.bytes.len() as u64 <= MAX_EXPORT_MEMBER_BYTES,
            "export package member '{}' exceeds the size limit",
            member.relative_path.display()
        );
        total_bytes = total_bytes
            .checked_add(member.bytes.len() as u64)
            .context("export package size overflow")?;
        anyhow::ensure!(
            total_bytes <= MAX_EXPORT_PACKAGE_BYTES,
            "export package exceeds the total size limit"
        );
        if main_member_name == Some(relative) {
            main_member = Some((member, path));
        } else {
            sidecars.push((member, path));
        }
    }
    if let Some(main_name) = main_member_name {
        anyhow::ensure!(
            main_member.is_some(),
            "export package is missing main member '{}'",
            main_name.display()
        );
    }

    let previous_main = if let Some((_, path)) = &main_member {
        read_regular_file_bounded(path, MAX_EXPORT_MEMBER_BYTES)?
    } else {
        None
    };
    let mut planned_sidecars = Vec::with_capacity(sidecars.len());
    for (member, path) in sidecars {
        match read_regular_file_bounded(&path, MAX_EXPORT_MEMBER_BYTES)? {
            Some(existing) if existing == member.bytes.as_ref() => {
                planned_sidecars.push(PlannedExportSidecar {
                    member,
                    path,
                    existed_identical: true,
                });
            }
            Some(_) => anyhow::bail!(
                "export package sidecar '{}' already exists with different bytes",
                path.display()
            ),
            None => planned_sidecars.push(PlannedExportSidecar {
                member,
                path,
                existed_identical: false,
            }),
        }
    }
    for sidecar in &planned_sidecars {
        let member = sidecar.member;
        let path = &sidecar.path;
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).with_context(|| {
                format!(
                    "failed to create export package directory for '{}'",
                    member.relative_path.display()
                )
            })?;
        }
    }
    if let Some((member, path)) = &main_member
        && let Some(parent) = path.parent()
    {
        std::fs::create_dir_all(parent).with_context(|| {
            format!(
                "failed to create export package directory for '{}'",
                member.relative_path.display()
            )
        })?;
    }

    let plan = ExportPublicationPlan {
        main_member: main_member
            .as_ref()
            .map(|(member, path)| (*member, path.clone(), previous_main.clone())),
        sidecars: planned_sidecars,
    };
    before_write(&plan)?;

    let mut rollback = ExportPackageRollback {
        main_path: main_member.as_ref().map(|(_, path)| path.clone()),
        previous_main: previous_main.clone(),
        published_main: None,
        created_sidecars: Vec::new(),
    };
    for sidecar in plan.sidecars {
        let member = sidecar.member;
        let path = sidecar.path;
        let current = match read_regular_file_bounded(&path, MAX_EXPORT_MEMBER_BYTES) {
            Ok(current) => current,
            Err(error) => {
                return abort_export_publication(
                    rollback,
                    error,
                    "failed to recheck export sidecar before publication",
                );
            }
        };
        if sidecar.existed_identical {
            if current.as_deref() != Some(member.bytes.as_ref()) {
                return abort_export_publication(
                    rollback,
                    anyhow::anyhow!(
                        "pre-existing export sidecar '{}' changed before publication",
                        path.display()
                    ),
                    "export sidecar preflight failed",
                );
            }
            continue;
        }
        if current.is_some() {
            return abort_export_publication(
                rollback,
                anyhow::anyhow!(
                    "export sidecar '{}' appeared before publication",
                    path.display()
                ),
                "export sidecar preflight failed",
            );
        }
        if let Err(error) = write_file_noclobber_atomically(&path, &member.bytes) {
            return match rollback.restore() {
                Ok(()) => Err(error).with_context(|| {
                    format!("failed to persist export sidecar '{}'", path.display())
                }),
                Err(rollback_error) => anyhow::bail!(
                    "failed to persist export sidecar '{}' ({error}); rollback failed ({rollback_error})",
                    path.display()
                ),
            };
        }
        rollback
            .created_sidecars
            .push((path.clone(), member.bytes.to_vec()));
        if let Err(error) = sync_external_parent(&path) {
            return match rollback.restore() {
                Ok(()) => Err(error).context("failed to flush external export sidecar directory"),
                Err(rollback_error) => anyhow::bail!(
                    "failed to flush external export sidecar directory ({error}); rollback failed ({rollback_error:#})"
                ),
            };
        }
        if let Err(error) = after_write(member, false) {
            return abort_export_publication(
                rollback,
                error,
                "failed to record external export sidecar installation",
            );
        }
    }
    if let Some((member, path)) = main_member {
        let current = match read_regular_file_bounded(&path, MAX_EXPORT_MEMBER_BYTES) {
            Ok(current) => current,
            Err(error) => {
                return abort_export_publication(
                    rollback,
                    error,
                    "failed to recheck external export main before publication",
                );
            }
        };
        if current != previous_main {
            return abort_export_publication(
                rollback,
                anyhow::anyhow!(
                    "external export main '{}' changed before publication",
                    path.display()
                ),
                "external export main preflight failed",
            );
        }
        if let Err(error) = autoeq_artifacts::write_file_atomically(&path, &member.bytes) {
            return match rollback.restore() {
                Ok(()) => Err(error)
                    .with_context(|| format!("failed to persist export main '{}'", path.display())),
                Err(rollback_error) => anyhow::bail!(
                    "failed to persist export main '{}' ({error}); rollback failed ({rollback_error})",
                    path.display()
                ),
            };
        }
        rollback.published_main = Some(member.bytes.to_vec());
        if let Err(error) = sync_external_parent(&path) {
            return match rollback.restore() {
                Ok(()) => Err(error).context("failed to flush external export main directory"),
                Err(rollback_error) => anyhow::bail!(
                    "failed to flush external export main directory ({error}); rollback failed ({rollback_error:#})"
                ),
            };
        }
        if let Err(error) = after_write(member, true) {
            return abort_export_publication(
                rollback,
                error,
                "failed to record external export main installation",
            );
        }
    }
    Ok(rollback)
}

fn abort_export_publication<T>(
    rollback: ExportPackageRollback,
    error: anyhow::Error,
    context: &str,
) -> anyhow::Result<T> {
    match rollback.restore() {
        Ok(()) => Err(error).context(context.to_string()),
        Err(rollback_error) => Err(anyhow::anyhow!(
            "{context} ({error:#}); rollback also failed ({rollback_error:#})"
        )),
    }
}

fn read_regular_file_bounded(path: &Path, maximum_bytes: u64) -> anyhow::Result<Option<Vec<u8>>> {
    let metadata = match std::fs::symlink_metadata(path) {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(error.into()),
    };
    anyhow::ensure!(
        metadata.file_type().is_file(),
        "export destination '{}' is not a regular file",
        path.display()
    );
    anyhow::ensure!(
        metadata.len() <= maximum_bytes,
        "export destination '{}' exceeds the size limit",
        path.display()
    );
    let file = std::fs::File::open(path)?;
    let mut bytes = Vec::with_capacity(metadata.len() as usize);
    file.take(maximum_bytes + 1).read_to_end(&mut bytes)?;
    anyhow::ensure!(
        bytes.len() as u64 == metadata.len() && bytes.len() as u64 <= maximum_bytes,
        "export destination '{}' changed size while being read",
        path.display()
    );
    Ok(Some(bytes))
}

fn write_file_noclobber_atomically(path: &Path, contents: &[u8]) -> std::io::Result<()> {
    let parent = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    let mut temporary = tempfile::NamedTempFile::new_in(parent)?;
    temporary.write_all(contents)?;
    temporary.as_file().sync_all()?;
    temporary
        .persist_noclobber(path)
        .map(|_| ())
        .map_err(|error| error.error)
}

fn file_matches(path: &Path, expected: &[u8]) -> anyhow::Result<bool> {
    Ok(read_regular_file_bounded(path, 256 * 1024 * 1024)?.is_some_and(|bytes| bytes == expected))
}

fn ensure_no_symlink_parent(root: &Path, relative: &Path) -> anyhow::Result<()> {
    let Some(parent) = relative.parent() else {
        return Ok(());
    };
    let mut current = root.to_path_buf();
    for component in parent.components() {
        current.push(component.as_os_str());
        match std::fs::symlink_metadata(&current) {
            Ok(metadata) => anyhow::ensure!(
                metadata.is_dir() && !metadata.file_type().is_symlink(),
                "export package parent '{}' is not a real directory",
                current.display()
            ),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => break,
            Err(error) => return Err(error.into()),
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn interrupted_external_transaction_fixture()
    -> (tempfile::TempDir, tempfile::TempDir, PathBuf, ExportPackage) {
        let native_dir = tempfile::tempdir().unwrap();
        let external_dir = tempfile::tempdir().unwrap();
        let native_output = native_dir.path().join("dsp.json");
        std::fs::write(&native_output, b"prior native graph").unwrap();
        std::fs::write(external_dir.path().join("room.yml"), b"prior config").unwrap();
        let package = ExportPackage::new(vec![
            ExportPackageMember::new("room.yml", b"new config".to_vec()).unwrap(),
            ExportPackageMember::new("impulse.wav", b"new impulse".to_vec()).unwrap(),
        ])
        .unwrap();
        let previous_native_sha256 = Some(autoeq_artifacts::sha256_hex(b"prior native graph"));
        let next_native_sha256 = autoeq_artifacts::sha256_hex(b"new native graph");
        let result = persist_export_package_with_hook(
            &package,
            external_dir.path(),
            Some(Path::new("room.yml")),
            |plan| {
                write_external_export_journal(
                    &native_output,
                    external_dir.path(),
                    &previous_native_sha256,
                    &next_native_sha256,
                    plan,
                )?;
                anyhow::bail!("simulated interruption after durable intent")
            },
            |_, _| Ok(()),
        );
        assert!(result.is_err());
        (native_dir, external_dir, native_output, package)
    }

    fn apply_journal_members(native_output: &Path, external_directory: &Path, only_sidecars: bool) {
        let journal_path = external_export_journal_path(native_output).unwrap();
        let journal_bytes = std::fs::read(journal_path).unwrap();
        let journal: ExternalExportTransactionJournal =
            serde_json::from_slice(&journal_bytes).unwrap();
        let native_parent = native_output.parent().unwrap();
        let transaction = native_parent.join(journal.transaction_directory);
        for member in &journal.members {
            if only_sidecars && member.is_main {
                continue;
            }
            if member.existed_identical || member.installed_by_transaction {
                continue;
            }
            let source = transaction.join(&member.staged_file);
            let destination = external_directory.join(&member.relative_path);
            let bytes = std::fs::read(source).unwrap();
            install_external_member(&destination, &bytes).unwrap();
            mark_external_member_installed(
                native_output,
                Path::new(&member.relative_path),
                member.is_main,
            )
            .unwrap();
        }
    }

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
            assert!(
                bind_final_convolution_artifacts(&mut result, source.path(), &store, 48_000.0)
                    .is_err(),
                "malformed convolution declaration was omitted from final evidence"
            );
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
                    .is_err(),
                    "invalid unretained resource was bound at {rate} Hz"
                );
                assert!(result.metadata.final_convolution_sha256.is_none());
            }
            let mut result = crate::test_fixtures::single_channel_room_result("left");
            result.channels = convolution_graph("impulse.wav").channels;
            result.channel_results.get_mut("left").unwrap().fir_coeffs = None;
            let bytes = mono_fir_bytes(&[0.5, -0.25], rate);
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
        store
            .write(&source.path().join(reference), &original)
            .unwrap();
        let mut result = crate::test_fixtures::single_channel_room_result("left");
        result.channels = convolution_graph(reference).channels;
        bind_final_convolution_artifacts(&mut result, source.path(), &store, 48_000.0).unwrap();
        let graph: DspGraph =
            serde_json::from_slice(&serde_json::to_vec(&result.to_dsp_chain_output()).unwrap())
                .unwrap();
        std::fs::write(source.path().join(reference), b"replacement").unwrap();
        let target = destination.path().join("room.yml");
        let error = export_dsp_chain_with_convolution_sidecars(
            &graph,
            ExportFormat::CamillaDsp,
            &target,
            48_000.0,
            source.path(),
        )
        .unwrap_err();
        assert!(
            error
                .to_string()
                .contains("changed since workflow completion"),
            "{error}"
        );
        assert!(!target.exists());
        std::fs::write(source.path().join(reference), original).unwrap();
        std::fs::write(destination.path().join(reference), b"occupied").unwrap();
        let packaged =
            package_convolution_sidecars(&graph, source.path(), destination.path()).unwrap();
        assert_eq!(convolution_reference(&packaged), "impulse_002.wav");
        assert!(
            packaged
                .metadata
                .as_ref()
                .unwrap()
                .final_convolution_sha256
                .as_ref()
                .unwrap()
                .contains_key("impulse_002.wav")
        );
        export_dsp_chain_with_convolution_sidecars(
            &packaged,
            ExportFormat::CamillaDsp,
            &target,
            48_000.0,
            destination.path(),
        )
        .unwrap();
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
        assert_eq!(
            result.metadata.final_convolution_sha256.as_ref().unwrap()["missing.wav"],
            None
        );
        std::fs::write(source.path().join("missing.wav"), b"newly appeared").unwrap();
        let error = export_dsp_chain_with_convolution_sidecars(
            &result.to_dsp_chain_output(),
            ExportFormat::CamillaDsp,
            &destination.path().join("room.yml"),
            48_000.0,
            source.path(),
        )
        .unwrap_err();
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
        ])
        .unwrap();
        // Keep the recorded identity, but replace a later member's actual bytes.
        package.members[1].bytes = b"stale or replaced impulse".to_vec().into();
        let error = persist_export_package(&package, destination.path(), None).unwrap_err();
        assert!(
            error.to_string().contains("content hash mismatch"),
            "{error}"
        );
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
        let package = ExportPackage {
            members: vec![member],
        };
        assert!(persist_export_package(&package, &destination, None).is_err());
        assert!(!destination.exists());
        assert!(!directory.path().join("escaped.wav").exists());
    }

    #[test]
    fn staged_export_is_rolled_back_when_native_publication_fails() {
        use roomeq_export::ExportPackageMember;

        let parent = tempfile::tempdir().unwrap();
        let staging = tempfile::tempdir_in(parent.path()).unwrap();
        let destination = parent.path().join("deployed");
        std::fs::create_dir_all(&destination).unwrap();
        let staged_main = staging.path().join("room.yml");
        let staged_package = ExportPackage::new(vec![
            ExportPackageMember::new("room.yml", b"new config".to_vec()).unwrap(),
            ExportPackageMember::new("impulse_002.wav", b"new impulse".to_vec()).unwrap(),
        ])
        .unwrap();
        let _stage_rollback =
            persist_export_package(&staged_package, staging.path(), Some(Path::new("room.yml")))
                .unwrap();
        let destination_main = destination.join("room.yml");
        std::fs::write(&destination_main, b"prior config").unwrap();

        let error = publish_staged_export_package_with(&staged_main, &destination_main, || {
            anyhow::bail!("native bundle commit failed")
        })
        .unwrap_err();

        assert!(
            error.to_string().contains("external export was restored"),
            "{error:#}"
        );
        assert_eq!(std::fs::read(destination_main).unwrap(), b"prior config");
        assert!(!destination.join("impulse_002.wav").exists());
    }

    #[test]
    fn external_transaction_recovers_old_root_at_each_publication_boundary() {
        for installed_stage in [0, 1, 2] {
            let (_native_dir, external_dir, native_output, _package) =
                interrupted_external_transaction_fixture();
            if installed_stage >= 1 {
                apply_journal_members(&native_output, external_dir.path(), true);
            }
            if installed_stage >= 2 {
                apply_journal_members(&native_output, external_dir.path(), false);
            }

            recover_pending_external_export_transaction(&native_output).unwrap();

            assert_eq!(
                std::fs::read(external_dir.path().join("room.yml")).unwrap(),
                b"prior config"
            );
            assert!(!external_dir.path().join("impulse.wav").exists());
            assert!(
                !external_export_journal_path(&native_output)
                    .unwrap()
                    .exists()
            );
        }
    }

    #[test]
    fn native_bundle_export_publication_commits_or_restores_as_one_recoverable_unit() {
        use roomeq_export::ExportPackageMember;

        let native_directory = tempfile::tempdir().unwrap();
        let external_directory = tempfile::tempdir().unwrap();
        let staging_directory = tempfile::tempdir().unwrap();
        let native_output = native_directory.path().join("dsp.json");
        let native_source = native_directory.path().join("candidate.json");
        let destination_main = external_directory.path().join("room.yml");
        let staged_main = staging_directory.path().join("room.yml");
        std::fs::write(&native_output, b"prior native graph").unwrap();
        std::fs::write(&native_source, b"next native graph").unwrap();
        std::fs::write(&destination_main, b"prior config").unwrap();
        let package = ExportPackage::new(vec![
            ExportPackageMember::new("room.yml", b"next config".to_vec()).unwrap(),
            ExportPackageMember::new("impulse.wav", b"next impulse".to_vec()).unwrap(),
        ])
        .unwrap();
        let _stage_rollback = persist_export_package(
            &package,
            staging_directory.path(),
            Some(Path::new("room.yml")),
        )
        .unwrap();

        let error = publish_staged_export_package_with_native_bundle(
            &staged_main,
            &destination_main,
            &native_output,
            &native_source,
            || anyhow::bail!("simulated native publication failure"),
        )
        .unwrap_err();
        assert!(
            error
                .to_string()
                .contains("recovered against the resulting native root"),
            "{error:#}"
        );
        assert_eq!(
            std::fs::read(&native_output).unwrap(),
            b"prior native graph"
        );
        assert_eq!(std::fs::read(&destination_main).unwrap(), b"prior config");
        assert!(!external_directory.path().join("impulse.wav").exists());

        publish_staged_export_package_with_native_bundle(
            &staged_main,
            &destination_main,
            &native_output,
            &native_source,
            || {
                autoeq_artifacts::write_file_atomically(&native_output, b"next native graph")?;
                Ok(())
            },
        )
        .unwrap();
        assert_eq!(std::fs::read(&destination_main).unwrap(), b"next config");
        assert_eq!(
            std::fs::read(external_directory.path().join("impulse.wav")).unwrap(),
            b"next impulse"
        );
        assert_eq!(std::fs::read(&native_output).unwrap(), b"next native graph");
        assert!(
            !external_export_journal_path(&native_output)
                .unwrap()
                .exists()
        );
    }

    #[test]
    fn coupled_native_publication_rejects_a_changed_staged_graph() {
        let native_directory = tempfile::tempdir().unwrap();
        let external_directory = tempfile::tempdir().unwrap();
        let staging_directory = tempfile::tempdir().unwrap();
        let native_output = native_directory.path().join("dsp.json");
        let native_source = native_directory.path().join("candidate.json");
        let destination_main = external_directory.path().join("room.yml");
        let staged_main = staging_directory.path().join("room.yml");
        std::fs::write(&native_output, b"prior native graph").unwrap();
        std::fs::write(&native_source, b"next native graph").unwrap();
        std::fs::write(&destination_main, b"prior config").unwrap();
        let package = ExportPackage::new(vec![
            ExportPackageMember::new("room.yml", b"next config".to_vec()).unwrap(),
            ExportPackageMember::new("impulse.wav", b"next impulse".to_vec()).unwrap(),
        ])
        .unwrap();
        let _stage_rollback = persist_export_package(
            &package,
            staging_directory.path(),
            Some(Path::new("room.yml")),
        )
        .unwrap();

        let error = publish_staged_export_package_with_native_bundle(
            &staged_main,
            &destination_main,
            &native_output,
            &native_source,
            || {
                std::fs::write(&native_source, b"changed after external intent")?;
                crate::output_bundle::publish_output_bundle_from_during_external_transaction_with_source_recovery(
                    &native_source,
                    &native_output,
                )
                .map_err(|error| anyhow::anyhow!(error.to_string()))?;
                Ok(())
            },
        )
        .unwrap_err();

        assert!(
            format!("{error:#}").contains("does not match the external transaction journal"),
            "{error:#}"
        );
        assert_eq!(
            std::fs::read(&native_output).unwrap(),
            b"prior native graph"
        );
        assert_eq!(std::fs::read(&destination_main).unwrap(), b"prior config");
        assert!(!external_directory.path().join("impulse.wav").exists());
        assert!(
            !external_export_journal_path(&native_output)
                .unwrap()
                .exists()
        );
    }

    #[test]
    fn external_transaction_finishes_when_new_native_root_was_committed() {
        let (native_dir, external_dir, native_output, _package) =
            interrupted_external_transaction_fixture();
        apply_journal_members(&native_output, external_dir.path(), false);
        std::fs::write(&native_output, b"new native graph").unwrap();

        recover_pending_external_export_transaction(&native_output).unwrap();

        assert_eq!(
            std::fs::read(external_dir.path().join("room.yml")).unwrap(),
            b"new config"
        );
        assert_eq!(
            std::fs::read(external_dir.path().join("impulse.wav")).unwrap(),
            b"new impulse"
        );
        assert!(
            !external_export_journal_path(&native_output)
                .unwrap()
                .exists()
        );
        drop(native_dir);
    }

    #[test]
    fn external_transaction_fails_closed_on_unknown_root_or_external_bytes() {
        let (_native_dir, external_dir, native_output, _package) =
            interrupted_external_transaction_fixture();
        std::fs::write(&native_output, b"concurrent native graph").unwrap();
        let error = recover_pending_external_export_transaction(&native_output).unwrap_err();
        assert!(
            error.to_string().contains("changed independently"),
            "{error:#}"
        );
        assert!(
            external_export_journal_path(&native_output)
                .unwrap()
                .exists()
        );
        assert_eq!(
            std::fs::read(external_dir.path().join("room.yml")).unwrap(),
            b"prior config"
        );

        let (_native_dir, external_dir, native_output, _package) =
            interrupted_external_transaction_fixture();
        apply_journal_members(&native_output, external_dir.path(), false);
        std::fs::write(external_dir.path().join("room.yml"), b"concurrent export").unwrap();
        let error = recover_pending_external_export_transaction(&native_output).unwrap_err();
        assert!(
            error.to_string().contains("changed independently"),
            "{error:#}"
        );
        assert_eq!(
            std::fs::read(external_dir.path().join("room.yml")).unwrap(),
            b"concurrent export"
        );
    }

    #[test]
    fn equal_previous_and_next_native_hashes_still_reject_an_independent_root() {
        let (_native_dir, external_dir, native_output, _package) =
            interrupted_external_transaction_fixture();
        let journal_path = external_export_journal_path(&native_output).unwrap();
        let bytes = std::fs::read(&journal_path).unwrap();
        let mut journal: ExternalExportTransactionJournal = serde_json::from_slice(&bytes).unwrap();
        journal.native_previous_sha256 = Some(journal.native_next_sha256.clone());
        std::fs::write(&journal_path, serde_json::to_vec(&journal).unwrap()).unwrap();
        std::fs::write(&native_output, b"independent native root").unwrap();

        let error = recover_pending_external_export_transaction(&native_output).unwrap_err();

        assert!(
            error.to_string().contains("changed independently"),
            "{error:#}"
        );
        assert!(journal_path.exists());
        assert_eq!(
            std::fs::read(external_dir.path().join("room.yml")).unwrap(),
            b"prior config"
        );
    }

    #[test]
    fn external_transaction_journal_rejects_unknown_fields() {
        let (_native_dir, external_dir, native_output, _package) =
            interrupted_external_transaction_fixture();
        let journal_path = external_export_journal_path(&native_output).unwrap();
        let bytes = std::fs::read(&journal_path).unwrap();
        let mut value: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        value
            .as_object_mut()
            .unwrap()
            .insert("future_field".to_string(), serde_json::json!(true));
        std::fs::write(&journal_path, serde_json::to_vec(&value).unwrap()).unwrap();

        let error = recover_pending_external_export_transaction(&native_output).unwrap_err();

        assert!(
            error.to_string().contains("journal is invalid"),
            "{error:#}"
        );
        assert!(journal_path.exists());
        assert_eq!(
            std::fs::read(external_dir.path().join("room.yml")).unwrap(),
            b"prior config"
        );

        let member = serde_json::json!({
            "relative_path": "impulse.wav",
            "sha256": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            "size_bytes": 0,
            "staged_file": "next/00001.member",
            "is_main": false,
            "existed_identical": false,
            "installed_by_transaction": false,
            "future_member_field": true
        });
        assert!(
            serde_json::from_value::<ExternalExportJournalMember>(member).is_err(),
            "unknown member metadata must be rejected"
        );
    }

    #[test]
    fn external_transaction_validates_previous_main_backup_before_restoring() {
        let (_native_dir, external_dir, native_output, _package) =
            interrupted_external_transaction_fixture();
        apply_journal_members(&native_output, external_dir.path(), false);
        let journal_bytes =
            std::fs::read(external_export_journal_path(&native_output).unwrap()).unwrap();
        let journal: ExternalExportTransactionJournal =
            serde_json::from_slice(&journal_bytes).unwrap();
        let backup = native_output
            .parent()
            .unwrap()
            .join(journal.transaction_directory)
            .join(journal.previous_main_file.unwrap());
        std::fs::write(&backup, b"corrupted prior config").unwrap();

        let error = recover_pending_external_export_transaction(&native_output).unwrap_err();

        assert!(
            error
                .to_string()
                .contains("backup failed integrity validation"),
            "{error:#}"
        );
        assert_eq!(
            std::fs::read(external_dir.path().join("room.yml")).unwrap(),
            b"new config"
        );
        assert!(
            external_export_journal_path(&native_output)
                .unwrap()
                .exists()
        );
    }

    #[test]
    fn external_transaction_preserves_member_when_install_marker_is_ambiguous() {
        let (_native_dir, external_dir, native_output, _package) =
            interrupted_external_transaction_fixture();
        let journal_path = external_export_journal_path(&native_output).unwrap();
        let journal_bytes = std::fs::read(&journal_path).unwrap();
        let journal: ExternalExportTransactionJournal =
            serde_json::from_slice(&journal_bytes).unwrap();
        let member = journal
            .members
            .iter()
            .find(|member| !member.is_main)
            .unwrap();
        let transaction = native_output
            .parent()
            .unwrap()
            .join(&journal.transaction_directory);
        let staged = std::fs::read(transaction.join(&member.staged_file)).unwrap();
        install_external_member(&external_dir.path().join(&member.relative_path), &staged).unwrap();

        let error = recover_pending_external_export_transaction(&native_output).unwrap_err();

        assert!(
            error
                .to_string()
                .contains("ambiguous installation ownership"),
            "{error:#}"
        );
        assert_eq!(
            std::fs::read(external_dir.path().join("impulse.wav")).unwrap(),
            b"new impulse"
        );
        assert!(journal_path.exists());
    }

    #[test]
    fn external_export_sidecar_conflict_preserves_prior_main() {
        use roomeq_export::ExportPackageMember;

        let parent = tempfile::tempdir().unwrap();
        let staging = tempfile::tempdir_in(parent.path()).unwrap();
        let destination = parent.path().join("deployed");
        std::fs::create_dir_all(&destination).unwrap();
        let package = ExportPackage::new(vec![
            ExportPackageMember::new("room.yml", b"new config".to_vec()).unwrap(),
            ExportPackageMember::new("impulse.wav", b"new impulse".to_vec()).unwrap(),
        ])
        .unwrap();
        let _stage_rollback =
            persist_export_package(&package, staging.path(), Some(Path::new("room.yml"))).unwrap();
        let destination_main = destination.join("room.yml");
        std::fs::write(&destination_main, b"prior config").unwrap();
        std::fs::write(destination.join("impulse.wav"), b"unrelated bytes").unwrap();

        let error = publish_staged_export_package_with(
            &staging.path().join("room.yml"),
            &destination_main,
            || panic!("native commit must not run after sidecar preflight fails"),
        )
        .unwrap_err();

        assert!(error.to_string().contains("different bytes"), "{error:#}");
        assert_eq!(std::fs::read(destination_main).unwrap(), b"prior config");
        assert_eq!(
            std::fs::read(destination.join("impulse.wav")).unwrap(),
            b"unrelated bytes"
        );
    }
    use roomeq_model::{ChannelDspChain, PluginConfigWrapper, default_config_version};
    use serde_json::json;
    use std::collections::HashMap;

    fn graph_with_plugins(plugins: Vec<PluginConfigWrapper>) -> DspGraph {
        DspGraph {
            version: default_config_version(),
            artifact_bundle_schema_version: None,
            deployed_source_curves: Default::default(),
            global_plugins: Vec::new(),
            channels: HashMap::from([(
                "left".to_string(),
                ChannelDspChain {
                    physical_correction_target: None,
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
                    joint_sub: None,
                    early_reflections: None,
                    t60_octaves: None,
                    speech_transmission: None,
                    waterfall: None,
                    resonance_decays: None,
                    wavelet: None,
                    early_late_curves: None,
                },
            )]),
            metadata: None,
            correction_decisions: None,
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
