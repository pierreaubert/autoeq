//! Adapts the optimizer's in-memory graph for native report composition.

use std::fs;
use std::io;
use std::path::Path;

use autoeq_report_wasm::PreparedReportData;
use roomeq_workflow::MEASUREMENTS_INDEX_FILENAME;
use roomeq_workflow::output_bundle::ARTIFACT_BUNDLE_MANIFEST_FILENAME;
use serde_json::{Value, json};

use crate::{OutputBundleVerification, ReportInputError, load_legacy_input};

/// Report graph and index evidence captured from an in-memory result.
#[derive(Debug)]
pub struct LiveReportInput {
    /// Original graph with missing indexed measurements filled in.
    pub graph: Value,
    /// Captured index, including deployed source curve provenance.
    pub measurement_index: Option<Value>,
    /// Captured measurements for symmetric monitor pairs.
    pub symmetric_pairs: Option<Value>,
}

/// Prepares a live optimizer result for the shared report composer.
///
/// Resolves indexed measurements and preserves the result's embedded target
/// and provenance using the same rules for every native application.
///
/// # Errors
/// Returns an error when hydration or prepared-input validation fails.
pub fn prepare_live_report(
    result: &Value,
    artifact_root: Option<&Path>,
) -> Result<PreparedReportData, ReportInputError> {
    let LiveReportInput {
        graph,
        measurement_index,
        symmetric_pairs,
    } = hydrate_live_report(result, artifact_root)?;
    PreparedReportData::new(
        graph,
        Some(report_provenance(result)),
        measurement_index,
        symmetric_pairs,
        target_snapshot(result),
        None,
    )
    .map_err(ReportInputError::Preparation)
}

/// Hydrates a live result using the canonical legacy bundle loader.
///
/// An unindexed artifact directory can contain generated FIR assets. It
/// contributes no measurement overlay and does not need to be staged.
/// A saved, manifested graph must use the saved bundle loader, which
/// verifies the original serialized bytes and all bound artifacts.
///
/// # Errors
/// Returns an error for a saved-bundle marker, an invalid artifact root,
/// or a rejected snapshot of indexed artifacts.
pub fn hydrate_live_report(
    graph: &Value,
    artifact_root: Option<&Path>,
) -> Result<LiveReportInput, ReportInputError> {
    if !graph["artifact_bundle_schema_version"].is_null() {
        return Err(ReportInputError::ManifestedGraphRequiresSavedBundle);
    }

    if let Some(root) = artifact_root {
        if !fs::symlink_metadata(root)?.is_dir() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "RoomEQ report artifact root must be a regular directory",
            )
            .into());
        }
        if entry_exists(&root.join(ARTIFACT_BUNDLE_MANIFEST_FILENAME))? {
            return Err(ReportInputError::ManifestedGraphRequiresSavedBundle);
        }
        if entry_exists(&root.join(MEASUREMENTS_INDEX_FILENAME))? {
            let graph_bytes = serde_json::to_vec(graph)
                .map_err(|error| ReportInputError::Serialization(error.to_string()))?;
            let input = load_legacy_input(&graph_bytes, Some(root))?;
            if input.verification() == OutputBundleVerification::ManifestVerified {
                return Err(ReportInputError::ManifestedGraphRequiresSavedBundle);
            }
            let graph = input.hydrated_report_json()?;
            let symmetric_pairs = input.symmetric_pairs().cloned();
            return Ok(LiveReportInput {
                graph,
                measurement_index: input.measurement_index,
                symmetric_pairs,
            });
        }
    }

    Ok(LiveReportInput {
        graph: graph.clone(),
        measurement_index: None,
        symmetric_pairs: None,
    })
}

fn entry_exists(path: &Path) -> io::Result<bool> {
    match fs::symlink_metadata(path) {
        Ok(_) => Ok(true),
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(false),
        Err(error) => Err(error),
    }
}

fn report_provenance(result: &Value) -> Value {
    let versions: Vec<_> = result["version"].as_str().into_iter().collect();
    let timestamps: Vec<_> = result["metadata"]["timestamp"]
        .as_str()
        .into_iter()
        .collect();
    json!({"versions":versions,"timestamps":timestamps})
}

fn target_snapshot(result: &Value) -> Value {
    let target = &result["metadata"]["target_curve"];
    if target["freq"].is_array() && target["spl"].is_array() {
        return json!({
            "schema_version":1,
            "state":"available",
            "resolution":"embedded_result",
            "target":target,
        });
    }
    json!({"schema_version":1,"state":"not_configured","target":null})
}
