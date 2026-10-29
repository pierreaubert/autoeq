//! Native filesystem boundary for canonical AutoEQ report inputs.
//!
//! This crate deliberately stays outside `autoeq-report-wasm`: it owns paths,
//! temporary staging, and the existing RoomEQ bundle loader, while the shared
//! report core continues to consume in-memory data and remains WASM-safe.

use std::fs::{self, File, OpenOptions};
use std::io::{self, Read, Write};
use std::path::{Component, Path, PathBuf};
use std::time::SystemTime;

use roomeq_model::DspGraph;
use roomeq_workflow::{FrozenBundleResource, FrozenOutputBundle, load_output_bundle_frozen};
use serde_json::Value;

mod dsp_replay;
#[doc(inline)]
pub use dsp_replay::{CapturedFirResource, DspReplayError, DspReplayReport, DspReplayState};
#[doc(inline)]
pub use roomeq_workflow::OutputBundleVerification;
/// Bounds match the frozen bundle loader's accepted asset-tree limits.
const MAX_LEGACY_ARTIFACT_BYTES: u64 = 512 * 1024 * 1024;
const MAX_LEGACY_ARTIFACT_FILES: usize = 50_000;
const MAX_LEGACY_ARTIFACT_ENTRIES: usize = 50_000;
const MAX_LEGACY_ARTIFACT_MEMBER_BYTES: u64 = 256 * 1024 * 1024;

/// A report input captured from one native output bundle snapshot.
///
/// `raw_graph_json` and `raw_graph` preserve fields unknown to the current
/// `DspGraph` schema. `graph` is the loader's hydrated typed graph, and
/// `resources` are immutable convolution bytes captured and verified by the
/// loader. Callers can choose the canonical representation appropriate to a
/// report section without discarding forward-compatible graph fields.
#[derive(Debug)]
pub struct LoadedReportInput {
    raw_graph_json: Vec<u8>,
    raw_graph: Value,
    measurement_index: Option<Value>,
    bundle: FrozenOutputBundle,
}

impl LoadedReportInput {
    /// Prepare each channel's small-signal transfer on its hydrated initial-curve grid.
    ///
    /// Captured FIR bytes are resolved only from verified immutable bundle resources;
    /// graph-level routing and unsupported channel realizations remain unavailable.
    /// The curve contains transfer magnitude/phase, not measured SPL plus correction.
    /// This is not per-driver acoustic parity.
    ///
    /// # Errors
    /// Returns an error when the requested sample rate is not a positive integer rate.
    pub fn prepare_dsp_replay(
        &self,
        sample_rate_hz: f64,
    ) -> Result<DspReplayReport, DspReplayError> {
        dsp_replay::prepare(self, sample_rate_hz)
    }

    /// Return the exact graph bytes read by the frozen loader.
    pub fn raw_graph_json(&self) -> &[u8] {
        &self.raw_graph_json
    }

    /// Return the parsed graph, including fields unknown to the typed schema.
    pub fn raw_graph(&self) -> &Value {
        &self.raw_graph
    }

    /// Return the typed graph with indexed measurement data hydrated.
    pub fn graph(&self) -> &DspGraph {
        self.bundle.output()
    }

    /// Return the captured measurement index, including future extension keys.
    pub fn measurement_index(&self) -> Option<&Value> {
        self.measurement_index.as_ref()
    }

    /// Return symmetric-pair measurements retained by the captured index.
    pub fn symmetric_pairs(&self) -> Option<&Value> {
        self.measurement_index
            .as_ref()?
            .get("symmetric_pairs")
            .filter(|value| value.is_object())
    }

    /// Return the captured bytes for an index-referenced measurement member.
    pub fn measurement_sidecar(&self, relative_path: &str) -> Option<&[u8]> {
        self.bundle.measurement_sidecar(relative_path)
    }

    /// Parse one captured JSON measurement sidecar without reopening its file.
    ///
    /// # Errors
    /// Returns an error if the captured sidecar is not valid JSON.
    pub fn measurement_sidecar_json(
        &self,
        relative_path: &str,
    ) -> Result<Option<Value>, ReportInputError> {
        self.measurement_sidecar(relative_path)
            .map(serde_json::from_slice)
            .transpose()
            .map_err(ReportInputError::InvalidJson)
    }

    /// Hydrate indexed measurements into the raw graph without reserializing unrelated fields.
    ///
    /// Only fields that the frozen loader actually hydrated are copied from
    /// the typed graph, and only when the raw graph omitted or set that field
    /// to `null`. Existing inline values—including unknown fields nested in
    /// those values—remain untouched. This keeps forward-compatible graph
    /// data while making indexed curves, impulse responses, and acoustic
    /// measurements available to report consumers.
    ///
    /// This preserves indexed acoustic sidecar extensions and keeps index
    /// provenance available through [`Self::measurement_index`] and
    /// [`Self::symmetric_pairs`]. Python's private FIR replay cache is outside
    /// this graph-only view. See [`KNOWN_PYTHON_HYDRATION_GAPS`].
    ///
    /// # Errors
    /// Returns an error if the hydrated typed graph cannot be serialized as JSON.
    pub fn hydrated_report_json(&self) -> Result<Value, ReportInputError> {
        let hydrated_graph = serde_json::to_value(self.bundle.output())
            .map_err(|error| ReportInputError::Serialization(error.to_string()))?;
        let mut report_graph = self.raw_graph.clone();
        merge_hydrated_measurements(&mut report_graph, &hydrated_graph);
        preserve_unknown_acoustic_sidecar_fields(
            &mut report_graph,
            &self.raw_graph,
            self.measurement_index.as_ref(),
            &self.bundle,
        )?;
        Ok(report_graph)
    }

    /// Build the graph-plus-provenance object consumed by native report adapters.
    ///
    /// The graph remains the existing hydrated RoomEQ JSON. Index metadata is
    /// carried alongside it instead of being inserted into the saved graph,
    /// so graph identity and payload verification retain their current meaning.
    ///
    /// # Errors
    /// Returns an error if the hydrated graph cannot be serialized.
    pub fn hydrated_report_input_json(&self) -> Result<Value, ReportInputError> {
        let mut input = serde_json::Map::new();
        input.insert("graph".to_owned(), self.hydrated_report_json()?);
        if let Some(index) = &self.measurement_index {
            input.insert("measurement_index".to_owned(), index.clone());
        }
        if let Some(symmetric_pairs) = self.symmetric_pairs() {
            input.insert("symmetric_pairs".to_owned(), symmetric_pairs.clone());
        }
        Ok(Value::Object(input))
    }

    /// Return whether a supported manifest verified this snapshot.
    pub fn verification(&self) -> OutputBundleVerification {
        self.bundle.verification()
    }

    /// Iterate over immutable convolution resources captured with this input.
    pub fn resources(&self) -> impl Iterator<Item = &FrozenBundleResource> {
        self.bundle.resources()
    }

    /// Return the frozen bundle for access to its graph identity and resources.
    pub fn frozen_bundle(&self) -> &FrozenOutputBundle {
        &self.bundle
    }
}

/// Load an existing saved RoomEQ bundle through its canonical snapshot loader.
///
/// The bundle loader verifies the manifest, binds resource bytes to the exact
/// graph bytes, and hydrates indexed measurements. No graph reserialization or
/// second loader is introduced here.
///
/// # Errors
/// Returns an error when the bundle is missing, incomplete, mutable during
/// loading, or rejected by the canonical frozen bundle loader.
pub fn load_saved_bundle(path: &Path) -> Result<LoadedReportInput, ReportInputError> {
    let bundle = load_output_bundle_frozen(path)
        .map_err(|error| ReportInputError::Bundle(error.to_string()))?;
    from_frozen_bundle(bundle)
}

/// Report hydration differences that cannot be represented by `DspGraph` today.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum ReportHydrationGap {
    /// Python derives private FIR replay-cache values from convolution WAV resources.
    FirReplayCache,
}

/// Python-loader metadata that is not present in the native report input.
pub const KNOWN_PYTHON_HYDRATION_GAPS: &[ReportHydrationGap] =
    &[ReportHydrationGap::FirReplayCache];

/// Load legacy graph JSON plus an optional artifact directory.
///
/// The graph bytes are written unchanged to a temporary `result.json`; an
/// optional artifact root is copied into the corresponding `result_files`
/// directory. The canonical frozen loader then performs graph parsing,
/// hydration, and resource validation. A non-null manifest schema marker is
/// rejected because this API has no original sibling manifest to authenticate
/// the supplied bytes against.
///
/// # Errors
/// Returns an error for invalid graph JSON, a manifest-stamped graph, an
/// unsafe or oversized artifact tree, a source tree that changes while being
/// staged, or any rejection from the canonical frozen bundle loader.
pub fn load_legacy_input(
    graph_json: &[u8],
    artifact_root: Option<&Path>,
) -> Result<LoadedReportInput, ReportInputError> {
    let raw_graph: Value = serde_json::from_slice(graph_json)?;
    if raw_graph
        .get("artifact_bundle_schema_version")
        .is_some_and(|value| !value.is_null())
    {
        return Err(ReportInputError::ManifestedGraphRequiresSavedBundle);
    }

    let staging = tempfile::tempdir()?;
    let graph_path = staging.path().join("result.json");
    fs::write(&graph_path, graph_json)?;
    if let Some(artifact_root) = artifact_root {
        copy_legacy_artifact_root(artifact_root, &staging.path().join("result_files"))?;
    }

    let bundle = load_output_bundle_frozen(&graph_path)
        .map_err(|error| ReportInputError::Bundle(error.to_string()))?;
    // The frozen object owns graph and resource bytes. The temporary staging
    // tree may be deleted before returning to the caller.
    from_frozen_bundle_with_raw(bundle, graph_json.to_vec(), raw_graph)
}

/// Failures at the native report-input boundary.
#[derive(Debug, thiserror::Error)]
pub enum ReportInputError {
    /// The supplied legacy graph could not be parsed as JSON.
    #[error("invalid report graph JSON: {0}")]
    InvalidJson(#[from] serde_json::Error),
    /// The hydrated typed graph could not be converted to a JSON value.
    #[error("could not serialize hydrated report graph: {0}")]
    Serialization(String),
    /// A manifest-stamped graph must be loaded with its saved bundle path.
    #[error(
        "manifest-stamped graph requires the saved bundle path so its exact bytes can be verified"
    )]
    ManifestedGraphRequiresSavedBundle,
    /// A filesystem staging operation failed.
    #[error("report input staging failed: {0}")]
    Io(#[from] io::Error),
    /// The canonical frozen bundle loader rejected the input.
    #[error("RoomEQ frozen bundle loader rejected report input: {0}")]
    Bundle(String),
}

const CHANNEL_MEASUREMENT_FIELDS: &[&str] = &[
    "initial_curve",
    "final_curve",
    "eq_response",
    "target_curve",
    "pre_ir",
    "post_ir",
    "early_late_curves",
    "waterfall",
    "resonance_decays",
    "wavelet",
];

const DRIVER_ACOUSTIC_SUFFIX: &str = "_measured_acoustics";

fn preserve_unknown_acoustic_sidecar_fields(
    report: &mut Value,
    raw_graph: &Value,
    measurement_index: Option<&Value>,
    bundle: &FrozenOutputBundle,
) -> Result<(), ReportInputError> {
    let Some(channel_indices) = measurement_index
        .and_then(|index| index.get("channels"))
        .and_then(Value::as_object)
    else {
        return Ok(());
    };
    let (Some(report_channels), Some(raw_channels)) = (
        report.get_mut("channels").and_then(Value::as_object_mut),
        raw_graph.get("channels").and_then(Value::as_object),
    ) else {
        return Ok(());
    };
    for (channel_name, channel_entry) in channel_indices {
        let Some(index_fields) = channel_entry.as_object() else {
            continue;
        };
        let (Some(report_drivers), Some(raw_drivers)) = (
            report_channels
                .get_mut(channel_name)
                .and_then(|channel| channel.get_mut("drivers"))
                .and_then(Value::as_array_mut),
            raw_channels
                .get(channel_name)
                .and_then(|channel| channel.get("drivers"))
                .and_then(Value::as_array),
        ) else {
            continue;
        };
        for (driver_index, (report_driver, raw_driver)) in
            report_drivers.iter_mut().zip(raw_drivers).enumerate()
        {
            if raw_driver
                .get("measured_acoustics")
                .is_some_and(|value| !value.is_null())
            {
                continue;
            }
            let prefix = format!("driver{driver_index}_");
            let Some(relative_path) = index_fields.iter().find_map(|(key, value)| {
                (key.starts_with(&prefix) && key.ends_with(DRIVER_ACOUSTIC_SUFFIX))
                    .then(|| value.as_str())
                    .flatten()
            }) else {
                continue;
            };
            let Some(bytes) = bundle.measurement_sidecar(relative_path) else {
                continue;
            };
            let sidecar: Value = serde_json::from_slice(bytes)?;
            let Some(destination) = report_driver
                .get_mut("measured_acoustics")
                .filter(|value| value.is_object())
            else {
                continue;
            };
            merge_missing_json_fields(destination, &sidecar);
        }
    }
    Ok(())
}

fn merge_missing_json_fields(destination: &mut Value, source: &Value) {
    let (Some(destination), Some(source)) = (destination.as_object_mut(), source.as_object())
    else {
        return;
    };
    for (key, source_value) in source {
        match destination.get_mut(key) {
            Some(destination_value)
                if destination_value.is_object() && source_value.is_object() =>
            {
                merge_missing_json_fields(destination_value, source_value);
            }
            Some(_) => {}
            None => {
                destination.insert(key.clone(), source_value.clone());
            }
        }
    }
}

fn merge_hydrated_measurements(raw: &mut Value, hydrated: &Value) {
    let (Some(raw_channels), Some(hydrated_channels)) = (
        raw.get_mut("channels").and_then(Value::as_object_mut),
        hydrated.get("channels").and_then(Value::as_object),
    ) else {
        return;
    };

    for (channel_name, hydrated_channel) in hydrated_channels {
        let Some(hydrated_channel) = hydrated_channel.as_object() else {
            continue;
        };
        let Some(raw_channel) = raw_channels
            .get_mut(channel_name)
            .and_then(Value::as_object_mut)
        else {
            continue;
        };
        for field in CHANNEL_MEASUREMENT_FIELDS {
            copy_if_missing(raw_channel, hydrated_channel, field);
        }

        let (Some(raw_drivers), Some(hydrated_drivers)) = (
            raw_channel.get_mut("drivers").and_then(Value::as_array_mut),
            hydrated_channel.get("drivers").and_then(Value::as_array),
        ) else {
            continue;
        };
        for (raw_driver, hydrated_driver) in raw_drivers.iter_mut().zip(hydrated_drivers) {
            let (Some(raw_driver), Some(hydrated_driver)) =
                (raw_driver.as_object_mut(), hydrated_driver.as_object())
            else {
                continue;
            };
            copy_if_missing(raw_driver, hydrated_driver, "initial_curve");
            copy_if_missing(raw_driver, hydrated_driver, "measured_acoustics");
        }
    }

    if let Some(hydrated_deployed) = hydrated
        .get("deployed_source_curves")
        .and_then(Value::as_object)
        && let Some(raw_object) = raw.as_object_mut()
    {
        if raw_object
            .get("deployed_source_curves")
            .is_none_or(Value::is_null)
        {
            raw_object.insert(
                "deployed_source_curves".to_owned(),
                Value::Object(serde_json::Map::new()),
            );
        }
        let Some(raw_deployed) = raw_object
            .get_mut("deployed_source_curves")
            .and_then(Value::as_object_mut)
        else {
            return;
        };
        for (name, curve) in hydrated_deployed {
            raw_deployed
                .entry(name.clone())
                .or_insert_with(|| curve.clone());
        }
    }
}

fn copy_if_missing(
    raw: &mut serde_json::Map<String, Value>,
    hydrated: &serde_json::Map<String, Value>,
    key: &str,
) {
    if raw.get(key).is_some_and(|value| !value.is_null()) {
        return;
    }
    if let Some(value) = hydrated.get(key).filter(|value| !value.is_null()) {
        raw.insert(key.to_owned(), value.clone());
    }
}

fn from_frozen_bundle(bundle: FrozenOutputBundle) -> Result<LoadedReportInput, ReportInputError> {
    let raw_graph_json = bundle.slim_graph_bytes().to_vec();
    let raw_graph = serde_json::from_slice(&raw_graph_json)?;
    let measurement_index = bundle
        .measurement_index_json()
        .map(serde_json::from_slice)
        .transpose()?;
    Ok(LoadedReportInput {
        raw_graph_json,
        raw_graph,
        measurement_index,
        bundle,
    })
}

fn from_frozen_bundle_with_raw(
    bundle: FrozenOutputBundle,
    raw_graph_json: Vec<u8>,
    raw_graph: Value,
) -> Result<LoadedReportInput, ReportInputError> {
    // The loader reads these exact bytes. Keep a defensive equality check here
    // so future loader changes cannot silently break legacy-field preservation.
    if bundle.slim_graph_bytes() != raw_graph_json {
        return Err(ReportInputError::Bundle(
            "frozen loader returned graph bytes different from the staged input".into(),
        ));
    }
    let measurement_index = bundle
        .measurement_index_json()
        .map(serde_json::from_slice)
        .transpose()?;
    Ok(LoadedReportInput {
        raw_graph_json,
        raw_graph,
        measurement_index,
        bundle,
    })
}

fn copy_legacy_artifact_root(source: &Path, destination: &Path) -> io::Result<()> {
    let before = snapshot_artifact_tree(source)?;
    let mut total_bytes = 0_u64;
    let mut file_count = 0_usize;

    // Validate the complete tree before copying any member. A collection of
    // individually valid files can still exceed the aggregate staging budget.
    for entry in before.values() {
        let ArtifactEntry::File { len, .. } = entry else {
            continue;
        };
        file_count = file_count.saturating_add(1);
        if file_count > MAX_LEGACY_ARTIFACT_FILES {
            return Err(invalid_artifact_tree(
                "legacy artifact tree exceeds the frozen loader's file-count limit",
            ));
        }
        if *len > MAX_LEGACY_ARTIFACT_MEMBER_BYTES {
            return Err(invalid_artifact_tree(
                "legacy artifact member exceeds the frozen loader's per-file size limit",
            ));
        }
        total_bytes = total_bytes.saturating_add(*len);
        if total_bytes > MAX_LEGACY_ARTIFACT_BYTES {
            return Err(invalid_artifact_tree(
                "legacy artifact tree exceeds the frozen loader's total-size limit",
            ));
        }
    }
    fs::create_dir(destination)?;

    for (relative, entry) in &before {
        let source_path = source.join(relative);
        let destination_path = destination.join(relative);
        if matches!(entry, ArtifactEntry::Directory) {
            fs::create_dir_all(&destination_path)?;
            continue;
        }

        let ArtifactEntry::File { len, .. } = entry else {
            unreachable!("directory entries returned above");
        };
        if let Some(parent) = destination_path.parent() {
            fs::create_dir_all(parent)?;
        }
        copy_file_bounded(&source_path, &destination_path, *len)?;
    }

    if snapshot_artifact_tree(source)? != before {
        return Err(io::Error::new(
            io::ErrorKind::WouldBlock,
            "legacy artifact tree changed while it was being staged",
        ));
    }
    for (relative, entry) in &before {
        if matches!(entry, ArtifactEntry::File { .. })
            && !files_equal(&source.join(relative), &destination.join(relative))?
        {
            return Err(io::Error::new(
                io::ErrorKind::WouldBlock,
                format!("legacy artifact changed while being staged: {relative:?}"),
            ));
        }
    }
    Ok(())
}

#[derive(Debug, PartialEq, Eq)]
enum ArtifactEntry {
    Directory,
    File {
        len: u64,
        modified: Option<SystemTime>,
    },
}

fn snapshot_artifact_tree(
    root: &Path,
) -> io::Result<std::collections::BTreeMap<PathBuf, ArtifactEntry>> {
    snapshot_artifact_tree_with_limit(root, MAX_LEGACY_ARTIFACT_ENTRIES)
}

fn snapshot_artifact_tree_with_limit(
    root: &Path,
    max_entries: usize,
) -> io::Result<std::collections::BTreeMap<PathBuf, ArtifactEntry>> {
    let root_metadata = fs::symlink_metadata(root)?;
    if !root_metadata.is_dir() || root_metadata.file_type().is_symlink() {
        return Err(invalid_artifact_tree(
            "legacy artifact root must be a regular directory",
        ));
    }

    let mut snapshot = std::collections::BTreeMap::new();
    let mut directories = vec![root.to_path_buf()];
    while let Some(directory) = directories.pop() {
        let mut entries = Vec::new();
        for entry in fs::read_dir(&directory)? {
            entries.push(entry?);
            if entries.len().saturating_add(snapshot.len()) > max_entries {
                return Err(invalid_artifact_tree(
                    "legacy artifact tree exceeds the frozen loader's entry-count limit",
                ));
            }
        }
        entries.sort_by_key(|entry| entry.file_name());
        for entry in entries {
            let path = entry.path();
            let relative = path
                .strip_prefix(root)
                .map_err(|_| invalid_artifact_tree("artifact path escaped its root"))?
                .to_path_buf();
            if relative
                .components()
                .any(|component| !matches!(component, Component::Normal(_)))
            {
                return Err(invalid_artifact_tree(
                    "artifact path contains an unsafe component",
                ));
            }
            let metadata = fs::symlink_metadata(&path)?;
            if metadata.file_type().is_symlink() {
                return Err(invalid_artifact_tree(format!(
                    "legacy artifact tree contains a symbolic link: {relative:?}"
                )));
            }
            let snapshot_entry = if metadata.is_dir() {
                directories.push(path);
                ArtifactEntry::Directory
            } else if metadata.is_file() {
                ArtifactEntry::File {
                    len: metadata.len(),
                    modified: metadata.modified().ok(),
                }
            } else {
                return Err(invalid_artifact_tree(format!(
                    "legacy artifact tree contains a non-file entry: {relative:?}"
                )));
            };
            snapshot.insert(relative, snapshot_entry);
            if snapshot.len() > max_entries {
                return Err(invalid_artifact_tree(
                    "legacy artifact tree exceeds the frozen loader's entry-count limit",
                ));
            }
        }
    }
    Ok(snapshot)
}

fn copy_file_bounded(source: &Path, destination: &Path, expected_len: u64) -> io::Result<()> {
    let source_file = File::open(source)?;
    let opened_metadata = source_file.metadata()?;
    if !opened_metadata.is_file() || opened_metadata.len() != expected_len {
        return Err(io::Error::new(
            io::ErrorKind::WouldBlock,
            "legacy artifact changed before it could be staged",
        ));
    }
    let mut reader = source_file.take(expected_len.saturating_add(1));
    let mut destination_file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(destination)?;
    let copied = io::copy(&mut reader, &mut destination_file)?;
    if copied != expected_len {
        return Err(io::Error::new(
            io::ErrorKind::WouldBlock,
            "legacy artifact changed while it was being copied",
        ));
    }
    destination_file.flush()?;
    Ok(())
}

fn files_equal(left: &Path, right: &Path) -> io::Result<bool> {
    const BUFFER_SIZE: usize = 64 * 1024;
    let mut left = File::open(left)?;
    let mut right = File::open(right)?;
    let mut left_buffer = [0_u8; BUFFER_SIZE];
    let mut right_buffer = [0_u8; BUFFER_SIZE];
    loop {
        let left_read = left.read(&mut left_buffer)?;
        let right_read = right.read(&mut right_buffer)?;
        if left_read != right_read || left_buffer[..left_read] != right_buffer[..right_read] {
            return Ok(false);
        }
        if left_read == 0 {
            return Ok(true);
        }
    }
}

fn invalid_artifact_tree(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, message.into())
}

#[cfg(test)]
#[path = "tests.rs"]
mod tests;
