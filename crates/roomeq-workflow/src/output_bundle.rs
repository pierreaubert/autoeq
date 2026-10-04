//! Filesystem bundle for RoomEQ native outputs.
//!
//! A run writes a small `dsp.json` next to a sibling `<stem>_files`
//! directory (e.g. `dsp.json` + `dsp_files/`). All generated WAV sidecars,
//! extracted measurement curves/CSVs, diagnostic text files, the run
//! manifest, and the run log live inside that directory; nothing is written
//! to the process working directory.
//!
//! The slim JSON keeps the exact `DspGraph` schema with DSP data (plugins,
//! metadata, ledger). Heavy measurement blobs (`CurveData`, `IrWaveform`,
//! waterfall/wavelet/early-late grids, deployed curves) are extracted to
//! CSV/JSON files in the assets directory and stripped from the saved JSON.
//! Publication keeps the canonical support-directory path for existing
//! consumers. Concurrent readers can see a brief missing-directory
//! window during the swap. The Rust loader recovers from the durable
//! journal; the Python loader refuses to return a partial result while a
//! journal remains. Non-Unix directory-entry flushes are best-effort.
//! The Python viewer (`scripts/src/loaders.py`) re-injects them from the
//! sibling directory, so plots are unchanged while `dsp.json` stays small.

use std::borrow::Cow;
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::io::{self, Read, Write};
use std::path::{Component, Path, PathBuf};
use std::sync::Arc;

pub use autoeq_artifacts::FsArtifactStore;
use autoeq_artifacts::{
    ArtifactBundleStaging, safe_artifact_path, sha256_hex, write_file_atomically,
};
use sha2::{Digest, Sha256};

use roomeq_export::checked_convolution_resource_references;
use roomeq_model::{
    ChannelEarlyLateCurves, ChannelResonanceDecays, ChannelWaterfall, ChannelWavelet, CurveData,
    DspGraph, IrWaveform, MeasuredRoomAcoustics, decision_ledger::GraphIdentity,
};

/// Name of the run log written inside the assets directory.
pub const RUN_LOG_FILENAME: &str = "roomeq.log";
/// Name of the run manifest written inside the assets directory.
pub const RUN_MANIFEST_FILENAME: &str = "manifest.json";

/// Describes whether a loaded bundle has a verified content manifest.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutputBundleVerification {
    /// A supported manifest bound the slim graph and every listed member.
    ManifestVerified,
    /// A legacy graph loaded without a supported content manifest.
    LegacyUnverified,
}

/// One convolution asset captured and verified while loading a native bundle.
#[derive(Debug, Clone)]
pub struct FrozenBundleResource {
    relative_path: String,
    sha256: String,
    bytes: Arc<[u8]>,
}

impl FrozenBundleResource {
    /// Return the portable graph-relative path bound to this resource.
    pub fn relative_path(&self) -> &str {
        &self.relative_path
    }

    /// Return the SHA-256 digest declared by the graph and checked against captured bytes.
    pub fn sha256(&self) -> &str {
        &self.sha256
    }

    /// Return the exact immutable bytes checked during bundle loading.
    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }
}

/// A loaded graph with immutable identities and convolution bytes from one verified snapshot.
#[derive(Debug)]
pub struct FrozenOutputBundle {
    output: DspGraph,
    slim_graph_bytes: Arc<[u8]>,
    raw_graph_sha256: String,
    slim_graph_identity: GraphIdentity,
    verification: OutputBundleVerification,
    resources: BTreeMap<String, FrozenBundleResource>,
}

impl FrozenOutputBundle {
    /// Return the hydrated graph loaded from the captured slim JSON bytes.
    pub fn output(&self) -> &DspGraph {
        &self.output
    }

    /// Return the exact slim graph bytes checked against the manifest.
    pub fn slim_graph_bytes(&self) -> &[u8] {
        &self.slim_graph_bytes
    }

    /// Return the SHA-256 digest of the exact slim graph bytes.
    pub fn raw_graph_sha256(&self) -> &str {
        &self.raw_graph_sha256
    }

    /// Return the canonical graph identity used by final ledger bindings.
    pub fn slim_graph_identity(&self) -> &GraphIdentity {
        &self.slim_graph_identity
    }

    /// Return whether the loaded bundle had a verifiable content manifest.
    pub fn verification(&self) -> OutputBundleVerification {
        self.verification
    }

    /// Return a captured convolution resource by its graph-relative path.
    pub fn resource(&self, relative_path: &str) -> Option<&FrozenBundleResource> {
        self.resources.get(relative_path)
    }

    /// Iterate over all captured convolution resources in path order.
    pub fn resources(&self) -> impl Iterator<Item = &FrozenBundleResource> {
        self.resources.values()
    }
}

/// Sibling assets directory for a native output path.
///
/// `dsp.json` -> `<parent>/dsp_files/`; `out.json` -> `<parent>/out_files/`.
/// A bare filename with no parent resolves against the current directory.
pub fn assets_dir_for(output_path: &Path) -> PathBuf {
    let parent = output_path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .map(Path::to_path_buf)
        .unwrap_or_else(|| PathBuf::from("."));
    let stem = output_path
        .file_stem()
        .and_then(|stem| stem.to_str())
        .filter(|stem| !stem.is_empty())
        .unwrap_or("dsp");
    parent.join(format!("{stem}_files"))
}

/// Candidate directories that may hold assets referenced by a native output.
///
/// The first entry is the output's parent directory (legacy layout: sidecars
/// next to the JSON); the second is the sibling `<stem>_files` directory
/// (new layout). Lookup helpers should try each in order.
pub fn candidate_asset_dirs(output_path: &Path) -> Vec<PathBuf> {
    let parent = output_path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .map(Path::to_path_buf)
        .unwrap_or_else(|| PathBuf::from("."));
    let assets = assets_dir_for(output_path);
    if assets == parent {
        vec![parent]
    } else {
        vec![parent, assets]
    }
}

/// Manifest path for a pipeline output. The manifest always lives inside the
/// sibling assets directory, e.g. `dsp.json` -> `dsp_files/manifest.json`.
pub fn manifest_path_for(output_path: &Path) -> PathBuf {
    assets_dir_for(output_path).join(RUN_MANIFEST_FILENAME)
}

/// Run-log path for a pipeline output, e.g. `dsp.json` -> `dsp_files/roomeq.log`.
pub fn run_log_path_for(output_path: &Path) -> PathBuf {
    assets_dir_for(output_path).join(RUN_LOG_FILENAME)
}

/// Resolve a bare convolution `ir_file` reference against the candidate
/// asset directories. Absolute references are returned unchanged.
pub fn resolve_convolution_path(reference: &str, output_path: &Path) -> PathBuf {
    let direct = Path::new(reference);
    if direct.is_absolute() {
        return direct.to_path_buf();
    }
    for dir in candidate_asset_dirs(output_path) {
        let candidate = dir.join(direct);
        if candidate.is_file() {
            return candidate;
        }
    }
    // Fall back to the legacy parent-relative join so error messages point
    // at the historically expected location.
    candidate_asset_dirs(output_path)
        .into_iter()
        .next()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(direct)
}

/// Read legacy convolution bytes from a path derived from the graph reference.
///
/// This compatibility helper performs an unbound filesystem read and does not
/// verify a bundle manifest. Strict native consumers must use
/// [`FrozenOutputBundle::resource`] from [`load_output_bundle_frozen`].
///
/// # Errors
/// Returns the underlying filesystem error when neither candidate path can be read.
pub fn read_convolution_bytes(reference: &str, output_path: &Path) -> std::io::Result<Vec<u8>> {
    let direct = Path::new(reference);
    if direct.is_absolute() {
        return std::fs::read(direct);
    }
    let mut last_error = None;
    for dir in candidate_asset_dirs(output_path) {
        match std::fs::read(dir.join(direct)) {
            Ok(bytes) => return Ok(bytes),
            Err(error) => last_error = Some(error),
        }
    }
    Err(last_error.unwrap_or_else(|| {
        std::io::Error::new(
            std::io::ErrorKind::NotFound,
            format!("convolution resource '{reference}' was not found"),
        )
    }))
}

fn sanitize_name(input: &str) -> String {
    let mut cleaned: String = input
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '-' || c == '_' {
                c
            } else {
                '_'
            }
        })
        .collect();
    while cleaned.contains("__") {
        cleaned = cleaned.replace("__", "_");
    }
    let cleaned = cleaned.trim_matches(|c| c == '_' || c == '-' || c == '.');
    if cleaned.is_empty() {
        "channel".to_string()
    } else {
        cleaned.to_string()
    }
}

fn validate_curve_data(curve: &CurveData) -> io::Result<()> {
    if curve.freq.len() != curve.spl.len() {
        return Err(io_invalid(
            "curve frequency and SPL arrays have different lengths",
        ));
    }
    if let Some(phase) = &curve.phase
        && phase.len() != curve.freq.len()
    {
        return Err(io_invalid(
            "curve phase array length does not match frequency data",
        ));
    }
    if curve
        .noise_floor_db
        .as_ref()
        .is_some_and(|values| values.len() != curve.freq.len())
        || curve
            .coherence
            .as_ref()
            .is_some_and(|values| values.len() != curve.freq.len())
    {
        return Err(io_invalid(
            "curve noise-floor and coherence arrays must match frequency data",
        ));
    }
    if curve
        .freq
        .iter()
        .any(|value| !value.is_finite() || *value <= 0.0)
        || curve.spl.iter().any(|value| !value.is_finite())
        || curve
            .phase
            .as_ref()
            .is_some_and(|phase| phase.iter().any(|value| !value.is_finite()))
        || curve
            .noise_floor_db
            .as_ref()
            .is_some_and(|values| values.iter().any(|value| !value.is_finite()))
        || curve
            .coherence
            .as_ref()
            .is_some_and(|values| values.iter().any(|value| !value.is_finite()))
        || curve
            .norm_range
            .is_some_and(|(start, end)| !start.is_finite() || !end.is_finite())
    {
        return Err(io_invalid("curve data must contain only finite values"));
    }
    if curve
        .coherence
        .as_ref()
        .is_some_and(|values| values.iter().any(|value| !(0.0..=1.0).contains(value)))
    {
        return Err(io_invalid(
            "curve coherence values must be between zero and one",
        ));
    }
    if curve
        .norm_range
        .is_some_and(|(start, end)| start <= 0.0 || start >= end)
    {
        return Err(io_invalid(
            "curve normalization range must be positive and ordered",
        ));
    }
    if curve.freq.windows(2).any(|pair| pair[0] >= pair[1]) {
        return Err(io_invalid("curve frequencies must be strictly increasing"));
    }
    Ok(())
}

fn validate_ir_data(ir: &IrWaveform) -> io::Result<()> {
    if ir.time_ms.len() != ir.amplitude.len() {
        return Err(io_invalid(
            "IR time and amplitude arrays have different lengths",
        ));
    }
    if ir.time_ms.iter().any(|value| !value.is_finite())
        || ir.amplitude.iter().any(|value| !value.is_finite())
    {
        return Err(io_invalid("IR data must contain only finite values"));
    }
    if ir.time_ms.windows(2).any(|pair| pair[0] >= pair[1]) {
        return Err(io_invalid("IR sample times must be strictly increasing"));
    }
    Ok(())
}

fn write_curve_csv(path: &Path, curve: &CurveData) -> std::io::Result<()> {
    validate_curve_data(curve)?;
    let mut out = String::with_capacity(curve.freq.len() * 32);
    let has_phase = curve.phase.is_some();
    if has_phase {
        out.push_str("freq,spl,phase\n");
    } else {
        out.push_str("freq,spl\n");
    }
    for (index, (freq, spl)) in curve.freq.iter().zip(curve.spl.iter()).enumerate() {
        if has_phase {
            let phase = curve.phase.as_ref().expect("phase presence checked")[index];
            out.push_str(&format!("{freq},{spl},{phase}\n"));
        } else {
            out.push_str(&format!("{freq},{spl}\n"));
        }
    }
    std::fs::write(path, out)
}

#[derive(serde::Serialize)]
struct CurveDataMetadata<'a> {
    freq: &'a [f64],
    spl: &'a [f64],
    #[serde(skip_serializing_if = "Option::is_none")]
    phase: Option<&'a [f64]>,
    #[serde(skip_serializing_if = "Option::is_none")]
    norm_range: Option<(f64, f64)>,
    #[serde(skip_serializing_if = "Option::is_none")]
    noise_floor_db: Option<&'a [f64]>,
    #[serde(skip_serializing_if = "Option::is_none")]
    coherence: Option<&'a [f64]>,
}

fn write_curve_metadata_json(path: &Path, curve: &CurveData) -> io::Result<()> {
    let metadata = CurveDataMetadata {
        freq: &curve.freq,
        spl: &curve.spl,
        phase: curve.phase.as_deref(),
        norm_range: curve.norm_range,
        noise_floor_db: curve.noise_floor_db.as_deref(),
        coherence: curve.coherence.as_deref(),
    };
    write_json_file(path, &metadata)
}

fn write_ir_csv(path: &Path, ir: &IrWaveform) -> std::io::Result<()> {
    validate_ir_data(ir)?;
    let mut out = String::with_capacity(ir.time_ms.len() * 32);
    out.push_str("time_ms,amplitude\n");
    for (time, amplitude) in ir.time_ms.iter().zip(ir.amplitude.iter()) {
        // Default float formatting round-trips; fixed decimal precision can
        // erase low-level native IR tails and shift capture time origins.
        out.push_str(&format!("{time},{amplitude}\n"));
    }
    std::fs::write(path, out)
}

fn write_json_file(path: &Path, value: &impl serde::Serialize) -> std::io::Result<()> {
    let json = serde_json::to_string_pretty(value).map_err(std::io::Error::other)?;
    std::fs::write(path, json)
}

/// Files extracted from one slimmed output, relative to the assets directory.
#[derive(Debug, Default)]
pub struct ExtractedMeasurementFiles {
    /// Relative file names written into the assets directory.
    pub files: Vec<String>,
    /// Per-channel curve/IR/diagnostic file mapping, also written as
    /// `measurements_index.json` when non-empty.
    pub index: serde_json::Value,
}

/// Name of the measurement index written into the assets directory.
pub const MEASUREMENTS_INDEX_FILENAME: &str = "measurements_index.json";

fn record_file(
    extracted: &mut ExtractedMeasurementFiles,
    index: &mut serde_json::Map<String, serde_json::Value>,
    channel: &str,
    kind: &str,
    file_name: &str,
) {
    extracted.files.push(file_name.to_string());
    let entry = index
        .entry(channel.to_string())
        .or_insert_with(|| serde_json::Value::Object(serde_json::Map::new()));
    if let Some(object) = entry.as_object_mut() {
        object.insert(
            kind.to_string(),
            serde_json::Value::String(file_name.to_string()),
        );
    }
}

fn record_curve_files(
    extracted: &mut ExtractedMeasurementFiles,
    index: &mut serde_json::Map<String, serde_json::Value>,
    channel: &str,
    kind: &str,
    csv_file_name: &str,
    metadata_file_name: &str,
) {
    record_file(extracted, index, channel, kind, csv_file_name);
    record_file(
        extracted,
        index,
        channel,
        &format!("{kind}_metadata"),
        metadata_file_name,
    );
}

fn extract_curve(
    dir: &Path,
    extracted: &mut ExtractedMeasurementFiles,
    index: &mut serde_json::Map<String, serde_json::Value>,
    channel: &str,
    kind: &str,
    file_name: String,
    curve: Option<CurveData>,
) -> Option<CurveData> {
    let curve = curve?;
    let path = dir.join(&file_name);
    let metadata_file_name = format!("{file_name}.json");
    let metadata_path = dir.join(&metadata_file_name);
    if write_curve_csv(&path, &curve).is_ok()
        && write_curve_metadata_json(&metadata_path, &curve).is_ok()
    {
        record_curve_files(
            extracted,
            index,
            channel,
            kind,
            &file_name,
            &metadata_file_name,
        );
        None
    } else {
        Some(curve)
    }
}

fn extract_ir(
    dir: &Path,
    extracted: &mut ExtractedMeasurementFiles,
    index: &mut serde_json::Map<String, serde_json::Value>,
    channel: &str,
    kind: &str,
    file_name: String,
    ir: Option<IrWaveform>,
) -> Option<IrWaveform> {
    let ir = ir?;
    let path = dir.join(&file_name);
    match write_ir_csv(&path, &ir) {
        Ok(()) => {
            record_file(extracted, index, channel, kind, &file_name);
            None
        }
        Err(_) => Some(ir),
    }
}

fn extract_json_blob<T: serde::Serialize>(
    dir: &Path,
    extracted: &mut ExtractedMeasurementFiles,
    index: &mut serde_json::Map<String, serde_json::Value>,
    channel: &str,
    kind: &str,
    file_name: String,
    value: Option<T>,
) -> Option<T> {
    let value = value?;
    let path = dir.join(&file_name);
    match write_json_file(&path, &value) {
        Ok(()) => {
            record_file(extracted, index, channel, kind, &file_name);
            None
        }
        Err(_) => Some(value),
    }
}

/// Strip heavy measurement blobs from `output`, writing them as CSV/JSON
/// files into `assets_dir`. DSP data (plugins, metadata, ledger) stays in
/// the JSON. On write failure the blob is kept inline so no data is lost.
pub fn extract_measurements_to_assets(
    output: &mut DspGraph,
    assets_dir: &Path,
) -> ExtractedMeasurementFiles {
    extract_measurements_to_assets_with_index_writer(output, assets_dir, &mut |path, value| {
        write_json_file(path, value)
    })
}

fn extract_measurements_to_assets_with_index_writer(
    output: &mut DspGraph,
    assets_dir: &Path,
    write_index: &mut impl FnMut(&Path, &serde_json::Value) -> io::Result<()>,
) -> ExtractedMeasurementFiles {
    let symmetric_pairs = crate::symmetric_report::pairs(output);
    let mut extracted = ExtractedMeasurementFiles::default();
    let mut channel_index = serde_json::Map::new();
    let mut deployed_index = serde_json::Map::new();
    let mut channels: Vec<String> = output.channels.keys().cloned().collect();
    channels.sort();
    for name in channels {
        let Some(chain) = output.channels.get_mut(&name) else {
            continue;
        };
        let tag = sanitize_name(&name);
        chain.initial_curve = extract_curve(
            assets_dir,
            &mut extracted,
            &mut channel_index,
            &name,
            "initial_curve",
            format!("{tag}__initial.csv"),
            chain.initial_curve.take(),
        );
        chain.final_curve = extract_curve(
            assets_dir,
            &mut extracted,
            &mut channel_index,
            &name,
            "final_curve",
            format!("{tag}__final.csv"),
            chain.final_curve.take(),
        );
        chain.eq_response = extract_curve(
            assets_dir,
            &mut extracted,
            &mut channel_index,
            &name,
            "eq_response",
            format!("{tag}__eq.csv"),
            chain.eq_response.take(),
        );
        chain.target_curve = extract_curve(
            assets_dir,
            &mut extracted,
            &mut channel_index,
            &name,
            "target_curve",
            format!("{tag}__target.csv"),
            chain.target_curve.take(),
        );
        chain.pre_ir = extract_ir(
            assets_dir,
            &mut extracted,
            &mut channel_index,
            &name,
            "pre_ir",
            format!("{tag}__pre_ir.csv"),
            chain.pre_ir.take(),
        );
        chain.post_ir = extract_ir(
            assets_dir,
            &mut extracted,
            &mut channel_index,
            &name,
            "post_ir",
            format!("{tag}__post_ir.csv"),
            chain.post_ir.take(),
        );
        chain.early_late_curves = extract_json_blob::<ChannelEarlyLateCurves>(
            assets_dir,
            &mut extracted,
            &mut channel_index,
            &name,
            "early_late_curves",
            format!("{tag}__early_late_curves.json"),
            chain.early_late_curves.take(),
        );
        chain.waterfall = extract_json_blob::<ChannelWaterfall>(
            assets_dir,
            &mut extracted,
            &mut channel_index,
            &name,
            "waterfall",
            format!("{tag}__waterfall.json"),
            chain.waterfall.take(),
        );
        chain.resonance_decays = extract_json_blob::<ChannelResonanceDecays>(
            assets_dir,
            &mut extracted,
            &mut channel_index,
            &name,
            "resonance_decays",
            format!("{tag}__resonance_decays.json"),
            chain.resonance_decays.take(),
        );
        chain.wavelet = extract_json_blob::<ChannelWavelet>(
            assets_dir,
            &mut extracted,
            &mut channel_index,
            &name,
            "wavelet",
            format!("{tag}__wavelet.json"),
            chain.wavelet.take(),
        );
        if let Some(drivers) = chain.drivers.as_mut() {
            for (index, driver) in drivers.iter_mut().enumerate() {
                let driver_tag = sanitize_name(&driver.name);
                driver.measured_acoustics = extract_json_blob::<MeasuredRoomAcoustics>(
                    assets_dir,
                    &mut extracted,
                    &mut channel_index,
                    &name,
                    &format!("driver{index}_{driver_tag}_measured_acoustics"),
                    format!("{tag}__driver{index}_{driver_tag}__measured_acoustics.json"),
                    driver.measured_acoustics.take(),
                );
                let kind = format!("driver{index}_{driver_tag}_initial_curve");
                let before = driver.initial_curve.take();
                driver.initial_curve = extract_curve(
                    assets_dir,
                    &mut extracted,
                    &mut channel_index,
                    &name,
                    &kind,
                    format!("{tag}__driver{index}_{driver_tag}__initial.csv"),
                    before,
                );
            }
        }
    }
    let mut deployed: Vec<String> = output.deployed_source_curves.keys().cloned().collect();
    deployed.sort();
    let mut deployed_metadata_index = serde_json::Map::new();
    for name in deployed {
        if let Some(curve) = output.deployed_source_curves.remove(&name) {
            let tag = sanitize_name(&name);
            let file_name = format!("deployed__{tag}.csv");
            let metadata_file_name = format!("{file_name}.json");
            let path = assets_dir.join(&file_name);
            let metadata_path = assets_dir.join(&metadata_file_name);
            if write_curve_csv(&path, &curve).is_ok()
                && write_curve_metadata_json(&metadata_path, &curve).is_ok()
            {
                extracted.files.push(file_name.clone());
                deployed_index.insert(name.clone(), serde_json::Value::String(file_name));
                extracted.files.push(metadata_file_name.clone());
                deployed_metadata_index.insert(name, serde_json::Value::String(metadata_file_name));
            } else {
                output.deployed_source_curves.insert(name, curve);
            }
        }
    }
    extracted.files.sort();
    if !channel_index.is_empty() || !deployed_index.is_empty() {
        let mut root = serde_json::Map::new();
        root.insert("symmetric_pairs".to_owned(), symmetric_pairs);
        root.insert(
            "channels".to_string(),
            serde_json::Value::Object(channel_index),
        );
        root.insert(
            "deployed_source_curves".to_string(),
            serde_json::Value::Object(deployed_index),
        );
        root.insert(
            "deployed_source_curve_metadata".to_string(),
            serde_json::Value::Object(deployed_metadata_index),
        );
        let index_value = serde_json::Value::Object(root);
        let path = assets_dir.join(MEASUREMENTS_INDEX_FILENAME);
        if write_index(&path, &index_value).is_ok() {
            extracted
                .files
                .push(MEASUREMENTS_INDEX_FILENAME.to_string());
            extracted.files.sort();
        }
        extracted.index = index_value;
    }
    extracted
}

/// File name of the integrity manifest for a published artifact bundle.
pub const ARTIFACT_BUNDLE_MANIFEST_FILENAME: &str = "artifact_bundle_manifest.json";

const BUNDLE_TRANSACTION_SUFFIX: &str = ".autoeq-transaction.json";
const BUNDLE_MANIFEST_SCHEMA_VERSION: u32 = 1;
// Native graphs may contain full optimization evidence, so allow substantially
// more data than sidecars while still bounding reads from untrusted files.
const MAX_NATIVE_GRAPH_BYTES: u64 = 512 * 1024 * 1024;
// A single measurement or FIR member should never require an unbounded read.
const MAX_BUNDLE_MEMBER_BYTES: u64 = 256 * 1024 * 1024;
const MAX_BUNDLE_TOTAL_BYTES: u64 = 512 * 1024 * 1024;
const MAX_RUN_MANIFEST_BYTES: u64 = 2 * 1024 * 1024;
const MAX_RUN_LOG_BYTES: u64 = 16 * 1024 * 1024;
const MAX_BUNDLE_MANIFEST_BYTES: u64 = 2 * 1024 * 1024;
const MAX_BUNDLE_MANIFEST_FILES: usize = 50_000;
const MAX_BUNDLE_MEMBER_PATH_BYTES: usize = 1_024;
const MAX_BUNDLE_JOURNAL_BYTES: u64 = 64 * 1024;

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
struct BundleFileRecord {
    size_bytes: u64,
    sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
struct BundleManifest {
    schema_version: u32,
    producer: String,
    producer_version: String,
    graph_schema_version: String,
    generation: String,
    graph_sha256: String,
    files: BTreeMap<String, BundleFileRecord>,
}

type CapturedBundleFiles = BTreeMap<String, Arc<[u8]>>;

#[derive(Debug, serde::Serialize, serde::Deserialize)]
struct BundleTransactionJournal {
    schema_version: u32,
    transaction_directory: String,
    generation: String,
    had_assets: bool,
    had_output: bool,
    old_output_sha256: Option<String>,
    new_output_sha256: String,
}

type PrepareHook<'a> =
    dyn FnMut(&mut DspGraph, &Path) -> Result<(), Box<dyn std::error::Error>> + 'a;
type RenameHook<'a> = dyn FnMut(&Path, &Path) -> io::Result<()> + 'a;
type WriteHook<'a> = dyn FnMut(&Path, &[u8]) -> io::Result<()> + 'a;
type SyncHook<'a> = dyn FnMut(&Path) -> io::Result<()> + 'a;

struct BundleHooks<'a> {
    prepare: &'a mut PrepareHook<'a>,
    rename: &'a mut RenameHook<'a>,
    write_root: &'a mut WriteHook<'a>,
    write_journal: &'a mut WriteHook<'a>,
    sync_parent: &'a mut SyncHook<'a>,
}

struct PreparedBundle<'a> {
    output_path: &'a Path,
    staging: ArtifactBundleStaging,
    stage_assets: &'a Path,
    output_bytes: &'a [u8],
    generation: String,
    had_assets: bool,
}

fn bundle_transaction_path(output_path: &Path) -> PathBuf {
    let mut file_name = output_path
        .file_name()
        .map(std::ffi::OsString::from)
        .unwrap_or_else(|| std::ffi::OsString::from("dsp.json"));
    file_name.push(BUNDLE_TRANSACTION_SUFFIX);
    output_path.with_file_name(file_name)
}

fn io_invalid(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message.into())
}

#[cfg(test)]
fn pause_after_test_publication_phase(phase: &str) {
    const CHILD_ENV: &str = "ROOMEQ_BUNDLE_CRASH_TEST_CHILD";
    const PHASE_ENV: &str = "ROOMEQ_BUNDLE_CRASH_TEST_PHASE";
    const READY_ENV: &str = "ROOMEQ_BUNDLE_CRASH_TEST_READY_FILE";

    if std::env::var(CHILD_ENV).as_deref() != Ok("1")
        || std::env::var(PHASE_ENV).as_deref() != Ok(phase)
    {
        return;
    }

    let ready_path = std::env::var_os(READY_ENV)
        .map(PathBuf::from)
        .expect("child process barrier path is provided");
    let temporary_path = ready_path.with_extension("ready.tmp");
    let mut marker = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&temporary_path)
        .expect("create child process barrier marker");
    writeln!(marker, "{phase}").expect("write child process barrier marker");
    marker
        .sync_all()
        .expect("sync child process barrier marker");
    drop(marker);
    std::fs::rename(&temporary_path, &ready_path).expect("publish child process barrier marker");

    loop {
        std::thread::park_timeout(std::time::Duration::from_millis(100));
    }
}

fn sha256_file_bounded(path: &Path, maximum_bytes: u64) -> io::Result<(u64, String)> {
    let file = std::fs::File::open(path)?;
    let mut file = file.take(maximum_bytes.saturating_add(1));
    let mut digest = Sha256::new();
    let mut total = 0_u64;
    let mut buffer = [0_u8; 64 * 1024];
    loop {
        let read = file.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        total = total
            .checked_add(read as u64)
            .ok_or_else(|| io_invalid("artifact bundle member size overflow"))?;
        if total > maximum_bytes {
            return Err(io_invalid("artifact bundle member exceeds its size limit"));
        }
        digest.update(&buffer[..read]);
    }
    let encoded = digest
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect();
    Ok((total, encoded))
}

fn sha256_file(path: &Path) -> io::Result<(u64, String)> {
    sha256_file_bounded(path, MAX_BUNDLE_MEMBER_BYTES)
}

fn is_sha256_hex(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn read_bounded_file(path: &Path, maximum_bytes: u64, label: &str) -> io::Result<Vec<u8>> {
    let metadata = std::fs::symlink_metadata(path)?;
    if metadata.file_type().is_symlink() || !metadata.is_file() {
        return Err(io_invalid(format!("{label} is not a regular file")));
    }
    if metadata.len() > maximum_bytes {
        return Err(io_invalid(format!("{label} exceeds its size limit")));
    }
    let file = std::fs::File::open(path)?;
    let mut bytes = Vec::with_capacity(metadata.len() as usize);
    file.take(maximum_bytes + 1).read_to_end(&mut bytes)?;
    if bytes.len() as u64 > maximum_bytes {
        return Err(io_invalid(format!("{label} exceeds its size limit")));
    }
    Ok(bytes)
}

fn read_bundle_member_bytes<'a>(
    assets: &Path,
    relative: &Path,
    manifest: Option<&BundleManifest>,
    captured_files: Option<&'a CapturedBundleFiles>,
    maximum_bytes: u64,
    label: &str,
) -> io::Result<Cow<'a, [u8]>> {
    autoeq_artifacts::validate_relative_artifact_path(relative)?;
    let path = if let Some(manifest) = manifest {
        let key = relative
            .to_str()
            .ok_or_else(|| io_invalid("artifact member path is not valid UTF-8"))?
            .replace('\\', "/");
        let expected = manifest.files.get(&key).ok_or_else(|| {
            io_invalid(format!(
                "{label} is not bound by the artifact bundle manifest"
            ))
        })?;
        if expected.size_bytes > maximum_bytes {
            return Err(io_invalid(format!("{label} exceeds its size limit")));
        }
        if let Some(captured_files) = captured_files {
            let bytes = captured_files.get(&key).ok_or_else(|| {
                io_invalid(format!(
                    "{label} is missing from the captured bundle snapshot"
                ))
            })?;
            if bytes.len() as u64 != expected.size_bytes || sha256_hex(bytes) != expected.sha256 {
                return Err(io_invalid(format!(
                    "{label} failed artifact bundle integrity validation"
                )));
            }
            if bytes.len() as u64 > maximum_bytes {
                return Err(io_invalid(format!("{label} exceeds its size limit")));
            }
            return Ok(Cow::Borrowed(bytes));
        }
        let path = checked_bundle_member(assets, relative)?;
        let bytes = read_bounded_file(&path, expected.size_bytes, label)?;
        if bytes.len() as u64 != expected.size_bytes || sha256_hex(&bytes) != expected.sha256 {
            return Err(io_invalid(format!(
                "{label} failed artifact bundle integrity validation"
            )));
        }
        return Ok(Cow::Owned(bytes));
    } else {
        safe_artifact_path(assets, relative)?
    };
    Ok(Cow::Owned(read_bounded_file(&path, maximum_bytes, label)?))
}

fn read_bundle_path_bytes<'a>(
    assets: &Path,
    path: &Path,
    manifest: Option<&BundleManifest>,
    captured_files: Option<&'a CapturedBundleFiles>,
    maximum_bytes: u64,
    label: &str,
) -> io::Result<Cow<'a, [u8]>> {
    let relative = path
        .strip_prefix(assets)
        .map_err(|_| io_invalid("artifact member path escapes its support directory"))?;
    read_bundle_member_bytes(
        assets,
        relative,
        manifest,
        captured_files,
        maximum_bytes,
        label,
    )
}

fn checked_bundle_member(root: &Path, relative: &Path) -> io::Result<PathBuf> {
    let path = safe_artifact_path(root, relative)?;
    let components: Vec<_> = relative
        .components()
        .filter_map(|component| match component {
            Component::Normal(value) => Some(value),
            _ => None,
        })
        .collect();
    let mut current = root.to_path_buf();
    for (index, component) in components.iter().enumerate() {
        current.push(component);
        let metadata = std::fs::symlink_metadata(&current)?;
        if metadata.file_type().is_symlink()
            || (index + 1 == components.len() && !metadata.is_file())
            || (index + 1 < components.len() && !metadata.is_dir())
        {
            return Err(io_invalid(format!(
                "artifact member path is not a regular file inside its bundle: {}",
                relative.display()
            )));
        }
    }
    Ok(path)
}

#[cfg(unix)]
fn sync_directory(path: &Path) -> io::Result<()> {
    std::fs::File::open(path)?.sync_all()
}

// std exposes no portable directory-handle sync on Windows. File contents and
// the journal are still flushed there, but directory-entry durability across
// sudden power loss is best-effort until the native Windows API is adopted.
#[cfg(not(unix))]
fn sync_directory(_path: &Path) -> io::Result<()> {
    Ok(())
}

fn sync_tree(path: &Path) -> io::Result<()> {
    for entry in std::fs::read_dir(path)? {
        let entry = entry?;
        let file_type = entry.file_type()?;
        if file_type.is_symlink() {
            return Err(io_invalid(format!(
                "artifact bundle contains a symlink: {}",
                entry.path().display()
            )));
        }
        if file_type.is_dir() {
            sync_tree(&entry.path())?;
        } else if file_type.is_file() {
            std::fs::File::open(entry.path())?.sync_all()?;
        } else {
            return Err(io_invalid(format!(
                "artifact bundle contains a non-file entry: {}",
                entry.path().display()
            )));
        }
    }
    sync_directory(path)
}

#[derive(Default)]
struct AssetCopyBudget {
    files: usize,
    total_bytes: u64,
}

fn copy_asset_tree(
    source: &Path,
    destination: &Path,
    budget: &mut AssetCopyBudget,
) -> io::Result<()> {
    let metadata = std::fs::symlink_metadata(source)?;
    if !metadata.is_dir() || metadata.file_type().is_symlink() {
        return Err(io_invalid(format!(
            "artifact support path is not a regular directory: {}",
            source.display()
        )));
    }
    std::fs::create_dir_all(destination)?;
    for entry in std::fs::read_dir(source)? {
        let entry = entry?;
        let file_type = entry.file_type()?;
        let target = destination.join(entry.file_name());
        if file_type.is_symlink() {
            return Err(io_invalid(format!(
                "artifact support directory contains a symlink: {}",
                entry.path().display()
            )));
        }
        if file_type.is_dir() {
            copy_asset_tree(&entry.path(), &target, budget)?;
        } else if file_type.is_file() {
            let metadata = std::fs::symlink_metadata(entry.path())?;
            if metadata.len() > MAX_BUNDLE_MEMBER_BYTES {
                return Err(io_invalid("artifact support member exceeds its size limit"));
            }
            budget.files = budget
                .files
                .checked_add(1)
                .ok_or_else(|| io_invalid("artifact support file count overflow"))?;
            budget.total_bytes = budget
                .total_bytes
                .checked_add(metadata.len())
                .ok_or_else(|| io_invalid("artifact support size overflow"))?;
            if budget.files > MAX_BUNDLE_MANIFEST_FILES
                || budget.total_bytes > MAX_BUNDLE_TOTAL_BYTES
            {
                return Err(io_invalid(
                    "artifact support tree exceeds its size or file-count limit",
                ));
            }
            let source_file = std::fs::File::open(entry.path())?;
            let opened_metadata = source_file.metadata()?;
            if !opened_metadata.is_file() || opened_metadata.len() != metadata.len() {
                return Err(io_invalid(
                    "artifact support member changed while being opened",
                ));
            }
            let mut destination_file = std::fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(target)?;
            let copied = io::copy(
                &mut source_file.take(metadata.len().saturating_add(1)),
                &mut destination_file,
            )?;
            if copied != metadata.len() {
                return Err(io_invalid(
                    "artifact support member changed while being copied",
                ));
            }
            destination_file.sync_all()?;
        } else {
            return Err(io_invalid(format!(
                "artifact support directory contains a non-file entry: {}",
                entry.path().display()
            )));
        }
    }
    Ok(())
}

fn copy_existing_asset_tree(
    source: &Path,
    destination: &Path,
    require_manifest: bool,
    root_graph: Option<&Path>,
) -> io::Result<()> {
    match load_manifest(source)? {
        Some(manifest) => {
            let manifest_path = source.join(ARTIFACT_BUNDLE_MANIFEST_FILENAME);
            let manifest_bytes = read_bounded_file(
                &manifest_path,
                MAX_BUNDLE_MANIFEST_BYTES,
                "artifact bundle manifest",
            )?;
            verify_bundle_tree_matches_manifest(source, &manifest)?;
            if let Some(root_graph) = root_graph {
                let root_bytes =
                    read_bounded_file(root_graph, MAX_NATIVE_GRAPH_BYTES, "native graph")?;
                if sha256_hex(&root_bytes) != manifest.graph_sha256 {
                    return Err(io_invalid(
                        "native graph does not match its artifact manifest",
                    ));
                }
            }
            copy_manifested_asset_tree(source, destination, &manifest, &manifest_bytes)?;
            verify_bundle_tree_matches_manifest(destination, &manifest)
        }
        None if require_manifest => Err(io_invalid(
            "marked artifact bundle is missing its integrity manifest",
        )),
        None => copy_asset_tree(source, destination, &mut AssetCopyBudget::default()),
    }
}

fn copy_manifested_member(
    source_assets: &Path,
    destination_assets: &Path,
    relative: &Path,
    expected: &BundleFileRecord,
) -> io::Result<()> {
    if expected.size_bytes > MAX_BUNDLE_MEMBER_BYTES {
        return Err(io_invalid("artifact bundle member exceeds its size limit"));
    }
    let source = checked_bundle_member(source_assets, relative)?;
    let destination = safe_artifact_path(destination_assets, relative)?;
    if let Some(parent) = destination.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let reader = std::fs::File::open(source)?;
    let mut reader = reader.take(expected.size_bytes.saturating_add(1));
    let mut writer = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(destination)?;
    let mut digest = Sha256::new();
    let mut total = 0_u64;
    let mut buffer = [0_u8; 64 * 1024];
    loop {
        let read = reader.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        total = total
            .checked_add(read as u64)
            .ok_or_else(|| io_invalid("artifact bundle member size overflow"))?;
        if total > expected.size_bytes {
            return Err(io_invalid("artifact bundle member changed during copy"));
        }
        digest.update(&buffer[..read]);
        writer.write_all(&buffer[..read])?;
    }
    let actual_digest = digest
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>();
    if total != expected.size_bytes || actual_digest != expected.sha256 {
        return Err(io_invalid("artifact bundle member changed during copy"));
    }
    writer.sync_all()?;
    Ok(())
}

fn copy_optional_mutable_member(
    source_assets: &Path,
    destination_assets: &Path,
    name: &str,
    maximum_bytes: u64,
) -> io::Result<()> {
    let source = source_assets.join(name);
    match std::fs::symlink_metadata(&source) {
        Ok(metadata) if metadata.is_file() && !metadata.file_type().is_symlink() => {}
        Ok(_) => {
            return Err(io_invalid(format!(
                "mutable bundle member is not a file: {name}"
            )));
        }
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(()),
        Err(error) => return Err(error),
    }
    let bytes = read_bounded_file(&source, maximum_bytes, "mutable bundle metadata")?;
    std::fs::write(destination_assets.join(name), bytes)
}

fn copy_manifested_asset_tree(
    source_assets: &Path,
    destination_assets: &Path,
    manifest: &BundleManifest,
    manifest_bytes: &[u8],
) -> io::Result<()> {
    let metadata = std::fs::symlink_metadata(source_assets)?;
    if !metadata.is_dir() || metadata.file_type().is_symlink() {
        return Err(io_invalid(
            "source artifact support path is not a regular directory",
        ));
    }
    std::fs::create_dir_all(destination_assets)?;
    for (name, record) in &manifest.files {
        let relative = PathBuf::from(name);
        autoeq_artifacts::validate_relative_artifact_path(&relative)?;
        copy_manifested_member(source_assets, destination_assets, &relative, record)?;
    }
    for (name, maximum_bytes) in [
        (RUN_MANIFEST_FILENAME, MAX_RUN_MANIFEST_BYTES),
        (RUN_LOG_FILENAME, MAX_RUN_LOG_BYTES),
    ] {
        copy_optional_mutable_member(source_assets, destination_assets, name, maximum_bytes)?;
    }
    write_file_atomically(
        &destination_assets.join(ARTIFACT_BUNDLE_MANIFEST_FILENAME),
        manifest_bytes,
    )
}

fn collect_previous_measurement_files(index: &serde_json::Value) -> io::Result<Vec<PathBuf>> {
    let mut files = Vec::new();
    for key in [
        "channels",
        "deployed_source_curves",
        "deployed_source_curve_metadata",
    ] {
        if let Some(entries) = index.get(key).and_then(serde_json::Value::as_object) {
            for value in entries.values() {
                if let Some(fields) = value.as_object() {
                    for filename in fields.values().filter_map(serde_json::Value::as_str) {
                        let path = PathBuf::from(filename);
                        autoeq_artifacts::validate_relative_artifact_path(&path)?;
                        files.push(path);
                    }
                } else if let Some(filename) = value.as_str() {
                    let path = PathBuf::from(filename);
                    autoeq_artifacts::validate_relative_artifact_path(&path)?;
                    files.push(path);
                }
            }
        }
    }
    Ok(files)
}

fn clear_previous_measurement_overlay(assets: &Path) -> io::Result<()> {
    let index_path = assets.join(MEASUREMENTS_INDEX_FILENAME);
    let previous_index = match std::fs::read(&index_path) {
        Ok(bytes) => Some(bytes),
        Err(error) if error.kind() == io::ErrorKind::NotFound => None,
        Err(error) => return Err(error),
    };
    if let Some(bytes) = previous_index {
        let index: serde_json::Value = serde_json::from_slice(&bytes)
            .map_err(|error| io_invalid(format!("invalid previous measurement index: {error}")))?;
        for relative in collect_previous_measurement_files(&index)? {
            let member = safe_artifact_path(assets, &relative)?;
            match std::fs::symlink_metadata(&member) {
                Ok(metadata) if metadata.file_type().is_symlink() || !metadata.is_file() => {
                    return Err(io_invalid(format!(
                        "previous measurement member is not a regular file: {}",
                        member.display()
                    )));
                }
                Ok(_) => std::fs::remove_file(member)?,
                Err(error) if error.kind() == io::ErrorKind::NotFound => {}
                Err(error) => return Err(error),
            }
        }
    }
    match std::fs::symlink_metadata(&index_path) {
        Ok(metadata) if metadata.file_type().is_symlink() || !metadata.is_file() => {
            Err(io_invalid(format!(
                "previous measurement index is not a regular file: {}",
                index_path.display()
            )))
        }
        Ok(_) => std::fs::remove_file(index_path),
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(error),
    }
}

fn has_inline_measurement_data(output: &DspGraph) -> bool {
    !output.deployed_source_curves.is_empty()
        || output.channels.values().any(|channel| {
            channel.initial_curve.is_some()
                || channel.final_curve.is_some()
                || channel.eq_response.is_some()
                || channel.target_curve.is_some()
                || channel.pre_ir.is_some()
                || channel.post_ir.is_some()
                || channel.early_late_curves.is_some()
                || channel.waterfall.is_some()
                || channel.resonance_decays.is_some()
                || channel.wavelet.is_some()
                || channel.drivers.as_ref().is_some_and(|drivers| {
                    drivers.iter().any(|driver| {
                        driver.measured_acoustics.is_some() || driver.initial_curve.is_some()
                    })
                })
        })
}

fn validate_extraction_complete(
    output: &DspGraph,
    extracted: &ExtractedMeasurementFiles,
) -> io::Result<()> {
    if has_inline_measurement_data(output)
        || (extracted.index != serde_json::Value::Null
            && !extracted
                .files
                .iter()
                .any(|file| file == MEASUREMENTS_INDEX_FILENAME))
    {
        return Err(io_invalid(
            "failed to extract every measurement asset; the prior output bundle is unchanged",
        ));
    }
    let mut unique = std::collections::HashSet::new();
    if extracted.files.iter().any(|file| !unique.insert(file)) {
        return Err(io_invalid(
            "measurement asset names collide after sanitization",
        ));
    }
    Ok(())
}

fn rewrite_convolution_plugin_resources(
    plugins: &mut [roomeq_model::PluginConfigWrapper],
    rewritten: &HashMap<String, String>,
) -> io::Result<()> {
    for plugin in plugins {
        if plugin.plugin_type != "convolution" {
            continue;
        }
        let value = plugin
            .parameters
            .get("ir_file")
            .and_then(serde_json::Value::as_str)
            .ok_or_else(|| io_invalid("convolution stage requires string field 'ir_file'"))?
            .to_owned();
        let replacement = rewritten.get(&value).ok_or_else(|| {
            io_invalid(format!(
                "convolution resource was not staged for reference '{value}'"
            ))
        })?;
        plugin.parameters["ir_file"] = serde_json::Value::String(replacement.clone());
    }
    Ok(())
}

fn rewrite_convolution_resource_references(
    output: &mut DspGraph,
    rewritten: &HashMap<String, String>,
) -> io::Result<()> {
    rewrite_convolution_plugin_resources(&mut output.global_plugins, rewritten)?;
    for chain in output.channels.values_mut() {
        rewrite_convolution_plugin_resources(&mut chain.plugins, rewritten)?;
        if let Some(drivers) = chain.drivers.as_mut() {
            for driver in drivers {
                rewrite_convolution_plugin_resources(&mut driver.plugins, rewritten)?;
            }
        }
    }
    Ok(())
}

fn source_convolution_path(reference: &str, source_assets: &Path) -> PathBuf {
    let reference = Path::new(reference);
    if reference.is_absolute() {
        reference.to_path_buf()
    } else {
        source_assets.join(reference)
    }
}

fn bind_convolution_resources(
    output: &mut DspGraph,
    source_assets: &Path,
    staged_assets: &Path,
) -> io::Result<Vec<String>> {
    let references = checked_convolution_resource_references(output)
        .map_err(|error| io_invalid(format!("invalid convolution references: {error}")))?;
    if references.is_empty() {
        return Ok(Vec::new());
    }
    let inventory = output
        .metadata
        .as_ref()
        .and_then(|metadata| metadata.final_convolution_sha256.as_ref())
        .ok_or_else(|| {
            io_invalid("convolution resources are not bound by final_convolution_sha256")
        })?;
    if inventory.len() != references.len() {
        return Err(io_invalid(
            "final convolution inventory does not exactly match graph references",
        ));
    }

    let mut rewritten = HashMap::new();
    let mut next_inventory = BTreeMap::new();
    let mut files = Vec::new();
    for reference in references {
        let expected = inventory
            .get(&reference)
            .and_then(Option::as_ref)
            .ok_or_else(|| {
                io_invalid(format!(
                    "convolution resource '{reference}' has no final SHA-256 binding"
                ))
            })?;
        let source_path = source_convolution_path(&reference, source_assets);
        let metadata = std::fs::symlink_metadata(&source_path).map_err(|error| {
            io::Error::new(
                error.kind(),
                format!(
                    "cannot read graph-bound convolution resource '{reference}' at '{}': {error}",
                    source_path.display()
                ),
            )
        })?;
        if metadata.file_type().is_symlink() || !metadata.is_file() {
            return Err(io_invalid(format!(
                "convolution resource '{reference}' is not a regular file"
            )));
        }
        let bytes = std::fs::read(&source_path)?;
        let digest = sha256_hex(&bytes);
        if &digest != expected {
            return Err(io_invalid(format!(
                "convolution resource '{reference}' failed final SHA-256 validation"
            )));
        }
        let rewritten_reference = format!("resources/{digest}.wav");
        let relative = PathBuf::from(&rewritten_reference);
        let destination = safe_artifact_path(staged_assets, &relative)?;
        if let Some(parent) = destination.parent() {
            std::fs::create_dir_all(parent)?;
        }
        match std::fs::read(&destination) {
            Ok(existing) if existing == bytes => {}
            Ok(_) => {
                return Err(io_invalid(format!(
                    "staged convolution resource content conflicts at '{rewritten_reference}'"
                )));
            }
            Err(error) if error.kind() == io::ErrorKind::NotFound => {
                write_file_atomically(&destination, &bytes)?;
            }
            Err(error) => return Err(error),
        }
        rewritten.insert(reference, rewritten_reference.clone());
        next_inventory.insert(rewritten_reference.clone(), Some(digest));
        files.push(rewritten_reference);
    }
    rewrite_convolution_resource_references(output, &rewritten)?;
    if let Some(metadata) = output.metadata.as_mut() {
        metadata.final_convolution_sha256 = Some(next_inventory);
    } else {
        return Err(io_invalid(
            "convolution resources require optimization metadata for final hashes",
        ));
    }
    files.sort();
    files.dedup();
    Ok(files)
}

fn validate_manifested_convolution_resources(
    output: &DspGraph,
    assets: &Path,
    manifest: Option<&BundleManifest>,
    captured_files: Option<&CapturedBundleFiles>,
) -> io::Result<()> {
    let references = checked_convolution_resource_references(output)
        .map_err(|error| io_invalid(format!("invalid convolution references: {error}")))?;
    let Some(manifest) = manifest else {
        // Legacy native outputs are allowed to reference external FIR files.
        return Ok(());
    };
    if references.is_empty() {
        if output
            .metadata
            .as_ref()
            .and_then(|metadata| metadata.final_convolution_sha256.as_ref())
            .is_some_and(|inventory| !inventory.is_empty())
        {
            return Err(io_invalid(
                "bundle convolution inventory contains unreferenced resources",
            ));
        }
        return Ok(());
    }
    let inventory = output
        .metadata
        .as_ref()
        .and_then(|metadata| metadata.final_convolution_sha256.as_ref())
        .ok_or_else(|| {
            io_invalid("manifested convolution resources lack final SHA-256 bindings")
        })?;
    if inventory.len() != references.len() {
        return Err(io_invalid(
            "manifested convolution inventory does not exactly match graph references",
        ));
    }
    for reference in references {
        let relative = PathBuf::from(&reference);
        autoeq_artifacts::validate_relative_artifact_path(&relative)?;
        let key = reference.replace('\\', "/");
        let record = manifest.files.get(&key).ok_or_else(|| {
            io_invalid(format!(
                "manifested graph convolution reference is not a bundled file: {reference}"
            ))
        })?;
        let expected = inventory
            .get(&reference)
            .and_then(Option::as_ref)
            .ok_or_else(|| {
                io_invalid(format!(
                    "manifested graph convolution reference has no SHA-256 binding: {reference}"
                ))
            })?;
        if expected != &record.sha256 {
            return Err(io_invalid(format!(
                "manifested convolution reference and final SHA-256 inventory disagree: {reference}"
            )));
        }
        let bytes = read_bundle_member_bytes(
            assets,
            &relative,
            Some(manifest),
            captured_files,
            MAX_BUNDLE_MEMBER_BYTES,
            "convolution resource",
        )?;
        if sha256_hex(bytes.as_ref()) != *expected {
            return Err(io_invalid(format!(
                "manifested convolution resource failed integrity validation: {reference}"
            )));
        }
    }
    Ok(())
}

fn collect_bundle_files(
    root: &Path,
    current: &Path,
    files: &mut BTreeMap<String, BundleFileRecord>,
) -> io::Result<()> {
    for entry in std::fs::read_dir(current)? {
        let entry = entry?;
        let file_type = entry.file_type()?;
        if file_type.is_symlink() {
            return Err(io_invalid(format!(
                "artifact bundle contains a symlink: {}",
                entry.path().display()
            )));
        }
        if file_type.is_dir() {
            collect_bundle_files(root, &entry.path(), files)?;
        } else if file_type.is_file() {
            let relative = entry
                .path()
                .strip_prefix(root)
                .map_err(|error| io_invalid(error.to_string()))?
                .to_path_buf();
            autoeq_artifacts::validate_relative_artifact_path(&relative)?;
            if relative == Path::new(ARTIFACT_BUNDLE_MANIFEST_FILENAME)
                || relative == Path::new(RUN_MANIFEST_FILENAME)
                || relative == Path::new(RUN_LOG_FILENAME)
            {
                continue;
            }
            let (size_bytes, sha256) = sha256_file(&entry.path())?;
            let name = relative
                .to_str()
                .ok_or_else(|| io_invalid("artifact member path is not valid UTF-8"))?
                .replace('\\', "/");
            files.insert(name, BundleFileRecord { size_bytes, sha256 });
        } else {
            return Err(io_invalid(format!(
                "artifact bundle contains a non-file entry: {}",
                entry.path().display()
            )));
        }
    }
    Ok(())
}

fn write_bundle_manifest(
    assets: &Path,
    generation: &str,
    graph_schema_version: &str,
    graph_sha256: &str,
) -> io::Result<()> {
    let mut files = BTreeMap::new();
    collect_bundle_files(assets, assets, &mut files)?;
    if files.len() > MAX_BUNDLE_MANIFEST_FILES {
        return Err(io_invalid("artifact bundle has too many files"));
    }
    let mut folded_names = BTreeSet::new();
    let mut total_bytes = 0_u64;
    for name in files.keys() {
        if name.len() > MAX_BUNDLE_MEMBER_PATH_BYTES || !folded_names.insert(name.to_lowercase()) {
            return Err(io_invalid(format!(
                "artifact bundle has an invalid or case-insensitive duplicate path: {name}"
            )));
        }
        let record = files
            .get(name)
            .ok_or_else(|| io_invalid("artifact member record disappeared"))?;
        total_bytes = total_bytes
            .checked_add(record.size_bytes)
            .ok_or_else(|| io_invalid("artifact bundle total size overflow"))?;
        if total_bytes > MAX_BUNDLE_TOTAL_BYTES {
            return Err(io_invalid("artifact bundle exceeds its total size limit"));
        }
    }
    let manifest = BundleManifest {
        schema_version: BUNDLE_MANIFEST_SCHEMA_VERSION,
        producer: "roomeq-workflow".to_owned(),
        producer_version: env!("CARGO_PKG_VERSION").to_owned(),
        graph_schema_version: graph_schema_version.to_owned(),
        generation: generation.to_owned(),
        graph_sha256: graph_sha256.to_owned(),
        files,
    };
    let bytes = serde_json::to_vec_pretty(&manifest).map_err(io::Error::other)?;
    if bytes.len() as u64 > MAX_BUNDLE_MANIFEST_BYTES {
        return Err(io_invalid(
            "artifact bundle manifest exceeds its size limit",
        ));
    }
    write_file_atomically(&assets.join(ARTIFACT_BUNDLE_MANIFEST_FILENAME), &bytes)
}

fn load_manifest(assets: &Path) -> io::Result<Option<BundleManifest>> {
    let path = assets.join(ARTIFACT_BUNDLE_MANIFEST_FILENAME);
    let metadata = match std::fs::symlink_metadata(&path) {
        Ok(metadata) if metadata.is_file() && !metadata.file_type().is_symlink() => metadata,
        Ok(_) => return Err(io_invalid("artifact bundle manifest is not a regular file")),
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(error),
    };
    if metadata.len() > MAX_BUNDLE_MANIFEST_BYTES {
        return Err(io_invalid(
            "artifact bundle manifest exceeds its size limit",
        ));
    }
    let bytes = read_bounded_file(&path, MAX_BUNDLE_MANIFEST_BYTES, "artifact bundle manifest")?;
    let manifest: BundleManifest = serde_json::from_slice(&bytes)
        .map_err(|error| io_invalid(format!("invalid artifact bundle manifest: {error}")))?;
    if manifest.schema_version != BUNDLE_MANIFEST_SCHEMA_VERSION {
        return Err(io_invalid(format!(
            "unsupported artifact bundle manifest schema {}",
            manifest.schema_version
        )));
    }
    if manifest.producer != "roomeq-workflow"
        || manifest.producer_version.is_empty()
        || manifest.graph_schema_version.is_empty()
        || manifest.generation.is_empty()
        || !is_sha256_hex(&manifest.graph_sha256)
    {
        return Err(io_invalid(
            "artifact bundle manifest has invalid provenance metadata",
        ));
    }
    if manifest.files.len() > MAX_BUNDLE_MANIFEST_FILES {
        return Err(io_invalid("artifact bundle manifest has too many files"));
    }
    let mut folded_names = BTreeSet::new();
    let mut total_member_bytes = 0_u64;
    for (name, expected) in &manifest.files {
        if name.len() > MAX_BUNDLE_MEMBER_PATH_BYTES
            || !folded_names.insert(name.to_lowercase())
            || !is_sha256_hex(&expected.sha256)
        {
            return Err(io_invalid(format!(
                "artifact bundle manifest has an invalid or duplicate file entry: {name}"
            )));
        }
        let relative = PathBuf::from(name);
        autoeq_artifacts::validate_relative_artifact_path(&relative)?;
        let path = checked_bundle_member(assets, &relative)?;
        let metadata = std::fs::symlink_metadata(&path)?;
        if metadata.file_type().is_symlink() || !metadata.is_file() {
            return Err(io_invalid(format!(
                "artifact bundle member is not a regular file: {}",
                path.display()
            )));
        }
        if metadata.len() != expected.size_bytes {
            return Err(io_invalid(format!(
                "artifact bundle member failed integrity validation: {name}"
            )));
        }
        if expected.size_bytes > MAX_BUNDLE_MEMBER_BYTES {
            return Err(io_invalid(format!(
                "artifact bundle member exceeds its size limit: {name}"
            )));
        }
        total_member_bytes = total_member_bytes
            .checked_add(expected.size_bytes)
            .ok_or_else(|| io_invalid("artifact bundle total size overflow"))?;
        if total_member_bytes > MAX_BUNDLE_TOTAL_BYTES {
            return Err(io_invalid("artifact bundle exceeds its total size limit"));
        }
        let (actual_size, actual_digest) = sha256_file_bounded(&path, expected.size_bytes)?;
        if actual_size != expected.size_bytes || actual_digest != expected.sha256 {
            return Err(io_invalid(format!(
                "artifact bundle member failed integrity validation: {name}"
            )));
        }
    }
    Ok(Some(manifest))
}

fn capture_manifested_bundle_files(
    assets: &Path,
    manifest: &BundleManifest,
) -> io::Result<CapturedBundleFiles> {
    let mut captured = BTreeMap::new();
    for (name, expected) in &manifest.files {
        let relative = PathBuf::from(name);
        let path = checked_bundle_member(assets, &relative)?;
        let metadata = std::fs::symlink_metadata(&path)?;
        if metadata.file_type().is_symlink() || !metadata.is_file() {
            return Err(io_invalid(format!(
                "artifact bundle member is not a regular file: {name}"
            )));
        }
        if expected.size_bytes > MAX_BUNDLE_MEMBER_BYTES {
            return Err(io_invalid(format!(
                "artifact bundle member exceeds its size limit: {name}"
            )));
        }
        let file = std::fs::File::open(&path)?;
        let mut bytes = Vec::with_capacity(expected.size_bytes as usize);
        file.take(expected.size_bytes.saturating_add(1))
            .read_to_end(&mut bytes)?;
        if bytes.len() as u64 != expected.size_bytes || sha256_hex(&bytes) != expected.sha256 {
            return Err(io_invalid(format!(
                "{} failed artifact bundle integrity validation",
                manifested_member_label(name)
            )));
        }
        captured.insert(name.clone(), Arc::<[u8]>::from(bytes));
    }
    Ok(captured)
}

fn manifested_member_label(name: &str) -> &'static str {
    if name == MEASUREMENTS_INDEX_FILENAME {
        "measurement index"
    } else if name.ends_with(".csv.json") {
        "curve metadata sidecar"
    } else if name.ends_with(".csv") {
        "curve CSV sidecar"
    } else {
        "bundle sidecar"
    }
}

fn transaction_directory(parent: &Path, name: &str) -> io::Result<PathBuf> {
    let relative = Path::new(name);
    autoeq_artifacts::validate_relative_artifact_path(relative)?;
    if relative.components().count() != 1 {
        return Err(io_invalid(
            "transaction directory must be one path component",
        ));
    }
    Ok(parent.join(relative))
}

fn remove_if_exists(path: &Path) -> io::Result<()> {
    match std::fs::symlink_metadata(path) {
        Ok(metadata) if metadata.is_dir() && !metadata.file_type().is_symlink() => {
            std::fs::remove_dir_all(path)
        }
        Ok(metadata) if metadata.is_file() && !metadata.file_type().is_symlink() => {
            std::fs::remove_file(path)
        }
        Ok(_) => Err(io_invalid(format!(
            "transaction path is not a regular file or directory: {}",
            path.display()
        ))),
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(error),
    }
}

fn rollback_transaction(
    output_path: &Path,
    assets: &Path,
    transaction_root: &Path,
    journal: &BundleTransactionJournal,
) -> io::Result<()> {
    let previous_assets = transaction_root.join("previous_assets");
    let current_assets_generation = load_manifest(assets)?.map(|manifest| manifest.generation);
    if previous_assets.exists() {
        if assets.exists() {
            if current_assets_generation.as_deref() != Some(journal.generation.as_str()) {
                return Err(io_invalid(
                    "cannot recover bundle transaction: current support directory is unknown",
                ));
            }
            remove_if_exists(assets)?;
        }
        std::fs::rename(&previous_assets, assets)?;
    } else if !journal.had_assets {
        if current_assets_generation.as_deref() == Some(journal.generation.as_str()) {
            remove_if_exists(assets)?;
        } else if assets.exists() {
            return Err(io_invalid(
                "cannot recover bundle transaction: unexpected support directory",
            ));
        }
    } else if !assets.exists()
        || current_assets_generation.as_deref() == Some(journal.generation.as_str())
    {
        return Err(io_invalid(
            "cannot recover bundle transaction: previous support directory is missing",
        ));
    }

    let output_bytes = match std::fs::read(output_path) {
        Ok(bytes) => Some(bytes),
        Err(error) if error.kind() == io::ErrorKind::NotFound => None,
        Err(error) => return Err(error),
    };
    let output_hash = output_bytes.as_deref().map(sha256_hex);
    if output_hash.as_deref() == Some(journal.new_output_sha256.as_str()) {
        if journal.had_output {
            let old_output = transaction_root.join("previous_output.json");
            let bytes = std::fs::read(old_output)?;
            if journal.old_output_sha256.as_deref() != Some(sha256_hex(&bytes).as_str()) {
                return Err(io_invalid(
                    "cannot recover bundle transaction: previous output backup failed integrity validation",
                ));
            }
            write_file_atomically(output_path, &bytes)?;
        } else {
            remove_if_exists(output_path)?;
        }
    } else if output_hash.as_deref() != journal.old_output_sha256.as_deref() {
        return Err(io_invalid(
            "cannot recover bundle transaction: output JSON changed independently",
        ));
    }
    Ok(())
}

/// Recover an interrupted publication before any loader returns bundle data.
///
/// A matching new root JSON and verified support manifest completes cleanup.
/// Otherwise, the prior root and support directory are restored. If the
/// journal or paths are inconsistent, loading fails without returning a
/// partial graph.
fn recover_pending_bundle(output_path: &Path) -> io::Result<()> {
    let journal_path = bundle_transaction_path(output_path);
    let bytes = match read_bounded_file(
        &journal_path,
        MAX_BUNDLE_JOURNAL_BYTES,
        "bundle transaction journal",
    ) {
        Ok(bytes) => bytes,
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(()),
        Err(error) => return Err(error),
    };
    let journal: BundleTransactionJournal = serde_json::from_slice(&bytes)
        .map_err(|error| io_invalid(format!("invalid bundle transaction journal: {error}")))?;
    if journal.schema_version != BUNDLE_MANIFEST_SCHEMA_VERSION {
        return Err(io_invalid(format!(
            "unsupported bundle transaction schema {}",
            journal.schema_version
        )));
    }
    if journal.had_output != journal.old_output_sha256.is_some() {
        return Err(io_invalid(
            "bundle transaction journal has inconsistent prior-output metadata",
        ));
    }
    if !is_sha256_hex(&journal.new_output_sha256)
        || journal
            .old_output_sha256
            .as_deref()
            .is_some_and(|digest| !is_sha256_hex(digest))
    {
        return Err(io_invalid(
            "bundle transaction journal has invalid output digests",
        ));
    }
    let parent = output_path
        .parent()
        .filter(|path| !path.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    let transaction_root = transaction_directory(parent, &journal.transaction_directory)?;
    let assets = assets_dir_for(output_path);
    let current_output = match std::fs::read(output_path) {
        Ok(bytes) => Some(bytes),
        Err(error) if error.kind() == io::ErrorKind::NotFound => None,
        Err(error) => return Err(error),
    };
    let current_hash = current_output.as_deref().map(sha256_hex);
    let root_is_new = current_hash.as_deref() == Some(journal.new_output_sha256.as_str());
    let committed_assets = load_manifest(&assets)
        .ok()
        .flatten()
        .is_some_and(|manifest| {
            manifest.generation == journal.generation
                && manifest.graph_sha256 == journal.new_output_sha256
        });
    if root_is_new && committed_assets {
        if remove_if_exists(&transaction_root).is_ok() && remove_if_exists(&journal_path).is_ok() {
            let _ = sync_directory(parent);
        }
        return Ok(());
    }
    if current_hash.as_deref() != journal.old_output_sha256.as_deref() && !root_is_new {
        return Err(io_invalid(
            "cannot recover bundle transaction: output JSON changed independently",
        ));
    }
    if root_is_new && journal.had_output {
        let expected = journal.old_output_sha256.as_deref().ok_or_else(|| {
            io_invalid("cannot recover bundle transaction: prior output hash is missing")
        })?;
        let backup = std::fs::read(transaction_root.join("previous_output.json"))?;
        if sha256_hex(&backup) != expected {
            return Err(io_invalid(
                "cannot recover bundle transaction: prior output backup failed integrity validation",
            ));
        }
    }
    rollback_transaction(output_path, &assets, &transaction_root, &journal)?;
    remove_if_exists(&transaction_root)?;
    remove_if_exists(&journal_path)?;
    sync_directory(parent)?;
    Ok(())
}

/// Recover pending native-bundle and coupled external-export publications.
///
/// The native bundle is resolved first because its root digest decides whether
/// the external package should be rolled back or completed.
pub fn recover_output_bundle_transactions(output_path: &Path) -> io::Result<()> {
    recover_pending_bundle(output_path)?;
    crate::export::recover_pending_external_export_transaction(output_path)
        .map_err(|error| io_invalid(format!("external export recovery failed: {error:#}")))
}

/// Return the bounded SHA-256 identity of a native root file, or `None` when
/// the root does not exist.
pub fn native_output_sha256(output_path: &Path) -> io::Result<Option<String>> {
    let bytes = match read_bounded_file(output_path, MAX_NATIVE_GRAPH_BYTES, "native output graph")
    {
        Ok(bytes) => bytes,
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(error),
    };
    Ok(Some(sha256_hex(&bytes)))
}

/// Save a DSP output as a small JSON plus sibling assets directory.
///
/// Publication uses unique staging, a durable journal, and atomic root JSON
/// replacement. Interrupted writes are recovered before a native loader
/// returns. The root JSON remains the binding referenced by existing
/// consumers; support assets are swapped at the canonical stem-files path.
/// Concurrent readers are not isolated from the short directory-swap window.
pub fn save_output_bundle(
    output: &mut DspGraph,
    output_path: &Path,
) -> Result<Vec<String>, Box<dyn std::error::Error>> {
    recover_output_bundle_transactions(output_path)?;
    save_output_bundle_using(
        output,
        output_path,
        &mut |from: &Path, to: &Path| std::fs::rename(from, to),
        &mut |path: &Path, bytes: &[u8]| write_file_atomically(path, bytes),
    )
}

/// Save a self-contained native bundle, binding every convolution resource
/// into the sibling support directory before publication.
///
/// The graph must carry a matching `final_convolution_sha256` entry for each
/// declared `ir_file`. Resource references are rewritten to portable,
/// content-addressed paths before the root JSON and support files are
/// published. Use this API for graphs that contain convolution stages.
///
/// # Errors
/// Returns an error when a referenced asset is unavailable or does not match
/// its final graph-bound digest, or when the bundle cannot be published.
pub fn save_output_bundle_with_resources(
    output: &mut DspGraph,
    output_path: &Path,
    source_assets: &Path,
) -> Result<Vec<String>, Box<dyn std::error::Error>> {
    save_output_bundle_with_resources_and_prepare(
        output,
        output_path,
        source_assets,
        &mut |_, _| Ok(()),
    )
}

/// Resource-aware bundle save with a callback that runs on the finalized
/// staged graph immediately before its bytes are manifested and published.
///
/// The callback runs after measurement extraction and convolution-reference
/// rewriting, so graph-bound evidence can be finalized against exactly the
/// bytes that will be committed. Returning an error leaves the prior root and
/// support directory unchanged.
///
/// # Errors
/// Returns an error when resource binding, callback validation, or bundle
/// publication fails.
pub fn save_output_bundle_with_resources_and_prepare(
    output: &mut DspGraph,
    output_path: &Path,
    source_assets: &Path,
    prepare: &mut impl FnMut(&mut DspGraph, &Path) -> Result<(), Box<dyn std::error::Error>>,
) -> Result<Vec<String>, Box<dyn std::error::Error>> {
    recover_output_bundle_transactions(output_path)?;
    let mut rename = |from: &Path, to: &Path| std::fs::rename(from, to);
    let mut write_root = |path: &Path, bytes: &[u8]| write_file_atomically(path, bytes);
    let mut write_journal = |path: &Path, bytes: &[u8]| write_file_atomically(path, bytes);
    let mut sync_parent = |path: &Path| sync_directory(path);
    let mut hooks = BundleHooks {
        prepare,
        rename: &mut rename,
        write_root: &mut write_root,
        write_journal: &mut write_journal,
        sync_parent: &mut sync_parent,
    };
    save_output_bundle_using_hooks_and_assets(
        output,
        output_path,
        Some(source_assets),
        true,
        &mut hooks,
    )
}

/// Publish an already validated native bundle to a new root path without
/// changing its graph bytes or ledger identity.
///
/// The source bundle must have a valid integrity manifest. Its support tree
/// and exact root JSON bytes are staged and installed with the same recovery
/// journal used by normal bundle saves.
///
/// # Errors
/// Returns an error when the source bundle is invalid or the destination
/// transaction cannot complete.
pub fn publish_output_bundle_from(
    source_output_path: &Path,
    destination_output_path: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    recover_output_bundle_transactions(source_output_path)?;
    recover_output_bundle_transactions(destination_output_path)?;
    publish_output_bundle_from_with_hook(source_output_path, destination_output_path, |_, _| Ok(()))
}

/// Publish the native half of a coupled transaction without first resolving
/// its external journal. The caller must already have resolved any previous
/// publication and durably recorded the new external intent.
///
/// This API is intended for the workflow export adapter. Ordinary consumers
/// should call [`publish_output_bundle_from`], which recovers both journals.
pub fn publish_output_bundle_from_during_external_transaction_with_source_recovery(
    source_output_path: &Path,
    destination_output_path: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    recover_output_bundle_transactions(source_output_path)?;
    recover_pending_bundle(destination_output_path)?;
    crate::export::validate_pending_external_export_source(
        destination_output_path,
        source_output_path,
    )?;
    publish_output_bundle_from_with_hook(source_output_path, destination_output_path, |_, _| Ok(()))
}

fn publish_output_bundle_from_with_hook(
    source_output_path: &Path,
    destination_output_path: &Path,
    after_source_validation: impl FnOnce(&Path, &Path) -> io::Result<()>,
) -> Result<(), Box<dyn std::error::Error>> {
    recover_pending_bundle(source_output_path)?;
    if source_output_path == destination_output_path {
        let _ = load_output_bundle(source_output_path)?;
        return Ok(());
    }
    let root_bytes = read_bounded_file(
        source_output_path,
        MAX_NATIVE_GRAPH_BYTES,
        "source native output graph",
    )?;
    let source_assets = assets_dir_for(source_output_path);
    let source_manifest_path = source_assets.join(ARTIFACT_BUNDLE_MANIFEST_FILENAME);
    let source_manifest_bytes = read_bounded_file(
        &source_manifest_path,
        MAX_BUNDLE_MANIFEST_BYTES,
        "source artifact bundle manifest",
    )?;
    let source_manifest = load_manifest(&source_assets)?
        .ok_or_else(|| io_invalid("source bundle has no integrity manifest"))?;
    if read_bounded_file(
        &source_manifest_path,
        MAX_BUNDLE_MANIFEST_BYTES,
        "source artifact bundle manifest",
    )? != source_manifest_bytes
    {
        return Err(io_invalid("source artifact bundle manifest changed during validation").into());
    }
    if source_manifest.graph_sha256 != sha256_hex(&root_bytes) {
        return Err(io_invalid("source graph does not match its artifact bundle manifest").into());
    }
    let graph: DspGraph = serde_json::from_slice(&root_bytes)?;
    let _ = load_output_bundle(source_output_path)?;
    if read_bounded_file(
        source_output_path,
        MAX_NATIVE_GRAPH_BYTES,
        "source native output graph",
    )? != root_bytes
        || read_bounded_file(
            &source_manifest_path,
            MAX_BUNDLE_MANIFEST_BYTES,
            "source artifact bundle manifest",
        )? != source_manifest_bytes
    {
        return Err(io_invalid("source bundle changed during validation").into());
    }
    after_source_validation(source_output_path, &source_assets)?;
    if read_bounded_file(
        source_output_path,
        MAX_NATIVE_GRAPH_BYTES,
        "source native output graph",
    )? != root_bytes
    {
        return Err(io_invalid("source graph changed after bundle validation").into());
    }
    if read_bounded_file(
        &source_manifest_path,
        MAX_BUNDLE_MANIFEST_BYTES,
        "source artifact bundle manifest",
    )? != source_manifest_bytes
    {
        return Err(io_invalid("source manifest changed after bundle validation").into());
    }
    verify_bundle_tree_matches_manifest(&source_assets, &source_manifest)?;

    recover_pending_bundle(destination_output_path)?;
    let parent = destination_output_path
        .parent()
        .filter(|path| !path.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    std::fs::create_dir_all(parent)?;
    let had_assets = match std::fs::symlink_metadata(assets_dir_for(destination_output_path)) {
        Ok(metadata) if metadata.is_dir() && !metadata.file_type().is_symlink() => true,
        Ok(_) => {
            return Err(io_invalid("destination support path is not a regular directory").into());
        }
        Err(error) if error.kind() == io::ErrorKind::NotFound => false,
        Err(error) => return Err(error.into()),
    };
    let staging = ArtifactBundleStaging::new(parent)?;
    let transaction_root = staging.root().to_path_buf();
    let stage_assets = transaction_root.join("next_assets");
    copy_manifested_asset_tree(
        &source_assets,
        &stage_assets,
        &source_manifest,
        &source_manifest_bytes,
    )?;
    verify_bundle_tree_matches_manifest(&stage_assets, &source_manifest)?;
    let generation = transaction_root
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| io_invalid("transaction directory name is not valid UTF-8"))?
        .to_string();
    write_bundle_manifest(
        &stage_assets,
        &generation,
        &graph.version,
        &sha256_hex(&root_bytes),
    )?;
    let mut prepare = |_: &mut DspGraph, _: &Path| Ok(());
    let mut rename = |from: &Path, to: &Path| std::fs::rename(from, to);
    let mut write_root = |path: &Path, bytes: &[u8]| write_file_atomically(path, bytes);
    let mut write_journal = |path: &Path, bytes: &[u8]| write_file_atomically(path, bytes);
    let mut sync_parent = |path: &Path| sync_directory(path);
    let mut hooks = BundleHooks {
        prepare: &mut prepare,
        rename: &mut rename,
        write_root: &mut write_root,
        write_journal: &mut write_journal,
        sync_parent: &mut sync_parent,
    };
    publish_prepared_bundle(
        PreparedBundle {
            output_path: destination_output_path,
            staging,
            stage_assets: &stage_assets,
            output_bytes: &root_bytes,
            generation,
            had_assets,
        },
        &mut hooks,
    )?;
    Ok(())
}

fn verify_bundle_tree_matches_manifest(assets: &Path, expected: &BundleManifest) -> io::Result<()> {
    let mut copied_files = BTreeMap::new();
    collect_bundle_files(assets, assets, &mut copied_files)?;
    if copied_files != expected.files {
        return Err(io_invalid(
            "copied source support files do not match the source bundle manifest",
        ));
    }
    let copied_manifest = load_manifest(assets)?
        .ok_or_else(|| io_invalid("copied source bundle is missing its integrity manifest"))?;
    if copied_manifest != *expected {
        return Err(io_invalid(
            "copied source bundle manifest differs from the validated source manifest",
        ));
    }
    Ok(())
}

fn save_output_bundle_using(
    output: &mut DspGraph,
    output_path: &Path,
    rename: &mut impl FnMut(&Path, &Path) -> io::Result<()>,
    write_root: &mut impl FnMut(&Path, &[u8]) -> io::Result<()>,
) -> Result<Vec<String>, Box<dyn std::error::Error>> {
    save_output_bundle_using_hooks(
        output,
        output_path,
        rename,
        write_root,
        &mut |path: &Path, bytes: &[u8]| write_file_atomically(path, bytes),
        &mut |path: &Path| sync_directory(path),
    )
}

fn save_output_bundle_using_hooks(
    output: &mut DspGraph,
    output_path: &Path,
    rename: &mut impl FnMut(&Path, &Path) -> io::Result<()>,
    write_root: &mut impl FnMut(&Path, &[u8]) -> io::Result<()>,
    write_journal: &mut impl FnMut(&Path, &[u8]) -> io::Result<()>,
    sync_parent: &mut impl FnMut(&Path) -> io::Result<()>,
) -> Result<Vec<String>, Box<dyn std::error::Error>> {
    let mut prepare = |_: &mut DspGraph, _: &Path| Ok(());
    let mut hooks = BundleHooks {
        prepare: &mut prepare,
        rename,
        write_root,
        write_journal,
        sync_parent,
    };
    save_output_bundle_using_hooks_and_assets(output, output_path, None, false, &mut hooks)
}

fn save_output_bundle_using_hooks_and_assets(
    output: &mut DspGraph,
    output_path: &Path,
    source_assets: Option<&Path>,
    allow_convolution_resources: bool,
    hooks: &mut BundleHooks<'_>,
) -> Result<Vec<String>, Box<dyn std::error::Error>> {
    recover_pending_bundle(output_path)?;
    output.validate().map_err(io_invalid)?;
    let references = checked_convolution_resource_references(output)
        .map_err(|error| io_invalid(format!("invalid convolution references: {error}")))?;
    if !allow_convolution_resources && !references.is_empty() {
        return Err(
            io_invalid("convolution resources require save_output_bundle_with_resources").into(),
        );
    }
    let parent = output_path
        .parent()
        .filter(|path| !path.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    std::fs::create_dir_all(parent)?;
    let assets = assets_dir_for(output_path);
    let had_assets = match std::fs::symlink_metadata(&assets) {
        Ok(metadata) if metadata.is_dir() && !metadata.file_type().is_symlink() => true,
        Ok(_) => {
            return Err(io_invalid(format!(
                "artifact support path is not a regular directory: {}",
                assets.display()
            ))
            .into());
        }
        Err(error) if error.kind() == io::ErrorKind::NotFound => false,
        Err(error) => return Err(error.into()),
    };

    let staging = ArtifactBundleStaging::new(parent)?;
    let transaction_root = staging.root().to_path_buf();
    let stage_assets = transaction_root.join("next_assets");
    std::fs::create_dir(&stage_assets)?;
    let require_source_manifest = output.artifact_bundle_schema_version.is_some();
    if let Some(source_assets) = source_assets {
        match std::fs::symlink_metadata(source_assets) {
            Ok(metadata) if metadata.is_dir() && !metadata.file_type().is_symlink() => {
                copy_existing_asset_tree(
                    source_assets,
                    &stage_assets,
                    require_source_manifest,
                    None,
                )?;
            }
            Ok(_) => {
                return Err(io_invalid(format!(
                    "source artifact support path is not a regular directory: {}",
                    source_assets.display()
                ))
                .into());
            }
            Err(error) if error.kind() == io::ErrorKind::NotFound => {}
            Err(error) => return Err(error.into()),
        }
    } else if had_assets {
        copy_existing_asset_tree(
            &assets,
            &stage_assets,
            require_source_manifest,
            Some(output_path),
        )?;
    }
    clear_previous_measurement_overlay(&stage_assets)?;

    let mut candidate = output.clone();
    candidate.artifact_bundle_schema_version = Some(BUNDLE_MANIFEST_SCHEMA_VERSION);
    let extracted = extract_measurements_to_assets(&mut candidate, &stage_assets);
    validate_extraction_complete(&candidate, &extracted)?;
    let mut files = extracted.files;
    if allow_convolution_resources {
        let source_assets = source_assets.ok_or_else(|| {
            io_invalid("resource-aware bundle save requires a source support directory")
        })?;
        files.extend(bind_convolution_resources(
            &mut candidate,
            source_assets,
            &stage_assets,
        )?);
    }
    candidate.validate().map_err(io_invalid)?;
    (hooks.prepare)(&mut candidate, &stage_assets)?;
    candidate.artifact_bundle_schema_version = Some(BUNDLE_MANIFEST_SCHEMA_VERSION);
    candidate.validate().map_err(io_invalid)?;
    let output_value = serde_json::to_value(&candidate)?;
    let output_bytes = serde_json::to_vec_pretty(&output_value)?;
    let generation = transaction_root
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| io_invalid("transaction directory name is not valid UTF-8"))?
        .to_string();
    write_bundle_manifest(
        &stage_assets,
        &generation,
        &candidate.version,
        &sha256_hex(&output_bytes),
    )?;
    files.push(ARTIFACT_BUNDLE_MANIFEST_FILENAME.to_string());
    files.sort();
    publish_prepared_bundle(
        PreparedBundle {
            output_path,
            staging,
            stage_assets: &stage_assets,
            output_bytes: &output_bytes,
            generation,
            had_assets,
        },
        hooks,
    )?;
    *output = candidate;
    Ok(files)
}

fn publish_prepared_bundle(
    prepared: PreparedBundle<'_>,
    hooks: &mut BundleHooks<'_>,
) -> io::Result<()> {
    let PreparedBundle {
        output_path,
        staging,
        stage_assets,
        output_bytes,
        generation,
        had_assets,
    } = prepared;
    let parent = output_path
        .parent()
        .filter(|path| !path.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    let assets = assets_dir_for(output_path);
    let transaction_root = staging.root().to_path_buf();
    let output_hash = sha256_hex(output_bytes);
    let old_output = match std::fs::read(output_path) {
        Ok(bytes) => Some(bytes),
        Err(error) if error.kind() == io::ErrorKind::NotFound => None,
        Err(error) => return Err(error),
    };
    let old_output_hash = old_output.as_deref().map(sha256_hex);
    if let Some(bytes) = old_output.as_deref() {
        let backup = transaction_root.join("previous_output.json");
        std::fs::write(&backup, bytes)?;
        std::fs::File::open(backup)?.sync_all()?;
        #[cfg(test)]
        pause_after_test_publication_phase("previous_output_backup_synced");
    }
    sync_tree(stage_assets)?;
    #[cfg(test)]
    pause_after_test_publication_phase("stage_assets_synced");
    sync_directory(&transaction_root)?;
    #[cfg(test)]
    pause_after_test_publication_phase("transaction_root_synced");
    let transaction_directory_name = generation.clone();
    let journal = BundleTransactionJournal {
        schema_version: BUNDLE_MANIFEST_SCHEMA_VERSION,
        transaction_directory: transaction_directory_name,
        generation,
        had_assets,
        had_output: old_output.is_some(),
        old_output_sha256: old_output_hash,
        new_output_sha256: output_hash,
    };
    let journal_bytes = serde_json::to_vec_pretty(&journal)?;
    let kept_transaction_root = staging.keep();
    let journal_path = bundle_transaction_path(output_path);
    if let Err(error) = (hooks.write_journal)(&journal_path, &journal_bytes) {
        if std::fs::symlink_metadata(&journal_path)
            .is_err_and(|error| error.kind() == io::ErrorKind::NotFound)
        {
            let _ = remove_if_exists(&kept_transaction_root);
        }
        return Err(error);
    }
    #[cfg(test)]
    pause_after_test_publication_phase("journal_written");
    // The journal may already be visible even when its parent sync fails.
    // Keep the transaction directory and backups so the next loader can
    // recover it; deleting them here would leave a durable dangling journal.
    (hooks.sync_parent)(parent)?;
    #[cfg(test)]
    pause_after_test_publication_phase("journal_parent_synced");

    let transaction = (|| -> io::Result<()> {
        if had_assets {
            (hooks.rename)(&assets, &kept_transaction_root.join("previous_assets"))?;
            #[cfg(test)]
            pause_after_test_publication_phase("previous_assets_renamed");
        }
        (hooks.rename)(stage_assets, &assets)?;
        #[cfg(test)]
        pause_after_test_publication_phase("candidate_assets_installed");
        (hooks.sync_parent)(parent)?;
        #[cfg(test)]
        pause_after_test_publication_phase("assets_parent_synced");
        (hooks.write_root)(output_path, output_bytes)?;
        #[cfg(test)]
        pause_after_test_publication_phase("root_written");
        (hooks.sync_parent)(parent)?;
        #[cfg(test)]
        pause_after_test_publication_phase("root_parent_synced");
        Ok(())
    })();
    if let Err(error) = transaction {
        recover_pending_bundle(output_path)?;
        let committed = std::fs::read(output_path)
            .ok()
            .is_some_and(|bytes| sha256_hex(&bytes) == journal.new_output_sha256)
            && load_manifest(&assets)
                .ok()
                .flatten()
                .is_some_and(|manifest| {
                    manifest.generation == journal.generation
                        && manifest.graph_sha256 == journal.new_output_sha256
                });
        if !committed {
            return Err(error);
        }
    }

    if remove_if_exists(&kept_transaction_root).is_ok() {
        #[cfg(test)]
        pause_after_test_publication_phase("transaction_directory_removed");
        if remove_if_exists(&bundle_transaction_path(output_path)).is_ok() {
            #[cfg(test)]
            pause_after_test_publication_phase("journal_removed");
            if sync_directory(parent).is_ok() {
                #[cfg(test)]
                pause_after_test_publication_phase("cleanup_parent_synced");
            }
        }
    }
    Ok(())
}

fn parse_f64_cell(cell: &str, what: &str) -> Result<f64, Box<dyn std::error::Error>> {
    cell.trim().parse::<f64>().map_err(|error| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("invalid {what} value {cell:?}: {error}"),
        )
        .into()
    })
}

fn read_curve_csv_bytes(
    bytes: &[u8],
    label: &str,
) -> Result<CurveData, Box<dyn std::error::Error>> {
    let text = std::str::from_utf8(bytes)?;
    let mut lines = text.lines();
    let header = lines.next().ok_or_else(|| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("curve CSV '{label}' is empty"),
        )
    })?;
    let columns: Vec<&str> = header.split(',').collect();
    if columns.len() < 2 || columns[0] != "freq" || columns[1] != "spl" {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("curve CSV '{label}' has an unexpected header"),
        )
        .into());
    }
    let has_phase = columns.get(2) == Some(&"phase");
    let mut freq = Vec::new();
    let mut spl = Vec::new();
    let mut phase = Vec::new();
    for line in lines {
        if line.trim().is_empty() {
            continue;
        }
        let cells: Vec<&str> = line.split(',').collect();
        freq.push(parse_f64_cell(
            cells.first().copied().unwrap_or(""),
            "frequency",
        )?);
        spl.push(parse_f64_cell(cells.get(1).copied().unwrap_or(""), "SPL")?);
        if has_phase {
            phase.push(parse_f64_cell(
                cells.get(2).copied().unwrap_or(""),
                "phase",
            )?);
        }
    }
    Ok(CurveData {
        freq,
        spl,
        phase: has_phase.then_some(phase),
        norm_range: None,
        noise_floor_db: None,
        coherence: None,
    })
}

fn indexed_member_path(
    assets: &Path,
    relative_text: &str,
    manifest: Option<&BundleManifest>,
    captured_files: Option<&CapturedBundleFiles>,
) -> io::Result<PathBuf> {
    let relative = PathBuf::from(relative_text);
    autoeq_artifacts::validate_relative_artifact_path(&relative)?;
    if let Some(manifest) = manifest {
        let key = relative
            .to_str()
            .ok_or_else(|| io_invalid("artifact member path is not valid UTF-8"))?
            .replace('\\', "/");
        if !manifest.files.contains_key(&key) {
            return Err(io_invalid(format!(
                "measurement index references unhashed bundle member: {relative_text}"
            )));
        }
        if let Some(captured_files) = captured_files {
            if !captured_files.contains_key(&key) {
                return Err(io_invalid(format!(
                    "measurement index references uncaptured bundle member: {relative_text}"
                )));
            }
            return Ok(assets.join(relative));
        }
        checked_bundle_member(assets, &relative)
    } else {
        safe_artifact_path(assets, &relative)
    }
}

fn indexed_member_is_available(
    assets: &Path,
    relative_text: &str,
    manifest: Option<&BundleManifest>,
    captured_files: Option<&CapturedBundleFiles>,
) -> io::Result<bool> {
    if let Some(manifest) = manifest {
        let key = relative_text.replace('\\', "/");
        if !manifest.files.contains_key(&key) {
            return Err(io_invalid(format!(
                "measurement index references unhashed bundle member: {relative_text}"
            )));
        }
        if let Some(captured_files) = captured_files {
            return Ok(captured_files.contains_key(&key));
        }
    }
    Ok(indexed_member_path(assets, relative_text, manifest, captured_files)?.is_file())
}

fn read_indexed_curve(
    assets: &Path,
    fields: &serde_json::Map<String, serde_json::Value>,
    kind: &str,
    csv_path: &Path,
    manifest: Option<&BundleManifest>,
    captured_files: Option<&CapturedBundleFiles>,
) -> io::Result<CurveData> {
    let has_metadata = fields.contains_key(&format!("{kind}_metadata"));
    let curve = if let Some(metadata) = fields.get(&format!("{kind}_metadata")) {
        let metadata = metadata.as_str().ok_or_else(|| {
            io_invalid(format!("measurement metadata path for '{kind}' is invalid"))
        })?;
        let relative = PathBuf::from(metadata);
        let bytes = read_bundle_member_bytes(
            assets,
            &relative,
            manifest,
            captured_files,
            MAX_BUNDLE_MEMBER_BYTES,
            "curve metadata sidecar",
        )?;
        serde_json::from_slice::<CurveData>(bytes.as_ref())
            .map_err(|error| io_invalid(format!("invalid curve metadata sidecar: {error}")))?
    } else {
        let bytes = read_bundle_path_bytes(
            assets,
            csv_path,
            manifest,
            captured_files,
            MAX_BUNDLE_MEMBER_BYTES,
            "curve CSV sidecar",
        )?;
        read_curve_csv_bytes(bytes.as_ref(), &csv_path.display().to_string())
            .map_err(|error| io_invalid(format!("invalid curve CSV sidecar: {error}")))?
    };
    if manifest.is_some() || has_metadata {
        validate_curve_data(&curve)?;
    }
    Ok(curve)
}

fn read_ir_csv_bytes(bytes: &[u8], label: &str) -> Result<IrWaveform, Box<dyn std::error::Error>> {
    let text = std::str::from_utf8(bytes)?;
    let mut lines = text.lines();
    let header = lines.next().ok_or_else(|| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("IR CSV '{label}' is empty"),
        )
    })?;
    if header != "time_ms,amplitude" {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("IR CSV '{label}' has an unexpected header"),
        )
        .into());
    }
    let mut time_ms = Vec::new();
    let mut amplitude = Vec::new();
    for line in lines {
        if line.trim().is_empty() {
            continue;
        }
        let cells: Vec<&str> = line.split(',').collect();
        time_ms.push(parse_f64_cell(
            cells.first().copied().unwrap_or(""),
            "time",
        )?);
        amplitude.push(parse_f64_cell(
            cells.get(1).copied().unwrap_or(""),
            "amplitude",
        )?);
    }
    let ir = IrWaveform { time_ms, amplitude };
    validate_ir_data(&ir)?;
    Ok(ir)
}

fn read_json_blob_bytes<T>(bytes: &[u8]) -> Result<T, Box<dyn std::error::Error>>
where
    T: serde::de::DeserializeOwned,
{
    Ok(serde_json::from_slice(bytes)?)
}

fn read_ir_member(
    assets: &Path,
    path: &Path,
    manifest: Option<&BundleManifest>,
    captured_files: Option<&CapturedBundleFiles>,
) -> Result<IrWaveform, Box<dyn std::error::Error>> {
    let bytes = read_bundle_path_bytes(
        assets,
        path,
        manifest,
        captured_files,
        MAX_BUNDLE_MEMBER_BYTES,
        "IR CSV sidecar",
    )?;
    read_ir_csv_bytes(bytes.as_ref(), &path.display().to_string())
}

fn read_json_member<T>(
    assets: &Path,
    path: &Path,
    manifest: Option<&BundleManifest>,
    captured_files: Option<&CapturedBundleFiles>,
) -> Result<T, Box<dyn std::error::Error>>
where
    T: serde::de::DeserializeOwned,
{
    let bytes = read_bundle_path_bytes(
        assets,
        path,
        manifest,
        captured_files,
        MAX_BUNDLE_MEMBER_BYTES,
        "bundle sidecar",
    )?;
    read_json_blob_bytes(bytes.as_ref())
}

fn optional_sidecar<T>(
    result: Result<T, Box<dyn std::error::Error>>,
    strict: bool,
) -> io::Result<Option<T>> {
    match result {
        Ok(value) => Ok(Some(value)),
        Err(error) if strict => Err(io_invalid(format!("invalid bundle sidecar: {error}"))),
        Err(_) => Ok(None),
    }
}

fn slim_graph_identity_and_validate_ledger(output: &DspGraph) -> io::Result<GraphIdentity> {
    let mut slim_graph = output.clone();
    let ledger = slim_graph.correction_decisions.take();
    let identity = crate::final_ledger::canonical_graph_identity(&slim_graph);
    if let Some(ledger) = ledger {
        crate::final_ledger::verify_final_binding(&ledger, &identity).map_err(io_invalid)?;
        let payload = serde_json::to_value(&slim_graph)
            .map_err(|error| io_invalid(format!("slim graph does not serialize: {error}")))?;
        if ledger
            .payload_binding
            .as_ref()
            .is_some_and(|binding| !binding.matches(&payload, &identity.fingerprint))
        {
            return Err(io_invalid(
                "decision ledger payload binding does not match the slim graph",
            ));
        }
        if ledger
            .acceptance_evidence
            .as_ref()
            .is_some_and(|evidence| !evidence.matches(&identity.fingerprint))
        {
            return Err(io_invalid(
                "acceptance evidence does not match the canonical slim graph identity",
            ));
        }
    }
    Ok(identity)
}

fn frozen_convolution_resources(
    output: &DspGraph,
    manifest: Option<&BundleManifest>,
    captured_files: &CapturedBundleFiles,
) -> io::Result<BTreeMap<String, FrozenBundleResource>> {
    let Some(manifest) = manifest else {
        return Ok(BTreeMap::new());
    };
    let references = checked_convolution_resource_references(output)
        .map_err(|error| io_invalid(format!("invalid convolution references: {error}")))?;
    let inventory = output
        .metadata
        .as_ref()
        .and_then(|metadata| metadata.final_convolution_sha256.as_ref());
    let mut resources = BTreeMap::new();
    for reference in references {
        let key = reference.replace('\\', "/");
        let record = manifest.files.get(&key).ok_or_else(|| {
            io_invalid(format!(
                "manifested graph convolution reference is not a bundled file: {reference}"
            ))
        })?;
        let expected = inventory
            .and_then(|inventory| inventory.get(&reference))
            .and_then(Option::as_ref)
            .ok_or_else(|| {
                io_invalid(format!(
                    "manifested convolution reference has no SHA-256 binding: {reference}"
                ))
            })?;
        if expected != &record.sha256 {
            return Err(io_invalid(format!(
                "manifested convolution reference and final SHA-256 inventory disagree: {reference}"
            )));
        }
        let bytes = captured_files.get(&key).ok_or_else(|| {
            io_invalid(format!(
                "manifested convolution resource is missing from the captured snapshot: {reference}"
            ))
        })?;
        if bytes.len() as u64 != record.size_bytes || sha256_hex(bytes) != *expected {
            return Err(io_invalid(format!(
                "manifested convolution resource failed integrity validation: {reference}"
            )));
        }
        resources.insert(
            reference.clone(),
            FrozenBundleResource {
                relative_path: reference,
                sha256: expected.clone(),
                bytes: Arc::clone(bytes),
            },
        );
    }
    Ok(resources)
}

fn frozen_output_bundle(
    output: DspGraph,
    slim_graph_bytes: Vec<u8>,
    slim_graph_identity: GraphIdentity,
    verification: OutputBundleVerification,
    resources: BTreeMap<String, FrozenBundleResource>,
) -> FrozenOutputBundle {
    let raw_graph_sha256 = sha256_hex(&slim_graph_bytes);
    FrozenOutputBundle {
        output,
        slim_graph_bytes: Arc::from(slim_graph_bytes),
        raw_graph_sha256,
        slim_graph_identity,
        verification,
        resources,
    }
}

/// Load a native output, re-injecting measurement blobs extracted into the
/// sibling `<stem>_files` directory.
///
/// Legacy outputs with embedded curves load unchanged: only fields that are
/// absent from the JSON are restored from `measurements_index.json`, and
/// missing asset files are skipped rather than treated as errors.
///
/// New manifested bundles are parsed from the same captured bytes that were
/// checked against their manifest. Use [`load_output_bundle_frozen`] when a
/// native consumer must retain graph identity and convolution bytes.
///
/// # Errors
/// Returns an error when graph JSON, a required manifest, or a listed bundle
/// member is invalid or fails integrity validation.
pub fn load_output_bundle(output_path: &Path) -> Result<DspGraph, Box<dyn std::error::Error>> {
    Ok(load_output_bundle_frozen(output_path)?.output)
}

#[cfg(test)]
fn load_output_bundle_with_hook(
    output_path: &Path,
    after_manifest_check: impl FnOnce(&Path) -> io::Result<()>,
) -> Result<DspGraph, Box<dyn std::error::Error>> {
    Ok(load_output_bundle_frozen_with_hooks(output_path, after_manifest_check, |_| Ok(()))?.output)
}

/// Load one immutable snapshot of a native graph and its manifested resources.
///
/// The returned graph is hydrated for existing report consumers. Its canonical
/// identity and resource map remain bound to the slim graph bytes before
/// hydration. Legacy outputs load with [`OutputBundleVerification::LegacyUnverified`]
/// and do not expose trusted convolution resources.
///
/// # Errors
/// Returns an error when the graph, ledger, evidence, manifest, or any listed
/// resource is invalid or changes before the immutable snapshot is captured.
pub fn load_output_bundle_frozen(
    output_path: &Path,
) -> Result<FrozenOutputBundle, Box<dyn std::error::Error>> {
    load_output_bundle_frozen_with_hooks(output_path, |_| Ok(()), |_| Ok(()))
}

fn load_output_bundle_frozen_with_hooks(
    output_path: &Path,
    after_manifest_check: impl FnOnce(&Path) -> io::Result<()>,
    after_snapshot_capture: impl FnOnce(&Path) -> io::Result<()>,
) -> Result<FrozenOutputBundle, Box<dyn std::error::Error>> {
    recover_output_bundle_transactions(output_path)?;
    let root_bytes = read_bounded_file(output_path, MAX_NATIVE_GRAPH_BYTES, "native output graph")?;
    let assets_dir = assets_dir_for(output_path);
    match std::fs::symlink_metadata(&assets_dir) {
        Ok(metadata) if !metadata.is_dir() || metadata.file_type().is_symlink() => {
            return Err(io_invalid("artifact support path is not a regular directory").into());
        }
        Ok(_) => {}
        Err(error) if error.kind() == io::ErrorKind::NotFound => {}
        Err(error) => return Err(error.into()),
    }
    let manifest = load_manifest(&assets_dir)?;
    if let Some(manifest) = &manifest
        && manifest.graph_sha256 != sha256_hex(&root_bytes)
    {
        return Err(io_invalid("native output graph failed bundle integrity validation").into());
    }
    after_manifest_check(&assets_dir)?;
    let captured_files = match manifest.as_ref() {
        Some(manifest) => capture_manifested_bundle_files(&assets_dir, manifest)?,
        None => CapturedBundleFiles::new(),
    };
    after_snapshot_capture(&assets_dir)?;
    let root_value: serde_json::Value = serde_json::from_slice(&root_bytes)?;
    match root_value.get("artifact_bundle_schema_version") {
        None | Some(serde_json::Value::Null) => {}
        Some(value) if value.as_u64() == Some(BUNDLE_MANIFEST_SCHEMA_VERSION as u64) => {
            if manifest.is_none() {
                return Err(io_invalid(
                    "native output requires a missing artifact bundle manifest",
                )
                .into());
            }
        }
        Some(_) => {
            return Err(io_invalid(
                "native output has an unsupported artifact bundle schema marker",
            )
            .into());
        }
    }
    let mut output: DspGraph = serde_json::from_slice(&root_bytes)?;
    if manifest.is_some() || output.artifact_bundle_schema_version.is_some() {
        output.validate().map_err(io_invalid)?;
    }
    let slim_graph_identity = if manifest.is_some() {
        slim_graph_identity_and_validate_ledger(&output)?
    } else {
        let mut slim_graph = output.clone();
        slim_graph.correction_decisions = None;
        crate::final_ledger::canonical_graph_identity(&slim_graph)
    };
    validate_manifested_convolution_resources(
        &output,
        &assets_dir,
        manifest.as_ref(),
        Some(&captured_files),
    )?;
    let resources = frozen_convolution_resources(&output, manifest.as_ref(), &captured_files)?;
    let verification = if manifest.is_some() {
        OutputBundleVerification::ManifestVerified
    } else {
        OutputBundleVerification::LegacyUnverified
    };
    let index_path = assets_dir.join(MEASUREMENTS_INDEX_FILENAME);
    if manifest.is_some() {
        if !captured_files.contains_key(MEASUREMENTS_INDEX_FILENAME) {
            match std::fs::symlink_metadata(&index_path) {
                Err(error) if error.kind() == io::ErrorKind::NotFound => {
                    return Ok(frozen_output_bundle(
                        output,
                        root_bytes,
                        slim_graph_identity,
                        verification,
                        resources,
                    ));
                }
                Ok(_) => {
                    return Err(io_invalid(
                        "measurement index is not bound by the bundle manifest",
                    )
                    .into());
                }
                Err(error) => return Err(error.into()),
            }
        }
    } else {
        match std::fs::symlink_metadata(&index_path) {
            Ok(metadata) if metadata.is_file() && !metadata.file_type().is_symlink() => {}
            Ok(_) => return Err(io_invalid("measurement index is not a regular file").into()),
            // Legacy outputs may omit the measurement index entirely.
            Err(error) if error.kind() == io::ErrorKind::NotFound => {
                return Ok(frozen_output_bundle(
                    output,
                    root_bytes,
                    slim_graph_identity,
                    verification,
                    resources,
                ));
            }
            Err(error) => return Err(error.into()),
        }
    }
    let index_bytes = read_bundle_member_bytes(
        &assets_dir,
        Path::new(MEASUREMENTS_INDEX_FILENAME),
        manifest.as_ref(),
        Some(&captured_files),
        MAX_BUNDLE_MANIFEST_BYTES,
        "measurement index",
    )?;
    let index: serde_json::Value = serde_json::from_slice(index_bytes.as_ref())?;
    let channels = index
        .get("channels")
        .and_then(|value| value.as_object())
        .cloned()
        .unwrap_or_default();
    for (name, entry) in &channels {
        let Some(chain) = output.channels.get_mut(name) else {
            if manifest.is_some() {
                return Err(io_invalid(format!(
                    "measurement index references unknown graph channel '{name}'"
                ))
                .into());
            }
            continue;
        };
        let Some(fields) = entry.as_object() else {
            if manifest.is_some() {
                return Err(
                    io_invalid(format!("measurement index entry for '{name}' is invalid")).into(),
                );
            }
            continue;
        };
        for (kind, file) in fields {
            let Some(file) = file.as_str() else {
                if manifest.is_some() {
                    return Err(io_invalid(format!(
                        "measurement index path for '{kind}' is invalid"
                    ))
                    .into());
                }
                continue;
            };
            let path =
                indexed_member_path(&assets_dir, file, manifest.as_ref(), Some(&captured_files))?;
            if !indexed_member_is_available(
                &assets_dir,
                file,
                manifest.as_ref(),
                Some(&captured_files),
            )? {
                if manifest.is_some() {
                    return Err(
                        io_invalid(format!("missing measurement bundle member: {file}")).into(),
                    );
                }
                continue;
            }
            match kind.as_str() {
                "initial_curve" if chain.initial_curve.is_none() => {
                    let curve = read_indexed_curve(
                        &assets_dir,
                        fields,
                        kind,
                        &path,
                        manifest.as_ref(),
                        Some(&captured_files),
                    );
                    chain.initial_curve = if manifest.is_some() {
                        Some(curve?)
                    } else {
                        curve.ok()
                    };
                }
                "final_curve" if chain.final_curve.is_none() => {
                    let curve = read_indexed_curve(
                        &assets_dir,
                        fields,
                        kind,
                        &path,
                        manifest.as_ref(),
                        Some(&captured_files),
                    );
                    chain.final_curve = if manifest.is_some() {
                        Some(curve?)
                    } else {
                        curve.ok()
                    };
                }
                "eq_response" if chain.eq_response.is_none() => {
                    let curve = read_indexed_curve(
                        &assets_dir,
                        fields,
                        kind,
                        &path,
                        manifest.as_ref(),
                        Some(&captured_files),
                    );
                    chain.eq_response = if manifest.is_some() {
                        Some(curve?)
                    } else {
                        curve.ok()
                    };
                }
                "target_curve" if chain.target_curve.is_none() => {
                    let curve = read_indexed_curve(
                        &assets_dir,
                        fields,
                        kind,
                        &path,
                        manifest.as_ref(),
                        Some(&captured_files),
                    );
                    chain.target_curve = if manifest.is_some() {
                        Some(curve?)
                    } else {
                        curve.ok()
                    };
                }
                "pre_ir" if chain.pre_ir.is_none() => {
                    chain.pre_ir = optional_sidecar(
                        read_ir_member(
                            &assets_dir,
                            &path,
                            manifest.as_ref(),
                            Some(&captured_files),
                        ),
                        manifest.is_some(),
                    )?;
                }
                "post_ir" if chain.post_ir.is_none() => {
                    chain.post_ir = optional_sidecar(
                        read_ir_member(
                            &assets_dir,
                            &path,
                            manifest.as_ref(),
                            Some(&captured_files),
                        ),
                        manifest.is_some(),
                    )?;
                }
                "early_late_curves" if chain.early_late_curves.is_none() => {
                    chain.early_late_curves = optional_sidecar(
                        read_json_member::<ChannelEarlyLateCurves>(
                            &assets_dir,
                            &path,
                            manifest.as_ref(),
                            Some(&captured_files),
                        ),
                        manifest.is_some(),
                    )?;
                }
                "waterfall" if chain.waterfall.is_none() => {
                    chain.waterfall = optional_sidecar(
                        read_json_member::<ChannelWaterfall>(
                            &assets_dir,
                            &path,
                            manifest.as_ref(),
                            Some(&captured_files),
                        ),
                        manifest.is_some(),
                    )?;
                }
                "resonance_decays" if chain.resonance_decays.is_none() => {
                    chain.resonance_decays = optional_sidecar(
                        read_json_member::<ChannelResonanceDecays>(
                            &assets_dir,
                            &path,
                            manifest.as_ref(),
                            Some(&captured_files),
                        ),
                        manifest.is_some(),
                    )?;
                }
                "wavelet" if chain.wavelet.is_none() => {
                    chain.wavelet = optional_sidecar(
                        read_json_member::<ChannelWavelet>(
                            &assets_dir,
                            &path,
                            manifest.as_ref(),
                            Some(&captured_files),
                        ),
                        manifest.is_some(),
                    )?;
                }
                _ => {
                    if let Some(rest) = kind.strip_prefix("driver")
                        && let Some((index_text, suffix)) = rest.split_once('_')
                        && suffix.ends_with("_measured_acoustics")
                        && let Ok(driver_index) = index_text.parse::<usize>()
                        && let Some(drivers) = chain.drivers.as_mut()
                        && let Some(driver) = drivers.get_mut(driver_index)
                        && driver.measured_acoustics.is_none()
                    {
                        driver.measured_acoustics = optional_sidecar(
                            read_json_member::<MeasuredRoomAcoustics>(
                                &assets_dir,
                                &path,
                                manifest.as_ref(),
                                Some(&captured_files),
                            ),
                            manifest.is_some(),
                        )?;
                    }
                    if let Some(rest) = kind.strip_prefix("driver")
                        && let Some((index_text, suffix)) = rest.split_once('_')
                        && suffix.ends_with("_initial_curve")
                        && let Ok(driver_index) = index_text.parse::<usize>()
                        && let Some(drivers) = chain.drivers.as_mut()
                        && let Some(driver) = drivers.get_mut(driver_index)
                        && driver.initial_curve.is_none()
                    {
                        let curve = read_indexed_curve(
                            &assets_dir,
                            fields,
                            kind,
                            &path,
                            manifest.as_ref(),
                            Some(&captured_files),
                        );
                        driver.initial_curve = if manifest.is_some() {
                            Some(curve?)
                        } else {
                            curve.ok()
                        };
                    }
                }
            }
        }
    }
    if let Some(deployed) = index
        .get("deployed_source_curves")
        .and_then(|v| v.as_object())
    {
        for (name, file) in deployed {
            if output.deployed_source_curves.contains_key(name) {
                continue;
            }
            let Some(file) = file.as_str() else {
                if manifest.is_some() {
                    return Err(
                        io_invalid(format!("deployed curve path for '{name}' is invalid")).into(),
                    );
                }
                continue;
            };
            let path =
                indexed_member_path(&assets_dir, file, manifest.as_ref(), Some(&captured_files))?;
            if !indexed_member_is_available(
                &assets_dir,
                file,
                manifest.as_ref(),
                Some(&captured_files),
            )? {
                if manifest.is_some() {
                    return Err(
                        io_invalid(format!("missing measurement bundle member: {file}")).into(),
                    );
                }
                continue;
            }
            let metadata_file = index
                .get("deployed_source_curve_metadata")
                .and_then(serde_json::Value::as_object)
                .and_then(|metadata| metadata.get(name));
            let mut metadata_fields = serde_json::Map::new();
            if let Some(metadata_file) = metadata_file {
                metadata_fields.insert(
                    "deployed_source_curves_metadata".into(),
                    metadata_file.clone(),
                );
            } else if manifest.is_some() && index.get("deployed_source_curve_metadata").is_some() {
                return Err(io_invalid(format!(
                    "deployed curve metadata path for '{name}' is invalid"
                ))
                .into());
            }
            let curve = read_indexed_curve(
                &assets_dir,
                &metadata_fields,
                "deployed_source_curves",
                &path,
                manifest.as_ref(),
                Some(&captured_files),
            );
            if manifest.is_some() {
                output.deployed_source_curves.insert(name.clone(), curve?);
            } else if let Ok(curve) = curve {
                output.deployed_source_curves.insert(name.clone(), curve);
            }
        }
    }
    Ok(frozen_output_bundle(
        output,
        root_bytes,
        slim_graph_identity,
        verification,
        resources,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn native_ir_sidecar_preserves_tails_and_time_origin() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("native.csv");
        let ir = IrWaveform {
            time_ms: vec![-0.1234567890123, 0.2098765443210333],
            amplitude: vec![0.1234567890123456, -1.234567890123456e-12],
        };
        write_ir_csv(&path, &ir).expect("save native IR");
        let bytes =
            read_bounded_file(&path, MAX_BUNDLE_MEMBER_BYTES, "IR CSV").expect("read saved IR");
        let restored = read_ir_csv_bytes(&bytes, "native.csv").expect("reload native IR");
        assert_eq!(restored.time_ms, ir.time_ms);
        assert_eq!(restored.amplitude, ir.amplitude);
    }
    use roomeq_model::CurveData;

    fn curve(freq: Vec<f64>, spl: Vec<f64>) -> CurveData {
        CurveData {
            freq,
            spl,
            phase: None,
            norm_range: None,
            noise_floor_db: None,
            coherence: None,
        }
    }

    #[test]
    fn assets_dir_uses_stem_files() {
        assert_eq!(
            assets_dir_for(Path::new("/tmp/out/dsp.json")),
            PathBuf::from("/tmp/out/dsp_files")
        );
        assert_eq!(
            assets_dir_for(Path::new("dsp.json")),
            PathBuf::from("./dsp_files")
        );
        assert_eq!(
            manifest_path_for(Path::new("/tmp/out/dsp.json")),
            PathBuf::from("/tmp/out/dsp_files/manifest.json")
        );
        assert_eq!(
            run_log_path_for(Path::new("/tmp/out/dsp.json")),
            PathBuf::from("/tmp/out/dsp_files/roomeq.log")
        );
    }

    #[test]
    fn bundle_extracts_curves_and_keeps_plugins() {
        let dir = tempfile::tempdir().expect("temp dir");
        let output_path = dir.path().join("dsp.json");
        let mut output = DspGraph::new("1");
        output.add_channel("L", Vec::new());
        let chain = output.channels.get_mut("L").expect("channel");
        chain.initial_curve = Some(curve(vec![100.0, 1000.0], vec![80.0, 81.0]));
        chain.final_curve = Some(curve(vec![100.0, 1000.0], vec![79.0, 80.0]));
        chain.pre_ir = Some(IrWaveform {
            time_ms: vec![0.0, 0.1],
            amplitude: vec![1.0, 0.5],
        });
        output.deployed_source_curves.insert(
            "L".to_string(),
            curve(vec![100.0, 1000.0], vec![79.5, 80.5]),
        );

        let files = save_output_bundle(&mut output, &output_path).expect("save bundle");

        assert!(output_path.is_file(), "slim JSON must exist");
        assert!(
            output.channels["L"].initial_curve.is_none(),
            "curves must be stripped from JSON"
        );
        assert!(
            output.channels["L"].pre_ir.is_none(),
            "IRs must be stripped from JSON"
        );
        assert!(
            output.deployed_source_curves.is_empty(),
            "deployed curves must be stripped from JSON"
        );
        assert!(
            output.channels["L"].plugins.is_empty(),
            "plugins stay in JSON (empty fixture)"
        );
        let assets = assets_dir_for(&output_path);
        assert!(assets.join("L__initial.csv").is_file());
        assert!(assets.join("L__final.csv").is_file());
        assert!(assets.join("L__pre_ir.csv").is_file());
        assert!(assets.join("deployed__L.csv").is_file());
        assert!(files.contains(&"L__initial.csv".to_string()));
        let slim = std::fs::read(&output_path).expect("read slim JSON");
        let full_csv = std::fs::read(assets.join("L__initial.csv")).expect("read csv");
        assert!(
            slim.len() < full_csv.len() + 500,
            "slim JSON should stay small, got {} bytes",
            slim.len()
        );
    }

    #[test]
    fn bundle_roundtrip_restores_curves_and_ir() {
        let dir = tempfile::tempdir().expect("temp dir");
        let output_path = dir.path().join("dsp.json");
        let mut output = DspGraph::new("1");
        output.add_channel("L", Vec::new());
        let chain = output.channels.get_mut("L").expect("channel");
        let source_curve = CurveData {
            freq: vec![123.45678901234567, 2345.678901234567],
            spl: vec![80.12345678901235, 81.23456789012345],
            phase: Some(vec![12.3456789012345, -23.456789012345]),
            norm_range: Some((100.1234567890123, 19000.98765432109)),
            noise_floor_db: Some(vec![20.12345678901235, 21.23456789012345]),
            coherence: Some(vec![0.9123456789012345, 0.9876543210987654]),
        };
        chain.initial_curve = Some(source_curve.clone());
        chain.post_ir = Some(IrWaveform {
            time_ms: vec![0.0, 0.1],
            amplitude: vec![1.0, 0.5],
        });
        output
            .deployed_source_curves
            .insert("L".to_string(), source_curve.clone());

        save_output_bundle(&mut output, &output_path).expect("save bundle");
        assert!(output.channels["L"].initial_curve.is_none());
        let assets = assets_dir_for(&output_path);

        let restored = load_output_bundle(&output_path).expect("load bundle");
        let initial = restored.channels["L"]
            .initial_curve
            .as_ref()
            .expect("initial curve restored");
        assert_eq!(initial.freq, source_curve.freq);
        assert_eq!(initial.spl, source_curve.spl);
        assert_eq!(
            initial.phase.as_ref().expect("phase restored"),
            source_curve.phase.as_ref().expect("phase source")
        );
        assert_eq!(
            initial.norm_range,
            source_curve.norm_range,
            "curve sidecar was {}",
            std::fs::read_to_string(assets.join("L__initial.csv.json")).unwrap()
        );
        assert_eq!(initial.noise_floor_db, source_curve.noise_floor_db);
        assert_eq!(initial.coherence, source_curve.coherence);
        let post_ir = restored.channels["L"]
            .post_ir
            .as_ref()
            .expect("post IR restored");
        assert_eq!(post_ir.amplitude, vec![1.0, 0.5]);
        assert_eq!(restored.deployed_source_curves["L"].freq, source_curve.freq);
        assert_eq!(restored.deployed_source_curves["L"].spl, source_curve.spl);
        assert_eq!(
            restored.deployed_source_curves["L"].norm_range,
            source_curve.norm_range
        );
        assert_eq!(
            restored.deployed_source_curves["L"].noise_floor_db,
            source_curve.noise_floor_db
        );
        assert_eq!(
            restored.deployed_source_curves["L"].coherence,
            source_curve.coherence
        );
    }

    #[test]
    fn legacy_embedded_output_loads_without_index() {
        let dir = tempfile::tempdir().expect("temp dir");
        let output_path = dir.path().join("legacy.json");
        let mut output = DspGraph::new("1");
        output.add_channel("L", Vec::new());
        output.channels.get_mut("L").expect("channel").initial_curve =
            Some(curve(vec![100.0], vec![80.0]));
        crate::output::save_dsp_chain(&output, &output_path).expect("save legacy");

        let restored = load_output_bundle(&output_path).expect("load legacy");
        assert_eq!(
            restored.channels["L"]
                .initial_curve
                .as_ref()
                .expect("embedded curve kept")
                .spl,
            vec![80.0]
        );
    }

    #[test]
    fn failed_index_write_rejects_extraction_instead_of_publishing_partial_bundle() {
        let dir = tempfile::tempdir().expect("temp dir");
        let mut output = DspGraph::new("1");
        output.add_channel("L", Vec::new());
        output.channels.get_mut("L").expect("channel").initial_curve =
            Some(curve(vec![100.0, 1000.0], vec![80.0, 81.0]));

        let extracted = extract_measurements_to_assets_with_index_writer(
            &mut output,
            dir.path(),
            &mut |path, value| {
                if path
                    .file_name()
                    .is_some_and(|name| name == MEASUREMENTS_INDEX_FILENAME)
                {
                    return Err(io::Error::other("simulated index write failure"));
                }
                write_json_file(path, value)
            },
        );

        assert!(output.channels["L"].initial_curve.is_none());
        assert!(
            validate_extraction_complete(&output, &extracted).is_err(),
            "a missing index must stop publication even if member files were written"
        );
    }

    fn existing_bundle_fixture(output_path: &Path) -> (Vec<u8>, PathBuf) {
        let mut prior = DspGraph::new("prior");
        prior.add_channel("L", Vec::new());
        crate::output::save_dsp_chain(&prior, output_path).expect("write prior graph");
        let assets = assets_dir_for(output_path);
        std::fs::create_dir_all(&assets).expect("create prior support");
        std::fs::write(assets.join("preserved.wav"), b"prior support")
            .expect("write support sentinel");
        (
            std::fs::read(output_path).expect("read prior graph"),
            assets,
        )
    }

    #[test]
    fn malformed_measurements_leave_the_previous_bundle_unchanged() {
        let malformed = [
            CurveData {
                freq: vec![100.0, 200.0],
                spl: vec![80.0],
                phase: None,
                norm_range: None,
                noise_floor_db: None,
                coherence: None,
            },
            CurveData {
                freq: vec![100.0, 90.0],
                spl: vec![80.0, 81.0],
                phase: None,
                norm_range: None,
                noise_floor_db: None,
                coherence: None,
            },
            CurveData {
                freq: vec![100.0, 200.0],
                spl: vec![80.0, 81.0],
                phase: Some(vec![0.0]),
                norm_range: None,
                noise_floor_db: None,
                coherence: None,
            },
            CurveData {
                freq: vec![100.0, 200.0],
                spl: vec![80.0, f64::NAN],
                phase: None,
                norm_range: None,
                noise_floor_db: None,
                coherence: None,
            },
            CurveData {
                freq: vec![100.0, 200.0],
                spl: vec![80.0, 81.0],
                phase: None,
                norm_range: None,
                noise_floor_db: None,
                coherence: Some(vec![0.9]),
            },
            CurveData {
                freq: vec![100.0, 200.0],
                spl: vec![80.0, 81.0],
                phase: None,
                norm_range: None,
                noise_floor_db: None,
                coherence: Some(vec![0.9, 1.1]),
            },
            CurveData {
                freq: vec![100.0, 200.0],
                spl: vec![80.0, 81.0],
                phase: None,
                norm_range: Some((1000.0, 500.0)),
                noise_floor_db: None,
                coherence: None,
            },
        ];
        for curve_data in malformed {
            let dir = tempfile::tempdir().expect("temp dir");
            let output_path = dir.path().join("dsp.json");
            let (old_root, assets) = existing_bundle_fixture(&output_path);
            let mut replacement = DspGraph::new("replacement");
            replacement.add_channel("L", Vec::new());
            replacement.channels.get_mut("L").unwrap().initial_curve = Some(curve_data);

            assert!(save_output_bundle(&mut replacement, &output_path).is_err());
            assert_eq!(std::fs::read(&output_path).unwrap(), old_root);
            assert_eq!(
                std::fs::read(assets.join("preserved.wav")).unwrap(),
                b"prior support"
            );
            assert!(
                replacement.channels["L"].initial_curve.is_some(),
                "failed save must retain malformed in-memory data for diagnosis"
            );
        }
    }

    #[test]
    fn malformed_ir_leaves_the_previous_bundle_unchanged() {
        let dir = tempfile::tempdir().expect("temp dir");
        let output_path = dir.path().join("dsp.json");
        let (old_root, assets) = existing_bundle_fixture(&output_path);
        let mut replacement = DspGraph::new("replacement");
        replacement.add_channel("L", Vec::new());
        replacement.channels.get_mut("L").unwrap().pre_ir = Some(IrWaveform {
            time_ms: vec![0.0, 0.1],
            amplitude: vec![1.0],
        });

        assert!(save_output_bundle(&mut replacement, &output_path).is_err());
        assert_eq!(std::fs::read(&output_path).unwrap(), old_root);
        assert_eq!(
            std::fs::read(assets.join("preserved.wav")).unwrap(),
            b"prior support"
        );
        assert!(replacement.channels["L"].pre_ir.is_some());
    }

    #[test]
    fn failed_publication_boundaries_preserve_the_previous_root_and_support() {
        for failing_boundary in 1..=3 {
            let dir = tempfile::tempdir().expect("temp dir");
            let output_path = dir.path().join("dsp.json");
            let (old_root, assets) = existing_bundle_fixture(&output_path);
            let mut replacement = DspGraph::new("replacement");
            replacement.add_channel("L", Vec::new());
            replacement
                .channels
                .get_mut("L")
                .expect("channel")
                .initial_curve = Some(curve(vec![100.0], vec![75.0]));

            let mut rename_count = 0;
            let result = save_output_bundle_using(
                &mut replacement,
                &output_path,
                &mut |from, to| {
                    rename_count += 1;
                    if rename_count == failing_boundary {
                        Err(io::Error::other("simulated rename failure"))
                    } else {
                        std::fs::rename(from, to)
                    }
                },
                &mut |path, bytes| {
                    if failing_boundary == 3 {
                        Err(io::Error::other("simulated root replacement failure"))
                    } else {
                        write_file_atomically(path, bytes)
                    }
                },
            );

            assert!(result.is_err(), "boundary {failing_boundary} should fail");
            assert_eq!(std::fs::read(&output_path).unwrap(), old_root);
            assert_eq!(
                std::fs::read(assets.join("preserved.wav")).unwrap(),
                b"prior support"
            );
            assert!(!bundle_transaction_path(&output_path).exists());
            assert!(
                replacement.channels["L"].initial_curve.is_some(),
                "failed save must not mutate the caller's graph"
            );
        }
    }

    #[test]
    fn journal_parent_sync_failure_keeps_recovery_data_until_recovery() {
        let dir = tempfile::tempdir().expect("temp dir");
        let output_path = dir.path().join("dsp.json");
        let (old_root, assets) = existing_bundle_fixture(&output_path);
        let mut replacement = DspGraph::new("replacement");
        replacement.add_channel("L", Vec::new());
        let mut sync_calls = 0;
        let result = save_output_bundle_using_hooks(
            &mut replacement,
            &output_path,
            &mut |from, to| std::fs::rename(from, to),
            &mut |path, bytes| write_file_atomically(path, bytes),
            &mut |path, bytes| write_file_atomically(path, bytes),
            &mut |path| {
                sync_calls += 1;
                if sync_calls == 1 {
                    Err(io::Error::other("simulated parent directory sync failure"))
                } else {
                    sync_directory(path)
                }
            },
        );

        assert!(result.is_err());
        assert_eq!(std::fs::read(&output_path).unwrap(), old_root);
        assert_eq!(
            std::fs::read(assets.join("preserved.wav")).unwrap(),
            b"prior support"
        );
        let journal_path = bundle_transaction_path(&output_path);
        let journal: BundleTransactionJournal =
            serde_json::from_slice(&std::fs::read(&journal_path).unwrap()).unwrap();
        let transaction_root = transaction_directory(dir.path(), &journal.transaction_directory)
            .expect("retained recovery directory");
        assert!(transaction_root.join("previous_output.json").is_file());

        recover_pending_bundle(&output_path).expect("recover after failed journal sync");
        assert!(!journal_path.exists());
        assert!(!transaction_root.exists());
        assert_eq!(std::fs::read(&output_path).unwrap(), old_root);
        assert_eq!(
            std::fs::read(assets.join("preserved.wav")).unwrap(),
            b"prior support"
        );
    }

    #[test]
    fn journal_write_failure_cleans_unpublished_transaction_data() {
        let dir = tempfile::tempdir().expect("temp dir");
        let output_path = dir.path().join("dsp.json");
        let (old_root, assets) = existing_bundle_fixture(&output_path);
        let mut replacement = DspGraph::new("replacement");
        replacement.add_channel("L", Vec::new());
        let result = save_output_bundle_using_hooks(
            &mut replacement,
            &output_path,
            &mut |from, to| std::fs::rename(from, to),
            &mut |path, bytes| write_file_atomically(path, bytes),
            &mut |_path, _bytes| Err(io::Error::other("simulated journal write failure")),
            &mut |path: &Path| sync_directory(path),
        );

        assert!(result.is_err());
        assert!(!bundle_transaction_path(&output_path).exists());
        assert_eq!(std::fs::read(&output_path).unwrap(), old_root);
        assert_eq!(
            std::fs::read(assets.join("preserved.wav")).unwrap(),
            b"prior support"
        );
        assert!(std::fs::read_dir(dir.path()).unwrap().all(|entry| {
            !entry
                .unwrap()
                .file_name()
                .to_string_lossy()
                .starts_with(".autoeq-bundle-")
        }));
    }

    #[test]
    fn recovery_rejects_corrupted_previous_output_backup_before_mutating_bundle() {
        let dir = tempfile::tempdir().expect("temp dir");
        let output_path = dir.path().join("dsp.json");
        let (old_root, assets) = existing_bundle_fixture(&output_path);
        let mut next = DspGraph::new("next");
        next.add_channel("L", Vec::new());
        let next_root = serde_json::to_vec_pretty(&next).unwrap();
        std::fs::write(&output_path, &next_root).unwrap();

        let staging = ArtifactBundleStaging::new(dir.path()).expect("transaction staging");
        let transaction_root = staging.root().to_path_buf();
        std::fs::write(
            transaction_root.join("previous_output.json"),
            b"corrupt backup",
        )
        .unwrap();
        let generation = transaction_root
            .file_name()
            .unwrap()
            .to_str()
            .unwrap()
            .to_string();
        let transaction_root = staging.keep();
        let journal = BundleTransactionJournal {
            schema_version: BUNDLE_MANIFEST_SCHEMA_VERSION,
            transaction_directory: generation.clone(),
            generation,
            had_assets: true,
            had_output: true,
            old_output_sha256: Some(sha256_hex(&old_root)),
            new_output_sha256: sha256_hex(&next_root),
        };
        write_file_atomically(
            &bundle_transaction_path(&output_path),
            &serde_json::to_vec(&journal).unwrap(),
        )
        .unwrap();

        let error = recover_pending_bundle(&output_path).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("prior output backup failed integrity")
        );
        assert_eq!(std::fs::read(&output_path).unwrap(), next_root);
        assert_eq!(
            std::fs::read(assets.join("preserved.wav")).unwrap(),
            b"prior support"
        );
        assert!(bundle_transaction_path(&output_path).exists());
        assert!(transaction_root.exists());
    }

    #[test]
    fn invalid_graph_does_not_replace_the_previous_bundle() {
        let dir = tempfile::tempdir().expect("temp dir");
        let output_path = dir.path().join("dsp.json");
        let (old_root, assets) = existing_bundle_fixture(&output_path);
        let mut invalid = DspGraph::new("1");

        assert!(save_output_bundle(&mut invalid, &output_path).is_err());
        assert_eq!(std::fs::read(&output_path).unwrap(), old_root);
        assert_eq!(
            std::fs::read(assets.join("preserved.wav")).unwrap(),
            b"prior support"
        );
    }

    fn convolution_graph(reference: &str, expected_sha256: &str) -> DspGraph {
        let mut output = DspGraph::new("1");
        output.add_channel(
            "L",
            vec![roomeq_model::Plugin {
                kind: "convolution".to_string(),
                parameters: serde_json::json!({"ir_file": reference}),
            }],
        );
        output.metadata = Some(
            serde_json::from_value(serde_json::json!({
                "pre_score": 0.0,
                "post_score": 0.0,
                "algorithm": "fixture",
                "iterations": 0,
                "timestamp": "fixture",
                "final_convolution_sha256": serde_json::Map::from_iter([(
                    reference.to_owned(),
                    serde_json::Value::String(expected_sha256.to_owned()),
                )]),
            }))
            .expect("optimization metadata"),
        );
        output
    }

    const PUBLICATION_CHILD_ENV: &str = "ROOMEQ_BUNDLE_CRASH_TEST_CHILD";
    const PUBLICATION_PHASE_ENV: &str = "ROOMEQ_BUNDLE_CRASH_TEST_PHASE";
    const PUBLICATION_READY_ENV: &str = "ROOMEQ_BUNDLE_CRASH_TEST_READY_FILE";
    const PUBLICATION_SOURCE_ENV: &str = "ROOMEQ_BUNDLE_CRASH_TEST_SOURCE_FILE";
    const PUBLICATION_DESTINATION_ENV: &str = "ROOMEQ_BUNDLE_CRASH_TEST_DESTINATION_FILE";
    const PUBLICATION_PHASES: &[&str] = &[
        "previous_output_backup_synced",
        "stage_assets_synced",
        "transaction_root_synced",
        "journal_written",
        "journal_parent_synced",
        "previous_assets_renamed",
        "candidate_assets_installed",
        "assets_parent_synced",
        "root_written",
        "root_parent_synced",
        "transaction_directory_removed",
        "journal_removed",
        "cleanup_parent_synced",
    ];

    struct ChildProcessGuard(Option<std::process::Child>);

    impl Drop for ChildProcessGuard {
        fn drop(&mut self) {
            if let Some(mut child) = self.0.take() {
                let _ = child.kill();
                let _ = child.wait();
            }
        }
    }

    fn test_float_pcm_wav(samples: &[f32]) -> Vec<u8> {
        let data_len = u32::try_from(std::mem::size_of_val(samples))
            .expect("fixture PCM length fits a WAV chunk");
        let sample_rate = 48_000_u32;
        let data_capacity =
            usize::try_from(data_len).expect("fixture PCM length fits this host's address space");
        let mut bytes = Vec::with_capacity(44 + data_capacity);
        bytes.extend_from_slice(b"RIFF");
        bytes.extend_from_slice(&(36_u32 + data_len).to_le_bytes());
        bytes.extend_from_slice(b"WAVEfmt ");
        bytes.extend_from_slice(&16_u32.to_le_bytes());
        bytes.extend_from_slice(&3_u16.to_le_bytes());
        bytes.extend_from_slice(&1_u16.to_le_bytes());
        bytes.extend_from_slice(&sample_rate.to_le_bytes());
        bytes.extend_from_slice(&(sample_rate * 4).to_le_bytes());
        bytes.extend_from_slice(&4_u16.to_le_bytes());
        bytes.extend_from_slice(&32_u16.to_le_bytes());
        bytes.extend_from_slice(b"data");
        bytes.extend_from_slice(&data_len.to_le_bytes());
        for sample in samples {
            bytes.extend_from_slice(&sample.to_le_bytes());
        }
        bytes
    }

    fn save_process_recovery_fixture(
        output_path: &Path,
        version: &str,
        frequencies_hz: [f64; 2],
        levels_db: [f64; 2],
        pcm_samples: &[f32],
    ) -> (FrozenOutputBundle, BundleManifest) {
        let source_assets = output_path.with_file_name(format!(
            "{}-source-assets",
            output_path
                .file_stem()
                .and_then(|stem| stem.to_str())
                .expect("fixture output has a UTF-8 file stem")
        ));
        std::fs::create_dir_all(&source_assets).expect("create fixture source assets");
        let pcm = test_float_pcm_wav(pcm_samples);
        std::fs::write(source_assets.join("impulse.wav"), &pcm)
            .expect("write fixture convolution PCM");
        let pcm_sha256 = sha256_hex(&pcm);
        let mut graph = convolution_graph("impulse.wav", &pcm_sha256);
        graph.version = version.to_string();
        graph
            .channels
            .get_mut("L")
            .expect("fixture channel")
            .initial_curve = Some(curve(frequencies_hz.to_vec(), levels_db.to_vec()));
        save_output_bundle_with_resources(&mut graph, output_path, &source_assets)
            .expect("save manifested fixture bundle");

        let frozen = load_output_bundle_frozen(output_path).expect("load fixture bundle");
        let resource_path = frozen.output().channels["L"].plugins[0].parameters["ir_file"]
            .as_str()
            .expect("bound fixture resource path");
        let resource = frozen
            .resource(resource_path)
            .expect("fixture resource is captured");
        assert_eq!(resource.bytes(), pcm);
        assert_eq!(resource.sha256(), pcm_sha256);
        let manifest = load_manifest(&assets_dir_for(output_path))
            .expect("read fixture manifest")
            .expect("fixture manifest exists");
        (frozen, manifest)
    }

    fn assert_recovered_generation(
        output_path: &Path,
        expected: &FrozenOutputBundle,
        expected_manifest: &BundleManifest,
    ) {
        let recovered = load_output_bundle_frozen(output_path).expect("recover native bundle");
        assert_eq!(
            recovered.verification(),
            OutputBundleVerification::ManifestVerified
        );
        assert_eq!(recovered.slim_graph_bytes(), expected.slim_graph_bytes());
        assert_eq!(recovered.raw_graph_sha256(), expected.raw_graph_sha256());
        assert_eq!(
            sha256_hex(recovered.slim_graph_bytes()),
            recovered.raw_graph_sha256()
        );
        assert_eq!(recovered.output().version, expected.output().version);
        let recovered_curve = recovered.output().channels["L"]
            .initial_curve
            .as_ref()
            .expect("recovered curve");
        let expected_curve = expected.output().channels["L"]
            .initial_curve
            .as_ref()
            .expect("expected curve");
        assert_eq!(recovered_curve.freq, expected_curve.freq);
        assert_eq!(recovered_curve.spl, expected_curve.spl);

        let recovered_reference = recovered.output().channels["L"].plugins[0].parameters["ir_file"]
            .as_str()
            .expect("recovered convolution reference");
        let expected_reference = expected.output().channels["L"].plugins[0].parameters["ir_file"]
            .as_str()
            .expect("expected convolution reference");
        assert_eq!(recovered_reference, expected_reference);
        let recovered_resource = recovered
            .resource(recovered_reference)
            .expect("recovered convolution PCM");
        let expected_resource = expected
            .resource(expected_reference)
            .expect("expected convolution PCM");
        assert_eq!(recovered_resource.sha256(), expected_resource.sha256());
        assert_eq!(recovered_resource.bytes(), expected_resource.bytes());

        let recovered_manifest = load_manifest(&assets_dir_for(output_path))
            .expect("read recovered manifest")
            .expect("recovered manifest exists");
        assert_eq!(recovered_manifest.graph_sha256, expected.raw_graph_sha256());
        assert_eq!(recovered_manifest.files, expected_manifest.files);
    }

    fn kill_child_at_publication_phase(
        phase: &str,
        source_path: &Path,
        destination_path: &Path,
        ready_path: &Path,
    ) {
        let executable = std::env::current_exe().expect("current test executable");
        let child = std::process::Command::new(executable)
            .arg("--exact")
            .arg("output_bundle::tests::publication_child_process_pause_helper")
            .arg("--nocapture")
            .arg("--test-threads=1")
            .env(PUBLICATION_CHILD_ENV, "1")
            .env(PUBLICATION_PHASE_ENV, phase)
            .env(PUBLICATION_READY_ENV, ready_path)
            .env(PUBLICATION_SOURCE_ENV, source_path)
            .env(PUBLICATION_DESTINATION_ENV, destination_path)
            .spawn()
            .expect("start child publication process");
        let mut child = ChildProcessGuard(Some(child));
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(20);
        loop {
            if ready_path.is_file() {
                let marker = std::fs::read_to_string(ready_path)
                    .expect("read child publication barrier marker");
                assert_eq!(
                    marker.trim_end(),
                    phase,
                    "child reached a different publication phase"
                );
                let process = child.0.as_mut().expect("child process remains owned");
                process.kill().expect("kill child at publication barrier");
                let status = process.wait().expect("wait for killed child");
                assert!(
                    !status.success(),
                    "child should have been killed at {phase}"
                );
                child.0.take();
                return;
            }
            if let Some(status) = child
                .0
                .as_mut()
                .expect("child process remains owned")
                .try_wait()
                .expect("check child process")
            {
                panic!("child exited before publication barrier {phase}: {status}");
            }
            if std::time::Instant::now() >= deadline {
                let process = child.0.as_mut().expect("child process remains owned");
                let _ = process.kill();
                let _ = process.wait();
                child.0.take();
                panic!("child did not reach publication barrier {phase}");
            }
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
    }

    #[test]
    fn publication_child_process_pause_helper() {
        if std::env::var(PUBLICATION_CHILD_ENV).as_deref() != Ok("1") {
            return;
        }
        let source_path = PathBuf::from(
            std::env::var_os(PUBLICATION_SOURCE_ENV).expect("candidate source path is provided"),
        );
        let destination_path = PathBuf::from(
            std::env::var_os(PUBLICATION_DESTINATION_ENV)
                .expect("publication destination path is provided"),
        );
        let mut candidate = load_output_bundle(&source_path).expect("load candidate bundle");
        let source_assets = assets_dir_for(&source_path);
        let mut prepare = |_: &mut DspGraph, _: &Path| Ok(());
        let mut rename = |from: &Path, to: &Path| std::fs::rename(from, to);
        let mut write_root = |path: &Path, bytes: &[u8]| write_file_atomically(path, bytes);
        let mut write_journal = |path: &Path, bytes: &[u8]| write_file_atomically(path, bytes);
        let mut sync_parent = |path: &Path| sync_directory(path);
        let mut hooks = BundleHooks {
            prepare: &mut prepare,
            rename: &mut rename,
            write_root: &mut write_root,
            write_journal: &mut write_journal,
            sync_parent: &mut sync_parent,
        };
        save_output_bundle_using_hooks_and_assets(
            &mut candidate,
            &destination_path,
            Some(&source_assets),
            true,
            &mut hooks,
        )
        .expect("child publication should pause before returning");
        panic!("child publication returned without reaching its requested barrier");
    }

    #[test]
    fn process_death_at_each_publication_boundary_recovers_one_complete_generation() {
        for phase in PUBLICATION_PHASES {
            let directory = tempfile::tempdir().expect("temporary fixture root");
            let destination_dir = directory.path().join("destination");
            let candidate_dir = directory.path().join("candidate");
            std::fs::create_dir_all(&destination_dir).expect("create destination directory");
            std::fs::create_dir_all(&candidate_dir).expect("create candidate directory");
            let destination_path = destination_dir.join("dsp.json");
            let candidate_path = candidate_dir.join("dsp.json");
            let ready_path = directory.path().join(format!("{phase}.ready"));
            let (previous, previous_manifest) = save_process_recovery_fixture(
                &destination_path,
                "previous-generation",
                [90.0, 180.0],
                [78.0, 81.0],
                &[0.25, -0.5, 0.125],
            );
            let (candidate, candidate_manifest) = save_process_recovery_fixture(
                &candidate_path,
                "candidate-generation",
                [125.0, 250.0],
                [83.0, 79.0],
                &[-0.75, 0.33, 0.11],
            );

            kill_child_at_publication_phase(phase, &candidate_path, &destination_path, &ready_path);

            let journal_path = bundle_transaction_path(&destination_path);
            let before_journal = matches!(
                *phase,
                "previous_output_backup_synced" | "stage_assets_synced" | "transaction_root_synced"
            );
            let journal_removed = matches!(*phase, "journal_removed" | "cleanup_parent_synced");
            assert_eq!(
                journal_path.is_file(),
                !before_journal && !journal_removed,
                "unexpected journal state after child death at {phase}"
            );
            let journal_transaction_root = if journal_path.is_file() {
                let journal: BundleTransactionJournal = serde_json::from_slice(
                    &std::fs::read(&journal_path).expect("read interrupted journal"),
                )
                .expect("parse interrupted journal");
                Some(
                    transaction_directory(&destination_dir, &journal.transaction_directory)
                        .expect("validate interrupted transaction path"),
                )
            } else {
                None
            };

            let committed = matches!(
                *phase,
                "root_written"
                    | "root_parent_synced"
                    | "transaction_directory_removed"
                    | "journal_removed"
                    | "cleanup_parent_synced"
            );
            let (expected, expected_manifest) = if committed {
                (&candidate, &candidate_manifest)
            } else {
                (&previous, &previous_manifest)
            };
            assert_recovered_generation(&destination_path, expected, expected_manifest);
            recover_output_bundle_transactions(&destination_path)
                .expect("second recovery is idempotent");
            let recovered_again =
                load_output_bundle_frozen(&destination_path).expect("load after second recovery");
            assert_recovered_generation(&destination_path, expected, expected_manifest);
            assert_eq!(
                recovered_again.slim_graph_bytes(),
                expected.slim_graph_bytes(),
                "second load changed the selected generation at {phase}"
            );
            assert!(
                !journal_path.exists(),
                "journal remains after recovery at {phase}"
            );

            let transaction_directories = std::fs::read_dir(&destination_dir)
                .expect("list destination directory")
                .collect::<std::io::Result<Vec<_>>>()
                .expect("read every destination directory entry")
                .into_iter()
                .filter(|entry| {
                    entry
                        .file_name()
                        .to_string_lossy()
                        .starts_with(".autoeq-bundle-")
                })
                .map(|entry| entry.path())
                .collect::<Vec<_>>();
            if before_journal {
                assert!(
                    transaction_directories.iter().any(|path| path.is_dir()),
                    "a killed pre-journal staging directory should remain unowned at {phase}"
                );
            } else {
                assert!(
                    transaction_directories.is_empty(),
                    "recovery left a transaction directory at {phase}"
                );
            }
            if let Some(transaction_root) = journal_transaction_root {
                assert!(
                    !transaction_root.exists(),
                    "recovery left the journal-owned directory at {phase}"
                );
            }

            let relocated_path = directory.path().join("relocated/dsp.json");
            publish_output_bundle_from(&destination_path, &relocated_path)
                .expect("relocate recovered generation");
            assert_recovered_generation(&relocated_path, expected, expected_manifest);
            let relocated =
                load_output_bundle_frozen(&relocated_path).expect("load relocated generation");
            assert_eq!(
                relocated.slim_graph_bytes(),
                expected.slim_graph_bytes(),
                "relocation changed the selected root at {phase}"
            );
        }
    }

    #[test]
    fn resource_save_rewrites_references_and_load_verifies_resource_hashes() {
        let dir = tempfile::tempdir().expect("temp dir");
        let output_path = dir.path().join("dsp.json");
        let source_assets = dir.path().join("run-assets");
        std::fs::create_dir_all(&source_assets).expect("source assets");
        let bytes = b"fixture FIR bytes";
        std::fs::write(source_assets.join("impulse.wav"), bytes).expect("write FIR");
        let hash = sha256_hex(bytes);
        let mut output = convolution_graph("impulse.wav", &hash);

        save_output_bundle_with_resources(&mut output, &output_path, &source_assets)
            .expect("save resource bundle");

        let reference = format!("resources/{hash}.wav");
        assert_eq!(
            output.channels["L"].plugins[0].parameters["ir_file"],
            reference
        );
        assert_eq!(
            &output
                .metadata
                .as_ref()
                .unwrap()
                .final_convolution_sha256
                .as_ref()
                .unwrap()[&reference],
            &Some(hash.clone())
        );
        let bundle_assets = assets_dir_for(&output_path);
        assert_eq!(
            std::fs::read(bundle_assets.join(&reference)).expect("bundled FIR"),
            bytes
        );
        let restored = load_output_bundle(&output_path).expect("load resource bundle");
        assert_eq!(
            restored.channels["L"].plugins[0].parameters["ir_file"],
            reference
        );

        std::fs::write(bundle_assets.join(&reference), b"tampered").expect("tamper FIR");
        let error = load_output_bundle(&output_path).unwrap_err();
        assert!(
            error.to_string().contains("integrity validation"),
            "{error}"
        );
    }

    #[test]
    fn frozen_bundle_exposes_manifest_bound_graph_identity_and_resource_bytes() {
        let dir = tempfile::tempdir().expect("temp dir");
        let output_path = dir.path().join("dsp.json");
        let source_assets = dir.path().join("run-assets");
        std::fs::create_dir_all(&source_assets).expect("source assets");
        let fir_bytes = b"frozen FIR bytes";
        std::fs::write(source_assets.join("impulse.wav"), fir_bytes).expect("write FIR");
        let fir_hash = sha256_hex(fir_bytes);
        let mut output = convolution_graph("impulse.wav", &fir_hash);
        output.channels.get_mut("L").unwrap().initial_curve =
            Some(curve(vec![30.25, 50.5, 100.75], vec![80.0, 81.0, 79.0]));
        save_output_bundle_with_resources(&mut output, &output_path, &source_assets)
            .expect("save resource bundle");
        let reference = output.channels["L"].plugins[0].parameters["ir_file"]
            .as_str()
            .expect("rewritten resource reference")
            .to_owned();

        let frozen = load_output_bundle_frozen(&output_path).expect("load frozen bundle");
        assert_eq!(
            frozen.verification(),
            OutputBundleVerification::ManifestVerified
        );
        assert_eq!(
            frozen.raw_graph_sha256(),
            sha256_hex(frozen.slim_graph_bytes())
        );
        let slim_graph: DspGraph =
            serde_json::from_slice(frozen.slim_graph_bytes()).expect("parse captured slim graph");
        assert_eq!(
            frozen.slim_graph_identity(),
            &slim_graph_identity_and_validate_ledger(&slim_graph).expect("slim identity")
        );
        let restored_curve = frozen.output().channels["L"]
            .initial_curve
            .as_ref()
            .expect("hydrated curve");
        assert_eq!(restored_curve.freq, vec![30.25, 50.5, 100.75]);
        assert_eq!(restored_curve.spl, vec![80.0, 81.0, 79.0]);
        let resource = frozen.resource(&reference).expect("frozen FIR resource");
        assert_eq!(resource.relative_path(), reference);
        assert_eq!(resource.sha256(), fir_hash);
        assert_eq!(resource.bytes(), fir_bytes);
    }

    #[test]
    fn frozen_loader_parses_the_exact_bytes_captured_before_path_replacement() {
        let dir = tempfile::tempdir().expect("temp dir");
        let output_path = dir.path().join("dsp.json");
        let source_assets = dir.path().join("run-assets");
        std::fs::create_dir_all(&source_assets).expect("source assets");
        let fir_bytes = b"captured FIR bytes";
        std::fs::write(source_assets.join("impulse.wav"), fir_bytes).expect("write FIR");
        let fir_hash = sha256_hex(fir_bytes);
        let expected_curve = curve(vec![30.25, 50.5, 100.75], vec![80.0, 81.0, 79.0]);
        let mut output = convolution_graph("impulse.wav", &fir_hash);
        output.channels.get_mut("L").unwrap().initial_curve = Some(expected_curve.clone());
        save_output_bundle_with_resources(&mut output, &output_path, &source_assets)
            .expect("save resource bundle");

        let assets = assets_dir_for(&output_path);
        let index: serde_json::Value = serde_json::from_slice(
            &std::fs::read(assets.join(MEASUREMENTS_INDEX_FILENAME)).expect("read index"),
        )
        .expect("parse index");
        let metadata_name = index["channels"]["L"]["initial_curve_metadata"]
            .as_str()
            .expect("curve metadata path")
            .to_owned();
        let fir_name = output.channels["L"].plugins[0].parameters["ir_file"]
            .as_str()
            .expect("rewritten resource reference")
            .to_owned();

        let frozen = load_output_bundle_frozen_with_hooks(
            &output_path,
            |_| Ok(()),
            |assets| {
                std::fs::write(assets.join(MEASUREMENTS_INDEX_FILENAME), b"{}")
                    .expect("replace measurement index");
                std::fs::write(assets.join(&metadata_name), b"{}").expect("replace metadata");
                std::fs::write(assets.join("L__initial.csv"), b"changed CSV").expect("replace CSV");
                std::fs::write(assets.join(&fir_name), b"changed FIR").expect("replace FIR");
                Ok(())
            },
        )
        .expect("parse the captured bundle snapshot");

        let restored_curve = frozen.output().channels["L"]
            .initial_curve
            .as_ref()
            .expect("hydrated curve");
        assert_eq!(restored_curve.freq, expected_curve.freq);
        assert_eq!(restored_curve.spl, expected_curve.spl);
        assert_eq!(
            frozen.resource(&fir_name).expect("captured FIR").bytes(),
            fir_bytes
        );
        assert_eq!(
            std::fs::read(assets.join(&fir_name)).unwrap(),
            b"changed FIR"
        );
    }

    #[test]
    fn frozen_slim_identity_rejects_stale_ledger_payload_and_evidence() {
        let mut graph = DspGraph::new("1");
        graph.artifact_bundle_schema_version = Some(BUNDLE_MANIFEST_SCHEMA_VERSION);
        graph.add_channel("L", Vec::new());
        crate::final_ledger::finalize_output_ledger(
            &mut graph,
            &[],
            &crate::final_ledger::ReconciliationEvents::default(),
        )
        .expect("finalize ledger");
        let identity = slim_graph_identity_and_validate_ledger(&graph).expect("valid ledger");
        graph
            .correction_decisions
            .as_mut()
            .unwrap()
            .acceptance_evidence =
            Some(roomeq_model::acceptance_evidence::AcceptanceEvidence::new(
                serde_json::json!({"diagnostic": "bound to the slim graph"}),
                &identity.fingerprint,
            ));
        assert_eq!(
            slim_graph_identity_and_validate_ledger(&graph).expect("valid evidence"),
            identity
        );

        let mut bad_payload = graph.clone();
        bad_payload
            .correction_decisions
            .as_mut()
            .unwrap()
            .payload_binding
            .as_mut()
            .unwrap()
            .sha256 = "0".repeat(64);
        let error = slim_graph_identity_and_validate_ledger(&bad_payload).unwrap_err();
        assert!(error.to_string().contains("payload binding"), "{error}");

        let mut bad_evidence = graph;
        let ledger = bad_evidence.correction_decisions.as_mut().unwrap();
        ledger.payload_binding = None;
        ledger.acceptance_evidence =
            Some(roomeq_model::acceptance_evidence::AcceptanceEvidence::new(
                serde_json::json!({"diagnostic": "bound to another graph"}),
                "different-graph",
            ));
        let error = slim_graph_identity_and_validate_ledger(&bad_evidence).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("acceptance evidence does not match the canonical slim graph"),
            "identity was {}: {error}",
            identity.fingerprint
        );
    }

    #[test]
    fn frozen_loader_marks_unmanifested_legacy_graph_unverified() {
        let dir = tempfile::tempdir().expect("temp dir");
        let output_path = dir.path().join("legacy.json");
        let mut graph = DspGraph::new("legacy");
        graph.add_channel("L", Vec::new());
        std::fs::write(
            &output_path,
            serde_json::to_vec_pretty(&graph).expect("serialize graph"),
        )
        .expect("write legacy output");

        let frozen = load_output_bundle_frozen(&output_path).expect("load legacy output");

        assert_eq!(
            frozen.verification(),
            OutputBundleVerification::LegacyUnverified
        );
        assert!(frozen.resources().next().is_none());
    }

    #[test]
    fn resource_save_failures_leave_the_previous_bundle_unchanged() {
        let dir = tempfile::tempdir().expect("temp dir");
        let output_path = dir.path().join("dsp.json");
        let (old_root, assets) = existing_bundle_fixture(&output_path);
        let source_assets = dir.path().join("run-assets");
        std::fs::create_dir_all(&source_assets).expect("source assets");
        std::fs::write(source_assets.join("impulse.wav"), b"actual FIR").expect("write FIR");
        let mut output = convolution_graph("impulse.wav", "incorrect-final-hash");

        let error = save_output_bundle_with_resources(&mut output, &output_path, &source_assets)
            .unwrap_err();

        assert!(error.to_string().contains("SHA-256 validation"), "{error}");
        assert_eq!(std::fs::read(&output_path).unwrap(), old_root);
        assert_eq!(
            std::fs::read(assets.join("preserved.wav")).unwrap(),
            b"prior support"
        );
        assert_eq!(
            output.channels["L"].plugins[0].parameters["ir_file"], "impulse.wav",
            "the input graph is unchanged when resource binding fails"
        );
    }

    #[test]
    fn basic_bundle_save_refuses_unowned_convolution_resources() {
        let dir = tempfile::tempdir().expect("temp dir");
        let output_path = dir.path().join("dsp.json");
        let (old_root, assets) = existing_bundle_fixture(&output_path);
        let mut output = convolution_graph("missing.wav", "unused");

        let error = save_output_bundle(&mut output, &output_path).unwrap_err();

        assert!(
            error
                .to_string()
                .contains("save_output_bundle_with_resources")
        );
        assert_eq!(std::fs::read(&output_path).unwrap(), old_root);
        assert_eq!(
            std::fs::read(assets.join("preserved.wav")).unwrap(),
            b"prior support"
        );
    }

    #[test]
    fn resource_prepare_callback_runs_after_refs_are_bound_and_can_abort_commit() {
        let dir = tempfile::tempdir().expect("temp dir");
        let output_path = dir.path().join("dsp.json");
        let (old_root, assets) = existing_bundle_fixture(&output_path);
        let source_assets = dir.path().join("run-assets");
        std::fs::create_dir_all(&source_assets).expect("source assets");
        let bytes = b"fixture FIR bytes";
        std::fs::write(source_assets.join("impulse.wav"), bytes).expect("write FIR");
        let hash = sha256_hex(bytes);
        let mut output = convolution_graph("impulse.wav", &hash);
        let mut saw_bound_candidate = false;

        let error = save_output_bundle_with_resources_and_prepare(
            &mut output,
            &output_path,
            &source_assets,
            &mut |candidate, staged_assets| {
                let reference = candidate.channels["L"].plugins[0].parameters["ir_file"]
                    .as_str()
                    .expect("rewritten reference");
                assert_eq!(reference, format!("resources/{hash}.wav"));
                assert_eq!(std::fs::read(staged_assets.join(reference)).unwrap(), bytes);
                saw_bound_candidate = true;
                Err(io_invalid("test preparation failure").into())
            },
        )
        .unwrap_err();

        assert!(saw_bound_candidate);
        assert!(error.to_string().contains("test preparation failure"));
        assert_eq!(std::fs::read(&output_path).unwrap(), old_root);
        assert_eq!(
            std::fs::read(assets.join("preserved.wav")).unwrap(),
            b"prior support"
        );
        assert_eq!(
            output.channels["L"].plugins[0].parameters["ir_file"], "impulse.wav",
            "the source graph is only replaced after commit"
        );
    }

    #[test]
    fn bundle_copy_publish_preserves_exact_graph_bytes_and_resource_bindings() {
        let source = tempfile::tempdir().expect("source dir");
        let destination = tempfile::tempdir().expect("destination dir");
        let source_path = source.path().join("dsp.json");
        let destination_path = destination.path().join("dsp.json");
        let source_assets = source.path().join("run-assets");
        std::fs::create_dir_all(&source_assets).expect("source assets");
        let bytes = b"fixture FIR bytes";
        std::fs::write(source_assets.join("impulse.wav"), bytes).expect("write FIR");
        let mut output = convolution_graph("impulse.wav", &sha256_hex(bytes));
        save_output_bundle_with_resources(&mut output, &source_path, &source_assets)
            .expect("save source bundle");
        let expected_root = std::fs::read(&source_path).expect("source root");

        publish_output_bundle_from(&source_path, &destination_path).expect("publish exact bundle");

        assert_eq!(std::fs::read(&destination_path).unwrap(), expected_root);
        let restored = load_output_bundle(&destination_path).expect("load published bundle");
        assert_eq!(
            restored.channels["L"].plugins[0].parameters["ir_file"],
            output.channels["L"].plugins[0].parameters["ir_file"]
        );
    }

    #[test]
    fn bundle_copy_publish_rejects_source_root_mutation_before_replacing_destination() {
        let source = tempfile::tempdir().expect("source dir");
        let destination = tempfile::tempdir().expect("destination dir");
        let source_path = source.path().join("dsp.json");
        let destination_path = destination.path().join("dsp.json");
        let mut source_graph = DspGraph::new("source");
        source_graph.add_channel("L", Vec::new());
        save_output_bundle(&mut source_graph, &source_path).expect("save source bundle");
        let (old_root, old_assets) = existing_bundle_fixture(&destination_path);

        let error = publish_output_bundle_from_with_hook(
            &source_path,
            &destination_path,
            |source_root, _| std::fs::write(source_root, b"changed source graph"),
        )
        .unwrap_err();

        assert!(
            error.to_string().contains("source graph changed"),
            "{error}"
        );
        assert_eq!(std::fs::read(&destination_path).unwrap(), old_root);
        assert_eq!(
            std::fs::read(old_assets.join("preserved.wav")).unwrap(),
            b"prior support"
        );
    }

    #[test]
    fn bundle_copy_publish_rejects_resource_mutation_before_replacing_destination() {
        let source = tempfile::tempdir().expect("source dir");
        let destination = tempfile::tempdir().expect("destination dir");
        let source_path = source.path().join("dsp.json");
        let destination_path = destination.path().join("dsp.json");
        let source_assets = source.path().join("run-assets");
        std::fs::create_dir_all(&source_assets).expect("source assets");
        let fir_bytes = b"fixture FIR bytes";
        std::fs::write(source_assets.join("impulse.wav"), fir_bytes).expect("write FIR");
        let mut source_graph = convolution_graph("impulse.wav", &sha256_hex(fir_bytes));
        save_output_bundle_with_resources(&mut source_graph, &source_path, &source_assets)
            .expect("save source bundle");
        let resource = source_graph.channels["L"].plugins[0].parameters["ir_file"]
            .as_str()
            .expect("rewritten resource reference")
            .to_owned();
        let (old_root, old_assets) = existing_bundle_fixture(&destination_path);

        let error =
            publish_output_bundle_from_with_hook(&source_path, &destination_path, |_, assets| {
                std::fs::write(assets.join(resource), b"changed FIR")
            })
            .unwrap_err();

        assert!(
            error.to_string().contains("copied source support files"),
            "{error}"
        );
        assert_eq!(std::fs::read(&destination_path).unwrap(), old_root);
        assert_eq!(
            std::fs::read(old_assets.join("preserved.wav")).unwrap(),
            b"prior support"
        );
    }

    #[test]
    fn manifested_member_copy_never_stages_past_the_declared_size() {
        let source = tempfile::tempdir().expect("source dir");
        let destination = tempfile::tempdir().expect("destination dir");
        let expected_bytes = b"old";
        std::fs::write(source.path().join("resource.wav"), b"new resource grew").unwrap();
        let record = BundleFileRecord {
            size_bytes: expected_bytes.len() as u64,
            sha256: sha256_hex(expected_bytes),
        };

        let error = copy_manifested_member(
            source.path(),
            destination.path(),
            Path::new("resource.wav"),
            &record,
        )
        .unwrap_err();

        assert!(error.to_string().contains("changed during copy"));
        assert!(
            std::fs::metadata(destination.path().join("resource.wav"))
                .unwrap()
                .len()
                <= record.size_bytes
        );
    }

    #[test]
    fn bundle_copy_publish_rejects_unlisted_source_files_before_commit() {
        let source = tempfile::tempdir().expect("source dir");
        let destination = tempfile::tempdir().expect("destination dir");
        let source_path = source.path().join("dsp.json");
        let destination_path = destination.path().join("dsp.json");
        let mut source_graph = DspGraph::new("source");
        source_graph.add_channel("L", Vec::new());
        save_output_bundle(&mut source_graph, &source_path).expect("save source bundle");
        let (old_root, old_assets) = existing_bundle_fixture(&destination_path);

        let error =
            publish_output_bundle_from_with_hook(&source_path, &destination_path, |_, assets| {
                std::fs::write(assets.join("unlisted.bin"), vec![0_u8; 1024])
            })
            .unwrap_err();

        assert!(
            error.to_string().contains("copied source support files"),
            "{error}"
        );
        assert_eq!(std::fs::read(&destination_path).unwrap(), old_root);
        assert_eq!(
            std::fs::read(old_assets.join("preserved.wav")).unwrap(),
            b"prior support"
        );
    }

    #[test]
    fn legacy_scratch_copy_refuses_oversized_members_before_copying_them() {
        let source = tempfile::tempdir().expect("source dir");
        let destination = tempfile::tempdir().expect("destination dir");
        let path = source.path().join("large.bin");
        let file = std::fs::File::create(&path).expect("create sparse file");
        file.set_len(MAX_BUNDLE_MEMBER_BYTES + 1)
            .expect("set sparse file size");

        let error =
            copy_existing_asset_tree(source.path(), destination.path(), false, None).unwrap_err();

        assert!(error.to_string().contains("member exceeds"), "{error}");
        assert!(!destination.path().join("large.bin").exists());
    }

    #[test]
    fn marked_graph_cannot_copy_support_without_its_manifest() {
        let source = tempfile::tempdir().expect("source dir");
        let destination = tempfile::tempdir().expect("destination dir");
        std::fs::write(source.path().join("response.csv"), b"1,2\n").unwrap();

        let error =
            copy_existing_asset_tree(source.path(), destination.path(), true, None).unwrap_err();

        assert!(
            error.to_string().contains("missing its integrity manifest"),
            "{error}"
        );
        assert!(!destination.path().join("response.csv").exists());
    }

    #[test]
    fn load_recovers_a_crash_after_support_swap_before_root_replace() {
        let dir = tempfile::tempdir().expect("temp dir");
        let output_path = dir.path().join("dsp.json");
        let (old_root, assets) = existing_bundle_fixture(&output_path);
        let old_hash = sha256_hex(&old_root);
        let mut next = DspGraph::new("next");
        next.add_channel("L", Vec::new());
        let next_bytes = serde_json::to_vec_pretty(&serde_json::to_value(&next).unwrap()).unwrap();
        let next_hash = sha256_hex(&next_bytes);

        let transaction = ArtifactBundleStaging::new(dir.path()).expect("transaction staging");
        let transaction_root = transaction.root().to_path_buf();
        let generation = transaction_root
            .file_name()
            .unwrap()
            .to_str()
            .unwrap()
            .to_string();
        let next_assets = transaction_root.join("next_assets");
        std::fs::create_dir(&next_assets).unwrap();
        std::fs::write(next_assets.join("new.wav"), b"new support").unwrap();
        write_bundle_manifest(&next_assets, &generation, &next.version, &next_hash).unwrap();
        std::fs::rename(&assets, transaction_root.join("previous_assets")).unwrap();
        std::fs::rename(&next_assets, &assets).unwrap();
        let transaction_root = transaction.keep();
        let journal = BundleTransactionJournal {
            schema_version: BUNDLE_MANIFEST_SCHEMA_VERSION,
            transaction_directory: generation.clone(),
            generation,
            had_assets: true,
            had_output: true,
            old_output_sha256: Some(old_hash),
            new_output_sha256: next_hash,
        };
        write_file_atomically(
            &bundle_transaction_path(&output_path),
            &serde_json::to_vec(&journal).unwrap(),
        )
        .unwrap();

        let recovered = load_output_bundle(&output_path).expect("recover prior bundle");
        assert_eq!(recovered.version, "prior");
        assert_eq!(
            std::fs::read(assets.join("preserved.wav")).unwrap(),
            b"prior support"
        );
        assert!(!bundle_transaction_path(&output_path).exists());
        assert!(!transaction_root.exists());
    }

    #[test]
    fn manifest_rejects_changed_graph_or_member_but_ignores_mutable_run_files() {
        let dir = tempfile::tempdir().expect("temp dir");
        let output_path = dir.path().join("dsp.json");
        let mut output = DspGraph::new("1");
        output.add_channel("L", Vec::new());
        output.channels.get_mut("L").expect("channel").initial_curve =
            Some(curve(vec![100.0], vec![80.0]));
        save_output_bundle(&mut output, &output_path).expect("save bundle");
        let assets = assets_dir_for(&output_path);
        std::fs::write(assets.join(RUN_LOG_FILENAME), "later run log").unwrap();
        std::fs::write(assets.join(RUN_MANIFEST_FILENAME), "later run manifest").unwrap();
        assert!(load_output_bundle(&output_path).is_ok());

        std::fs::write(assets.join("L__initial.csv"), b"freq,spl\\n100,0\\n").unwrap();
        assert!(
            load_output_bundle(&output_path).is_err(),
            "changed support bytes must be rejected before overlays are restored"
        );
    }

    #[test]
    fn manifested_loader_rechecks_the_bytes_after_manifest_verification() {
        let dir = tempfile::tempdir().expect("temp dir");
        let output_path = dir.path().join("dsp.json");
        let mut output = DspGraph::new("1");
        output.add_channel("L", Vec::new());
        output.channels.get_mut("L").expect("channel").initial_curve =
            Some(curve(vec![100.0], vec![80.0]));
        save_output_bundle(&mut output, &output_path).expect("save bundle");

        let error = load_output_bundle_with_hook(&output_path, |assets| {
            std::fs::write(assets.join(MEASUREMENTS_INDEX_FILENAME), b"{}")
        })
        .expect_err("index mutation after manifest validation must fail closed");
        assert!(
            error
                .to_string()
                .contains("measurement index failed artifact bundle integrity"),
            "{error}"
        );
    }

    #[test]
    fn manifested_curve_parser_uses_rechecked_sidecar_bytes() {
        let dir = tempfile::tempdir().expect("temp dir");
        let output_path = dir.path().join("dsp.json");
        let mut output = DspGraph::new("1");
        output.add_channel("L", Vec::new());
        output.channels.get_mut("L").expect("channel").initial_curve =
            Some(curve(vec![100.0], vec![80.0]));
        save_output_bundle(&mut output, &output_path).expect("save bundle");
        let assets = assets_dir_for(&output_path);
        let index: serde_json::Value = serde_json::from_slice(
            &std::fs::read(assets.join(MEASUREMENTS_INDEX_FILENAME)).expect("read index"),
        )
        .expect("parse index");
        let metadata_name = index["channels"]["L"]["initial_curve_metadata"]
            .as_str()
            .expect("metadata sidecar name")
            .to_owned();

        let error = load_output_bundle_with_hook(&output_path, |assets| {
            std::fs::write(assets.join(metadata_name), b"{}")
        })
        .expect_err("metadata mutation after manifest validation must fail closed");
        assert!(
            error
                .to_string()
                .contains("curve metadata sidecar failed artifact bundle integrity"),
            "{error}"
        );
    }

    #[test]
    fn new_bundle_marker_requires_manifest_and_unknown_marker_fails_closed() {
        let dir = tempfile::tempdir().expect("temp dir");
        let output_path = dir.path().join("dsp.json");
        let mut output = DspGraph::new("1");
        output.add_channel("L", Vec::new());
        save_output_bundle(&mut output, &output_path).expect("save bundle");
        assert_eq!(output.artifact_bundle_schema_version, Some(1));

        std::fs::remove_file(assets_dir_for(&output_path).join(ARTIFACT_BUNDLE_MANIFEST_FILENAME))
            .expect("remove required manifest");
        let error = load_output_bundle(&output_path).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("requires a missing artifact bundle manifest")
        );

        let mut root: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&output_path).unwrap()).unwrap();
        root["artifact_bundle_schema_version"] = serde_json::json!(2);
        std::fs::write(&output_path, serde_json::to_vec(&root).unwrap()).unwrap();
        let error = load_output_bundle(&output_path).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("unsupported artifact bundle schema marker")
        );
    }

    #[test]
    fn manifested_graph_without_marker_still_runs_native_graph_validation() {
        let dir = tempfile::tempdir().expect("temp dir");
        let output_path = dir.path().join("dsp.json");
        let mut output = DspGraph::new("1");
        output.add_channel("L", Vec::new());
        save_output_bundle(&mut output, &output_path).expect("save bundle");

        let mut root: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&output_path).expect("read root"))
                .expect("parse root");
        root["artifact_bundle_schema_version"] = serde_json::Value::Null;
        root["channels"] = serde_json::json!({});
        let root_bytes = serde_json::to_vec(&root).expect("serialize malformed graph");
        std::fs::write(&output_path, &root_bytes).expect("replace root");
        let assets = assets_dir_for(&output_path);
        let manifest: BundleManifest = serde_json::from_slice(
            &std::fs::read(assets.join(ARTIFACT_BUNDLE_MANIFEST_FILENAME)).expect("read manifest"),
        )
        .expect("parse manifest");
        write_bundle_manifest(
            &assets,
            &manifest.generation,
            &manifest.graph_schema_version,
            &sha256_hex(&root_bytes),
        )
        .expect("rebind malformed root to manifest");

        let error = load_output_bundle_frozen(&output_path).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("DSP graph requires at least one channel"),
            "{error}"
        );
    }

    #[test]
    fn manifest_rejects_casefold_duplicate_and_oversized_metadata() {
        let dir = tempfile::tempdir().expect("temp dir");
        let output_path = dir.path().join("dsp.json");
        let mut output = DspGraph::new("1");
        output.add_channel("L", Vec::new());
        save_output_bundle(&mut output, &output_path).expect("save bundle");
        let assets = assets_dir_for(&output_path);
        let manifest_path = assets.join(ARTIFACT_BUNDLE_MANIFEST_FILENAME);
        let mut manifest: BundleManifest =
            serde_json::from_slice(&std::fs::read(&manifest_path).unwrap()).unwrap();
        let member_bytes = b"collision fixture";
        std::fs::write(assets.join("A"), member_bytes).unwrap();
        let record = BundleFileRecord {
            size_bytes: member_bytes.len() as u64,
            sha256: sha256_hex(member_bytes),
        };
        manifest.files.insert("A".to_owned(), record.clone());
        manifest.files.insert("a".to_owned(), record);
        std::fs::write(&manifest_path, serde_json::to_vec(&manifest).unwrap()).unwrap();
        let error = load_output_bundle(&output_path).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("invalid or duplicate file entry")
        );

        std::fs::write(
            &manifest_path,
            vec![b' '; MAX_BUNDLE_MANIFEST_BYTES as usize + 1],
        )
        .unwrap();
        let error = load_output_bundle(&output_path).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("manifest exceeds its size limit")
        );
    }

    #[test]
    fn manifested_curve_metadata_is_validated_after_hash_verification() {
        let dir = tempfile::tempdir().expect("temp dir");
        let output_path = dir.path().join("dsp.json");
        let mut output = DspGraph::new("1");
        output.add_channel("L", Vec::new());
        let mut native_curve = curve(vec![100.0, 1000.0], vec![80.0, 81.0]);
        native_curve.coherence = Some(vec![0.9, 0.95]);
        output.channels.get_mut("L").unwrap().initial_curve = Some(native_curve);
        save_output_bundle(&mut output, &output_path).expect("save bundle");

        let assets = assets_dir_for(&output_path);
        let index: serde_json::Value = serde_json::from_slice(
            &std::fs::read(assets.join(MEASUREMENTS_INDEX_FILENAME)).unwrap(),
        )
        .unwrap();
        let metadata_file = index["channels"]["L"]["initial_curve_metadata"]
            .as_str()
            .unwrap();
        let metadata_path = assets.join(metadata_file);
        let mut curve_value: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&metadata_path).unwrap()).unwrap();
        curve_value["coherence"] = serde_json::json!([0.9, 1.1]);
        std::fs::write(
            &metadata_path,
            serde_json::to_vec_pretty(&curve_value).unwrap(),
        )
        .unwrap();
        let manifest: BundleManifest = serde_json::from_slice(
            &std::fs::read(assets.join(ARTIFACT_BUNDLE_MANIFEST_FILENAME)).unwrap(),
        )
        .unwrap();
        write_bundle_manifest(
            &assets,
            &manifest.generation,
            &manifest.graph_schema_version,
            &manifest.graph_sha256,
        )
        .unwrap();

        let error = load_output_bundle(&output_path).unwrap_err();
        assert!(error.to_string().contains("coherence"), "{error}");
    }

    #[test]
    fn manifest_rejects_root_json_changes() {
        let dir = tempfile::tempdir().expect("temp dir");
        let output_path = dir.path().join("dsp.json");
        let mut output = DspGraph::new("1");
        output.add_channel("L", Vec::new());
        save_output_bundle(&mut output, &output_path).expect("save bundle");
        std::fs::write(&output_path, br#"{"version":"mutated"}"#).unwrap();

        assert!(load_output_bundle(&output_path).is_err());
    }

    #[test]
    fn convolution_lookup_prefers_parent_then_assets_dir() {
        let dir = tempfile::tempdir().expect("temp dir");
        let output_path = dir.path().join("dsp.json");
        let assets = assets_dir_for(&output_path);
        std::fs::create_dir_all(&assets).expect("assets dir");
        std::fs::write(assets.join("L_fir_48000hz.wav"), b"waves").expect("write wav");
        let resolved = resolve_convolution_path("L_fir_48000hz.wav", &output_path);
        assert_eq!(resolved, assets.join("L_fir_48000hz.wav"));
        let bytes = read_convolution_bytes("L_fir_48000hz.wav", &output_path).expect("read");
        assert_eq!(bytes, b"waves");
    }
}
