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
//! The Python viewer (`scripts/src/loaders.py`) re-injects them from the
//! sibling directory, so plots are unchanged while `dsp.json` stays small.

use std::path::{Path, PathBuf};

use roomeq_model::{
    ChannelEarlyLateCurves, ChannelResonanceDecays, ChannelWaterfall, ChannelWavelet, CurveData,
    DspGraph, IrWaveform, MeasuredRoomAcoustics,
};

/// Name of the run log written inside the assets directory.
pub const RUN_LOG_FILENAME: &str = "roomeq.log";
/// Name of the run manifest written inside the assets directory.
pub const RUN_MANIFEST_FILENAME: &str = "manifest.json";

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

/// Read convolution bytes, checking the output parent first and the sibling
/// assets directory second.
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

fn write_curve_csv(path: &Path, curve: &CurveData) -> std::io::Result<()> {
    let mut out = String::with_capacity(curve.freq.len() * 24);
    let has_phase = curve.phase.as_ref().is_some_and(|p| !p.is_empty());
    if has_phase {
        out.push_str("freq,spl,phase\n");
    } else {
        out.push_str("freq,spl\n");
    }
    for (index, (freq, spl)) in curve.freq.iter().zip(curve.spl.iter()).enumerate() {
        if has_phase {
            let phase = curve
                .phase
                .as_ref()
                .and_then(|p| p.get(index))
                .copied()
                .unwrap_or(f64::NAN);
            out.push_str(&format!("{freq:.6},{spl:.6},{phase:.6}\n"));
        } else {
            out.push_str(&format!("{freq:.6},{spl:.6}\n"));
        }
    }
    std::fs::write(path, out)
}

fn write_ir_csv(path: &Path, ir: &IrWaveform) -> std::io::Result<()> {
    let mut out = String::with_capacity(ir.time_ms.len() * 24);
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
    match write_curve_csv(&path, &curve) {
        Ok(()) => {
            record_file(extracted, index, channel, kind, &file_name);
            None
        }
        Err(_) => Some(curve),
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
    for name in deployed {
        if let Some(curve) = output.deployed_source_curves.remove(&name) {
            let tag = sanitize_name(&name);
            let file_name = format!("deployed__{tag}.csv");
            let path = assets_dir.join(&file_name);
            match write_curve_csv(&path, &curve) {
                Ok(()) => {
                    extracted.files.push(file_name.clone());
                    deployed_index.insert(name.clone(), serde_json::Value::String(file_name));
                }
                Err(_) => {
                    output.deployed_source_curves.insert(name, curve);
                }
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
        let index_value = serde_json::Value::Object(root);
        let path = assets_dir.join(MEASUREMENTS_INDEX_FILENAME);
        if write_json_file(&path, &index_value).is_ok() {
            extracted
                .files
                .push(MEASUREMENTS_INDEX_FILENAME.to_string());
            extracted.files.sort();
        }
        extracted.index = index_value;
    }
    extracted
}

/// Save a DSP output as a small JSON plus sibling assets directory.
///
/// Creates `<stem>_files/`, extracts measurement blobs there, then writes
/// the slim JSON to `output_path`. Returns the relative asset file names
/// written (excluding the JSON itself).
pub fn save_output_bundle(
    output: &mut DspGraph,
    output_path: &Path,
) -> Result<Vec<String>, Box<dyn std::error::Error>> {
    let assets_dir = assets_dir_for(output_path);
    std::fs::create_dir_all(&assets_dir)?;
    if let Some(parent) = output_path.parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent)?;
    }
    let extracted = extract_measurements_to_assets(output, &assets_dir);
    crate::output::save_dsp_chain(output, output_path)?;
    Ok(extracted.files)
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

fn read_curve_csv(path: &Path) -> Result<CurveData, Box<dyn std::error::Error>> {
    let text = std::fs::read_to_string(path)?;
    let mut lines = text.lines();
    let header = lines.next().ok_or_else(|| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("curve CSV '{}' is empty", path.display()),
        )
    })?;
    let columns: Vec<&str> = header.split(',').collect();
    if columns.len() < 2 || columns[0] != "freq" || columns[1] != "spl" {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("curve CSV '{}' has an unexpected header", path.display()),
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

fn read_ir_csv(path: &Path) -> Result<IrWaveform, Box<dyn std::error::Error>> {
    let text = std::fs::read_to_string(path)?;
    let mut lines = text.lines();
    let header = lines.next().ok_or_else(|| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("IR CSV '{}' is empty", path.display()),
        )
    })?;
    if header != "time_ms,amplitude" {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("IR CSV '{}' has an unexpected header", path.display()),
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
    Ok(IrWaveform { time_ms, amplitude })
}

fn read_json_blob<T>(path: &Path) -> Result<T, Box<dyn std::error::Error>>
where
    T: serde::de::DeserializeOwned,
{
    let text = std::fs::read_to_string(path)?;
    Ok(serde_json::from_str(&text)?)
}

/// Load a native output, re-injecting measurement blobs extracted into the
/// sibling `<stem>_files` directory.
///
/// Legacy outputs with embedded curves load unchanged: only fields that are
/// absent from the JSON are restored from `measurements_index.json`, and
/// missing asset files are skipped rather than treated as errors.
pub fn load_output_bundle(output_path: &Path) -> Result<DspGraph, Box<dyn std::error::Error>> {
    let text = std::fs::read_to_string(output_path)?;
    let mut output: DspGraph = serde_json::from_str(&text)?;
    let assets_dir = assets_dir_for(output_path);
    let index_path = assets_dir.join(MEASUREMENTS_INDEX_FILENAME);
    let Ok(index_text) = std::fs::read_to_string(&index_path) else {
        return Ok(output);
    };
    let index: serde_json::Value = serde_json::from_str(&index_text)?;
    let channels = index
        .get("channels")
        .and_then(|value| value.as_object())
        .cloned()
        .unwrap_or_default();
    for (name, entry) in &channels {
        let Some(chain) = output.channels.get_mut(name) else {
            continue;
        };
        let Some(fields) = entry.as_object() else {
            continue;
        };
        for (kind, file) in fields {
            let Some(file) = file.as_str() else {
                continue;
            };
            let path = assets_dir.join(file);
            if !path.is_file() {
                continue;
            }
            match kind.as_str() {
                "initial_curve" if chain.initial_curve.is_none() => {
                    chain.initial_curve = read_curve_csv(&path).ok();
                }
                "final_curve" if chain.final_curve.is_none() => {
                    chain.final_curve = read_curve_csv(&path).ok();
                }
                "eq_response" if chain.eq_response.is_none() => {
                    chain.eq_response = read_curve_csv(&path).ok();
                }
                "target_curve" if chain.target_curve.is_none() => {
                    chain.target_curve = read_curve_csv(&path).ok();
                }
                "pre_ir" if chain.pre_ir.is_none() => {
                    chain.pre_ir = read_ir_csv(&path).ok();
                }
                "post_ir" if chain.post_ir.is_none() => {
                    chain.post_ir = read_ir_csv(&path).ok();
                }
                "early_late_curves" if chain.early_late_curves.is_none() => {
                    chain.early_late_curves = read_json_blob::<ChannelEarlyLateCurves>(&path).ok();
                }
                "waterfall" if chain.waterfall.is_none() => {
                    chain.waterfall = read_json_blob::<ChannelWaterfall>(&path).ok();
                }
                "resonance_decays" if chain.resonance_decays.is_none() => {
                    chain.resonance_decays = read_json_blob::<ChannelResonanceDecays>(&path).ok();
                }
                "wavelet" if chain.wavelet.is_none() => {
                    chain.wavelet = read_json_blob::<ChannelWavelet>(&path).ok();
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
                        driver.measured_acoustics =
                            read_json_blob::<MeasuredRoomAcoustics>(&path).ok();
                    }
                    if let Some(rest) = kind.strip_prefix("driver")
                        && let Some((index_text, suffix)) = rest.split_once('_')
                        && suffix.ends_with("_initial_curve")
                        && let Ok(driver_index) = index_text.parse::<usize>()
                        && let Some(drivers) = chain.drivers.as_mut()
                        && let Some(driver) = drivers.get_mut(driver_index)
                        && driver.initial_curve.is_none()
                    {
                        driver.initial_curve = read_curve_csv(&path).ok();
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
                continue;
            };
            let path = assets_dir.join(file);
            if !path.is_file() {
                continue;
            }
            if let Ok(curve) = read_curve_csv(&path) {
                output.deployed_source_curves.insert(name.clone(), curve);
            }
        }
    }
    Ok(output)
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
        let restored = read_ir_csv(&path).expect("reload native IR");
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
        chain.initial_curve = Some(CurveData {
            freq: vec![100.0, 1000.0],
            spl: vec![80.0, 81.0],
            phase: Some(vec![0.0, 10.0]),
            norm_range: None,
            noise_floor_db: None,
            coherence: None,
        });
        chain.post_ir = Some(IrWaveform {
            time_ms: vec![0.0, 0.1],
            amplitude: vec![1.0, 0.5],
        });
        output.deployed_source_curves.insert(
            "L".to_string(),
            curve(vec![100.0, 1000.0], vec![79.5, 80.5]),
        );

        save_output_bundle(&mut output, &output_path).expect("save bundle");
        assert!(output.channels["L"].initial_curve.is_none());

        let restored = load_output_bundle(&output_path).expect("load bundle");
        let initial = restored.channels["L"]
            .initial_curve
            .as_ref()
            .expect("initial curve restored");
        assert_eq!(initial.spl, vec![80.0, 81.0]);
        assert_eq!(
            initial.phase.as_ref().expect("phase restored"),
            &vec![0.0, 10.0]
        );
        let post_ir = restored.channels["L"]
            .post_ir
            .as_ref()
            .expect("post IR restored");
        assert_eq!(post_ir.amplitude, vec![1.0, 0.5]);
        assert_eq!(restored.deployed_source_curves["L"].spl, vec![79.5, 80.5]);
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
