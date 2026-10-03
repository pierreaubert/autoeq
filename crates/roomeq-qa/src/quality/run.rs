use super::apply::apply_group_delay_qa_passthrough_eq;
use super::apply::apply_mutation;
use super::apply::apply_option_override;
use super::apply::apply_qa_overrides;
use super::apply::clamp_strict_measured_maxeval;
use super::consts::CORRECTION_OUT_OF_BAND_SPREAD_DB;
use super::consts::CROSS_MODE_BASS_MAX_RMS_DB;
use super::consts::CROSS_MODE_BASS_MEDIAN_RMS_DB;
use super::consts::CROSS_MODE_FR_MAX_DIFF_DB;
use super::consts::CROSS_MODE_MAIN_MEDIAN_RMS_DB;
use super::consts::CROSS_MODE_RATIO_LIMIT;
use super::consts::CROSS_MODE_SCORE_RATIO_LIMIT;
use super::consts::CROSS_MODE_TIMING_MAX_STD_MS;
use super::consts::CROSS_MODE_UPPER_MEDIAN_RMS_DB;
use super::consts::FIR_MUTATIONS;
use super::consts::IIR_MUTATIONS;
use super::consts::MIXED_MUTATIONS;
use super::consts::MIXED_PHASE_MUTATIONS;
use super::consts::SAMPLE_RATE;
use super::consts::TEMP_DIR_COUNTER;
use super::group::group_delay_std_dev;
use super::group_delay_qa_profile::disable_option;
use super::group_delay_qa_profile::prepare_option_measurement_paths;
use super::metric_scorecard::MetricScorecard;
use super::metric_scorecard::compare_scorecards;
use super::metric_scorecard::compute_scorecard;
use super::metric_scorecard::evaluate_scorecard;
use super::metric_scorecard::placeholder_scorecard;
use super::misc::convergence_epsilon;
use super::misc::level_matched_rms_curve_difference_db;
use super::misc::load_config_for_generic_path;
use super::misc::load_config_for_path;
use super::misc::max_curve_difference_db;
use super::mutation::Mutation;
use super::option::isolate_schroeder_split_from_multi_measurement;
use super::option::option_gd_profile;
use super::option::option_is_group_delay;
use super::option::option_needs_gd_trusted_measurements;
use super::option::option_needs_multi_measurement_paths;
use super::option::option_needs_multisub_multi_seat_paths;
use super::option_override::OptionOverride;
use super::types::TestResult;
use super::validate::validate_option_effect;
use anyhow::{Context, Result};
use roomeq_engine::room_result::RoomOptimizationResult;
use roomeq_model::{Curve, ProcessingMode, RoomConfig, TargetResponseConfig, TargetShape};
use roomeq_workflow::load_config;
use std::fmt::Write as _;
use std::path::Path;
use std::sync::atomic::Ordering;

pub(super) struct AssessedOptimization {
    result: RoomOptimizationResult,
    pub(super) scorecard: MetricScorecard,
}

impl std::ops::Deref for AssessedOptimization {
    type Target = RoomOptimizationResult;
    fn deref(&self) -> &Self::Target {
        &self.result
    }
}

#[cfg(test)]
pub(super) mod finalization_diagnostic {
    use super::*;
    use anyhow::{Context, bail, ensure};
    use roomeq_workflow::{
        FinalizationDiagnosticSink, FinalizationDiagnosticTrial, RoomPipeline, RoomPipelineRequest,
    };
    use serde::Serialize;
    use sha2::{Digest, Sha256};
    use std::collections::BTreeMap;
    use std::fs::{self, File, OpenOptions};
    use std::io::{Read, Write};
    use std::path::{Path, PathBuf};
    use std::process::Command;
    use std::sync::Mutex;

    const MAX_EVENT_BYTES: usize = 128 * 1024 * 1024;
    const MAX_INPUT_FILES: usize = 128;
    const MAX_INPUT_FILE_BYTES: u64 = 256 * 1024 * 1024;
    const MAX_INPUT_TREE_BYTES: u64 = 1024 * 1024 * 1024;

    #[derive(Debug, Clone, Serialize, PartialEq, Eq)]
    struct FileIdentity {
        path: String,
        bytes: u64,
        sha256: String,
    }

    #[derive(Debug, Clone, Serialize)]
    struct EventIdentity {
        bytes: usize,
        sha256: String,
    }

    /// Creates one fresh evidence directory and publishes each JSON event
    /// atomically without replacing any existing event.
    pub(super) struct FinalizationDiagnosticDirectory {
        root: PathBuf,
        events: Mutex<BTreeMap<String, EventIdentity>>,
    }

    impl FinalizationDiagnosticDirectory {
        pub(super) fn create(root: &Path) -> Result<Self> {
            fs::create_dir(root)
                .with_context(|| format!("create fresh diagnostic root {}", root.display()))?;
            let metadata = fs::symlink_metadata(root)?;
            ensure!(
                metadata.file_type().is_dir() && !metadata.file_type().is_symlink(),
                "diagnostic root is not a real directory"
            );
            Ok(Self {
                root: root.to_path_buf(),
                events: Mutex::new(BTreeMap::new()),
            })
        }

        fn event_inventory(&self) -> Result<BTreeMap<String, EventIdentity>> {
            Ok(self
                .events
                .lock()
                .map_err(|_| anyhow::anyhow!("diagnostic event inventory mutex was poisoned"))?
                .clone())
        }
    }

    impl FinalizationDiagnosticSink for FinalizationDiagnosticDirectory {
        fn write_event(&self, name: &str, json: &[u8]) -> std::io::Result<()> {
            if name.is_empty()
                || !name
                    .bytes()
                    .all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'-')
            {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    "event name must use lowercase ASCII, digits, and hyphens",
                ));
            }
            if json.len() > MAX_EVENT_BYTES {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("diagnostic event exceeds {MAX_EVENT_BYTES} bytes"),
                ));
            }
            if serde_json::from_slice::<serde_json::Value>(json).is_err() {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "diagnostic event is not valid JSON",
                ));
            }
            let mut events = self.events.lock().map_err(|_| {
                std::io::Error::other("diagnostic event inventory mutex was poisoned")
            })?;
            if events.contains_key(name) {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::AlreadyExists,
                    format!("duplicate diagnostic event '{name}'"),
                ));
            }

            let final_path = self.root.join(format!("{name}.json"));
            let temp_path = self
                .root
                .join(format!(".{name}.{}.pending", std::process::id()));
            let mut file = OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&temp_path)?;
            let publish_result = (|| {
                file.write_all(json)?;
                file.sync_all()?;
                fs::hard_link(&temp_path, &final_path)?;
                fs::remove_file(&temp_path)?;
                sync_directory(&self.root)?;
                Ok::<(), std::io::Error>(())
            })();
            if publish_result.is_err() {
                let _ = fs::remove_file(&temp_path);
            }
            publish_result?;
            events.insert(
                name.to_string(),
                EventIdentity {
                    bytes: json.len(),
                    sha256: hex_sha256(json),
                },
            );
            Ok(())
        }
    }

    #[cfg(unix)]
    fn sync_directory(path: &Path) -> std::io::Result<()> {
        File::open(path)?.sync_all()
    }

    #[cfg(not(unix))]
    fn sync_directory(_path: &Path) -> std::io::Result<()> {
        // Directory syncing is not portable through std on every target.
        Ok(())
    }

    fn hex_sha256(bytes: &[u8]) -> String {
        Sha256::digest(bytes)
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect()
    }

    fn hash_file(path: &Path) -> Result<(u64, String)> {
        let metadata = fs::symlink_metadata(path)?;
        ensure!(
            metadata.file_type().is_file(),
            "input is not a regular file: {}",
            path.display()
        );
        ensure!(
            metadata.len() <= MAX_INPUT_FILE_BYTES,
            "input exceeds the per-file evidence cap: {}",
            path.display()
        );
        let mut file = File::open(path)?;
        let mut digest = Sha256::new();
        let mut bytes = 0_u64;
        let mut buffer = [0_u8; 64 * 1024];
        loop {
            let read = file.read(&mut buffer)?;
            if read == 0 {
                break;
            }
            bytes = bytes
                .checked_add(read as u64)
                .context("input byte count overflow")?;
            ensure!(
                bytes <= MAX_INPUT_FILE_BYTES && bytes <= metadata.len(),
                "input changed or exceeded its recorded size while hashing: {}",
                path.display()
            );
            digest.update(&buffer[..read]);
        }
        ensure!(
            bytes == metadata.len(),
            "input changed while hashing: {}",
            path.display()
        );
        let sha256 = digest
            .finalize()
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect();
        Ok((bytes, sha256))
    }

    fn collect_tree(root: &Path) -> Result<Vec<FileIdentity>> {
        fn visit(
            root: &Path,
            path: &Path,
            files: &mut Vec<FileIdentity>,
            total_bytes: &mut u64,
        ) -> Result<()> {
            let mut entries = fs::read_dir(path)?
                .collect::<std::io::Result<Vec<_>>>()
                .with_context(|| format!("read input directory {}", path.display()))?;
            entries.sort_by_key(|entry| entry.file_name());
            for entry in entries {
                let entry_path = entry.path();
                let file_type = fs::symlink_metadata(&entry_path)?.file_type();
                if file_type.is_symlink() {
                    bail!(
                        "refusing symlink in captured input tree: {}",
                        entry_path.display()
                    );
                }
                if file_type.is_dir() {
                    visit(root, &entry_path, files, total_bytes)?;
                    continue;
                }
                ensure!(
                    file_type.is_file(),
                    "unsupported input entry: {}",
                    entry_path.display()
                );
                ensure!(
                    files.len() < MAX_INPUT_FILES,
                    "input tree exceeds file-count cap"
                );
                let (bytes, sha256) = hash_file(&entry_path)?;
                *total_bytes = total_bytes
                    .checked_add(bytes)
                    .context("input tree byte count overflow")?;
                ensure!(
                    *total_bytes <= MAX_INPUT_TREE_BYTES,
                    "input tree exceeds byte cap"
                );
                files.push(FileIdentity {
                    path: entry_path
                        .strip_prefix(root)
                        .expect("walked path must remain under root")
                        .to_string_lossy()
                        .into_owned(),
                    bytes,
                    sha256,
                });
            }
            Ok(())
        }

        let mut files = Vec::new();
        let mut total_bytes = 0_u64;
        visit(root, root, &mut files, &mut total_bytes)?;
        Ok(files)
    }

    fn collect_source_files(root: &Path) -> Result<Vec<FileIdentity>> {
        let paths = [
            "Cargo.toml",
            "Cargo.lock",
            ".cargo/config.toml",
            "crates/roomeq-qa/Cargo.toml",
            "crates/roomeq-qa/src/quality/run.rs",
            "crates/roomeq-qa/src/quality/tests.rs",
            "crates/roomeq-quality/Cargo.toml",
            "crates/roomeq-quality/src/quality.rs",
            "crates/roomeq-engine/Cargo.toml",
            "crates/roomeq-engine/src/room_result.rs",
            "crates/roomeq-engine/src/output/create.rs",
            "crates/roomeq-workflow/Cargo.toml",
            "crates/roomeq-workflow/src/lib.rs",
            "crates/roomeq-workflow/src/pipeline.rs",
            "crates/roomeq-workflow/src/room_optimization.rs",
            "crates/roomeq-workflow/src/room_optimization/finalization.rs",
            "crates/roomeq-workflow/src/room_optimization/seat_replay.rs",
        ];
        paths
            .into_iter()
            .map(|relative| {
                let path = root.join(relative);
                let (bytes, sha256) = hash_file(&path)
                    .with_context(|| format!("inventory source file {}", path.display()))?;
                Ok(FileIdentity {
                    path: relative.to_string(),
                    bytes,
                    sha256,
                })
            })
            .collect()
    }

    fn git_output(root: &Path, arguments: &[&str]) -> Result<String> {
        let output = Command::new("git")
            .arg("-C")
            .arg(root)
            .args(arguments)
            .output()
            .with_context(|| format!("run git {}", arguments.join(" ")))?;
        ensure!(
            output.status.success(),
            "git {} failed: {}",
            arguments.join(" "),
            String::from_utf8_lossy(&output.stderr)
        );
        Ok(String::from_utf8(output.stdout)?.trim().to_string())
    }

    fn resolved_cargo_metadata(root: &Path) -> Result<Vec<u8>> {
        let output = Command::new("cargo")
            .args(["metadata", "--format-version", "1", "--locked", "--offline"])
            .current_dir(root)
            .output()
            .context("run locked offline cargo metadata for diagnostic provenance")?;
        ensure!(
            output.status.success(),
            "cargo metadata failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        ensure!(
            output.stdout.len() <= MAX_EVENT_BYTES,
            "cargo metadata exceeds diagnostic event cap"
        );
        Ok(output.stdout)
    }

    fn assert_resolved_run_config(
        root: &Path,
        event: &serde_json::Value,
        project_root: &Path,
    ) -> Result<()> {
        let config = event
            .get("config")
            .context("optimized diagnostic is missing its resolved run config")?;
        ensure!(
            config
                .get("serialized")
                .and_then(serde_json::Value::as_bool)
                == Some(true)
                && config
                    .get("value")
                    .is_some_and(serde_json::Value::is_object),
            "optimized diagnostic did not retain a serialized resolved run config"
        );
        let config_path = root.join("resolved-run-config.json");
        let config_metadata = fs::metadata(&config_path)
            .with_context(|| format!("missing resolved run config at {}", config_path.display()))?;
        ensure!(
            config_metadata.len() <= MAX_EVENT_BYTES as u64,
            "resolved run config exceeds the read cap"
        );
        let config_bytes = fs::read(&config_path)?;
        let saved_config: serde_json::Value = serde_json::from_slice(&config_bytes)
            .context("parse the retained resolved run config")?;
        let run_inputs_path = root.join("run-inputs.json");
        let run_inputs_metadata = fs::metadata(&run_inputs_path).with_context(|| {
            format!(
                "missing run-inputs receipt at {}",
                run_inputs_path.display()
            )
        })?;
        ensure!(
            run_inputs_metadata.len() <= MAX_EVENT_BYTES as u64,
            "run-inputs receipt exceeds the read cap"
        );
        let run_inputs: serde_json::Value = serde_json::from_slice(&fs::read(&run_inputs_path)?)
            .context("parse run-inputs receipt for resolved config")?;

        let mut finalizer_config = config.get("value").expect("validated above").clone();
        normalize_and_verify_loaded_measurements(&mut finalizer_config, project_root, &run_inputs)?;
        ensure!(
            saved_config == finalizer_config,
            "retained resolved run config differs from the finalizer config after checking loaded measurement snapshots"
        );

        ensure!(
            run_inputs
                .get("resolved_config_bytes")
                .and_then(serde_json::Value::as_u64)
                == Some(config_bytes.len() as u64),
            "resolved run config length differs from its run-inputs receipt"
        );
        let config_sha256 = hex_sha256(&config_bytes);
        ensure!(
            run_inputs
                .get("resolved_config_sha256")
                .and_then(serde_json::Value::as_str)
                == Some(config_sha256.as_str()),
            "resolved run config hash differs from its run-inputs receipt"
        );
        ensure!(
            run_inputs
                .get("resolved_config_event")
                .and_then(serde_json::Value::as_str)
                == Some("resolved-run-config.json"),
            "run-inputs receipt does not identify the retained resolved config"
        );

        // Workflow metadata may omit a duplicate embedded copy. If present,
        // that copy must still be serialized completely.
        if let Some(effective_config) = event
            .pointer("/result/effective_config")
            .filter(|value| !value.is_null())
        {
            ensure!(
                effective_config
                    .get("serialized")
                    .and_then(serde_json::Value::as_bool)
                    == Some(true)
                    && effective_config
                        .get("value")
                        .is_some_and(serde_json::Value::is_object),
                "optimized diagnostic effective config was not serialized"
            );
        }
        Ok(())
    }

    fn normalize_and_verify_loaded_measurements(
        finalizer_config: &mut serde_json::Value,
        project_root: &Path,
        run_inputs: &serde_json::Value,
    ) -> Result<()> {
        visit_loaded_measurements(finalizer_config, project_root, run_inputs)
    }

    fn visit_loaded_measurements(
        value: &mut serde_json::Value,
        project_root: &Path,
        run_inputs: &serde_json::Value,
    ) -> Result<()> {
        match value {
            serde_json::Value::Array(values) => {
                for value in values {
                    visit_loaded_measurements(value, project_root, run_inputs)?;
                }
            }
            serde_json::Value::Object(object)
                if object.contains_key("original") || object.contains_key("loaded_response") =>
            {
                ensure!(
                    object.len() == 2
                        && object.contains_key("original")
                        && object.contains_key("loaded_response"),
                    "loaded measurement snapshot has an unexpected shape"
                );
                let mut original = object
                    .get("original")
                    .expect("validated loaded measurement shape")
                    .clone();
                let loaded_response = object
                    .get("loaded_response")
                    .expect("validated loaded measurement shape");
                verify_loaded_measurement_source(
                    &original,
                    loaded_response,
                    project_root,
                    run_inputs,
                )?;
                visit_loaded_measurements(&mut original, project_root, run_inputs)?;
                *value = original;
            }
            serde_json::Value::Object(object) => {
                for value in object.values_mut() {
                    visit_loaded_measurements(value, project_root, run_inputs)?;
                }
            }
            _ => {}
        }
        Ok(())
    }

    fn verify_loaded_measurement_source(
        original: &serde_json::Value,
        loaded_response: &serde_json::Value,
        project_root: &Path,
        run_inputs: &serde_json::Value,
    ) -> Result<()> {
        let measurement_path = measurement_source_path(original)
            .context("loaded measurement original reference has no path")?;
        let config_source = run_inputs
            .get("config_source_paths")
            .and_then(serde_json::Value::as_array)
            .and_then(|paths| paths.first())
            .and_then(serde_json::Value::as_str)
            .context("run-inputs receipt omitted the primary config source path")?;
        let config_source_path = Path::new(config_source);
        ensure!(
            !config_source_path.is_absolute()
                && config_source_path
                    .components()
                    .all(|component| matches!(component, std::path::Component::Normal(_))),
            "run-inputs config source path is not workspace-relative"
        );
        let canonical_project = project_root
            .canonicalize()
            .context("canonicalize diagnostic project root")?;
        let config_path = canonical_project.join(config_source_path);
        let input_root = config_path
            .parent()
            .context("primary config source has no parent directory")?
            .canonicalize()
            .context("resolve primary config input directory")?;
        ensure!(
            input_root.starts_with(&canonical_project),
            "primary config input directory escaped the project root"
        );
        let source_path = Path::new(measurement_path);
        let source_path = if source_path.is_absolute() {
            source_path.to_path_buf()
        } else {
            input_root.join(source_path)
        };
        let source_path = source_path
            .canonicalize()
            .context("resolve loaded measurement source")?;
        ensure!(
            source_path.starts_with(&input_root),
            "loaded measurement source escaped the captured input directory"
        );
        let relative_path = source_path
            .strip_prefix(&input_root)
            .expect("source path checked under input root")
            .to_string_lossy()
            .replace('\\', "/");
        let inventory = run_inputs
            .get("input_fixture_files")
            .and_then(serde_json::Value::as_array)
            .context("run-inputs receipt omitted captured input inventory")?;
        let entry = inventory
            .iter()
            .find(|entry| {
                entry.get("path").and_then(serde_json::Value::as_str) == Some(&relative_path)
            })
            .with_context(|| {
                format!("loaded measurement is absent from input inventory: {relative_path}")
            })?;
        let (source_bytes, source_sha256) = hash_file(&source_path)
            .with_context(|| format!("hash loaded measurement source {relative_path}"))?;
        ensure!(
            entry.get("bytes").and_then(serde_json::Value::as_u64) == Some(source_bytes)
                && entry.get("sha256").and_then(serde_json::Value::as_str)
                    == Some(source_sha256.as_str()),
            "loaded measurement source differs from captured inventory: {relative_path}"
        );
        let parsed_curve =
            autoeq_measurements::read::read_curve_from_csv(&source_path.to_path_buf())
                .map_err(|error| anyhow::anyhow!(error.to_string()))
                .with_context(|| format!("parse loaded measurement source {relative_path}"))?;
        let parsed_curve = serde_json::to_value(parsed_curve)?;
        ensure!(
            &parsed_curve == loaded_response,
            "loaded measurement snapshot differs from its inventoried source: {relative_path}"
        );
        Ok(())
    }

    fn measurement_source_path(reference: &serde_json::Value) -> Option<&str> {
        reference
            .get("original")
            .and_then(measurement_source_path)
            .or_else(|| reference.get("path").and_then(serde_json::Value::as_str))
            .or_else(|| reference.as_str())
    }

    fn assert_iir_only_events(
        root: &Path,
        project_root: &Path,
        require_complete: bool,
    ) -> Result<()> {
        let mut post_alignment_identity = None;
        let mut replay_identity = None;
        let mut final_identity = None;
        for event_name in [
            "optimized-pre-finalization",
            "zero-strength-output-prepared",
            "zero-strength-output-required-attenuation",
            "zero-strength-output-post-safety-pre-alignment",
            "zero-strength-output-post-alignment",
            "zero-strength-output-useful-output-replay",
            "finalization-result",
        ] {
            let path = root.join(format!("{event_name}.json"));
            if !path.exists() {
                ensure!(
                    !require_complete,
                    "successful diagnostic omitted required event {event_name}"
                );
                continue;
            }
            let metadata = fs::metadata(&path)?;
            ensure!(
                metadata.len() <= MAX_EVENT_BYTES as u64,
                "diagnostic event exceeds the read cap: {}",
                path.display()
            );
            let bytes = fs::read(&path)?;
            let event: serde_json::Value = serde_json::from_slice(&bytes)
                .with_context(|| format!("parse graph diagnostic {}", path.display()))?;
            if event_name == "optimized-pre-finalization" {
                assert_resolved_run_config(root, &event, project_root)?;
            }
            if event_name == "zero-strength-output-required-attenuation" {
                continue;
            }
            if event_name == "zero-strength-output-useful-output-replay" {
                let records = event
                    .get("replay_records")
                    .and_then(serde_json::Value::as_array)
                    .with_context(|| "useful-output event omitted replay records")?;
                ensure!(
                    !records.is_empty(),
                    "useful-output event retained no replay records"
                );
                let serialized_hash = event
                    .get("replayed_serialized_dsp_graph_projection_sha256")
                    .and_then(serde_json::Value::as_str)
                    .context("useful-output event omitted serialized graph hash")?;
                let playback_hash = event
                    .get("replayed_playback_graph_sha256")
                    .and_then(serde_json::Value::as_str)
                    .context("useful-output event omitted playback graph hash")?;
                let descriptor = event
                    .get("trial_descriptor")
                    .context("useful-output event omitted trial descriptor")?
                    .clone();
                for record in records {
                    ensure!(
                        record
                            .get("replayed_serialized_dsp_graph_projection_sha256")
                            .and_then(serde_json::Value::as_str)
                            == Some(serialized_hash)
                            && record
                                .get("replayed_playback_graph_sha256")
                                .and_then(serde_json::Value::as_str)
                                == Some(playback_hash),
                        "useful-output replay graph identity differs from its event"
                    );
                }
                replay_identity = Some((
                    descriptor,
                    serialized_hash.to_string(),
                    playback_hash.to_string(),
                ));
                continue;
            }
            let channels = event
                .pointer("/result/dsp_graph/channels")
                .and_then(serde_json::Value::as_object)
                .with_context(|| format!("missing serialized channel graph in {event_name}"))?;
            for (channel, chain) in channels {
                let plugins = chain
                    .get("plugins")
                    .and_then(serde_json::Value::as_array)
                    .with_context(|| format!("missing plugin list for channel {channel}"))?;
                ensure!(
                    plugins.iter().all(|plugin| plugin
                        .get("plugin_type")
                        .and_then(serde_json::Value::as_str)
                        != Some("convolution")),
                    "IIR diagnostic event {event_name} contains convolution on {channel}"
                );
            }
            if event_name == "zero-strength-output-post-alignment" {
                let descriptor = event
                    .get("trial_descriptor")
                    .context("post-alignment event omitted trial descriptor")?
                    .clone();
                let serialized_hash = event
                    .get("serialized_dsp_graph_projection_sha256")
                    .and_then(serde_json::Value::as_str)
                    .context("post-alignment event omitted serialized graph hash")?;
                let playback_hash = event
                    .get("playback_graph_sha256")
                    .and_then(serde_json::Value::as_str)
                    .context("post-alignment event omitted playback graph hash")?;
                post_alignment_identity = Some((
                    descriptor,
                    serialized_hash.to_string(),
                    playback_hash.to_string(),
                ));
            } else if event_name == "finalization-result" {
                let descriptor = event
                    .get("attempted_target_trial_descriptor")
                    .context("final result omitted attempted target descriptor")?
                    .clone();
                let target_serialized_hash = event
                    .get("attempted_target_post_alignment_serialized_dsp_graph_projection_sha256")
                    .and_then(serde_json::Value::as_str)
                    .context("final result omitted target serialized graph hash")?;
                let target_playback_hash = event
                    .get("attempted_target_post_alignment_playback_graph_sha256")
                    .and_then(serde_json::Value::as_str)
                    .context("final result omitted target playback graph hash")?;
                let final_playback_hash = event
                    .get("playback_graph_sha256")
                    .and_then(serde_json::Value::as_str)
                    .context("final result omitted final playback graph hash")?;
                let equals_final = event
                    .get("attempted_target_graph_equals_final_playback_graph")
                    .and_then(serde_json::Value::as_bool)
                    .context("final result omitted playback graph equality")?;
                ensure!(
                    equals_final == (target_playback_hash == final_playback_hash),
                    "final result playback graph equality does not match its hashes"
                );
                final_identity = Some((
                    descriptor,
                    target_serialized_hash.to_string(),
                    target_playback_hash.to_string(),
                ));
            }
        }
        match (post_alignment_identity.as_ref(), replay_identity.as_ref()) {
            (Some(post), Some(replay)) => ensure!(
                post == replay,
                "useful-output replay hashes or trial descriptor differ from post-alignment event"
            ),
            _ if require_complete => {
                bail!("successful diagnostic is missing cross-event playback identity")
            }
            _ => {}
        }
        match (post_alignment_identity.as_ref(), final_identity.as_ref()) {
            (Some(post), Some(final_result)) => ensure!(
                post == final_result,
                "final result target hashes or trial descriptor differ from post-alignment event"
            ),
            _ if require_complete => {
                bail!("successful diagnostic is missing final target graph identity")
            }
            _ => {}
        }
        Ok(())
    }

    pub(in crate::quality) fn run_one_iir(
        project_root: &Path,
        diagnostic_root: &Path,
        maxeval: usize,
    ) -> Result<()> {
        let input_root = project_root.join("data_tests/roomeq/measured/5.1.4_genelec");
        let base_config_path = input_root.join("recordings.json");
        let override_path = input_root.join("optimiser-iir.json");
        let output = FinalizationDiagnosticDirectory::create(diagnostic_root)?;
        let source_before = collect_source_files(project_root)?;
        let inputs_before = collect_tree(&input_root)?;
        let metadata_bytes = resolved_cargo_metadata(project_root)?;
        let metadata_value: serde_json::Value =
            serde_json::from_slice(&metadata_bytes).context("parse resolved cargo metadata")?;
        output.write_event("cargo-metadata", &metadata_bytes)?;
        let executable = std::env::current_exe().context("resolve QA test executable")?;
        let (executable_bytes, executable_sha256) = hash_file(&executable)?;
        let build_identity = serde_json::json!({
            "schema_version": 1,
            "event": "build-identity",
            "test_executable_path": executable,
            "test_executable_bytes": executable_bytes,
            "test_executable_sha256": executable_sha256,
            "cargo_metadata_bytes": metadata_bytes.len(),
            "cargo_metadata_sha256": hex_sha256(&metadata_bytes),
            "cargo_metadata_package_count": metadata_value["packages"].as_array().map(Vec::len),
            "cargo_metadata_workspace_members": metadata_value["workspace_members"],
        });
        output.write_event(
            "build-identity",
            &serde_json::to_vec_pretty(&build_identity)?,
        )?;
        let (mut config, _) = super::super::misc::load_config_for_path(
            &base_config_path,
            Some(&override_path),
            ProcessingMode::LowLatency,
            true,
        )?;
        clamp_strict_measured_maxeval(&mut config, maxeval);
        let resolved_config = serde_json::to_vec(&serde_json::to_value(&config)?)?;
        output.write_event("resolved-run-config", &resolved_config)?;
        let run_inputs = serde_json::json!({
            "schema_version": 1,
            "event": "run-inputs",
            "worktree_head": git_output(project_root, &["rev-parse", "HEAD"] )?,
            "worktree_status": git_output(project_root, &["status", "--porcelain=v1", "--untracked-files=all"] )?,
            "sample_rate_hz": SAMPLE_RATE,
            "processing_mode": "low_latency_iir",
            "expected_no_convolution": true,
            "maxeval_requested": maxeval,
            "maxeval_effective": config.optimizer.max_iter,
            "seed": config.optimizer.seed,
            "resolved_config_bytes": resolved_config.len(),
            "resolved_config_sha256": hex_sha256(&resolved_config),
            "resolved_config_event": "resolved-run-config.json",
            "config_source_paths": [
                base_config_path.strip_prefix(project_root)?.to_string_lossy(),
                override_path.strip_prefix(project_root)?.to_string_lossy(),
            ],
            "source_files": &source_before,
            "input_fixture_files": &inputs_before,
        });
        output.write_event("run-inputs", &serde_json::to_vec_pretty(&run_inputs)?)?;

        let run_directory = tempfile::tempdir().context("create temporary RoomEQ run directory")?;
        let result = RoomPipeline::new(RoomPipelineRequest {
            config: &config,
            sample_rate: SAMPLE_RATE,
            output_dir: Some(run_directory.path()),
            probe_arrival_overrides: None,
        })
        .with_finalization_diagnostic_sink(FinalizationDiagnosticTrial::ZeroStrengthOutput, &output)
        .run(None)
        .map_err(|error| anyhow::anyhow!(error.to_string()));
        let no_convolution_events =
            assert_iir_only_events(&output.root, project_root, result.is_ok());

        let source_after = collect_source_files(project_root)?;
        let inputs_after = collect_tree(&input_root)?;
        let unchanged = source_before == source_after && inputs_before == inputs_after;
        let mode_contract = result.as_ref().ok().map(|result| {
            let convolution_plugin_count = result
                .channels
                .values()
                .flat_map(|chain| chain.plugins.iter())
                .filter(|plugin| plugin.plugin_type == "convolution")
                .count();
            let retained_fir_channels = result
                .channel_results
                .values()
                .filter(|channel| channel.fir_coeffs.is_some())
                .count();
            serde_json::json!({
                "convolution_plugin_count": convolution_plugin_count,
                "retained_fir_channels": retained_fir_channels,
                "no_convolution": convolution_plugin_count == 0 && retained_fir_channels == 0,
            })
        });
        let summary = match &result {
            Ok(result) => serde_json::json!({
                "schema_version": 1,
                "event": "qa-run-summary",
                "status": "completed",
                "derived_outcome": result.metadata.correction_acceptance.as_ref().map(|report| report.derived_outcome()),
                "stage_outcomes": &result.metadata.stage_outcomes,
                "mode_contract": &mode_contract,
                "no_convolution_event_check": no_convolution_events.as_ref().map(|_| true).unwrap_or(false),
                "no_convolution_event_error": no_convolution_events.as_ref().err().map(ToString::to_string),
                "source_and_fixture_unchanged": unchanged,
            }),
            Err(error) => serde_json::json!({
                "schema_version": 1,
                "event": "qa-run-summary",
                "status": "failed",
                "error": format!("{error:#}"),
                "source_and_fixture_unchanged": unchanged,
            }),
        };
        output.write_event("qa-run-summary", &serde_json::to_vec_pretty(&summary)?)?;
        let manifest = serde_json::json!({
            "schema_version": 1,
            "event": "run-manifest",
            "claim_scope": "single IIR workflow finalization diagnostic; retained trace is not acoustic acceptance",
            "source_and_fixture_unchanged": unchanged,
            "source_files_after": source_after,
            "input_fixture_files_after": inputs_after,
            "events": output.event_inventory()?,
        });
        output.write_event("run-manifest", &serde_json::to_vec_pretty(&manifest)?)?;
        ensure!(
            unchanged,
            "source or fixture bytes changed during diagnostic run"
        );
        no_convolution_events?;
        match result {
            Ok(_) => {
                ensure!(
                    mode_contract
                        .as_ref()
                        .and_then(|contract| contract["no_convolution"].as_bool())
                        == Some(true),
                    "IIR diagnostic unexpectedly emitted convolution content"
                );
                Ok(())
            }
            Err(error) => Err(error),
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn directory_sink_publishes_atomically_and_refuses_duplicate_or_unsafe_names() {
            let parent = tempfile::tempdir().unwrap();
            let root = parent.path().join("capture");
            let sink = FinalizationDiagnosticDirectory::create(&root).unwrap();
            sink.write_event("first-event", br#"{"ok":true}"#).unwrap();
            let before = fs::read(root.join("first-event.json")).unwrap();
            assert!(sink.write_event("first-event", b"{}").is_err());
            assert!(sink.write_event("../escape", b"{}").is_err());
            assert!(sink.write_event("invalid-json", b"not-json").is_err());
            assert_eq!(fs::read(root.join("first-event.json")).unwrap(), before);
            assert!(!root.join("../escape.json").exists());
            assert!(FinalizationDiagnosticDirectory::create(&root).is_err());
        }

        #[test]
        fn input_inventory_refuses_symlinks_and_detects_byte_changes() {
            let parent = tempfile::tempdir().unwrap();
            let root = parent.path().join("inputs");
            fs::create_dir(&root).unwrap();
            let input = root.join("measurement.csv");
            fs::write(&input, b"freq,spl\n20,80\n").unwrap();
            let first = collect_tree(&root).unwrap();
            fs::write(&input, b"freq,spl\n20,81\n").unwrap();
            let second = collect_tree(&root).unwrap();
            assert_ne!(first, second);
            #[cfg(unix)]
            {
                std::os::unix::fs::symlink(&input, root.join("measurement-link.csv")).unwrap();
                assert!(collect_tree(&root).is_err());
            }
        }

        #[test]
        fn measured_diagnostic_requires_the_exact_serialized_run_config() {
            let parent = tempfile::tempdir().unwrap();
            let root = parent.path();
            let config = serde_json::json!({
                "optimizer": { "max_iter": 600_000 },
                "processing_mode": "low_latency_iir"
            });
            let config_bytes = serde_json::to_vec(&config).unwrap();
            fs::write(root.join("resolved-run-config.json"), &config_bytes).unwrap();
            fs::write(
                root.join("run-inputs.json"),
                serde_json::to_vec(&serde_json::json!({
                    "resolved_config_bytes": config_bytes.len(),
                    "resolved_config_sha256": hex_sha256(&config_bytes),
                    "resolved_config_event": "resolved-run-config.json"
                }))
                .unwrap(),
            )
            .unwrap();
            let valid = serde_json::json!({
                "config": { "serialized": true, "value": config.clone() },
                "result": { "effective_config": null }
            });
            assert_resolved_run_config(root, &valid, root).unwrap();

            let missing = serde_json::json!({"result": {"effective_config": null}});
            assert!(assert_resolved_run_config(root, &missing, root).is_err());
            let unavailable = serde_json::json!({
                "config": {"serialized": false, "reason": "in-memory"},
                "result": { "effective_config": null }
            });
            assert!(assert_resolved_run_config(root, &unavailable, root).is_err());
            let malformed_effective = serde_json::json!({
                "config": { "serialized": true, "value": config },
                "result": {
                    "effective_config": {"serialized": false, "reason": "unavailable"}
                }
            });
            assert!(assert_resolved_run_config(root, &malformed_effective, root).is_err());

            let mut changed_bytes = config_bytes.clone();
            changed_bytes.push(b' ');
            fs::write(root.join("resolved-run-config.json"), changed_bytes).unwrap();
            assert!(assert_resolved_run_config(root, &valid, root).is_err());
        }

        #[test]
        fn loaded_measurement_snapshots_normalize_only_after_source_verification() {
            let project = tempfile::tempdir().unwrap();
            let input_root = project
                .path()
                .join("data_tests/roomeq/measured/finalization-test");
            fs::create_dir_all(&input_root).unwrap();
            let config_source = input_root.join("recordings.json");
            fs::write(&config_source, b"{}\n").unwrap();
            let measurement_path = input_root.join("measurement.csv");
            let measurement_bytes =
                b"frequency,spl,phase\n20,80,0\n40,81,-10\n80,82,-20\n160,83,-30\n";
            fs::write(&measurement_path, measurement_bytes).unwrap();
            let loaded_curve =
                autoeq_measurements::read::read_curve_from_csv(&measurement_path).unwrap();
            let original = serde_json::json!({
                "name": "test speaker",
                "path": measurement_path.to_string_lossy(),
            });
            let saved_config = serde_json::json!({
                "speakers": { "left": original.clone() },
                "optimizer": { "max_iter": 1 },
            });
            let runtime_config = serde_json::json!({
                "speakers": {
                    "left": {
                        "original": original,
                        "loaded_response": serde_json::to_value(&loaded_curve).unwrap(),
                    }
                },
                "optimizer": { "max_iter": 1 },
            });
            let config_bytes = serde_json::to_vec(&saved_config).unwrap();
            let root = project.path();
            fs::write(root.join("resolved-run-config.json"), &config_bytes).unwrap();
            fs::write(
                root.join("run-inputs.json"),
                serde_json::to_vec(&serde_json::json!({
                    "config_source_paths": ["data_tests/roomeq/measured/finalization-test/recordings.json"],
                    "input_fixture_files": [{
                        "path": "measurement.csv",
                        "bytes": measurement_bytes.len(),
                        "sha256": hex_sha256(measurement_bytes),
                    }],
                    "resolved_config_bytes": config_bytes.len(),
                    "resolved_config_sha256": hex_sha256(&config_bytes),
                    "resolved_config_event": "resolved-run-config.json",
                }))
                .unwrap(),
            )
            .unwrap();
            let event = serde_json::json!({
                "config": { "serialized": true, "value": runtime_config },
                "result": { "effective_config": null },
            });

            assert_resolved_run_config(root, &event, root).unwrap();

            let mut changed_curve = event.clone();
            changed_curve["config"]["value"]["speakers"]["left"]["loaded_response"]["spl"]["data"]
                [1] = serde_json::json!(99.0);
            assert!(
                assert_resolved_run_config(root, &changed_curve, root)
                    .unwrap_err()
                    .to_string()
                    .contains("snapshot differs")
            );

            let mut changed_reference = event.clone();
            changed_reference["config"]["value"]["speakers"]["left"]["original"]["name"] =
                serde_json::json!("different speaker");
            assert!(assert_resolved_run_config(root, &changed_reference, root).is_err());

            fs::write(
                &measurement_path,
                b"frequency,spl,phase\n20,80,0\n40,90,-10\n80,82,-20\n160,83,-30\n",
            )
            .unwrap();
            assert!(
                assert_resolved_run_config(root, &event, root)
                    .unwrap_err()
                    .to_string()
                    .contains("differs from captured inventory")
            );
        }

        #[test]
        #[ignore = "artifact-only replay; set A09_RUN02_ARTIFACT_ROOT to the preserved run02 output"]
        fn retained_run02_artifacts_match_loaded_sources_and_cross_event_identity() -> Result<()> {
            let artifact_root = PathBuf::from(
                std::env::var_os("A09_RUN02_ARTIFACT_ROOT")
                    .context("set A09_RUN02_ARTIFACT_ROOT to the preserved run02 output")?,
            );
            let project_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("../..")
                .canonicalize()?;
            assert_iir_only_events(&artifact_root, &project_root, false)?;

            let summary: serde_json::Value =
                serde_json::from_slice(&fs::read(artifact_root.join("qa-run-summary.json"))?)?;
            ensure!(
                summary.get("status").and_then(serde_json::Value::as_str) == Some("completed"),
                "preserved run02 workflow status changed"
            );
            ensure!(
                summary
                    .get("no_convolution_event_error")
                    .and_then(serde_json::Value::as_str)
                    .is_some_and(|error| error.contains("retained resolved run config differs")),
                "preserved run02 config-mismatch failure was not retained"
            );
            ensure!(
                summary
                    .get("no_convolution_event_check")
                    .and_then(serde_json::Value::as_bool)
                    == Some(false),
                "preserved run02 diagnostic-check failure was not retained"
            );
            let replay: serde_json::Value = serde_json::from_slice(&fs::read(
                artifact_root.join("zero-strength-output-useful-output-replay.json"),
            )?)?;
            ensure!(
                replay
                    .get("replay_records")
                    .and_then(serde_json::Value::as_array)
                    .is_some_and(|records| records.len() == 2),
                "preserved run02 replay record count changed"
            );
            Ok(())
        }

        #[test]
        fn successful_diagnostic_requires_all_trace_events_but_failure_keeps_partial_trace() {
            let parent = tempfile::tempdir().unwrap();
            assert!(assert_iir_only_events(parent.path(), parent.path(), true).is_err());
            assert!(assert_iir_only_events(parent.path(), parent.path(), false).is_ok());
            let config = serde_json::json!({"optimizer": {"max_iter": 1}});
            let config_bytes = serde_json::to_vec(&config).unwrap();
            fs::write(
                parent.path().join("resolved-run-config.json"),
                &config_bytes,
            )
            .unwrap();
            fs::write(
                parent.path().join("run-inputs.json"),
                serde_json::to_vec(&serde_json::json!({
                    "resolved_config_bytes": config_bytes.len(),
                    "resolved_config_sha256": hex_sha256(&config_bytes),
                    "resolved_config_event": "resolved-run-config.json"
                }))
                .unwrap(),
            )
            .unwrap();
            fs::write(
                parent.path().join("optimized-pre-finalization.json"),
                serde_json::to_vec(&serde_json::json!({
                    "config": {"serialized": true, "value": config},
                    "result": {"dsp_graph": {"channels": {}}}
                }))
                .unwrap(),
            )
            .unwrap();
            assert!(
                assert_iir_only_events(parent.path(), parent.path(), true)
                    .unwrap_err()
                    .to_string()
                    .contains("zero-strength-output-prepared")
            );
            assert!(assert_iir_only_events(parent.path(), parent.path(), false).is_ok());
        }
    }
}

pub(super) fn run_optimization(
    config: &RoomConfig,
    seed_runs: usize,
) -> Result<AssessedOptimization> {
    let id = TEMP_DIR_COUNTER.fetch_add(1, Ordering::Relaxed);
    let temp_dir = std::env::temp_dir().join(format!("roomeq_qa_{}_{}", std::process::id(), id));
    std::fs::create_dir_all(&temp_dir)?;
    let result = if seed_runs == 1 {
        roomeq_workflow::optimize_room(config, SAMPLE_RATE, None, Some(&temp_dir))
            .map_err(|error| anyhow::anyhow!(error.to_string()))
    } else {
        crate::optimize_room(config, SAMPLE_RATE, Some(&temp_dir))
    };
    let result = result.and_then(|result| {
        let electrical = super::electrical::assess(&result, SAMPLE_RATE, &temp_dir);
        let mut scorecard = super::metric_scorecard::compute_result_scorecard(&result, electrical);
        let bundle_root = super::misc::find_project_root()?.join("target/qa/electrical-replays");
        std::fs::create_dir_all(&bundle_root)?;
        let bundle = bundle_root.join(format!(
            "{}-{id}-{}",
            std::process::id(),
            chrono::Utc::now().timestamp_micros()
        ));
        match super::electrical::retain_replay_bundle(&result, SAMPLE_RATE, &temp_dir, &bundle) {
            Ok(()) => scorecard.replay_bundle = Some(bundle),
            Err(error) => {
                scorecard.max_boost_db = f64::INFINITY;
                scorecard.electrical = Some(Err(format!(
                    "failed to retain final replay bundle: {error:#}"
                )));
            }
        }
        Ok(AssessedOptimization { result, scorecard })
    });
    let _ = std::fs::remove_dir_all(&temp_dir);
    result
}

pub(super) fn run_stereo_workflow_tests(
    name: &str,
    base_config_path: &Path,
    override_config_path: Option<&Path>,
    maxeval: usize,
    seed_runs: usize,
) -> Result<(String, Vec<TestResult>)> {
    let mut out = String::new();
    let mut results = Vec::new();

    writeln!(out, "\n--- {} (IIR workflow) ---", name).unwrap();

    let mut baseline_scorecard: Option<MetricScorecard> = None;

    for mutation in IIR_MUTATIONS {
        let (mut config, _, _validation) = load_config(base_config_path, override_config_path)?;
        apply_qa_overrides(&mut config, &format!("{name}:iir:{mutation}"), maxeval);
        apply_mutation(&mut config, *mutation);

        let result = run_optimization(&config, seed_runs)
            .with_context(|| format!("{} IIR {}", name, mutation))?;

        let pre = result.combined_pre_score;
        let scorecard = compute_scorecard(&result);

        let (pass, reason) =
            evaluate_scorecard(*mutation, pre, &scorecard, &mut baseline_scorecard);

        let status = if pass { "PASS" } else { "FAIL" };
        writeln!(
            out,
            "  IIR {:>14}: {}  {}  ({})",
            mutation.to_string(),
            scorecard,
            status,
            reason
        )
        .unwrap();

        results.push(TestResult {
            label: format!("{} IIR {}", name, mutation),
            pre_score: pre,
            scorecard,
            pass,
            reason,
        });
    }

    Ok((out, results))
}

/// Exercise a non-IIR override through the production config-loading path.
///
/// The generic-path matrix deliberately mutates processing modes in memory;
/// this smoke gate is separate so a broken or misleading checked-in FIR or
/// Hybrid override cannot remain hidden behind that mutation.
pub(super) fn run_workflow_override_smoke(
    name: &str,
    mode_name: &str,
    expected_mode: ProcessingMode,
    base_config_path: &Path,
    override_config_path: &Path,
    maxeval: usize,
    seed_runs: usize,
) -> Result<(String, Vec<TestResult>)> {
    let mut out = String::new();
    let (mut config, _, _validation) = load_config(base_config_path, Some(override_config_path))?;
    anyhow::ensure!(
        config.optimizer.processing_mode == expected_mode,
        "{} workflow override {} claims {:?}, but its merged config selects {:?}",
        name,
        override_config_path.display(),
        expected_mode,
        config.optimizer.processing_mode
    );

    apply_qa_overrides(
        &mut config,
        &format!(
            "{name}:workflow:{}:baseline",
            mode_name.to_ascii_lowercase()
        ),
        maxeval,
    );
    let result = run_optimization(&config, seed_runs)
        .with_context(|| format!("{name} {mode_name} workflow baseline"))?;
    let pre = result.combined_pre_score;
    let scorecard = compute_scorecard(&result);
    let mut baseline_scorecard = None;
    let (pass, reason) =
        evaluate_scorecard(Mutation::Baseline, pre, &scorecard, &mut baseline_scorecard);
    let status = if pass { "PASS" } else { "FAIL" };
    writeln!(
        out,
        "  {mode_name:>10} workflow: {scorecard} {status} ({reason})"
    )
    .unwrap();

    Ok((
        out,
        vec![TestResult {
            label: format!("{name} {mode_name} workflow baseline"),
            pre_score: pre,
            scorecard,
            pass,
            reason,
        }],
    ))
}

pub(super) fn run_generic_path_tests(
    name: &str,
    base_config_path: &Path,
    override_config_dir: &Path,
    maxeval: usize,
    seed_runs: usize,
) -> Result<(String, Vec<TestResult>)> {
    let mut out = String::new();
    let mut results = Vec::new();

    writeln!(out, "\n--- Generic Path ({}, all modes) ---", name).unwrap();

    let modes: &[(&str, ProcessingMode, &str, &[Mutation])] = &[
        (
            "IIR",
            ProcessingMode::LowLatency,
            "optimiser-iir.json",
            IIR_MUTATIONS,
        ),
        (
            "FIR",
            ProcessingMode::PhaseLinear,
            "optimiser-fir.json",
            FIR_MUTATIONS,
        ),
        (
            "Mixed",
            ProcessingMode::Hybrid,
            "optimiser-mixed.json",
            MIXED_MUTATIONS,
        ),
        (
            "MixedPhase",
            ProcessingMode::MixedPhase,
            "../modes/optimiser-mixed-phase.json",
            MIXED_PHASE_MUTATIONS,
        ),
    ];

    let mut mode_baselines: Vec<(&str, f64)> = Vec::new();

    for (mode_name, processing_mode, override_file, mutations) in modes {
        let scenario_override = override_config_dir.join(override_file);
        let shared_override = override_config_dir
            .parent()
            .unwrap_or(override_config_dir)
            .join("modes")
            .join(
                Path::new(override_file)
                    .file_name()
                    .unwrap_or_else(|| override_file.as_ref()),
            );
        let override_path = if scenario_override.exists() {
            scenario_override
        } else {
            shared_override
        };
        let mut baseline_scorecard: Option<MetricScorecard> = None;

        for mutation in *mutations {
            let (mut config, _) = load_config_for_generic_path(
                base_config_path,
                Some(&override_path),
                processing_mode.clone(),
            )?;
            apply_qa_overrides(
                &mut config,
                &format!("{name}:generic:{mode_name}:{mutation}"),
                maxeval,
            );
            // Generic-path cases compare processing modes and budget/filter
            // mutations. Keep their scalar objective aligned with the flat-loss
            // scorecard and runtime acceptance metric; psychoacoustic and
            // asymmetric objectives have dedicated option-effect cases.
            config.optimizer.psychoacoustic = false;
            config.optimizer.asymmetric_loss = false;
            apply_mutation(&mut config, *mutation);

            let result = run_optimization(&config, seed_runs)
                .with_context(|| format!("{} {} generic {}", name, mode_name, mutation))?;

            let pre = result.combined_pre_score;
            let scorecard = compute_scorecard(&result);

            let (pass, reason) =
                evaluate_scorecard(*mutation, pre, &scorecard, &mut baseline_scorecard);

            // Record baseline for cross-mode comparison
            if matches!(mutation, Mutation::Baseline) {
                mode_baselines.push((mode_name, scorecard.flat_loss));
            }

            let status = if pass { "PASS" } else { "FAIL" };
            writeln!(
                out,
                "  {} {:>14}: {}  {}  ({})",
                mode_name,
                mutation.to_string(),
                scorecard,
                status,
                reason
            )
            .unwrap();

            results.push(TestResult {
                label: format!("{} generic {} {}", name, mode_name, mutation),
                pre_score: pre,
                scorecard,
                pass,
                reason,
            });
        }
    }

    // Cross-mode comparison
    if mode_baselines.len() >= 2 {
        let scores: Vec<f64> = mode_baselines.iter().map(|(_, s)| *s).collect();
        let min_score = scores.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_score = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let ratio = if min_score > 0.0 {
            max_score / min_score
        } else {
            f64::INFINITY
        };
        let pass = ratio <= CROSS_MODE_RATIO_LIMIT;
        let status = if pass { "PASS" } else { "FAIL" };

        let mode_scores: String = mode_baselines
            .iter()
            .map(|(name, score)| format!("{}={:.4}", name, score))
            .collect::<Vec<_>>()
            .join(" ");

        writeln!(
            out,
            "\n  Cross-mode: {} ratio={:.2}x  {}",
            mode_scores, ratio, status
        )
        .unwrap();

        results.push(TestResult {
            label: format!("{} cross-mode", name),
            pre_score: 0.0,
            scorecard: placeholder_scorecard(ratio),
            pass,
            reason: format!("ratio={:.2}x (limit={:.1}x)", ratio, CROSS_MODE_RATIO_LIMIT),
        });
    }

    Ok((out, results))
}

fn median(mut values: Vec<f64>) -> Option<f64> {
    values.retain(|value| value.is_finite());
    if values.is_empty() {
        return None;
    }
    values.sort_by(f64::total_cmp);
    let middle = values.len() / 2;
    Some(if values.len().is_multiple_of(2) {
        (values[middle - 1] + values[middle]) * 0.5
    } else {
        values[middle]
    })
}

pub(super) fn deployed_final_curve(
    result: &RoomOptimizationResult,
    channel: &str,
) -> Option<Curve> {
    result
        .deployed_source_curves
        .get(channel)
        .cloned()
        .or_else(|| {
            result
                .channel_results
                .get(channel)
                .map(|channel| channel.final_curve.clone())
        })
        .or_else(|| {
            result
                .channels
                .get(channel)
                .and_then(|chain| chain.final_curve.clone())
                .map(Curve::from)
        })
}

fn redirected_main_channels(result: &RoomOptimizationResult) -> Vec<String> {
    let routed: std::collections::BTreeSet<String> = result
        .metadata
        .bass_management
        .as_ref()
        .and_then(|report| report.routing_graph.as_ref())
        .into_iter()
        .flat_map(|graph| graph.routes.iter())
        .filter(|route| route.route_kind == "main_highpass_to_self")
        .map(|route| route.source_channel.clone())
        .collect();
    if !routed.is_empty() {
        return routed.into_iter().collect();
    }
    result
        .channel_results
        .keys()
        .filter(|name| {
            let lower = name.to_ascii_lowercase();
            !lower.contains("lfe") && !lower.contains("sub")
        })
        .cloned()
        .collect()
}

fn correction_passband_violations(result: &RoomOptimizationResult) -> Vec<String> {
    let mut violations = Vec::new();
    for (channel_name, channel_result) in &result.channel_results {
        let Some(reliable_upper_hz) = roomeq_engine::spectral_align::reliable_upper_passband_hz(
            &channel_result.initial_curve,
        ) else {
            continue;
        };
        let Some(chain) = result.channels.get(channel_name) else {
            continue;
        };
        for plugin in &chain.plugins {
            if plugin.plugin_type != "eq" {
                continue;
            }
            let Some(filters) = plugin
                .parameters
                .get("filters")
                .and_then(serde_json::Value::as_array)
            else {
                continue;
            };
            let stage = plugin
                .parameters
                .get("label")
                .or_else(|| plugin.parameters.get("room_eq_correction_stage"))
                .or_else(|| plugin.parameters.get("room_eq_stage"))
                .and_then(serde_json::Value::as_str)
                .unwrap_or("eq");
            for frequency_hz in filters
                .iter()
                .filter_map(|filter| filter.get("freq").and_then(serde_json::Value::as_f64))
            {
                if frequency_hz > reliable_upper_hz {
                    violations.push(format!(
                        "{channel_name}:{stage}:{frequency_hz:.1}Hz>{reliable_upper_hz:.1}Hz"
                    ));
                }
            }
        }

        let lower_channel_name = channel_name.to_ascii_lowercase();
        let is_bass_output =
            lower_channel_name.contains("lfe") || lower_channel_name.starts_with("sub");
        if !is_bass_output && let Some(eq_response) = chain.eq_response.as_ref() {
            let audit_start_hz = reliable_upper_hz * 2.0_f64.powf(1.0 / 6.0);
            let mut minimum_db = f64::INFINITY;
            let mut maximum_db = f64::NEG_INFINITY;
            let mut sample_count = 0usize;
            for (frequency_hz, level_db) in eq_response.freq.iter().zip(&eq_response.spl) {
                if *frequency_hz >= audit_start_hz && level_db.is_finite() {
                    minimum_db = minimum_db.min(*level_db);
                    maximum_db = maximum_db.max(*level_db);
                    sample_count += 1;
                }
            }
            let spread_db = maximum_db - minimum_db;
            if sample_count >= 3 && spread_db > CORRECTION_OUT_OF_BAND_SPREAD_DB {
                violations.push(format!(
                    "{channel_name}:out-of-band correction spread {spread_db:.2}dB>{CORRECTION_OUT_OF_BAND_SPREAD_DB:.2}dB above {audit_start_hz:.1}Hz"
                ));
            }
        }
    }
    violations
}

/// Complete channel and mode coverage for one strict parity band.
#[derive(Debug)]
pub(super) struct CrossModeBandComparison {
    pub(super) expected_comparisons: usize,
    pub(super) available_comparisons: usize,
    pub(super) unavailable: Vec<String>,
    pub(super) median_rms: f64,
    pub(super) max_rms: f64,
}

impl CrossModeBandComparison {
    pub(super) fn passes(&self, median_limit: f64, max_limit: Option<f64>) -> bool {
        self.expected_comparisons > 0
            && self.available_comparisons == self.expected_comparisons
            && self.unavailable.is_empty()
            && self.median_rms <= median_limit
            && max_limit.is_none_or(|limit| self.max_rms <= limit)
    }
}

pub(super) fn expected_parity_main_channels(
    config: &RoomConfig,
) -> std::collections::BTreeSet<String> {
    // Topology workflows publish logical roles, not measurement keys.
    let names: Vec<_> = match &config.system {
        Some(system) => system.speakers.keys().collect(),
        None => config.speakers.keys().collect(),
    };
    names
        .into_iter()
        .filter(|channel| !super::misc::is_lfe_or_sub_channel(channel))
        .cloned()
        .collect()
}

pub(super) fn compare_cross_mode_band(
    channel_curves: &[(String, Vec<Option<Curve>>)],
    mode_names: &[&str],
    fmin: f64,
    fmax: f64,
) -> CrossModeBandComparison {
    let mut differences = Vec::new();
    let mut unavailable = Vec::new();
    let mut expected_comparisons = 0;
    if channel_curves.is_empty() || mode_names.len() < 2 {
        unavailable.push("no declared channels or fewer than two modes".to_string());
    }
    for (channel, curves) in channel_curves {
        for first in 0..mode_names.len() {
            for second in (first + 1)..mode_names.len() {
                expected_comparisons += 1;
                let rms = curves
                    .get(first)
                    .and_then(Option::as_ref)
                    .zip(curves.get(second).and_then(Option::as_ref))
                    .and_then(|(first, second)| {
                        level_matched_rms_curve_difference_db(first, second, fmin, fmax)
                    });
                if let Some(rms) = rms {
                    differences.push(rms);
                } else {
                    unavailable.push(format!(
                        "{channel} {} vs {} ({fmin}..{fmax}Hz)",
                        mode_names[first], mode_names[second]
                    ));
                }
            }
        }
    }
    let available_comparisons = differences.len();
    let median_rms = median(differences.clone()).unwrap_or(f64::INFINITY);
    let max_rms = differences
        .into_iter()
        .reduce(f64::max)
        .unwrap_or(f64::INFINITY);
    CrossModeBandComparison {
        expected_comparisons,
        available_comparisons,
        unavailable,
        median_rms,
        max_rms,
    }
}

/// Require accepted correction in the requested family for convergence evidence.
pub(super) fn strict_cross_mode_correction(
    result: &RoomOptimizationResult,
    requested: &ProcessingMode,
) -> std::result::Result<(), String> {
    let report = result
        .metadata
        .correction_acceptance
        .as_ref()
        .ok_or_else(|| "missing correction acceptance report".to_string())?;
    let outcome = report.derived_outcome();
    if outcome != roomeq_model::RoomEqOutcome::Accepted || !report.violations.is_empty() {
        return Err(format!(
            "correction is {outcome:?} (decision={:?}, violations={:?})",
            report.decision, report.violations
        ));
    }
    // Metadata can precede graph conversion or contain a stale family label.
    // Inspect the emitted channel and driver plugins instead.
    let realized = roomeq_model::assess_realized_processing(&result.channels);
    if let Some(reason) = roomeq_model::processing_fallback_reason(Some(requested), &realized) {
        return Err(reason);
    }
    Ok(())
}

pub(super) fn run_cross_mode_convergence_tests(
    name: &str,
    base_config_path: &Path,
    override_config_dir: &Path,
    preserve_system: bool,
    strict: bool,
    maxeval: usize,
    seed_runs: usize,
) -> Result<(String, Vec<TestResult>)> {
    let mut out = String::new();
    let mut results = Vec::new();

    writeln!(out, "\n--- {} (cross-mode convergence) ---", name).unwrap();

    let modes: &[(&str, ProcessingMode, &str)] = &[
        ("IIR", ProcessingMode::LowLatency, "optimiser-iir.json"),
        ("FIR", ProcessingMode::PhaseLinear, "optimiser-fir.json"),
        ("Hybrid", ProcessingMode::Hybrid, "optimiser-mixed.json"),
        (
            "MixedPhase",
            ProcessingMode::MixedPhase,
            "optimiser-mixed-phase.json",
        ),
    ];

    // Run every production processing mode and collect comparable artifacts.
    let mut mode_results: Vec<(&str, RoomOptimizationResult)> = Vec::new();
    let mut expected_main_channels = std::collections::BTreeSet::new();
    let mut unavailable_corrections = Vec::new();

    for (mode_name, processing_mode, override_file) in modes {
        let override_path = override_config_dir.join(override_file);
        let (mut config, _) = load_config_for_path(
            base_config_path,
            Some(&override_path),
            processing_mode.clone(),
            preserve_system,
        )?;
        if !strict {
            apply_qa_overrides(
                &mut config,
                &format!("{name}:cross-mode:{mode_name}"),
                maxeval,
            );
        } else {
            clamp_strict_measured_maxeval(&mut config, maxeval);
            expected_main_channels.extend(expected_parity_main_channels(&config));
        }
        // Strict measured regressions exercise the checked-in production
        // fixture unchanged except for clamping optimizer.max_iter. Filter
        // count, algorithm, population, seed, and every acoustic option remain
        // the checked-in production values.

        let result = run_optimization(&config, seed_runs)
            .with_context(|| format!("{} {} cross-mode", name, mode_name))?;

        writeln!(
            out,
            "  {}: post={:.4} (pre={:.4})",
            mode_name, result.combined_post_score, result.combined_pre_score
        )
        .unwrap();

        let pre = result.combined_pre_score;
        let scorecard = compute_scorecard(&result);
        let (pass, reason) = if strict {
            let acceptance = strict_cross_mode_correction(&result, processing_mode);
            let finite = pre.is_finite()
                && scorecard.flat_loss.is_finite()
                && scorecard.max_boost_db.is_finite();
            let reason = match acceptance {
                Err(reason) => Some(reason),
                Ok(()) if !finite => Some("non-finite measured-mode metrics".to_string()),
                Ok(()) => None,
            };
            if let Some(reason) = reason {
                unavailable_corrections.push(format!("{mode_name}: {reason}"));
                (false, reason)
            } else {
                (
                    true,
                    "accepted requested-mode correction with finite metrics".to_string(),
                )
            }
        } else {
            let mut baseline_scorecard = None;
            evaluate_scorecard(Mutation::Baseline, pre, &scorecard, &mut baseline_scorecard)
        };
        results.push(TestResult {
            label: format!("{name} {mode_name} correction"),
            pre_score: pre,
            scorecard,
            pass,
            reason,
        });

        if strict {
            let violations = correction_passband_violations(&result);
            let pass = violations.is_empty();
            results.push(TestResult {
                label: format!("{name} {mode_name} measured-passband correction bounds"),
                pre_score: 0.0,
                scorecard: placeholder_scorecard(if pass { 0.0 } else { f64::INFINITY }),
                pass,
                reason: if pass {
                    "all correction stages stay within each measured passband".to_string()
                } else {
                    format!("out-of-passband correction: {}", violations.join(", "))
                },
            });
        }

        mode_results.push((mode_name, result.result));
    }

    // Keep rejected/baseline curves for diagnostics, but never promote their
    // agreement to successful correction convergence.
    let corrections_available = unavailable_corrections.is_empty();
    let acceptance_detail = if corrections_available {
        String::new()
    } else {
        format!(
            "; unavailable corrections: {}",
            unavailable_corrections.join("; ")
        )
    };

    // CM-1: Frequency-response convergence from the final deployed channel
    // curves. Strict cases use level-matched RMS bands; legacy generic cases
    // retain their historical broad maximum-difference smoke gate.
    if strict {
        // Configuration owns the expected channels. A missing channel in every
        // result must still contribute unavailable comparisons.
        let channel_curves: Vec<_> = expected_main_channels
            .into_iter()
            .map(|channel| {
                let curves = mode_results
                    .iter()
                    .map(|(_, result)| deployed_final_curve(result, &channel))
                    .collect();
                (channel, curves)
            })
            .collect();
        let mode_names: Vec<_> = mode_results.iter().map(|(name, _)| *name).collect();
        let bands = [
            (
                "bass",
                25.0,
                250.0,
                CROSS_MODE_BASS_MEDIAN_RMS_DB,
                Some(CROSS_MODE_BASS_MAX_RMS_DB),
            ),
            ("main", 100.0, 10_000.0, CROSS_MODE_MAIN_MEDIAN_RMS_DB, None),
            (
                "upper",
                300.0,
                10_000.0,
                CROSS_MODE_UPPER_MEDIAN_RMS_DB,
                None,
            ),
        ];
        for (band_name, fmin, fmax, median_limit, max_limit) in bands {
            let comparison = compare_cross_mode_band(&channel_curves, &mode_names, fmin, fmax);
            let median_rms = comparison.median_rms;
            let max_rms = comparison.max_rms;
            let pass = corrections_available && comparison.passes(median_limit, max_limit);
            let coverage = format!(
                "comparisons={}/{}",
                comparison.available_comparisons, comparison.expected_comparisons
            );
            let unavailable = if comparison.unavailable.is_empty() {
                String::new()
            } else {
                format!(", unavailable: {}", comparison.unavailable.join("; "))
            };
            let status = if pass { "PASS" } else { "FAIL" };
            writeln!(
                out,
                "  CM-1 {band_name} parity: median_rms={median_rms:.2}dB max_rms={max_rms:.2}dB {coverage}  {status}"
            )
            .unwrap();
            results.push(TestResult {
                label: format!("{name} CM-1 {band_name} parity"),
                pre_score: 0.0,
                scorecard: placeholder_scorecard(max_rms),
                pass,
                reason: format!(
                    "median_rms={median_rms:.2}dB (limit={median_limit:.2}dB), max_rms={max_rms:.2}dB{}, {coverage}{unavailable}{acceptance_detail}",
                    max_limit.map_or_else(String::new, |limit| format!(" (limit={limit:.2}dB)"))
                ),
            });
        }
    } else {
        let channel_names = redirected_main_channels(&mode_results[0].1);
        let mut cm1_max_diff = 0.0_f64;
        for ch_name in &channel_names {
            let curves: Vec<Curve> = mode_results
                .iter()
                .filter_map(|(_, result)| deployed_final_curve(result, ch_name))
                .collect();
            if curves.len() >= 2 {
                let curve_refs: Vec<&Curve> = curves.iter().collect();
                let diff = max_curve_difference_db(&curve_refs, 20.0, 500.0);
                cm1_max_diff = cm1_max_diff.max(diff);
            }
        }
        let pass = cm1_max_diff <= CROSS_MODE_FR_MAX_DIFF_DB;
        let status = if pass { "PASS" } else { "FAIL" };
        writeln!(
            out,
            "  CM-1 FR convergence: max_diff={cm1_max_diff:.2}dB (limit={CROSS_MODE_FR_MAX_DIFF_DB:.1}dB)  {status}"
        )
        .unwrap();
        results.push(TestResult {
            label: format!("{name} CM-1 FR convergence"),
            pre_score: 0.0,
            scorecard: placeholder_scorecard(cm1_max_diff),
            pass,
            reason: format!(
                "max_diff={cm1_max_diff:.2}dB (limit={CROSS_MODE_FR_MAX_DIFF_DB:.1}dB)"
            ),
        });
    }

    // CM-2: strict home-cinema cases keep group-delay dispersion bounded for
    // every mode. Hybrid is a frequency-band split and does not promise lower
    // group-delay dispersion than IIR; MixedPhase is validated independently.
    if strict {
        let channel_names = redirected_main_channels(&mode_results[0].1);
        let mut by_mode = vec![Vec::new(); mode_results.len()];
        for channel in &channel_names {
            for (mode_index, (_, result)) in mode_results.iter().enumerate() {
                if let Some(curve) = deployed_final_curve(result, channel)
                    && let Some(gd_std) = group_delay_std_dev(&curve, 100.0, 1_000.0)
                {
                    by_mode[mode_index].push(gd_std);
                }
            }
        }
        let medians: Vec<f64> = by_mode
            .into_iter()
            .map(|values| median(values).unwrap_or(f64::INFINITY))
            .collect();
        let pass = corrections_available
            && medians
                .iter()
                .all(|value| value.is_finite() && *value <= CROSS_MODE_TIMING_MAX_STD_MS);
        let detail = mode_results
            .iter()
            .zip(&medians)
            .map(|((mode_name, _), value)| format!("{mode_name}={value:.2}ms"))
            .collect::<Vec<_>>()
            .join(" ");
        let status = if pass { "PASS" } else { "FAIL" };
        writeln!(
            out,
            "  CM-2 timing sanity: {detail} limit={CROSS_MODE_TIMING_MAX_STD_MS:.2}ms  {status}"
        )
        .unwrap();
        results.push(TestResult {
            label: format!("{name} CM-2 timing sanity"),
            pre_score: 0.0,
            scorecard: placeholder_scorecard(
                medians.iter().copied().fold(f64::NEG_INFINITY, f64::max),
            ),
            pass,
            reason: format!(
                "{detail}; every mode must remain <= {CROSS_MODE_TIMING_MAX_STD_MS:.2}ms{acceptance_detail}"
            ),
        });
    } else {
        let channel_names: Vec<String> =
            mode_results[0].1.channel_results.keys().cloned().collect();
        let mut iir_gd_max = 0.0_f64;
        let mut fir_gd_max = 0.0_f64;
        let mut mixed_gd_max = 0.0_f64;
        let mut has_phase = false;
        for ch_name in &channel_names {
            for (mode_name, result) in &mode_results {
                if let Some(ch) = result.channel_results.get(ch_name)
                    && let Some(gd_std) = group_delay_std_dev(&ch.final_curve, 20.0, 500.0)
                {
                    has_phase = true;
                    match *mode_name {
                        "IIR" => iir_gd_max = iir_gd_max.max(gd_std),
                        "FIR" => fir_gd_max = fir_gd_max.max(gd_std),
                        "Mixed" => mixed_gd_max = mixed_gd_max.max(gd_std),
                        _ => {}
                    }
                }
            }
        }
        if has_phase {
            let max_gd = iir_gd_max.max(fir_gd_max).max(mixed_gd_max);
            let pass = max_gd < 50.0;
            let status = if pass { "PASS" } else { "FAIL" };
            writeln!(
                out,
                "  CM-2 GD flatness: IIR={iir_gd_max:.2}ms FIR={fir_gd_max:.2}ms Mixed={mixed_gd_max:.2}ms  {status}"
            )
            .unwrap();
            results.push(TestResult {
                label: format!("{name} CM-2 GD flatness"),
                pre_score: 0.0,
                scorecard: placeholder_scorecard(fir_gd_max.max(mixed_gd_max)),
                pass,
                reason: format!(
                    "IIR={iir_gd_max:.2}ms FIR={fir_gd_max:.2}ms Mixed={mixed_gd_max:.2}ms"
                ),
            });
        } else {
            writeln!(out, "  CM-2 GD flatness: SKIP (no phase data)").unwrap();
        }
    }

    // CM-3: Score convergence (ratio of max/min post scores)
    {
        let scores: Vec<f64> = mode_results
            .iter()
            .map(|(_, r)| r.combined_post_score)
            .collect();
        let min_score = scores.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_score = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let ratio = if min_score > 0.0 {
            max_score / min_score
        } else {
            f64::INFINITY
        };
        let cm3_pass = corrections_available && ratio <= CROSS_MODE_SCORE_RATIO_LIMIT;
        let status = if cm3_pass { "PASS" } else { "FAIL" };

        let mode_scores: String = mode_results
            .iter()
            .map(|(name, r)| format!("{}={:.4}", name, r.combined_post_score))
            .collect::<Vec<_>>()
            .join(" ");

        writeln!(
            out,
            "  CM-3 Score convergence: {} ratio={:.2}x (limit={:.1}x)  {}",
            mode_scores, ratio, CROSS_MODE_SCORE_RATIO_LIMIT, status
        )
        .unwrap();

        results.push(TestResult {
            label: format!("{} CM-3 score convergence", name),
            pre_score: 0.0,
            scorecard: placeholder_scorecard(ratio),
            pass: cm3_pass,
            reason: format!(
                "{} ratio={:.2}x (limit={:.1}x){acceptance_detail}",
                mode_scores, ratio, CROSS_MODE_SCORE_RATIO_LIMIT
            ),
        });
    }

    Ok((out, results))
}

#[allow(clippy::too_many_arguments)]
pub(super) fn run_option_effect_test(
    name: &str,
    fem_dir: &Path,
    fem_subdir: &str,
    optim_dir: &Path,
    optim_subdir: &str,
    options: &[OptionOverride],
    maxeval: usize,
    seed_runs: usize,
) -> Result<(String, Vec<TestResult>)> {
    let mut out = String::new();
    let mut results = Vec::new();

    let options_str: String = options
        .iter()
        .map(|o| o.to_string())
        .collect::<Vec<_>>()
        .join(" + ");
    writeln!(out, "\n--- {} ({}) ---", name, options_str).unwrap();

    let base_config_path = fem_dir.join(format!("{}/config.json", fem_subdir));
    let override_path = optim_dir.join(format!("{}/optimiser-iir.json", optim_subdir));
    let override_path = if override_path.exists() {
        Some(override_path)
    } else {
        None
    };

    let needs_multi_measurement = options.iter().any(option_needs_multi_measurement_paths);
    let needs_gd_trusted_measurements = options.iter().any(option_needs_gd_trusted_measurements);
    let needs_multisub_multi_seat = options.iter().any(option_needs_multisub_multi_seat_paths);
    let gd_profile = options.iter().find_map(option_gd_profile);
    let isolate_group_delay = options.iter().any(option_is_group_delay);

    // BroadbandTargetMatching needs a target tilt to have something to match.
    // When the combo doesn't include an explicit TargetTilt, both baseline and
    // option get a default -0.8 dB/oct tilt so the only variable is broadband.
    let has_broadband = options
        .iter()
        .any(|o| matches!(o, OptionOverride::BroadbandTargetMatching));
    let has_tilt = options
        .iter()
        .any(|o| matches!(o, OptionOverride::TargetTilt { .. }));
    let default_target_response = if has_broadband && !has_tilt {
        Some(TargetResponseConfig {
            shape: TargetShape::Custom,
            slope_db_per_octave: -0.8,
            ..TargetResponseConfig::default()
        })
    } else {
        None
    };

    // Load and run baseline (all options disabled)
    let (mut baseline_config, _, _validation) =
        load_config(&base_config_path, override_path.as_deref())?;
    apply_qa_overrides(
        &mut baseline_config,
        &format!("{name}:option-baseline"),
        maxeval,
    );
    for option in options {
        disable_option(&mut baseline_config, option);
    }
    isolate_schroeder_split_from_multi_measurement(&mut baseline_config, options);
    if let Some(ref tr) = default_target_response {
        baseline_config.optimizer.target_response = Some(tr.clone());
    }
    if isolate_group_delay {
        apply_group_delay_qa_passthrough_eq(&mut baseline_config);
    }
    prepare_option_measurement_paths(
        &mut baseline_config,
        fem_dir,
        fem_subdir,
        needs_multi_measurement,
        needs_gd_trusted_measurements,
        needs_multisub_multi_seat,
        gd_profile,
    )?;

    let baseline_result = run_optimization(&baseline_config, seed_runs)
        .with_context(|| format!("{} baseline", name))?;

    writeln!(
        out,
        "  baseline: post={:.4} (pre={:.4})",
        baseline_result.combined_post_score, baseline_result.combined_pre_score
    )
    .unwrap();

    // Load and run with all options enabled
    let (mut option_config, _, _validation) =
        load_config(&base_config_path, override_path.as_deref())?;
    apply_qa_overrides(
        &mut option_config,
        &format!("{name}:option-enabled"),
        maxeval,
    );
    for option in options {
        apply_option_override(&mut option_config, option);
    }
    isolate_schroeder_split_from_multi_measurement(&mut option_config, options);
    if let Some(ref tr) = default_target_response {
        option_config.optimizer.target_response = Some(tr.clone());
    }
    if isolate_group_delay {
        apply_group_delay_qa_passthrough_eq(&mut option_config);
    }
    prepare_option_measurement_paths(
        &mut option_config,
        fem_dir,
        fem_subdir,
        needs_multi_measurement,
        needs_gd_trusted_measurements,
        needs_multisub_multi_seat,
        gd_profile,
    )?;

    let option_result = run_optimization(&option_config, seed_runs)
        .with_context(|| format!("{} with-options", name))?;

    writeln!(
        out,
        "  with-options: post={:.4} (pre={:.4})",
        option_result.combined_post_score, option_result.combined_pre_score
    )
    .unwrap();

    // Validate each per-option invariant individually
    let mut all_pass = true;
    for option in options {
        let (pass, reason) = validate_option_effect(
            option,
            &baseline_config,
            &baseline_result,
            &option_config,
            &option_result,
            options,
        );

        let status = if pass { "PASS" } else { "FAIL" };
        writeln!(out, "  {}: {}  ({})", option, status, reason).unwrap();

        if !pass {
            all_pass = false;
            results.push(TestResult {
                label: format!("{} [{}]", name, option),
                pre_score: option_result.combined_pre_score,
                scorecard: compute_scorecard(&option_result),
                pass: false,
                reason,
            });
        }
    }

    // Combo-level scorecard check: compare option result against baseline
    // using the multi-metric scorecard. Combos with multiple options face
    // conflicting constraints (e.g., schroeder split + asymmetric loss) that
    // shrink the feasible region. Allow a small convergence margin that scales
    // with the number of options.
    let option_scorecard = compute_scorecard(&option_result);
    let baseline_scorecard = compute_scorecard(&baseline_result);

    let convergence_margin = match options.len() {
        0..=1 => option_result.combined_pre_score * 0.01, // 1% — optimizer budget is tight, allow noise
        2..=3 => option_result.combined_pre_score * 0.05, // 5% for 2-3 options
        _ => option_result.combined_pre_score * 0.15,     // 15% for 4+ options
    };
    // Target-reshaping options (tilt, broadband matching) deliberately move the
    // response away from flat, so the flat-loss convergence gate is not a valid
    // acceptance criterion for combos containing them; the per-option validators
    // above (tilt slope error, broadband shelves, double-tilt check) are the
    // authoritative gates in that case.
    let target_reshaped = options.iter().any(OptionOverride::reshapes_target);
    let converged = option_result.combined_post_score
        < option_result.combined_pre_score
            + convergence_margin
            // Degenerate flat-in/flat-out fixtures (e.g. group-delay isolation
            // with 0 dB EQ bounds) score exactly 0 == 0: count equality as
            // non-regression instead of failing the strict comparison.
            + convergence_epsilon(option_result.combined_pre_score);

    // Run scorecard comparison (informational for option tests — per-option
    // validators remain the primary gates, but EPA/peak/GD violations are surfaced)
    let scorecard_checks = compare_scorecards(&baseline_scorecard, &option_scorecard);
    let scorecard_failures: Vec<String> = scorecard_checks
        .iter()
        .filter(|(_, pass, _)| !pass)
        .map(|(name, _, detail)| format!("{}: {}", name, detail))
        .collect();

    if !converged && target_reshaped {
        writeln!(
            out,
            "  convergence: SKIP  (flat-loss gate n/a: option set reshapes the target; post {:.6} vs pre {:.6} informational)",
            option_result.combined_post_score, option_result.combined_pre_score
        )
        .unwrap();
    } else if !converged {
        all_pass = false;
        let reason = format!(
            "no convergence: post {:.6} >= pre {:.6} (+{:.6} margin)",
            option_result.combined_post_score,
            option_result.combined_pre_score,
            convergence_margin + convergence_epsilon(option_result.combined_pre_score)
        );
        writeln!(out, "  convergence: FAIL  ({})", reason).unwrap();
        results.push(TestResult {
            label: format!("{} [convergence]", name),
            pre_score: option_result.combined_pre_score,
            scorecard: option_scorecard.clone(),
            pass: false,
            reason,
        });
    }

    // Scorecard failures are blocking quality failures, not warnings.
    if !scorecard_failures.is_empty() {
        all_pass = false;
        let reason = scorecard_failures.join("; ");
        writeln!(out, "  scorecard: FAIL [{}]", reason).unwrap();
        results.push(TestResult {
            label: format!("{} [scorecard]", name),
            pre_score: option_result.combined_pre_score,
            scorecard: option_scorecard.clone(),
            pass: false,
            reason,
        });
    }
    // Registry improvement is correction quality (the option run's own
    // uncorrected input -> corrected output). The per-option validators above
    // independently compare the requested effect against the default baseline.
    // Requiring every tuning value to outperform the default tuning would make
    // parameter sweeps fail even when they are safe, effective, and distinct.
    // If everything passed, push a single PASS result.
    if all_pass {
        results.push(TestResult {
            label: name.to_string(),
            pre_score: option_result.combined_pre_score,
            scorecard: option_scorecard,
            pass: true,
            reason: format!(
                "all {} invariants pass [{}]",
                options.len(),
                compute_scorecard(&option_result)
            ),
        });
    }

    Ok((out, results))
}
