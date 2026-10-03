//! Integration tests for the roomeq binary

#[cfg(unix)]
use std::collections::BTreeMap;
use std::fs;
#[cfg(unix)]
use std::io::{self, Read};
#[cfg(unix)]
use std::path::Path;
use std::path::PathBuf;
#[cfg(unix)]
use std::process::{Child, Command, ExitStatus, Stdio};
#[cfg(unix)]
use std::sync::{Arc, Mutex, mpsc};
#[cfg(unix)]
use std::thread;
#[cfg(unix)]
use std::time::{Duration, Instant};

mod common;

use common::binary_runner::{BinaryRunner, ProcessBinaryRunner, run_roomeq};

#[cfg(unix)]
const MAX_CAPTURED_CHILD_OUTPUT_BYTES: usize = 2 * 1024 * 1024;
#[cfg(unix)]
const MAX_PROGRESS_LINE_BYTES: usize = 16 * 1024;
#[cfg(unix)]
const ROOM_CONFIGURATION_TIMEOUT: Duration = Duration::from_secs(90);
#[cfg(unix)]
const ROOM_OPTIMIZATION_READINESS_TIMEOUT: Duration = Duration::from_secs(30);
#[cfg(unix)]
const ROOM_CANCELLATION_TIMEOUT: Duration = Duration::from_secs(20);

#[cfg(unix)]
#[derive(Default)]
struct CapturedChildOutput {
    stdout: Vec<u8>,
    stderr: Vec<u8>,
    truncated: bool,
}

#[cfg(unix)]
struct SpawnedRoomEq {
    child: Option<Child>,
    terminal_status: Option<ExitStatus>,
    output: Arc<Mutex<CapturedChildOutput>>,
    progress_lines: mpsc::Receiver<String>,
    reader_threads: Vec<thread::JoinHandle<()>>,
}

#[cfg(unix)]
impl SpawnedRoomEq {
    fn spawn(config_path: &Path, output_path: &Path, frequency_samples: usize) -> io::Result<Self> {
        let mut child = Command::new(env!("CARGO_BIN_EXE_roomeq"))
            .args([
                "--config",
                config_path.to_str().expect("UTF-8 config path"),
                "--output",
                output_path.to_str().expect("UTF-8 output path"),
                "--sample-rate",
                "48000",
                "--freq-samples",
            ])
            .arg(frequency_samples.to_string())
            .env("RUST_LOG", "info")
            .env("RAYON_NUM_THREADS", "2")
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()?;
        let stdout = child.stdout.take().expect("piped stdout");
        let stderr = child.stderr.take().expect("piped stderr");
        let output = Arc::new(Mutex::new(CapturedChildOutput::default()));
        let (progress_sender, progress_lines) = mpsc::sync_channel(256);
        let stdout_output = Arc::clone(&output);
        let stderr_output = Arc::clone(&output);
        let reader_threads = vec![
            thread::spawn(move || capture_stdout(stdout, stdout_output)),
            thread::spawn(move || capture_stderr(stderr, stderr_output, progress_sender)),
        ];

        Ok(Self {
            child: Some(child),
            terminal_status: None,
            output,
            progress_lines,
            reader_threads,
        })
    }

    fn try_wait(&mut self) -> io::Result<Option<ExitStatus>> {
        if let Some(status) = self.terminal_status {
            return Ok(Some(status));
        }
        let Some(child) = self.child.as_mut() else {
            return Ok(None);
        };
        let status = child.try_wait()?;
        if let Some(status) = status {
            self.terminal_status = Some(status);
            self.child = None;
        }
        Ok(status)
    }

    fn send_interrupt(&mut self) -> io::Result<()> {
        if self.try_wait()?.is_some() {
            return Err(io::Error::new(
                io::ErrorKind::NotConnected,
                "RoomEQ exited before the test could send SIGINT",
            ));
        }
        let process_id = self.child.as_ref().expect("running child is retained").id();
        let status = Command::new("kill")
            .args(["-INT", &process_id.to_string()])
            .status()?;
        if !status.success() {
            return Err(io::Error::other(format!(
                "kill -INT {process_id} exited with {status}"
            )));
        }
        Ok(())
    }

    fn wait_for_parallel_progress(&mut self, timeout: Duration) -> Result<Vec<String>, String> {
        let deadline = Instant::now() + timeout;
        let mut stereo_route_seen = false;
        let mut left_progress_seen = false;
        let mut right_progress_seen = false;
        let mut readiness_evidence = Vec::new();

        loop {
            if let Some(status) = self.try_wait().map_err(|error| error.to_string())? {
                return Err(format!(
                    "RoomEQ exited before both stereo workers reported progress ({status}); {}",
                    self.output_text()
                ));
            }
            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                return Err(format!(
                    "timed out waiting for stereo route and both channel workers; {}",
                    self.output_text()
                ));
            }
            let wait = remaining.min(Duration::from_millis(100));
            match self.progress_lines.recv_timeout(wait) {
                Ok(line) => {
                    if !stereo_route_seen && line.contains("Selected Stereo 2.0 workflow") {
                        stereo_route_seen = true;
                        readiness_evidence.push(line.clone());
                    }
                    let iteration = line
                        .split_once("iter ")
                        .and_then(|(_, progress)| progress.split_once('/'))
                        .and_then(|(iteration, _)| iteration.trim().parse::<usize>().ok());
                    if iteration.is_some_and(|iteration| iteration >= 100) {
                        if !left_progress_seen && line.contains("[L]") {
                            left_progress_seen = true;
                            readiness_evidence.push(line.clone());
                        }
                        if !right_progress_seen && line.contains("[R]") {
                            right_progress_seen = true;
                            readiness_evidence.push(line.clone());
                        }
                    }
                    if stereo_route_seen && left_progress_seen && right_progress_seen {
                        if let Some(status) = self.try_wait().map_err(|error| error.to_string())? {
                            return Err(format!(
                                "RoomEQ exited after readiness but before SIGINT ({status}); {}",
                                self.output_text()
                            ));
                        }
                        return Ok(readiness_evidence);
                    }
                }
                Err(mpsc::RecvTimeoutError::Timeout) => {}
                Err(mpsc::RecvTimeoutError::Disconnected) => {
                    return Err(format!(
                        "RoomEQ closed its progress stream before readiness; {}",
                        self.output_text()
                    ));
                }
            }
        }
    }

    fn wait_until_exit(&mut self, timeout: Duration) -> io::Result<ExitStatus> {
        let deadline = Instant::now() + timeout;
        loop {
            if let Some(status) = self.try_wait()? {
                self.join_readers();
                return Ok(status);
            }
            if Instant::now() >= deadline {
                return Err(io::Error::new(
                    io::ErrorKind::TimedOut,
                    "timed out waiting for RoomEQ child process",
                ));
            }
            thread::sleep(Duration::from_millis(20));
        }
    }

    fn output_text(&self) -> String {
        let output = self.output.lock().expect("captured output mutex");
        let mut rendered = format!(
            "stdout:\n{}\nstderr:\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        if output.truncated {
            rendered.push_str("\n[child output truncated at 2 MiB per stream]");
        }
        rendered
    }

    fn save_output(&self, directory: &Path, name: &str) -> io::Result<bool> {
        let output = self.output.lock().expect("captured output mutex");
        fs::write(directory.join(format!("{name}.stdout.log")), &output.stdout)?;
        fs::write(directory.join(format!("{name}.stderr.log")), &output.stderr)?;
        Ok(output.truncated)
    }

    fn join_readers(&mut self) {
        for reader in self.reader_threads.drain(..) {
            let _ = reader.join();
        }
    }
}

#[cfg(unix)]
impl Drop for SpawnedRoomEq {
    fn drop(&mut self) {
        if let Some(child) = self.child.as_mut() {
            let _ = child.kill();
            let _ = child.wait();
            self.child = None;
        }
        self.join_readers();
    }
}

#[cfg(unix)]
fn capture_stdout(mut reader: impl Read, output: Arc<Mutex<CapturedChildOutput>>) {
    let mut bytes = [0_u8; 4096];
    loop {
        match reader.read(&mut bytes) {
            Ok(0) | Err(_) => return,
            Ok(count) => append_captured_output(&output, false, &bytes[..count]),
        }
    }
}

#[cfg(unix)]
fn capture_stderr(
    reader: impl Read,
    output: Arc<Mutex<CapturedChildOutput>>,
    progress_sender: mpsc::SyncSender<String>,
) {
    let mut reader = reader;
    let mut bytes = [0_u8; 4096];
    let mut line = Vec::with_capacity(MAX_PROGRESS_LINE_BYTES);
    let mut discard_line = false;
    loop {
        let count = match reader.read(&mut bytes) {
            Ok(0) => {
                if !discard_line && !line.is_empty() {
                    let _ = progress_sender.try_send(String::from_utf8_lossy(&line).into_owned());
                }
                return;
            }
            Ok(count) => count,
            Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
            Err(_) => return,
        };
        append_captured_output(&output, true, &bytes[..count]);
        for &byte in &bytes[..count] {
            if byte == b'\n' {
                if !discard_line {
                    let _ = progress_sender.try_send(String::from_utf8_lossy(&line).into_owned());
                }
                line.clear();
                discard_line = false;
            } else if !discard_line {
                if line.len() < MAX_PROGRESS_LINE_BYTES {
                    line.push(byte);
                } else {
                    line.clear();
                    discard_line = true;
                }
            }
        }
    }
}

#[cfg(unix)]
fn append_captured_output(output: &Arc<Mutex<CapturedChildOutput>>, is_stderr: bool, bytes: &[u8]) {
    let mut output = output.lock().expect("captured output mutex");
    let stream = if is_stderr {
        &mut output.stderr
    } else {
        &mut output.stdout
    };
    let remaining = MAX_CAPTURED_CHILD_OUTPUT_BYTES.saturating_sub(stream.len());
    let retained = bytes.len().min(remaining);
    stream.extend_from_slice(&bytes[..retained]);
    output.truncated |= retained < bytes.len();
}

#[cfg(unix)]
fn write_signal_test_config(directory: &Path, max_iterations: usize) -> PathBuf {
    let fixture_path =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/data/roomeq/test_config_stereo.json");
    let fixture_directory = fixture_path.parent().expect("fixture parent");
    let mut config: serde_json::Value =
        serde_json::from_slice(&fs::read(&fixture_path).expect("read checked-in stereo fixture"))
            .expect("parse checked-in stereo fixture");
    for speaker in ["left", "right"] {
        let reference = config["speakers"][speaker]
            .as_str()
            .expect("fixture speaker path");
        config["speakers"][speaker] = serde_json::json!(fixture_directory.join(reference));
    }
    config["system"] = serde_json::json!({
        "model": "stereo",
        "speakers": { "L": "left", "R": "right" }
    });
    config["optimizer"]["max_iter"] = serde_json::json!(max_iterations);
    config["optimizer"]["algorithm"] = serde_json::json!("autoeq:de");
    config["optimizer"]["population"] = serde_json::json!(36);
    config["optimizer"]["seed"] = serde_json::json!(7);
    config["optimizer"]["strategy"] = serde_json::json!("best1bin");

    let config_path = directory.join(format!("signal-room-{max_iterations}.json"));
    fs::write(
        &config_path,
        serde_json::to_vec_pretty(&config).expect("serialize signal-test config"),
    )
    .expect("write signal-test config");
    config_path
}

#[cfg(unix)]
fn collect_asset_files(
    root: &Path,
    current: &Path,
    files: &mut BTreeMap<PathBuf, Vec<u8>>,
) -> io::Result<()> {
    for entry in fs::read_dir(current)? {
        let entry = entry?;
        if entry.file_type()?.is_dir() {
            collect_asset_files(root, &entry.path(), files)?;
        } else if entry.file_type()?.is_file() {
            let path = entry.path();
            let path_from_root = path.strip_prefix(root).map_err(io::Error::other)?;
            files.insert(path_from_root.to_path_buf(), fs::read(path)?);
        }
    }
    Ok(())
}

#[cfg(unix)]
fn snapshot_room_bundle(output_path: &Path) -> io::Result<BTreeMap<PathBuf, Vec<u8>>> {
    let parent = output_path
        .parent()
        .filter(|path| !path.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    let mut files = BTreeMap::new();
    files.insert(PathBuf::from("native-output.json"), fs::read(output_path)?);
    let assets = roomeq_workflow::assets_dir_for(output_path);
    if assets.exists() {
        collect_asset_files(parent, &assets, &mut files)?;
    }
    Ok(files)
}

#[cfg(unix)]
fn room_stage_directories(directory: &Path) -> io::Result<Vec<PathBuf>> {
    let mut stages = Vec::new();
    for entry in fs::read_dir(directory)? {
        let entry = entry?;
        let name = entry.file_name();
        let Some(name) = name.to_str() else {
            continue;
        };
        if name.starts_with(".roomeq-attempt-")
            || name.starts_with(".roomeq-artifacts-")
            || name.starts_with(".roomeq-fallback-")
        {
            stages.push(entry.path());
        }
    }
    Ok(stages)
}

#[cfg(unix)]
#[test]
fn room_cli_sigint_drains_stereo_optimization_and_preserves_prior_bundle() {
    let directory = tempfile::TempDir::new().expect("create RoomEQ signal test directory");
    let output_path = directory.path().join("room-output.json");
    let config_path = write_signal_test_config(directory.path(), 100);

    let mut baseline =
        SpawnedRoomEq::spawn(&config_path, &output_path, 64).expect("spawn baseline RoomEQ CLI");
    let baseline_status = baseline
        .wait_until_exit(ROOM_CONFIGURATION_TIMEOUT)
        .unwrap_or_else(|error| {
            panic!(
                "baseline RoomEQ run did not finish: {error}; {}",
                baseline.output_text()
            )
        });
    assert!(
        baseline_status.success(),
        "baseline RoomEQ run failed: {}",
        baseline.output_text()
    );
    assert_eq!(
        baseline.try_wait().expect("cached baseline status"),
        Some(baseline_status)
    );
    roomeq_workflow::load_output_bundle(&output_path)
        .expect("baseline must produce a valid prior bundle");
    let manifest = roomeq_workflow::bundle_manifest_path_for(&output_path);
    assert!(manifest.is_file(), "baseline bundle manifest is present");
    let prior_bundle = snapshot_room_bundle(&output_path).expect("snapshot prior graph and assets");

    // The source curves, stereo mapping, DE seed/population, rate, PEQ count,
    // sample grid, and two-thread Rayon pool are identical to the baseline.
    // Only the optimizer iteration ceiling increases to provide a bounded
    // window for observing live progress and sending SIGINT.
    let config_path = write_signal_test_config(directory.path(), 100_000);
    let mut interrupted = SpawnedRoomEq::spawn(&config_path, &output_path, 64)
        .expect("spawn RoomEQ CLI to interrupt");
    let readiness_evidence = interrupted
        .wait_for_parallel_progress(ROOM_OPTIMIZATION_READINESS_TIMEOUT)
        .unwrap_or_else(|error| panic!("RoomEQ readiness failed: {error}"));
    eprintln!(
        "RoomEQ SIGINT test observed route and both channel progress events:\n{}",
        readiness_evidence.join("\n")
    );
    interrupted
        .send_interrupt()
        .expect("send SIGINT to the owned RoomEQ process");
    let interrupted_status = interrupted
        .wait_until_exit(ROOM_CANCELLATION_TIMEOUT)
        .unwrap_or_else(|error| {
            panic!(
                "RoomEQ did not drain after SIGINT within the bounded wait: {error}; {}",
                interrupted.output_text()
            )
        });
    assert_eq!(
        interrupted.try_wait().expect("cached terminal status"),
        Some(interrupted_status),
        "repeated status checks must retain the already-reaped child status"
    );
    assert!(
        interrupted_status.code().is_some(),
        "Tokio should handle SIGINT and the CLI should exit normally, not by signal"
    );
    assert!(
        !interrupted_status.success(),
        "cancelled run must not report success"
    );
    let child_output = interrupted.output_text();
    assert!(
        child_output
            .to_ascii_lowercase()
            .contains("stopped by observer")
            || child_output
                .to_ascii_lowercase()
                .contains("cancelled before"),
        "expected an explicit cancellation error; {child_output}"
    );

    assert_eq!(
        snapshot_room_bundle(&output_path).expect("snapshot preserved bundle"),
        prior_bundle,
        "cancellation during optimization must preserve the full graph and asset bundle"
    );
    roomeq_workflow::load_output_bundle(&output_path)
        .expect("preserved prior bundle remains loadable");
    assert!(
        room_stage_directories(directory.path())
            .expect("inspect output parent")
            .is_empty(),
        "cancelled optimization must remove its private staging directories"
    );

    if std::env::var_os("ROOMEQ_KEEP_SIGNAL_TEST_ARTIFACTS").is_some() {
        assert!(
            !baseline
                .save_output(directory.path(), "baseline")
                .expect("save baseline child output"),
            "baseline child output exceeded its bounded capture"
        );
        assert!(
            !interrupted
                .save_output(directory.path(), "interrupted")
                .expect("save interrupted child output"),
            "interrupted child output exceeded its bounded capture"
        );
        let retained_path = directory.keep();
        eprintln!(
            "Retained RoomEQ SIGINT child evidence: directory={}, baseline_config={}, cancel_config={}, canonical_bundle={}, assets={}",
            retained_path.display(),
            retained_path.join("signal-room-100.json").display(),
            retained_path.join("signal-room-100000.json").display(),
            retained_path.join("room-output.json").display(),
            retained_path.join("room-output_files").display()
        );
    }
}

#[cfg(unix)]
#[test]
fn room_cli_cobra_publishes_a_loadable_stereo_bundle() {
    let directory = tempfile::TempDir::new().expect("COBRA RoomEQ test directory");
    let config_path = write_signal_test_config(directory.path(), 24);
    let mut config: serde_json::Value =
        serde_json::from_slice(&fs::read(&config_path).unwrap()).unwrap();
    config["optimizer"]["algorithm"] = serde_json::json!("autoeq:cobra");
    config["optimizer"]["num_filters"] = serde_json::json!(1);
    config["optimizer"]["refine"] = serde_json::json!(false);
    fs::write(&config_path, serde_json::to_vec_pretty(&config).unwrap()).unwrap();
    let output_path = directory.path().join("cobra-room.json");
    let mut child =
        SpawnedRoomEq::spawn(&config_path, &output_path, 64).expect("spawn COBRA RoomEQ");
    let status = child
        .wait_until_exit(ROOM_CONFIGURATION_TIMEOUT)
        .unwrap_or_else(|error| {
            panic!(
                "COBRA RoomEQ did not finish: {error}; {}",
                child.output_text()
            )
        });
    assert!(
        status.success(),
        "COBRA RoomEQ failed: {}",
        child.output_text()
    );
    roomeq_workflow::load_output_bundle(&output_path).expect("COBRA bundle is valid and loadable");
    let graph: serde_json::Value =
        serde_json::from_slice(&fs::read(&output_path).unwrap()).unwrap();
    assert_eq!(graph["metadata"]["algorithm"], "autoeq:cobra");
    assert!(graph["channels"].get("L").is_some());
    assert!(graph["channels"].get("R").is_some());
    let evidence = graph["metadata"]["optimizer_evidence"].to_string();
    assert!(
        evidence.contains("AutoEQ COBRA:"),
        "missing actual COBRA run evidence: {evidence}"
    );
    assert!(room_stage_directories(directory.path()).unwrap().is_empty());
}

fn centered_rms_in_band(curve: &serde_json::Value, min_hz: f64, max_hz: f64) -> f64 {
    let frequencies = curve["freq"].as_array().expect("curve frequency array");
    let spl = curve["spl"].as_array().expect("curve SPL array");
    assert_eq!(frequencies.len(), spl.len());
    let values: Vec<f64> = frequencies
        .iter()
        .zip(spl)
        .filter_map(|(frequency, spl)| {
            let frequency = frequency.as_f64()?;
            (min_hz..=max_hz)
                .contains(&frequency)
                .then(|| spl.as_f64())
                .flatten()
        })
        .collect();
    assert!(values.len() >= 3, "insufficient score bins: {values:?}");
    assert!(values.iter().all(|value| value.is_finite()));
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    (values
        .iter()
        .map(|value| (value - mean).powi(2))
        .sum::<f64>()
        / values.len() as f64)
        .sqrt()
}

#[test]
fn test_roomeq_stereo_config() {
    let temp_dir = tempfile::TempDir::new().expect("Failed to create temp dir");
    let output_path = temp_dir.path().join("output.json");

    let config_path =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/data/roomeq/test_config_stereo.json");

    // Run roomeq binary
    let runner = ProcessBinaryRunner::new();
    let output = runner
        .run(
            "roomeq",
            &[
                "--config",
                config_path.to_str().unwrap(),
                "--output",
                output_path.to_str().unwrap(),
                "--sample-rate",
                "48000",
            ],
        )
        .expect("Failed to execute roomeq");

    // Check that it ran successfully
    if !output.status.success() {
        eprintln!("stdout: {}", String::from_utf8_lossy(&output.stdout));
        eprintln!("stderr: {}", String::from_utf8_lossy(&output.stderr));
        panic!("roomeq failed with status: {}", output.status);
    }

    // Verify output file was created
    assert!(output_path.exists(), "Output file was not created");

    // Parse and validate output, restoring measurement blobs extracted
    // into the sibling assets directory.
    let bundle =
        roomeq_workflow::load_output_bundle(&output_path).expect("Failed to load output bundle");
    let json: serde_json::Value =
        serde_json::to_value(&bundle).expect("Failed to serialize output bundle");

    // Verify structure
    assert!(json.get("channels").is_some(), "Missing 'channels' field");
    assert!(json.get("metadata").is_some(), "Missing 'metadata' field");

    let channels = json["channels"]
        .as_object()
        .expect("channels should be an object");

    // Should have left and right channels
    assert!(channels.contains_key("left"), "Missing 'left' channel");
    assert!(channels.contains_key("right"), "Missing 'right' channel");

    // Validate left channel has plugins
    let left_channel = &channels["left"];
    assert!(
        left_channel.get("channel").is_some(),
        "Missing channel name"
    );
    assert!(left_channel.get("plugins").is_some(), "Missing plugins");

    let plugins = left_channel["plugins"]
        .as_array()
        .expect("plugins should be an array");

    // Should have at least an EQ plugin
    assert!(!plugins.is_empty(), "No plugins in DSP chain");

    // Check for EQ plugin
    let has_eq = plugins.iter().any(|p| {
        p.get("plugin_type")
            .and_then(|t| t.as_str())
            .map(|t| t == "eq")
            .unwrap_or(false)
    });
    assert!(has_eq, "Missing EQ plugin in DSP chain");

    let metadata = json["metadata"].as_object().expect("metadata object");
    let pre = metadata["pre_score"].as_f64().expect("finite pre_score");
    let post = metadata["post_score"].as_f64().expect("finite post_score");
    assert!(pre.is_finite() && post.is_finite());
    assert!(post <= pre, "stereo RoomEQ score worsened: {pre} -> {post}");

    for channel_name in ["left", "right"] {
        let channel = &channels[channel_name];
        let initial = &channel["initial_curve"];
        let final_curve = &channel["final_curve"];
        let independently_scored_pre = centered_rms_in_band(initial, 20.0, 20_000.0);
        let independently_scored_post = centered_rms_in_band(final_curve, 20.0, 20_000.0);
        assert!(
            independently_scored_post + 0.01 < independently_scored_pre,
            "{channel_name} correction did not materially improve independently computed flatness: {independently_scored_pre} -> {independently_scored_post}"
        );
    }
}

#[test]
fn test_roomeq_multidriver_missing_phase_exports_rejected_diagnostic() {
    let temp_dir = tempfile::TempDir::new().expect("Failed to create temp dir");
    let output_path = temp_dir.path().join("output_multidriver.json");

    let config_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/data/roomeq/test_config_multidriver.json");

    // Run roomeq binary
    let output = run_roomeq(&[
        "--config",
        config_path.to_str().unwrap(),
        "--output",
        output_path.to_str().unwrap(),
        "--sample-rate",
        "48000",
        "--verbose",
    ]);

    // Magnitude-only branches cannot establish coherent playback.
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("not approved for playback"));

    // An unapproved diagnostic is retained inside its attempt directory; it
    // must not replace or create the canonical playback output.
    assert!(
        !output_path.exists(),
        "rejected diagnostic must not publish canonical output"
    );
    let mut retained_attempts = Vec::new();
    for entry in fs::read_dir(temp_dir.path()).expect("inspect output parent") {
        let entry = entry.expect("read output-parent entry");
        if entry
            .file_name()
            .to_string_lossy()
            .starts_with(".roomeq-attempt-")
        {
            retained_attempts.push(entry.path());
        }
    }
    assert_eq!(
        retained_attempts.len(),
        1,
        "expected exactly one retained diagnostic attempt"
    );
    let diagnostic_path =
        retained_attempts[0].join(output_path.file_name().expect("output filename"));
    assert!(
        diagnostic_path.is_file(),
        "retained diagnostic is missing: {}",
        diagnostic_path.display()
    );

    // Parse and validate output
    let json_str = fs::read_to_string(&diagnostic_path).expect("Failed to read diagnostic output");
    let json: serde_json::Value =
        serde_json::from_str(&json_str).expect("Failed to parse output JSON");

    let acceptance = &json["metadata"]["correction_acceptance"];
    assert_eq!(acceptance["accepted"], false);
    assert_eq!(acceptance["outcome"], "insufficient_evidence");
    assert!(
        acceptance["violations"]
            .as_array()
            .unwrap()
            .iter()
            .any(|reason| {
                reason
                    .as_str()
                    .is_some_and(|reason| reason.contains("phase"))
            })
    );

    // Verify the rejected diagnostic retains its full topology.
    let channels = json["channels"]
        .as_object()
        .expect("channels should be an object");

    // Should have left channel
    assert!(channels.contains_key("left"), "Missing 'left' channel");

    let left_channel = &channels["left"];
    let plugins = left_channel["plugins"]
        .as_array()
        .expect("plugins should be an array");

    // Multi-driver exports keep active-crossover DSP on each driver branch.
    // The EQ computed on the combined (summed) response is intentionally
    // placed at channel level, upstream of the crossover split — see
    // `build_multidriver_dsp_chain` ("Build combined EQ (applied to summed
    // output)"). Final channel-level alignment may add one labelled gain;
    // crossover processing must remain on the driver branches.
    for plugin in plugins {
        let combined_eq = plugin["plugin_type"] == "eq";
        let level_alignment = plugin["plugin_type"] == "gain"
            && plugin["parameters"]["label"] == "final_channel_level_alignment"
            && plugin["parameters"]["gain_db"]
                .as_f64()
                .is_some_and(f64::is_finite);
        let safety_headroom = plugin["plugin_type"] == "gain"
            && plugin["parameters"]["label"] == "final_electrical_headroom"
            && plugin["parameters"]["room_eq_safety_gain"] == true
            && plugin["parameters"]["gain_db"]
                .as_f64()
                .is_some_and(|gain| gain.is_finite() && gain <= 0.0);
        assert!(
            combined_eq || level_alignment || safety_headroom,
            "unexpected multi-driver channel-level processing: {plugin}"
        );
    }
    assert!(
        plugins
            .iter()
            .filter(|plugin| plugin["parameters"]["label"] == "final_channel_level_alignment")
            .count()
            <= 1,
        "final channel-level alignment must not be applied more than once"
    );
    assert!(
        plugins
            .iter()
            .filter(|plugin| { plugin["parameters"]["label"] == "final_electrical_headroom" })
            .count()
            <= 1,
        "headroom attenuation must have a single channel-level owner"
    );

    let drivers = left_channel["drivers"]
        .as_array()
        .expect("drivers should be an array");
    assert_eq!(drivers.len(), 2, "Expected woofer/tweeter driver branches");

    for driver in drivers {
        let driver_plugins = driver["plugins"]
            .as_array()
            .expect("driver plugins should be an array");
        assert!(
            !driver_plugins.is_empty(),
            "No plugins in multi-driver branch"
        );
        assert!(
            driver_plugins.iter().any(|p| {
                p.get("plugin_type")
                    .and_then(|t| t.as_str())
                    .map(|t| t == "crossover")
                    .unwrap_or(false)
            }),
            "Missing crossover plugin in multi-driver branch"
        );
    }

    // Verify we can parse the optimizer metadata
    let metadata = json["metadata"]
        .as_object()
        .expect("metadata should be an object");
    assert!(
        metadata.contains_key("algorithm"),
        "Missing algorithm in metadata"
    );
    assert!(
        metadata.contains_key("iterations"),
        "Missing iterations in metadata"
    );
}

#[test]
fn test_roomeq_multidriver_known_phase_exports_approved_playback() {
    let directory = tempfile::tempdir().unwrap();
    // Ideal unit-gain drivers have a known impulse response (a delta), hence
    // zero phase. These generated data do not invent phase for measured files.
    let mut csv = String::from("freq,spl,phase\n");
    for index in 0..=200 {
        let frequency = 20.0 * 1000.0_f64.powf(index as f64 / 200.0);
        csv.push_str(&format!("{frequency},80,0\n"));
    }
    fs::write(directory.path().join("woofer.csv"), &csv).unwrap();
    fs::write(directory.path().join("tweeter.csv"), &csv).unwrap();
    let config: serde_json::Value = serde_json::json!({
        "speakers": {"left": {
            "name": "Synthetic ideal two-way",
            "measurements": ["woofer.csv", "tweeter.csv"],
            "crossover": "split"
        }},
        "crossovers": {"split": {"type": "LR24", "frequency": 1000.0}},
        "optimizer": {
            "algorithm": "autoeq:cobyla", "seed": 7, "max_iter": 100,
            "num_filters": 3, "min_freq": 100.0, "max_freq": 10000.0,
            "min_q": 0.5, "max_q": 10.0, "min_db": -12.0, "max_db": 12.0,
            "loss_type": "flat"
        }
    });
    let config_path = directory.path().join("config.json");
    let output_path = directory.path().join("output.json");
    fs::write(&config_path, serde_json::to_vec_pretty(&config).unwrap()).unwrap();
    let output = run_roomeq(&[
        "--config",
        config_path.to_str().unwrap(),
        "--output",
        output_path.to_str().unwrap(),
        "--sample-rate",
        "48000",
    ]);
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let graph: serde_json::Value = serde_json::from_slice(&fs::read(output_path).unwrap()).unwrap();
    // A flat ideal system needs no correction. Approved unchanged playback
    // is distinct from claiming an accepted improvement.
    let acceptance = &graph["metadata"]["correction_acceptance"];
    assert_eq!(acceptance["outcome"], "unchanged", "{acceptance}");
    assert!(
        acceptance
            .get("violations")
            .is_none_or(|value| value == &serde_json::json!([]))
    );
    let drivers = graph["channels"]["left"]["drivers"].as_array().unwrap();
    assert_eq!(drivers.len(), 2);
    for driver in drivers {
        assert!(
            driver["plugins"]
                .as_array()
                .unwrap()
                .iter()
                .any(|plugin| plugin["plugin_type"] == "crossover")
        );
    }
}

#[test]
fn test_roomeq_invalid_config() {
    let temp_dir = tempfile::TempDir::new().expect("Failed to create temp dir");
    let output_path = temp_dir.path().join("output_invalid.json");
    let config_path = temp_dir.path().join("invalid_config.json");

    // Create invalid config
    fs::write(&config_path, r#"{"invalid": "config"}"#).expect("Failed to write invalid config");

    // Run roomeq binary - should fail
    let output = run_roomeq(&[
        "--config",
        config_path.to_str().unwrap(),
        "--output",
        output_path.to_str().unwrap(),
    ]);

    // Should fail
    assert!(
        !output.status.success(),
        "roomeq should fail with invalid config"
    );
    assert!(!output_path.exists());
    assert!(
        String::from_utf8_lossy(&output.stderr)
            .to_ascii_lowercase()
            .contains("speakers")
    );
}

#[test]
fn test_roomeq_missing_measurement() {
    let temp_dir = tempfile::TempDir::new().expect("Failed to create temp dir");
    let output_path = temp_dir.path().join("output_missing.json");
    let config_path = temp_dir.path().join("missing_measurement_config.json");

    // Create config with non-existent measurement
    let config = serde_json::json!({
        "speakers": {
            "left": "nonexistent_file.csv"
        },
        "optimizer": {
            "num_filters": 3,
            "algorithm": "nlopt:cobyla",
            "max_iter": 100,
            "min_freq": 20.0,
            "max_freq": 20000.0,
            "min_q": 0.5,
            "max_q": 10.0,
            "min_db": -12.0,
            "max_db": 12.0,
            "loss_type": "flat"
        }
    });

    fs::write(&config_path, serde_json::to_string(&config).unwrap())
        .expect("Failed to write config");

    // Run roomeq binary - should fail
    let output = run_roomeq(&[
        "--config",
        config_path.to_str().unwrap(),
        "--output",
        output_path.to_str().unwrap(),
    ]);

    // Should fail
    assert!(
        !output.status.success(),
        "roomeq should fail with missing measurement file"
    );
    assert!(!output_path.exists());
    assert!(
        String::from_utf8_lossy(&output.stderr).contains("nonexistent_file.csv"),
        "missing path should be reported: {}",
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn test_roomeq_help() {
    // Test that --help works
    let output = run_roomeq(&["--help"]);

    assert!(output.status.success(), "roomeq --help should succeed");

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("Automatic equalization"),
        "Help text should contain description"
    );
    assert!(
        stdout.contains("--config"),
        "Help text should mention --config"
    );
    assert!(
        stdout.contains("--output"),
        "Help text should mention --output"
    );
}
