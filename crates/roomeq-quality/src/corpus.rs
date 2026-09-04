use std::path::{Path, PathBuf};

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

// Repository-backed acoustic corpus support.
use super::{QaTier, QualityBaselineMetrics};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum CorpusProvenance {
    RealMeasurement,
    Fem,
    Synthetic,
}

impl CorpusProvenance {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::RealMeasurement => "real_measurement",
            Self::Fem => "fem",
            Self::Synthetic => "synthetic",
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum QualityGateMode {
    #[default]
    ReportOnly,
    Enforce,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct HeldOutMeasurement {
    pub channel: String,
    pub path: PathBuf,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct CorpusRobustnessConfig {
    pub seeds: Vec<u64>,
    pub noise_peak_db: f64,
    pub coherence_floor: f64,
    /// Optional deterministic per-curve SPL calibration offset (± dB).
    ///
    /// Models the level mismatch between a training capture and a later
    /// verification capture. Each curve gets `± level_calibration_error_db`
    /// with a seed-derived sign.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub level_calibration_error_db: Option<f64>,
    /// Optional fraction of seats dropped per robustness seed (`0..1`).
    ///
    /// Models a missing listening position: the rescoring subset omits
    /// `floor(curve_count * seat_dropout_fraction)` curves at a seed-derived
    /// offset, keeping at least one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub seat_dropout_fraction: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct AcousticCorpusScenario {
    pub id: String,
    pub tier: QaTier,
    pub provenance: CorpusProvenance,
    pub topology: String,
    /// Routed result channels included in the scorecard. Empty means all.
    #[serde(default)]
    pub channels: Vec<String>,
    pub sample_rate: f64,
    pub config: PathBuf,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub override_config: Option<PathBuf>,
    /// Optional matched algorithm/objective candidate. The corpus evaluator
    /// runs this against the same measurements, seed, band, and held-out data.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub candidate_override_config: Option<PathBuf>,
    pub evaluation_band_hz: [f64; 2],
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub schroeder_hz: Option<f64>,
    #[serde(default)]
    pub held_out: Vec<HeldOutMeasurement>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub robustness: Option<CorpusRobustnessConfig>,
    #[serde(default)]
    pub gate_mode: QualityGateMode,
}

impl AcousticCorpusScenario {
    pub fn validate(&self) -> Result<(), String> {
        if self.id.trim().is_empty() {
            return Err("acoustic corpus scenario id must not be empty".to_string());
        }
        if !self.sample_rate.is_finite() || self.sample_rate <= 0.0 {
            return Err(format!("scenario '{}' has invalid sample rate", self.id));
        }
        let [low, high] = self.evaluation_band_hz;
        if !low.is_finite() || !high.is_finite() || low <= 0.0 || high <= low {
            return Err(format!(
                "scenario '{}' has an invalid evaluation band",
                self.id
            ));
        }
        if self
            .schroeder_hz
            .is_some_and(|value| !value.is_finite() || value <= 0.0)
        {
            return Err(format!(
                "scenario '{}' has an invalid Schroeder frequency",
                self.id
            ));
        }
        if !self.config.is_file() {
            return Err(format!(
                "scenario '{}' config does not exist: {}",
                self.id,
                self.config.display()
            ));
        }
        if let Some(path) = &self.override_config
            && !path.is_file()
        {
            return Err(format!(
                "scenario '{}' override does not exist: {}",
                self.id,
                path.display()
            ));
        }
        if let Some(path) = &self.candidate_override_config
            && !path.is_file()
        {
            return Err(format!(
                "scenario '{}' candidate override does not exist: {}",
                self.id,
                path.display()
            ));
        }
        for measurement in &self.held_out {
            if measurement.channel.trim().is_empty() || !measurement.path.is_file() {
                return Err(format!(
                    "scenario '{}' has invalid held-out measurement '{}': {}",
                    self.id,
                    measurement.channel,
                    measurement.path.display()
                ));
            }
            if !self.channels.is_empty() && !self.channels.contains(&measurement.channel) {
                return Err(format!(
                    "scenario '{}' held-out channel '{}' is not selected for scoring",
                    self.id, measurement.channel
                ));
            }
        }
        if let Some(robustness) = &self.robustness
            && (robustness.seeds.is_empty()
                || !robustness.noise_peak_db.is_finite()
                || robustness.noise_peak_db < 0.0
                || !robustness.coherence_floor.is_finite()
                || !(0.0..=1.0).contains(&robustness.coherence_floor)
                || robustness
                    .level_calibration_error_db
                    .is_some_and(|value| !value.is_finite() || value < 0.0)
                || robustness
                    .seat_dropout_fraction
                    .is_some_and(|value| !value.is_finite() || !(0.0..1.0).contains(&value)))
        {
            return Err(format!(
                "scenario '{}' has an invalid robustness configuration",
                self.id
            ));
        }
        // Intake rule: an enforced gate is a generalization claim, so it needs
        // at least two held-out measurements covering every scored channel. A
        // single-position capture stays report-only until a second seat exists.
        if self.gate_mode == QualityGateMode::Enforce {
            if self.held_out.len() < 2 {
                return Err(format!(
                    "scenario '{}' is enforced but provides fewer than two held-out measurements; add a second seat or downgrade to report-only",
                    self.id
                ));
            }
            if !self.channels.is_empty() {
                for channel in &self.channels {
                    if !self
                        .held_out
                        .iter()
                        .any(|measurement| &measurement.channel == channel)
                    {
                        return Err(format!(
                            "scenario '{}' is enforced but scored channel '{channel}' has no held-out measurement",
                            self.id
                        ));
                    }
                }
            }
        }
        Ok(())
    }

    fn resolve_paths(&mut self, base: &Path) {
        self.config = resolve(base, &self.config);
        self.override_config = self
            .override_config
            .as_ref()
            .map(|path| resolve(base, path));
        self.candidate_override_config = self
            .candidate_override_config
            .as_ref()
            .map(|path| resolve(base, path));
        for measurement in &mut self.held_out {
            measurement.path = resolve(base, &measurement.path);
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct AcousticCorpusManifest {
    pub version: String,
    pub scenarios: Vec<AcousticCorpusScenario>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct AcousticCorpusBaselineEntry {
    pub id: String,
    #[serde(flatten)]
    pub metrics: QualityBaselineMetrics,
}

/// Execution platform for which an optimizer-derived acoustic baseline was
/// calibrated. Fixed seeds make runs repeatable on one platform, but floating
/// point and parallel evaluation order can still lead iterative optimizers to
/// a different solution on another OS or architecture.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct AcousticBaselinePlatform {
    pub os: String,
    pub arch: String,
}

impl AcousticBaselinePlatform {
    pub fn current() -> Self {
        Self {
            os: std::env::consts::OS.to_string(),
            arch: std::env::consts::ARCH.to_string(),
        }
    }

    pub fn label(&self) -> String {
        format!("{}-{}", self.os, self.arch)
    }

    fn validate(&self) -> Result<(), String> {
        if self.os.trim().is_empty() || self.arch.trim().is_empty() {
            return Err(
                "acoustic corpus baseline platform needs non-empty os and arch".to_string(),
            );
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct AcousticCorpusBaseline {
    pub version: String,
    pub platform: AcousticBaselinePlatform,
    /// Informational compiler provenance. Platform compatibility deliberately
    /// remains keyed to OS and architecture; compiler changes are still
    /// visible in artifacts and remain subject to the numeric quality gates.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub generated_with_rustc: Option<String>,
    pub scenarios: Vec<AcousticCorpusBaselineEntry>,
}

impl AcousticCorpusBaseline {
    pub fn load(path: &Path) -> Result<Self, String> {
        let json = std::fs::read_to_string(path)
            .map_err(|error| format!("failed to read corpus baseline: {error}"))?;
        let baseline: Self = serde_json::from_str(&json)
            .map_err(|error| format!("failed to parse corpus baseline: {error}"))?;
        baseline.platform.validate()?;
        if baseline
            .generated_with_rustc
            .as_ref()
            .is_some_and(|version| version.trim().is_empty())
        {
            return Err(
                "acoustic corpus baseline generated_with_rustc must not be empty".to_string(),
            );
        }
        let mut ids = std::collections::HashSet::new();
        if let Some(duplicate) = baseline
            .scenarios
            .iter()
            .map(|scenario| scenario.id.as_str())
            .find(|id| !ids.insert(*id))
        {
            return Err(format!(
                "duplicate acoustic corpus baseline id '{duplicate}'"
            ));
        }
        Ok(baseline)
    }

    pub fn validate_for_platform(&self, expected: &AcousticBaselinePlatform) -> Result<(), String> {
        expected.validate()?;
        if self.platform != *expected {
            return Err(format!(
                "acoustic corpus baseline is calibrated for {}, but this run is {}; select or recalibrate the matching platform baseline",
                self.platform.label(),
                expected.label()
            ));
        }
        Ok(())
    }

    pub fn get(&self, id: &str) -> Option<&QualityBaselineMetrics> {
        self.scenarios
            .iter()
            .find(|scenario| scenario.id == id)
            .map(|scenario| &scenario.metrics)
    }
}

impl AcousticCorpusManifest {
    pub fn load(path: &Path) -> Result<Self, String> {
        let path = path.canonicalize().map_err(|error| {
            format!(
                "failed to resolve corpus manifest '{}': {error}",
                path.display()
            )
        })?;
        let json = std::fs::read_to_string(&path)
            .map_err(|error| format!("failed to read corpus manifest: {error}"))?;
        let mut manifest: Self = serde_json::from_str(&json)
            .map_err(|error| format!("failed to parse corpus manifest: {error}"))?;
        let base = path.parent().unwrap_or_else(|| Path::new("."));
        for scenario in &mut manifest.scenarios {
            scenario.resolve_paths(base);
            scenario.validate()?;
        }
        if manifest.scenarios.is_empty() {
            return Err("acoustic corpus manifest must contain scenarios".to_string());
        }
        let mut ids = std::collections::HashSet::new();
        if let Some(duplicate) = manifest
            .scenarios
            .iter()
            .map(|scenario| scenario.id.as_str())
            .find(|id| !ids.insert(*id))
        {
            return Err(format!(
                "duplicate acoustic corpus scenario id '{duplicate}'"
            ));
        }
        Ok(manifest)
    }

    pub fn scenarios_for(&self, tier: QaTier) -> impl Iterator<Item = &AcousticCorpusScenario> {
        self.scenarios
            .iter()
            .filter(move |scenario| tier.includes(scenario.tier))
    }
}

fn resolve(base: &Path, path: &Path) -> PathBuf {
    if path.is_absolute() {
        path.to_path_buf()
    } else {
        base.join(path)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn workspace_root() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
    }

    #[test]
    fn acoustic_baseline_rejects_a_different_execution_platform() {
        let expected = AcousticBaselinePlatform {
            os: "linux".to_string(),
            arch: "x86_64".to_string(),
        };
        let baseline = AcousticCorpusBaseline {
            version: "2.0.0".to_string(),
            platform: AcousticBaselinePlatform {
                os: "macos".to_string(),
                arch: "aarch64".to_string(),
            },
            generated_with_rustc: Some("rustc test".to_string()),
            scenarios: Vec::new(),
        };

        let error = baseline
            .validate_for_platform(&expected)
            .expect_err("a platform-specific baseline must not cross platforms");
        assert!(error.contains("macos-aarch64"));
        assert!(error.contains("linux-x86_64"));
    }

    #[test]
    fn acoustic_baseline_accepts_its_declared_execution_platform() {
        let platform = AcousticBaselinePlatform {
            os: "linux".to_string(),
            arch: "x86_64".to_string(),
        };
        let baseline = AcousticCorpusBaseline {
            version: "2.0.0".to_string(),
            platform: platform.clone(),
            generated_with_rustc: None,
            scenarios: Vec::new(),
        };

        baseline
            .validate_for_platform(&platform)
            .expect("matching platform must be accepted");
    }

    #[test]
    fn repository_baselines_declare_their_execution_platform() {
        let root = workspace_root().join("data_tests/roomeq/acoustic_corpus");
        for (file, os, arch) in [
            ("baseline.json", "macos", "aarch64"),
            ("baseline.linux-x86_64.json", "linux", "x86_64"),
        ] {
            let expected = AcousticBaselinePlatform {
                os: os.to_string(),
                arch: arch.to_string(),
            };
            let baseline = AcousticCorpusBaseline::load(&root.join(file))
                .unwrap_or_else(|error| panic!("failed to load {file}: {error}"));

            baseline
                .validate_for_platform(&expected)
                .unwrap_or_else(|error| panic!("invalid platform in {file}: {error}"));
            assert_eq!(baseline.scenarios.len(), 8, "unexpected corpus in {file}");
            assert!(
                baseline.generated_with_rustc.is_some(),
                "missing compiler provenance in {file}"
            );
        }
    }

    #[test]
    fn repository_held_out_curves_are_not_training_measurements() {
        let path = workspace_root().join("data_tests/roomeq/acoustic_corpus/manifest.json");
        let manifest =
            AcousticCorpusManifest::load(&path).expect("repository acoustic manifest should load");
        for scenario in &manifest.scenarios {
            let config = std::fs::read_to_string(&scenario.config).unwrap_or_else(|error| {
                panic!("failed to read {}: {error}", scenario.config.display())
            });
            for held_out in &scenario.held_out {
                let file_name = held_out
                    .path
                    .file_name()
                    .and_then(|name| name.to_str())
                    .expect("held-out path should have a UTF-8 file name");
                assert!(
                    !config.contains(file_name),
                    "scenario '{}' reuses training measurement '{}' as held-out data",
                    scenario.id,
                    held_out.path.display()
                );
            }
        }
    }

    fn valid_scenario() -> AcousticCorpusScenario {
        AcousticCorpusScenario {
            id: "validation-contract".to_string(),
            tier: QaTier::Pr,
            provenance: CorpusProvenance::Synthetic,
            topology: "stereo_2_0".to_string(),
            channels: Vec::new(),
            sample_rate: 48_000.0,
            config: Path::new(env!("CARGO_MANIFEST_DIR")).join("Cargo.toml"),
            override_config: None,
            candidate_override_config: None,
            evaluation_band_hz: [20.0, 20_000.0],
            schroeder_hz: None,
            held_out: Vec::new(),
            robustness: None,
            gate_mode: QualityGateMode::ReportOnly,
        }
    }

    #[test]
    fn qa_tiers_include_lower_cost_scenarios() {
        assert!(QaTier::Nightly.includes(QaTier::Pr));
        assert!(QaTier::Release.includes(QaTier::Weekly));
        assert!(!QaTier::Pr.includes(QaTier::Nightly));
    }

    #[test]
    fn repository_manifest_has_real_and_held_out_pr_evidence() {
        let path = workspace_root().join("data_tests/roomeq/acoustic_corpus/manifest.json");
        let manifest = AcousticCorpusManifest::load(&path).expect("repository corpus manifest");
        assert!(
            manifest
                .scenarios
                .iter()
                .all(|scenario| scenario.config.is_absolute()),
            "corpus paths must be fully resolved before RoomEQ validation"
        );
        let pr: Vec<_> = manifest.scenarios_for(QaTier::Pr).collect();
        assert!(
            pr.iter()
                .any(|scenario| scenario.provenance == CorpusProvenance::RealMeasurement)
        );
        assert!(pr.iter().any(|scenario| !scenario.held_out.is_empty()));
        assert!(
            pr.iter()
                .all(|scenario| scenario.gate_mode == QualityGateMode::Enforce)
        );
        assert!(pr.iter().any(|scenario| scenario.topology == "stereo_2_1"));
        assert!(
            manifest
                .scenarios_for(QaTier::Nightly)
                .any(|scenario| scenario.topology == "stereo_2_2_mso")
        );
        let promoted_mso = manifest
            .scenarios
            .iter()
            .find(|scenario| scenario.id == "fem_small_stereo_22_mso")
            .expect("promoted MSO scenario");
        assert!(
            promoted_mso
                .override_config
                .as_ref()
                .is_some_and(|path| path.ends_with("qa_optimizer.json"))
        );
        assert!(
            promoted_mso
                .candidate_override_config
                .as_ref()
                .is_some_and(|path| path.ends_with("qa_optimizer_headroom_smooth.json"))
        );
        assert!(
            manifest
                .scenarios
                .iter()
                .find(|scenario| scenario.id == "measured_stereo_t7v")
                .and_then(|scenario| scenario.candidate_override_config.as_ref())
                .is_some_and(|path| path.ends_with("qa_optimizer_alternate_de.json"))
        );
        assert!(
            manifest
                .scenarios_for(QaTier::Nightly)
                .any(|scenario| scenario.topology == "home_cinema_5_1")
        );
        let baseline_path = path.with_file_name("baseline.json");
        let baseline = AcousticCorpusBaseline::load(&baseline_path).expect("repository baseline");
        assert_eq!(manifest.version, baseline.version);
        // Enforced scenarios pin a baseline; report-only scenarios (e.g. a
        // single-position measured capture awaiting a second seat) are covered
        // by the missing-baseline advisory instead.
        assert!(
            manifest
                .scenarios
                .iter()
                .filter(|scenario| scenario.gate_mode == QualityGateMode::Enforce)
                .all(|scenario| baseline.get(&scenario.id).is_some())
        );
    }

    #[test]
    fn enforced_scenarios_need_two_held_out_measurements_per_scored_channel() {
        let mut scenario = valid_scenario();
        scenario.gate_mode = QualityGateMode::Enforce;
        let error = scenario
            .validate()
            .expect_err("enforced scenario without held-out data must be rejected");
        assert!(error.contains("fewer than two held-out measurements"));

        scenario.held_out = vec![
            HeldOutMeasurement {
                channel: "L".to_string(),
                path: scenario.config.clone(),
            },
            HeldOutMeasurement {
                channel: "L".to_string(),
                path: scenario.config.clone(),
            },
        ];
        scenario.channels = vec!["L".to_string(), "R".to_string()];
        let error = scenario
            .validate()
            .expect_err("enforced scenario must cover every scored channel with held-out data");
        assert!(error.contains("'R' has no held-out measurement"));

        scenario.channels = vec!["L".to_string()];
        scenario
            .validate()
            .expect("covered held-out channels must validate");
    }

    #[test]
    fn report_only_scenarios_may_wait_for_a_second_seat() {
        let scenario = valid_scenario();
        assert_eq!(scenario.gate_mode, QualityGateMode::ReportOnly);
        scenario.validate().expect(
            "single-position measured captures stay report-only until a second seat exists",
        );
    }

    #[test]
    fn robustness_rejects_negative_level_error_and_full_dropout() {
        let mut scenario = valid_scenario();
        scenario.robustness = Some(CorpusRobustnessConfig {
            seeds: vec![1],
            noise_peak_db: 0.2,
            coherence_floor: 0.8,
            level_calibration_error_db: Some(-1.0),
            seat_dropout_fraction: None,
        });
        assert!(scenario.validate().is_err());

        scenario.robustness = Some(CorpusRobustnessConfig {
            seeds: vec![1],
            noise_peak_db: 0.2,
            coherence_floor: 0.8,
            level_calibration_error_db: Some(1.0),
            seat_dropout_fraction: Some(1.0),
        });
        assert!(scenario.validate().is_err());

        scenario.robustness = Some(CorpusRobustnessConfig {
            seeds: vec![1],
            noise_peak_db: 0.2,
            coherence_floor: 0.8,
            level_calibration_error_db: Some(1.0),
            seat_dropout_fraction: Some(0.5),
        });
        scenario
            .validate()
            .expect("bounded degradation knobs must validate");
    }

    #[test]
    fn repository_manifest_resolves_paths_from_a_relative_manifest_path() {
        let path = Path::new("../../data_tests/roomeq/acoustic_corpus/manifest.json");
        let manifest = AcousticCorpusManifest::load(path).expect("relative repository manifest");

        assert!(
            manifest.scenarios.iter().all(|scenario| {
                scenario.config.is_absolute()
                    && scenario
                        .override_config
                        .as_ref()
                        .is_none_or(|path| path.is_absolute())
                    && scenario
                        .candidate_override_config
                        .as_ref()
                        .is_none_or(|path| path.is_absolute())
                    && scenario
                        .held_out
                        .iter()
                        .all(|measurement| measurement.path.is_absolute())
            }),
            "all corpus paths must be absolute even when the manifest argument is relative"
        );
    }

    #[test]
    fn missing_candidate_override_is_rejected() {
        let mut scenario = valid_scenario();
        scenario.candidate_override_config =
            Some(Path::new(env!("CARGO_MANIFEST_DIR")).join("definitely-missing-candidate.json"));

        assert!(scenario.validate().is_err());
    }
}
