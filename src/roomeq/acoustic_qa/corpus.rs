use std::path::{Path, PathBuf};

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

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
                || !(0.0..=1.0).contains(&robustness.coherence_floor))
        {
            return Err(format!(
                "scenario '{}' has an invalid robustness configuration",
                self.id
            ));
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

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct AcousticCorpusBaseline {
    pub version: String,
    pub scenarios: Vec<AcousticCorpusBaselineEntry>,
}

impl AcousticCorpusBaseline {
    pub fn load(path: &Path) -> Result<Self, String> {
        let json = std::fs::read_to_string(path)
            .map_err(|error| format!("failed to read corpus baseline: {error}"))?;
        let baseline: Self = serde_json::from_str(&json)
            .map_err(|error| format!("failed to parse corpus baseline: {error}"))?;
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

    pub fn get(&self, id: &str) -> Option<&QualityBaselineMetrics> {
        self.scenarios
            .iter()
            .find(|scenario| scenario.id == id)
            .map(|scenario| &scenario.metrics)
    }
}

impl AcousticCorpusManifest {
    pub fn load(path: &Path) -> Result<Self, String> {
        let json = std::fs::read_to_string(path)
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

    #[test]
    fn qa_tiers_include_lower_cost_scenarios() {
        assert!(QaTier::Nightly.includes(QaTier::Pr));
        assert!(QaTier::Release.includes(QaTier::Weekly));
        assert!(!QaTier::Pr.includes(QaTier::Nightly));
    }

    #[test]
    fn repository_manifest_has_real_and_held_out_pr_evidence() {
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("data_tests/roomeq/acoustic_corpus/manifest.json");
        let manifest = AcousticCorpusManifest::load(&path).expect("repository corpus manifest");
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
                .is_some_and(|path| path.ends_with("qa_optimizer_headroom_smooth.json"))
        );
        assert!(promoted_mso.candidate_override_config.is_none());
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
        assert!(
            manifest
                .scenarios
                .iter()
                .all(|scenario| baseline.get(&scenario.id).is_some())
        );
    }
}
