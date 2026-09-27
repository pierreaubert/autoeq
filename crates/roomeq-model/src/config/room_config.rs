use super::ctc_config::CtcConfig;
use super::default::{default_config_version, validate_config_version};
use super::measured_ir_config::MeasuredIrSource;
use super::optimizer_config::OptimizerConfig;
use super::provenance_config::ProvenanceConfig;
use super::reporting_config::ReportingConfig;
use super::speaker_config::SpeakerConfig;
use super::types::CrossoverConfig;
use super::types::RecordingConfiguration;
use super::types::SystemConfig;
use super::types::TargetCurveConfig;
use super::{ConfigValidationReport, ValidationStage};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap};

/// Complete room configuration
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[schemars(transform = allow_legacy_description)]
pub struct RoomConfig {
    /// Configuration schema version. Version 3.0.x intentionally rejects the
    /// legacy stereo-LFE representation.
    #[serde(default = "default_config_version")]
    pub version: String,
    /// System configuration (v3) separating logical inputs from physical outputs.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub system: Option<SystemConfig>,
    /// Map of channel name to speaker configuration
    pub speakers: HashMap<String, SpeakerConfig>,
    /// Optional crossover configuration for multi-driver groups
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub crossovers: Option<HashMap<String, CrossoverConfig>>,
    /// Optional target curve (freq, spl)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub target_curve: Option<TargetCurveConfig>,
    /// Optimizer configuration
    #[serde(default)]
    pub optimizer: OptimizerConfig,
    /// Versioned measurement-sidecar references and validation policy.
    #[serde(default, skip_serializing_if = "is_default_provenance")]
    pub provenance: ProvenanceConfig,
    /// Recording configuration (device settings, signal parameters used during capture)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub recording_config: Option<RecordingConfiguration>,
    /// Measured room impulse responses per channel, backing the R1–R5
    /// acoustic report fields. Keys are output channel names; values are
    /// `time_ms,amplitude` CSVs captured in the room. Channels without an
    /// entry keep synthesized IRs and their report cells stay pending.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub measured_impulse_responses: BTreeMap<String, MeasuredIrSource>,
    /// Cross-talk cancellation / binaural-aware correction configuration.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ctc: Option<CtcConfig>,
    /// Report-policy inputs needing an explicit operator declaration (for
    /// example the T60 flatness tolerance). Absent policies leave the viewer
    /// to its documented standard defaults; they change no acceptance math.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reporting: Option<ReportingConfig>,
    /// Pre-fetched CEA2034 data (runtime only, not serialized).
    #[serde(skip)]
    #[schemars(skip)]
    pub cea2034_cache: Option<HashMap<String, crate::SpinoramaBundle>>,
}

/// Older configuration fixtures carry a descriptive label that Serde accepts
/// and ignores. Keep the generated input schema in sync with that accepted
/// form without opening the schema to arbitrary unknown configuration keys.
fn allow_legacy_description(schema: &mut schemars::Schema) {
    schema
        .ensure_object()
        .entry("properties")
        .or_insert_with(|| serde_json::json!({}))
        .as_object_mut()
        .expect("RoomConfig schema properties are an object")
        .insert(
            "description".into(),
            serde_json::json!({ "type": "string" }),
        );
}

impl Default for RoomConfig {
    fn default() -> Self {
        Self {
            version: default_config_version(),
            system: None,
            speakers: HashMap::new(),
            crossovers: None,
            target_curve: None,
            optimizer: OptimizerConfig::default(),
            provenance: ProvenanceConfig::default(),
            recording_config: None,
            measured_impulse_responses: BTreeMap::new(),
            ctc: None,
            reporting: None,
            cea2034_cache: None,
        }
    }
}

impl RoomConfig {
    /// Declared T60 flatness tolerance for report summary cells, if the
    /// operator supplied a `reporting` policy. Structural validation rejects
    /// nonfinite/nonpositive tolerances before any run, so topology plumbing
    /// carries this value without re-checking it.
    #[must_use]
    pub fn report_t60_tolerance_s(&self) -> Option<f64> {
        self.reporting
            .as_ref()
            .and_then(|reporting| reporting.t60_flatness_tolerance_s)
    }

    /// Validate the serialized configuration schema version.
    pub fn validate_version(&self) -> Result<(), String> {
        validate_config_version(&self.version)
    }

    /// Validate invariants required before any RoomEQ engine run.
    pub fn validate_structure(&self) -> Result<(), String> {
        let report = self.validation_report();
        report.errors().next().cloned().map_or(Ok(()), Err)
    }

    /// Run the engine-neutral portion of the canonical validation pipeline.
    ///
    /// This intentionally runs only schema/version and structural validation.
    /// Callers must inspect [`ConfigValidationReport::production_ready`] rather
    /// than treating this report as resolved-resource/acoustic/export evidence.
    pub fn validation_report(&self) -> ConfigValidationReport {
        let mut report = ConfigValidationReport::new();
        let version_errors = self.validate_version().err().into_iter().collect();
        report.record(ValidationStage::SchemaVersion, version_errors, Vec::new());
        report.record(
            ValidationStage::Structural,
            self.structural_errors(),
            Vec::new(),
        );
        report
    }

    fn structural_errors(&self) -> Vec<String> {
        let mut errors = Vec::new();
        if !self.optimizer.max_crossover_cancellation_db.is_finite()
            || self.optimizer.max_crossover_cancellation_db < 0.0
        {
            errors.push(
                "optimizer.max_crossover_cancellation_db must be finite and nonnegative".into(),
            );
        }
        if let Some(reporting) = &self.reporting
            && let Err(error) = reporting.validate()
        {
            errors.push(error);
        }
        if let Err(error) = self.optimizer.finalization.validate() {
            errors.push(error);
        }
        if self
            .optimizer
            .fir
            .as_ref()
            .is_some_and(|fir| fir.placement == super::FirPlacement::PerDriver)
        {
            if !self.version.starts_with("3.0.") {
                errors.push("fir.placement=per_driver requires config version 3.0.x".into());
            }
            if !matches!(
                self.optimizer.processing_mode,
                super::ProcessingMode::PhaseLinear
                    | super::ProcessingMode::Hybrid
                    | super::ProcessingMode::MixedPhase
            ) {
                errors.push("fir.placement=per_driver requires phase_linear, hybrid or mixed_phase processing".into());
            }
            // Shared routed outputs use one common kernel deployed per physical
            // branch, not independent inversions of logical-source objectives.
            if self.speakers.values().any(|s| {
                !matches!(
                    s,
                    SpeakerConfig::Group(_)
                        | SpeakerConfig::Topology(_)
                        | SpeakerConfig::Single(_)
                        | SpeakerConfig::MultiSub(_)
                )
            }) {
                errors.push(
                    "fir.placement=per_driver does not support arrays or supporting-source outputs"
                        .into(),
                );
            }
        }
        if self.speakers.is_empty() {
            errors.push("room configuration requires at least one speaker".to_string());
        }
        for (name, speaker) in &self.speakers {
            if name.trim().is_empty() {
                errors.push("speaker names must not be empty".to_string());
            }
            if let SpeakerConfig::Topology(topology) = speaker
                && let Err(error) = topology.validate()
            {
                errors.push(format!(
                    "speaker group '{name}' has invalid topology: {error}"
                ));
            }
        }
        if let Some(system) = &self.system {
            for (role, measurement_key) in &system.speakers {
                if !self.speakers.contains_key(measurement_key) {
                    errors.push(format!(
                        "system.speakers role '{role}' references missing speaker measurement '{measurement_key}'"
                    ));
                }
            }
            match system.model {
                crate::SystemModel::Stereo => {
                    let has_exact_lr = system.speakers.len() == 2
                        && system.speakers.contains_key("L")
                        && system.speakers.contains_key("R");
                    if !has_exact_lr {
                        errors.push(
                            "stereo system.speakers must contain exactly logical inputs 'L' and 'R'; system.speakers.LFE is invalid in v3. Replace it with system.subwoofers.outputs: [{\"id\":\"Sub1\",\"speaker\":\"sub\"}]"
                                .to_string(),
                        );
                    }
                }
                crate::SystemModel::HomeCinema => {
                    if system
                        .speakers
                        .keys()
                        .any(|role| role.eq_ignore_ascii_case("LFE"))
                    {
                        errors.push(
                            "home_cinema system.speakers must not map an LFE measurement in v3; the logical LFE programme input is implicit and physical measurements belong in system.subwoofers.outputs"
                                .to_string(),
                        );
                    }
                }
                crate::SystemModel::Custom => {}
            }
            if let Some(subwoofers) = system.subwoofers.as_ref() {
                let expected = match system.model {
                    crate::SystemModel::Stereo => 1..=2,
                    _ => 1..=usize::MAX,
                };
                if !expected.contains(&subwoofers.outputs.len()) {
                    errors.push(format!(
                        "system.subwoofers.outputs has {} entries; {} requires {}",
                        subwoofers.outputs.len(),
                        match system.model {
                            crate::SystemModel::Stereo => "stereo",
                            crate::SystemModel::HomeCinema => "home_cinema",
                            crate::SystemModel::Custom => "custom",
                        },
                        if matches!(system.model, crate::SystemModel::Stereo) {
                            "one or two physical sub outputs"
                        } else {
                            "at least one physical sub output"
                        }
                    ));
                }
                let mut output_ids = std::collections::HashSet::new();
                for output in &subwoofers.outputs {
                    if output.id.trim().is_empty() || !output_ids.insert(output.id.as_str()) {
                        errors.push(format!(
                            "system.subwoofers.outputs contains an empty or duplicate output id '{}'",
                            output.id
                        ));
                    }
                    if !self.speakers.contains_key(&output.speaker) {
                        errors.push(format!(
                            "physical sub output '{}' references missing speaker measurement '{}'",
                            output.id, output.speaker
                        ));
                    }
                }
                if let Some(crossover) = subwoofers.crossover.as_ref()
                    && crossover.as_list().len() != subwoofers.outputs.len()
                {
                    errors.push(format!(
                        "system.subwoofers.crossover has {} entries but system.subwoofers.outputs has {}; v3 requires one crossover per physical output",
                        crossover.as_list().len(),
                        subwoofers.outputs.len()
                    ));
                }
            }
        }
        if let Some(crossovers) = &self.crossovers {
            for (name, crossover) in crossovers {
                if crossover
                    .crossover_type
                    .parse::<crate::CrossoverType>()
                    .is_err()
                {
                    errors.push(format!(
                        "Crossover '{name}' has unsupported type '{}'",
                        crossover.crossover_type
                    ));
                }
            }
        }
        errors.extend(self.optimizer.gain_envelope_errors());
        errors.extend(self.provenance.structural_errors(self.speakers.keys()));
        if !self.optimizer.min_freq.is_finite()
            || !self.optimizer.max_freq.is_finite()
            || self.optimizer.min_freq <= 0.0
            || self.optimizer.min_freq >= self.optimizer.max_freq
        {
            errors.push(format!(
                "optimizer frequency range must be finite, positive, and increasing; got [{}, {}] Hz",
                self.optimizer.min_freq, self.optimizer.max_freq
            ));
        }
        errors
    }

    /// Resolve relative paths in this room configuration against a base directory
    pub fn resolve_paths(&mut self, base_dir: &std::path::Path) {
        let base_dir = if base_dir.is_absolute() {
            base_dir.to_path_buf()
        } else {
            std::env::current_dir()
                .map(|current_dir| current_dir.join(base_dir))
                .unwrap_or_else(|_| base_dir.to_path_buf())
        };
        for speaker in self.speakers.values_mut() {
            speaker.resolve_paths(&base_dir);
        }
        if let Some(TargetCurveConfig::Path(ref mut path)) = self.target_curve
            && path.is_relative()
        {
            *path = base_dir.join(&*path);
        }
        if let Some(target_response) = self.optimizer.target_response.as_mut()
            && let Some(path) = target_response.curve_path.as_mut()
            && path.is_relative()
        {
            *path = base_dir.join(&*path);
        }
        if let Some(ctc) = &mut self.ctc {
            ctc.resolve_paths(&base_dir);
        }
        self.provenance.resolve_paths(&base_dir);
        for source in self.measured_impulse_responses.values_mut() {
            if source.path.is_relative() {
                source.path = base_dir.join(&source.path);
            }
        }
    }
}

fn is_default_provenance(provenance: &ProvenanceConfig) -> bool {
    provenance.measurements.is_empty()
        && provenance.validation_mode == super::ProvenanceValidationMode::Warn
}
