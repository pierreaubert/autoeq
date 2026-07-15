use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// Ordered stages in the canonical RoomEQ configuration validation pipeline.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum ValidationStage {
    SchemaVersion,
    Structural,
    ResolvedResource,
    Acoustic,
    ExportTarget,
}

impl ValidationStage {
    pub const ALL: [Self; 5] = [
        Self::SchemaVersion,
        Self::Structural,
        Self::ResolvedResource,
        Self::Acoustic,
        Self::ExportTarget,
    ];
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum ValidationStageStatus {
    NotRun,
    NotApplicable,
    Passed,
    Failed,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct ValidationStageReport {
    pub stage: ValidationStage,
    pub status: ValidationStageStatus,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub errors: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub warnings: Vec<String>,
}

/// Serializable evidence describing exactly which validation strengths ran.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct ConfigValidationReport {
    pub pipeline_version: String,
    pub stages: Vec<ValidationStageReport>,
}

impl Default for ConfigValidationReport {
    fn default() -> Self {
        Self::new()
    }
}

impl ConfigValidationReport {
    pub const PIPELINE_VERSION: &'static str = "1.0.0";

    pub fn new() -> Self {
        Self {
            pipeline_version: Self::PIPELINE_VERSION.to_string(),
            stages: ValidationStage::ALL
                .into_iter()
                .map(|stage| ValidationStageReport {
                    stage,
                    status: ValidationStageStatus::NotRun,
                    errors: Vec::new(),
                    warnings: Vec::new(),
                })
                .collect(),
        }
    }

    pub fn record(&mut self, stage: ValidationStage, errors: Vec<String>, warnings: Vec<String>) {
        let entry = self.stage_mut(stage);
        entry.status = if errors.is_empty() {
            ValidationStageStatus::Passed
        } else {
            ValidationStageStatus::Failed
        };
        entry.errors = errors;
        entry.warnings = warnings;
    }

    pub fn mark_not_applicable(&mut self, stage: ValidationStage) {
        let entry = self.stage_mut(stage);
        entry.status = ValidationStageStatus::NotApplicable;
        entry.errors.clear();
        entry.warnings.clear();
    }

    pub fn stage(&self, stage: ValidationStage) -> &ValidationStageReport {
        self.stages
            .iter()
            .find(|entry| entry.stage == stage)
            .expect("canonical validation report contains every stage")
    }

    pub fn stage_ran(&self, stage: ValidationStage) -> bool {
        matches!(
            self.stage(stage).status,
            ValidationStageStatus::Passed | ValidationStageStatus::Failed
        )
    }

    pub fn is_valid(&self) -> bool {
        self.stages
            .iter()
            .all(|stage| stage.status != ValidationStageStatus::Failed)
    }

    /// True only after every production stage either passed or was explicitly
    /// declared inapplicable. A structural-only report can never return true.
    pub fn production_ready(&self) -> bool {
        self.stages.iter().all(|stage| {
            matches!(
                stage.status,
                ValidationStageStatus::Passed | ValidationStageStatus::NotApplicable
            )
        })
    }

    pub fn errors(&self) -> impl Iterator<Item = &String> {
        self.stages.iter().flat_map(|stage| stage.errors.iter())
    }

    pub fn warnings(&self) -> impl Iterator<Item = &String> {
        self.stages.iter().flat_map(|stage| stage.warnings.iter())
    }

    fn stage_mut(&mut self, stage: ValidationStage) -> &mut ValidationStageReport {
        self.stages
            .iter_mut()
            .find(|entry| entry.stage == stage)
            .expect("canonical validation report contains every stage")
    }
}
