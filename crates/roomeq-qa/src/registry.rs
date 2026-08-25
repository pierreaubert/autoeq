//! Declarative RoomEQ QA scenario inventory.

use anyhow::{Result, bail};
use clap::ValueEnum;
use roomeq_engine::room_result::RoomOptimizationResult;
use roomeq_model::{MeasurementSource, RoomConfig, SpeakerConfig, SubwooferStrategy, SystemModel};
use serde::Deserialize;
use std::collections::HashSet;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, ValueEnum)]
#[serde(rename_all = "snake_case")]
pub enum QaTier {
    Pr,
    Nightly,
    Weekly,
}

/// What a QA case is allowed to prove.
///
/// Safety smoke cases may accept a runtime revert. Functional and quality
/// cases must retain the requested correction; otherwise a safe fallback can
/// conceal a missing or incorrectly realized feature.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum QaGatePurpose {
    Safety,
    Functional,
    #[default]
    Quality,
}

impl QaTier {
    pub fn includes(self, candidate: Self) -> bool {
        let rank = |tier| match tier {
            Self::Pr => 0,
            Self::Nightly => 1,
            Self::Weekly => 2,
        };
        rank(candidate) <= rank(self)
    }
}

#[derive(Debug, Clone, Copy, Deserialize)]
pub struct ScenarioExpect {
    pub improvement_min_pct: f64,
    pub max_post_score: f64,
    pub max_boost_db: f64,
    #[serde(default)]
    pub allow_safe_revert: bool,
    #[serde(default)]
    pub gate_purpose: QaGatePurpose,
}

impl ScenarioExpect {
    pub fn accepts_safe_revert(self) -> bool {
        self.gate_purpose == QaGatePurpose::Safety && self.allow_safe_revert
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct ScenarioFamily {
    pub id: String,
    pub scenario: String,
    pub solver: String,
    pub tier: QaTier,
    pub modes: Vec<String>,
    pub claims: Vec<String>,
    pub expect: ScenarioExpect,
}

#[derive(Debug, Clone, Deserialize)]
pub struct SuiteSpec {
    pub id: String,
    pub runner: String,
    pub tier: QaTier,
    pub claims: Vec<String>,
    #[serde(default)]
    pub cases: Vec<String>,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct HomeCinemaRuntimeExpect {
    pub redirected_bass: Option<bool>,
    #[serde(default)]
    pub adaptive_allpass: bool,
    #[serde(default)]
    pub height_alignment: bool,
    #[serde(default)]
    pub all_channel_multi_seat: bool,
    #[serde(default)]
    pub multi_seat_attempted: bool,
    #[serde(default)]
    pub multi_sub: bool,
    pub channel_count: Option<usize>,
    pub physical_sub_count: Option<usize>,
    #[serde(default)]
    pub channel_matching: bool,
    #[serde(default)]
    pub timing_alignment: bool,
    #[serde(default)]
    pub excursion_protection: bool,
    #[serde(default)]
    pub schroeder_split: bool,
    #[serde(default)]
    pub modal_basis: bool,
    pub fir_phase: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct HomeCinemaSpec {
    pub id: String,
    pub scenario: String,
    pub solver: String,
    pub mode: String,
    pub override_config: String,
    pub tier: QaTier,
    pub claims: Vec<String>,
    pub expect: ScenarioExpect,
    #[serde(default)]
    pub runtime: HomeCinemaRuntimeExpect,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum QualityCaseKind {
    Workflow,
    Generic,
    CrossModeConvergence,
    OptionEffect,
}

#[derive(Debug, Clone, Deserialize)]
pub struct QualityCaseSpec {
    pub id: String,
    pub name: String,
    pub kind: QualityCaseKind,
    pub fem_subdir: String,
    pub optim_subdir: String,
    #[serde(default)]
    pub config_path: Option<String>,
    #[serde(default)]
    pub override_dir: Option<String>,
    #[serde(default)]
    pub preserve_system: bool,
    #[serde(default)]
    pub strict_cross_mode: bool,
    #[serde(default)]
    pub options: Vec<String>,
    pub tier: QaTier,
    pub claims: Vec<String>,
    pub expect: ScenarioExpect,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ScenarioRegistry {
    pub version: u32,
    pub families: Vec<ScenarioFamily>,
    pub home_cinema: Vec<HomeCinemaSpec>,
    pub quality_cases: Vec<QualityCaseSpec>,
    pub suites: Vec<SuiteSpec>,
}

pub fn load_registry() -> Result<ScenarioRegistry> {
    let registry: ScenarioRegistry = serde_json::from_str(include_str!("registry.json"))?;
    registry.validate()?;
    Ok(registry)
}

impl ScenarioRegistry {
    pub fn validate(&self) -> Result<()> {
        if self.version != 1 {
            bail!("unsupported RoomEQ QA registry version {}", self.version);
        }
        let mut ids = HashSet::new();
        for (id, claims, expect) in self
            .families
            .iter()
            .map(|entry| (&entry.id, &entry.claims, entry.expect))
            .chain(
                self.home_cinema
                    .iter()
                    .map(|entry| (&entry.id, &entry.claims, entry.expect)),
            )
        {
            if !ids.insert(id) {
                bail!("duplicate RoomEQ QA registry id '{id}'");
            }
            if claims.is_empty() {
                bail!("RoomEQ QA registry entry '{id}' has no claims");
            }
            if !expect.improvement_min_pct.is_finite()
                || expect.improvement_min_pct <= 0.0
                || !expect.max_post_score.is_finite()
                || expect.max_post_score <= 0.0
                || !expect.max_boost_db.is_finite()
                || expect.max_boost_db <= 0.0
            {
                bail!("RoomEQ QA registry entry '{id}' has an invalid expect block");
            }
            if expect.allow_safe_revert != (expect.gate_purpose == QaGatePurpose::Safety) {
                bail!(
                    "RoomEQ QA registry entry '{id}' must allow safe reversion exactly when gate_purpose is safety"
                );
            }
        }

        for family in &self.families {
            if family.expect.gate_purpose != QaGatePurpose::Quality {
                bail!("RoomEQ QA family '{}' must use a quality gate", family.id);
            }
            if family.modes.is_empty() {
                bail!("RoomEQ QA registry family '{}' has no modes", family.id);
            }
            if !matches!(family.solver.as_str(), "fem" | "fast-hybrid") {
                bail!(
                    "RoomEQ QA registry family '{}' has unsupported solver '{}'",
                    family.id,
                    family.solver
                );
            }
            for mode in &family.modes {
                if !matches!(mode.as_str(), "iir" | "fir" | "mixed" | "mixed_phase") {
                    bail!(
                        "RoomEQ QA registry family '{}' has unsupported mode '{}'",
                        family.id,
                        mode
                    );
                }
            }
        }
        for case in &self.home_cinema {
            if case.expect.gate_purpose != QaGatePurpose::Functional {
                bail!(
                    "RoomEQ QA home-cinema case '{}' must use a functional gate",
                    case.id
                );
            }
            if !matches!(case.solver.as_str(), "fem" | "fast-hybrid") {
                bail!(
                    "RoomEQ QA registry case '{}' has unsupported solver '{}'",
                    case.id,
                    case.solver
                );
            }
            if !matches!(case.mode.as_str(), "iir" | "fir" | "mixed" | "mixed_phase") {
                bail!(
                    "RoomEQ QA registry case '{}' has unsupported mode '{}'",
                    case.id,
                    case.mode
                );
            }
            if case.override_config.trim().is_empty() {
                bail!("RoomEQ QA registry case '{}' has no override", case.id);
            }
        }
        for case in &self.quality_cases {
            if !ids.insert(&case.id) {
                bail!("duplicate RoomEQ QA registry id '{}'", case.id);
            }
            if case.claims.is_empty() {
                bail!("quality case '{}' has no claims", case.id);
            }
            let expect = case.expect;
            if !expect.improvement_min_pct.is_finite()
                || expect.improvement_min_pct <= 0.0
                || !expect.max_post_score.is_finite()
                || expect.max_post_score <= 0.0
                || !expect.max_boost_db.is_finite()
                || expect.max_boost_db <= 0.0
            {
                bail!("quality case '{}' has an invalid expect block", case.id);
            }
            if expect.allow_safe_revert != (expect.gate_purpose == QaGatePurpose::Safety) {
                bail!(
                    "quality case '{}' must allow safe reversion exactly when gate_purpose is safety",
                    case.id
                );
            }
            if matches!(case.kind, QualityCaseKind::CrossModeConvergence)
                && expect.gate_purpose != QaGatePurpose::Functional
            {
                bail!(
                    "cross-mode quality case '{}' must use a functional gate",
                    case.id
                );
            }
            if matches!(case.kind, QualityCaseKind::OptionEffect) && case.options.is_empty() {
                bail!("quality option-effect case '{}' has no options", case.id);
            }
            if !matches!(case.kind, QualityCaseKind::OptionEffect) && !case.options.is_empty() {
                bail!("non-option quality case '{}' declares options", case.id);
            }
            if case.config_path.is_some() != case.override_dir.is_some() {
                bail!(
                    "quality case '{}' must specify config_path and override_dir together",
                    case.id
                );
            }
            if (case.preserve_system || case.strict_cross_mode)
                && !matches!(case.kind, QualityCaseKind::CrossModeConvergence)
            {
                bail!(
                    "quality case '{}' enables cross-mode-only controls for a non-cross-mode kind",
                    case.id
                );
            }
            for claim in &case.claims {
                let active = match claim.as_str() {
                    "workflow" => matches!(case.kind, QualityCaseKind::Workflow),
                    "generic_modes" => matches!(case.kind, QualityCaseKind::Generic),
                    "cross_mode" => {
                        matches!(case.kind, QualityCaseKind::CrossModeConvergence)
                    }
                    "option_effect" => matches!(case.kind, QualityCaseKind::OptionEffect),
                    option => {
                        matches!(case.kind, QualityCaseKind::OptionEffect)
                            && case.options.iter().any(|configured| configured == option)
                    }
                };
                if !active {
                    bail!(
                        "quality case '{}' claim '{}' is not activated by its kind/options",
                        case.id,
                        claim
                    );
                }
            }
        }
        let mut runners = HashSet::new();
        for suite in &self.suites {
            if !ids.insert(&suite.id) {
                bail!("duplicate RoomEQ QA registry id '{}'", suite.id);
            }
            if suite.claims.is_empty() {
                bail!("RoomEQ QA registry suite '{}' has no claims", suite.id);
            }
            if !runners.insert(&suite.runner) {
                bail!("duplicate RoomEQ QA registry runner '{}'", suite.runner);
            }
        }
        Ok(())
    }

    pub fn families_for(&self, tier: QaTier) -> impl Iterator<Item = &ScenarioFamily> {
        self.families
            .iter()
            .filter(move |entry| tier.includes(entry.tier))
    }

    pub fn home_cinema_for(&self, tier: QaTier) -> impl Iterator<Item = &HomeCinemaSpec> {
        self.home_cinema
            .iter()
            .filter(move |entry| tier.includes(entry.tier))
    }

    pub fn suite_for_runner(&self, runner: &str) -> Option<&SuiteSpec> {
        self.suites.iter().find(|entry| entry.runner == runner)
    }

    pub fn quality_cases_for(&self, tier: QaTier) -> impl Iterator<Item = &QualityCaseSpec> {
        self.quality_cases
            .iter()
            .filter(move |entry| tier.includes(entry.tier))
    }
}

fn source_is_multi(source: &MeasurementSource) -> bool {
    matches!(
        source,
        MeasurementSource::Multiple(_) | MeasurementSource::InMemoryMultiple(_)
    )
}

fn speaker_has_multi_measurement(speaker: &SpeakerConfig) -> bool {
    match speaker {
        SpeakerConfig::Single(source) => source_is_multi(source),
        SpeakerConfig::Group(group) => group.measurements.iter().any(source_is_multi),
        SpeakerConfig::Topology(topology) => topology
            .drivers
            .iter()
            .any(|driver| source_is_multi(&driver.measurement)),
        SpeakerConfig::MultiSub(group) => group.subwoofers.iter().any(source_is_multi),
        SpeakerConfig::Dba(config) => config.front.iter().chain(&config.rear).any(source_is_multi),
        SpeakerConfig::Cardioid(config) => {
            source_is_multi(&config.front) || source_is_multi(&config.rear)
        }
        SpeakerConfig::SupportingSource(group) => {
            source_is_multi(&group.primary) || source_is_multi(&group.support)
        }
    }
}

/// Assert that a loaded effective config actually activates each structural
/// feature claimed by its registry entry.
pub fn verify_config_claims(config: &RoomConfig, claims: &[String]) -> Vec<String> {
    let mut failures = Vec::new();
    let system = config.system.as_ref();
    let has_sub = system.and_then(|value| value.subwoofers.as_ref()).is_some();
    for claim in claims {
        let satisfied = match claim.as_str() {
            "cardioid" => config
                .speakers
                .values()
                .any(|speaker| matches!(speaker, SpeakerConfig::Cardioid(_))),
            "mso" => config
                .speakers
                .values()
                .any(|speaker| matches!(speaker, SpeakerConfig::MultiSub(_)))
                || system
                    .and_then(|value| value.subwoofers.as_ref())
                    .is_some_and(|value| value.config == SubwooferStrategy::Mso),
            "four_subwoofers" => config.speakers.values().any(|speaker| {
                matches!(speaker, SpeakerConfig::MultiSub(group) if group.subwoofers.len() >= 4)
            }),
            "speaker_group" => config.speakers.values().any(|speaker| {
                matches!(speaker, SpeakerConfig::Group(_) | SpeakerConfig::Topology(_))
            }),
            "multi_measurement" | "multi_seat" => config
                .speakers
                .values()
                .any(speaker_has_multi_measurement),
            "bass_management" => has_sub,
            "home_cinema" | "surround" => system
                .is_some_and(|value| value.model == SystemModel::HomeCinema),
            "height_channels" => system.is_some_and(|value| {
                value.speakers.keys().any(|role| {
                    let role = role.to_ascii_uppercase();
                    role.starts_with('T') || role.contains("HEIGHT")
                })
            }),
            "stereo" => system.is_some_and(|value| value.model == SystemModel::Stereo),
            "full_range" => !has_sub,
            "multi_sub" => config
                .speakers
                .values()
                .any(|speaker| matches!(speaker, SpeakerConfig::MultiSub(_)))
                || system
                    .and_then(|value| value.subwoofers.as_ref())
                    .is_some_and(|value| value.config == SubwooferStrategy::Mso),
            // Home-cinema entries carry these as explicit runtime expectations;
            // validate_home_cinema_result checks the corresponding stage/report.
            "lfe_only"
            | "redirected_bass"
            | "linear_phase_fir"
            | "kirkeby_fir"
            | "hybrid"
            | "mixed_phase"
            | "adaptive_allpass"
            | "height_alignment"
            | "channel_matching"
            | "timing_alignment"
            | "excursion_protection"
            | "schroeder_split"
            | "modal_basis" => true,
            other => {
                failures.push(format!("unsupported structural claim '{other}'"));
                continue;
            }
        };
        if !satisfied {
            failures.push(format!("claimed feature '{claim}' is not active"));
        }
    }
    failures
}

/// Verify runtime evidence for claims whose activation is observable only
/// after the workflow executes.
pub fn verify_result_claims(result: &RoomOptimizationResult, claims: &[String]) -> Vec<String> {
    let mut failures = Vec::new();
    for claim in claims {
        let satisfied = match claim.as_str() {
            "cardioid" | "mso" | "four_subwoofers" | "speaker_group" => {
                result.channels.values().any(|chain| {
                    chain
                        .drivers
                        .as_ref()
                        .is_some_and(|drivers| drivers.len() >= 2)
                })
            }
            "bass_management" => result
                .metadata
                .bass_management
                .as_ref()
                .is_some_and(|report| report.enabled),
            "home_cinema" | "surround" | "height_channels" => {
                result.metadata.home_cinema_layout.is_some()
            }
            "multi_seat" => {
                result.metadata.multi_seat_coverage.is_some()
                    || result.metadata.multi_seat_correction.is_some()
            }
            // These are input-shape claims and were already checked before run.
            "multi_measurement" | "stereo" | "full_range" => true,
            // Home-cinema claims are checked against their typed runtime
            // expectations by validate_home_cinema_result.
            "lfe_only"
            | "redirected_bass"
            | "linear_phase_fir"
            | "kirkeby_fir"
            | "hybrid"
            | "mixed_phase"
            | "adaptive_allpass"
            | "height_alignment"
            | "channel_matching"
            | "timing_alignment"
            | "excursion_protection"
            | "schroeder_split"
            | "modal_basis"
            | "multi_sub" => true,
            other => {
                failures.push(format!("unsupported runtime claim '{other}'"));
                continue;
            }
        };
        if !satisfied {
            failures.push(format!(
                "claimed feature '{claim}' produced no runtime evidence"
            ));
        }
    }
    failures
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn safe_revert_acceptance_requires_both_safety_purpose_and_opt_in() {
        let expect = |gate_purpose, allow_safe_revert| ScenarioExpect {
            improvement_min_pct: 0.0,
            max_post_score: 0.0,
            max_boost_db: 0.0,
            allow_safe_revert,
            gate_purpose,
        };

        assert!(expect(QaGatePurpose::Safety, true).accepts_safe_revert());
        assert!(!expect(QaGatePurpose::Safety, false).accepts_safe_revert());
        assert!(!expect(QaGatePurpose::Functional, true).accepts_safe_revert());
        assert!(!expect(QaGatePurpose::Quality, true).accepts_safe_revert());
    }

    #[test]
    fn registry_is_valid_and_has_all_execution_tiers() {
        let registry = load_registry().unwrap();
        assert_eq!(registry.families.len(), 19);
        assert_eq!(registry.home_cinema.len(), 15);
        assert!(
            registry
                .families
                .iter()
                .any(|entry| entry.tier == QaTier::Pr)
        );
        assert!(
            registry
                .families
                .iter()
                .any(|entry| entry.tier == QaTier::Nightly)
        );
        assert!(
            registry
                .families
                .iter()
                .any(|entry| entry.tier == QaTier::Weekly)
        );
        let all_expectations = registry
            .families
            .iter()
            .map(|entry| (entry.id.as_str(), entry.expect))
            .chain(
                registry
                    .home_cinema
                    .iter()
                    .map(|entry| (entry.id.as_str(), entry.expect)),
            )
            .chain(
                registry
                    .quality_cases
                    .iter()
                    .map(|entry| (entry.id.as_str(), entry.expect)),
            )
            .collect::<Vec<_>>();
        assert!(
            all_expectations
                .iter()
                .all(|(_, expect)| expect.improvement_min_pct > 0.0),
            "all quality-bearing registry entries must require strict improvement"
        );
        let safe_reverts = all_expectations
            .iter()
            .filter(|(_, expect)| expect.allow_safe_revert)
            .map(|(id, _)| *id)
            .collect::<Vec<_>>();
        assert!(
            safe_reverts.is_empty(),
            "functional and quality registry gates must not accept safe reversion"
        );
        for runner in [
            "synthetic",
            "features",
            "acoustic",
            "integration",
            "quality",
            "fuzzer",
        ] {
            assert!(
                registry.suite_for_runner(runner).is_some(),
                "missing {runner}"
            );
        }
    }

    #[test]
    fn quality_case_claims_must_match_the_executed_kind_or_option() {
        let mut registry = load_registry().unwrap();
        registry.quality_cases[0].claims = vec!["spatial_robustness".to_string()];
        let error = registry.validate().unwrap_err().to_string();
        assert!(error.contains("not activated"), "{error}");
    }

    #[test]
    fn unknown_runtime_claims_fail_closed() {
        let result = RoomOptimizationResult {
            channels: Default::default(),
            channel_results: Default::default(),
            combined_pre_score: 0.0,
            combined_post_score: 0.0,
            metadata: roomeq_model::OptimizationMetadata {
                pre_score: 0.0,
                post_score: 0.0,
                algorithm: "test".to_string(),
                loss_type: None,
                iterations: 0,
                timestamp: String::new(),
                inter_channel_deviation: None,
                epa_per_channel: None,
                epa_multichannel: None,
                group_delay: None,
                mixed_phase_per_channel: None,
                perceptual_metrics: None,
                home_cinema_layout: None,
                multi_seat_coverage: None,
                multi_seat_correction: None,
                bass_management: None,
                timing_diagnostics: None,
                ctc: None,
                perceptual_policy: None,
                bootstrap_uncertainty: None,
                validation_bundle: None,
                supporting_source: None,
                correction_acceptance: None,
                optimizer_evidence: None,
                stage_outcomes: Vec::new(),
                effective_config: None,
            },
        };
        assert_eq!(
            verify_result_claims(&result, &["future_claim".to_string()]),
            ["unsupported runtime claim 'future_claim'"]
        );
    }
}
