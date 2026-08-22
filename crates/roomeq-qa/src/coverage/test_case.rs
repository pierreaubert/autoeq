use super::consts::OPTIM_CONFIG_DIR;
use super::home_cinema::HomeCinemaExpectations;
use super::misc::project_root;
use super::processing_method::ProcessingMethod;
use super::solver::Solver;
use crate::registry::{ScenarioExpect, verify_config_claims};
use anyhow::{Result, bail};
use roomeq_model::RoomConfig;
use roomeq_workflow::load_merged_config_strict;
use std::path::PathBuf;

#[derive(Debug, Clone)]
#[allow(dead_code)]
pub(super) struct TestCase {
    pub(super) registry_id: String,
    pub(super) scenario: String,
    pub(super) description: String,
    pub(super) solver: Solver,
    pub(super) method: ProcessingMethod,
    pub(super) case_name: Option<String>,
    pub(super) override_file: Option<PathBuf>,
    pub(super) home_cinema_expectations: Option<HomeCinemaExpectations>,
    pub(super) claims: Vec<String>,
    pub(super) expect: ScenarioExpect,
}

impl TestCase {
    pub(super) fn name(&self) -> String {
        self.case_name.clone().unwrap_or_else(|| {
            format!(
                "{} {} {}",
                self.scenario,
                self.solver.name(),
                self.method.name()
            )
        })
    }

    pub(super) fn config_path(&self) -> PathBuf {
        let base = self.solver.dir();
        project_root()
            .join(base)
            .join(&self.scenario)
            .join("config.json")
    }

    pub(super) fn override_path(&self) -> PathBuf {
        let optim_dir = project_root().join(OPTIM_CONFIG_DIR);
        if let Some(override_file) = &self.override_file {
            return optim_dir.join(override_file);
        }
        let scenario_override = optim_dir
            .join(&self.scenario)
            .join(self.method.config_file());
        if scenario_override.exists() {
            scenario_override
        } else {
            optim_dir.join("modes").join(self.method.config_file())
        }
    }
}

pub(super) fn print_matrix(test_cases: &[TestCase]) {
    println!("Test Matrix ({} cases):\n", test_cases.len());
    println!("{:<30} {:>6} {:>8}", "Scenario", "Solver", "Mode");
    println!("{:-<30} {:-<6} {:-<8}", "", "", "");

    for tc in test_cases {
        println!(
            "{:<30} {:>6} {:>8}",
            tc.scenario,
            tc.solver.name(),
            tc.method.name()
        );
    }
}

pub(super) fn load_config_for_test(tc: &TestCase) -> Result<(RoomConfig, PathBuf)> {
    let config_path = tc.config_path();
    let override_path = tc.override_path();
    let override_path = override_path.exists().then_some(override_path);
    let (room_config, config_dir) =
        load_merged_config_strict(&config_path, override_path.as_deref())?;

    let expected_mode = tc.method.mode();
    if let Some(path) = override_path.as_deref() {
        let override_value: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(path)?)?;
        if override_value
            .pointer("/optimizer/processing_mode")
            .is_some()
        {
            log::warn!(
                "QA override {} explicitly selects processing mode {:?}",
                path.display(),
                room_config.optimizer.processing_mode
            );
        }
    }
    if room_config.optimizer.processing_mode != expected_mode {
        bail!(
            "QA case '{}' claims {:?}, but its effective config selects {:?}; fix the override instead of forcing the mode in the runner",
            tc.name(),
            expected_mode,
            room_config.optimizer.processing_mode
        );
    }
    let claim_failures = verify_config_claims(&room_config, &tc.claims);
    if !claim_failures.is_empty() {
        bail!(
            "QA registry entry '{}' has false claims: {}",
            tc.registry_id,
            claim_failures.join("; ")
        );
    }

    Ok((room_config, config_dir))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::coverage::home_cinema::build_home_cinema_matrix;
    use crate::coverage::processing_method::build_test_matrix_for_tier;
    use crate::registry::QaTier;
    use roomeq_workflow::deserialize_room_config_strict;
    use std::fs;
    use std::path::Path;

    fn collect_json_files(directory: &Path, files: &mut Vec<PathBuf>) {
        for entry in fs::read_dir(directory)
            .unwrap_or_else(|error| panic!("failed to read {}: {error}", directory.display()))
        {
            let path = entry.expect("directory entry should be readable").path();
            if path.is_dir() {
                collect_json_files(&path, files);
            } else if path
                .extension()
                .is_some_and(|extension| extension == "json")
            {
                files.push(path);
            }
        }
    }

    #[test]
    fn all_repository_room_configs_strictly_deserialize() {
        let root = project_root();
        let mut files = Vec::new();
        collect_json_files(
            &root.join("data_tests/roomeq/generate/optimiser-config"),
            &mut files,
        );
        collect_json_files(&root.join("data_tests/roomeq/generate/fem"), &mut files);
        collect_json_files(
            &root.join("data_tests/roomeq/generate/fast-hybrid"),
            &mut files,
        );
        collect_json_files(&root.join("data_tests/roomeq/acoustic_corpus"), &mut files);
        files.retain(|path| {
            let name = path
                .file_name()
                .and_then(|name| name.to_str())
                .unwrap_or("");
            name == "config.json"
                || name.starts_with("optimiser-")
                || name.starts_with("qa_optimizer")
                || path
                    .components()
                    .any(|component| component.as_os_str() == "home_cinema")
                || path
                    .components()
                    .any(|component| component.as_os_str() == "modes")
                || path
                    .components()
                    .any(|component| component.as_os_str() == "multi_measurement")
        });
        files.sort();
        assert!(!files.is_empty(), "RoomEQ config lint selected no files");

        let mut failures = Vec::new();
        for path in files {
            let result = fs::read_to_string(&path)
                .map_err(anyhow::Error::from)
                .and_then(|json| {
                    let mut value: serde_json::Value =
                        serde_json::from_str(&json).map_err(anyhow::Error::from)?;
                    if let Some(object) = value.as_object_mut() {
                        object
                            .entry("speakers")
                            .or_insert_with(|| serde_json::json!({}));
                    }
                    Ok(value)
                })
                .and_then(deserialize_room_config_strict);
            if let Err(error) = result {
                failures.push(format!("{}: {error:#}", path.display()));
            }
        }
        assert!(
            failures.is_empty(),
            "repository RoomEQ config lint failed:\n{}",
            failures.join("\n")
        );
    }

    #[test]
    fn registered_coverage_configs_strictly_deserialize() {
        let mut cases = build_test_matrix_for_tier(QaTier::Weekly, false, None, None);
        cases.extend(build_home_cinema_matrix(QaTier::Weekly, None, None));
        let mut failures = Vec::new();
        for test_case in cases {
            if let Err(error) = load_config_for_test(&test_case) {
                failures.push(format!("{}: {error:#}", test_case.name()));
            }
        }
        assert!(
            failures.is_empty(),
            "registered RoomEQ configs failed strict lint:\n{}",
            failures.join("\n")
        );
    }
}
