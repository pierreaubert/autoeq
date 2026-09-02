use anyhow::{Result, anyhow};
use roomeq_engine::room_result::RoomOptimizationResult;
use std::path::{Path, PathBuf};

/// Compute average EPA post-preference across channels.
pub(super) fn avg_epa_preference(result: &RoomOptimizationResult) -> Option<f64> {
    let epa = result.metadata.epa_per_channel.as_ref()?;
    if epa.is_empty() {
        return None;
    }
    let sum: f64 = epa.values().map(|m| m.post.preference).sum();
    Some(sum / epa.len() as f64)
}

pub(super) fn discover_recordings(
    project_root: &Path,
    scenario: Option<&str>,
) -> Result<Vec<(String, PathBuf)>> {
    let qa_data_dir = project_root.join("data_tests/roomeq/measured");
    if !qa_data_dir.exists() {
        return Err(anyhow!("QA data directory not found: {:?}", qa_data_dir));
    }

    let mut recordings = Vec::new();
    for entry in std::fs::read_dir(&qa_data_dir)? {
        let entry = entry?;
        if !entry.file_type()?.is_dir() {
            continue;
        }
        let recordings_json = entry.path().join("recordings.json");
        if recordings_json.exists() {
            let name = entry.file_name().to_string_lossy().to_string();
            recordings.push((name, recordings_json));
        }
    }

    recordings.sort_by(|a, b| a.0.cmp(&b.0));

    if recordings.is_empty() {
        return Err(anyhow!("No recordings found in {:?}", qa_data_dir));
    }

    select_recordings(recordings, scenario, &qa_data_dir)
}

fn select_recordings(
    recordings: Vec<(String, PathBuf)>,
    scenario: Option<&str>,
    qa_data_dir: &Path,
) -> Result<Vec<(String, PathBuf)>> {
    let Some(scenario) = scenario else {
        return Ok(recordings);
    };

    let available = recordings
        .iter()
        .map(|(name, _)| name.as_str())
        .collect::<Vec<_>>()
        .join(", ");
    recordings
        .into_iter()
        .find(|(name, _)| name == scenario)
        .map(|recording| vec![recording])
        .ok_or_else(|| {
            anyhow!(
                "Recording '{scenario}' not found in {:?}. Available recordings: {available}",
                qa_data_dir
            )
        })
}

pub(super) fn find_project_root() -> Result<PathBuf> {
    let mut dir = std::env::current_dir()?;
    loop {
        let cargo_toml = dir.join("Cargo.toml");
        if cargo_toml.exists() {
            let content = std::fs::read_to_string(&cargo_toml)?;
            if content.contains("[workspace]") {
                return Ok(dir);
            }
        }
        if !dir.pop() {
            return Err(anyhow!(
                "Could not find project root (Cargo.toml with [workspace])"
            ));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::select_recordings;
    use std::path::{Path, PathBuf};

    fn recordings() -> Vec<(String, PathBuf)> {
        vec![
            (
                "2.0_fidelia".to_string(),
                PathBuf::from("2.0_fidelia/recordings.json"),
            ),
            (
                "5.1_kef".to_string(),
                PathBuf::from("5.1_kef/recordings.json"),
            ),
        ]
    }

    #[test]
    fn selecting_a_scenario_returns_only_that_recording() {
        let selected =
            select_recordings(recordings(), Some("5.1_kef"), Path::new("measured")).unwrap();
        assert_eq!(selected.len(), 1);
        assert_eq!(selected[0].0, "5.1_kef");
    }

    #[test]
    fn selecting_an_unknown_scenario_lists_available_recordings() {
        let error = select_recordings(recordings(), Some("missing"), Path::new("measured"))
            .unwrap_err()
            .to_string();
        assert!(error.contains("Recording 'missing' not found"));
        assert!(error.contains("2.0_fidelia, 5.1_kef"));
    }
}
