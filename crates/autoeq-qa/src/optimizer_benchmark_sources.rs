//! Resolve measured benchmark inputs from their declared manifest paths.

// Rust guideline compliant 2026-02-21

use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::collections::BTreeSet;
use std::fs;
use std::path::{Component, Path};

/// One immutable read of the files declared by a benchmark fixture.
#[derive(Debug)]
pub(crate) struct SourceSnapshot {
    bytes_by_path: BTreeMap<String, Vec<u8>>,
    hashes_by_path: BTreeMap<String, String>,
}

impl SourceSnapshot {
    /// Return the bytes read for one declared relative path.
    pub(crate) fn bytes_for(&self, relative_path: &str) -> Result<&[u8], String> {
        self.bytes_by_path
            .get(relative_path)
            .map(Vec::as_slice)
            .ok_or_else(|| format!("source path '{relative_path}' was not snapshotted"))
    }

    /// Return hashes computed from the exact retained file bytes.
    pub(crate) fn hashes(&self) -> &BTreeMap<String, String> {
        &self.hashes_by_path
    }
}

/// Read and hash every declared source file once.
///
/// Consumers can parse data from [`SourceSnapshot::bytes_for`] and report the
/// paired hashes from [`SourceSnapshot::hashes`], keeping input identity tied
/// to the same byte buffers. Paths receive lexical repository-relative
/// validation; symlinks are not resolved by this helper.
pub(crate) fn snapshot_declared_sources(
    root: &Path,
    declared_paths: &[String],
) -> Result<SourceSnapshot, String> {
    let mut bytes_by_path = BTreeMap::new();
    let mut hashes_by_path = BTreeMap::new();

    for declared in declared_paths {
        validate_repository_relative_path(declared)?;
        if bytes_by_path.contains_key(declared) {
            return Err(format!("duplicate declared source path '{declared}'"));
        }
        let bytes = fs::read(root.join(declared))
            .map_err(|error| format!("cannot snapshot source path '{declared}': {error}"))?;
        hashes_by_path.insert(declared.clone(), sha256_hex(&bytes));
        bytes_by_path.insert(declared.clone(), bytes);
    }

    Ok(SourceSnapshot {
        bytes_by_path,
        hashes_by_path,
    })
}

fn sha256_hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

/// Paths for the six measured room curves used by the fixed benchmark case.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct RoomInputPaths<'a> {
    pub(crate) training_left: &'a str,
    pub(crate) training_right: &'a str,
    pub(crate) held_out_left_1: &'a str,
    pub(crate) held_out_left_2: &'a str,
    pub(crate) held_out_right_1: &'a str,
    pub(crate) held_out_right_2: &'a str,
}

/// Resolve channel roles from declared paths and reject ambiguous inventories.
///
/// Paths are borrowed from `declared_paths`; callers that hash that same list
/// therefore identify the exact path strings used to load each room curve.
/// The six CSV roles are identified by unique file names and must share one
/// measurement directory. Other declared paths may record source or license
/// material, but they cannot take the place of a measured curve.
pub(crate) fn resolve_room_input_paths(
    declared_paths: &[String],
) -> Result<RoomInputPaths<'_>, String> {
    let mut unique_paths = BTreeSet::new();
    let mut roles: [Option<&str>; 6] = [None; 6];
    let mut measurement_parent: Option<&Path> = None;

    for declared in declared_paths {
        let path = validate_repository_relative_path(declared)?;
        if !unique_paths.insert(declared.as_str()) {
            return Err(format!("duplicate declared source path '{declared}'"));
        }

        let Some(file_name) = path.file_name().and_then(|value| value.to_str()) else {
            continue;
        };
        let Some(role_index) = room_role_index(file_name) else {
            continue;
        };
        if roles[role_index].replace(declared.as_str()).is_some() {
            return Err(format!("duplicate room input role for '{file_name}'"));
        }

        let parent = path
            .parent()
            .ok_or_else(|| format!("room input '{declared}' has no parent directory"))?;
        if let Some(expected_parent) = measurement_parent {
            if parent != expected_parent {
                return Err(format!(
                    "room input '{declared}' is outside the shared measurement directory"
                ));
            }
        } else {
            measurement_parent = Some(parent);
        }
    }

    let mut take_role = |index: usize, name: &str| {
        roles[index]
            .take()
            .ok_or_else(|| format!("room benchmark manifest is missing the declared {name} curve"))
    };

    Ok(RoomInputPaths {
        training_left: take_role(0, "training-left")?,
        training_right: take_role(1, "training-right")?,
        held_out_left_1: take_role(2, "held-out-left-1")?,
        held_out_left_2: take_role(3, "held-out-left-2")?,
        held_out_right_1: take_role(4, "held-out-right-1")?,
        held_out_right_2: take_role(5, "held-out-right-2")?,
    })
}

fn validate_repository_relative_path(value: &str) -> Result<&Path, String> {
    if value.is_empty()
        || value.contains('\\')
        || value.contains(':')
        || value.as_bytes().contains(&0)
    {
        return Err(format!("invalid declared source path '{value}'"));
    }

    let path = Path::new(value);
    let mut normal_components = Vec::new();
    for component in path.components() {
        match component {
            Component::Normal(part) => normal_components.push(part.to_string_lossy()),
            Component::CurDir
            | Component::ParentDir
            | Component::RootDir
            | Component::Prefix(_) => {
                return Err(format!(
                    "declared source path must stay inside the repository: '{value}'"
                ));
            }
        }
    }
    let canonical = normal_components.join("/");
    if normal_components.is_empty() || canonical != value {
        return Err(format!(
            "declared source path is not canonical repository-relative text: '{value}'"
        ));
    }
    Ok(path)
}

fn room_role_index(file_name: &str) -> Option<usize> {
    match file_name {
        "L.csv" => Some(0),
        "R.csv" => Some(1),
        "L_heldout_1.csv" => Some(2),
        "L_heldout_2.csv" => Some(3),
        "R_heldout_1.csv" => Some(4),
        "R_heldout_2.csv" => Some(5),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::{RoomInputPaths, resolve_room_input_paths, snapshot_declared_sources};
    use std::fs;
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    const MEASUREMENT_DIR: &str = "data_tests/roomeq/measured/2.0_8361a";

    fn declared_paths() -> Vec<String> {
        [
            "L.csv",
            "R.csv",
            "L_heldout_1.csv",
            "L_heldout_2.csv",
            "R_heldout_1.csv",
            "R_heldout_2.csv",
        ]
        .into_iter()
        .map(|file| format!("{MEASUREMENT_DIR}/{file}"))
        .chain([
            "data_tests/roomeq/acoustic_corpus/PROVENANCE.md".to_string(),
            "scripts/generate_roomeq_held_out.py".to_string(),
        ])
        .collect()
    }

    struct TemporaryDirectory(PathBuf);

    impl Drop for TemporaryDirectory {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    fn temporary_directory() -> TemporaryDirectory {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system time is after the Unix epoch")
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "autoeq-qa-source-snapshot-{}-{nonce}",
            std::process::id()
        ));
        fs::create_dir(&path).expect("create isolated test directory");
        TemporaryDirectory(path)
    }

    #[test]
    fn declared_order_does_not_change_explicit_room_roles() {
        let paths = declared_paths();
        let resolved = resolve_room_input_paths(&paths).expect("room role map");

        assert_eq!(
            resolved,
            RoomInputPaths {
                training_left: "data_tests/roomeq/measured/2.0_8361a/L.csv",
                training_right: "data_tests/roomeq/measured/2.0_8361a/R.csv",
                held_out_left_1: "data_tests/roomeq/measured/2.0_8361a/L_heldout_1.csv",
                held_out_left_2: "data_tests/roomeq/measured/2.0_8361a/L_heldout_2.csv",
                held_out_right_1: "data_tests/roomeq/measured/2.0_8361a/R_heldout_1.csv",
                held_out_right_2: "data_tests/roomeq/measured/2.0_8361a/R_heldout_2.csv",
            }
        );
        for path in [
            resolved.training_left,
            resolved.training_right,
            resolved.held_out_left_1,
            resolved.held_out_left_2,
            resolved.held_out_right_1,
            resolved.held_out_right_2,
        ] {
            assert!(paths.iter().any(|declared| declared == path));
        }

        let mut reversed = paths.clone();
        reversed.reverse();
        assert_eq!(resolve_room_input_paths(&reversed), Ok(resolved));
    }

    #[test]
    fn changed_or_missing_room_path_is_refused() {
        let mut paths = declared_paths();
        let left = paths
            .iter_mut()
            .find(|path| path.ends_with("/L.csv"))
            .expect("training-left path");
        *left = format!("{MEASUREMENT_DIR}/L_changed.csv");

        let error = resolve_room_input_paths(&paths).expect_err("changed role must refuse");
        assert!(error.contains("training-left"));
    }

    #[test]
    fn duplicate_role_and_mixed_measurement_directories_are_refused() {
        let mut duplicate = declared_paths();
        let duplicate_path = duplicate[0].clone();
        duplicate.push(duplicate_path);
        assert!(
            resolve_room_input_paths(&duplicate)
                .expect_err("duplicate path must refuse")
                .contains("duplicate declared source path")
        );

        let mut duplicate_role = declared_paths();
        duplicate_role.push("data_tests/roomeq/measured/other/L.csv".to_string());
        assert!(
            resolve_room_input_paths(&duplicate_role)
                .expect_err("duplicate role under another path must refuse")
                .contains("duplicate room input role")
        );

        let mut mixed = declared_paths();
        let right = mixed
            .iter_mut()
            .find(|path| path.ends_with("/R.csv"))
            .expect("training-right path");
        *right = "data_tests/roomeq/measured/other/R.csv".to_string();
        assert!(
            resolve_room_input_paths(&mixed)
                .expect_err("mixed measurement directories must refuse")
                .contains("shared measurement directory")
        );
    }

    #[test]
    fn unsafe_noncanonical_and_unrecognized_curve_paths_are_refused() {
        for path in [
            "/tmp/L.csv",
            "data_tests/../L.csv",
            "data_tests//roomeq/L.csv",
            "C:/measurements/L.csv",
            "data_tests\\L.csv",
        ] {
            let paths = vec![path.to_string()];
            assert!(resolve_room_input_paths(&paths).is_err(), "accepted {path}");
        }

        let mut paths = declared_paths();
        paths.retain(|path| !path.ends_with("/L_heldout_2.csv"));
        assert!(
            resolve_room_input_paths(&paths)
                .expect_err("missing held-out curve must refuse")
                .contains("held-out-left-2")
        );
    }

    #[test]
    fn snapshot_parser_and_hash_use_the_same_original_bytes_after_file_changes() {
        let directory = temporary_directory();
        let relative_path = "curve.csv";
        let source = directory.0.join(relative_path);
        let original = b"frequency_hz,spl_db\n100,81.5\n200,79.0\n";
        fs::write(&source, original).expect("write initial source bytes");

        let snapshot = snapshot_declared_sources(&directory.0, &[relative_path.to_string()])
            .expect("snapshot declared source");
        fs::write(&source, b"frequency_hz,spl_db\n100,-999\n")
            .expect("mutate source after snapshot");

        let bytes = snapshot
            .bytes_for(relative_path)
            .expect("retained input bytes");
        let mut reader = csv::Reader::from_reader(bytes);
        let rows = reader
            .records()
            .collect::<Result<Vec<_>, _>>()
            .expect("parse retained CSV bytes");
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].get(1), Some("81.5"));
        assert_eq!(rows[1].get(1), Some("79.0"));

        assert_eq!(
            snapshot.hashes().get(relative_path),
            Some(&super::sha256_hex(bytes))
        );
        assert_ne!(bytes, fs::read(source).expect("read changed source bytes"));
    }
}
