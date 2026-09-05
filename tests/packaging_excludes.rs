//! Regression test: the root `autoeq` package must exclude QA fixtures,
//! helper scripts, editor backups, and REW `.mdat` files from the published
//! archive. The `exclude = [...]` list must live under `[package]` (packaging
//! scope); under `[workspace]` it only excludes workspace members and
//! `cargo package` ships everything.
use std::process::Command;

fn package_file_list() -> Vec<String> {
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let output = Command::new("cargo")
        .args(["package", "-p", "autoeq", "--list", "--allow-dirty"])
        .current_dir(manifest_dir)
        .output()
        .expect("failed to run `cargo package -p autoeq --list --allow-dirty`");
    assert!(
        output.status.success(),
        "cargo package --list failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8_lossy(&output.stdout)
        .lines()
        .map(str::to_string)
        .collect()
}

fn is_excluded_path(path: &str) -> bool {
    path.starts_with("data_tests/")
        || path.starts_with("data_generated/")
        || path.starts_with("scripts/")
        || path.ends_with(".mdat")
        || path.ends_with('~')
}

#[test]
fn published_package_excludes_qa_scripts_backups_and_mdat() {
    let files = package_file_list();
    assert!(
        !files.is_empty(),
        "expected a non-empty `cargo package --list` output"
    );
    let leaked: Vec<&String> = files
        .iter()
        .filter(|p| is_excluded_path(p))
        .collect();
    assert!(
        leaked.is_empty(),
        "published package must not contain QA/script/backup/mdat files, leaked: {leaked:?}"
    );
}
