#!/usr/bin/env python3
"""Run the Cargo-member test matrix and the declared numerical QA cases."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
import pathlib
import platform
import re
import shlex
import shutil
import subprocess
import sys
import tomllib
from typing import Any


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_POLICY = REPO_ROOT / "scripts" / "crate_partition_policy.json"
RUNNING_TEST_TARGET = re.compile(r"Running tests/([^\s/]+)\.rs(?:\s|\()")
TEST_RESULT = re.compile(
    r"test result: (?:ok|FAILED)\.\s+(\d+) passed;\s+(\d+) failed;\s+(\d+) ignored;"
)
QA_RESULT_PREFIX = "QA_RESULT:"
PINNED_REQUIREMENT = re.compile(r"^([A-Za-z0-9_.-]+)\s*==")
GENERATED_PATH_PARTS = {
    ".git",
    ".tokensave",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "target",
    ".DS_Store",
}


def unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def load_policy(policy_path: pathlib.Path) -> dict[str, Any]:
    return json.loads(
        policy_path.read_text(encoding="utf-8"), object_pairs_hook=unique_json_object
    )


def focused_tests(policy_path: pathlib.Path) -> dict[str, str]:
    policy = load_policy(policy_path)
    return policy["focused_tests"]


def command_for_package(
    tests: dict[str, str], package: str, *, release: bool
) -> list[str]:
    try:
        command = shlex.split(tests[package])
    except KeyError as error:
        available = ", ".join(sorted(tests))
        raise ValueError(
            f"unknown focused-test package {package!r}; expected one of: {available}"
        ) from error
    if release and "--release" not in command:
        try:
            separator = command.index("--")
        except ValueError:
            separator = len(command)
        command.insert(separator, "--release")
    return command


def cargo_program() -> str:
    return os.environ.get("CARGO", "cargo")


def cargo_metadata() -> dict[str, Any]:
    command = [
        cargo_program(),
        "metadata",
        "--format-version",
        "1",
        "--no-deps",
        "--locked",
    ]
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode:
        raise RuntimeError("cargo metadata failed:\n" + completed.stderr.rstrip())
    return json.loads(completed.stdout)


def workspace_packages(metadata: dict[str, Any]) -> dict[str, dict[str, Any]]:
    member_ids = set(metadata["workspace_members"])
    packages = {
        package["name"]: package
        for package in metadata["packages"]
        if package["id"] in member_ids
    }
    if len(packages) != len(member_ids):
        raise ValueError("Cargo metadata contains duplicate workspace package names")
    return packages


def validate_package_matrix(
    policy: dict[str, Any], packages: dict[str, dict[str, Any]]
) -> list[str]:
    errors: list[str] = []
    tests = policy["focused_tests"]
    expected_count = policy.get("workspace_package_count")
    if expected_count is not None and len(packages) != expected_count:
        errors.append(
            f"workspace package count changed: expected {expected_count}, found {len(packages)}"
        )
    missing = sorted(set(packages) - set(tests))
    extra = sorted(set(tests) - set(packages))
    errors.extend(f"focused test command missing for {name}" for name in missing)
    errors.extend(f"focused test command names absent package {name}" for name in extra)

    for package_name, command_text in tests.items():
        try:
            arguments = shlex.split(command_text)
        except ValueError as error:
            errors.append(f"invalid focused test command for {package_name}: {error}")
            continue
        expected = ["cargo", "test", "-p", package_name]
        if arguments[:4] != expected:
            errors.append(
                f"focused test command for {package_name} must start with "
                + " ".join(expected)
            )
        if "--locked" not in arguments:
            errors.append(f"focused test command for {package_name} must use --locked")
        package = packages.get(package_name)
        if package is None:
            continue
        feature_names = set(package.get("features", {}))
        requested_features: set[str] = set()
        for index, argument in enumerate(arguments):
            if argument in {"--features", "-F"} and index + 1 < len(arguments):
                requested_features.update(arguments[index + 1].split(","))
            elif argument.startswith("--features="):
                requested_features.update(argument.split("=", 1)[1].split(","))
        invalid_features = sorted(requested_features - feature_names)
        if invalid_features:
            errors.append(
                f"focused test command for {package_name} requests unknown Cargo features: "
                + ", ".join(invalid_features)
            )
        library_kinds = {"lib", "rlib", "cdylib", "dylib", "staticlib", "proc-macro"}
        if "--lib" in arguments and not any(
            (set(target.get("kind", [])) | set(target.get("crate_types", []))) & library_kinds
            for target in package.get("targets", [])
        ):
            errors.append(f"focused test command for {package_name} requests a missing lib target")
    return errors


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def working_tree_snapshot(root: pathlib.Path) -> dict[str, Any]:
    """Hash tracked and non-ignored files, excluding generated caches and builds."""
    root = root.resolve()
    completed = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        cwd=root,
        check=False,
        capture_output=True,
    )
    if completed.returncode == 0:
        paths = [
            pathlib.Path(os.fsdecode(item))
            for item in completed.stdout.split(b"\0")
            if item
        ]
        method = "git-ls-files"
    else:
        paths = []
        for directory, names, files in os.walk(root):
            names[:] = sorted(
                name for name in names if name not in GENERATED_PATH_PARTS
            )
            for file_name in files:
                paths.append(pathlib.Path(directory, file_name).relative_to(root))
        method = "filesystem-fallback"

    digest = hashlib.sha256()
    included = 0
    for relative in sorted(set(paths), key=lambda item: item.as_posix()):
        if GENERATED_PATH_PARTS.intersection(relative.parts):
            continue
        path = root / relative
        encoded_path = relative.as_posix().encode("utf-8", errors="surrogateescape")
        digest.update(len(encoded_path).to_bytes(8, "big"))
        digest.update(encoded_path)
        if not path.is_file():
            digest.update(b"<missing>")
            included += 1
            continue
        file_digest = hashlib.sha256()
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                file_digest.update(chunk)
        digest.update(file_digest.digest())
        included += 1
    return {
        "sha256": digest.hexdigest(),
        "file_count": included,
        "method": method,
    }


def git_head_and_dirty_files(root: pathlib.Path) -> tuple[str | None, list[str]]:
    head = subprocess.run(
        ["git", "rev-parse", "--verify", "HEAD"],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    status = subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    return (
        head.stdout.strip() if head.returncode == 0 else None,
        status.stdout.splitlines() if status.returncode == 0 else [],
    )


def git_repository_root(path: pathlib.Path) -> pathlib.Path | None:
    completed = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        cwd=path,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode:
        return None
    return pathlib.Path(completed.stdout.strip()).resolve()


def full_cargo_metadata() -> dict[str, Any]:
    command = [
        cargo_program(),
        "metadata",
        "--format-version",
        "1",
        "--locked",
        "--offline",
    ]
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode:
        raise RuntimeError(
            "full Cargo metadata failed while resolving local dependency provenance:\n"
            + completed.stderr.rstrip()
        )
    return json.loads(completed.stdout)


def resolved_external_local_packages(
    metadata: dict[str, Any],
) -> list[dict[str, Any]]:
    """Return resolved path packages outside this workspace, not config guesses."""
    workspace_members = set(metadata.get("workspace_members", []))
    packages = [
        package
        for package in metadata.get("packages", [])
        if package.get("id") not in workspace_members and package.get("source") is None
    ]
    return sorted(packages, key=lambda package: package["id"])


def local_path_dependency_provenance(
    metadata: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    metadata = metadata if metadata is not None else full_cargo_metadata()
    repositories: dict[str, dict[str, Any]] = {}
    for package in resolved_external_local_packages(metadata):
        manifest_path = pathlib.Path(package["manifest_path"]).resolve()
        if not manifest_path.is_file():
            raise RuntimeError(
                "resolved local Cargo package is unavailable: "
                f"{package['name']} {package['version']} at {manifest_path}"
            )
        dependency_path = manifest_path.parent
        git_root = git_repository_root(dependency_path)
        repository_root = git_root or dependency_path
        package_relative = pathlib.PurePosixPath(
            os.path.relpath(dependency_path, repository_root).replace(os.sep, "/")
        )
        repository_key = str(repository_root)
        existing_repository = repositories.get(repository_key)
        if existing_repository is None:
            head, dirty_files = git_head_and_dirty_files(repository_root)
            tree = working_tree_snapshot(repository_root)
        else:
            head = existing_repository["git_head"]
            dirty_files = existing_repository["dirty_files"]
            tree = existing_repository["working_tree"]
        repository = repositories.setdefault(
            repository_key,
            {
                "path_from_repository": os.path.relpath(repository_root, REPO_ROOT).replace(
                    os.sep, "/"
                ),
                "available": True,
                "git_checkout": git_root is not None,
                "git_head": head,
                "dirty_files": dirty_files,
                "working_tree": tree,
                "packages": [],
            },
        )
        repository["packages"].append({
            "id": package["id"],
            "name": package["name"],
            "version": package["version"],
            "package_relative_path": package_relative.as_posix(),
            "manifest_path": (package_relative / "Cargo.toml").as_posix(),
            "source": package.get("source"),
        })
    result = list(repositories.values())
    for repository in result:
        repository["packages"].sort(key=lambda item: item["name"])
    return sorted(result, key=lambda item: item["path_from_repository"])


def capture_source_provenance() -> dict[str, Any]:
    head, dirty_files = git_head_and_dirty_files(REPO_ROOT)
    lock_path = REPO_ROOT / "Cargo.lock"
    local_dependencies = local_path_dependency_provenance()
    cargo_config_path = REPO_ROOT / ".cargo" / "config.toml"
    return {
        "git_head": head,
        "dirty_files": dirty_files,
        "working_tree": working_tree_snapshot(REPO_ROOT),
        "cargo_lock_sha256": sha256_file(lock_path) if lock_path.is_file() else None,
        "cargo_config_sha256": sha256_file(cargo_config_path)
        if cargo_config_path.is_file()
        else None,
        "local_path_repositories": local_dependencies,
    }


def source_provenance_digest(snapshot: dict[str, Any]) -> str:
    encoded = json.dumps(snapshot, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def command_version(command: list[str]) -> str | None:
    try:
        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as error:
        return f"unavailable: {error}"
    if completed.returncode:
        return f"unavailable: {completed.stderr.strip()}"
    return (completed.stdout or completed.stderr).strip()


def run_environment_provenance() -> dict[str, Any]:
    requirement_path = REPO_ROOT / "scripts" / "autoeq-qa-requirements.txt"
    requirement_pins: dict[str, str] = {}
    requirements: dict[str, str | None] = {}
    if requirement_path.is_file():
        for line in requirement_path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            match = PINNED_REQUIREMENT.match(stripped)
            if match:
                name = match.group(1)
                requirement_pins[name] = stripped.split("==", 1)[1].strip()
                try:
                    requirements[name] = importlib.metadata.version(name)
                except importlib.metadata.PackageNotFoundError:
                    requirements[name] = None
    cargo = shlex.split(cargo_program())
    rustc = shlex.split(os.environ.get("RUSTC", ""))
    if not rustc:
        cargo_executable = shutil.which(cargo[0]) if cargo else None
        if cargo_executable:
            sibling_rustc = pathlib.Path(cargo_executable).with_name("rustc")
            rustc = [str(sibling_rustc)] if sibling_rustc.exists() else ["rustc"]
        else:
            rustc = ["rustc"]
    return {
        "platform": {
            "platform": platform.platform(),
            "system": platform.system(),
            "machine": platform.machine(),
        },
        "cargo_version": command_version(cargo + ["--version"]),
        "rustc_version": command_version(rustc + ["--version", "--verbose"]),
        "cargo_target": os.environ.get("CARGO_BUILD_TARGET", "host default"),
        "rustflags": os.environ.get("RUSTFLAGS"),
        "python": {
            "executable": sys.executable,
            "version": sys.version,
            "required_dependencies_file": relative_path(requirement_path),
            "required_dependencies_sha256": sha256_file(requirement_path)
            if requirement_path.is_file()
            else None,
            "requirement_pins": requirement_pins,
            "dependencies": requirements,
            "requirements_match": all(
                requirements.get(name) == version
                for name, version in requirement_pins.items()
            ),
        },
    }


def provenance_report(
    before: dict[str, Any],
    after: dict[str, Any],
    environment: dict[str, Any],
    *,
    inventory_inputs_match: bool = True,
    inventory_input_errors: list[str] | None = None,
) -> dict[str, Any]:
    before_digest = source_provenance_digest(before)
    after_digest = source_provenance_digest(after)
    available = all(
        item["available"] for item in before["local_path_repositories"]
    ) and all(item["available"] for item in after["local_path_repositories"])
    requirements_match = environment["python"]["requirements_match"]
    return {
        "before": before,
        "after": after,
        "before_sha256": before_digest,
        "after_sha256": after_digest,
        "unchanged_during_run": before_digest == after_digest,
        "local_path_repositories_available": available,
        "python_requirements_match": requirements_match,
        "inventory_inputs_match_at_start": inventory_inputs_match,
        "inventory_input_errors": inventory_input_errors or [],
        "passed": before_digest == after_digest
        and available
        and requirements_match
        and inventory_inputs_match,
    }


def verify_inventory_input_hashes(
    inventory: dict[str, Any],
    target_files: dict[str, dict[str, str | None]] | None = None,
) -> list[str]:
    errors: list[str] = []
    manifest_path = REPO_ROOT / inventory["manifest"]
    current_manifest_hash = sha256_file(manifest_path) if manifest_path.is_file() else None
    if current_manifest_hash != inventory["manifest_sha256"]:
        errors.append("QA manifest changed after inventory creation")

    for case in inventory["cases"]:
        for path_key, hash_key in (
            ("script", "script_sha256"),
            ("golden", "golden_sha256"),
            ("test_path", "test_sha256"),
        ):
            path = REPO_ROOT / case[path_key]
            current_hash = sha256_file(path) if path.is_file() else None
            if current_hash != case[hash_key]:
                errors.append(
                    f"QA case {case['id']} input changed after inventory: {case[path_key]}"
                )

    for target, info in (target_files or {}).items():
        path = REPO_ROOT / info["test_file"]
        current_hash = sha256_file(path) if path.is_file() else None
        if current_hash != info["test_file_sha256"]:
            errors.append(f"Rust target source changed before execution: {target}")
    return errors


def relative_path(path: pathlib.Path) -> str:
    return path.relative_to(REPO_ROOT).as_posix()


def qa_inventory(
    policy: dict[str, Any],
    packages: dict[str, dict[str, Any]],
    *,
    allow_missing_goldens: bool = False,
) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    qa = policy["numerical_qa"]
    manifest_path = REPO_ROOT / qa["manifest"]
    if not manifest_path.is_file():
        return {}, [f"QA manifest is missing: {qa['manifest']}"]
    manifest_bytes = manifest_path.read_bytes()
    manifest = tomllib.loads(manifest_bytes.decode("utf-8"))
    cases = manifest.get("case", [])
    expected_cases = qa["expected_case_count"]
    if len(cases) != expected_cases:
        errors.append(f"QA manifest case count: expected {expected_cases}, found {len(cases)}")

    ids = [case.get("id", "") for case in cases]
    if len(set(ids)) != len(ids) or any(not case_id for case_id in ids):
        errors.append("QA manifest case IDs must be present and unique")

    runner_counts = Counter(case.get("runner", "rust") for case in cases)
    expected_runner_counts = qa["expected_runner_counts"]
    if dict(sorted(runner_counts.items())) != dict(sorted(expected_runner_counts.items())):
        errors.append(
            "QA manifest runner counts differ: expected "
            f"{expected_runner_counts}, found {dict(sorted(runner_counts.items()))}"
        )
    unknown_runners = sorted(set(runner_counts) - {"rust", "python"})
    if unknown_runners:
        errors.append("QA manifest has unsupported runners: " + ", ".join(unknown_runners))

    qa_root = manifest_path.parent
    manifest_scripts = {case.get("script", "") for case in cases}
    actual_scripts = {path.name for path in (qa_root / "wolfram").glob("*.wls")}
    if manifest_scripts != actual_scripts:
        errors.append(
            "Wolfram script membership differs from manifest: "
            f"missing={sorted(manifest_scripts - actual_scripts)}, "
            f"unlisted={sorted(actual_scripts - manifest_scripts)}"
        )
    manifest_goldens = {case.get("golden", "") for case in cases}
    actual_goldens = {path.name for path in (qa_root / "wolfram" / "goldens").glob("*.json")}
    missing_goldens = sorted(manifest_goldens - actual_goldens)
    unlisted_goldens = sorted(actual_goldens - manifest_goldens)
    if unlisted_goldens or (missing_goldens and not allow_missing_goldens):
        errors.append(
            "Wolfram golden membership differs from manifest: "
            f"missing={missing_goldens}, unlisted={unlisted_goldens}"
        )

    rust_cases = [case for case in cases if case.get("runner", "rust") == "rust"]
    python_cases = [case for case in cases if case.get("runner", "rust") == "python"]
    if len(rust_cases) != expected_runner_counts.get("rust"):
        errors.append("Rust manifest case count does not match its expected runner count")
    if len(python_cases) != expected_runner_counts.get("python"):
        errors.append("Python manifest case count does not match its expected runner count")

    unique_test_names: set[str] = set()
    evidence_cases: list[dict[str, Any]] = []
    for case in cases:
        runner = case.get("runner", "rust")
        test_name = case.get("test", "")
        if not test_name or test_name in unique_test_names:
            errors.append(f"QA manifest test names must be present and unique: {test_name!r}")
        unique_test_names.add(test_name)
        if not case.get("family") or not case.get("owner"):
            errors.append(f"QA manifest case {case.get('id')} must declare family and owner")

        script_path = qa_root / "wolfram" / case.get("script", "")
        golden_path = qa_root / "wolfram" / "goldens" / case.get("golden", "")
        if not script_path.is_file():
            errors.append(f"QA case {case.get('id')} is missing script {relative_path(script_path)}")
        if not golden_path.is_file() and not allow_missing_goldens:
            errors.append(f"QA case {case.get('id')} is missing golden {relative_path(golden_path)}")

        if runner == "rust":
            test_path = qa_root / "tests" / f"{test_name}.rs"
            if not test_path.is_file():
                errors.append(f"QA case {case.get('id')} is missing Rust test {relative_path(test_path)}")
        elif runner == "python":
            test_path = qa_root / "tests" / f"{test_name}.py"
            if not test_path.is_file():
                errors.append(f"QA case {case.get('id')} is missing Python runner {relative_path(test_path)}")
        else:
            test_path = qa_root / "tests" / test_name

        item: dict[str, Any] = {
            "id": case.get("id"),
            "family": case.get("family"),
            "owner": case.get("owner"),
            "runner": runner,
            "test": test_name,
            "test_path": relative_path(test_path),
            "script": relative_path(script_path),
            "script_sha256": sha256_file(script_path) if script_path.is_file() else None,
            "golden": relative_path(golden_path),
            "golden_sha256": sha256_file(golden_path) if golden_path.is_file() else None,
            "test_sha256": sha256_file(test_path) if test_path.is_file() else None,
            "tolerance": case.get("tolerance"),
            "tolerance_kind": case.get("tolerance_kind"),
            "tolerance_value": case.get("tolerance_value"),
            "status": "not_run",
        }
        tolerance_value = item["tolerance_value"]
        if (
            not isinstance(tolerance_value, (int, float))
            or isinstance(tolerance_value, bool)
            or not math.isfinite(float(tolerance_value))
            or tolerance_value < 0
        ):
            errors.append(
                f"QA case {case.get('id')} must declare a finite non-negative tolerance_value"
            )
        if item["tolerance_kind"] not in {"abs", "rel"}:
            errors.append(
                f"QA case {case.get('id')} has unsupported tolerance_kind: "
                f"{item['tolerance_kind']!r}"
            )
        evidence_cases.append(item)

    package_name = qa["package"]
    package = packages.get(package_name)
    if package is None:
        errors.append(f"QA package is not a Cargo workspace member: {package_name}")
        test_targets: set[str] = set()
    else:
        test_targets = {
            target["name"]
            for target in package.get("targets", [])
            if "test" in target.get("kind", [])
        }
    rust_target_names = {case["test"] for case in rust_cases}
    expected_target_count = qa["expected_rust_test_target_count"]
    if len(test_targets) != expected_target_count:
        errors.append(
            f"Rust QA test-target count: expected {expected_target_count}, found {len(test_targets)}"
        )
    missing_targets = sorted(rust_target_names - test_targets)
    if missing_targets:
        errors.append("Rust manifest cases missing Cargo test targets: " + ", ".join(missing_targets))

    auxiliary_target_list = qa.get("additional_rust_test_targets", [])
    auxiliary_targets = set(auxiliary_target_list)
    if len(auxiliary_targets) != len(auxiliary_target_list):
        errors.append("declared auxiliary Rust QA targets must be unique")
    overlap = sorted(auxiliary_targets & rust_target_names)
    if overlap:
        errors.append("auxiliary Rust targets also appear in the manifest: " + ", ".join(overlap))
    undeclared_targets = sorted(test_targets - rust_target_names - auxiliary_targets)
    missing_auxiliary = sorted(auxiliary_targets - test_targets)
    if undeclared_targets:
        errors.append("unregistered Rust QA test targets: " + ", ".join(undeclared_targets))
    if missing_auxiliary:
        errors.append("declared auxiliary Rust QA targets are missing: " + ", ".join(missing_auxiliary))

    python_names = {case["test"] + ".py" for case in python_cases}
    auxiliary_python = set(qa.get("auxiliary_python_files", []))
    actual_python = {path.name for path in (qa_root / "tests").glob("*.py")}
    if actual_python != python_names | auxiliary_python:
        errors.append(
            "Python runner membership differs from manifest: "
            f"missing={sorted((python_names | auxiliary_python) - actual_python)}, "
            f"unlisted={sorted(actual_python - python_names - auxiliary_python)}"
        )

    inventory = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "manifest": relative_path(manifest_path),
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "expected_case_count": expected_cases,
        "case_count": len(cases),
        "missing_golden_count": len(missing_goldens),
        "expected_runner_counts": expected_runner_counts,
        "runner_counts": dict(sorted(runner_counts.items())),
        "expected_rust_test_target_count": expected_target_count,
        "rust_test_target_count": len(test_targets),
        "rust_manifest_target_count": len(rust_target_names),
        "python_runner_count": len(python_cases),
        "status": "valid" if not errors else "invalid",
        "cases": evidence_cases,
    }
    return inventory, errors


def evidence_path(policy: dict[str, Any], filename: str) -> pathlib.Path:
    path = REPO_ROOT / policy["numerical_qa"]["evidence_directory"] / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def write_evidence(path: pathlib.Path, report: dict[str, Any]) -> None:
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"QA evidence: {relative_path(path)}")


def print_errors(errors: list[str]) -> None:
    for error in errors:
        print(f"- {error}", file=sys.stderr)


def parse_python_qa_result(
    case: dict[str, Any], stdout: str, return_code: int
) -> tuple[str, dict[str, Any] | None]:
    """Require exactly one passing QA_RESULT for the expected manifest case."""
    records: list[dict[str, Any]] = []
    marker_count = 0
    malformed_count = 0
    for line in stdout.splitlines():
        if "QA_RESULT" in line:
            marker_count += 1
        try:
            value = json.loads(line, object_pairs_hook=unique_json_object)
        except (json.JSONDecodeError, ValueError):
            if "QA_RESULT" in line:
                malformed_count += 1
            continue
        if isinstance(value, dict) and "QA_RESULT" in value:
            records.append(value)
    if return_code != 0 or len(records) != 1 or marker_count != 1 or malformed_count:
        return "failed", records[0] if len(records) == 1 else None
    result = records[0]
    if (
        result.get("QA_RESULT") is not True
        or result.get("case") != case["id"]
        or result.get("pass") is not True
        or result.get("tolerance_kind") != case["tolerance_kind"]
        or not qa_result_values_match(case, result)
    ):
        return "failed", result
    return "passed", result


def _finite_nonnegative_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and value >= 0
    )


def qa_result_values_match(case: dict[str, Any], result: dict[str, Any]) -> bool:
    """Check numerical evidence against the manifest's machine-readable contract."""
    expected_tolerance = case.get("tolerance_value")
    tolerance = result.get("tolerance")
    max_abs_error = result.get("max_abs_error")
    max_rel_error = result.get("max_rel_error")
    if not all(
        _finite_nonnegative_number(value)
        for value in (expected_tolerance, tolerance, max_abs_error, max_rel_error)
    ):
        return False
    if not math.isclose(
        float(tolerance), float(expected_tolerance), rel_tol=1e-12, abs_tol=0.0
    ):
        return False
    observed_error = max_rel_error if case["tolerance_kind"] == "rel" else max_abs_error
    if observed_error > expected_tolerance:
        return False
    provenance = result.get("provenance")
    return isinstance(provenance, str) and bool(provenance.strip())


def parse_rust_qa_records(output: str) -> tuple[list[dict[str, Any]], list[str]]:
    """Read QA_RESULT records even when libtest text precedes the marker."""
    records: list[dict[str, Any]] = []
    errors: list[str] = []
    current_target: str | None = None
    for line_number, line in enumerate(output.splitlines(), start=1):
        running = RUNNING_TEST_TARGET.search(line)
        if running:
            current_target = pathlib.Path(running.group(1)).stem
        marker = line.find(QA_RESULT_PREFIX)
        if marker >= 0:
            encoded = line[marker + len(QA_RESULT_PREFIX) :].strip()
            try:
                record = json.loads(encoded, object_pairs_hook=unique_json_object)
            except (json.JSONDecodeError, ValueError) as error:
                errors.append(f"line {line_number}: malformed QA_RESULT JSON: {error}")
            else:
                if not isinstance(record, dict):
                    errors.append(f"line {line_number}: QA_RESULT must be a JSON object")
                else:
                    records.append({**record, "_cargo_target": current_target})
        if TEST_RESULT.search(line):
            current_target = None
    return records, errors


def validate_rust_qa_records(
    cases: list[dict[str, Any]],
    records: list[dict[str, Any]],
    malformed: list[str],
    expected_provenance: str | None = None,
) -> tuple[dict[str, dict[str, Any]], list[str]]:
    """Require exactly one valid numerical record for every Rust manifest case."""
    if expected_provenance is None:
        expected_provenance = (
            "live-engine" if "WOLFRAMSCRIPT" in os.environ else "checked-in-golden"
        )
    by_id: dict[str, list[dict[str, Any]]] = {}
    errors = list(malformed)
    expected_ids = {case["id"] for case in cases}
    for record in records:
        case_id = record.get("case")
        if not isinstance(case_id, str):
            errors.append("QA_RESULT has no string case id")
            continue
        by_id.setdefault(case_id, []).append(record)
        if case_id not in expected_ids:
            errors.append(f"unexpected Rust QA_RESULT case id: {case_id}")

    validated: dict[str, dict[str, Any]] = {}
    for case in cases:
        case_id = case["id"]
        matches = by_id.get(case_id, [])
        case_errors: list[str] = []
        if len(matches) != 1:
            case_errors.append(
                f"expected exactly one QA_RESULT for {case_id}, found {len(matches)}"
            )
        elif not rust_qa_record_matches(case, matches[0], expected_provenance):
            case_errors.append(
                f"QA_RESULT for {case_id} failed target, identity, pass, tolerance, error, or provenance checks"
            )
        if case_errors:
            errors.extend(case_errors)
        validated[case_id] = {
            "status": "passed" if not case_errors else "failed",
            "record": matches[0] if matches else None,
            "records": matches,
            "record_count": len(matches),
            "errors": case_errors,
        }
    return validated, errors


def rust_qa_record_matches(
    case: dict[str, Any],
    result: dict[str, Any],
    expected_provenance: str | None = None,
) -> bool:
    if expected_provenance is None:
        expected_provenance = (
            "live-engine" if "WOLFRAMSCRIPT" in os.environ else "checked-in-golden"
        )
    return (
        result.get("_cargo_target") == case["test"]
        and result.get("case") == case["id"]
        and result.get("pass") is True
        and result.get("tolerance_kind") == case["tolerance_kind"]
        and result.get("provenance") == expected_provenance
        and qa_result_values_match(case, result)
    )


def run_python_cases(
    inventory: dict[str, Any],
    policy: dict[str, Any],
) -> int:
    input_hash_errors = verify_inventory_input_hashes(inventory)
    provenance_before = capture_source_provenance()
    environment_provenance = run_environment_provenance()
    results: list[dict[str, Any]] = []
    for case in inventory["cases"]:
        if case["runner"] != "python":
            continue
        command = [sys.executable, str(REPO_ROOT / case["test_path"])]
        print("--- " + case["test_path"], flush=True)
        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.stdout:
            print(completed.stdout, end="" if completed.stdout.endswith("\n") else "\n")
        if completed.stderr:
            print(completed.stderr, file=sys.stderr, end="" if completed.stderr.endswith("\n") else "\n")

        status, qa_result = parse_python_qa_result(
            case, completed.stdout, completed.returncode
        )
        results.append({
            "id": case["id"],
            "family": case["family"],
            "owner": case["owner"],
            "test": case["test"],
            "test_path": case["test_path"],
            "test_sha256": case["test_sha256"],
            "status": status,
            "return_code": completed.returncode,
            "qa_result": qa_result,
            "script_sha256": case["script_sha256"],
            "golden_sha256": case["golden_sha256"],
            "tolerance": case["tolerance"],
            "tolerance_kind": case["tolerance_kind"],
            "tolerance_value": case["tolerance_value"],
        })

    expected = policy["numerical_qa"]["expected_runner_counts"]["python"]
    executed = len(results)
    passed = sum(result["status"] == "passed" for result in results)
    failed = executed - passed
    provenance_after = capture_source_provenance()
    provenance = provenance_report(
        provenance_before,
        provenance_after,
        environment_provenance,
        inventory_inputs_match=not input_hash_errors,
        inventory_input_errors=input_hash_errors,
    )
    report = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "runner": "python",
        "manifest_sha256": inventory["manifest_sha256"],
        "environment": environment_provenance,
        "source_provenance": provenance,
        "expected_case_count": expected,
        "executed_case_count": executed,
        "passed_case_count": passed,
        "failed_case_count": failed,
        "status": "passed"
        if executed == expected and failed == 0 and provenance["passed"]
        else "failed",
        "cases": results,
    }
    write_evidence(evidence_path(policy, "autoeq-qa-python-results.json"), report)
    print(f"Python comparisons: {passed}/{expected} passed; {failed} failed")
    if not provenance["passed"]:
        print("Python QA provenance gate FAILED", file=sys.stderr)
    return 0 if report["status"] == "passed" else 1


def parse_rust_case_results(output: str) -> dict[str, dict[str, int]]:
    current_target: str | None = None
    results: dict[str, dict[str, int]] = {}
    for line in output.splitlines():
        running = RUNNING_TEST_TARGET.search(line)
        if running:
            current_target = pathlib.Path(running.group(1)).stem
            continue
        summary = TEST_RESULT.search(line)
        if summary and current_target is not None:
            result = {
                "passed": int(summary.group(1)),
                "failed": int(summary.group(2)),
                "ignored": int(summary.group(3)),
            }
            if current_target in results:
                # A repeated target summary is ambiguous; preserve it so the
                # coverage gate reports the duplicate instead of hiding it.
                results[current_target]["summary_count"] += 1
            else:
                result["summary_count"] = 1
                results[current_target] = result
            current_target = None
    return results


def rust_target_reports(
    target_names: list[str],
    results: dict[str, dict[str, int]],
    target_files: dict[str, dict[str, str | None]] | None = None,
) -> list[dict[str, Any]]:
    """Summarize every expected Cargo target, rejecting missing and inert runs."""
    reports: list[dict[str, Any]] = []
    for name in target_names:
        result = results.get(name)
        active = 0 if result is None else result["passed"] + result["failed"]
        status = "not_run"
        if result is not None and active > 0:
            status = (
                "passed"
                if result["failed"] == 0 and result.get("summary_count") == 1
                else "failed"
            )
        reports.append({
            "target": name,
            **(target_files or {}).get(name, {}),
            "status": status,
            "reported": result is not None,
            "executed": active > 0,
            "test_result": result,
        })
    return reports


def rust_target_coverage_satisfied(
    reports: list[dict[str, Any]], expected_count: int
) -> bool:
    return len(reports) == expected_count and all(
        report["status"] == "passed" for report in reports
    )


def rust_case_status(
    target_status: str, qa_status: str, qa_record_count: int
) -> str:
    if target_status == "passed" and qa_status == "passed":
        return "passed"
    if target_status != "not_run" or qa_record_count > 0:
        return "failed"
    return "not_run"


def run_rust_cases(
    inventory: dict[str, Any],
    packages: dict[str, dict[str, Any]],
    policy: dict[str, Any],
    *,
    release: bool,
) -> int:
    qa = policy["numerical_qa"]
    package = packages[qa["package"]]
    test_target_metadata = [
        target
        for target in package.get("targets", [])
        if "test" in target.get("kind", [])
    ]
    test_targets = sorted(target["name"] for target in test_target_metadata)
    target_files = {
        target["name"]: {
            "test_file": relative_path(pathlib.Path(target["src_path"])),
            "test_file_sha256": sha256_file(pathlib.Path(target["src_path"]))
            if pathlib.Path(target["src_path"]).is_file()
            else None,
        }
        for target in test_target_metadata
    }
    command = [
        cargo_program(),
        "test",
        "-p",
        qa["package"],
        "--locked",
        "--no-fail-fast",
        "--lib",
    ]
    if release:
        command.append("--release")
    for target in test_targets:
        command.extend(["--test", target])
    command.extend(["--", "--nocapture"])

    print("Running all registered Rust QA targets:", shlex.join(command), flush=True)
    input_hash_errors = verify_inventory_input_hashes(inventory, target_files)
    provenance_before = capture_source_provenance()
    environment_provenance = run_environment_provenance()
    environment = os.environ.copy()
    environment["CARGO_TERM_COLOR"] = "never"
    process = subprocess.Popen(
        command,
        cwd=REPO_ROOT,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    output_lines: list[str] = []
    log_path = evidence_path(policy, "autoeq-qa-rust-run.log")
    assert process.stdout is not None
    with log_path.open("w", encoding="utf-8") as log_file:
        for line in process.stdout:
            output_lines.append(line)
            log_file.write(line)
            log_file.flush()
            print(line, end="", flush=True)
    return_code = process.wait()
    provenance_after = capture_source_provenance()
    provenance = provenance_report(
        provenance_before,
        provenance_after,
        environment_provenance,
        inventory_inputs_match=not input_hash_errors,
        inventory_input_errors=input_hash_errors,
    )
    target_results = parse_rust_case_results("".join(output_lines))
    rust_cases = [case for case in inventory["cases"] if case["runner"] == "rust"]
    qa_records, malformed_qa_records = parse_rust_qa_records("".join(output_lines))
    expected_provenance = (
        "live-engine" if "WOLFRAMSCRIPT" in environment else "checked-in-golden"
    )
    qa_validation, qa_validation_errors = validate_rust_qa_records(
        rust_cases, qa_records, malformed_qa_records, expected_provenance
    )

    target_reports = rust_target_reports(test_targets, target_results, target_files)
    reports_by_name = {report["target"]: report for report in target_reports}
    case_results: list[dict[str, Any]] = []
    for case in inventory["cases"]:
        if case["runner"] != "rust":
            continue
        target_report = reports_by_name[case["test"]]
        qa_result = qa_validation[case["id"]]
        case_status = rust_case_status(
            target_report["status"], qa_result["status"], qa_result["record_count"]
        )
        case_results.append({
            "id": case["id"],
            "family": case["family"],
            "owner": case["owner"],
            "test": case["test"],
            "test_path": case["test_path"],
            "test_sha256": case["test_sha256"],
            "status": case_status,
            "test_result": target_report["test_result"],
            "qa_result": qa_result["record"],
            "qa_result_records": qa_result["records"],
            "qa_result_record_count": qa_result["record_count"],
            "qa_result_errors": qa_result["errors"],
            "script_sha256": case["script_sha256"],
            "golden_sha256": case["golden_sha256"],
            "tolerance": case["tolerance"],
            "tolerance_kind": case["tolerance_kind"],
            "tolerance_value": case["tolerance_value"],
        })

    expected = qa["expected_runner_counts"]["rust"]
    executed = sum(result["status"] != "not_run" for result in case_results)
    passed = sum(result["status"] == "passed" for result in case_results)
    failed = sum(result["status"] == "failed" for result in case_results)
    not_run = sum(result["status"] == "not_run" for result in case_results)
    target_count_expected = qa["expected_rust_test_target_count"]
    target_count_reported = sum(item["reported"] for item in target_reports)
    target_count_executed = sum(item["executed"] for item in target_reports)
    target_count_passed = sum(item["status"] == "passed" for item in target_reports)
    target_count_failed = sum(item["status"] == "failed" for item in target_reports)
    report = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "runner": "rust",
        "command": command,
        "log_file": relative_path(log_path),
        "manifest_sha256": inventory["manifest_sha256"],
        "environment": environment_provenance,
        "source_provenance": provenance,
        "expected_case_count": expected,
        "executed_case_count": executed,
        "passed_case_count": passed,
        "failed_case_count": failed,
        "not_run_case_count": not_run,
        "expected_qa_result_count": expected,
        "observed_qa_result_count": len(qa_records),
        "malformed_qa_result_count": len(malformed_qa_records),
        "qa_result_records": qa_records,
        "qa_result_errors": qa_validation_errors,
        "expected_test_target_count": target_count_expected,
        "reported_test_target_count": target_count_reported,
        "executed_test_target_count": target_count_executed,
        "passed_test_target_count": target_count_passed,
        "failed_test_target_count": target_count_failed,
        "test_targets": target_reports,
        "cargo_return_code": return_code,
        "status": "passed"
        if return_code == 0 and provenance["passed"] and executed == expected and passed == expected and len(qa_records) == expected and not qa_validation_errors and target_count_executed == target_count_expected
        and target_count_passed == target_count_expected
        and rust_target_coverage_satisfied(target_reports, target_count_expected)
        else "failed",
        "cases": case_results,
    }
    write_evidence(evidence_path(policy, "autoeq-qa-rust-results.json"), report)
    print(f"Rust QA log: {relative_path(log_path)}")
    print(
        f"Rust comparisons: {passed}/{expected} passed; {failed} failed; "
        f"{report['not_run_case_count']} not run; "
        f"{target_count_executed}/{target_count_expected} targets executed and "
        f"{target_count_passed} passed, {target_count_failed} failed"
    )
    if not provenance["passed"]:
        print("Rust QA provenance gate FAILED", file=sys.stderr)
    return 0 if report["status"] == "passed" else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=pathlib.Path, default=DEFAULT_POLICY)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--list-json", action="store_true")
    action.add_argument("--package")
    action.add_argument("--check-qa-manifest", action="store_true")
    action.add_argument("--qa-runner", choices=("rust", "python"))
    parser.add_argument("--allow-missing-goldens", action="store_true")
    parser.add_argument("--release", action="store_true")
    arguments = parser.parse_args()
    if arguments.allow_missing_goldens and not arguments.check_qa_manifest:
        parser.error("--allow-missing-goldens is only valid with --check-qa-manifest")

    policy_path = arguments.policy.resolve()
    policy = load_policy(policy_path)
    try:
        metadata = cargo_metadata()
        packages = workspace_packages(metadata)
    except (RuntimeError, ValueError, json.JSONDecodeError) as error:
        print(str(error), file=sys.stderr)
        return 1

    package_errors = validate_package_matrix(policy, packages)
    if arguments.list_json or arguments.package:
        if package_errors:
            print_errors(package_errors)
            return 1
        if arguments.list_json:
            print(json.dumps(sorted(packages)))
            return 0
        try:
            command = command_for_package(
                policy["focused_tests"], arguments.package, release=arguments.release
            )
        except ValueError as error:
            parser.error(str(error))
        command[0] = cargo_program()
        return subprocess.run(command, cwd=REPO_ROOT, check=False).returncode

    inventory, qa_errors = qa_inventory(
        policy, packages, allow_missing_goldens=arguments.allow_missing_goldens
    )
    errors = package_errors + qa_errors
    if errors:
        print_errors(errors)
        if inventory:
            inventory["status"] = "invalid"
            inventory["errors"] = errors
            write_evidence(evidence_path(policy, "autoeq-qa-manifest.json"), inventory)
        return 1

    if arguments.check_qa_manifest:
        write_evidence(evidence_path(policy, "autoeq-qa-manifest.json"), inventory)
        print(
            "QA manifest PASS: "
            f"{inventory['case_count']} cases, "
            f"{inventory['runner_counts'].get('rust', 0)} Rust, "
            f"{inventory['runner_counts'].get('python', 0)} Python, "
            f"{inventory['rust_test_target_count']} Rust test targets, "
            f"{inventory['missing_golden_count']} missing goldens"
        )
        return 0

    if arguments.qa_runner == "python":
        return run_python_cases(inventory, policy)
    return run_rust_cases(inventory, packages, policy, release=arguments.release)


if __name__ == "__main__":
    raise SystemExit(main())
