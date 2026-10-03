#!/usr/bin/env python3
"""Enforce the crate-partition dependency graph and migration ratchets."""

from __future__ import annotations

import argparse
import difflib
import hashlib
import io
import json
import os
import pathlib
import re
import shlex
import subprocess
import sys
import tarfile
from collections import defaultdict
from typing import Any


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_POLICY = REPO_ROOT / "scripts" / "crate_partition_policy.json"
TEST_ATTRIBUTE = re.compile(
    r"#\s*\[\s*(?:[A-Za-z_][A-Za-z0-9_]*::)?test"
    r"(?:\s*\([^]]*\))?\s*\]"
)
UNSAFE_RUST = re.compile(
    r"\bunsafe\s*(?:\{|fn\b|impl\b|trait\b|extern\b)|"
    r"#\s*\[\s*unsafe\s*\("
)
ENVIRONMENT_MUTATION = re.compile(
    r"\b(?:std\s*::\s*)?env\s*::\s*(?:set_var|remove_var)\s*\("
)
NDARRAY_SLICE_MACRO = re.compile(r"\b(?:ndarray\s*::\s*)?s\s*!\s*\[")


def unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def load_json(path: pathlib.Path) -> dict[str, Any]:
    return json.loads(
        path.read_text(encoding="utf-8"), object_pairs_hook=unique_json_object
    )


def cargo_metadata(repo_root: pathlib.Path) -> dict[str, Any]:
    command = [
        os.environ.get("CARGO", "cargo"),
        "metadata",
        "--format-version",
        "1",
        "--no-deps",
        "--locked",
    ]
    completed = subprocess.run(
        command,
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode:
        raise RuntimeError(
            "cargo metadata failed:\n" + completed.stderr.rstrip()
        )
    return json.loads(completed.stdout)


def workspace_packages(metadata: dict[str, Any]) -> dict[str, dict[str, Any]]:
    member_ids = set(metadata["workspace_members"])
    packages: dict[str, dict[str, Any]] = {}
    for package in metadata["packages"]:
        if package["id"] not in member_ids:
            continue
        name = package["name"]
        if name in packages:
            raise ValueError(f"duplicate workspace package name: {name}")
        packages[name] = package
    return packages


WorkspaceEdge = tuple[str, str, str]


def workspace_edges(
    packages: dict[str, dict[str, Any]],
) -> set[WorkspaceEdge]:
    package_names = set(packages)
    return {
        (package_name, dependency["name"], dependency.get("kind") or "normal")
        for package_name, package in packages.items()
        for dependency in package["dependencies"]
        if dependency["name"] in package_names
    }


def dependency_cycles(
    package_names: set[str],
    edges: set[WorkspaceEdge],
    kinds: frozenset[str] = frozenset({"normal", "build"}),
) -> list[list[str]]:
    graph = {name: set() for name in package_names}
    for source, destination, kind in edges:
        if kind in kinds:
            graph[source].add(destination)

    state = {name: 0 for name in package_names}
    stack: list[str] = []
    cycles: list[list[str]] = []

    def visit(node: str) -> None:
        state[node] = 1
        stack.append(node)
        for dependency in sorted(graph[node]):
            if state[dependency] == 0:
                visit(dependency)
            elif state[dependency] == 1:
                start = stack.index(dependency)
                cycle = stack[start:] + [dependency]
                if cycle not in cycles:
                    cycles.append(cycle)
        stack.pop()
        state[node] = 2

    for package_name in sorted(package_names):
        if state[package_name] == 0:
            visit(package_name)
    return cycles


def exception_pairs(policy: dict[str, Any]) -> set[WorkspaceEdge]:
    return {
        (
            exception["from"],
            exception["to"],
            exception.get("kind", "normal"),
        )
        for exception in policy["temporary_exceptions"]
    }


def check_dependency_policy(
    packages: dict[str, dict[str, Any]], policy: dict[str, Any]
) -> tuple[set[WorkspaceEdge], list[str]]:
    errors: list[str] = []
    edges = workspace_edges(packages)
    root_package = policy["root_package"]
    terminal_consumers = set(policy["terminal_consumers"])
    allowed_by_kind = {
        "normal": policy["allowed_direct_dependencies"],
        "dev": policy.get("allowed_dev_dependencies", {}),
        "build": policy.get("allowed_build_dependencies", {}),
    }
    allowed_by_kind = {
        kind: {
            package: set(dependencies)
            for package, dependencies in package_dependencies.items()
        }
        for kind, package_dependencies in allowed_by_kind.items()
    }
    policy_packages = set().union(
        *(set(package_dependencies) for package_dependencies in allowed_by_kind.values())
    )
    expected_package_count = policy.get("workspace_package_count")
    if expected_package_count is not None and len(packages) != expected_package_count:
        errors.append(
            f"workspace package count changed: {len(packages)}, expected {expected_package_count}"
        )

    missing_policy = set(packages) - policy_packages - terminal_consumers
    for package_name in sorted(missing_policy):
        errors.append(f"workspace package is missing from policy: {package_name}")
    unknown_policy_packages = policy_packages | terminal_consumers
    for package_name in sorted(unknown_policy_packages - set(packages)):
        errors.append(f"dependency policy names absent package: {package_name}")

    exceptions = policy["temporary_exceptions"]
    temporary_edges = exception_pairs(policy)
    if len(temporary_edges) != len(exceptions):
        errors.append("temporary exception edges must be unique")

    for exception in exceptions:
        kind = exception.get("kind", "normal")
        edge = (exception.get("from", ""), exception.get("to", ""), kind)
        if kind not in allowed_by_kind:
            errors.append(
                f"temporary exception {edge[0]} -> {edge[1]} has invalid dependency kind {kind!r}"
            )
        if not re.fullmatch(r"WP(?:[1-9]|1[01])", exception.get("remove_by", "")):
            errors.append(
                f"temporary exception {edge[0]} -> {edge[1]} has no valid remove_by WP"
            )
        if not exception.get("reason", "").strip():
            errors.append(
                f"temporary exception {edge[0]} -> {edge[1]} has no reason"
            )
        if edge not in edges:
            errors.append(
                f"stale temporary exception must be removed: {edge[0]} -> {edge[1]}"
            )
        if kind in allowed_by_kind and edge[1] in allowed_by_kind[kind].get(edge[0], set()):
            errors.append(
                f"temporary exception is already allowed: {edge[0]} -> {edge[1]}"
            )

    for source, destination, kind in sorted(edges):
        if source != root_package and destination == root_package:
            errors.append(
                f"workspace crate depends on root facade: {source} -> {destination}"
            )
            continue
        if source in terminal_consumers:
            continue
        if destination in allowed_by_kind.get(kind, {}).get(source, set()):
            continue
        if (source, destination, kind) in temporary_edges:
            continue
        errors.append(
            f"forbidden {kind} workspace edge: {source} -> {destination}"
        )

    for cycle in dependency_cycles(set(packages), edges):
        errors.append(
            "workspace dependency cycle (normal/build): " + " -> ".join(cycle)
        )

    required_external = policy.get("required_external_dependencies", {})
    for package_name, required_by_kind in required_external.items():
        package = packages.get(package_name)
        if package is None:
            errors.append(
                f"external dependency policy names absent package: {package_name}"
            )
            continue
        actual_external = {
            (dependency["name"], dependency.get("kind") or "normal")
            for dependency in package["dependencies"]
            if dependency["name"] not in packages
        }
        for kind, dependency_names in required_by_kind.items():
            if kind not in allowed_by_kind:
                errors.append(
                    f"external dependency policy for {package_name} has invalid kind {kind!r}"
                )
                continue
            for dependency_name in dependency_names:
                if (dependency_name, kind) not in actual_external:
                    errors.append(
                        f"required external dependency is missing: {package_name} -> {dependency_name} ({kind})"
                    )
    return edges, errors


def rust_files(path: pathlib.Path) -> list[pathlib.Path]:
    if not path.exists():
        return []
    return sorted(path.rglob("*.rs"))


def rust_line_count(path: pathlib.Path) -> int:
    return sum(
        len(file_path.read_text(encoding="utf-8", errors="replace").splitlines())
        for file_path in rust_files(path)
    )


def rust_test_count(path: pathlib.Path) -> int:
    return sum(
        len(TEST_ATTRIBUTE.findall(
            file_path.read_text(encoding="utf-8", errors="replace")
        ))
        for file_path in rust_files(path)
    )


def root_metrics(repo_root: pathlib.Path) -> dict[str, int]:
    source = repo_root / "src"
    return {
        "root_rust_loc": rust_line_count(source),
        "root_roomeq_rust_loc": rust_line_count(source / "roomeq"),
        "root_binary_rust_loc": rust_line_count(source / "bin"),
        "root_unit_tests": rust_test_count(source),
    }


def check_metric_budgets(
    metrics: dict[str, int], policy: dict[str, Any]
) -> list[str]:
    errors: list[str] = []
    for name, budget in policy["metric_budgets"].items():
        value = metrics[name]
        if value > budget:
            errors.append(f"metric increased: {name}={value}, budget={budget}")
    return errors


def normalized_source_fingerprint(path: pathlib.Path) -> str | None:
    source = path.read_text(encoding="utf-8", errors="replace")
    normalized = re.sub(r"\s+", "", source)
    if len(normalized) < 200:
        return None
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def duplicate_source_ownership(repo_root: pathlib.Path) -> list[list[pathlib.Path]]:
    root_files = rust_files(repo_root / "src")
    crate_files = sorted((repo_root / "crates").glob("*/src/**/*.rs"))
    groups: dict[str, list[pathlib.Path]] = defaultdict(list)
    for file_path in root_files + crate_files:
        fingerprint = normalized_source_fingerprint(file_path)
        if fingerprint:
            groups[fingerprint].append(file_path)

    duplicates: list[list[pathlib.Path]] = []
    root_set = set(root_files)
    crate_set = set(crate_files)
    for paths in groups.values():
        if len(paths) < 2:
            continue
        if root_set.intersection(paths) and crate_set.intersection(paths):
            duplicates.append(sorted(paths))
    return sorted(duplicates, key=lambda paths: str(paths[0]))


def extract_root_public_api(source: str) -> list[str]:
    """Extract the root facade declarations that control compatibility paths."""
    lines = source.splitlines()
    declarations: list[str] = []
    macro_export = False
    index = 0
    while index < len(lines):
        stripped = lines[index].strip()
        if stripped == "#[macro_export]":
            macro_export = True
        elif macro_export:
            match = re.search(r"macro_rules!\s+([A-Za-z_][A-Za-z0-9_]*)", stripped)
            if match:
                declarations.append(f"macro {match.group(1)}")
                macro_export = False
        if stripped.startswith(("pub use ", "pub mod ", "pub extern crate ")):
            parts = [stripped]
            while ";" not in parts[-1]:
                index += 1
                if index >= len(lines):
                    raise ValueError("unterminated public facade declaration")
                parts.append(lines[index].strip())
            declarations.append(re.sub(r"\s+", " ", " ".join(parts)).strip())
        index += 1
    return declarations


def baseline_lines(path: pathlib.Path) -> list[str]:
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def check_public_api(repo_root: pathlib.Path, policy: dict[str, Any]) -> list[str]:
    source_path = repo_root / policy["public_api"]["source"]
    baseline_path = repo_root / policy["public_api"]["baseline"]
    actual = extract_root_public_api(source_path.read_text(encoding="utf-8"))
    expected = baseline_lines(baseline_path)
    if actual == expected:
        return []
    diff = "\n".join(
        difflib.unified_diff(
            expected,
            actual,
            fromfile=str(baseline_path.relative_to(repo_root)),
            tofile=str(source_path.relative_to(repo_root)),
            lineterm="",
        )
    )
    return ["root public facade changed without updating its baseline:\n" + diff]


def check_schema_baseline_files(
    repo_root: pathlib.Path, policy: dict[str, Any]
) -> list[str]:
    errors: list[str] = []
    for schema_kind, relative_path in policy["schema_baselines"].items():
        path = repo_root / relative_path
        try:
            json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            errors.append(f"invalid {schema_kind} schema baseline {relative_path}: {error}")
    return errors


def check_focused_tests(
    packages: dict[str, dict[str, Any]], policy: dict[str, Any]
) -> list[str]:
    errors: list[str] = []
    focused_tests = policy["focused_tests"]
    missing = set(packages) - set(focused_tests)
    extra = set(focused_tests) - set(packages)
    for package_name in sorted(missing):
        errors.append(f"focused test command missing for {package_name}")
    for package_name in sorted(extra):
        errors.append(f"focused test command names absent package {package_name}")
    for package_name, command in focused_tests.items():
        arguments = shlex.split(command)
        expected = ["cargo", "test", "-p", package_name]
        if arguments[:4] != expected:
            errors.append(
                f"focused test command for {package_name} must start with "
                + " ".join(expected)
            )
        if "--locked" not in arguments:
            errors.append(
                f"focused test command for {package_name} must include --locked"
            )
        package = packages.get(package_name)
        if package is not None:
            features: set[str] = set()
            for index, argument in enumerate(arguments):
                if argument in ("--features", "-F") and index + 1 < len(arguments):
                    features.update(arguments[index + 1].split(","))
                elif argument.startswith("--features="):
                    features.update(argument.split("=", 1)[1].split(","))
                elif argument.startswith("-F") and argument != "-F":
                    features.update(argument[2:].split(","))
            if "--all-features" not in arguments:
                unknown_features = features - set(package["features"])
                for feature in sorted(unknown_features):
                    errors.append(
                        f"focused test command for {package_name} names unknown feature {feature}"
                    )
            if "--lib" in arguments and not any(
                "lib" in target["kind"]
                or any(
                    crate_type in {"lib", "rlib", "cdylib"}
                    for crate_type in target.get("crate_types", [])
                )
                for target in package["targets"]
            ):
                errors.append(
                    f"focused test command for {package_name} selects a missing library target"
                )
    return errors


def check_crate_documentation(
    packages: dict[str, dict[str, Any]],
) -> list[str]:
    errors: list[str] = []
    for package_name, package in sorted(packages.items()):
        package_dir = pathlib.Path(package["manifest_path"]).parent
        for file_name in ("README.md", "CHANGELOG.md"):
            path = package_dir / file_name
            if not path.is_file():
                errors.append(
                    f"workspace package {package_name} is missing {file_name}"
                )
            elif not path.read_text(encoding="utf-8").strip():
                errors.append(
                    f"workspace package {package_name} has empty {file_name}"
                )
    return errors


def workspace_owned_rust_files(
    packages: dict[str, dict[str, Any]],
) -> list[pathlib.Path]:
    files: set[pathlib.Path] = set()
    for package in packages.values():
        package_dir = pathlib.Path(package["manifest_path"]).parent
        for directory in ("src", "tests", "examples", "benches"):
            files.update(rust_files(package_dir / directory))
        build_script = package_dir / "build.rs"
        if build_script.is_file():
            files.add(build_script)
    return sorted(files)


def toml_table(source: str, name: str) -> str | None:
    match = re.search(
        rf"^\[{re.escape(name)}\]\s*$\n(.*?)(?=^\[|\Z)",
        source,
        flags=re.MULTILINE | re.DOTALL,
    )
    return match.group(1) if match else None


def check_workspace_safety(
    repo_root: pathlib.Path, packages: dict[str, dict[str, Any]]
) -> list[str]:
    errors: list[str] = []
    root_manifest = (repo_root / "Cargo.toml").read_text(encoding="utf-8")
    rust_lints = toml_table(root_manifest, "workspace.lints.rust") or ""
    if not re.search(
        r'^unsafe_code\s*=\s*(?:"forbid"|\{[^}]*\blevel\s*=\s*"forbid"[^}]*\})\s*$',
        rust_lints,
        flags=re.MULTILINE,
    ):
        errors.append("workspace Rust lint unsafe_code must be set to forbid")

    for package_name, package in sorted(packages.items()):
        manifest_path = pathlib.Path(package["manifest_path"])
        manifest = manifest_path.read_text(encoding="utf-8")
        package_lints = toml_table(manifest, "lints") or ""
        if not re.search(
            r"^workspace\s*=\s*true\s*$", package_lints, flags=re.MULTILINE
        ):
            errors.append(
                f"workspace package {package_name} does not inherit workspace lints"
            )

    patterns = (
        ("unsafe Rust syntax", UNSAFE_RUST),
        ("process-environment mutation", ENVIRONMENT_MUTATION),
        ("unsafe-expanding ndarray slice macro", NDARRAY_SLICE_MACRO),
    )
    for file_path in workspace_owned_rust_files(packages):
        source = file_path.read_text(encoding="utf-8", errors="replace")
        for description, pattern in patterns:
            for match in pattern.finditer(source):
                line = source.count("\n", 0, match.start()) + 1
                relative = file_path.relative_to(repo_root)
                errors.append(f"{description}: {relative}:{line}")
    return errors


def package_metrics(
    packages: dict[str, dict[str, Any]], policy: dict[str, Any]
) -> list[tuple[str, int, int, str]]:
    rows: list[tuple[str, int, int, str]] = []
    for package_name, package in sorted(packages.items()):
        source = pathlib.Path(package["manifest_path"]).parent / "src"
        rows.append(
            (
                package_name,
                rust_line_count(source),
                rust_test_count(source),
                policy["focused_tests"].get(package_name, "MISSING"),
            )
        )
    return rows


def policy_from_git(
    repo_root: pathlib.Path, policy_path: pathlib.Path, reference: str
) -> tuple[dict[str, Any] | None, str | None]:
    if not reference:
        return None, None
    verified = subprocess.run(
        ["git", "rev-parse", "--verify", f"{reference}^{{commit}}"],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    if verified.returncode:
        return None, f"cannot resolve baseline ref {reference!r}"
    relative_path = policy_path.relative_to(repo_root).as_posix()
    exists = subprocess.run(
        ["git", "cat-file", "-e", f"{reference}:{relative_path}"],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    if exists.returncode:
        return None, None
    shown = subprocess.run(
        ["git", "show", f"{reference}:{relative_path}"],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    if shown.returncode:
        return None, f"cannot read policy from baseline ref {reference!r}"
    return json.loads(shown.stdout, object_pairs_hook=unique_json_object), None


def check_monotonic_ratchets(
    current: dict[str, Any],
    baseline: dict[str, Any],
    baseline_source_metrics: dict[str, int],
) -> list[str]:
    errors: list[str] = []
    added_exceptions = exception_pairs(current) - exception_pairs(baseline)
    for source, destination, kind in sorted(added_exceptions):
        kind_suffix = "" if kind == "normal" else f" ({kind})"
        errors.append(
            f"temporary exception list may only shrink: added {source} -> {destination}{kind_suffix}"
        )
    baseline_budgets = baseline.get("metric_budgets", {})
    for name, current_budget in current["metric_budgets"].items():
        baseline_budget = baseline_budgets.get(name)
        if baseline_budget is not None and current_budget > baseline_budget:
            observed_base = baseline_source_metrics.get(name)
            if observed_base is None or current_budget > observed_base:
                errors.append(
                    f"metric budget increase exceeds measured base source: {name} "
                    f"{baseline_budget} -> {current_budget}, base source {observed_base}"
                )
    return errors


def source_metrics_at_git_ref(repo_root: pathlib.Path, reference: str) -> dict[str, int]:
    completed = subprocess.run(
        ["git", "archive", "--format=tar", reference, "src"],
        cwd=repo_root,
        check=False,
        capture_output=True,
    )
    if completed.returncode:
        raise RuntimeError(
            f"cannot read Rust source metrics from baseline ref {reference!r}: "
            + completed.stderr.decode("utf-8", errors="replace").rstrip()
        )
    metrics = {
        "root_rust_loc": 0,
        "root_roomeq_rust_loc": 0,
        "root_binary_rust_loc": 0,
        "root_unit_tests": 0,
    }
    with tarfile.open(fileobj=io.BytesIO(completed.stdout), mode="r:") as archive:
        for member in archive.getmembers():
            if not member.isfile() or not member.name.endswith(".rs"):
                continue
            source_file = archive.extractfile(member)
            if source_file is None:
                continue
            source = source_file.read().decode("utf-8", errors="replace")
            line_count = len(source.splitlines())
            metrics["root_rust_loc"] += line_count
            if member.name.startswith("src/roomeq/"):
                metrics["root_roomeq_rust_loc"] += line_count
            if member.name.startswith("src/bin/"):
                metrics["root_binary_rust_loc"] += line_count
            metrics["root_unit_tests"] += len(TEST_ATTRIBUTE.findall(source))
    return metrics


def print_report(
    packages: dict[str, dict[str, Any]],
    policy: dict[str, Any],
    edges: set[WorkspaceEdge],
    metrics: dict[str, int],
) -> None:
    print("Crate-partition fitness report")
    print(
        f"workspace: {len(packages)} packages, {len(edges)} direct internal edges, "
        f"{len(policy['temporary_exceptions'])} temporary exceptions"
    )
    print(
        "normal/build dependency cycles: "
        f"{len(dependency_cycles(set(packages), edges))}"
    )
    print("temporary dependency exceptions:")
    for exception in policy["temporary_exceptions"]:
        print(
            f"  {exception['from']} -> {exception['to']} "
            f"(remove by {exception['remove_by']})"
        )
    for metric_name, value in metrics.items():
        budget = policy["metric_budgets"][metric_name]
        print(f"{metric_name}: {value} (budget {budget})")
    print("\nFocused crate tests and ownership:")
    print("package | src LOC | test functions | focused command")
    for package_name, lines, tests, command in package_metrics(packages, policy):
        print(f"{package_name} | {lines} | {tests} | {command}")

    consumers: dict[str, list[str]] = defaultdict(list)
    for source, destination, kind in sorted(edges):
        consumers[destination].append(f"{source} ({kind})")
    print("\nDirect workspace consumers:")
    for package_name in sorted(packages):
        names = ", ".join(consumers[package_name]) or "none"
        print(f"{package_name} <- {names}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=pathlib.Path, default=DEFAULT_POLICY)
    parser.add_argument(
        "--baseline-ref",
        default="",
        help="Git ref whose exception list and metric budgets are upper bounds",
    )
    arguments = parser.parse_args()

    policy_path = arguments.policy.resolve()
    policy = load_json(policy_path)
    metadata = cargo_metadata(REPO_ROOT)
    packages = workspace_packages(metadata)
    edges, errors = check_dependency_policy(packages, policy)
    metrics = root_metrics(REPO_ROOT)
    errors.extend(check_metric_budgets(metrics, policy))
    errors.extend(check_focused_tests(packages, policy))
    errors.extend(check_crate_documentation(packages))
    errors.extend(check_workspace_safety(REPO_ROOT, packages))
    errors.extend(check_public_api(REPO_ROOT, policy))
    errors.extend(check_schema_baseline_files(REPO_ROOT, policy))

    duplicates = duplicate_source_ownership(REPO_ROOT)
    for paths in duplicates:
        relative = [str(path.relative_to(REPO_ROOT)) for path in paths]
        errors.append("duplicate root/crate source ownership: " + ", ".join(relative))

    baseline, baseline_error = policy_from_git(
        REPO_ROOT, policy_path, arguments.baseline_ref
    )
    if baseline_error:
        errors.append(baseline_error)
    elif baseline is not None:
        try:
            baseline_metrics = source_metrics_at_git_ref(REPO_ROOT, arguments.baseline_ref)
        except RuntimeError as error:
            errors.append(str(error))
        else:
            errors.extend(
                check_monotonic_ratchets(policy, baseline, baseline_metrics)
            )
    elif arguments.baseline_ref:
        print("baseline ref predates WP0 policy; monotonic comparison bootstrapped")

    print_report(packages, policy, edges, metrics)
    print(f"normalized duplicate root/crate source groups: {len(duplicates)}")
    if errors:
        print("\nCrate-partition fitness FAILED:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1
    print("\nCrate-partition fitness PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
