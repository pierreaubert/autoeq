#!/usr/bin/env python3
"""Enforce the crate-partition dependency graph and migration ratchets."""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import os
import pathlib
import re
import shlex
import subprocess
import sys
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


def load_json(path: pathlib.Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


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


def workspace_edges(
    packages: dict[str, dict[str, Any]],
) -> set[tuple[str, str]]:
    package_names = set(packages)
    return {
        (package_name, dependency["name"])
        for package_name, package in packages.items()
        for dependency in package["dependencies"]
        if dependency["name"] in package_names
    }


def dependency_cycles(
    package_names: set[str], edges: set[tuple[str, str]]
) -> list[list[str]]:
    graph = {name: set() for name in package_names}
    for source, destination in edges:
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


def exception_pairs(policy: dict[str, Any]) -> set[tuple[str, str]]:
    return {
        (exception["from"], exception["to"])
        for exception in policy["temporary_exceptions"]
    }


def check_dependency_policy(
    packages: dict[str, dict[str, Any]], policy: dict[str, Any]
) -> tuple[set[tuple[str, str]], list[str]]:
    errors: list[str] = []
    edges = workspace_edges(packages)
    root_package = policy["root_package"]
    terminal_consumers = set(policy["terminal_consumers"])
    allowed = {
        package: set(dependencies)
        for package, dependencies in policy["allowed_direct_dependencies"].items()
    }

    missing_policy = set(packages) - set(allowed) - terminal_consumers
    for package_name in sorted(missing_policy):
        errors.append(f"workspace package is missing from policy: {package_name}")

    exceptions = policy["temporary_exceptions"]
    temporary_edges = exception_pairs(policy)
    if len(temporary_edges) != len(exceptions):
        errors.append("temporary exception edges must be unique")

    for exception in exceptions:
        edge = (exception.get("from", ""), exception.get("to", ""))
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
        if edge[1] in allowed.get(edge[0], set()):
            errors.append(
                f"temporary exception is already allowed: {edge[0]} -> {edge[1]}"
            )

    for source, destination in sorted(edges):
        if source != root_package and destination == root_package:
            errors.append(
                f"workspace crate depends on root facade: {source} -> {destination}"
            )
            continue
        if source in terminal_consumers:
            continue
        if destination in allowed.get(source, set()):
            continue
        if (source, destination) in temporary_edges:
            continue
        errors.append(f"forbidden workspace edge: {source} -> {destination}")

    for cycle in dependency_cycles(set(packages), edges):
        errors.append("workspace dependency cycle: " + " -> ".join(cycle))
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
    return json.loads(shown.stdout), None


def check_monotonic_ratchets(
    current: dict[str, Any], baseline: dict[str, Any]
) -> list[str]:
    errors: list[str] = []
    added_exceptions = exception_pairs(current) - exception_pairs(baseline)
    for source, destination in sorted(added_exceptions):
        errors.append(
            f"temporary exception list may only shrink: added {source} -> {destination}"
        )
    baseline_budgets = baseline.get("metric_budgets", {})
    for name, current_budget in current["metric_budgets"].items():
        baseline_budget = baseline_budgets.get(name)
        if baseline_budget is not None and current_budget > baseline_budget:
            errors.append(
                f"metric budget may only shrink: {name} {baseline_budget} -> {current_budget}"
            )
    return errors


def print_report(
    packages: dict[str, dict[str, Any]],
    policy: dict[str, Any],
    edges: set[tuple[str, str]],
    metrics: dict[str, int],
) -> None:
    print("Crate-partition fitness report")
    print(
        f"workspace: {len(packages)} packages, {len(edges)} direct internal edges, "
        f"{len(policy['temporary_exceptions'])} temporary exceptions"
    )
    print(f"dependency cycles: {len(dependency_cycles(set(packages), edges))}")
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
    for source, destination in sorted(edges):
        consumers[destination].append(source)
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
        errors.extend(check_monotonic_ratchets(policy, baseline))
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
