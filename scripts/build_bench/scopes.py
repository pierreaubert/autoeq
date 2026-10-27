"""Verify corresponding semantic test groups across ownership changes."""
from __future__ import annotations

import json
from pathlib import Path
import re
import subprocess

from .execution import timed


def listed_tests(text: str) -> list[str]:
    return [line.removesuffix(": test") for line in text.splitlines()
            if line.endswith(": test")]


def validate_listing(text: str, expected: dict[str, bool]) -> None:
    if not expected:
        raise ValueError("a matching test scope must not be empty")
    actual = listed_tests(text)
    missing = sorted(expected.keys() - set(actual))
    extra = sorted(set(actual) - expected.keys())
    if missing or extra or len(actual) != len(expected):
        raise RuntimeError(f"test scope mismatch: missing={missing[:5]}, extra={extra[:5]}, "
                           f"expected={len(expected)}, listed={len(actual)}")


def validate_result(text: str, expected: dict[str, bool]) -> None:
    summaries = re.findall(r"^test result: ok\. (\d+) passed; (\d+) failed; (\d+) ignored;", text, re.MULTILINE)
    if len(summaries) != 1:
        raise RuntimeError("matching test execution did not emit exactly one successful summary")
    passed, failed, ignored = map(int, summaries[0])
    if (passed + ignored != len(expected) or failed or ignored != sum(expected.values())):
        raise RuntimeError(f"test execution scope changed: passed={passed}, ignored={ignored}, "
                           f"expected={len(expected)}, expected_ignored={sum(expected.values())}")


def resolve_identities(available: list[str], expected: dict[str, bool], aliases: dict) -> dict[str, str]:
    if not expected:
        raise ValueError("a matching test scope must not be empty")
    names = set(available)
    resolved = {}
    for logical in expected:
        matches = [name for name in [logical, *aliases.get(logical, [])] if name in names]
        if len(matches) != 1:
            raise RuntimeError(f"test identity {logical!r} has {len(matches)} explicit matches: {matches}")
        resolved[logical] = matches[0]
    if len(set(resolved.values())) != len(resolved):
        raise RuntimeError("multiple logical tests resolved to the same physical test")
    return resolved


def scope_for_compiler_profile(scope: dict, profile: dict | None) -> dict:
    """Apply only source-declared cfg gates, using the actual Cargo artifact."""
    conditions = scope.get("profile_conditions", {})
    excluded = {}
    for logical, condition in conditions.items():
        if logical not in scope["tests"] or set(condition) != {"debug_assertions"}:
            raise ValueError("invalid explicit profile condition: " + logical)
        if profile is None or not isinstance(profile.get("debug_assertions"), bool):
            raise ValueError("profile-gated scope requires Cargo's debug_assertions value")
        required = condition["debug_assertions"]
        if not isinstance(required, bool):
            raise ValueError("profile condition must be boolean")
        if profile["debug_assertions"] != required:
            excluded[logical] = condition
    return {**scope, "tests": {name: ignored for name, ignored in scope["tests"].items()
                               if name not in excluded},
            "all_profile_test_count": len(scope["tests"]),
            "profile_excluded_tests": excluded, "compiler_profile": profile}


def execute_matching_tests(binary: str, scope: dict, cwd: Path, env: dict,
                           output: Path, compiler_profile: dict | None = None) -> dict:
    scope = scope_for_compiler_profile(scope, compiler_profile)
    expected = scope["tests"]
    output.parent.mkdir(parents=True, exist_ok=True)
    available = subprocess.run([binary, "--list", "--format", "terse"], cwd=cwd, env=env,
                               capture_output=True, text=True, check=False)
    output.with_suffix(".available-tests.txt").write_text(available.stdout)
    if available.returncode:
        raise RuntimeError("could not list owning test target: " + available.stderr[-1500:])
    resolved = resolve_identities(listed_tests(available.stdout), expected, scope.get("aliases", {}))
    physical = {name: expected[logical] for logical, name in resolved.items()}
    command = [binary, "--test-threads=1", "--exact", *sorted(physical)]
    listing = subprocess.run([*command, "--list", "--format", "terse"], cwd=cwd, env=env,
                             capture_output=True, text=True, check=False)
    output.with_suffix(".selected-tests.txt").write_text(listing.stdout)
    if listing.returncode:
        raise RuntimeError("could not list matching test scope: " + listing.stderr[-1500:])
    validate_listing(listing.stdout, physical)
    identity_file = output.with_suffix(".identities.json")
    identity_file.write_text(json.dumps({**scope, "resolved_identities": resolved}, indent=2, sort_keys=True) + "\n")
    result = timed(command, cwd, env, output)
    if result["exit_code"]:
        raise RuntimeError("matching test execution failed: " + Path(result["stderr"]).read_text()[-1500:])
    validate_result(Path(result["stdout"]).read_text(), expected)
    result.update(logical_test_count=len(expected), logical_ignored_count=sum(expected.values()),
                  logical_test_identity_file=str(identity_file), exact_test_scope_verified=True,
                  profile_excluded_tests=scope["profile_excluded_tests"],
                  all_profile_test_count=scope["all_profile_test_count"])
    return result
