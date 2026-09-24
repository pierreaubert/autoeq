#!/usr/bin/env python3
"""Execute exact F01-F15 software controls without promoting incomplete gates."""

import json
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "qa/registry/defect-matrix-pr.json"
OUTPUT = ROOT / "target/qa/defect-matrix-pr"
SAFE_NAME = re.compile(r"[A-Za-z0-9_-]+")
RESULT = re.compile(r"test result: ok\. (\d+) passed; 0 failed;")


def command_for(spec: dict, *, list_only: bool = False, exact_name: str = "") -> list[str]:
    package = spec["package"]
    if not SAFE_NAME.fullmatch(package):
        raise ValueError(f"invalid package in defect matrix: {package!r}")
    command = ["cargo", "test", "-p", package]
    if features := spec.get("features"):
        if not SAFE_NAME.fullmatch(features):
            raise ValueError(f"invalid features in defect matrix: {features!r}")
        command.extend(["--features", features])
    if target := spec.get("target"):
        if not SAFE_NAME.fullmatch(target):
            raise ValueError(f"invalid target in defect matrix: {target!r}")
        command.extend(["--test", target])
    else:
        command.append("--lib")
    if list_only:
        command.extend(["--", "--list"])
    else:
        command.extend([exact_name, "--", "--exact"])
    return command


def run_command(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command, cwd=ROOT, capture_output=True, text=True, check=False, timeout=600
    )


def test_key(spec: dict) -> tuple[str, str, str, str]:
    return (
        spec["package"], spec.get("target", ""), spec.get("features", ""), spec["filter"]
    )


def main() -> int:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    cases = manifest["cases"]
    expected_ids = {f"F{index:02d}" for index in range(1, 16)}
    if {case["id"] for case in cases} != expected_ids or len(cases) != 15:
        raise ValueError("defect matrix must cover F01-F15 exactly once")
    pairwise_path = ROOT / manifest["pairwise_manifest"]
    pairwise = json.loads(pairwise_path.read_text(encoding="utf-8"))
    pairwise_rows = pairwise["rows"]
    if not 1 <= len(pairwise_rows) <= pairwise["max_rows"]:
        raise ValueError("pairwise manifest has no bounded row set")

    OUTPUT.mkdir(parents=True, exist_ok=True)
    specs = [manifest["pairwise_test"]]
    specs.extend(test for case in cases for test in case["tests"])
    specs.extend(mutation["test"] for mutation in manifest["high_risk_mutations"])
    unique = {test_key(spec): spec for spec in specs}
    listed: dict[tuple[str, str, str], set[str]] = {}
    results = []
    failed = False
    for key, spec in unique.items():
        package, target, features, selected_filter = key
        group = (package, target, features)
        if group not in listed:
            listing = run_command(command_for(spec, list_only=True))
            (OUTPUT / f"list-{package}-{target or 'lib'}.log").write_text(
                listing.stdout + listing.stderr, encoding="utf-8"
            )
            if listing.returncode:
                raise RuntimeError(f"failed to list tests in {package}/{target or 'lib'}")
            listed[group] = {
                line.removesuffix(": test")
                for line in listing.stdout.splitlines()
                if line.endswith(": test")
            }
        matches = [
            name for name in listed[group]
            if name == selected_filter or name.endswith("::" + selected_filter)
        ]
        if len(matches) != 1:
            raise ValueError(f"{key}: expected one exact test, found {matches}")
        exact_name = matches[0]
        command = command_for(spec, exact_name=exact_name)
        completed = run_command(command)
        log_name = f"{package}-{selected_filter}.log"
        (OUTPUT / log_name).write_text(
            completed.stdout + completed.stderr, encoding="utf-8"
        )
        passed = completed.returncode == 0 and RESULT.search(completed.stdout) is not None
        results.append({
            "package": package,
            "target": target or "lib",
            "test": exact_name,
            "passed": passed,
            "log_path": str((OUTPUT / log_name).relative_to(ROOT)),
        })
        print(f"{'PASS' if passed else 'FAIL'} {package} {exact_name}", flush=True)
        failed |= not passed

    summary = {
        "version": 1,
        "manifest": str(MANIFEST.relative_to(ROOT)),
        "pairwise_rows": len(pairwise_rows),
        "tests": results,
        "software_controls_passed": not failed,
        "end_to_end_complete": not failed and all(
            case["coverage"] != "partial" for case in cases
        ),
        "remaining": [
            {"id": case["id"], "reason": case["remaining"]}
            for case in cases if case["coverage"] == "partial"
        ],
    }
    (OUTPUT / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"{sum(result['passed'] for result in results)}/{len(results)} exact software tests passed; "
        f"end-to-end complete={summary['end_to_end_complete']}",
        flush=True,
    )
    return 1 if failed else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (KeyError, ValueError, RuntimeError, subprocess.TimeoutExpired) as error:
        print(f"defect matrix failed: {error}", file=sys.stderr)
        sys.exit(2)
