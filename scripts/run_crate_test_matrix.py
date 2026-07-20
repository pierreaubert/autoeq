#!/usr/bin/env python3
"""Expose and execute the crate-partition focused-test matrix for CI."""

from __future__ import annotations

import argparse
import json
import pathlib
import shlex
import subprocess


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_POLICY = REPO_ROOT / "scripts" / "crate_partition_policy.json"


def focused_tests(policy_path: pathlib.Path) -> dict[str, str]:
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=pathlib.Path, default=DEFAULT_POLICY)
    parser.add_argument("--list-json", action="store_true")
    parser.add_argument("--package")
    parser.add_argument("--release", action="store_true")
    arguments = parser.parse_args()

    tests = focused_tests(arguments.policy.resolve())
    if arguments.list_json:
        if arguments.package:
            parser.error("--list-json and --package are mutually exclusive")
        print(json.dumps(sorted(tests)))
        return 0
    if not arguments.package:
        parser.error("one of --list-json or --package is required")

    try:
        command = command_for_package(
            tests, arguments.package, release=arguments.release
        )
    except ValueError as error:
        parser.error(str(error))
    return subprocess.run(command, cwd=REPO_ROOT, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
