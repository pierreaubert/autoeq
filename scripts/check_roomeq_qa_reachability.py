#!/usr/bin/env python3
"""Fail when a declared RoomEQ QA suite has no scheduled or PR entry point."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "crates/roomeq-qa/src/registry.json"
WORKFLOWS = ROOT / ".github/workflows"
JUSTFILES = (ROOT / "Justfile", ROOT / "builds/qa/qa-roomeq.just")

RUNNER_COMMANDS = {
    "synthetic": "roomeq-qa-synthetic",
    "features": "roomeq-qa-features",
    "acoustic": "roomeq-qa-acoustic",
    "integration": "roomeq_generated_data_test",
    "quality": "roomeq-qa-quality",
    "fuzzer": "roomeq-fuzzer",
}

RECIPE_HEADER = re.compile(r"^([A-Za-z][A-Za-z0-9_-]*)(?:\s+[^:]*)?:\s*(.*)$")
JUST_CALL = re.compile(r"\bjust\s+([A-Za-z][A-Za-z0-9_-]*)")


def parse_just_recipes(text: str) -> dict[str, str]:
    recipes: dict[str, list[str]] = {}
    current: str | None = None
    for line in text.splitlines():
        if line and not line[0].isspace():
            match = RECIPE_HEADER.match(line)
            if match and not line.startswith(("alias ", "import ", "mod ")):
                current = match.group(1)
                recipes.setdefault(current, []).append(match.group(2))
                continue
        if current is not None:
            recipes[current].append(line)
    return {name: "\n".join(lines) for name, lines in recipes.items()}


def load_recipes() -> dict[str, str]:
    recipes: dict[str, str] = {}
    for path in JUSTFILES:
        recipes.update(parse_just_recipes(path.read_text(encoding="utf-8")))
    return recipes


def reachable_workflow_text() -> tuple[str, set[str]]:
    workflow_text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted(WORKFLOWS.glob("*.y*ml"))
    )
    recipes = load_recipes()
    pending = list(JUST_CALL.findall(workflow_text))
    visited: set[str] = set()
    expanded = [workflow_text]
    while pending:
        recipe = pending.pop()
        if recipe in visited:
            continue
        visited.add(recipe)
        body = recipes.get(recipe, "")
        expanded.append(body)
        pending.extend(JUST_CALL.findall(body))
        header = body.splitlines()[0] if body else ""
        pending.extend(
            token
            for token in re.findall(r"\b[A-Za-z][A-Za-z0-9_-]*\b", header)
            if token in recipes
        )
    return "\n".join(expanded), visited


def missing_suite_runners() -> dict[str, list[str]]:
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    suites_by_runner: dict[str, list[str]] = {}
    for suite in registry["suites"]:
        suites_by_runner.setdefault(suite["runner"], []).append(suite["id"])

    unknown = sorted(set(suites_by_runner) - set(RUNNER_COMMANDS))
    if unknown:
        raise ValueError(f"QA reachability checker has no command mapping for: {unknown}")

    workflow_text, _ = reachable_workflow_text()
    return {
        runner: suite_ids
        for runner, suite_ids in sorted(suites_by_runner.items())
        if RUNNER_COMMANDS[runner] not in workflow_text
    }


def main() -> int:
    missing = missing_suite_runners()
    if missing:
        for runner, suites in missing.items():
            print(
                f"unreachable RoomEQ QA runner '{runner}' "
                f"({RUNNER_COMMANDS[runner]}): {', '.join(suites)}",
                file=sys.stderr,
            )
        return 1
    print("All declared RoomEQ QA suite runners are reachable from CI or schedules.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
