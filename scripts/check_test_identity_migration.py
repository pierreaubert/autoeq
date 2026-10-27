#!/usr/bin/env python3
"""Verify exact nextest identities and ignored states across target relocations."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def compare(before: dict, after: dict, owners: dict[str, str | dict[str, str]]) -> tuple[list[str], list[list[str]]]:
    errors = []
    rows = []
    suites = after["rust-suites"]
    for old_id, old_suite in before["rust-suites"].items():
        cases = old_suite["testcases"]
        if not cases:
            continue
        target = old_suite["binary-name"]
        if target not in owners:
            errors.append(f"missing ownership mapping: {old_id}")
            continue
        owner = owners[target]
        if isinstance(owner, dict):
            # This is the sole obsolete test contract permitted by the plan.
            expected = "published_package_excludes_qa_scripts_backups_and_mdat"
            if (target != "packaging_excludes" or old_suite.get("package-name") != "autoeq"
                    or set(cases) != {expected} or not owner.get("retired")):
                errors.append(f"unsupported test retirement: {old_id}")
                continue
            if any(suite.get("package-name") == "autoeq" for suite in suites.values()):
                errors.append("packaging test cannot retire while the root package exists")
                continue
            rows.append([old_id, expected, "<retired root packaging contract>", str(cases[expected]["ignored"]).lower()])
            continue
        new_id = f"{owner}::{target}"
        new_suite = suites.get(new_id)
        if new_suite is None:
            errors.append(f"missing target: {new_id}")
            continue
        if new_id != old_id and old_id in suites:
            errors.append(f"old target still compiled: {old_id}")
        new_cases = new_suite["testcases"]
        for missing in sorted(cases.keys() - new_cases.keys()):
            errors.append(f"missing test: {old_id}::{missing}")
        for extra in sorted(new_cases.keys() - cases.keys()):
            errors.append(f"unmapped added test: {new_id}::{extra}")
        for name in sorted(cases.keys() & new_cases.keys()):
            for field in ("kind", "ignored"):
                if cases[name][field] != new_cases[name][field]:
                    errors.append(f"changed {field}: {old_id}::{name}")
            rows.append([old_id, name, new_id, str(cases[name]["ignored"]).lower()])
    return errors, rows


def compare_full_inventory(before: dict, after: dict, manifest: dict) -> tuple[list[str], list[list[str]]]:
    """Account for every case, with exact relocations and explicit additions."""
    def flatten(inventory):
        return {(suite_id, name): case for suite_id, suite in inventory["rust-suites"].items()
                for name, case in suite["testcases"].items()}

    old, new = flatten(before), flatten(after)
    moves = {}
    errors, rows = [], []
    for move in manifest.get("relocations", []):
        source = (move["from_suite"], move["name"])
        destination = (move["to_suite"], move.get("to_name", move["name"]))
        if source not in old:
            errors.append(f"relocation source does not exist: {source}")
        if source in moves:
            errors.append(f"duplicate relocation source: {source}")
        moves[source] = destination
    additions = {(item["suite"], item["name"]) for item in manifest.get("additions", [])}
    expected = set()
    for source, case in sorted(old.items()):
        destination = moves.get(source, source)
        if destination in expected:
            errors.append(f"duplicate relocation destination: {destination}")
        expected.add(destination)
        if destination not in new:
            errors.append(f"missing test: {source} -> {destination}")
            continue
        for field in ("kind", "ignored"):
            if case[field] != new[destination][field]:
                errors.append(f"changed {field}: {source} -> {destination}")
        rows.append([source[0], source[1], destination[0], str(case["ignored"]).lower()])
    for overlap in sorted(expected & additions):
        errors.append(f"declared addition already existed: {overlap}")
    for missing in sorted(additions - new.keys()):
        errors.append(f"missing declared addition: {missing}")
    for extra in sorted(new.keys() - expected - additions):
        errors.append(f"unmapped added test: {extra}")
    return errors, rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", type=Path, action="append", required=True)
    parser.add_argument("--after", type=Path, required=True)
    mapping = parser.add_mutually_exclusive_group(required=True)
    mapping.add_argument("--owners", type=Path)
    mapping.add_argument("--full-relocations", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    after = json.loads(args.after.read_text())
    owners = json.loads(args.owners.read_text()) if args.owners else None
    errors = []
    rows = []
    seen = set()
    for path in args.before:
        before = json.loads(path.read_text())
        repeated = seen & before["rust-suites"].keys()
        if repeated:
            errors.append(f"duplicate baseline target(s): {sorted(repeated)}")
        seen.update(before["rust-suites"])
        if args.full_relocations:
            if len(args.before) != 1:
                parser.error("full inventory comparison requires exactly one baseline")
            new_errors, new_rows = compare_full_inventory(before, after, json.loads(args.full_relocations.read_text()))
        else:
            new_errors, new_rows = compare(before, after, owners)
        errors.extend(new_errors)
        rows.extend(new_rows)
    if errors:
        print("\n".join(errors))
        return 1
    with args.output.open("w", newline="") as output:
        writer = csv.writer(output, delimiter="\t", lineterminator="\n")
        writer.writerow(["old_binary_id", "test_name", "new_binary_id", "ignored"])
        writer.writerows(rows)
    retired = sum(row[2].startswith("<retired") for row in rows)
    if args.full_relocations:
        print(f"Full inventory: {len(rows)} preserved identities; exact relocations and declared additions verified")
        return 0
    print(f"Test identities accounted for: {len(rows) - retired} preserved, {retired} obsolete root packaging contract(s); zero missing or extra")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
