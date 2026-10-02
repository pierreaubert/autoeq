#!/usr/bin/env python3
"""Check the checked-in PR covering-array manifest matches its declared axes."""
import json
from pathlib import Path
import sys

MAX_SUPPORTED_ROWS = 24


def validate_manifest(data: dict) -> list[str]:
    errors = []
    if not isinstance(data, dict):
        return ["parameter matrix manifest must be an object"]
    dimensions = data.get("dimensions")
    rows = data.get("rows", [])
    maximum = data.get("max_rows", 24)
    if not isinstance(dimensions, dict) or not dimensions:
        return ["parameter matrix dimensions must be a non-empty object"]
    if (
        not isinstance(maximum, int)
        or isinstance(maximum, bool)
        or maximum < 1
        or maximum > MAX_SUPPORTED_ROWS
    ):
        return [f"parameter matrix max_rows must be an integer from 1..{MAX_SUPPORTED_ROWS}"]
    if not isinstance(rows, list) or not rows or len(rows) > maximum:
        errors.append("parameter matrix must contain 1..max_rows generated rows")
        return errors
    domains = list(dimensions.values())
    if any(not isinstance(domain, list) or len(domain) < 2 for domain in domains):
        errors.append("every parameter dimension must declare at least two values")
    for name, domain in dimensions.items():
        if not isinstance(domain, list):
            continue
        if len(domain) != len(set(map(json.dumps, domain))):
            errors.append(f"parameter dimension {name!r} repeats a semantic value")
    for index, row in enumerate(rows):
        if not isinstance(row, list) or len(row) != len(domains):
            errors.append(f"parameter matrix row {index} has the wrong width")
            continue
        for axis, value in enumerate(row):
            if not isinstance(value, int) or isinstance(value, bool):
                errors.append(f"parameter matrix row {index} axis {axis} is not an integer")
            elif axis < len(domains) and not 0 <= value < len(domains[axis]):
                errors.append(f"parameter matrix row {index} axis {axis} is outside its domain")
    encoded_rows = [json.dumps(row, separators=(",", ":")) for row in rows]
    if len(set(encoded_rows)) != len(rows):
        errors.append("parameter matrix contains duplicate rows")
    return errors


def main() -> int:
    path = Path("qa/registry/parameter-matrix-pr.json")
    data = json.loads(path.read_text(encoding="utf-8"))
    errors = validate_manifest(data)
    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    print(f"validated {len(data['rows'])} checked-in pairwise rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
