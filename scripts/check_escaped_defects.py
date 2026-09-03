#!/usr/bin/env python3
"""Validate that every registered escaped defect has blocking ownership."""
import json
from pathlib import Path
import sys

required = {"id", "stage", "invariant", "regression_test", "pr_recipe", "mutant_fixture"}
path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("qa/registry/escaped-defects.json")
data = json.loads(path.read_text(encoding="utf-8"))
defects = data.get("defects", [])
if not defects:
    raise SystemExit("escaped-defects registry is empty")
ids = [item.get("id") for item in defects]
if len(ids) != len(set(ids)):
    raise SystemExit("escaped-defects registry contains duplicate IDs")
for item in defects:
    missing = required - item.keys()
    if missing:
        raise SystemExit(f"{item.get('id', '<unknown>')} missing {sorted(missing)}")
    fixture = Path(item["mutant_fixture"]) / "README.md"
    if not fixture.exists():
        raise SystemExit(f"{item['id']} references missing fixture {fixture}")
print(f"validated {len(defects)} escaped-defect registrations")
