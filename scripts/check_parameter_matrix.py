#!/usr/bin/env python3
"""Check the checked-in PR covering-array manifest remains bounded and shaped."""
import json
from pathlib import Path

path = Path("qa/registry/parameter-matrix-pr.json")
data = json.loads(path.read_text(encoding="utf-8"))
rows = data.get("rows", [])
if not rows or len(rows) > data.get("max_rows", 24):
    raise SystemExit("parameter matrix must contain 1..max_rows generated rows")
if any(len(row) != len(data["dimensions"]) or any(value not in (0, 1, 2) for value in row) for row in rows):
    raise SystemExit("parameter matrix rows must contain nine ternary dimension values")
print(f"validated {len(rows)} checked-in pairwise rows")
