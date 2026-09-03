#!/usr/bin/env python3
"""Turn a cargo-mutants run into a blocking, machine-readable QA summary."""
import json
from pathlib import Path
import sys

root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("target/qa/roomeq-mutants-gate-purpose")
outcomes_path = root / "mutants.out" / "outcomes.json"
if not outcomes_path.exists():
    raise SystemExit(f"missing cargo-mutants results: {outcomes_path}")
data = json.loads(outcomes_path.read_text(encoding="utf-8"))
total = int(data.get("total_mutants", 0))
caught = int(data.get("caught", 0))
missed = int(data.get("missed", 0))
timeouts = int(data.get("timeout", 0))
unviable = int(data.get("unviable", 0))
if total <= 0:
    raise SystemExit("mutation run generated no mutants")
kill_rate = caught / total
summary = {
    "total_mutants": total,
    "caught": caught,
    "missed": missed,
    "timeout": timeouts,
    "unviable": unviable,
    "kill_rate": kill_rate,
    "blocking": missed == 0 and timeouts == 0 and unviable == 0 and kill_rate >= 1.0,
}
(root / "mutation-summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
if not summary["blocking"]:
    raise SystemExit(f"mutation gate failed: {json.dumps(summary, sort_keys=True)}")
print(f"mutation gate passed: {caught}/{total} caught (kill_rate={kill_rate:.3f})")
