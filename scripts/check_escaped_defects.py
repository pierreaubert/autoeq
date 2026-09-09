#!/usr/bin/env python3
"""Resolve ownership; registration alone does not prove mutation execution."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
from escaped_defect_ownership import ROOT, check, execute_regression

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("registry", nargs="?", type=Path, default=ROOT / "qa/registry/escaped-defects.json")
    parser.add_argument("--execute", action="store_true", help="run every exactly resolved regression")
    parser.add_argument("--evidence", type=Path, help="write status and exact-test execution evidence")
    args = parser.parse_args()
    report = {"status": "running", "executions": []}
    def persist():
        if args.evidence:
            args.evidence.parent.mkdir(parents=True, exist_ok=True)
            args.evidence.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    try:
        persist()  # invalidate any cached success before discovery or execution
        path = args.registry
        report["registry_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        count = check(path)
        if args.execute:
            for defect in json.loads(path.read_text(encoding="utf-8"))["defects"]:
                execution = execute_regression(defect["regression_test"])
                report["executions"].append({"defect_id": defect["id"], **execution})
                persist()
        report["status"] = "passed" if args.execute else "ownership_resolved_only"
        persist()
    except (ValueError, KeyError, OSError, TypeError) as error:
        report["status"] = "failed"
        report["error"] = str(error)
        persist()
        print(error, file=sys.stderr)
        raise SystemExit(1)
    print(f"resolved {count} escaped-defect owners; executed {len(report['executions'])} exact regressions; mutant execution is a separate required gate")
