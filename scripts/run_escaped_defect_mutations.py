#!/usr/bin/env python3
"""Run every registered semantic mutant; retain incomplete and failed runs."""
import argparse
import hashlib
import json
from pathlib import Path

if __package__:
    from .escaped_defect_ownership import ROOT, check
    from .escaped_defect_mutation import run as run_mutant
else:
    from escaped_defect_ownership import ROOT, check
    from escaped_defect_mutation import run as run_mutant


def execute(registry, evidence):
    report = {"status": "running", "mutants": []}

    def persist():
        evidence.parent.mkdir(parents=True, exist_ok=True)
        evidence.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    persist()
    try:
        report["registry_sha256"] = hashlib.sha256(registry.read_bytes()).hexdigest()
        count = check(registry)
        for defect in json.loads(registry.read_text(encoding="utf-8"))["defects"]:
            report["active_defect"] = defect["id"]
            persist()
            artifact = run_mutant(ROOT / defect["mutant_fixture"] / "manifest.json")
            report["mutants"].append({
                "defect_id": defect["id"],
                "evidence": str(artifact.relative_to(ROOT)),
                "evidence_sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
            })
            persist()
        if len(report["mutants"]) != count:
            raise ValueError("mutation execution count differs from validated registry")
        report.pop("active_defect", None)
        report["status"] = "passed"
        persist()
        return report
    except Exception as error:
        report["status"] = "failed"
        report["error"] = str(error)
        persist()
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=ROOT / "qa/registry/escaped-defects.json")
    parser.add_argument("--evidence", type=Path, default=ROOT / "target/qa/escaped-defects/mutation-run.json")
    args = parser.parse_args()
    execute(args.registry, args.evidence)
