#!/usr/bin/env python3
"""Reintroduce each October 18 correctness defect in an isolated source snapshot."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
JOINT = "crates/roomeq-engine/src/multisub/joint_objective.rs"
PREPROCESS = "crates/roomeq-workflow/src/topology/bass_management/preprocess.rs"
CORRELATED = "crates/roomeq-workflow/src/room_optimization/seat_replay/correlated_bass.rs"
CASES = {
    "C1": (PREPROCESS, "roomeq-workflow",
           "topology::bass_management::preprocess::tests::preprocess_dba_expands_array_controls_to_every_physical_driver",
           "            curves\n                .into_iter()\n                .enumerate()",
           "            curves\n                .into_iter()\n                .take(1)\n                .enumerate()"),
    "C2": (JOINT, "roomeq-engine",
           "multisub::joint_objective::tests::joint_dispatch_uses_and_validates_configured_primary_seat",
           "result.per_seat_post.get(primary_seat).cloned()", "result.per_seat_post.first().cloned()"),
    "C3": (JOINT, "roomeq-engine",
           "multisub::joint_objective::tests::joint_dispatch_preserves_aligned_arrays_with_flat_and_zero_slope_targets",
           ".map(|value| baseline_level + value - shape_level)", ".map(|value| value - shape_level)"),
    "C4": (JOINT, "roomeq-engine",
           "multisub::joint_objective::tests::broadband_alignment_is_feasible_for_every_sub_order",
           "    upper[0] = 0.0;", "    upper[0] = 0.0;\n    upper[subs] = 0.0;"),
    "C5": (CORRELATED, "roomeq-workflow",
           "room_optimization::seat_replay::correlated_bass::tests::final_serialized_routes_reject_opposed_correlated_bass_at_every_seat",
           "passed: loss <= limit + 1e-6,", "passed: true,"),
}


def require_execution(log, name, passed):
    expected = "ok" if passed else "FAILED"
    executions = re.findall(r"^test (\S+) \.\.\. (ok|FAILED|ignored)$", log, re.MULTILINE)
    if executions != [(name, expected)]:
        raise ValueError(f"expected only {name} ... {expected}; got {executions}")
    summary = "1 passed; 0 failed; 0 ignored" if passed else "0 passed; 1 failed; 0 ignored"
    if summary not in log:
        raise ValueError("missing exact Rust test execution summary")


def source_snapshot(destination):
    listing = subprocess.run(["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
                             cwd=ROOT, capture_output=True, check=True).stdout
    inventory = hashlib.sha256()
    for name in sorted(set(os.fsdecode(x) for x in listing.split(b"\0") if x)):
        source = ROOT / name
        if not source.is_file():
            continue
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target, follow_symlinks=True)
        inventory.update(name.encode() + b"\0" + hashlib.sha256(target.read_bytes()).digest())
    return inventory.hexdigest()


def run(output, cases):
    report = {"schema_version": 1, "status": "running", "cases": [], "requested": list(cases)}
    output.mkdir(parents=True, exist_ok=True)
    # Cargo releases its build lock before executing tests. Sharing a target
    # directory lets a concurrent mutant replace an ordinary test executable
    # between compilation and execution, even with isolated source trees.
    environment = os.environ.copy()
    environment["CARGO_TARGET_DIR"] = str((output / "cargo-target").resolve())
    report["target_dir"] = environment["CARGO_TARGET_DIR"]

    def persist():
        (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")

    persist()
    try:
        if not cases or len(set(cases)) != len(cases):
            raise ValueError("expected distinct nonempty mutation cases")
        # Keep the snapshot beside the checkout so workspace ../math-audio
        # dependencies resolve identically, without modifying the real sources.
        with tempfile.TemporaryDirectory(prefix="autoeq-oct18-mutants-", dir=ROOT.parent) as tmp:
            snapshot = Path(tmp)
            report["source_snapshot_sha256"] = source_snapshot(snapshot)
            persist()
            for case in cases:
                source_name, package, name, before, after = CASES[case]
                report["active"] = case
                persist()
                source = snapshot / source_name
                original = source.read_text()
                if original.count(before) != 1:
                    raise ValueError(f"{case}: mutation must match exactly one source site")
                command = ["cargo", "test", "--release", "--offline", "--locked", "-p", package,
                           "--lib", name, "--", "--exact", "--nocapture"]
                baseline = subprocess.run(command, cwd=snapshot, capture_output=True, text=True,
                                          check=False, timeout=900, env=environment)
                (output / f"{case}-baseline.log").write_text(baseline.stdout + baseline.stderr)
                if baseline.returncode:
                    raise ValueError(f"{case}: unmutated baseline failed")
                require_execution(baseline.stdout, name, True)
                source.write_text(original.replace(before, after))
                mutated_sha256 = hashlib.sha256(source.read_bytes()).hexdigest()
                try:
                    mutant = subprocess.run(command, cwd=snapshot, capture_output=True, text=True,
                                            check=False, timeout=900, env=environment)
                finally:
                    source.write_text(original)
                (output / f"{case}-mutant.log").write_text(mutant.stdout + mutant.stderr)
                # An executed Rust assertion failure proves the mutant built;
                # compile errors, zero tests, ignored tests and timeouts cannot pass.
                if mutant.returncode != 101:
                    raise ValueError(f"{case}: mutant survived or did not fail a Rust test")
                require_execution(mutant.stdout, name, False)
                report["cases"].append({"id": case, "status": "caught", "source": source_name,
                    "regression": name, "command": command,
                    "source_sha256": hashlib.sha256(original.encode()).hexdigest(),
                    "mutated_sha256": mutated_sha256,
                    "baseline_log_sha256": hashlib.sha256((output / f"{case}-baseline.log").read_bytes()).hexdigest(),
                    "mutant_log_sha256": hashlib.sha256((output / f"{case}-mutant.log").read_bytes()).hexdigest()})
                persist()
        if len(report["cases"]) != len(cases):
            raise ValueError("incomplete mutation matrix")
        report.pop("active", None)
        report["status"] = "passed"
    except Exception as error:
        report["status"] = "failed"
        report["error"] = str(error)
        persist()
        raise
    persist()
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "target/qa/audit-oct18-mutants")
    parser.add_argument("--case", action="append", choices=CASES)
    args = parser.parse_args()
    result = run(args.output, args.case or list(CASES))
    print(f"{result['status']}: {len(result['cases'])} built and caught audit mutants")
