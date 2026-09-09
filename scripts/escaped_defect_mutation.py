"""Run one preregistered source mutant in cargo-mutants' isolated copy."""
import hashlib
import json
import re
from pathlib import Path
import subprocess
import tempfile

try:
    from .escaped_defect_ownership import ROOT, execute_regression
except ImportError:
    from escaped_defect_ownership import ROOT, execute_regression


def require_single_mutant(discovery):
    if not isinstance(discovery, list) or len(discovery) != 1:
        raise ValueError("mutation discovery must select exactly one preregistered mutant")


def require_caught(report):
    if any(report.get(key) != value for key, value in {
        "total_mutants": 1, "caught": 1, "missed": 0, "timeout": 0, "unviable": 0,
    }.items()):
        raise ValueError("expected exactly one caught mutant; survivor, timeout, unviable or empty runs fail")
    baselines = [o for o in report.get("outcomes", []) if o.get("scenario") == "Baseline"]
    if len(baselines) != 1 or baselines[0].get("summary") != "Success":
        raise ValueError("mutation evidence requires a successful unmutated baseline")
    mutants = [o for o in report.get("outcomes", []) if isinstance(o.get("scenario"), dict)
               and "Mutant" in o["scenario"]]
    if len(mutants) != 1 or mutants[0].get("summary") != "CaughtMutant":
        raise ValueError("missing caught mutant execution outcome")
    phases = {p["phase"]: p["process_status"] for p in mutants[0].get("phase_results", [])}
    if phases.get("Build") != "Success" or phases.get("Test") != {"Failure": 101}:
        raise ValueError("mutant must build successfully and fail its Rust regression test")


def require_regression_log(log, name, expected):
    """Bind a successful baseline/caught status to the selected test's execution."""
    executions = re.findall(r"^test (\S+) \.\.\. (ok|FAILED|ignored)$", log, re.MULTILINE)
    if executions != [(name, expected)]:
        raise ValueError(f"expected only registered regression {name!r} with status {expected}; got {executions!r}")
    summaries = re.findall(r"test result: (?:ok|FAILED)\. (\d+) passed; (\d+) failed; (\d+) ignored;", log)
    totals = tuple(sum(int(summary[i]) for summary in summaries) for i in range(3))
    wanted = (1, 0, 0) if expected == "ok" else (0, 1, 0)
    if totals != wanted:
        raise ValueError(f"regression log has unexpected execution totals: {totals}")


def verify_execution_logs(report, directory, name):
    hashes = {}
    for outcome in report["outcomes"]:
        path = (directory / outcome["log_path"]).resolve()
        if not path.is_relative_to(directory.resolve()):
            raise ValueError("mutation log path escapes its artifact directory")
        expected = "ok" if outcome["scenario"] == "Baseline" else "FAILED"
        require_regression_log(path.read_text(), name, expected)
        hashes[outcome["log_path"]] = hashlib.sha256(path.read_bytes()).hexdigest()
    return hashes


def run(manifest_path):
    manifest = json.loads(Path(manifest_path).read_text())
    regression = execute_regression(manifest["regression_test"])
    parent = ROOT / "target/qa/escaped-defects"
    parent.mkdir(parents=True, exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix=manifest["id"] + "-", dir=parent))
    mutation = manifest["mutation"]
    source = (ROOT / mutation["file"]).resolve()
    test_source = (ROOT / manifest["test_source"]).resolve()
    if not source.is_relative_to(ROOT) or not test_source.is_relative_to(ROOT):
        raise ValueError("mutation source paths must stay in the repository")
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in (source, test_source)}
    command = [
        "cargo", "mutants", "--no-config", "--test-tool", "cargo",
        "--package", mutation["package"], "--test-package", manifest["regression_test"]["package"],
        "--file", mutation["file"], "--re", mutation["selector"],
        "--baseline", "run", "--timeout", "120", "--build-timeout", "600",
        "--cargo-arg=--offline", "--cargo-arg=--lib", "--cargo-arg=--package=" + manifest["regression_test"]["package"],
        "--cargo-test-arg=" + manifest["regression_test"]["name"],
        "--cargo-test-arg=--", "--cargo-test-arg=--exact",
        "-o", str(output),
    ]
    if "scope_diff" in mutation:
        scope = (ROOT / mutation["scope_diff"]).resolve()
        if not scope.is_relative_to(ROOT):
            raise ValueError("mutation scope must stay in the repository")
        hashes[str(scope.relative_to(ROOT))] = hashlib.sha256(scope.read_bytes()).hexdigest()
        command.extend(["--in-diff", mutation["scope_diff"]])
    discovery = subprocess.run(command + ["--list", "--json"], cwd=ROOT,
                               text=True, capture_output=True, check=False)
    if discovery.returncode:
        raise ValueError("mutation discovery failed\n" + discovery.stderr)
    require_single_mutant(json.loads(discovery.stdout))
    result = subprocess.run(command, cwd=ROOT, check=False)
    report = json.loads((output / "mutants.out/outcomes.json").read_text())
    require_caught(report)
    log_hashes = verify_execution_logs(report, output / "mutants.out", manifest["regression_test"]["name"])
    if result.returncode:
        raise ValueError(f"mutation runner exited {result.returncode}")
    evidence = {"defect": manifest["id"], "regression": regression,
                "source_sha256": hashes, "execution_log_sha256": log_hashes, "mutation_command": command,
                "outcomes": report}
    artifact = output / "evidence.json"
    artifact.write_text(json.dumps(evidence, indent=2) + "\n")
    print(f"caught registered mutant; evidence: {artifact}")
    return artifact
