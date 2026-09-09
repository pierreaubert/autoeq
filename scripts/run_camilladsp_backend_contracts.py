#!/usr/bin/env python3
"""Required (never silently skipped) CamillaDSP PCM export contracts."""
import json
import os
from pathlib import Path
import re
import shutil
import subprocess


REQUIRED_TESTS = {
    "room_optimization::gd::tests::tool_contract_camilladsp_fractional_gd_matches_exported_response",
    "tests::conformance::tool_contract_camilladsp_pcm_preserves_polarity_and_delay",
    "tests::conformance::tool_contract_camilladsp_pcm_processes_convolution_sidecar",
    "tests::conformance::tool_contract_camilladsp_pcm_matches_linkwitz_riley_crossover_gain",
    "tests::conformance::tool_contract_camilladsp_pcm_matches_peaking_filter_gain",
    "tests::conformance::tool_contract_camilladsp_pcm_preserves_routed_channel_matrix",
    "tests::realized_transfer::tool_contract_camilladsp_multisub_coherent_peak_at_all_rates",
}


def main():
    root = Path(__file__).resolve().parents[1]
    artifact = root / "target/qa/camilladsp-backend-contracts.json"
    artifact.parent.mkdir(parents=True, exist_ok=True)
    record = {"status": "running", "scope": "PCM contracts and sampled steady-state physical-sub peaks"}
    artifact.write_text(json.dumps(record, indent=2))
    try:
        requested = os.environ.get("ROOMEQ_CAMILLADSP_BIN", "camilladsp")
        binary = shutil.which(requested)
        if not binary:
            raise RuntimeError("CamillaDSP is required; install it or set ROOMEQ_CAMILLADSP_BIN")
        version = subprocess.run([binary, "--version"], capture_output=True, text=True,
                                 check=True, timeout=10)
        record.update(binary=binary, version=version.stdout.strip())
        environment = dict(os.environ, ROOMEQ_CAMILLADSP_BIN=binary)
        command = ["cargo", "test", "-p", "roomeq-export", "-p", "roomeq-workflow", "--lib", "tool_contract_camilladsp",
                   "--no-default-features", "--", "--nocapture"]
        completed = subprocess.run(command, cwd=root, env=environment, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, timeout=180)
        log = artifact.with_suffix(".log")
        log.write_text(completed.stdout)
        record.update(command=command, returncode=completed.returncode, log=str(log))
        print(completed.stdout, end="")
        summaries = re.findall(
            r"test result: (\w+)\. (\d+) passed; (\d+) failed; (\d+) ignored",
            completed.stdout,
        )
        passed = sum(int(summary[1]) for summary in summaries)
        if (completed.returncode or len(summaries) != 2
                or any(status != "ok" or int(failed) or int(ignored)
                       for status, _, failed, ignored in summaries)
                or passed < len(REQUIRED_TESTS)):
            raise RuntimeError("backend contracts failed or both crate suites did not execute")
        if "skipping optional PCM backend" in completed.stdout:
            raise RuntimeError("required backend contract was skipped")
        executed = set(re.findall(r"test (\S+) \.\.\. ok", completed.stdout))
        missing = sorted(REQUIRED_TESTS - executed)
        record.update(required_tests=sorted(REQUIRED_TESTS), missing_tests=missing)
        if missing:
            raise RuntimeError(f"required backend tests did not pass: {missing}")
        record.update(status="passed", tests_passed=passed)
    except Exception as error:
        record.update(status="failed", error=str(error))
        raise
    finally:
        artifact.write_text(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
