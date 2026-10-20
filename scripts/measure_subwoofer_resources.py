#!/usr/bin/env python3
"""Measure each October 18 workload in a fresh process; retain failures as evidence."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
WORKLOADS = (
    "joint_2_1_128", "joint_4_4_512", "joint_8_8_1024",
    "cinema_2_2_44100", "cinema_4_3_48000", "cinema_8_4_96000",
    "accepted_cinema_48000", "dba_2_3_48000",
    "hybrid_cinema_8192_48000", "fir_cinema_16384_96000",
    "fir_minimum_8192_44100", "fir_linear_8192_48000", "fir_minimum_16384_96000",
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def worker(executable, workload, timeout):
    # This worker has exactly one child, so RUSAGE_CHILDREN's maximum RSS and
    # cumulative CPU times describe that child, not previous cases or Cargo.
    import resource
    start = time.monotonic()
    result = subprocess.run([str(executable), workload], capture_output=True,
                            text=True, timeout=timeout, check=False)
    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    if result.returncode:
        raise ValueError(f"{workload} exited {result.returncode}\n{result.stdout}\n{result.stderr}")
    payload = json.loads(result.stdout)
    if payload.get("workload") != workload:
        raise ValueError("runner returned the wrong workload identity")
    payload["process"] = {
        "wall_ms": (time.monotonic() - start) * 1000.0,
        "user_cpu_ms": usage.ru_utime * 1000.0,
        "system_cpu_ms": usage.ru_stime * 1000.0,
        "peak_rss_bytes": usage.ru_maxrss * (1 if sys.platform == "darwin" else 1024),
        "scope": "entire fresh workload process including fixtures, validation and reporting; excludes Cargo and Python",
    }
    return payload


def validate_sample(sample, max_runtime_ms, max_peak_rss_bytes):
    process = sample["process"]
    operation = sample["resources"]
    for key in ("wall_ms", "user_cpu_ms", "system_cpu_ms", "peak_rss_bytes"):
        if not isinstance(process[key], (int, float)) or not math.isfinite(process[key]) or process[key] < 0:
            raise ValueError(f"invalid process measurement: {key}")
    if process["peak_rss_bytes"] <= 0 or process["peak_rss_bytes"] > max_peak_rss_bytes:
        raise ValueError("peak RSS is missing or exceeds the resource budget")
    if process["wall_ms"] > max_runtime_ms:
        raise ValueError("workload runtime exceeds the resource budget")
    for key in ("allocations", "bytes_allocated", "reallocations", "bytes_deallocated", "deallocations"):
        if not isinstance(operation[key], int) or operation[key] < 0:
            raise ValueError(f"invalid allocation measurement: {key}")
    if operation["allocations"] == 0 or operation["bytes_allocated"] == 0:
        raise ValueError("workload did not record allocation activity")
    if not math.isfinite(operation["wall_ms"]) or operation["wall_ms"] <= 0:
        raise ValueError("invalid operation runtime")
    if not isinstance(sample.get("quality"), dict) or not sample["quality"]:
        raise ValueError("resource measurements require acoustic outcome evidence")


def run_suite(executable, output, samples=3, workloads=WORKLOADS,
              max_runtime_ms=900000, max_peak_rss_bytes=4 * 1024**3):
    report = {"schema_version": 1, "status": "running", "samples": [], "requested_workloads": list(workloads),
              "samples_per_workload": samples, "platform": platform.platform(),
              "budgets": {"max_runtime_ms": max_runtime_ms, "max_peak_rss_bytes": max_peak_rss_bytes}}

    def persist():
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    persist()  # Invalidate any previous success before starting work.
    try:
        report["executable_sha256"] = digest(executable)
        report["input_sha256"] = {name: digest(ROOT / name) for name in (
            "Cargo.lock", "crates/roomeq-workflow/examples/subwoofer_resources.rs",
            "crates/roomeq-workflow/benches/fixtures/subwoofers.rs",
            "scripts/measure_subwoofer_resources.py",
        )}
        if samples < 1 or not workloads or len(set(workloads)) != len(workloads):
            raise ValueError("expected positive sample count and distinct nonempty workloads")
        for workload in workloads:
            if workload not in WORKLOADS:
                raise ValueError(f"unknown workload {workload}")
            for repetition in range(samples):
                report["active"] = {"workload": workload, "repetition": repetition}
                persist()
                # An independent Python worker prevents accumulated child CPU/RSS
                # from contaminating later samples. The Rust process also starts cold.
                result = subprocess.run([
                    sys.executable, str(Path(__file__).resolve()), "--worker",
                    str(executable.resolve()), workload, str(max_runtime_ms / 1000.0),
                ], capture_output=True, text=True, check=False,
                    timeout=max_runtime_ms / 1000.0 + 30)
                if result.returncode:
                    raise ValueError(f"{workload} worker failed\n{result.stderr}")
                sample = json.loads(result.stdout)
                validate_sample(sample, max_runtime_ms, max_peak_rss_bytes)
                sample["repetition"] = repetition
                report["samples"].append(sample)
                persist()
        if len(report["samples"]) != samples * len(workloads):
            raise ValueError("incomplete workload matrix")
        report.pop("active", None)
        report["status"] = "passed"
    except Exception as error:
        report["status"] = "failed"
        report["error"] = str(error)
        persist()
        raise
    persist()
    return report


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--worker":
        print(json.dumps(worker(Path(sys.argv[2]), sys.argv[3], float(sys.argv[4])), allow_nan=False))
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", type=Path, help="already-built subwoofer_resources example")
    parser.add_argument("--output", type=Path, default=ROOT / "target/qa/subwoofer-resources.json")
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--workload", action="append", choices=WORKLOADS)
    parser.add_argument("--max-runtime-ms", type=float, default=900000)
    parser.add_argument("--max-peak-rss-mib", type=float, default=4096)
    args = parser.parse_args()
    if os.name != "posix":
        parser.error("fresh-process RSS/CPU measurement requires macOS or Linux")
    if not all(math.isfinite(x) and x > 0 for x in (args.max_runtime_ms, args.max_peak_rss_mib)):
        parser.error("resource budgets must be finite and positive")
    executable = args.executable
    if executable is None:
        # A build failure must also invalidate a previous successful report.
        build_report = {"schema_version": 1, "status": "running", "stage": "build", "samples": []}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(build_report, indent=2) + "\n")
        try:
            build = subprocess.run(["cargo", "build", "--release", "--offline", "--locked",
                                    "-p", "roomeq-workflow", "--example", "subwoofer_resources",
                                    "--message-format=json"], cwd=ROOT, capture_output=True, text=True, check=True)
            artifacts = [json.loads(line) for line in build.stdout.splitlines()]
            executable = next(Path(item["executable"]) for item in artifacts
                              if item.get("reason") == "compiler-artifact"
                              and item.get("target", {}).get("name") == "subwoofer_resources"
                              and item.get("executable"))
        except Exception as error:
            build_report.update(status="failed", error=str(error))
            args.output.write_text(json.dumps(build_report, indent=2) + "\n")
            raise
    report = run_suite(executable, args.output, args.samples, args.workload or WORKLOADS,
                       args.max_runtime_ms, args.max_peak_rss_mib * 1024**2)
    print(f"{report['status']}: {len(report['samples'])} fresh-process samples; {args.output}")


if __name__ == "__main__":
    main()
