#!/usr/bin/env python3
"""Verify exact CLI continuation after signals and refuse stale checkpoint inputs."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import time


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(65536), b""):
            digest.update(block)
    return digest.hexdigest()


def read_state(path):
    with path.open("rb") as source:
        data = source.read(32 * 1024 * 1024 + 1)
    if len(data) > 32 * 1024 * 1024:
        raise RuntimeError("checkpoint exceeds the production 32 MiB limit")
    return json.loads(data)


def source_identity(root, binary, curve):
    paths = subprocess.check_output(["git", "ls-files", "-z"], cwd=root).split(b"\0")
    digest = hashlib.sha256()
    for relative in sorted(path for path in paths if path):
        digest.update(relative + b"\0")
        digest.update((root / os.fsdecode(relative)).read_bytes())
        digest.update(b"\0")
    return {
        "head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        "status": subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True),
        "tracked_source_sha256": digest.hexdigest(),
        "runner_sha256": sha256(Path(__file__)),
        "lock_sha256": sha256(root / "Cargo.lock"),
        "binary_sha256": sha256(binary),
        "curve_sha256": sha256(curve),
    }


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def run_process(command, directory, environment, timeout, interrupt_path=None, interrupt_signal=None):
    directory.mkdir()
    with (directory / "process.log").open("wb") as log:
        child = subprocess.Popen(command, cwd=directory, env=environment, stdout=log, stderr=log)
        observed_generation = None
        deadline = time.monotonic() + timeout
        try:
            if interrupt_path is not None:
                while child.poll() is None and time.monotonic() < deadline:
                    try:
                        state = read_state(interrupt_path)["checkpoint"]
                    except (FileNotFoundError, json.JSONDecodeError):
                        time.sleep(0.002)
                        continue
                    if state["generation"] >= 2 and state["terminal"] is None:
                        observed_generation = state["generation"]
                        child.send_signal(interrupt_signal)
                        break
                    time.sleep(0.002)
                require(observed_generation is not None,
                        "no nonterminal barrier was observed before interruption deadline")
            returncode = child.wait(timeout=max(0.001, deadline - time.monotonic()))
            return returncode, observed_generation
        finally:
            if child.poll() is None:
                child.kill()
                child.wait(timeout=10)


def check_contracts(binary, output, timeout, record):
    root = Path(__file__).resolve().parents[1]
    curve = output / "analytic.csv"
    curve.write_text("frequency,spl\n" + "".join(
        f"{frequency:.17g},{level:.17g}\n"
        for frequency, level in (
            (f, 4 * math.exp(-0.5 * (math.log(f / 650) / 0.4) ** 2)
             - 3 * math.exp(-0.5 * (math.log(f / 5000) / 0.3) ** 2))
            for f in (40 * (20000 / 40) ** (index / 255) for index in range(256))
        )))
    base = [str(binary), "--curve", str(curve), "--algo", "autoeq:de",
            "--num-filters", "2", "--population", "48", "--maxeval", "6000",
            "--seed", "91827", "--no-parallel", "--tolerance", "1e-12",
            "--atolerance", "1e-12", "--min-freq", "40", "--max-freq", "20000"]
    environment = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    record["before"] = source_identity(root, binary, curve)

    def command(name, flags):
        return base + ["--output", str(output / name / "report.html")] + flags

    baseline_path = output / "baseline.json"
    baseline_command = command("baseline", ["--checkpoint-exact", str(baseline_path)])
    status, _ = run_process(baseline_command, output / "baseline", environment, timeout)
    require(status == 0, f"uninterrupted baseline exited with {status}")
    baseline = read_state(baseline_path)["checkpoint"]
    require(baseline["terminal"]["finalized"], "baseline checkpoint is not finalized")
    record["baseline_command"] = baseline_command
    print("Uninterrupted CLI baseline passed.", flush=True)

    for name, interruption in [("sigint", signal.SIGINT), ("sigkill", signal.SIGKILL)]:
        path = output / f"{name}.json"
        initial_command = command(name, ["--checkpoint-exact", str(path)])
        status, observed = run_process(initial_command, output / name, environment, timeout,
                                       path, interruption)
        require(status == -interruption, f"{name} process exited with {status}")
        saved = read_state(path)["checkpoint"]
        require(saved["terminal"] is None, f"{name} checkpoint was already terminal")
        item = {"interruption": name, "exit_status": status,
                "observed_generation": observed, "persisted_generation": saved["generation"],
                "saved_checkpoint_sha256": sha256(path), "command": initial_command}
        resume_name = name + "-resumed"
        resumed_command = command(resume_name, ["--resume-exact", str(path)])
        status, _ = run_process(resumed_command, output / resume_name, environment, timeout)
        require(status == 0, f"{name} resumed process exited with {status}")
        require(read_state(path)["checkpoint"] == baseline,
                f"{name} full terminal DE checkpoint differs from uninterrupted baseline")
        item.update(resume_exit_status=status, full_terminal_checkpoint_equal=True,
                    resume_command=resumed_command)
        record["interruptions"].append(item)
        print(f"{name}: resumed full terminal DE state equals baseline.", flush=True)

    valid_path = output / "sigkill.json"
    valid_bytes = valid_path.read_bytes()
    malformed = output / "malformed.json"
    malformed.write_text('{"checkpoint":')
    for name, option, value, path, expected in [
        ("changed-budget", "--maxeval", "6001", valid_path, "identity"),
        ("changed-seed", "--seed", "91828", valid_path, "identity"),
        ("malformed-state", None, None, malformed, "load exact DE checkpoint"),
    ]:
        invocation = command(name, ["--resume-exact", str(path)])
        if option:
            invocation[invocation.index(option) + 1] = value
        status, _ = run_process(invocation, output / name, environment, timeout)
        require(status != 0, f"{name} unexpectedly succeeded")
        log = (output / name / "process.log").read_text()
        require(expected in log, f"{name} did not report the expected checkpoint refusal")
        require(valid_path.read_bytes() == valid_bytes, f"{name} replaced the valid checkpoint")
        require(not (output / name / "iir-autoeq-flat.txt").exists(), f"{name} published a preset")
        require(not (output / name / "report.html").exists(), f"{name} published a report")
        record["refusals"].append({"case": name, "exit_status": status,
                                   "saved_checkpoint_unchanged": True, "output_published": False,
                                   "command": invocation})
    record["after"] = source_identity(root, binary, curve)
    require(record["before"] == record["after"], "source, lock, binary or input changed during checks")
    print("Three process refusals passed without checkpoint or output replacement.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", required=True, type=Path, help="built AutoEQ CLI executable")
    parser.add_argument("--output", required=True, type=Path, help="new private evidence directory")
    parser.add_argument("--timeout", type=float, default=120, help="positive per-process deadline in seconds")
    args = parser.parse_args()
    if os.name != "posix":
        parser.error("these SIGINT/SIGKILL process contracts require a POSIX host")
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("--timeout must be finite and positive")
    binary = args.binary.resolve()
    if not binary.is_file() or not os.access(binary, os.X_OK):
        parser.error("--binary must name an executable file")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    record = {"status": "running", "scope": "same-build CLI process continuation",
              "audio_device_opened": False, "interruptions": [], "refusals": []}
    try:
        check_contracts(binary, output, args.timeout, record)
        record["status"] = "passed"
    except (Exception, KeyboardInterrupt) as error:
        record.update(status="failed", error=str(error) or type(error).__name__)
        raise
    finally:
        (output / "results.json").write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    main()
