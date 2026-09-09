#!/usr/bin/env python3
"""Required selected-matrix PCM replay; stale/partial evidence never passes."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import uuid

ROOT = Path(__file__).resolve().parents[1]
TEST = "tests::realized_transfer::parameter_matrix_backend_complex_transfer"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_row(row, run_id):
    directory = Path(row["replay_bundle"]["directory"]).resolve()
    allowed = (ROOT / "target/qa/roomeq-parameter-bundles").resolve()
    if not directory.is_relative_to(allowed):
        raise ValueError("bundle outside matrix artifact directory")
    proof = json.loads((directory / "backend-complex-transfer.json").read_text())
    if (proof["status"] != "passed" or proof["run_id"] != run_id
            or proof["row"] != row["row"]):
        raise ValueError("missing, failed, or stale backend row")
    if (proof["sample_rate_hz"] != row["sample_rate_hz"]
            or proof["requested_axes"] != row["requested_axes"]
            or proof["unexecuted_axes"] != row["unexecuted_axes"]):
        raise ValueError("backend row identity changed")
    required = {row["replay_bundle"]["request"], row["replay_bundle"]["selected_output"]}
    graph = json.loads((directory / row["replay_bundle"]["selected_output"]).read_text())
    routing = ((graph.get("metadata") or {}).get("bass_management") or {}).get("routing_graph")
    if routing and routing.get("routes"):
        inputs, outputs = routing["input_channels"], routing["output_channels"]
    else:
        inputs = outputs = sorted(graph["channels"])
    if proof["input_channels"] != inputs or proof["output_channels"] != outputs:
        raise ValueError("backend channel inventory differs from graph")
    for chain in graph["channels"].values():
        required.update(p["parameters"]["ir_file"] for p in chain["plugins"]
                        if p["plugin_type"] == "convolution")
    if set(proof["artifact_sha256"]) != required:
        raise ValueError("incomplete graph/request/sidecar hash inventory")
    for name, expected in proof["artifact_sha256"].items():
        path = (directory / name).resolve()
        if not path.is_relative_to(directory) or digest(path) != expected:
            raise ValueError(f"artifact changed: {name}")
    if digest(directory / "backend-camilladsp.yaml") != proof["yaml_sha256"]:
        raise ValueError("rendered YAML changed")
    expected_paths = {(i, o) for i in proof["input_channels"] for o in proof["output_channels"]}
    paths = [(p["input"], p["output"]) for p in proof["comparisons"]]
    if not expected_paths or set(paths) != expected_paths or len(paths) != len(expected_paths):
        raise ValueError("incomplete or duplicate input/output replay")
    if (len(proof["frequencies_hz"]) != 49 or any(
            not math.isfinite(f) or abs(f - 20 * 1000 ** (i / 48)) > 1e-8
            for i, f in enumerate(proof["frequencies_hz"]))):
        raise ValueError("unexpected frequency inventory")
    for comparison in proof["comparisons"]:
        if (comparison["frequency_count"] != 49
                or not math.isfinite(comparison["max_absolute_complex_error"])):
            raise ValueError("invalid transfer comparison")
        samples = comparison.get("complex_samples")
        if not isinstance(samples, list) or len(samples) != 49:
            raise ValueError("missing sampled complex transfer evidence")
        errors = []
        for sample in samples:
            if not isinstance(sample, dict):
                raise ValueError("invalid complex transfer sample")
            actual, expected = sample.get("actual"), sample.get("expected")
            if (not isinstance(actual, list) or not isinstance(expected, list)
                    or len(actual) != 2 or len(expected) != 2
                    or not all(isinstance(value, (int, float)) and math.isfinite(value)
                               for value in actual + expected)):
                raise ValueError("invalid complex transfer sample")
            error = math.hypot(actual[0] - expected[0], actual[1] - expected[1])
            limit = 0.0001 + 0.01 * math.hypot(*expected)
            if not math.isfinite(error) or error > limit:
                raise ValueError("sampled complex transfer exceeds tolerance")
            errors.append(error)
        if not math.isclose(comparison["max_absolute_complex_error"], max(errors),
                            rel_tol=1e-12, abs_tol=1e-15):
            raise ValueError("complex error summary disagrees with samples")
    return proof


def main(argv=None):
    argparse.ArgumentParser(description=__doc__).parse_args(argv)
    artifact = ROOT / "target/qa/roomeq-matrix-backend.json"
    artifact.parent.mkdir(parents=True, exist_ok=True)
    run_id = uuid.uuid4().hex
    record = {"status": "running", "run_id": run_id,
              "scope": "sampled_linear_electrical_transfer_not_acoustic_or_clipping_certification"}
    artifact.write_text(json.dumps(record, indent=2))
    try:
        matrix = ROOT / "target/qa/roomeq-parameter-matrix.json"
        rows = json.loads(matrix.read_text())
        if not isinstance(rows, list) or [r["row"] for r in rows] != list(range(16)):
            raise ValueError("completed ordered 16-row matrix required")
        record["matrix_sha256"] = digest(matrix)
        binary = shutil.which(os.environ.get("ROOMEQ_CAMILLADSP_BIN", "camilladsp"))
        if not binary:
            raise RuntimeError("CamillaDSP binary is required")
        environment = dict(os.environ, ROOMEQ_CAMILLADSP_BIN=binary,
                           ROOMEQ_PARAMETER_MATRIX=str(matrix), ROOMEQ_BACKEND_RUN_ID=run_id)
        command = ["cargo", "test", "-p", "roomeq-export", "--lib", TEST,
                   "--", "--exact", "--ignored", "--nocapture"]
        record.update(command=command, backend=binary, backend_sha256=digest(Path(binary)))
        process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, start_new_session=True)
        try:
            output, _ = process.communicate(timeout=600)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            output, _ = process.communicate()
            artifact.with_suffix(".log").write_text(output)
            raise RuntimeError("matrix backend exceeded 600-second process-group deadline")
        artifact.with_suffix(".log").write_text(output)
        print(output, end="")
        if (process.returncode or f"test {TEST} ... ok" not in output
                or "test result: ok. 1 passed; 0 failed; 0 ignored;" not in output
                or "skipping optional" in output):
            raise RuntimeError("required backend test failed or did not execute")
        executed = [int(x) for x in re.findall(r"matrix backend row (\d+):", output)]
        if executed != list(range(16)) or digest(matrix) != record["matrix_sha256"]:
            raise RuntimeError("matrix changed or exact row execution inventory is missing")
        record["rows"] = [validate_row(row, run_id) for row in rows]
        record["status"] = "passed"
    except Exception as error:
        record.update(status="failed", error=str(error))
        raise
    finally:
        artifact.write_text(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
