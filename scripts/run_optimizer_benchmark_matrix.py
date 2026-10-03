#!/usr/bin/env python3
"""Run the fixed optimizer benchmark as one bounded process per cell.

The Rust CLI owns fixture loading and optimization. This supervisor owns the
per-cell watchdog and evidence files; it does not aggregate away refusals or
failed optimizer outcomes. A recorded matrix is execution evidence, not an
assertion that every backend produced an acceptable correction.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import struct
import subprocess
import sys
import tempfile
import time
from typing import Any


INVENTORY_SCHEMA = "autoeq.optimizer_benchmark_cell_inventory/v1"
RUN_SCHEMA = "autoeq.optimizer_benchmark_matrix_run/v1"
RESULT_SCHEMA = "autoeq.optimizer_benchmark_cell_result/v1"
EXPECTED_CELLS = 841
RATE_CANARY_INVENTORY_SCHEMA = "autoeq.optimizer_rate_canary_cell_inventory/v1"
RATE_CANARY_SPEC_SCHEMA = "autoeq.optimizer_rate_canary_cell/v1"
RATE_CANARY_EXPECTED_CELLS = 112
RATE_CANARY_SAMPLE_RATES_HZ = (44_100, 96_000)
RATE_CANARY_CASE_IDS = frozenset({
    "analytic_headphone_peq",
    "asr_beyerdynamic_dt1990pro",
    "measured_stereo_8361a",
})
RATE_CANARY_REGISTERED_BACKENDS = frozenset({
    "autoeq:cobyla",
    "autoeq:cobra",
    "autoeq:isres",
    "autoeq:cmaes",
    "autoeq:bo",
    "autoeq:nsga2",
    "autoeq:nsga3",
    "autoeq:de",
    "mh:de",
    "mh:pso",
    "mh:rga",
    "mh:tlbo",
    "mh:firefly",
})
INVENTORY_WATCHDOG_SECONDS = 60.0
ALLOWED_OUTCOMES = {
    "completed",
    "budget_refused",
    "observer_stopped",
    "callback_unsupported",
    "timed_out",
    "backend_failure",
    "invalid_candidate",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def optional_sha256_file(path: Path) -> str | None:
    return sha256_file(path) if path.is_file() else None


def canonical_json_bytes(value: Any) -> bytes:
    """Preserve parsed key order, matching serde's struct field order."""
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON object key: {key}")
        value[key] = item
    return value


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(value, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as destination:
            destination.write(payload)
            destination.flush()
            os.fsync(destination.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def safe_cell_dir_name(index: int, cell_id: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", cell_id).strip("._-")[:120]
    return f"{index:04d}-{slug or 'cell'}"


def process_group(command: list[str], cwd: Path, environment: dict[str, str],
                  timeout_seconds: float, stdout_path: Path, stderr_path: Path
                  ) -> tuple[int | None, bool]:
    """Run one child, killing and reaping its process group at the watchdog."""
    with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=stdout,
            stderr=stderr,
            start_new_session=(os.name == "posix"),
        )
        timed_out = False
        try:
            process.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            timed_out = True
            kill_process_group(process)
            process.wait()
        finally:
            if process.poll() is None:
                kill_process_group(process)
                process.wait()
        return process.returncode, timed_out


def kill_process_group(process: subprocess.Popen[bytes]) -> None:
    try:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGKILL)
        else:
            process.kill()
    except ProcessLookupError:
        pass


def validate_rate_canary_profile(cells: list[dict[str, Any]]) -> None:
    """Check the declared 112-cell rate canary Cartesian product."""
    rates = set(RATE_CANARY_SAMPLE_RATES_HZ)
    cases = {spec["case_id"] for spec in cells}
    analytic_case = "analytic_headphone_peq"
    if cases != RATE_CANARY_CASE_IDS:
        raise ValueError("rate canary must contain the three fixed cases and analytic headphone case")

    def decoded_rate(spec: dict[str, Any]) -> int:
        bits = spec.get("sample_rate_hz_bits")
        if type(bits) is not int or not 0 <= bits < 2**64:
            raise ValueError(f"invalid sample rate in {spec.get('cell_id')}")
        value = struct.unpack(">d", bits.to_bytes(8, "big"))[0]
        if not math.isfinite(value) or value not in rates:
            raise ValueError(f"rate canary has unsupported sample rate {value!r}")
        return int(value)

    for spec in cells:
        rate = decoded_rate(spec)
        prefix = f"rate{rate}hz:"
        if not spec["cell_id"].startswith(prefix):
            raise ValueError(f"rate canary cell ID does not encode {rate} Hz: {spec['cell_id']}")
        f_min = struct.unpack(">d", spec["frequency_min_hz_bits"].to_bytes(8, "big"))[0]
        f_max = struct.unpack(">d", spec["frequency_max_hz_bits"].to_bytes(8, "big"))[0]
        if not (math.isfinite(f_min) and math.isfinite(f_max) and 0 < f_min <= f_max < rate / 2):
            raise ValueError(f"rate canary band is not strictly below Nyquist: {spec['cell_id']}")
        if spec.get("seed") != 42:
            raise ValueError(f"rate canary must use seed 42: {spec['cell_id']}")

    ordinary = [spec for spec in cells if spec["purpose"] == "ordinary"]
    backends = {spec["backend"] for spec in ordinary}
    if backends != RATE_CANARY_REGISTERED_BACKENDS:
        missing = sorted(RATE_CANARY_REGISTERED_BACKENDS - backends)
        unexpected = sorted(backends - RATE_CANARY_REGISTERED_BACKENDS)
        raise ValueError(
            "ordinary rate-canary backend set differs from the registered 13-backend plan; "
            f"missing={missing}, unexpected={unexpected}"
        )
    ordinary_keys = {
        (spec["case_id"], spec["backend"], decoded_rate(spec), spec["seed"],
         spec["root_search_budget"], spec["stage_search_budget"])
        for spec in ordinary
    }
    expected_ordinary = {
        (case, backend, rate, 42, 128, 128)
        for case in cases for backend in backends for rate in rates
    }
    if len(ordinary) != 78 or ordinary_keys != expected_ordinary:
        raise ValueError("ordinary rate-canary cells do not match the 13x3x2 cap-128 product")

    for purpose, stage_budget in (("adaptive", 128), ("refinement", 256)):
        rows = [spec for spec in cells if spec["purpose"] == purpose]
        keys = {
            (spec["case_id"], decoded_rate(spec), spec["backend"],
             spec["root_search_budget"], spec["stage_search_budget"])
            for spec in rows
        }
        expected = {
            (case, rate, "autoeq:de", 512, stage_budget)
            for case in cases for rate in rates
        }
        if len(rows) != 6 or keys != expected:
            raise ValueError(f"{purpose} rate-canary cells do not match the DE three-case/two-rate plan")

    pareto = [spec for spec in cells if spec["purpose"] == "pareto_front"]
    pareto_keys = {
        (spec["case_id"], decoded_rate(spec), spec["backend"], spec["root_search_budget"])
        for spec in pareto
    }
    expected_pareto = {
        (case, rate, backend, 512)
        for case in cases for rate in rates
        for backend in ("autoeq:nsga2", "autoeq:nsga3", "autoeq:bo")
    }
    if len(pareto) != 18 or pareto_keys != expected_pareto:
        raise ValueError("Pareto rate-canary cells do not match the three-backend product")

    observer_stop = [spec for spec in cells if spec["purpose"] == "observer_stop"]
    observer_unsupported = [spec for spec in cells if spec["purpose"] == "observer_unsupported"]
    expected_stops = {(analytic_case, rate, "autoeq:cobra", 128) for rate in rates}
    expected_refusals = {(analytic_case, rate, "autoeq:cobyla", 128) for rate in rates}
    if len(observer_stop) != 2 or {
        (spec["case_id"], decoded_rate(spec), spec["backend"], spec["root_search_budget"])
        for spec in observer_stop
    } != expected_stops:
        raise ValueError("observer-stop rate-canary cells do not match COBRA at both rates")
    if len(observer_unsupported) != 2 or {
        (spec["case_id"], decoded_rate(spec), spec["backend"], spec["root_search_budget"])
        for spec in observer_unsupported
    } != expected_refusals:
        raise ValueError("observer-refusal rate-canary cells do not match COBYLA at both rates")


def validate_inventory(payload: bytes, *, rate_canary: bool = False) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    try:
        inventory = json.loads(payload, object_pairs_hook=reject_duplicate_keys)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"CLI returned invalid inventory JSON: {error}") from error
    expected_schema = RATE_CANARY_INVENTORY_SCHEMA if rate_canary else INVENTORY_SCHEMA
    expected_cell_count = RATE_CANARY_EXPECTED_CELLS if rate_canary else EXPECTED_CELLS
    expected_spec_schema = RATE_CANARY_SPEC_SCHEMA if rate_canary else "autoeq.optimizer_benchmark_cell/v1"
    if not isinstance(inventory, dict) or inventory.get("schema") != expected_schema:
        raise ValueError("CLI returned an unknown cell inventory schema")
    cells = inventory.get("cells")
    declared_count = inventory.get("expected_cell_count")
    inventory_sha = inventory.get("spec_inventory_sha256")
    if (type(declared_count) is not int or declared_count != expected_cell_count
            or not isinstance(cells, list) or len(cells) != declared_count
            or not isinstance(inventory_sha, str) or len(inventory_sha) != 64):
        raise ValueError("CLI inventory is incomplete or has invalid identity fields")
    ids: list[str] = []
    for spec in cells:
        if not isinstance(spec, dict) or not isinstance(spec.get("cell_id"), str):
            raise ValueError("cell inventory contains a malformed spec")
        if spec.get("schema") != expected_spec_schema:
            raise ValueError(f"unknown cell spec schema for {spec.get('cell_id')}")
        watchdog = spec.get("process_watchdog_millis")
        if type(watchdog) is not int or watchdog <= 0:
            raise ValueError(f"invalid process watchdog for {spec['cell_id']}")
        if rate_canary:
            for field in ("case_id", "backend", "purpose"):
                if not isinstance(spec.get(field), str) or not spec[field]:
                    raise ValueError(f"rate-canary cell {spec['cell_id']} has invalid {field}")
            for field in ("sample_rate_hz_bits", "frequency_min_hz_bits", "frequency_max_hz_bits"):
                if type(spec.get(field)) is not int or not 0 <= spec[field] < 2**64:
                    raise ValueError(f"rate-canary cell {spec['cell_id']} has invalid {field}")
            if type(spec.get("seed")) is not int or type(spec.get("root_search_budget")) is not int \
                    or type(spec.get("stage_search_budget")) is not int:
                raise ValueError(f"rate-canary cell {spec['cell_id']} has invalid budget or seed")
        ids.append(spec["cell_id"])
    if len(set(ids)) != len(ids):
        raise ValueError("cell inventory contains duplicate IDs")
    computed_sha = sha256_bytes(canonical_json_bytes(cells))
    if computed_sha != inventory_sha:
        raise ValueError("cell inventory content does not match its declared SHA-256")
    if rate_canary:
        if inventory.get("sample_rate_scope") != "digital_filter_realization_hz; measurement_capture_rate_not_asserted":
            raise ValueError("rate-canary inventory does not distinguish realization rate from capture rate")
        if inventory.get("sample_rates_hz") != list(RATE_CANARY_SAMPLE_RATES_HZ):
            raise ValueError("rate-canary inventory sample-rate declaration is incorrect")
        validate_rate_canary_profile(cells)
    return inventory, cells


def validate_result(result_path: Path, spec: dict[str, Any], inventory: dict[str, Any],
                    binary_path: Path, binary_sha256: str) -> dict[str, Any]:
    try:
        result = json.loads(result_path.read_bytes(), object_pairs_hook=reject_duplicate_keys)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"cell did not write valid result JSON: {error}") from error
    expected_spec_sha = sha256_bytes(canonical_json_bytes(spec))
    if (not isinstance(result, dict)
            or result.get("schema") != RESULT_SCHEMA
            or result.get("cell_id") != spec["cell_id"]
            or result.get("spec") != spec
            or result.get("spec_sha256") != expected_spec_sha
            or result.get("matrix_expected_cell_count") != inventory["expected_cell_count"]
            or result.get("matrix_spec_inventory_sha256") != inventory["spec_inventory_sha256"]
            or result.get("executable_sha256") != binary_sha256
            or result.get("outcome") not in ALLOWED_OUTCOMES):
        raise ValueError(f"cell result identity or outcome does not match {spec['cell_id']}")
    try:
        reported_binary = Path(result["executable_path"]).resolve()
    except (KeyError, OSError, TypeError) as error:
        raise ValueError(f"cell omitted its executable identity: {error}") from error
    if reported_binary != binary_path:
        raise ValueError("cell reported a different executable path")
    return result


def run_one_cell(index: int, spec: dict[str, Any], inventory: dict[str, Any],
                 inventory_sha256: str, binary: Path, binary_sha256: str,
                 repository: Path, output: Path, environment: dict[str, str],
                 *, rate_canary: bool = False,
                 ) -> dict[str, Any]:
    cell_dir = output / "cells" / safe_cell_dir_name(index, spec["cell_id"])
    cell_dir.mkdir(parents=True, exist_ok=False)
    spec_path = cell_dir / "cell-spec.json"
    partial_result = cell_dir / "cell-result.partial.json"
    final_result = cell_dir / "cell-result.json"
    spec_bytes = canonical_json_bytes(spec) + b"\n"
    spec_path.write_bytes(spec_bytes)
    command = [str(binary), "--cell-spec", str(spec_path), "--output", str(partial_result)]
    if rate_canary:
        command.append("--rate-canary")
    watchdog_millis = spec.get("process_watchdog_millis")
    if type(watchdog_millis) is not int or watchdog_millis <= 0:
        raise ValueError(f"invalid watchdog for cell {spec['cell_id']}")
    started_at = utc_now()
    started = time.monotonic()
    return_code: int | None = None
    timed_out = False
    interrupted = False
    error: str | None = None
    result: dict[str, Any] | None = None
    try:
        return_code, timed_out = process_group(
            command,
            repository,
            environment,
            watchdog_millis / 1000,
            cell_dir / "stdout.log",
            cell_dir / "stderr.log",
        )
        if timed_out:
            error = f"child exceeded {watchdog_millis} ms process watchdog"
        elif return_code != 0:
            error = f"child exited with status {return_code}"
        else:
            result = validate_result(partial_result, spec, inventory, binary, binary_sha256)
            os.replace(partial_result, final_result)
    except KeyboardInterrupt:
        interrupted = True
        error = "supervisor interrupted while waiting; child process group was killed and reaped"
    except Exception as caught:
        error = f"{type(caught).__name__}: {caught}"
    elapsed_millis = int((time.monotonic() - started) * 1000)
    receipt = {
        "schema": "autoeq.optimizer_benchmark_cell_process/v1",
        "cell_id": spec["cell_id"],
        "spec_sha256": sha256_bytes(canonical_json_bytes(spec)),
        "inventory_sha256": inventory_sha256,
        "binary_path": str(binary),
        "binary_sha256": binary_sha256,
        "command": command,
        "working_directory": str(repository),
        "environment_sha256": environment_digest(environment),
        "cell_directory": str(cell_dir),
        "spec_path": str(spec_path),
        "stdout_path": str(cell_dir / "stdout.log"),
        "stderr_path": str(cell_dir / "stderr.log"),
        "partial_result_path": str(partial_result),
        "result_path": str(final_result) if final_result.is_file() else None,
        "started_at_utc": started_at,
        "exit_code": return_code,
        "watchdog_timed_out": timed_out,
        "elapsed_millis": elapsed_millis,
        "stdout_sha256": optional_sha256_file(cell_dir / "stdout.log"),
        "stderr_sha256": optional_sha256_file(cell_dir / "stderr.log"),
        "partial_result_retained": partial_result.is_file(),
        "result_sha256": sha256_file(final_result) if final_result.is_file() else None,
        "runner_status": (
            "interrupted" if interrupted else
            "result_recorded" if result is not None and error is None else "failed"
        ),
        "optimizer_outcome": result["outcome"] if result is not None else None,
        "error": error,
        "finished_at_utc": utc_now(),
    }
    receipt["cell_directory"] = str(cell_dir)
    write_json_atomic(cell_dir / "cell-process.json", receipt)
    return receipt


def environment_digest(environment: dict[str, str]) -> str:
    stable = "\0".join(f"{key}={environment[key]}" for key in sorted(environment))
    return hashlib.sha256(stable.encode("utf-8", errors="surrogateescape")).hexdigest()


def run(args: argparse.Namespace) -> int:
    rate_canary = bool(getattr(args, "rate_canary", False))
    binary = args.binary.resolve(strict=True)
    if not binary.is_file() or not os.access(binary, os.X_OK):
        raise ValueError("--binary must name an executable file")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    repository = Path(__file__).resolve().parents[1]
    environment = dict(os.environ)
    environment.update(
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
    )
    binary_sha256 = sha256_file(binary)
    list_command = [str(binary), "--list-cell-specs"]
    if rate_canary:
        list_command.append("--rate-canary")
    inventory_start = time.monotonic()
    code: int | None = None
    timed_out = False
    try:
        code, timed_out = process_group(
            list_command,
            repository,
            environment,
            INVENTORY_WATCHDOG_SECONDS,
            output / "cell-spec-list.stdout.json",
            output / "cell-spec-list.stderr.log",
        )
        if timed_out or code != 0:
            raise RuntimeError(f"spec-list process ended code={code} timeout={timed_out}")
        inventory_bytes = (output / "cell-spec-list.stdout.json").read_bytes()
        inventory, all_specs = validate_inventory(inventory_bytes, rate_canary=rate_canary)
    except KeyboardInterrupt:
        write_json_atomic(output / "matrix-run.json", {
            "schema": RUN_SCHEMA,
            "matrix_mode": "rate_canary" if rate_canary else "legacy_48khz",
            "status": "interrupted",
            "error": "supervisor interrupted while the CLI was listing cell specs",
            "binary_path": str(binary),
            "binary_sha256": binary_sha256,
            "cell_spec_list_command": list_command,
            "cell_spec_list_exit_code": code,
            "cell_spec_list_watchdog_timed_out": timed_out,
            "cell_spec_list_elapsed_millis": int((time.monotonic() - inventory_start) * 1000),
            "cell_spec_list_stdout_sha256": optional_sha256_file(
                output / "cell-spec-list.stdout.json"
            ),
            "cell_spec_list_stderr_sha256": optional_sha256_file(
                output / "cell-spec-list.stderr.log"
            ),
            "complete_matrix": False,
            "cells": [],
        })
        return 130
    except Exception as error:
        write_json_atomic(output / "matrix-run.json", {
            "schema": RUN_SCHEMA,
            "matrix_mode": "rate_canary" if rate_canary else "legacy_48khz",
            "status": "failed",
            "error": f"{type(error).__name__}: {error}",
            "binary_path": str(binary),
            "binary_sha256": binary_sha256,
            "cell_spec_list_command": list_command,
            "cell_spec_list_exit_code": code,
            "cell_spec_list_watchdog_timed_out": timed_out,
            "cell_spec_list_elapsed_millis": int((time.monotonic() - inventory_start) * 1000),
            "cell_spec_list_stdout_sha256": optional_sha256_file(
                output / "cell-spec-list.stdout.json"
            ),
            "cell_spec_list_stderr_sha256": optional_sha256_file(
                output / "cell-spec-list.stderr.log"
            ),
        })
        return 1

    inventory_sha256 = sha256_bytes(inventory_bytes)
    requested = args.cell_id
    available_ids = {spec["cell_id"] for spec in all_specs}
    selection_error = None
    if len(requested) != len(set(requested)):
        selection_error = "--cell-id was repeated"
    else:
        missing_ids = [cell_id for cell_id in requested if cell_id not in available_ids]
        if missing_ids:
            selection_error = f"unknown cell IDs: {', '.join(missing_ids)}"
    if selection_error:
        write_json_atomic(output / "matrix-run.json", {
            "schema": RUN_SCHEMA,
            "matrix_mode": "rate_canary" if rate_canary else "legacy_48khz",
            "status": "failed",
            "error": selection_error,
            "binary_path": str(binary),
            "binary_sha256": binary_sha256,
            "cell_spec_list_command": list_command,
            "cell_spec_list_exit_code": code,
            "cell_spec_list_watchdog_timed_out": timed_out,
            "cell_spec_list_stdout_sha256": inventory_sha256,
            "cell_spec_list_stderr_sha256": sha256_file(output / "cell-spec-list.stderr.log"),
            "spec_inventory_sha256": inventory["spec_inventory_sha256"],
            "expected_cell_count": inventory["expected_cell_count"],
            "selected_cell_ids": requested,
            "complete_matrix": False,
            "cells": [],
        })
        return 1
    selected_ids = set(requested)
    selected_specs = [
        spec for spec in all_specs if not selected_ids or spec["cell_id"] in selected_ids
    ]

    run_record: dict[str, Any] = {
        "schema": RUN_SCHEMA,
        "matrix_mode": "rate_canary" if rate_canary else "legacy_48khz",
        "status": "running",
        "started_at_utc": utc_now(),
        "repository": str(repository),
        "binary_path": str(binary),
        "binary_sha256": binary_sha256,
        "cell_spec_list_command": list_command,
        "cell_spec_list_exit_code": code,
        "cell_spec_list_watchdog_timed_out": timed_out,
        "cell_spec_list_stdout_sha256": inventory_sha256,
        "cell_spec_list_stderr_sha256": sha256_file(output / "cell-spec-list.stderr.log"),
        "cell_spec_list_elapsed_millis": int((time.monotonic() - inventory_start) * 1000),
        "spec_inventory_sha256": inventory["spec_inventory_sha256"],
        "expected_cell_count": inventory["expected_cell_count"],
        "selected_cell_count": len(selected_specs),
        "selected_cell_ids": [spec["cell_id"] for spec in selected_specs],
        "planned_cell_ids": [spec["cell_id"] for spec in selected_specs],
        "attempted_cell_ids": [],
        "valid_result_cell_ids": [],
        "runner_failed_cell_ids": [],
        "missing_cell_ids": [spec["cell_id"] for spec in selected_specs],
        "unresolved_result_cell_ids": [spec["cell_id"] for spec in selected_specs],
        "complete_matrix": False,
        "environment_sha256": environment_digest(environment),
        "environment_overrides": {
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
        },
        "cells": [],
    }
    write_json_atomic(output / "matrix-run.json", run_record)
    failures = 0
    outcomes: Counter[str] = Counter()
    try:
        for index, spec in enumerate(selected_specs, start=1):
            try:
                receipt = run_one_cell(
                    index,
                    spec,
                    inventory,
                    inventory["spec_inventory_sha256"],
                    binary,
                    binary_sha256,
                    repository,
                    output,
                    environment,
                    rate_canary=rate_canary,
                )
            except Exception as error:
                receipt = {
                    "cell_id": spec["cell_id"],
                    "spec_sha256": sha256_bytes(canonical_json_bytes(spec)),
                    "runner_status": "failed",
                    "optimizer_outcome": None,
                    "error": f"{type(error).__name__}: {error}",
                    "cell_directory": None,
                }
            run_record["cells"].append(receipt)
            run_record["attempted_cell_ids"] = [
                item["cell_id"] for item in run_record["cells"]
            ]
            run_record["valid_result_cell_ids"] = [
                item["cell_id"] for item in run_record["cells"]
                if item["runner_status"] == "result_recorded"
            ]
            run_record["runner_failed_cell_ids"] = [
                item["cell_id"] for item in run_record["cells"]
                if item["runner_status"] != "result_recorded"
            ]
            attempted_ids = set(run_record["attempted_cell_ids"])
            valid_ids = set(run_record["valid_result_cell_ids"])
            run_record["missing_cell_ids"] = [
                cell_id for cell_id in run_record["planned_cell_ids"]
                if cell_id not in attempted_ids
            ]
            run_record["unresolved_result_cell_ids"] = [
                cell_id for cell_id in run_record["planned_cell_ids"]
                if cell_id not in valid_ids
            ]
            if receipt["runner_status"] == "interrupted":
                run_record.update(
                    status="interrupted",
                    finished_at_utc=utc_now(),
                    completed_cell_count=len(run_record["cells"]),
                    runner_failure_count=failures + 1,
                    complete_matrix=False,
                )
                write_json_atomic(output / "matrix-run.json", run_record)
                return 130
            if receipt["runner_status"] != "result_recorded":
                failures += 1
            elif receipt["optimizer_outcome"]:
                outcomes[receipt["optimizer_outcome"]] += 1
            run_record["completed_cell_count"] = len(run_record["cells"])
            run_record["runner_failure_count"] = failures
            run_record["optimizer_outcome_counts"] = dict(sorted(outcomes.items()))
            write_json_atomic(output / "matrix-run.json", run_record)
    except KeyboardInterrupt:
        run_record.update(status="interrupted", finished_at_utc=utc_now())
        write_json_atomic(output / "matrix-run.json", run_record)
        return 130

    all_cells_selected = len(selected_specs) == inventory["expected_cell_count"]
    all_cells_recorded = len(run_record["cells"]) == len(selected_specs)
    run_record.update(
        status="recorded" if failures == 0 and all_cells_recorded else "incomplete",
        finished_at_utc=utc_now(),
        completed_cell_count=len(run_record["cells"]),
        runner_failure_count=failures,
        optimizer_outcome_counts=dict(sorted(outcomes.items())),
        complete_matrix=all_cells_selected and all_cells_recorded and failures == 0,
        note=(
            "complete_matrix means every planned process produced a validated typed result; "
            "optimizer refusals, timeouts, failures, and invalid candidates remain outcomes"
        ),
    )
    write_json_atomic(output / "matrix-run.json", run_record)
    return 0 if failures == 0 and all_cells_recorded else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", required=True, type=Path, help="built optimizer-benchmark executable")
    parser.add_argument("--output", required=True, type=Path, help="new evidence directory; must not exist")
    parser.add_argument(
        "--rate-canary", action="store_true",
        help="run the separate 44.1/96 kHz realization-rate inventory",
    )
    parser.add_argument(
        "--cell-id",
        action="append",
        default=[],
        help="run only this exact ID (repeatable for bounded smoke runs)",
    )
    args = parser.parse_args(argv)
    try:
        return run(args)
    except Exception as error:
        print(f"optimizer matrix runner failed: {type(error).__name__}: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
