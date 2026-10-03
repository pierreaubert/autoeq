#!/usr/bin/env python3
"""Validate and summarize AutoEQ optimizer benchmark matrix evidence.

The default mode writes an explicit partial or complete summary. Pass
``--require-complete`` to make identity, receipt, schema, counter, metric, and
feasibility failures return a nonzero status. This tool never changes the run
directory; its output directory must be new.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics
import struct
import sys
from typing import Any

if __package__:
    from . import run_optimizer_benchmark_matrix as runner
else:
    import run_optimizer_benchmark_matrix as runner


ANALYSIS_SCHEMA = "autoeq.optimizer_benchmark_matrix_analysis/v2"
CELL_SPEC_SCHEMA = "autoeq.optimizer_benchmark_cell/v1"
PROCESS_SCHEMA = "autoeq.optimizer_benchmark_cell_process/v1"
ALLOWED_OUTCOMES = runner.ALLOWED_OUTCOMES
COUNT_FIELDS = (
    "evaluation_budget",
    "evaluations_started",
    "evaluations_completed",
    "evaluations_failed",
    "evaluations_refused",
    "evaluations_in_flight",
    "component_evaluations_started",
    "component_evaluations_completed",
    "validation_evaluations_started",
    "validation_evaluations_completed",
    "validation_evaluations_failed",
    "validation_evaluations_refused",
    "validation_evaluations_in_flight",
    "validation_component_evaluations_started",
    "validation_component_evaluations_completed",
)
FLAG_FIELDS = ("cancellation_requested", "deadline_reached", "budget_exhausted")
HASH_RE = set("0123456789abcdef")
FEASIBILITY_TOLERANCE = 1e-9


class AnalysisInputError(ValueError):
    """The run cannot be interpreted as a matrix snapshot."""


def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def reject_nonfinite_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number is forbidden: {value}")


def parse_finite_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"non-finite JSON number is forbidden: {value}")
    return parsed


def parse_json(data: bytes, label: str) -> Any:
    try:
        return json.loads(
            data,
            object_pairs_hook=reject_duplicate_keys,
            parse_float=parse_finite_float,
            parse_constant=reject_nonfinite_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise AnalysisInputError(f"invalid {label}: {error}") from error


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def finite(value: Any) -> bool:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def exact_int(value: Any) -> bool:
    return type(value) is int


def valid_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in HASH_RE for character in value)
    )


def quantile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def distribution(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"n": 0, "min": None, "p25": None, "median": None, "p75": None, "max": None}
    return {
        "n": len(values),
        "min": min(values),
        "p25": quantile(values, 0.25),
        "median": statistics.median(values),
        "p75": quantile(values, 0.75),
        "max": max(values),
    }


def add_problem(problems: list[dict[str, str]], code: str, message: str,
                cell_id: str | None = None) -> None:
    record = {"code": code, "message": message}
    if cell_id is not None:
        record["cell_id"] = cell_id
    problems.append(record)


def validate_inventory(payload: bytes) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    inventory = parse_json(payload, "printed cell-spec inventory")
    if not isinstance(inventory, dict) or inventory.get("schema") != runner.INVENTORY_SCHEMA:
        raise AnalysisInputError("printed inventory has an unsupported schema")
    cells = inventory.get("cells")
    expected = inventory.get("expected_cell_count")
    if not exact_int(expected) or expected <= 0 or not isinstance(cells, list) or expected != len(cells):
        raise AnalysisInputError("printed inventory count does not match its cells")
    declared_hash = inventory.get("spec_inventory_sha256")
    if not valid_sha256(declared_hash):
        raise AnalysisInputError("printed inventory has an invalid SHA-256 field")
    if sha256_bytes(canonical_json_bytes(cells)) != declared_hash:
        raise AnalysisInputError("printed inventory content does not match its SHA-256")
    ids: list[str] = []
    for spec in cells:
        if not isinstance(spec, dict) or spec.get("schema") != CELL_SPEC_SCHEMA:
            raise AnalysisInputError("printed inventory contains a malformed cell spec")
        cell_id = spec.get("cell_id")
        if not isinstance(cell_id, str) or not cell_id.strip():
            raise AnalysisInputError("printed inventory contains an empty cell ID")
        ids.append(cell_id)
        for field in ("manifest_sha256", "fixture_sha256"):
            if not valid_sha256(spec.get(field)):
                raise AnalysisInputError(f"cell {cell_id} has invalid {field}")
        for field in ("root_search_budget", "stage_search_budget", "filter_count", "seed"):
            minimum = 0 if field == "seed" else 1
            if not exact_int(spec.get(field)) or spec[field] < minimum:
                raise AnalysisInputError(f"cell {cell_id} has invalid integer field {field}")
        if spec["stage_search_budget"] > spec["root_search_budget"]:
            raise AnalysisInputError(f"cell {cell_id} stage budget exceeds root budget")
        for field in ("cooperative_deadline_millis", "process_watchdog_millis", "population_size"):
            if not exact_int(spec.get(field)) or spec[field] <= 0:
                raise AnalysisInputError(f"cell {cell_id} has invalid positive integer field {field}")
        for field in ("case_id", "backend", "purpose", "optimizer_version", "multi_strategy", "local_refiner"):
            if not isinstance(spec.get(field), str) or not spec[field].strip():
                raise AnalysisInputError(f"cell {cell_id} has invalid string field {field}")
        if type(spec.get("bo_ehvi")) is not bool:
            raise AnalysisInputError(f"cell {cell_id} has invalid bo_ehvi flag")
        for field in (
            "sample_rate_hz_bits", "frequency_min_hz_bits", "frequency_max_hz_bits",
            "min_q_bits", "max_q_bits", "min_gain_db_bits", "max_gain_db_bits",
        ):
            if not exact_int(spec.get(field)) or not 0 <= spec[field] < 2**64:
                raise AnalysisInputError(f"cell {cell_id} has invalid float-bit field {field}")
    if len(set(ids)) != len(ids):
        raise AnalysisInputError("printed inventory contains duplicate cell IDs")
    for spec in cells:
        decoded = {
            field: struct.unpack(">d", spec[field].to_bytes(8, "big"))[0]
            for field in (
                "sample_rate_hz_bits", "frequency_min_hz_bits", "frequency_max_hz_bits",
                "min_q_bits", "max_q_bits", "min_gain_db_bits", "max_gain_db_bits",
            )
        }
        if any(not math.isfinite(value) for value in decoded.values()):
            raise AnalysisInputError(f"cell {spec['cell_id']} has non-finite encoded bounds")
        if not (
            decoded["sample_rate_hz_bits"] > 0
            and 0 < decoded["frequency_min_hz_bits"] <= decoded["frequency_max_hz_bits"] < decoded["sample_rate_hz_bits"] / 2
            and 0 < decoded["min_q_bits"] <= decoded["max_q_bits"]
            and decoded["min_gain_db_bits"] <= decoded["max_gain_db_bits"]
        ):
            raise AnalysisInputError(f"cell {spec['cell_id']} has inconsistent encoded constraints")
    return inventory, cells


def safe_cell_directory(cells_root: Path, receipt: dict[str, Any]) -> Path:
    raw = receipt.get("cell_directory")
    if not isinstance(raw, str) or not raw:
        raise ValueError("receipt has no cell directory")
    path = Path(raw).resolve(strict=True)
    root = cells_root.resolve(strict=True)
    if path.parent != root or not path.is_dir():
        raise ValueError("cell directory is outside the run's cells directory")
    return path


def verify_child_file(receipt: dict[str, Any], path_field: str, hash_field: str,
                      cell_directory: Path, problems: list[dict[str, str]],
                      cell_id: str, code: str) -> Path | None:
    raw_path = receipt.get(path_field)
    expected_hash = receipt.get(hash_field)
    if not isinstance(raw_path, str) or not valid_sha256(expected_hash):
        add_problem(problems, code, f"invalid {path_field} or {hash_field}", cell_id)
        return None
    try:
        path = Path(raw_path).resolve(strict=True)
        if path.parent != cell_directory or not path.is_file():
            raise ValueError("path is outside the cell directory or not a file")
        if sha256_file(path) != expected_hash:
            raise ValueError("file SHA-256 differs from receipt")
        return path
    except (OSError, ValueError) as error:
        add_problem(problems, code, f"{path_field}: {error}", cell_id)
        return None


def validate_counter_snapshot(snapshot: Any, label: str, expected_budget: int,
                              problems: list[dict[str, str]], cell_id: str) -> dict[str, int | bool] | None:
    if not isinstance(snapshot, dict):
        add_problem(problems, "counter_schema", f"{label} is not an object", cell_id)
        return None
    values: dict[str, int | bool] = {}
    for field in COUNT_FIELDS:
        value = snapshot.get(field)
        if not exact_int(value) or value < 0:
            add_problem(problems, "counter_schema", f"{label}.{field} is not a non-negative integer", cell_id)
        else:
            values[field] = value
    for field in FLAG_FIELDS:
        value = snapshot.get(field)
        if type(value) is not bool:
            add_problem(problems, "counter_schema", f"{label}.{field} is not boolean", cell_id)
        else:
            values[field] = value
    if len(values) != len(COUNT_FIELDS) + len(FLAG_FIELDS):
        return None
    if values["evaluation_budget"] != expected_budget:
        add_problem(problems, "counter_budget", f"{label} budget differs from its declared quota", cell_id)
    if values["evaluations_started"] != (
        values["evaluations_completed"] + values["evaluations_failed"] + values["evaluations_in_flight"]
    ):
        add_problem(problems, "counter_conservation", f"{label} search admissions do not conserve", cell_id)
    if values["validation_evaluations_started"] != (
        values["validation_evaluations_completed"] + values["validation_evaluations_failed"]
        + values["validation_evaluations_in_flight"]
    ):
        add_problem(problems, "counter_conservation", f"{label} validation admissions do not conserve", cell_id)
    if values["evaluations_in_flight"] != 0 or values["validation_evaluations_in_flight"] != 0:
        add_problem(problems, "counter_in_flight", f"{label} returned with evaluations in flight", cell_id)
    if values["evaluations_started"] > expected_budget:
        add_problem(problems, "counter_budget", f"{label} exceeded its search quota", cell_id)
    for started, completed in (
        ("component_evaluations_started", "component_evaluations_completed"),
        ("validation_component_evaluations_started", "validation_component_evaluations_completed"),
    ):
        if values[started] < values[completed]:
            add_problem(problems, "counter_conservation", f"{label} has more completed components than started", cell_id)
    return values


def validate_stage_evidence(result: dict[str, Any], spec: dict[str, Any],
                            problems: list[dict[str, str]]) -> dict[str, Any]:
    cell_id = spec["cell_id"]
    rows = result.get("stage_evidence")
    summary: dict[str, Any] = {
        "stage_count": 0,
        "search_started_by_stage": [],
        "search_started_total_by_stages": 0,
    }
    if not isinstance(rows, list):
        add_problem(problems, "stage_schema", "stage_evidence is not an array", cell_id)
        return summary
    summary["stage_count"] = len(rows)
    root_budget = spec["root_search_budget"]
    stage_budget = spec["stage_search_budget"]
    previous_root_started = 0
    stage_search_total = 0
    for index, stage in enumerate(rows):
        label = f"stage {index}"
        if not isinstance(stage, dict):
            add_problem(problems, "stage_schema", f"{label} is not an object", cell_id)
            continue
        prior_root_started = previous_root_started
        remaining_before_stage = max(0, root_budget - prior_root_started)
        run_counts = validate_counter_snapshot(
            stage.get("run_counters_at_dispatch_return"),
            f"{label} run counters", root_budget, problems, cell_id,
        )
        local_raw = stage.get("stage_counters_at_dispatch_return")
        local_counts = None
        if local_raw is None:
            add_problem(problems, "stage_counters_missing", f"{label} omits stage-local counters", cell_id)
        else:
            local_counts = validate_counter_snapshot(
                local_raw, f"{label} stage counters", stage_budget, problems, cell_id
            )
            if local_counts is not None:
                stage_search_total += int(local_counts["evaluations_started"])
                summary["search_started_by_stage"].append(local_counts["evaluations_started"])
        profile = stage.get("profile")
        if profile is None:
            add_problem(problems, "profile_missing", f"{label} omits its resolved budget profile", cell_id)
        else:
            if not isinstance(profile, dict):
                add_problem(problems, "profile_schema", f"{label} profile is not an object", cell_id)
            else:
                dimension = profile.get("parameter_dimension")
                lower = profile.get("lower_bounds")
                upper = profile.get("upper_bounds")
                if not exact_int(dimension) or dimension <= 0 or not isinstance(lower, list) or not isinstance(upper, list):
                    add_problem(problems, "profile_bounds", f"{label} has invalid dimension/bounds", cell_id)
                elif len(lower) != dimension or len(upper) != dimension or any(
                    not finite(lo) or not finite(hi) or lo > hi for lo, hi in zip(lower, upper)
                ):
                    add_problem(problems, "profile_bounds", f"{label} bounds are malformed or non-finite", cell_id)
                quota = profile.get("stage_evaluation_budget")
                effective = profile.get("effective_evaluation_limit")
                if not exact_int(quota) or quota != stage_budget:
                    add_problem(problems, "profile_budget", f"{label} stage quota differs from cell spec", cell_id)
                if not exact_int(effective) or effective < 0 or effective > stage_budget:
                    add_problem(problems, "profile_budget", f"{label} effective limit exceeds stage quota", cell_id)
                elif effective > remaining_before_stage:
                    add_problem(problems, "profile_budget", f"{label} effective limit exceeds remaining root quota", cell_id)
                if not isinstance(profile.get("resolved_backend"), str) or not profile["resolved_backend"]:
                    add_problem(problems, "profile_identity", f"{label} omits resolved backend identity", cell_id)
        if run_counts is not None:
            current = int(run_counts["evaluations_started"])
            if current < prior_root_started:
                add_problem(problems, "counter_monotonicity", f"{label} root search count decreased", cell_id)
            delta = current - prior_root_started
            local_started = int(local_counts["evaluations_started"]) if local_counts is not None else 0
            if delta != local_started:
                add_problem(problems, "counter_attribution", f"{label} local admissions do not match cumulative root delta", cell_id)
            previous_root_started = current
            if local_counts is not None and current < int(local_counts["evaluations_started"]):
                add_problem(problems, "counter_attribution", f"{label} local count exceeds root count", cell_id)
    summary["search_started_total_by_stages"] = stage_search_total
    return summary


def pair_metrics(rows: Any, label: str, cell_id: str,
                 problems: list[dict[str, str]]) -> list[tuple[str, float, float]] | None:
    if not isinstance(rows, list):
        add_problem(problems, "metric_schema", f"{label} is not an array", cell_id)
        return None
    output: list[tuple[str, float, float]] = []
    seen: set[str] = set()
    valid = True
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            add_problem(problems, "metric_schema", f"{label}[{index}] is not an object", cell_id)
            valid = False
            continue
        measurement_id = row.get("measurement_id")
        if not isinstance(measurement_id, str) or not measurement_id.strip():
            add_problem(problems, "metric_identity", f"{label}[{index}] has empty measurement ID", cell_id)
            valid = False
            continue
        if measurement_id in seen:
            add_problem(problems, "metric_duplicate", f"{label} repeats measurement ID {measurement_id!r}", cell_id)
            valid = False
            continue
        seen.add(measurement_id)
        baseline = row.get("baseline_loss")
        final = row.get("final_loss")
        if not finite(baseline) or not finite(final):
            add_problem(problems, "metric_nonfinite", f"{label}[{index}] baseline/final loss must be finite", cell_id)
            valid = False
            continue
        output.append((measurement_id, float(baseline), float(final)))
    return output if valid else None


def float_from_bits(spec: dict[str, Any], field: str, cell_id: str,
                    problems: list[dict[str, str]]) -> float | None:
    raw = spec.get(field)
    if not exact_int(raw) or not 0 <= raw < 2**64:
        add_problem(problems, "constraint_spec", f"{field} is not a u64 bit pattern", cell_id)
        return None
    value = struct.unpack(">d", raw.to_bytes(8, "big"))[0]
    if not math.isfinite(value):
        add_problem(problems, "constraint_spec", f"{field} decodes to a non-finite value", cell_id)
        return None
    return value


def validate_realized(result: dict[str, Any], spec: dict[str, Any],
                      problems: list[dict[str, str]]) -> dict[str, Any] | None:
    cell_id = spec["cell_id"]
    realized = result.get("realized")
    if not isinstance(realized, dict):
        add_problem(problems, "realized_missing", "completed candidate has no realized summary", cell_id)
        return None
    rows = realized.get("filter_parameters_hz_q_gain_db")
    active = realized.get("active_filter_count")
    top_active = result.get("active_filter_count")
    if not isinstance(rows, list) or not exact_int(active) or active != len(rows) or top_active != active:
        add_problem(problems, "realized_shape", "filter rows and active-filter counts disagree", cell_id)
        return None
    valid = True
    if active > spec["filter_count"]:
        add_problem(problems, "realized_filter_count", "active filters exceed the requested filter limit", cell_id)
        valid = False
    sample_rate = float_from_bits(spec, "sample_rate_hz_bits", cell_id, problems)
    f_min = float_from_bits(spec, "frequency_min_hz_bits", cell_id, problems)
    f_max = float_from_bits(spec, "frequency_max_hz_bits", cell_id, problems)
    q_min = float_from_bits(spec, "min_q_bits", cell_id, problems)
    q_max = float_from_bits(spec, "max_q_bits", cell_id, problems)
    gain_min = float_from_bits(spec, "min_gain_db_bits", cell_id, problems)
    gain_max = float_from_bits(spec, "max_gain_db_bits", cell_id, problems)
    limits = (sample_rate, f_min, f_max, q_min, q_max, gain_min, gain_max)
    if any(value is None for value in limits):
        valid = False
    elif not (sample_rate > 0 and 0 < f_min <= f_max < sample_rate / 2 and 0 < q_min <= q_max and gain_min <= gain_max):
        add_problem(problems, "constraint_spec", "cell frequency/Q/gain bounds are inconsistent", cell_id)
        valid = False
    frequencies: list[float] = []
    qs: list[float] = []
    gains: list[float] = []
    for index, row in enumerate(rows):
        if not isinstance(row, list) or len(row) != 3 or any(not finite(value) for value in row):
            add_problem(problems, "realized_nonfinite", f"filter row {index} is malformed/non-finite", cell_id)
            valid = False
            continue
        frequency, q, gain = (float(value) for value in row)
        frequencies.append(frequency)
        qs.append(q)
        gains.append(gain)
        if sample_rate is not None and not 0 < frequency < sample_rate / 2:
            add_problem(problems, "realized_frequency", f"filter {index} is outside Nyquist bounds", cell_id)
            valid = False
        if f_min is not None and f_max is not None and not (
            f_min - FEASIBILITY_TOLERANCE <= frequency <= f_max + FEASIBILITY_TOLERANCE
        ):
            add_problem(problems, "realized_frequency", f"filter {index} is outside requested correction band", cell_id)
            valid = False
        if q_min is not None and q_max is not None and not (
            q_min - FEASIBILITY_TOLERANCE <= q <= q_max + FEASIBILITY_TOLERANCE
        ):
            add_problem(problems, "realized_q", f"filter {index} is outside requested Q bounds", cell_id)
            valid = False
        if gain_min is not None and gain_max is not None and not (
            gain_min - FEASIBILITY_TOLERANCE <= gain <= gain_max + FEASIBILITY_TOLERANCE
        ):
            add_problem(problems, "realized_gain", f"filter {index} is outside requested gain bounds", cell_id)
            valid = False
    scalar_expectations = {
        "max_abs_gain_db": max(map(abs, gains), default=0.0),
        "max_q": max(qs, default=0.0),
        "transfer_db_min": None,
        "transfer_db_max": None,
        "transfer_db_rms": None,
        "maximum_bound_violation": None,
    }
    for field in ("max_abs_gain_db", "max_q", "transfer_db_min", "transfer_db_max", "transfer_db_rms", "maximum_bound_violation"):
        if not finite(realized.get(field)):
            add_problem(problems, "realized_nonfinite", f"realized.{field} is missing/non-finite", cell_id)
            valid = False
    for field, expected in scalar_expectations.items():
        if expected is not None and finite(realized.get(field)) and not math.isclose(
            float(realized[field]), expected, rel_tol=1e-10, abs_tol=FEASIBILITY_TOLERANCE
        ):
            add_problem(problems, "realized_summary_mismatch", f"realized.{field} disagrees with emitted filter rows", cell_id)
            valid = False
    if finite(realized.get("maximum_bound_violation")) and realized["maximum_bound_violation"] > FEASIBILITY_TOLERANCE:
        add_problem(problems, "realized_infeasible", "reported maximum bound violation exceeds tolerance", cell_id)
        valid = False
    if finite(realized.get("transfer_db_rms")) and realized["transfer_db_rms"] < 0:
        add_problem(problems, "realized_transfer", "transfer RMS cannot be negative", cell_id)
        valid = False
    if finite(realized.get("transfer_db_min")) and finite(realized.get("transfer_db_max")) and realized["transfer_db_min"] > realized["transfer_db_max"]:
        add_problem(problems, "realized_transfer", "minimum transfer exceeds maximum transfer", cell_id)
        valid = False
    parameters = result.get("parameters_log10_hz_q_gain_db")
    if not isinstance(parameters, list) or any(not finite(value) for value in parameters):
        add_problem(problems, "candidate_parameters", "completed candidate parameter vector is missing/non-finite", cell_id)
        valid = False
    elif len(parameters) != 3 * active:
        add_problem(problems, "candidate_parameters", "completed parameter vector dimension differs from emitted active-filter count", cell_id)
        valid = False
    if not finite(result.get("optimizer_loss_after_engine_normalization")):
        add_problem(problems, "optimizer_loss", "completed candidate optimizer loss is missing/non-finite", cell_id)
        valid = False
    return realized if valid else None


def validate_completed_quality(result: dict[str, Any], spec: dict[str, Any],
                               problems: list[dict[str, str]]) -> dict[str, Any] | None:
    cell_id = spec["cell_id"]
    local_problems: list[dict[str, str]] = []
    training = pair_metrics(result.get("training_source_metrics"), "training_source_metrics", cell_id, local_problems)
    heldout = pair_metrics(result.get("held_out_source_metrics"), "held_out_source_metrics", cell_id, local_problems)
    selected = heldout if isinstance(heldout, list) and heldout else training
    expected = result.get("comparison_measurements_expected")
    available = result.get("comparison_measurements_available")
    comparison_loss = result.get("comparison_loss")
    if not exact_int(expected) or expected <= 0 or not exact_int(available):
        add_problem(local_problems, "comparison_count", "comparison counts must be positive integers", cell_id)
    elif not isinstance(selected, list) or len(selected) != expected or available != expected:
        add_problem(local_problems, "comparison_count", "comparison pair is missing or contains malformed/duplicate measurements", cell_id)
    if not finite(comparison_loss):
        add_problem(local_problems, "comparison_loss", "comparison_loss is missing or non-finite", cell_id)
    elif isinstance(selected, list) and selected:
        expected_loss = max(row[2] for row in selected)
        if not math.isclose(float(comparison_loss), expected_loss, rel_tol=1e-10, abs_tol=1e-12):
            add_problem(local_problems, "comparison_loss", "reported comparison loss does not match worst paired final loss", cell_id)
    if training is None or not training:
        add_problem(local_problems, "training_metrics", "completed result needs finite unique training pairs", cell_id)
    if heldout is None:
        add_problem(local_problems, "heldout_metrics", "held-out metric rows are malformed", cell_id)
    root_counters = result.get("root_counters")
    if not isinstance(root_counters, dict) or not exact_int(root_counters.get("evaluations_started")) or not exact_int(
        root_counters.get("validation_evaluations_started")
    ):
        add_problem(local_problems, "counter_schema", "completed result has malformed root evaluation counters", cell_id)
    realized = validate_realized(result, spec, local_problems)
    if local_problems:
        problems.extend(local_problems)
        return None
    assert isinstance(training, list) and isinstance(heldout, list)
    primary = heldout if heldout else training
    baseline_worst = max(row[1] for row in primary)
    final_worst = max(row[2] for row in primary)
    return {
        "baseline_worst": baseline_worst,
        "final_worst": final_worst,
        "delta_worst": final_worst - baseline_worst,
        "comparison_loss": float(comparison_loss),
        "training_count": len(training),
        "heldout_count": len(heldout),
        "heldout_deltas": [final - baseline for _, baseline, final in heldout],
        "heldout_worst_delta": (
            max(row[2] for row in heldout) - max(row[1] for row in heldout)
            if heldout else None
        ),
        "optimizer_loss": float(result["optimizer_loss_after_engine_normalization"]),
        "search_evaluations": root_counters["evaluations_started"],
        "validation_evaluations": root_counters["validation_evaluations_started"],
        "engine_elapsed_millis": result.get("engine_elapsed_millis"),
        "active_filter_count": realized["active_filter_count"],
        "max_abs_gain_db": realized["max_abs_gain_db"],
        "max_q": realized["max_q"],
        "transfer_db_rms": realized["transfer_db_rms"],
    }


def verify_result(receipt: dict[str, Any], spec: dict[str, Any], inventory_sha: str,
                  inventory_count: int, binary_path: Path, binary_sha: str,
                  repository_path: Path | None, environment_sha: str | None, cells_root: Path,
                  problems: list[dict[str, str]]) -> tuple[dict[str, Any] | None, dict[str, Any] | None, bool]:
    cell_id = spec["cell_id"]
    problem_start = len(problems)
    if receipt.get("schema") != PROCESS_SCHEMA:
        add_problem(problems, "receipt_schema", "unexpected child process receipt schema", cell_id)
    if receipt.get("cell_id") != cell_id:
        add_problem(problems, "receipt_identity", "receipt cell ID differs from plan", cell_id)
    expected_spec_sha = sha256_bytes(canonical_json_bytes(spec))
    for key, expected in (("spec_sha256", expected_spec_sha), ("inventory_sha256", inventory_sha),
                          ("binary_sha256", binary_sha)):
        if receipt.get(key) != expected:
            add_problem(problems, "receipt_identity", f"receipt {key} does not match matrix inputs", cell_id)
    try:
        if Path(receipt.get("binary_path", "")).resolve(strict=True) != binary_path:
            add_problem(problems, "receipt_identity", "receipt binary path differs from matrix binary path", cell_id)
    except (OSError, TypeError, ValueError):
        add_problem(problems, "receipt_identity", "receipt has invalid binary path", cell_id)
    if environment_sha is not None and receipt.get("environment_sha256") != environment_sha:
        add_problem(problems, "receipt_identity", "receipt environment hash differs from matrix", cell_id)
    try:
        cell_dir = safe_cell_directory(cells_root, receipt)
    except (OSError, ValueError) as error:
        add_problem(problems, "receipt_path", str(error), cell_id)
        return None, None, False
    process_path = cell_dir / "cell-process.json"
    try:
        process = parse_json(process_path.read_bytes(), f"{cell_id} on-disk process receipt")
        if process != receipt:
            add_problem(problems, "receipt_mismatch", "embedded receipt differs from on-disk process receipt", cell_id)
    except (OSError, AnalysisInputError) as error:
        add_problem(problems, "receipt_missing", str(error), cell_id)
    stdout = verify_child_file(receipt, "stdout_path", "stdout_sha256", cell_dir, problems, cell_id, "stdout_integrity")
    stderr = verify_child_file(receipt, "stderr_path", "stderr_sha256", cell_dir, problems, cell_id, "stderr_integrity")
    _ = stdout, stderr
    try:
        spec_path = Path(receipt.get("spec_path", "")).resolve(strict=True)
        if spec_path.parent != cell_dir:
            raise ValueError("spec file is outside cell directory")
        spec_file = parse_json(spec_path.read_bytes(), f"{cell_id} child spec")
        if spec_file != spec or canonical_json_bytes(spec_file) != canonical_json_bytes(spec):
            raise ValueError("child spec differs from printed inventory")
    except (OSError, ValueError, AnalysisInputError) as error:
        add_problem(problems, "spec_integrity", str(error), cell_id)
    command = receipt.get("command")
    expected_command = [
        str(binary_path), "--cell-spec", str(cell_dir / "cell-spec.json"),
        "--output", str(cell_dir / "cell-result.partial.json"),
    ]
    if command != expected_command:
        add_problem(problems, "receipt_command", "child command does not match its bound spec/output paths", cell_id)
    if repository_path is not None:
        try:
            if Path(receipt.get("working_directory", "")).resolve(strict=True) != repository_path:
                add_problem(problems, "receipt_identity", "child working directory differs from matrix repository", cell_id)
        except (OSError, TypeError, ValueError):
            add_problem(problems, "receipt_identity", "child working directory is invalid", cell_id)
    status = receipt.get("runner_status")
    if status != "result_recorded":
        if receipt.get("result_sha256") is not None:
            add_problem(problems, "receipt_schema", "failed process has an unexpected result hash", cell_id)
        if status not in ("failed", "interrupted"):
            add_problem(problems, "receipt_schema", "unknown runner status", cell_id)
        return None, None, False
    if type(receipt.get("exit_code")) is not int or receipt.get("exit_code") != 0:
        add_problem(problems, "receipt_exit", "result_recorded child did not exit with status 0", cell_id)
    if receipt.get("watchdog_timed_out") is not False:
        add_problem(problems, "receipt_exit", "result_recorded child has timeout flag set or missing", cell_id)
    if not finite(receipt.get("elapsed_millis")) or receipt["elapsed_millis"] < 0:
        add_problem(problems, "receipt_metrics", "child elapsed time is missing/non-finite/negative", cell_id)
    result_path = verify_child_file(receipt, "result_path", "result_sha256", cell_dir, problems, cell_id, "result_integrity")
    if result_path is None:
        return None, None, False
    try:
        result = parse_json(result_path.read_bytes(), f"{cell_id} result")
    except AnalysisInputError as error:
        add_problem(problems, "result_schema", str(error), cell_id)
        return None, None, False
    if not isinstance(result, dict):
        add_problem(problems, "result_schema", "result is not an object", cell_id)
        return None, None, False
    # The result can remain useful as a typed outcome even when its metrics or
    # counters fail independent checks. Track identity separately so such a
    # row remains visible but cannot enter a quality distribution.
    identity_problem_start = len(problems)
    for key, expected in (
        ("schema", runner.RESULT_SCHEMA),
        ("cell_id", cell_id),
        ("spec", spec),
        ("spec_sha256", expected_spec_sha),
        ("matrix_expected_cell_count", inventory_count),
        ("matrix_spec_inventory_sha256", inventory_sha),
        ("executable_sha256", binary_sha),
    ):
        if result.get(key) != expected:
            add_problem(problems, "result_identity", f"result {key} differs from plan", cell_id)
    if not exact_int(result.get("matrix_expected_cell_count")):
        add_problem(problems, "result_identity", "result expected-cell count is not an integer", cell_id)
    if not isinstance(result.get("outcome"), str) or result.get("outcome") not in ALLOWED_OUTCOMES:
        add_problem(problems, "result_schema", "result has unknown typed outcome", cell_id)
    if receipt.get("optimizer_outcome") != result.get("outcome"):
        add_problem(problems, "receipt_outcome", "child receipt outcome differs from result", cell_id)
    try:
        if Path(result.get("executable_path", "")).resolve(strict=True) != binary_path:
            add_problem(problems, "result_identity", "result executable path differs from matrix", cell_id)
    except (OSError, TypeError, ValueError):
        add_problem(problems, "result_identity", "result has invalid executable path", cell_id)
    hashes = result.get("input_source_hashes")
    if not isinstance(hashes, dict) or any(not isinstance(path, str) or not valid_sha256(value) for path, value in hashes.items()):
        add_problem(problems, "source_identity", "result has malformed input source hashes", cell_id)
    identity_problem_end = len(problems)
    identity_ok = not any(
        problem.get("code") in {
            "receipt_schema", "receipt_identity", "receipt_path", "receipt_mismatch",
            "receipt_missing", "stdout_integrity", "stderr_integrity", "spec_integrity",
            "result_integrity", "result_schema", "result_identity", "source_identity",
        }
        for problem in problems[problem_start:identity_problem_end]
    )
    if not finite(result.get("source_normalization_reference_hz")) or result["source_normalization_reference_hz"] <= 0:
        add_problem(problems, "result_metrics", "source normalization reference is invalid", cell_id)
    for field in ("elapsed_millis", "engine_elapsed_millis", "source_metric_score_calls", "callback_invocations"):
        value = result.get(field)
        if field in ("source_metric_score_calls", "callback_invocations"):
            if not exact_int(value) or value < 0:
                add_problem(problems, "result_metrics", f"{field} is not a non-negative integer", cell_id)
        elif not finite(value) or value < 0:
            add_problem(problems, "result_metrics", f"{field} is missing/non-finite/negative", cell_id)
    root_counts = validate_counter_snapshot(result.get("root_counters"), "root counters", spec["root_search_budget"], problems, cell_id)
    stage_summary = validate_stage_evidence(result, spec, problems)
    if result.get("outcome") == "completed" and stage_summary["stage_count"] == 0:
        add_problem(problems, "stage_missing", "completed result has no controlled optimizer dispatch stage", cell_id)
    if root_counts is not None and stage_summary["search_started_total_by_stages"] != root_counts["evaluations_started"]:
        add_problem(problems, "counter_attribution", "sum of stage search admissions differs from root total", cell_id)
    outcome = result.get("outcome")
    purpose = spec.get("purpose")
    if purpose == "observer_unsupported":
        if outcome != "callback_unsupported":
            add_problem(problems, "purpose_outcome", "observer-unsupported cell did not return its declared refusal", cell_id)
        if root_counts is not None and (
            root_counts["evaluations_started"] != 0
            or root_counts["validation_evaluations_started"] != 0
            or result.get("source_metric_score_calls") != 0
        ):
            add_problem(problems, "refusal_scored", "callback-unsupported cell scored an objective", cell_id)
    elif outcome == "callback_unsupported":
        add_problem(problems, "purpose_outcome", "callback-unsupported outcome used outside that purpose", cell_id)
    if purpose == "observer_stop":
        if outcome != "observer_stopped":
            add_problem(problems, "purpose_outcome", "observer-stop cell did not stop at its callback", cell_id)
        if root_counts is not None and not root_counts["cancellation_requested"]:
            add_problem(problems, "stop_evidence", "observer stop has no latched cancellation evidence", cell_id)
        callback_count = result.get("callback_invocations")
        if not exact_int(callback_count) or callback_count <= 0:
            add_problem(problems, "stop_evidence", "observer stop has no callback invocation", cell_id)
    elif outcome == "observer_stopped":
        add_problem(problems, "purpose_outcome", "observer-stop outcome used outside that purpose", cell_id)
    if outcome == "timed_out" and root_counts is not None and not root_counts["deadline_reached"]:
        stage_rows = result.get("stage_evidence")
        stage_timed_out = any(
            isinstance(stage, dict)
            and isinstance(stage.get("evidence"), dict)
            and stage["evidence"].get("termination") == "timed_out"
            for stage in (stage_rows if isinstance(stage_rows, list) else [])
        )
        if not stage_timed_out:
            add_problem(problems, "timeout_evidence", "timed_out outcome has no deadline/typed stage evidence", cell_id)
    quality = None
    if outcome == "completed":
        quality = validate_completed_quality(result, spec, problems)
    elif outcome in ("backend_failure", "invalid_candidate"):
        add_problem(problems, "optimizer_failure", f"typed outcome is {outcome}", cell_id)
    return result, quality, identity_ok


def verify_run(run_dir: Path) -> dict[str, Any]:
    run_dir = run_dir.resolve(strict=True)
    matrix_bytes = (run_dir / "matrix-run.json").read_bytes()
    inventory_bytes = (run_dir / "cell-spec-list.stdout.json").read_bytes()
    matrix = parse_json(matrix_bytes, "matrix-run.json")
    if not isinstance(matrix, dict) or matrix.get("schema") != runner.RUN_SCHEMA:
        raise AnalysisInputError("matrix-run.json has an unsupported schema")
    inventory, specs = validate_inventory(inventory_bytes)
    problems: list[dict[str, str]] = []
    inventory_sha = inventory["spec_inventory_sha256"]
    spec_ids = [spec["cell_id"] for spec in specs]
    id_set = set(spec_ids)

    def id_list(field: str) -> list[str]:
        value = matrix.get(field)
        if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
            add_problem(problems, "matrix_id_schema", f"matrix {field} is not a string array")
            return []
        if len(set(value)) != len(value):
            add_problem(problems, "matrix_duplicate_ids", f"matrix {field} contains duplicate IDs")
        return value

    selected_ids = id_list("selected_cell_ids")
    planned_ids = id_list("planned_cell_ids")
    attempted_ids = id_list("attempted_cell_ids")
    valid_claimed_ids = id_list("valid_result_cell_ids")
    runner_failed_claimed_ids = id_list("runner_failed_cell_ids")
    missing_claimed_ids = id_list("missing_cell_ids")
    unresolved_claimed_ids = id_list("unresolved_result_cell_ids")
    for name, values in (("selected", selected_ids), ("planned", planned_ids), ("attempted", attempted_ids)):
        unknown = set(values) - id_set
        if unknown:
            add_problem(problems, "matrix_unknown_ids", f"{name} IDs absent from printed inventory: {sorted(unknown)[:5]}")
    if planned_ids != selected_ids:
        add_problem(problems, "matrix_selection", "planned IDs differ from selected IDs")
    if [cell_id for cell_id in spec_ids if cell_id in set(selected_ids)] != selected_ids:
        add_problem(problems, "matrix_selection", "selected IDs are not in printed-inventory order")
    for field, expected in (
        ("expected_cell_count", len(specs)),
        ("selected_cell_count", len(selected_ids)),
    ):
        if not exact_int(matrix.get(field)) or matrix[field] != expected:
            add_problem(problems, "matrix_count", f"matrix {field} does not match independently counted IDs")
    if matrix.get("spec_inventory_sha256") != inventory_sha:
        add_problem(problems, "matrix_identity", "matrix inventory digest differs from printed inventory")
    if matrix.get("cell_spec_list_stdout_sha256") != sha256_bytes(inventory_bytes):
        add_problem(problems, "matrix_identity", "matrix inventory stdout digest differs from captured bytes")
    inventory_stdout = run_dir / "cell-spec-list.stdout.json"
    if not inventory_stdout.is_file() or inventory_stdout.read_bytes() != inventory_bytes:
        add_problem(problems, "matrix_identity", "printed inventory snapshot file is missing or differs")
    try:
        binary_path = Path(matrix.get("binary_path", "")).resolve(strict=True)
        binary_sha = sha256_file(binary_path)
        if not valid_sha256(matrix.get("binary_sha256")) or binary_sha != matrix.get("binary_sha256"):
            add_problem(problems, "binary_identity", "current executable SHA differs from matrix identity")
    except (OSError, TypeError, ValueError):
        binary_path = Path(matrix.get("binary_path", "")).resolve()
        binary_sha = None
        add_problem(problems, "binary_identity", "matrix executable is unavailable for SHA verification")
    if matrix.get("cell_spec_list_exit_code") != 0 or type(matrix.get("cell_spec_list_exit_code")) is not int:
        add_problem(problems, "inventory_execution", "cell-spec listing did not exit successfully")
    if matrix.get("cell_spec_list_watchdog_timed_out") is not False:
        add_problem(problems, "inventory_execution", "cell-spec listing was timed out or lacks explicit false")
    expected_list_command = [str(binary_path), "--list-cell-specs"]
    if matrix.get("cell_spec_list_command") != expected_list_command:
        add_problem(problems, "inventory_execution", "recorded inventory command does not identify the matrix binary")
    environment_sha = matrix.get("environment_sha256")
    if not valid_sha256(environment_sha):
        environment_sha = None
        add_problem(problems, "matrix_identity", "matrix environment SHA-256 is malformed")
    try:
        repository_path = Path(matrix.get("repository", "")).resolve(strict=True)
        if not repository_path.is_dir():
            raise ValueError("repository is not a directory")
    except (OSError, TypeError, ValueError):
        repository_path = None
        add_problem(problems, "matrix_identity", "matrix repository path is unavailable")
    for path_name, hash_name in (
        ("cell-spec-list.stdout.json", "cell_spec_list_stdout_sha256"),
        ("cell-spec-list.stderr.log", "cell_spec_list_stderr_sha256"),
    ):
        path = run_dir / path_name
        try:
            actual_hash = sha256_file(path)
            if actual_hash != matrix.get(hash_name):
                add_problem(problems, "inventory_log_integrity", f"{path_name} hash differs from matrix")
        except OSError as error:
            add_problem(problems, "inventory_log_integrity", f"{path_name} unavailable: {error}")

    receipt_rows = matrix.get("cells")
    if not isinstance(receipt_rows, list):
        raise AnalysisInputError("matrix cells field is not an array")
    receipts: dict[str, dict[str, Any]] = {}
    for receipt in receipt_rows:
        if not isinstance(receipt, dict) or not isinstance(receipt.get("cell_id"), str):
            add_problem(problems, "receipt_schema", "matrix contains malformed child receipt")
            continue
        cell_id = receipt["cell_id"]
        if cell_id in receipts:
            add_problem(problems, "duplicate_receipt", "matrix contains duplicate child receipt", cell_id)
        receipts[cell_id] = receipt
    receipt_ids = list(receipts)
    if receipt_ids != attempted_ids:
        add_problem(problems, "attempted_receipts", "attempted IDs differ from ordered receipt IDs")
    if attempted_ids != planned_ids[:len(attempted_ids)]:
        add_problem(problems, "attempt_order", "attempted IDs are not a planned prefix")
    computed_missing = [cell_id for cell_id in planned_ids if cell_id not in receipts]
    if missing_claimed_ids != computed_missing:
        add_problem(problems, "matrix_count", "missing IDs differ from recomputed planned-minus-attempted")

    specs_by_id = {spec["cell_id"]: spec for spec in specs}
    global_quality_identity_ok = not any(
        problem.get("code") in {
            "matrix_identity", "binary_identity", "inventory_execution", "inventory_log_integrity"
        }
        for problem in problems
    )
    results: dict[str, dict[str, Any]] = {}
    qualities: dict[str, dict[str, Any]] = {}
    computed_runner_failed: list[str] = []
    identity_unverified_ids: list[str] = []
    for cell_id, receipt in receipts.items():
        spec = specs_by_id.get(cell_id)
        if spec is None:
            add_problem(problems, "receipt_unknown_id", "receipt ID is absent from inventory", cell_id)
            computed_runner_failed.append(cell_id)
            continue
        cell_problems_start = len(problems)
        result, quality, identity_ok = verify_result(
            receipt, spec, inventory_sha, len(specs), binary_path, binary_sha or "",
            repository_path, environment_sha, run_dir / "cells", problems
        )
        if receipt.get("runner_status") != "result_recorded":
            computed_runner_failed.append(cell_id)
        elif result is None:
            pass
        elif identity_ok:
            results[cell_id] = result
            if quality is not None and global_quality_identity_ok and len(problems) == cell_problems_start:
                qualities[cell_id] = quality
        else:
            identity_unverified_ids.append(cell_id)
    if runner_failed_claimed_ids != computed_runner_failed:
        add_problem(problems, "matrix_count", "runner-failed IDs differ from receipts")
    if valid_claimed_ids != [cell_id for cell_id in attempted_ids if receipts.get(cell_id, {}).get("runner_status") == "result_recorded"]:
        add_problem(problems, "matrix_count", "valid-result IDs differ from runner receipt statuses")
    if unresolved_claimed_ids != [cell_id for cell_id in planned_ids if cell_id not in set(valid_claimed_ids)]:
        add_problem(problems, "matrix_count", "unresolved-result IDs differ from planned minus claimed valid")
    if not exact_int(matrix.get("completed_cell_count")) or matrix["completed_cell_count"] != len(receipt_rows):
        add_problem(problems, "matrix_count", "completed count differs from receipt count")
    if not exact_int(matrix.get("runner_failure_count")) or matrix["runner_failure_count"] != len(computed_runner_failed):
        add_problem(problems, "matrix_count", "runner failure count differs from receipts")

    outcome_counts = Counter(
        result.get("outcome") for result in results.values()
    )
    recorded_outcomes = matrix.get("optimizer_outcome_counts")
    if not isinstance(recorded_outcomes, dict) or any(not exact_int(v) or v < 0 for v in recorded_outcomes.values()):
        add_problem(problems, "outcome_counts", "recorded outcome counts are malformed")
    elif dict(sorted(outcome_counts.items())) != dict(sorted(recorded_outcomes.items())):
        add_problem(problems, "outcome_counts", "recorded optimizer outcome counts differ from verified results")

    runner_execution_complete = (
        selected_ids == spec_ids
        and planned_ids == spec_ids
        and attempted_ids == planned_ids
        and len(computed_runner_failed) == 0
        and len([r for r in receipts.values() if r.get("runner_status") == "result_recorded"]) == len(planned_ids)
    )
    if type(matrix.get("complete_matrix")) is not bool or matrix["complete_matrix"] != runner_execution_complete:
        add_problem(problems, "completeness_flag", "complete_matrix flag differs from recomputed execution completeness")
    expected_status = "recorded" if not computed_runner_failed and len(receipt_ids) == len(planned_ids) else None
    if expected_status is not None and matrix.get("status") != expected_status:
        add_problem(problems, "matrix_status", "matrix status contradicts independently recomputed receipts")
    if len(receipt_rows) != len(planned_ids):
        # A partial/interrupted run is a valid summary, but not strict-complete.
        pass
    actual_dirs: set[str] = set()
    cells_dir = run_dir / "cells"
    if cells_dir.is_dir():
        actual_dirs = {child.name for child in cells_dir.iterdir() if child.is_dir()}
    receipt_dirs = {
        Path(receipt["cell_directory"]).name
        for receipt in receipt_rows
        if isinstance(receipt, dict) and isinstance(receipt.get("cell_directory"), str)
    }
    unexpected_dirs = sorted(actual_dirs - receipt_dirs)
    if unexpected_dirs:
        add_problem(problems, "unbound_child_artifact", f"unreceipted cell directories exist: {unexpected_dirs[:5]}")

    return {
        "matrix": matrix,
        "inventory": inventory,
        "specs": specs,
        "specs_by_id": specs_by_id,
        "receipts": receipts,
        "results": results,
        "qualities": qualities,
        "problems": problems,
        "selected_ids": selected_ids,
        "planned_ids": planned_ids,
        "attempted_ids": attempted_ids,
        "valid_claimed_ids": valid_claimed_ids,
        "missing_ids": computed_missing,
        "unresolved_ids": [cell_id for cell_id in planned_ids if cell_id not in results],
        "runner_failed_ids": computed_runner_failed,
        "identity_unverified_ids": identity_unverified_ids,
        "execution_complete": runner_execution_complete,
        "binary_path": str(binary_path),
        "binary_sha256_current": binary_sha,
        "inventory_bytes": inventory_bytes,
        "matrix_bytes": matrix_bytes,
        "unexpected_cell_directories": unexpected_dirs,
    }


def build_report(run_dir: Path) -> tuple[dict[str, Any], dict[str, bytes]]:
    state = verify_run(run_dir)
    matrix = state["matrix"]
    specs = state["specs"]
    results = state["results"]
    qualities = state["qualities"]
    receipts = state["receipts"]
    planned = state["planned_ids"]
    groups: dict[tuple[str, str, str, int], dict[str, Any]] = {}
    for spec in specs:
        key = (spec.get("purpose", "<missing>"), spec.get("backend", "<missing>"),
               spec.get("case_id", "<missing>"), spec.get("root_search_budget", -1))
        group = groups.setdefault(key, {
            "planned": 0, "outcomes": Counter(), "runner_failures": 0,
            "missing": 0, "completed": 0, "quality_eligible": 0,
            "quality_excluded": 0, "comparison_delta": [], "final_worst": [],
            "optimizer_loss": [], "heldout_worst_delta": [], "heldout_deltas": [],
            "search_evals": [], "validation_evals": [], "engine_millis": [],
            "ineligible_ids": [], "identity_unverified": 0,
        })
        group["planned"] += 1
    for cell_id in planned:
        spec = state["specs_by_id"].get(cell_id)
        if spec is None:
            continue
        key = (spec.get("purpose", "<missing>"), spec.get("backend", "<missing>"),
               spec.get("case_id", "<missing>"), spec.get("root_search_budget", -1))
        group = groups[key]
        receipt = receipts.get(cell_id)
        result = results.get(cell_id)
        if receipt is None:
            group["missing"] += 1
            group["outcomes"]["not_started_missing"] += 1
            continue
        if result is None:
            if cell_id in state["identity_unverified_ids"]:
                group["identity_unverified"] += 1
                group["outcomes"]["identity_unverified"] += 1
            else:
                group["runner_failures"] += 1
                group["outcomes"][f"runner_{receipt.get('runner_status', 'unknown')}"] += 1
            continue
        outcome = result.get("outcome", "<missing>")
        group["outcomes"][outcome] += 1
        if outcome != "completed":
            continue
        group["completed"] += 1
        quality = qualities.get(cell_id)
        if quality is None:
            group["quality_excluded"] += 1
            group["ineligible_ids"].append(cell_id)
            continue
        group["quality_eligible"] += 1
        group["comparison_delta"].append(quality["delta_worst"])
        group["final_worst"].append(quality["final_worst"])
        group["optimizer_loss"].append(quality["optimizer_loss"])
        if quality["heldout_worst_delta"] is not None:
            group["heldout_worst_delta"].append(quality["heldout_worst_delta"])
            group["heldout_deltas"].extend(quality["heldout_deltas"])
        for target, value in (("search_evals", quality["search_evaluations"]),
                              ("validation_evals", quality["validation_evaluations"]),
                              ("engine_millis", quality["engine_elapsed_millis"])):
            if finite(value):
                group[target].append(float(value))
    group_rows = []
    for (purpose, backend, case_id, cap), group in sorted(groups.items()):
        group_rows.append({
            "purpose": purpose, "backend": backend, "case_id": case_id, "root_cap": cap,
            "planned": group["planned"],
            "returned": sum(group["outcomes"].values()),
            "missing": group["missing"], "runner_failures": group["runner_failures"],
            "identity_unverified": group["identity_unverified"],
            "outcomes": dict(sorted(group["outcomes"].items())),
            "completed": group["completed"],
            "quality_eligible_completed": group["quality_eligible"],
            "quality_excluded_completed": group["quality_excluded"],
            "quality_excluded_ids": group["ineligible_ids"],
            "paired_final_minus_baseline_worst": distribution(group["comparison_delta"]),
            "paired_final_worst_loss": distribution(group["final_worst"]),
            "optimizer_loss_after_engine_normalization": distribution(group["optimizer_loss"]),
            "heldout_final_minus_baseline_worst": distribution(group["heldout_worst_delta"]),
            "heldout_measurement_final_minus_baseline": distribution(group["heldout_deltas"]),
            "search_evaluations_started": distribution(group["search_evals"]),
            "validation_evaluations_started": distribution(group["validation_evals"]),
            "engine_elapsed_millis": distribution(group["engine_millis"]),
        })
    outcomes = Counter(result.get("outcome", "<missing>") for result in results.values())
    by_purpose: dict[str, Counter[str]] = defaultdict(Counter)
    by_backend: dict[str, Counter[str]] = defaultdict(Counter)
    for spec in specs:
        cell_id = spec["cell_id"]
        result = results.get(cell_id)
        receipt = receipts.get(cell_id)
        outcome = result.get("outcome") if result else (
            f"runner_{receipt.get('runner_status', 'unknown')}" if receipt else "not_started_missing"
        )
        by_purpose[str(spec.get("purpose"))][str(outcome)] += 1
        by_backend[str(spec.get("backend"))][str(outcome)] += 1
    errors = state["problems"]
    gate_failures = list(errors)
    if not state["execution_complete"]:
        gate_failures.append({"code": "incomplete_matrix", "message": "selected and attempted IDs do not cover the full inventory with validated results"})
    quality_excluded = [cell_id for cell_id, result in results.items()
                        if result.get("outcome") == "completed" and cell_id not in qualities]
    for cell_id in quality_excluded:
        if not any(problem.get("cell_id") == cell_id for problem in errors):
            gate_failures.append({"code": "quality_exclusion", "cell_id": cell_id,
                                  "message": "completed candidate is not eligible for quality distributions"})
    report = {
        "schema": ANALYSIS_SCHEMA,
        "analysis_created_at_utc": datetime.now(timezone.utc).isoformat(),
        "read_only": True,
        "run_directory": str(Path(run_dir).resolve()),
        "matrix_run_sha256": sha256_bytes(state["matrix_bytes"]),
        "matrix_run_sha256_after_read": sha256_bytes((Path(run_dir) / "matrix-run.json").read_bytes()),
        "matrix_run_changed_during_analysis": state["matrix_bytes"] != (Path(run_dir) / "matrix-run.json").read_bytes(),
        "matrix_status_recorded": matrix.get("status"),
        "matrix_complete_flag_recorded": matrix.get("complete_matrix"),
        "matrix_execution_complete_recomputed": state["execution_complete"],
        "inventory_expected_count_recomputed": len(specs),
        "selected_count_recomputed": len(state["selected_ids"]),
        "planned_count_recomputed": len(state["planned_ids"]),
        "unselected_inventory_cell_ids": [
            spec["cell_id"] for spec in specs if spec["cell_id"] not in set(state["selected_ids"])
        ],
        "attempted_count_recomputed": len(state["attempted_ids"]),
        "independently_verified_result_count": len(results),
        "independently_verified_result_ids": [cell_id for cell_id in planned if cell_id in results],
        "quality_eligible_completed_count": len(qualities),
        "quality_excluded_completed_count": len(quality_excluded),
        "quality_excluded_completed_ids": quality_excluded,
        "missing_cell_ids_recomputed": state["missing_ids"],
        "unresolved_cell_ids_recomputed": state["unresolved_ids"],
        "runner_failed_cell_ids_recomputed": state["runner_failed_ids"],
        "identity_unverified_cell_ids_recomputed": state["identity_unverified_ids"],
        "verified_outcome_counts": dict(sorted(outcomes.items())),
        "purpose_outcome_counts_over_inventory": {key: dict(sorted(value.items())) for key, value in sorted(by_purpose.items())},
        "backend_outcome_counts_over_inventory": {key: dict(sorted(value.items())) for key, value in sorted(by_backend.items())},
        "identity": {
            "binary_path": state["binary_path"],
            "binary_sha256_recorded": matrix.get("binary_sha256"),
            "binary_sha256_current": state["binary_sha256_current"],
            "spec_inventory_sha256": state["inventory"]["spec_inventory_sha256"],
            "inventory_stdout_sha256_recorded": matrix.get("cell_spec_list_stdout_sha256"),
            "inventory_stdout_sha256_actual": sha256_bytes(state["inventory_bytes"]),
            "receipt_artifact_directories_unbound": state["unexpected_cell_directories"],
        },
        "integrity_and_validation_problems": errors,
        "strict_gate_failures": gate_failures,
        "strict_gate_pass": not gate_failures,
        "performance_summary_policy": [
            "Only completed cells with independently validated finite paired metrics and emitted-filter feasibility contribute to quality distributions.",
            "Timeouts, refusals, invalid candidates, backend failures, runner failures, and missing cells remain separate outcomes.",
            "Primary paired delta is max(final_loss)-max(baseline_loss) on one unique complete metric set; negative means lower final loss.",
            "Held-out deltas use only explicit held_out_source_metrics; they are not substituted with training rows.",
            "Elapsed time is descriptive; it is not a controlled timing ranking.",
            "The measured_stereo_8361a fixture is perturbation-derived evidence, not an independent measured capture.",
            "Repeated seed labels for deterministic backends are not independent randomized trials.",
        ],
        "groups_by_backend_case_purpose_cap": group_rows,
        "analyzer_sha256": sha256_file(Path(__file__).resolve()),
    }
    return report, {"matrix-run.snapshot.json": state["matrix_bytes"],
                    "cell-spec-list.stdout.snapshot.json": state["inventory_bytes"]}


def render_markdown(report: dict[str, Any]) -> str:
    status = "complete" if report["matrix_execution_complete_recomputed"] else "partial/incomplete"
    lines = [
        "# Optimizer benchmark matrix analysis", "",
        f"Run is **{status}** by independently recomputed IDs; recorded complete flag: `{report['matrix_complete_flag_recorded']}`.",
        f"Verified results: **{report['independently_verified_result_count']} / {report['inventory_expected_count_recomputed']}**; "
        f"quality-eligible completed: **{report['quality_eligible_completed_count']}**; "
        f"quality-excluded completed: **{report['quality_excluded_completed_count']}**.",
        f"Strict gate: **{'PASS' if report['strict_gate_pass'] else 'FAIL'}** ({len(report['strict_gate_failures'])} issue(s)).",
        "", "Partial runs retain planned, missing, unresolved, and failed IDs. Typed timeouts and refusals are never counted as quality results.",
        "Elapsed times are descriptive only. Repeated seed labels for deterministic backends do not represent independent randomized trials.",
        "The 8361 stereo holdouts are perturbation-derived evidence, not independent measured captures.",
        "", "## Outcomes by purpose", "", "| Purpose | Outcomes over full inventory |", "|---|---|",
    ]
    for purpose, outcomes in report["purpose_outcome_counts_over_inventory"].items():
        lines.append(f"| {purpose} | " + ", ".join(f"{key}: {value}" for key, value in outcomes.items()) + " |")
    lines.extend([
        "", "## Completed, feasible quality distributions", "",
        "Only completed rows with finite unique paired metrics and feasible realized filters contribute.",
        "`Δ worst loss` is final minus baseline; negative means the paired worst loss decreased.", "",
        "| Purpose | Backend | Case | Cap | Planned | Returned | Completed | Quality n | Excluded | Timeout | Median Δ worst | Median held-out Δ worst |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in report["groups_by_backend_case_purpose_cap"]:
        outcomes = row["outcomes"]
        primary = row["paired_final_minus_baseline_worst"]
        heldout = row["heldout_final_minus_baseline_worst"]
        fmt = lambda value: "—" if value is None else f"{value:.6g}"
        lines.append(
            f"| {row['purpose']} | {row['backend']} | {row['case_id']} | {row['root_cap']} | "
            f"{row['planned']} | {row['returned']} | {row['completed']} | {row['quality_eligible_completed']} | "
            f"{row['quality_excluded_completed']} | {outcomes.get('timed_out', 0)} | "
            f"{fmt(primary['median'])} | {fmt(heldout['median'])} |"
        )
    lines.extend(["", "## Strict gate failures", ""])
    if report["strict_gate_failures"]:
        for issue in report["strict_gate_failures"]:
            suffix = f" [{issue['cell_id']}]" if issue.get("cell_id") else ""
            lines.append(f"- `{issue['code']}`{suffix}: {issue['message']}")
    else:
        lines.append("None.")
    return "\n".join(lines) + "\n"


def write_analysis(run_dir: Path, output_dir: Path) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite analysis output: {output_dir}")
    report, snapshots = build_report(run_dir)
    matrix_after = (Path(run_dir) / "matrix-run.json").read_bytes()
    inventory_snapshot = snapshots["cell-spec-list.stdout.snapshot.json"]
    inventory_after = (Path(run_dir) / "cell-spec-list.stdout.json").read_bytes()
    report["matrix_run_sha256_after_read"] = sha256_bytes(matrix_after)
    report["matrix_run_changed_during_analysis"] = report["matrix_run_sha256"] != report["matrix_run_sha256_after_read"]
    report["inventory_stdout_sha256_after_read"] = sha256_bytes(inventory_after)
    report["inventory_changed_during_analysis"] = inventory_snapshot != inventory_after
    if report["matrix_run_changed_during_analysis"] or report["inventory_changed_during_analysis"]:
        changed = []
        if report["matrix_run_changed_during_analysis"]:
            changed.append("matrix-run.json")
        if report["inventory_changed_during_analysis"]:
            changed.append("cell-spec-list.stdout.json")
        issue = {"code": "input_mutated", "message": f"analysis inputs changed during read: {', '.join(changed)}"}
        report["integrity_and_validation_problems"].append(issue)
        report["strict_gate_failures"].append(issue)
        report["strict_gate_pass"] = False
    output_dir.mkdir(parents=True, exist_ok=False)
    for name, data in snapshots.items():
        (output_dir / name).write_bytes(data)
    (output_dir / "matrix-analysis.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8"
    )
    (output_dir / "summary.md").write_text(render_markdown(report), encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True, help="existing matrix run directory")
    parser.add_argument("--output", type=Path, required=True, help="new analysis output directory")
    parser.add_argument("--require-complete", action="store_true",
                        help="return nonzero unless all inventory cells and strict validations pass")
    args = parser.parse_args(argv)
    if args.output.exists():
        print(f"analysis failed: refusing to overwrite existing output: {args.output}", file=sys.stderr)
        return 1 if args.require_complete else 2
    try:
        report = write_analysis(args.run, args.output)
    except (OSError, AnalysisInputError, ValueError) as error:
        if not args.output.exists():
            args.output.mkdir(parents=True, exist_ok=False)
        failure = {
            "schema": ANALYSIS_SCHEMA,
            "read_only": True,
            "run_directory": str(args.run),
            "strict_mode": args.require_complete,
            "strict_gate_pass": False,
            "fatal_error": f"{type(error).__name__}: {error}",
        }
        (args.output / "analysis-error.json").write_text(
            json.dumps(failure, indent=2) + "\n", encoding="utf-8"
        )
        print(f"analysis failed: {failure['fatal_error']}", file=sys.stderr)
        return 1 if args.require_complete else 2
    print(json.dumps({
        "analysis_output": str(args.output.resolve()),
        "strict_gate_pass": report["strict_gate_pass"],
        "verified_results": report["independently_verified_result_count"],
        "inventory_count": report["inventory_expected_count_recomputed"],
        "quality_eligible_completed": report["quality_eligible_completed_count"],
        "gate_failure_count": len(report["strict_gate_failures"]),
    }, indent=2))
    return 0 if not args.require_complete or report["strict_gate_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
