#!/usr/bin/env python3
"""Wolfram cross-check: held-out perturbation contract (PY05).

Oracle: crates/autoeq-qa/wolfram/py05_heldout_contract.wls (independent
closed form of the magnitude/phase deltas). Drives the real
scripts/generate_roomeq_held_out.py generate() on a scratch CSV and
compares the 4-decimal rounded output.
Tolerance: A, absolute error <= 6e-5 dB/deg (4-decimal CSV rounding).
"""

import importlib.util
import json
import sys
import tempfile
from pathlib import Path

from qa_support.metrics import maximum_error_metrics, require_finite_numbers

CASE_ID = "autoeq-qa.py05-heldout-contract.v1"
TOL_ABS = 6e-5

ROOT = Path(__file__).resolve().parents[3]
GOLDEN = ROOT / "crates/autoeq-qa/wolfram/goldens/py05_heldout_contract.json"
MODULE = ROOT / "scripts/generate_roomeq_held_out.py"


def fail(message):
    print(f"FAIL {CASE_ID}: {message}", file=sys.stderr)
    sys.exit(1)


def _validate_fixture(ref):
    """Validate held-out arrays before the generator can hide invalid inputs."""
    grid = ref["grid_hz"]
    response_keys = (
        "base_spl_db",
        "base_phase_deg",
        "heldout_spl_db",
        "heldout_phase_deg",
    )
    if any(len(ref[key]) != len(grid) for key in response_keys):
        raise ValueError("golden frequency and response array lengths differ")
    inputs = [*grid, *(value for key in response_keys for value in ref[key])]
    require_finite_numbers(inputs, "golden held-out data")
    return grid


def _validate_output_grid(actual, expected):
    """Require CSV frequency rows to match the golden grid within its tolerance."""
    if len(actual) != len(expected):
        raise ValueError("generated frequency and expected grid lengths differ")
    pairs = list(zip(actual, expected))
    max_abs_error, _ = maximum_error_metrics(pairs)
    if max_abs_error > TOL_ABS:
        raise ValueError(
            f"generated frequency grid differs from golden by "
            f"{max_abs_error:.3e} Hz (limit {TOL_ABS:.1e} Hz)"
        )
    return pairs


def main():
    if not GOLDEN.is_file():
        fail(f"missing golden {GOLDEN}")
    ref = json.loads(GOLDEN.read_text())
    if ref.get("case") != CASE_ID or ref.get("schema_version") != 1:
        fail("golden case identity mismatch")
    spec = importlib.util.spec_from_file_location("gen_held_out", MODULE)
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception as error:
        fail(f"cannot load generate_roomeq_held_out: {error}")
    try:
        grid = _validate_fixture(ref)
    except (KeyError, TypeError, ValueError) as error:
        fail(f"invalid held-out fixture: {error}")
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "L.csv"
        dst = Path(tmp) / "L_heldout_1.csv"
        with src.open("w", encoding="utf-8") as handle:
            handle.write("frequency_hz,spl_db,phase_deg\n")
            for f, s, p in zip(grid, ref["base_spl_db"], ref["base_phase_deg"]):
                handle.write(f"{f},{s},{p}\n")
        mod.generate(src, dst, int(ref["position"]), ref["channel"])
        lines = dst.read_text(encoding="utf-8").strip().splitlines()
    if len(lines) != len(grid) + 1:
        fail(f"row count {len(lines)} != {len(grid) + 1}")
    output_rows = []
    for i, line in enumerate(lines[1:]):
        try:
            f, s, p = (float(value) for value in line.split(","))
        except ValueError as error:
            fail(f"malformed output row {i}: {error}")
        output_rows.append((f, s, p))
    try:
        require_finite_numbers(
            (value for row in output_rows for value in row), "generated held-out row"
        )
        grid_comparisons = _validate_output_grid(
            [frequency for frequency, _, _ in output_rows], grid
        )
        comparisons = list(grid_comparisons)
        for i, (_, spl, phase) in enumerate(output_rows):
            comparisons.extend(((spl, ref["heldout_spl_db"][i]),
                                (phase, ref["heldout_phase_deg"][i])))
        max_err, max_rel_err = maximum_error_metrics(comparisons)
    except ValueError as error:
        fail(f"invalid finite comparison: {error}")
    for i, (f, s, p) in enumerate(output_rows):
        frequency_error = abs(f - grid[i])
        if frequency_error > TOL_ABS:
            fail(
                f"frequency[{i}]: got={f:.6f} expected={grid[i]:.6f} "
                f"abs_err={frequency_error:.3e}"
            )
        for name, g, e in (("spl", s, ref["heldout_spl_db"][i]),
                           ("phase", p, ref["heldout_phase_deg"][i])):
            err = abs(g - e)
            if err > TOL_ABS:
                fail(f"{name}[{f} Hz]: got={g:.6f} expected={e:.6f} abs_err={err:.3e}")
    print(json.dumps({
        "QA_RESULT": True, "case": CASE_ID, "pass": True,
        "max_abs_error": max_err, "max_rel_error": max_rel_err,
        "tolerance": TOL_ABS,
        "tolerance_kind": "abs", "provenance": "wolfram-engine-15.0.0",
    }))


if __name__ == "__main__":
    main()
