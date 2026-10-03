#!/usr/bin/env python3
"""Wolfram cross-check: triangular-log viewer smoothing (PY01).

Oracle: crates/autoeq-qa/wolfram/py01_dsp_smoothing.wls (independent
triangular-log weighting from the published viewer definition).
Compares scripts/src/dsp.py smooth_octave against the engine golden.
Tolerance: A, absolute error <= 1e-9 dB. Run from workspace root.
"""

import json
import sys
from pathlib import Path

from qa_support.metrics import maximum_error_metrics, require_finite_numbers

CASE_ID = "autoeq-qa.py01-dsp-smoothing.v1"
TOL_ABS = 1e-9

ROOT = Path(__file__).resolve().parents[3]
GOLDEN = ROOT / "crates/autoeq-qa/wolfram/goldens/py01_dsp_smoothing.json"


def fail(message):
    print(f"FAIL {CASE_ID}: {message}", file=sys.stderr)
    sys.exit(1)


def main():
    sys.path.insert(0, str(ROOT))
    try:
        from scripts.src.dsp import smooth_octave
    except ImportError as error:
        fail(f"cannot import smooth_octave: {error}")
    if not GOLDEN.is_file():
        fail(f"missing golden {GOLDEN}")
    ref = json.loads(GOLDEN.read_text())
    if ref.get("case") != CASE_ID or ref.get("schema_version") != 1:
        fail("golden case identity mismatch")
    freqs = ref["grid_hz"]
    spl = ref["spl_db"]
    expected = ref["smoothed_db"]
    if not (len(freqs) == len(spl) == len(expected)):
        fail("golden grid/spl/smoothed length mismatch")
    try:
        require_finite_numbers(
            [*freqs, *spl, *expected, ref["octave_fraction"]],
            "golden smoothing data",
        )
    except ValueError as error:
        fail(f"invalid finite golden input: {error}")
    got = smooth_octave(list(freqs), list(spl), float(ref["octave_fraction"]))
    if len(got) != len(expected):
        fail(f"length {len(got)} != golden {len(expected)}")
    try:
        max_err, max_rel_err = maximum_error_metrics(zip(got, expected))
    except ValueError as error:
        fail(f"invalid finite comparison: {error}")
    for f, g, e in zip(freqs, got, expected):
        err = abs(g - e)
        if err > TOL_ABS:
            fail(f"f={f} Hz: got={g:.12f} expected={e:.12f} abs_err={err:.3e}")
    print(json.dumps({
        "QA_RESULT": True, "case": CASE_ID, "pass": True,
        "max_abs_error": max_err, "max_rel_error": max_rel_err,
        "tolerance": TOL_ABS,
        "tolerance_kind": "abs", "provenance": "wolfram-engine-15.0.0",
    }))


if __name__ == "__main__":
    main()
