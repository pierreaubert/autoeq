#!/usr/bin/env python3
"""Wolfram cross-check: QA-helper scorer coefficient signs (H02).

Oracle: crates/autoeq-qa/wolfram/h02_scorer_sign.wls (independent RBJ
cookbook evaluation). Compares scripts/src/dsp.py biquad_coefficients
(Rust-canonical a1,a2,b0,b1,b2 order) plus a direct SOS transfer replay.
Tolerance: N, relative error <= 1e-9 on complex response; the coefficient
vector must expose a deliberate sign flip (negative control).
"""

import cmath
import json
import math
import sys
from pathlib import Path

from qa_support.metrics import maximum_error_metrics, require_finite_numbers

CASE_ID = "autoeq-qa.h02-scorer-sign.v1"
TOL_REL = 1e-9
TOL_COEFF = 1e-12

ROOT = Path(__file__).resolve().parents[3]
GOLDEN = ROOT / "crates/autoeq-qa/wolfram/goldens/h02_scorer_sign.json"


def fail(message):
    print(f"FAIL {CASE_ID}: {message}", file=sys.stderr)
    sys.exit(1)


def _h02_error_metrics(coefficients, expected_coefficients, complex_pairs):
    """Measure coefficient and transfer errors with H02's zero-reference rules."""
    if len(coefficients) != 5 or len(expected_coefficients) != 5:
        raise ValueError("coefficient vectors must both contain five values")
    coefficient_abs, coefficient_rel = maximum_error_metrics(
        zip(coefficients, expected_coefficients)
    )
    response_abs = 0.0
    response_rel = 0.0
    if not complex_pairs:
        raise ValueError("at least one complex response comparison is required")
    for index, (actual, expected) in enumerate(complex_pairs):
        require_finite_numbers(
            (actual.real, actual.imag, expected.real, expected.imag),
            f"complex response[{index}]",
        )
        absolute_error = abs(actual - expected)
        if expected == 0:
            if absolute_error != 0.0:
                raise ValueError(
                    f"complex response[{index}] has a zero reference and nonzero error"
                )
            relative_error = 0.0
        else:
            relative_error = absolute_error / abs(expected)
        if not math.isfinite(absolute_error) or not math.isfinite(relative_error):
            raise ValueError(f"complex response[{index}] produced a non-finite error")
        response_abs = max(response_abs, absolute_error)
        response_rel = max(response_rel, relative_error)
    return max(coefficient_abs, response_abs), max(coefficient_rel, response_rel)


def main():
    sys.path.insert(0, str(ROOT))
    try:
        from scripts.src.dsp import biquad_coefficients
    except ImportError as error:
        fail(f"cannot import biquad_coefficients: {error}")
    if not GOLDEN.is_file():
        fail(f"missing golden {GOLDEN}")
    ref = json.loads(GOLDEN.read_text())
    if ref.get("case") != CASE_ID or ref.get("schema_version") != 1:
        fail("golden case identity mismatch")
    sr = float(ref["sample_rate_hz"])
    center_hz = float(ref["center_hz"])
    q = float(ref["q"])
    gain_db = float(ref["gain_db"])
    expected_coeffs = ref["coeffs_a1_a2_b0_b1_b2"]
    grid = ref["grid_hz"]
    expected_responses = ref["response_re_im"]
    if len(expected_coeffs) != 5:
        fail(f"golden coefficient vector length {len(expected_coeffs)} != 5")
    if len(grid) != len(expected_responses):
        fail("golden grid/complex-response length mismatch")
    if any(not isinstance(pair, list) or len(pair) != 2 for pair in expected_responses):
        fail("golden complex responses must contain real/imaginary pairs")
    try:
        require_finite_numbers(
            [sr, center_hz, q, gain_db, *expected_coeffs, *grid,
             *(value for pair in expected_responses for value in pair)],
            "golden biquad data",
        )
    except ValueError as error:
        fail(f"invalid finite golden input: {error}")
    got = biquad_coefficients("peak", center_hz, sr, q, gain_db)
    if len(got) != 5:
        fail(f"coefficient vector length {len(got)} != 5")
    for name, g, e in zip(("a1", "a2", "b0", "b1", "b2"), got, expected_coeffs):
        denom = abs(e) if e != 0 else 1.0
        err = abs(g - e) / denom
        if err > TOL_COEFF:
            fail(f"{name}: got={g:.15f} expected={e:.15f} rel_err={err:.3e}")
    a1, a2, b0, b1, b2 = got
    complex_pairs = []
    for f, (re, im) in zip(grid, expected_responses):
        z = cmath.exp(-1j * 2 * math.pi * f / sr)
        h = (b0 + b1 * z + b2 * z * z) / (1 + a1 * z + a2 * z * z)
        complex_pairs.append((h, complex(re, im)))
    for f, (h, ref_h) in zip(grid, complex_pairs):
        num = abs(h - ref_h)
        err = 0.0 if num == 0.0 else (num / abs(ref_h) if ref_h != 0 else math.inf)
        if err > TOL_REL:
            fail(f"H({f} Hz): got={h!r} expected={ref_h!r} rel_err={err:.3e}")
    try:
        max_abs_error, max_rel_error = _h02_error_metrics(
            got, expected_coeffs, complex_pairs
        )
    except ValueError as error:
        fail(f"invalid finite comparison: {error}")
    print(json.dumps({
        "QA_RESULT": True, "case": CASE_ID, "pass": True,
        "max_abs_error": max_abs_error, "max_rel_error": max_rel_error,
        "tolerance": TOL_REL, "tolerance_kind": "rel",
        "provenance": "wolfram-engine-15.0.0",
    }))


if __name__ == "__main__":
    main()
