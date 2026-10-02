#!/usr/bin/env python3
"""Wolfram cross-check: report-side band-mean statistic (PY02).

Oracle: crates/autoeq-qa/wolfram/py02_band_statistics.wls (independent
arithmetic mean over the closed band). Compares
scripts/src/acoustic_report.py band_mean, including the empty-band None
contract. Tolerance: A, absolute error <= 1e-9 dB.
"""

import json
import sys
from pathlib import Path

from qa_support.metrics import maximum_error_metrics, require_finite_numbers

CASE_ID = "autoeq-qa.py02-band-statistics.v1"
TOL_ABS = 1e-9

ROOT = Path(__file__).resolve().parents[3]
GOLDEN = ROOT / "crates/autoeq-qa/wolfram/goldens/py02_band_statistics.json"


def fail(message):
    print(f"FAIL {CASE_ID}: {message}", file=sys.stderr)
    sys.exit(1)


def main():
    sys.path.insert(0, str(ROOT))
    try:
        from scripts.src.acoustic_report import band_mean
    except ImportError as error:
        fail(f"cannot import band_mean: {error}")
    if not GOLDEN.is_file():
        fail(f"missing golden {GOLDEN}")
    ref = json.loads(GOLDEN.read_text())
    if ref.get("case") != CASE_ID or ref.get("schema_version") != 1:
        fail("golden case identity mismatch")
    if len(ref["grid_hz"]) != len(ref["spl_db"]):
        fail("golden grid/SPL length mismatch")
    try:
        require_finite_numbers(
            [*ref["grid_hz"], *ref["spl_db"], ref["band_lo_hz"],
             ref["band_hi_hz"], ref["band_mean_db"]],
            "golden band-statistics data",
        )
    except ValueError as error:
        fail(f"invalid finite golden input: {error}")
    got = band_mean(ref["grid_hz"], ref["spl_db"], ref["band_lo_hz"], ref["band_hi_hz"])
    expected = ref["band_mean_db"]
    if got is None:
        fail("band_mean returned None for a non-empty band")
    try:
        max_abs_err, max_rel_err = maximum_error_metrics([(got, expected)])
    except ValueError as error:
        fail(f"invalid finite comparison: {error}")
    if max_abs_err > TOL_ABS:
        fail(f"band_mean={got:.12f} expected={expected:.12f} abs_err={max_abs_err:.3e}")
    if band_mean(ref["grid_hz"], ref["spl_db"], 6000.0, 7000.0) is not None:
        fail("empty band must return None")
    print(json.dumps({
        "QA_RESULT": True, "case": CASE_ID, "pass": True,
        "max_abs_error": max_abs_err, "max_rel_error": max_rel_err,
        "tolerance": TOL_ABS,
        "tolerance_kind": "abs", "provenance": "wolfram-engine-15.0.0",
    }))


if __name__ == "__main__":
    main()
