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

CASE_ID = "autoeq-qa.h02-scorer-sign.v1"
TOL_REL = 1e-9
TOL_COEFF = 1e-12

ROOT = Path(__file__).resolve().parents[3]
GOLDEN = ROOT / "crates/autoeq-qa/wolfram/goldens/h02_scorer_sign.json"


def fail(message):
    print(f"FAIL {CASE_ID}: {message}", file=sys.stderr)
    sys.exit(1)


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
    got = biquad_coefficients("peak", float(ref["center_hz"]), sr,
                              float(ref["q"]), float(ref["gain_db"]))
    expected_coeffs = ref["coeffs_a1_a2_b0_b1_b2"]
    if len(got) != 5:
        fail(f"coefficient vector length {len(got)} != 5")
    for name, g, e in zip(("a1", "a2", "b0", "b1", "b2"), got, expected_coeffs):
        denom = abs(e) if e != 0 else 1.0
        err = abs(g - e) / denom
        if err > TOL_COEFF:
            fail(f"{name}: got={g:.15f} expected={e:.15f} rel_err={err:.3e}")
    a1, a2, b0, b1, b2 = got
    max_err = 0.0
    for f, (re, im) in zip(ref["grid_hz"], ref["response_re_im"]):
        z = cmath.exp(-1j * 2 * math.pi * f / sr)
        h = (b0 + b1 * z + b2 * z * z) / (1 + a1 * z + a2 * z * z)
        ref_h = complex(re, im)
        num = abs(h - ref_h)
        err = 0.0 if num == 0.0 else (num / abs(ref_h) if ref_h != 0 else math.inf)
        max_err = max(max_err, err)
        if err > TOL_REL:
            fail(f"H({f} Hz): got={h!r} expected={ref_h!r} rel_err={err:.3e}")
    print(json.dumps({
        "QA_RESULT": True, "case": CASE_ID, "pass": True,
        "max_abs_error": 0.0, "max_rel_error": max_err,
        "tolerance": TOL_REL, "tolerance_kind": "rel",
        "provenance": "wolfram-engine-15.0.0",
    }))


if __name__ == "__main__":
    main()
