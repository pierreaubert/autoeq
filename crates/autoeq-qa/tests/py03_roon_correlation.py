#!/usr/bin/env python3
"""Wolfram cross-check: Roon verifier inter-channel delay (PY03).

Oracle: crates/autoeq-qa/wolfram/py03_roon_correlation.wls (independent
full linear correlation, ArgMax|.| minus (len-1)). Compares
scripts/roon-qa/verify_capture.py relative_delay_samples on synthetic PCM.
Tolerance: exact integer samples (defect class: sign flip / off-by-one).
"""

import importlib.util
import json
import sys
from pathlib import Path

CASE_ID = "autoeq-qa.py03-roon-correlation.v1"

ROOT = Path(__file__).resolve().parents[3]
GOLDEN = ROOT / "crates/autoeq-qa/wolfram/goldens/py03_roon_correlation.json"
MODULE = ROOT / "scripts/roon-qa/verify_capture.py"


def fail(message):
    print(f"FAIL {CASE_ID}: {message}", file=sys.stderr)
    sys.exit(1)


def main():
    import numpy as np

    if not GOLDEN.is_file():
        fail(f"missing golden {GOLDEN}")
    ref = json.loads(GOLDEN.read_text())
    if ref.get("case") != CASE_ID or ref.get("schema_version") != 1:
        fail("golden case identity mismatch")
    spec = importlib.util.spec_from_file_location("verify_capture", MODULE)
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception as error:
        fail(f"cannot load verify_capture: {error}")
    left = np.asarray(ref["left"], dtype=np.float64)
    right = np.asarray(ref["right"], dtype=np.float64)
    if len(left) != len(right):
        fail("golden left/right length mismatch")
    data = np.column_stack([left, right])
    got = mod.relative_delay_samples(data)
    expected = int(ref["expected_delay_samples"])
    if got != expected:
        fail(f"delay={got} samples expected={expected} samples")
    print(json.dumps({
        "QA_RESULT": True, "case": CASE_ID, "pass": True,
        "max_abs_error": 0, "tolerance": 0,
        "tolerance_kind": "abs", "provenance": "wolfram-engine-15.0.0",
    }))


if __name__ == "__main__":
    main()
