#!/usr/bin/env python3
"""Wolfram cross-check: converter SPL/phase decode + rounding (PY04).

Oracle: crates/autoeq-qa/wolfram/py04_converter_decode.wls (independent
SPL=20 log10|H|, phase=Arg(H) in degrees, 6-decimal CSV rounding).
Compares utils/msop2csv.py decode_response plus the f"{v:.6f}" CSV
format used by mdat2csv/msop2csv exporters.
Tolerance: A, absolute error <= 1e-9 (native units); rounding exact.
"""

import importlib.util
import json
import sys
from pathlib import Path

CASE_ID = "autoeq-qa.py04-converter-decode.v1"
TOL_ABS = 1e-9

ROOT = Path(__file__).resolve().parents[3]
GOLDEN = ROOT / "crates/autoeq-qa/wolfram/goldens/py04_converter_decode.json"
MODULE = ROOT / "utils/msop2csv.py"


def fail(message):
    print(f"FAIL {CASE_ID}: {message}", file=sys.stderr)
    sys.exit(1)


def main():
    if not GOLDEN.is_file():
        fail(f"missing golden {GOLDEN}")
    ref = json.loads(GOLDEN.read_text())
    if ref.get("case") != CASE_ID or ref.get("schema_version") != 1:
        fail("golden case identity mismatch")
    spec = importlib.util.spec_from_file_location("msop2csv", MODULE)
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception as error:
        fail(f"cannot load msop2csv: {error}")
    freq = ref["grid_hz"]
    pairs = [v for pair in zip(ref["re"], ref["im"]) for v in pair]
    spl, phase = mod.decode_response(list(freq), list(pairs))
    if not (len(spl) == len(phase) == len(freq)):
        fail("decode_response length mismatch")
    max_err = 0.0
    for i, f in enumerate(freq):
        for name, g, e in (("spl", spl[i], ref["spl_db"][i]),
                           ("phase", phase[i], ref["phase_deg"][i])):
            err = abs(g - e)
            max_err = max(max_err, err)
            if err > TOL_ABS:
                fail(f"{name}[{f} Hz]: got={g:.12f} expected={e:.12f} abs_err={err:.3e}")
    for i in range(len(freq)):
        if f"{spl[i]:.6f}" != ref["spl_rounded"][i]:
            fail(f"spl rounding row {i}: {spl[i]:.6f} != {ref['spl_rounded'][i]}")
        if f"{phase[i]:.6f}" != ref["phase_rounded"][i]:
            fail(f"phase rounding row {i}: {phase[i]:.6f} != {ref['phase_rounded'][i]}")
    print(json.dumps({
        "QA_RESULT": True, "case": CASE_ID, "pass": True,
        "max_abs_error": max_err, "tolerance": TOL_ABS,
        "tolerance_kind": "abs", "provenance": "wolfram-engine-15.0.0",
    }))


if __name__ == "__main__":
    main()
