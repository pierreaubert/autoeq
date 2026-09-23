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

CASE_ID = "autoeq-qa.py05-heldout-contract.v1"
TOL_ABS = 6e-5

ROOT = Path(__file__).resolve().parents[3]
GOLDEN = ROOT / "crates/autoeq-qa/wolfram/goldens/py05_heldout_contract.json"
MODULE = ROOT / "scripts/generate_roomeq_held_out.py"


def fail(message):
    print(f"FAIL {CASE_ID}: {message}", file=sys.stderr)
    sys.exit(1)


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
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "L.csv"
        dst = Path(tmp) / "L_heldout_1.csv"
        with src.open("w", encoding="utf-8") as handle:
            handle.write("frequency_hz,spl_db,phase_deg\n")
            for f, s, p in zip(ref["grid_hz"], ref["base_spl_db"], ref["base_phase_deg"]):
                handle.write(f"{f},{s},{p}\n")
        mod.generate(src, dst, int(ref["position"]), ref["channel"])
        lines = dst.read_text(encoding="utf-8").strip().splitlines()
    if len(lines) != len(ref["grid_hz"]) + 1:
        fail(f"row count {len(lines)} != {len(ref['grid_hz']) + 1}")
    max_err = 0.0
    for i, line in enumerate(lines[1:]):
        f, s, p = (float(v) for v in line.split(","))
        for name, g, e in (("spl", s, ref["heldout_spl_db"][i]),
                           ("phase", p, ref["heldout_phase_deg"][i])):
            err = abs(g - e)
            max_err = max(max_err, err)
            if err > TOL_ABS:
                fail(f"{name}[{f} Hz]: got={g:.6f} expected={e:.6f} abs_err={err:.3e}")
    print(json.dumps({
        "QA_RESULT": True, "case": CASE_ID, "pass": True,
        "max_abs_error": max_err, "tolerance": TOL_ABS,
        "tolerance_kind": "abs", "provenance": "wolfram-engine-15.0.0",
    }))


if __name__ == "__main__":
    main()
