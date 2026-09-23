#!/usr/bin/env python3
"""Wolfram cross-check: FIR test-helper bin-rounding budget (H01).

Oracle: crates/autoeq-qa/wolfram/h01_fir_bin_rounding.wls (independent
direct-sum DFT at bin centers and exact-frequency DTFT). Replays the
tests/fir_tests/compute.rs helper math (N=16 zero-padded FFT,
bin=Round(f/step), dB=20 log10(max(mag,1e-10))) with numpy, and budgets
the sampling approximation against the exact DTFT separately.
Tolerance: N, absolute error <= 1e-6 dB on the rounded-bin value.
"""

import json
import sys
from pathlib import Path

CASE_ID = "autoeq-qa.h01-fir-bin-rounding.v1"
TOL_ABS = 1e-6

ROOT = Path(__file__).resolve().parents[3]
GOLDEN = ROOT / "crates/autoeq-qa/wolfram/goldens/h01_fir_bin_rounding.json"


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
    taps = np.asarray(ref["taps"], dtype=np.float64)
    sr = float(ref["sample_rate_hz"])
    nfft = int(ref["fft_size"])
    spectrum = np.fft.rfft(np.pad(taps, (0, nfft - len(taps))), n=nfft)
    step = sr / nfft
    max_err = 0.0
    for f, b, e_bin, e_exact in zip(ref["grid_hz"], ref["bins"],
                                    ref["bin_db"], ref["exact_db"]):
        got_bin = int(round(f / step))
        if got_bin != int(b):
            fail(f"f={f} Hz: bin={got_bin} expected={b}")
        got_db = 20.0 * np.log10(max(abs(spectrum[got_bin]), 1e-10))
        err = abs(got_db - e_bin)
        max_err = max(max_err, err)
        if err > TOL_ABS:
            fail(f"f={f} Hz: bin_db={got_db:.9f} expected={e_bin:.9f} abs_err={err:.3e}")
        k = np.arange(len(taps))
        exact = 20.0 * np.log10(max(abs(np.sum(
            taps * np.exp(-2j * np.pi * f * k / sr))), 1e-10))
        gap = abs(e_bin - e_exact)
        if abs(exact - e_exact) > 1e-9:
            fail(f"f={f} Hz: exact DTFT mismatch (oracle inconsistent)")
        print(f"note {CASE_ID}: f={f:g} Hz bin={b} "
              f"sampling-gap={gap:.6f} dB", file=sys.stderr)
    print(json.dumps({
        "QA_RESULT": True, "case": CASE_ID, "pass": True,
        "max_abs_error": max_err, "tolerance": TOL_ABS,
        "tolerance_kind": "abs", "provenance": "wolfram-engine-15.0.0",
    }))


if __name__ == "__main__":
    main()
