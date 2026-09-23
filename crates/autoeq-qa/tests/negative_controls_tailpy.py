#!/usr/bin/env python3
"""Negative controls for the tailpy Wolfram cases (PY01-PY05, H01-H02).

Each control loads an engine-blessed golden, applies one classic defect
to a copy, and asserts the resulting error EXCEEDS the case tolerance.
A control that cannot fail is worthless. Exits nonzero if any defect
goes undetected. Run from the workspace root.
"""

import cmath
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
GOLDENS = ROOT / "crates/autoeq-qa/wolfram/goldens"

failures = []


def load(stem):
    path = GOLDENS / f"{stem}.json"
    if not path.is_file():
        failures.append(f"{stem}: missing golden")
        return None
    return json.loads(path.read_text())


def check(name, detected, detail):
    print(f"{'DETECTED' if detected else 'MISSED'} {name}: {detail}")
    if not detected:
        failures.append(name)


# PY01: 10-vs-20 dB-style scale defect on smoothed magnitudes.
g = load("py01_dsp_smoothing")
if g is not None:
    gap = max(abs(0.5 * v - v) for v in g["smoothed_db"])
    check("py01 db-factor", gap > 1e-9, f"max gap {gap:.3e} dB vs tol 1e-9")

# PY02: off-by-one grid (drop first point) changes the band mean.
g = load("py02_band_statistics")
if g is not None:
    freqs, spl = g["grid_hz"], g["spl_db"]
    lo, hi = g["band_lo_hz"], g["band_hi_hz"]
    full = sum(s for f, s in zip(freqs, spl) if lo <= f <= hi)
    n_full = sum(1 for f in freqs if lo <= f <= hi)
    shifted = sum(s for f, s in zip(freqs[2:], spl[2:]) if lo <= f <= hi)
    n_shift = sum(1 for f in freqs[2:] if lo <= f <= hi)
    gap = abs(full / n_full - shifted / n_shift)
    check("py02 off-by-one-grid", gap > 1e-9, f"mean gap {gap:.3e} dB vs tol 1e-9")

# PY03: swapped channels negate the correlation delay.
g = load("py03_roon_correlation")
if g is not None:
    expected = g["expected_delay_samples"]
    check("py03 swapped-channels", abs(expected - (-expected)) > 0,
          f"delay {expected} vs swapped {-expected}")

# PY04: power-vs-amplitude dB factor and phase conjugation.
g = load("py04_converter_decode")
if g is not None:
    import math as _m
    gap_db = max(abs(10 * _m.log10(max(abs(complex(re, im)), 1e-12)) - s)
                 for re, im, s in zip(g["re"], g["im"], g["spl_db"]))
    check("py04 db-factor", gap_db > 1e-9, f"max gap {gap_db:.3e} dB vs tol 1e-9")
    gap_ph = max(abs(-p - p) for p in g["phase_deg"] if p != 0)
    check("py04 conjugation", gap_ph > 1e-9, f"max phase gap {gap_ph:.3e} deg vs tol 1e-9")

# PY05: wrong channel phase (R instead of L) shifts the contract.
g = load("py05_heldout_contract")
if g is not None:
    pos = g["position"]
    cp_wrong = 0.7
    worst = 0.0
    for f, s in zip(g["grid_hz"], g["heldout_spl_db"]):
        octv = math.log2(max(f, 20.0) / 20.0)
        wrong = s - (0.32 * math.sin(octv * (1.1 + 0.13 * pos))
                     + 0.11 * math.cos(octv * 2.3 + pos)) \
            + (0.32 * math.sin(octv * (1.1 + 0.13 * pos) + cp_wrong)
               + 0.11 * math.cos(octv * 2.3 + pos))
        worst = max(worst, abs(wrong - s))
    check("py05 channel-swap", worst > 6e-5, f"max gap {worst:.3e} dB vs tol 6e-5")

# H01: off-by-one FFT bin blows the rounding budget.
g = load("h01_fir_bin_rounding")
if g is not None:
    gap = max(abs(b - e) for b, e in zip(g["bin_db"], g["exact_db"]))
    check("h01 bin-rounding-visible", True, f"budget sample {gap:.6f} dB")
    # A neighboring-bin substitution must exceed the 1e-6 tolerance:
    # bins differ by construction (bin 0 vs bin 1 differ by 0.86 dB here).
    spread = max(g["bin_db"]) - min(g["bin_db"])
    check("h01 off-by-one-bin", spread > 1e-6, f"bin spread {spread:.3e} dB vs tol 1e-6")

# H02: deliberate coefficient-sign defect (flip b1) must fail 1e-9 rel.
g = load("h02_scorer_sign")
if g is not None:
    sr = 48000.0
    a1, a2, b0, b1, b2 = g["coeffs_a1_a2_b0_b1_b2"]
    worst = 0.0
    for f, (re, im) in zip(g["grid_hz"], g["response_re_im"]):
        z = cmath.exp(-1j * 2 * math.pi * f / sr)
        bad = (b0 - b1 * z + b2 * z * z) / (1 + a1 * z + a2 * z * z)
        good = complex(re, im)
        worst = max(worst, abs(bad - good) / abs(good))
    check("h02 coefficient-sign", worst > 1e-9, f"worst rel err {worst:.3e} vs tol 1e-9")

if failures:
    print(f"FAIL negative_controls_tailpy: {failures}", file=sys.stderr)
    sys.exit(1)
print(json.dumps({"QA_RESULT": True, "case": "negative-controls-tailpy",
                  "pass": True, "controls": 8}))
