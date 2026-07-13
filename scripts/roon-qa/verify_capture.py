#!/usr/bin/env python3
"""Compare private BlackHole captures with the exported Roon manifest/archive."""

from __future__ import annotations

import argparse
import io
import json
import math
import zipfile
from pathlib import Path

import numpy as np
from scipy import signal
from scipy.io import wavfile


def read_capture(path: Path) -> tuple[int, np.ndarray]:
    rate, data = wavfile.read(path)
    data = np.asarray(data, dtype=np.float64)
    if data.ndim == 1:
        data = data[:, None]
    if rate != 48_000 or data.shape[1] < 2:
        raise ValueError(f"{path}: expected at least two channels at 48 kHz")
    return rate, data


def transfer(reference: np.ndarray, measured: np.ndarray, rate: int):
    size = min(len(reference), len(measured))
    reference = reference[:size]
    measured = measured[:size]
    freq, pxx = signal.welch(reference, rate, nperseg=min(65536, size))
    _, pyx = signal.csd(measured, reference, rate, nperseg=min(65536, size))
    response = pyx / np.maximum(pxx, np.finfo(float).tiny)
    valid = pxx > np.max(pxx) * 1e-8
    return freq, response, valid


def biquad(band: dict, rate: int):
    kind = band["type"]
    frequency = float(band["frequency"])
    q = float(band["q"])
    gain = float(band["gain"])
    w0 = 2 * math.pi * frequency / rate
    c, s = math.cos(w0), math.sin(w0)
    alpha = s / (2 * q)
    a = 10 ** (gain / 40)
    if kind == "Peak/Dip":
        b = [1 + alpha * a, -2 * c, 1 - alpha * a]
        aa = [1 + alpha / a, -2 * c, 1 - alpha / a]
    elif kind == "Low Pass":
        b = [(1 - c) / 2, 1 - c, (1 - c) / 2]
        aa = [1 + alpha, -2 * c, 1 - alpha]
    elif kind == "High Pass":
        b = [(1 + c) / 2, -(1 + c), (1 + c) / 2]
        aa = [1 + alpha, -2 * c, 1 - alpha]
    elif kind == "Band Pass":
        b = [alpha, 0, -alpha]
        aa = [1 + alpha, -2 * c, 1 - alpha]
    elif kind == "Band Stop":
        b = [1, -2 * c, 1]
        aa = [1 + alpha, -2 * c, 1 - alpha]
    elif kind in ("Low Shelf", "High Shelf"):
        # Match math_audio_iir_fir::Biquad's shelf convention exactly.
        two = math.sqrt(2 * a) * s
        if kind == "Low Shelf":
            b = [a * ((a + 1) - (a - 1) * c + two),
                 2 * a * ((a - 1) - (a + 1) * c),
                 a * ((a + 1) - (a - 1) * c - two)]
            aa = [(a + 1) + (a - 1) * c + two,
                  -2 * ((a - 1) + (a + 1) * c),
                  (a + 1) + (a - 1) * c - two]
        else:
            b = [a * ((a + 1) + (a - 1) * c + two),
                 -2 * a * ((a - 1) + (a + 1) * c),
                 a * ((a + 1) + (a - 1) * c - two)]
            aa = [(a + 1) - (a - 1) * c + two,
                  2 * ((a - 1) - (a + 1) * c),
                  (a + 1) - (a - 1) * c - two]
    else:
        raise ValueError(f"unsupported Roon band type {kind!r}")
    return np.asarray(b) / aa[0], np.asarray(aa) / aa[0]


def expected_iir(manifest: dict, frequencies: np.ndarray, rate: int):
    channel = manifest["channels"]["left"]
    response = np.full(frequencies.shape, 10 ** (channel.get("headroom_gain_db", 0) / 20), complex)
    for band in channel.get("parametric_eq", {}).get("bands", []):
        b, a = biquad(band, rate)
        _, h = signal.freqz(b, a, worN=frequencies, fs=rate)
        response *= h
    return response


def expected_fir(archive: Path, frequencies: np.ndarray, rate: int):
    with zipfile.ZipFile(archive) as bundle:
        names = sorted(name for name in bundle.namelist() if name.startswith("filters/00_") and name.endswith(".wav"))
        if len(names) != 1:
            raise ValueError("Roon archive must contain exactly one first-channel mono WAV")
        ir_rate, taps = wavfile.read(io.BytesIO(bundle.read(names[0])))
    if ir_rate != rate or np.asarray(taps).ndim != 1:
        raise ValueError("archived first-channel IR has the wrong rate or channel count")
    _, response = signal.freqz(np.asarray(taps, dtype=np.float64), worN=frequencies, fs=rate)
    return response


def magnitude_error(measured, expected, frequencies, valid):
    band = (frequencies >= 30) & (frequencies <= 18_000) & valid
    delta = 20 * np.log10(np.maximum(np.abs(measured[band]), 1e-12)) \
        - 20 * np.log10(np.maximum(np.abs(expected[band]), 1e-12))
    return float(np.max(np.abs(delta))), float(np.sqrt(np.mean(delta * delta)))


def relative_delay_samples(data: np.ndarray) -> int:
    left, right = data[:, 0], data[:, 1]
    correlation = signal.correlate(left, right, mode="full", method="fft")
    return int(np.argmax(np.abs(correlation)) - (len(right) - 1))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--captures", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--roon-version", required=True)
    parser.add_argument("--zone", required=True)
    parser.add_argument("--ui-readback", type=Path, required=True)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text())
    rate, baseline = read_capture(args.captures / "baseline.wav")
    modes = {name: read_capture(args.captures / f"{name}.wav")[1]
             for name in ("iir", "fir", "combined")}
    frequencies, measured_iir, valid = transfer(baseline[:, 0], modes["iir"][:, 0], rate)
    iir = expected_iir(manifest, frequencies, rate)
    expected = {"iir": iir}
    if "convolution_archive" in manifest:
        fir = expected_fir(args.manifest.parent / manifest["convolution_archive"]["file"], frequencies, rate)
    else:
        fir = np.ones_like(iir)
    expected.update(fir=fir, combined=iir * fir)

    results = {}
    for mode in ("iir", "fir", "combined"):
        _, measured, mode_valid = transfer(baseline[:, 0], modes[mode][:, 0], rate)
        peak, rms = magnitude_error(measured, expected[mode], frequencies, valid & mode_valid)
        results[mode] = {"peak_magnitude_error_db": peak, "rms_magnitude_error_db": rms}
        if peak > 0.5:
            raise SystemExit(f"{mode} magnitude error {peak:.3f} dB exceeds 0.5 dB")

    measured_delay = relative_delay_samples(modes["iir"]) - relative_delay_samples(baseline)
    left_delay = manifest["channels"].get("left", {}).get("delay_ms", 0)
    right_delay = manifest["channels"].get("right", {}).get("delay_ms", 0)
    expected_delay = round((left_delay - right_delay) * rate / 1000)
    delay_error = abs(measured_delay - expected_delay)
    if delay_error > 2:
        raise SystemExit(f"relative delay error {delay_error} samples exceeds two samples")

    report = {
        "roon_version": args.roon_version,
        "qa_zone": args.zone,
        "manifest": str(args.manifest),
        "response": results,
        "relative_delay": {
            "measured_samples": measured_delay,
            "expected_samples": expected_delay,
            "error_samples": delay_error,
        },
        "ui_readback": json.loads(args.ui_readback.read_text()),
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
