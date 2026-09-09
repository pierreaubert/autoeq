"""File I/O functions for loading roomeq data files."""

import json
import sys
import math
import struct
from pathlib import Path


def read_fir_wav(filepath: Path) -> tuple[int, list[float]]:
    """Read mono IEEE-float FIR sidecars, including WAVE_FORMAT_EXTENSIBLE."""
    raw = filepath.read_bytes()
    if raw[:4] != b"RIFF" or raw[8:12] != b"WAVE":
        raise ValueError(f"Not a RIFF WAVE FIR: {filepath}")
    fmt = payload = None
    offset = 12
    while offset + 8 <= len(raw):
        kind, size = struct.unpack_from("<4sI", raw, offset)
        start = offset + 8
        if start + size > len(raw):
            raise ValueError(f"Truncated FIR WAV: {filepath}")
        if kind == b"fmt ":
            fmt = raw[start:start + size]
        elif kind == b"data":
            payload = raw[start:start + size]
        offset = start + size + (size & 1)
    if fmt is None or len(fmt) < 16 or payload is None:
        raise ValueError(f"Missing FIR WAV chunks: {filepath}")
    encoding, channels, rate, _, _, bits = struct.unpack_from("<HHIIHH", fmt)
    if encoding == 65534 and len(fmt) >= 40:
        encoding = struct.unpack_from("<H", fmt, 24)[0]
    if encoding != 3 or channels != 1 or bits not in (32, 64) or rate == 0:
        raise ValueError(f"Expected mono float FIR WAV: {filepath}")
    width = bits // 8
    if not payload or len(payload) % width:
        raise ValueError(f"Invalid FIR WAV samples: {filepath}")
    taps = [value[0] for value in struct.iter_unpack("<f" if bits == 32 else "<d", payload)]
    if not all(math.isfinite(v) for v in taps):
        raise ValueError(f"Non-finite FIR WAV: {filepath}")
    return rate, taps


def load_roomeq_json(filepath: Path) -> dict:
    """Load and parse roomeq JSON output file."""
    if not filepath.exists():
        print(f"Error: File not found: {filepath}")
        sys.exit(1)
    with open(filepath, "r") as f:
        data = json.load(f)
    # Bind physical sidecars before plots replay a driver's plugin chain.
    # These private in-memory fields are never written back to the DSP JSON.
    for channel in (data.get("channels") or {}).values():
        for driver in channel.get("drivers") or []:
            for plugin in driver.get("plugins") or []:
                parameters = plugin.get("parameters") or {}
                if plugin.get("plugin_type") == "convolution" and parameters.get("room_eq_fir_placement") == "per_driver":
                    rate, taps = read_fir_wav(filepath.parent / parameters["ir_file"])
                    parameters["_fir_sample_rate"] = rate
                    parameters["_fir_taps"] = taps
    return data
