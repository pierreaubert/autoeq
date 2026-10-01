#!/usr/bin/env python3
"""
Extract frequency response data (freq, SPL, phase) from MSO .msop files.

MSO (Multi-Sub Optimizer) project files are gzip-compressed blobs. The
decompressed payload embeds one data block per imported measurement with
this layout (all integers little-endian):

    [0x01][int32 name_len][name bytes (UTF-8)]
    [0x01 0x00][int32 point_count]
    [point_count float64 frequencies in Hz]
    [point_count interleaved float64 (real, imag) complex response pairs]

The complex response is stored as linear-domain real/imaginary parts, so
the familiar FRD quantities are recovered exactly as:

    spl_db    = 20 * log10(hypot(real, imag))
    phase_deg = atan2(imag, real) in degrees, wrapped to [-180, +180]

The (real, imag) order follows the universal C-struct convention and is
confirmed by causality: decoded subwoofer responses show the expected
lagging (decreasing) phase trend with frequency. Swapping the order would
only negate the exported phase; SPL is unaffected either way.

Like mdat2csv.py, block discovery is heuristic: every 0x01 0x00 marker
with a plausible point count is validated by checking that the next
point_count doubles form a strictly increasing (linear or log-spaced)
frequency axis inside sane bounds. Accidental matches are effectively
impossible, and measurement names referenced elsewhere in the project
(filter configs, graph definitions) never carry such a block, so only
real measurements are extracted.

Usage:
    python3 msop2csv.py <file.msop> [output_dir]
"""

import gzip
import math
import os
import re
import struct
import sys

GZIP_MAGIC = b'\x1f\x8b'
BLOCK_MARKER = b'\x01\x00'
MIN_POINTS = 8
MAX_NAME_LEN = 1024


def load_msop_bytes(filepath):
    """Read a .msop file, decompressing the gzip layer when present."""
    with open(filepath, 'rb') as f:
        raw = f.read()
    if raw[:2] == GZIP_MAGIC:
        try:
            return gzip.decompress(raw)
        except (OSError, EOFError) as exc:
            raise ValueError(f"Not a valid MSO .msop file: {exc}") from exc
    return raw


def _axis_spacing(freq):
    """
    Classify a validated-increasing axis as 'linear', 'log', or None.
    Returns None when the axis is neither uniformly nor geometrically spaced.
    """
    diffs = [b - a for a, b in zip(freq, freq[1:])]
    median = sorted(diffs)[len(diffs) // 2]
    if median <= 0:
        return None
    if max(abs(d - median) for d in diffs) <= 1e-3 * max(1.0, abs(median)):
        return 'linear'
    if freq[0] > 0:
        ratios = [b / a for a, b in zip(freq, freq[1:])]
        median_ratio = sorted(ratios)[len(ratios) // 2]
        if median_ratio > 1 and max(abs(r - median_ratio) for r in ratios) <= 1e-9:
            return 'log'
    return None


def _valid_frequency_axis(freq):
    """Check plausible measurement frequency-axis bounds and spacing."""
    if len(freq) < MIN_POINTS:
        return False
    if any(not math.isfinite(v) for v in freq):
        return False
    if any(b <= a for a, b in zip(freq, freq[1:])):
        return False
    if not (0.1 <= freq[0] <= 1000 and 5 <= freq[-1] <= 100000):
        return False
    return _axis_spacing(freq) is not None


def _valid_complex_block(pairs):
    """Check the interleaved (real, imag) pairs are finite and non-silent."""
    if any(not math.isfinite(v) for v in pairs):
        return False
    peak = 0.0
    for i in range(0, len(pairs), 2):
        peak = max(peak, math.hypot(pairs[i], pairs[i + 1]))
    return 0.0 < peak < 1e15


def recover_block_name(data, marker):
    """
    Recover the measurement name stored just before a block marker.

    Layout working backwards from marker: name bytes, int32 name length,
    ideally a 0x01 record byte. Returns None when no candidate decodes.
    """
    fallback = None
    for length in range(1, MAX_NAME_LEN + 1):
        len_pos = marker - 4 - length
        if len_pos < 0:
            break
        if struct.unpack('<i', data[len_pos:len_pos + 4])[0] != length:
            continue
        try:
            name = data[len_pos + 4:marker].decode('utf-8')
        except UnicodeDecodeError:
            continue
        if not name.strip() or not all(32 <= ord(c) < 127 for c in name):
            continue
        marked = len_pos > 0 and data[len_pos - 1] == 0x01
        if marked:
            return name
        if fallback is None:
            fallback = name
    return fallback


def find_measurement_blocks(data):
    """
    Scan decompressed bytes for measurement blocks.
    Returns a list of (marker_offset, point_count) tuples in file order.
    """
    blocks = []
    pos = 0
    while True:
        marker = data.find(BLOCK_MARKER, pos)
        if marker == -1:
            break
        if marker + 6 <= len(data):
            count = struct.unpack('<i', data[marker + 2:marker + 6])[0]
            need = 6 + 3 * count * 8
            if MIN_POINTS <= count and marker + need <= len(data):
                axis_end = marker + 6 + count * 8
                try:
                    freq = struct.unpack(f'<{count}d', data[marker + 6:axis_end])
                except struct.error:
                    freq = None
                if freq is not None and _valid_frequency_axis(freq):
                    pairs = struct.unpack(f'<{2 * count}d', data[axis_end:axis_end + 2 * count * 8])
                    if _valid_complex_block(pairs):
                        blocks.append((marker, count))
        pos = marker + 1
    return blocks


def decode_response(freq, pairs):
    """Convert stored (freq, interleaved complex) data to SPL/phase lists."""
    spl = []
    phase = []
    for i in range(len(freq)):
        real = pairs[2 * i]
        imag = pairs[2 * i + 1]
        spl.append(20 * math.log10(math.hypot(real, imag)))
        ph = math.degrees(math.atan2(imag, real))
        phase.append(ph)
    return spl, phase


def parse_msop(filepath):
    """
    Parse an MSO .msop file and extract measurements.
    Returns a list of measurement dicts with 'freq', 'spl', 'phase', 'name'.
    """
    data = load_msop_bytes(filepath)
    if len(data) < 64:
        raise ValueError("Not a valid MSO .msop file: payload too small")

    blocks = find_measurement_blocks(data)
    if not blocks:
        raise ValueError("No measurement data found in .msop file")

    measurements = []
    used_names = set()
    for index, (marker, count) in enumerate(blocks):
        axis_end = marker + 6 + count * 8
        freq = list(struct.unpack(f'<{count}d', data[marker + 6:axis_end]))
        pairs = struct.unpack(f'<{2 * count}d', data[axis_end:axis_end + 2 * count * 8])
        spl, phase = decode_response(freq, pairs)

        name = recover_block_name(data, marker)
        if not name:
            name = f"measurement_{index + 1}"
        base = name
        suffix = 2
        while name in used_names:
            name = f"{base}_{suffix}"
            suffix += 1
        used_names.add(name)

        measurements.append({
            'name': name,
            'freq': freq,
            'spl': spl,
            'phase': phase,
        })
    return measurements


def sanitize_filename(name, max_len=100):
    """Make a string safe for use as a filename."""
    name = re.sub(r'[^\w\-. ]', '_', name)
    name = re.sub(r'_+', '_', name).strip('_')
    if len(name) > max_len:
        name = name[:max_len]
    return name


def sanitize_identifier(name, max_len=100):
    """Create a stable underscore-separated filename or JSON object key."""
    identifier = sanitize_filename(name, max_len=max_len).replace(' ', '_')
    return re.sub(r'_+', '_', identifier).strip('_')


def export_csv(measurement, output_dir):
    """Export a single measurement to CSV."""
    os.makedirs(output_dir, exist_ok=True)
    name = sanitize_identifier(measurement['name'])
    filepath = os.path.join(output_dir, f"{name}.csv")

    freqs = measurement['freq']
    spl = measurement['spl']
    phase = measurement['phase']

    with open(filepath, 'w') as f:
        f.write("freq_hz,spl_db,phase_deg\n")
        for i in range(len(freqs)):
            s = f"{spl[i]:.6f}" if spl is not None else ""
            p = f"{phase[i]:.6f}" if phase is not None else ""
            f.write(f"{freqs[i]:.6f},{s},{p}\n")

    return filepath


def export_recordings_json(measurements, csv_paths, output_dir):
    """
    Export a recordings.json file conforming to the roomeq input_schema.json.
    Each measurement becomes a speaker entry referencing its CSV file.
    """
    import json

    speakers = {}
    for m, csv_path in zip(measurements, csv_paths):
        if csv_path is None:
            continue
        # Use relative path from recordings.json location
        rel_path = os.path.relpath(csv_path, output_dir)
        key = sanitize_identifier(m['name'])
        speakers[key] = {
            "path": rel_path,
            "name": m['name'],
        }

    config = {
        "version": "2.1.0",
        "speakers": speakers,
    }

    json_path = os.path.join(output_dir, "recordings.json")
    with open(json_path, 'w') as f:
        json.dump(config, f, indent=2)
        f.write('\n')

    return json_path


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <file.msop> [output_dir]")
        sys.exit(1)

    msop_path = sys.argv[1]
    if len(sys.argv) > 2:
        output_dir = sys.argv[2]
    else:
        output_dir = os.path.splitext(msop_path)[0] + "_csv"

    print(f"Parsing: {msop_path}")
    measurements = parse_msop(msop_path)

    print(f"Found {len(measurements)} measurements")
    csv_paths = []
    for i, m in enumerate(measurements):
        has_spl = m['spl'] is not None
        has_phase = m['phase'] is not None
        spl_range = ""
        if has_spl:
            spl_range = f" SPL=[{min(m['spl']):.1f}..{max(m['spl']):.1f}]"
        phase_range = ""
        if has_phase:
            phase_range = f" Phase=[{min(m['phase']):.1f}..{max(m['phase']):.1f}]"

        print(f"  [{i + 1}] {m['name']}: {len(m['freq'])} pts, "
              f"{m['freq'][0]:.1f}-{m['freq'][-1]:.0f} Hz, "
              f"{'SPL:yes' if has_spl else 'SPL:NO'}{spl_range}, "
              f"{'Phase:yes' if has_phase else 'Phase:NO'}{phase_range}")

        if has_spl or has_phase:
            filepath = export_csv(m, output_dir)
            csv_paths.append(filepath)
            print(f"      -> {filepath}")
        else:
            csv_paths.append(None)
            print("      -> SKIPPED (no SPL or phase data found)")

    json_path = export_recordings_json(measurements, csv_paths, output_dir)
    print(f"\n  recordings.json -> {json_path}")


if __name__ == '__main__':
    main()
