"""File I/O functions for loading roomeq data files."""

import csv
import json
import sys
import math
import struct
from pathlib import Path


class RoomEqData(dict):
    """Keep the source directory outside the serialized payload being verified."""

    def __init__(self, payload, source_directory):
        super().__init__(payload)
        self.source_directory = source_directory
        # Sibling `<stem>_files` directory holding sidecars/curves when present.
        self.assets_directory = source_directory
        # Measurement index re-injected from external CSV/JSON files, keyed
        # by channel/field. Payload verification strips these before hashing
        # so the ledger binds the slim saved bytes, not the viewer overlay.
        self.measurement_index = {}
        # Set when the loader installs a missing `deployed_source_curves`
        # dict to hold external curves. Payload verification drops the key
        # again once overlaid entries are stripped, since Rust omits it
        # when empty and the ledger binds the slim saved bytes.
        self.deployed_source_curves_created = False


class FirParameters(dict):
    """Expose plot-only replay caches without changing the serialized graph."""

    def __init__(self, parameters, rate, taps):
        super().__init__(parameters)
        self._replay_cache = {"_fir_sample_rate": rate, "_fir_taps": taps}

    def get(self, key, default=None):
        if key in self._replay_cache:
            return self._replay_cache[key]
        return super().get(key, default)


def assets_dir_for(json_path: Path) -> Path:
    """Sibling assets directory for a native output path.

    `dsp.json` -> `<parent>/dsp_files/`; mirrors
    `roomeq-workflow/src/output_bundle.rs`.
    """
    parent = json_path.parent
    stem = json_path.stem or "dsp"
    return parent / f"{stem}_files"


def candidate_asset_dirs(json_path: Path):
    """Parent first (legacy layout), then the sibling assets directory."""
    parent = json_path.parent
    assets = assets_dir_for(json_path)
    if assets == parent:
        return [parent]
    return [parent, assets]


def resolve_asset(json_path: Path, reference: str) -> Path:
    """Resolve a bare sidecar/curve reference against candidate directories."""
    candidate = Path(reference)
    if candidate.is_absolute():
        return candidate
    for directory in candidate_asset_dirs(json_path):
        resolved = directory / candidate
        if resolved.is_file():
            return resolved
    return json_path.parent / candidate


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


def read_curve_csv(path: Path) -> dict:
    """Read a measurement curve CSV written by the output bundle."""
    freq: list[float] = []
    spl: list[float] = []
    phase: list[float] = []
    has_phase = False
    with path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "freq" not in reader.fieldnames or "spl" not in reader.fieldnames:
            raise ValueError(f"Invalid curve CSV: {path}")
        has_phase = "phase" in reader.fieldnames
        for row in reader:
            freq.append(float(row["freq"]))
            spl.append(float(row["spl"]))
            if has_phase:
                raw = (row.get("phase") or "").strip()
                phase.append(float(raw) if raw not in {"", "nan", "NaN", "NAN"} else float("nan"))
    curve: dict = {"freq": freq, "spl": spl}
    if has_phase:
        curve["phase"] = phase
    return curve


def read_ir_csv(path: Path) -> dict:
    """Read an impulse-response CSV written by the output bundle."""
    time_ms: list[float] = []
    amplitude: list[float] = []
    with path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "time_ms" not in reader.fieldnames or "amplitude" not in reader.fieldnames:
            raise ValueError(f"Invalid IR CSV: {path}")
        for row in reader:
            time_ms.append(float(row["time_ms"]))
            amplitude.append(float(row["amplitude"]))
    return {"time_ms": time_ms, "amplitude": amplitude}


_CURVE_FIELDS = ("initial_curve", "final_curve", "eq_response", "target_curve")
_IR_FIELDS = ("pre_ir", "post_ir")
_BLOB_FIELDS = ("early_late_curves", "waterfall", "resonance_decays", "wavelet")


def _apply_measurement_overlay(data: dict, json_path: Path) -> dict:
    """Re-inject external measurement files into an in-memory copy.

    Returns the overlay index `{channel: {field: filename}}` plus a
    `deployed_source_curves` entry. Missing files are skipped so legacy
    outputs (fully embedded) and slim outputs (fully external) both load.
    """
    overlay: dict = {}
    assets = assets_dir_for(json_path)
    index_path = assets / "measurements_index.json"
    index: dict = {}
    if index_path.is_file():
        try:
            index = json.loads(index_path.read_text())
        except (ValueError, OSError):
            index = {}
    channels_index = (index.get("channels") or {}) if isinstance(index, dict) else {}
    deployed_index = (index.get("deployed_source_curves") or {}) if isinstance(index, dict) else {}
    if isinstance(index, dict) and isinstance(index.get("symmetric_pairs"), dict):
        if isinstance(data, RoomEqData):
            data.symmetric_pairs = index["symmetric_pairs"]

    for name, channel in (data.get("channels") or {}).items():
        if not isinstance(channel, dict):
            continue
        entry = channels_index.get(name) if isinstance(channels_index, dict) else None
        entry = entry if isinstance(entry, dict) else {}
        for field in _CURVE_FIELDS + _IR_FIELDS:
            if channel.get(field) is not None:
                continue
            filename = entry.get(field)
            if not isinstance(filename, str):
                continue
            path = assets / filename
            if not path.is_file():
                continue
            try:
                channel[field] = read_ir_csv(path) if field in _IR_FIELDS else read_curve_csv(path)
            except (ValueError, OSError):
                continue
            overlay.setdefault(name, {})[field] = filename
        for field in _BLOB_FIELDS:
            if channel.get(field) is not None:
                continue
            filename = entry.get(field)
            if not isinstance(filename, str):
                continue
            path = assets / filename
            if not path.is_file():
                continue
            try:
                channel[field] = json.loads(path.read_text())
            except (ValueError, OSError):
                continue
            overlay.setdefault(name, {})[field] = filename
        drivers = channel.get("drivers") or []
        for driver_index, driver in enumerate(drivers):
            if not isinstance(driver, dict) or driver.get("measured_acoustics") is not None:
                continue
            for kind, filename in entry.items():
                if (kind.startswith(f"driver{driver_index}_")
                        and kind.endswith("_measured_acoustics") and isinstance(filename, str)):
                    try:
                        driver["measured_acoustics"] = json.loads((assets / filename).read_text())
                        overlay.setdefault(name, {})[kind] = filename
                    except (ValueError, OSError):
                        pass
        for driver_index, driver in enumerate(drivers):
            if not isinstance(driver, dict) or driver.get("initial_curve") is not None:
                continue
            kind = None
            filename = None
            if isinstance(entry, dict):
                for key, value in entry.items():
                    if key.startswith(f"driver{driver_index}_") and key.endswith("_initial_curve") and isinstance(value, str):
                        kind, filename = key, value
                        break
            if filename is None:
                continue
            path = assets / filename
            if not path.is_file():
                continue
            try:
                driver["initial_curve"] = read_curve_csv(path)
            except (ValueError, OSError):
                continue
            overlay.setdefault(name, {})[kind] = filename

    deployed = data.get("deployed_source_curves")
    if deployed is None:
        # Rust omits the key when empty (skip_serializing_if), so only
        # install it when at least one external curve will actually load.
        # An untracked empty dict would break payload verification, which
        # hashes the slim saved bytes.
        loadable = isinstance(deployed_index, dict) and any(
            isinstance(filename, str) and (assets / filename).is_file()
            for filename in deployed_index.values()
        )
        if not loadable:
            return overlay
        deployed = {}
        data["deployed_source_curves"] = deployed
        if isinstance(data, RoomEqData):
            data.deployed_source_curves_created = True
    if isinstance(deployed, dict) and isinstance(deployed_index, dict):
        for name, filename in deployed_index.items():
            if name in deployed or not isinstance(filename, str):
                continue
            path = assets / filename
            if not path.is_file():
                continue
            try:
                deployed[name] = read_curve_csv(path)
            except (ValueError, OSError):
                continue
            overlay.setdefault("deployed_source_curves", {})[name] = filename
    return overlay


def load_roomeq_json(filepath: Path) -> dict:
    """Load and parse roomeq JSON output file."""
    if not filepath.exists():
        print(f"Error: File not found: {filepath}")
        sys.exit(1)
    filepath = Path(filepath)
    with open(filepath, "r") as f:
        payload = json.load(f)
    assets = assets_dir_for(filepath)
    source_directory = assets if assets.is_dir() else filepath.resolve().parent
    data = RoomEqData(payload, source_directory)
    data.assets_directory = assets
    # Re-inject slim-output measurement files so plots see the same curves
    # as legacy embedded outputs. Overlay keys are tracked for payload
    # verification, which strips them before hashing.
    try:
        data.measurement_index = _apply_measurement_overlay(data, filepath)
    except (OSError, ValueError):
        data.measurement_index = {}
    # Bind physical sidecars before plots replay a driver's plugin chain.
    # These private in-memory fields are never written back to the DSP JSON.
    for channel in (data.get("channels") or {}).values():
        for driver in channel.get("drivers") or []:
            for plugin in driver.get("plugins") or []:
                parameters = plugin.get("parameters") or {}
                if plugin.get("plugin_type") == "convolution" and parameters.get("room_eq_fir_placement") == "per_driver":
                    resolved = resolve_asset(filepath, parameters["ir_file"])
                    rate, taps = read_fir_wav(resolved)
                    plugin["parameters"] = FirParameters(parameters, rate, taps)
    return data
