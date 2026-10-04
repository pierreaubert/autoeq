"""File I/O functions for loading roomeq data files."""

import csv
import hashlib
import io
import json
import sys
import math
import re
import struct
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Optional, Union


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
        # Keep manifested bundle bytes in a private snapshot while this loaded
        # object is alive, so later consumers cannot reopen changed originals.
        self._bundle_snapshot_owner = None
        # Set when the loader installs a missing `deployed_source_curves`
        # dict to hold external curves. Payload verification drops the key
        # again once overlaid entries are stripped, since Rust omits it
        # when empty and the ledger binds the slim saved bytes.
        self.deployed_source_curves_created = False

    def close(self):
        """Release the private manifested-bundle snapshot when no longer needed."""
        owner = getattr(self, "_bundle_snapshot_owner", None)
        if hasattr(self, "_bundle_snapshot_owner"):
            self._bundle_snapshot_owner = None
        if owner is not None:
            owner.cleanup()

    def __del__(self):
        self.close()


class FirParameters(dict):
    """Expose plot-only replay caches without changing the serialized graph."""

    def __init__(self, parameters, rate, taps):
        super().__init__(parameters)
        self._replay_cache = {"_fir_sample_rate": rate, "_fir_taps": taps}

    def get(self, key, default=None):
        if key in self._replay_cache:
            return self._replay_cache[key]
        return super().get(key, default)


MAX_ARTIFACT_MANIFEST_BYTES = 2 * 1024 * 1024
MAX_ARTIFACT_MANIFEST_FILES = 50_000
MAX_ARTIFACT_MEMBER_PATH_BYTES = 1_024
MAX_ARTIFACT_MEMBER_BYTES = 256 * 1024 * 1024
MAX_NATIVE_GRAPH_BYTES = 512 * 1024 * 1024


@dataclass(frozen=True)
class _BundleMemberIdentity:
    size_bytes: int
    sha256: str


def _read_bounded_bytes(path: Path, maximum_bytes: int, label: str) -> bytes:
    if path.stat().st_size > maximum_bytes:
        raise ValueError(f"{label} exceeds its size limit")
    with path.open("rb") as handle:
        data = handle.read(maximum_bytes + 1)
    if len(data) > maximum_bytes:
        raise ValueError(f"{label} exceeds its size limit")
    return data


def _read_verified_bundle_member_bytes(
        assets: Path, filename: str,
        verified_members: dict[str, _BundleMemberIdentity],
        maximum_bytes: int = MAX_ARTIFACT_MEMBER_BYTES) -> bytes:
    identity = verified_members.get(filename)
    if identity is None:
        raise ValueError(f"Bundle member is not bound by the manifest: {filename!r}")
    if identity.size_bytes > maximum_bytes:
        raise ValueError(f"Bundle member exceeds its size limit: {filename!r}")
    path = _safe_bundle_member(assets, filename)
    raw = _read_bounded_bytes(path, min(identity.size_bytes, maximum_bytes),
                              f"Bundle member {filename!r}")
    if len(raw) != identity.size_bytes or hashlib.sha256(raw).hexdigest() != identity.sha256:
        raise ValueError(f"Bundle member failed integrity validation: {filename!r}")
    return raw


def _snapshot_verified_bundle_members(
        source_assets: Path,
        verified_members: dict[str, _BundleMemberIdentity]) -> tuple[object, Path]:
    owner = tempfile.TemporaryDirectory(prefix="roomeq-bundle-")
    root = Path(owner.name)
    try:
        for filename, identity in verified_members.items():
            if identity.size_bytes > MAX_ARTIFACT_MEMBER_BYTES:
                raise ValueError(f"Bundle member exceeds its size limit: {filename!r}")
            source = _safe_bundle_member(source_assets, filename)
            destination = root.joinpath(*PurePosixPath(filename).parts)
            destination.parent.mkdir(parents=True, exist_ok=True)
            digest = hashlib.sha256()
            copied = 0
            with source.open("rb") as reader, destination.open("xb") as writer:
                while copied <= identity.size_bytes:
                    remaining_plus_one = identity.size_bytes - copied + 1
                    chunk = reader.read(min(64 * 1024, remaining_plus_one))
                    if not chunk:
                        break
                    copied += len(chunk)
                    if copied > identity.size_bytes:
                        raise ValueError(f"Bundle member changed during snapshot: {filename!r}")
                    digest.update(chunk)
                    writer.write(chunk)
            if copied != identity.size_bytes or digest.hexdigest() != identity.sha256:
                raise ValueError(f"Bundle member changed during snapshot: {filename!r}")
        return owner, root
    except Exception:
        owner.cleanup()
        raise


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


def _safe_bundle_member(assets: Path, filename: str) -> Path:
    """Resolve one portable bundle member without allowing traversal or symlinks."""
    if (not filename or len(filename.encode("utf-8")) > MAX_ARTIFACT_MEMBER_PATH_BYTES
            or "\\" in filename or ":" in filename):
        raise ValueError(f"Invalid bundle member path: {filename!r}")
    components = filename.split("/")
    if any(part in {"", ".", ".."} for part in components):
        raise ValueError(f"Invalid bundle member path: {filename!r}")
    for part in components:
        if (any(ord(character) < 32 or character in '<>:"|?*' for character in part)
                or part.endswith((".", " "))):
            raise ValueError(f"Invalid bundle member path: {filename!r}")
        device_stem = part.split(".", 1)[0].upper().rstrip(" .")
        if (device_stem in {"CON", "PRN", "AUX", "NUL"}
                or re.fullmatch(r"(?:COM|LPT)(?:[1-9]|¹|²|³)", device_stem)):
            raise ValueError(f"Invalid bundle member path: {filename!r}")
    relative = PurePosixPath(filename)
    if relative.is_absolute() or relative.parts != tuple(components):
        raise ValueError(f"Invalid bundle member path: {filename!r}")
    root = assets.resolve()
    candidate = root.joinpath(*relative.parts)
    current = root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError(f"Bundle member path contains a symlink: {filename!r}")
    try:
        candidate.resolve(strict=False).relative_to(root)
    except ValueError as error:
        raise ValueError(f"Bundle member escapes its root: {filename!r}") from error
    return candidate


def _verify_artifact_bundle_manifest(
        json_path: Path, graph_bytes: bytes) -> Optional[dict[str, _BundleMemberIdentity]]:
    """Check graph and immutable support hashes before loading the measurement overlay."""
    manifest_path = assets_dir_for(json_path) / "artifact_bundle_manifest.json"
    assets = assets_dir_for(json_path)
    if assets.is_symlink() or (assets.exists() and not assets.is_dir()):
        raise ValueError(f"Artifact support path is not a regular directory: {assets}")
    if manifest_path.is_symlink():
        raise ValueError(f"Artifact bundle manifest is not a regular file: {manifest_path}")
    if not manifest_path.exists():
        return None
    if not manifest_path.is_file():
        raise ValueError(f"Artifact bundle manifest is not a regular file: {manifest_path}")
    try:
        manifest_bytes = _read_bounded_bytes(
            manifest_path, MAX_ARTIFACT_MANIFEST_BYTES, "Artifact bundle manifest")
        manifest = json.loads(manifest_bytes)
    except (OSError, ValueError) as error:
        raise ValueError(f"Invalid artifact bundle manifest: {error}") from error
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
        raise ValueError("Unsupported artifact bundle manifest schema")
    if (manifest.get("producer") != "roomeq-workflow"
            or not manifest.get("producer_version")
            or not manifest.get("graph_schema_version")
            or not manifest.get("generation")):
        raise ValueError("Artifact bundle manifest has invalid provenance metadata")
    graph_hash = manifest.get("graph_sha256")
    if (not isinstance(graph_hash, str)
            or not re.fullmatch(r"[0-9a-f]{64}", graph_hash)
            or graph_hash != hashlib.sha256(graph_bytes).hexdigest()):
        raise ValueError("Native output graph failed artifact bundle integrity validation")
    entries = manifest.get("files")
    if not isinstance(entries, dict) or len(entries) > MAX_ARTIFACT_MANIFEST_FILES:
        raise ValueError("Artifact bundle manifest has no file map")
    verified: dict[str, _BundleMemberIdentity] = {}
    folded_names: set[str] = set()
    for filename, expected in entries.items():
        if not isinstance(filename, str) or not isinstance(expected, dict):
            raise ValueError("Artifact bundle manifest has an invalid file entry")
        if filename.casefold() in folded_names:
            raise ValueError(f"Artifact bundle manifest has a case-insensitive duplicate: {filename!r}")
        folded_names.add(filename.casefold())
        expected_hash = expected.get("sha256")
        expected_size = expected.get("size_bytes")
        if (not isinstance(expected_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", expected_hash)
                or not isinstance(expected_size, int) or isinstance(expected_size, bool)
                or expected_size < 0):
            raise ValueError(f"Artifact bundle manifest has an invalid file digest: {filename!r}")
        if (filename == "measurements_index.json"
                and expected_size > MAX_ARTIFACT_MANIFEST_BYTES):
            raise ValueError("Measurement index exceeds its size limit")
        path = _safe_bundle_member(assets_dir_for(json_path), filename)
        try:
            if expected_size > MAX_ARTIFACT_MEMBER_BYTES:
                raise ValueError(f"Artifact bundle member exceeds its size limit: {filename!r}")
            if path.stat().st_size != expected_size:
                raise ValueError(f"Artifact bundle member failed integrity validation: {filename}")
            digest = hashlib.sha256()
            actual_size = 0
            with path.open("rb") as member:
                while actual_size <= expected_size:
                    chunk = member.read(min(64 * 1024, expected_size - actual_size + 1))
                    if not chunk:
                        break
                    actual_size += len(chunk)
                    if actual_size > expected_size:
                        break
                    digest.update(chunk)
        except OSError as error:
            raise ValueError(f"Missing artifact bundle member {filename!r}: {error}") from error
        if actual_size != expected_size or digest.hexdigest() != expected_hash:
            raise ValueError(f"Artifact bundle member failed integrity validation: {filename}")
        verified[filename] = _BundleMemberIdentity(expected_size, expected_hash)
    return verified


def _convolution_plugins(payload: dict):
    """Yield native convolution stages without skipping malformed plugin lists."""
    global_plugins = payload.get("global_plugins", [])
    if not isinstance(global_plugins, list):
        raise ValueError("Native graph global_plugins must be a list")
    yield from global_plugins

    channels = payload.get("channels", {})
    if not isinstance(channels, dict):
        raise ValueError("Native graph channels must be an object")
    for name, channel in channels.items():
        if not isinstance(channel, dict):
            raise ValueError(f"Native graph channel {name!r} must be an object")
        plugins = channel.get("plugins", [])
        if not isinstance(plugins, list):
            raise ValueError(f"Native graph channel {name!r} plugins must be a list")
        yield from plugins
        drivers = channel.get("drivers") or []
        if not isinstance(drivers, list):
            raise ValueError(f"Native graph channel {name!r} drivers must be a list")
        for index, driver in enumerate(drivers):
            if not isinstance(driver, dict):
                raise ValueError(f"Native graph channel {name!r} driver {index} must be an object")
            plugins = driver.get("plugins", [])
            if not isinstance(plugins, list):
                raise ValueError(
                    f"Native graph channel {name!r} driver {index} plugins must be a list")
            yield from plugins


def _verify_manifested_convolution_resources(
        payload: dict, assets: Path,
        verified_members: Optional[dict[str, _BundleMemberIdentity]]) -> None:
    """Require every manifested graph FIR to be a hash-bound bundle member.

    Unmanifested legacy graphs retain their historical ability to refer to
    external FIR paths. Once a bundle manifest exists, all convolution stages
    must use portable relative member paths and match the graph hash inventory.
    """
    if verified_members is None:
        return

    references = set()
    for plugin in _convolution_plugins(payload):
        if not isinstance(plugin, dict):
            raise ValueError("Native graph plugin entries must be objects")
        if plugin.get("plugin_type") != "convolution":
            continue
        parameters = plugin.get("parameters")
        if not isinstance(parameters, dict) or not isinstance(parameters.get("ir_file"), str):
            raise ValueError("Manifested convolution stage requires string field 'ir_file'")
        reference = parameters["ir_file"]
        _safe_bundle_member(assets, reference)
        references.add(reference)

    metadata = payload.get("metadata")
    inventory = metadata.get("final_convolution_sha256") if isinstance(metadata, dict) else None
    if not references:
        if inventory not in (None, {}):
            raise ValueError("Manifested convolution inventory contains unreferenced resources")
        return
    if not isinstance(inventory, dict) or set(inventory) != references:
        raise ValueError(
            "Manifested convolution inventory does not exactly match graph references")
    for reference in references:
        expected = inventory.get(reference)
        if not isinstance(expected, str) or len(expected) != 64:
            raise ValueError(
                f"Manifested convolution reference {reference!r} has no SHA-256 binding")
        actual = verified_members.get(reference)
        if actual is None:
            raise ValueError(
                f"Manifested graph convolution reference is not a bundled file: {reference!r}")
        if expected != actual.sha256:
            raise ValueError(
                f"Manifested convolution resource {reference!r} does not match its SHA-256 binding")


def resolve_asset(json_path: Path, reference: str) -> Path:
    """Resolve a bare sidecar/curve reference against candidate directories."""
    candidate = Path(reference)
    if candidate.is_absolute() or PureWindowsPath(reference).is_absolute():
        return candidate
    if "\\" in reference or ":" in reference or ".." in PurePosixPath(reference).parts:
        raise ValueError(f"Invalid relative asset path: {reference!r}")
    for directory in candidate_asset_dirs(json_path):
        resolved = directory / candidate
        if resolved.is_file():
            return resolved
    return json_path.parent / candidate


def read_fir_wav(filepath: Union[Path, bytes]) -> tuple[int, list[float]]:
    """Read mono IEEE-float FIR sidecars, including WAVE_FORMAT_EXTENSIBLE."""
    raw = filepath if isinstance(filepath, bytes) else filepath.read_bytes()
    display_path = filepath if isinstance(filepath, Path) else "verified bundle bytes"
    if raw[:4] != b"RIFF" or raw[8:12] != b"WAVE":
        raise ValueError(f"Not a RIFF WAVE FIR: {display_path}")
    fmt = payload = None
    offset = 12
    while offset + 8 <= len(raw):
        kind, size = struct.unpack_from("<4sI", raw, offset)
        start = offset + 8
        if start + size > len(raw):
            raise ValueError(f"Truncated FIR WAV: {display_path}")
        if kind == b"fmt ":
            fmt = raw[start:start + size]
        elif kind == b"data":
            payload = raw[start:start + size]
        offset = start + size + (size & 1)
    if fmt is None or len(fmt) < 16 or payload is None:
        raise ValueError(f"Missing FIR WAV chunks: {display_path}")
    encoding, channels, rate, _, _, bits = struct.unpack_from("<HHIIHH", fmt)
    if encoding == 65534 and len(fmt) >= 40:
        encoding = struct.unpack_from("<H", fmt, 24)[0]
    if encoding != 3 or channels != 1 or bits not in (32, 64) or rate == 0:
        raise ValueError(f"Expected mono float FIR WAV: {display_path}")
    width = bits // 8
    if not payload or len(payload) % width:
        raise ValueError(f"Invalid FIR WAV samples: {display_path}")
    taps = [value[0] for value in struct.iter_unpack("<f" if bits == 32 else "<d", payload)]
    if not all(math.isfinite(v) for v in taps):
        raise ValueError(f"Non-finite FIR WAV: {display_path}")
    return rate, taps


def _open_curve_text(path_or_bytes):
    if isinstance(path_or_bytes, bytes):
        return io.StringIO(path_or_bytes.decode("utf-8"), newline="")
    return path_or_bytes.open("r", newline="")


def read_curve_csv(path: Union[Path, bytes]) -> dict:
    """Read a measurement curve CSV written by the output bundle."""
    freq: list[float] = []
    spl: list[float] = []
    phase: list[float] = []
    has_phase = False
    with _open_curve_text(path) as handle:
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


def read_ir_csv(path: Union[Path, bytes]) -> dict:
    """Read an impulse-response CSV written by the output bundle."""
    time_ms: list[float] = []
    amplitude: list[float] = []
    with _open_curve_text(path) as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "time_ms" not in reader.fieldnames or "amplitude" not in reader.fieldnames:
            raise ValueError(f"Invalid IR CSV: {path}")
        for row in reader:
            time_ms.append(float(row["time_ms"]))
            amplitude.append(float(row["amplitude"]))
    if len(time_ms) != len(amplitude):
        raise ValueError(f"Mismatched IR arrays: {path}")
    if any(not math.isfinite(value) for value in time_ms + amplitude):
        raise ValueError(f"Non-finite IR values: {path}")
    if any(left >= right for left, right in zip(time_ms, time_ms[1:])):
        raise ValueError(f"IR sample times must be strictly increasing: {path}")
    return {"time_ms": time_ms, "amplitude": amplitude}


_CURVE_FIELDS = ("initial_curve", "final_curve", "eq_response", "target_curve")
_IR_FIELDS = ("pre_ir", "post_ir")
_BLOB_FIELDS = ("early_late_curves", "waterfall", "resonance_decays", "wavelet")


def _validate_curve_data(curve: dict) -> None:
    freq = curve.get("freq")
    spl = curve.get("spl")
    if not isinstance(freq, list) or not isinstance(spl, list) or len(freq) != len(spl):
        raise ValueError("Curve frequency and SPL arrays must be lists with matching lengths")

    def numeric_values(values, field):
        if not isinstance(values, list) or any(
                not isinstance(value, (int, float)) or isinstance(value, bool)
                or not math.isfinite(value) for value in values):
            raise ValueError(f"Curve {field} values must be finite numbers")
        return values

    numeric_values(freq, "frequency")
    numeric_values(spl, "SPL")
    if any(value <= 0 for value in freq):
        raise ValueError("Curve frequencies must be positive")
    if any(left >= right for left, right in zip(freq, freq[1:])):
        raise ValueError("Curve frequencies must be strictly increasing")

    for field in ("phase", "noise_floor_db", "coherence"):
        if field not in curve or curve[field] is None:
            continue
        values = numeric_values(curve[field], field)
        if len(values) != len(freq):
            raise ValueError(f"Curve {field} length does not match frequency data")
        if field == "coherence" and any(value < 0 or value > 1 for value in values):
            raise ValueError("Curve coherence values must be between zero and one")

    if "norm_range" in curve and curve["norm_range"] is not None:
        norm_range = curve["norm_range"]
        if (not isinstance(norm_range, (list, tuple)) or len(norm_range) != 2
                or any(not isinstance(value, (int, float)) or isinstance(value, bool)
                       or not math.isfinite(value) for value in norm_range)):
            raise ValueError("Curve norm_range must contain two finite frequencies")
        low, high = norm_range
        if low <= 0 or low >= high:
            raise ValueError("Curve norm_range must be positive and ordered")


def _read_indexed_curve(assets: Path, csv_filename: str, metadata_filename=None,
                        verified_members: Optional[dict[str, _BundleMemberIdentity]] = None) -> dict:
    if metadata_filename is not None:
        if not isinstance(metadata_filename, str):
            raise ValueError(f"Invalid curve metadata member: {metadata_filename!r}")
        metadata_path = _safe_bundle_member(assets, metadata_filename)
        if verified_members is not None and metadata_filename not in verified_members:
            raise ValueError(f"Measurement index references unhashed member: {metadata_filename!r}")
        if not metadata_path.is_file():
            if verified_members is not None:
                raise ValueError(f"Missing measurement member: {metadata_filename!r}")
        else:
            metadata_bytes = (
                _read_verified_bundle_member_bytes(assets, metadata_filename, verified_members)
                if verified_members is not None else metadata_path.read_bytes())
            curve = json.loads(metadata_bytes)
            if not isinstance(curve, dict):
                raise ValueError(f"Invalid curve metadata member: {metadata_filename!r}")
            _validate_curve_data(curve)
            return curve
    csv_path = _safe_bundle_member(assets, csv_filename)
    if verified_members is not None and csv_filename not in verified_members:
        raise ValueError(f"Measurement index references unhashed member: {csv_filename!r}")
    if not csv_path.is_file():
        if verified_members is not None:
            raise ValueError(f"Missing measurement member: {csv_filename!r}")
        raise FileNotFoundError(csv_path)
    curve_bytes = (
        _read_verified_bundle_member_bytes(assets, csv_filename, verified_members)
        if verified_members is not None else None)
    curve = read_curve_csv(curve_bytes if curve_bytes is not None else csv_path)
    if verified_members is not None:
        _validate_curve_data(curve)
    return curve


def _apply_measurement_overlay(
        data: dict, json_path: Path,
        verified_members: Optional[dict[str, _BundleMemberIdentity]] = None,
        assets_override: Optional[Path] = None) -> dict:
    """Re-inject external measurement files into an in-memory copy.

    Returns the overlay index `{channel: {field: filename}}` plus a
    `deployed_source_curves` entry. Missing files are skipped so legacy
    outputs (fully embedded) and slim outputs (fully external) both load.
    """
    overlay: dict = {}
    assets = assets_override or assets_dir_for(json_path)
    index_path = assets / "measurements_index.json"
    index: dict = {}
    if index_path.is_file():
        try:
            if verified_members is not None:
                index_bytes = _read_verified_bundle_member_bytes(
                    assets, "measurements_index.json", verified_members,
                    maximum_bytes=MAX_ARTIFACT_MANIFEST_BYTES)
            else:
                index_bytes = index_path.read_bytes()
            index = json.loads(index_bytes)
        except (ValueError, OSError):
            if verified_members is not None:
                raise
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
            path = _safe_bundle_member(assets, filename)
            if verified_members is not None and filename not in verified_members:
                raise ValueError(f"Measurement index references unhashed member: {filename!r}")
            if not path.is_file():
                if verified_members is not None:
                    raise ValueError(f"Missing measurement member: {filename!r}")
                continue
            try:
                if field in _IR_FIELDS:
                    ir_bytes = (
                        _read_verified_bundle_member_bytes(assets, filename, verified_members)
                        if verified_members is not None else None)
                    channel[field] = read_ir_csv(ir_bytes if ir_bytes is not None else path)
                else:
                    metadata_filename = entry.get(f"{field}_metadata")
                    if metadata_filename is not None and not isinstance(metadata_filename, str):
                        if verified_members is not None:
                            raise ValueError(f"Invalid curve metadata member: {metadata_filename!r}")
                        metadata_filename = None
                    channel[field] = _read_indexed_curve(
                        assets, filename, metadata_filename, verified_members)
            except (ValueError, OSError):
                if verified_members is not None:
                    raise
                continue
            overlay.setdefault(name, {})[field] = filename
        for field in _BLOB_FIELDS:
            if channel.get(field) is not None:
                continue
            filename = entry.get(field)
            if not isinstance(filename, str):
                continue
            path = _safe_bundle_member(assets, filename)
            if verified_members is not None and filename not in verified_members:
                raise ValueError(f"Measurement index references unhashed member: {filename!r}")
            if not path.is_file():
                if verified_members is not None:
                    raise ValueError(f"Missing measurement member: {filename!r}")
                continue
            try:
                blob_bytes = (
                    _read_verified_bundle_member_bytes(assets, filename, verified_members)
                    if verified_members is not None else path.read_bytes())
                channel[field] = json.loads(blob_bytes)
            except (ValueError, OSError):
                if verified_members is not None:
                    raise
                continue
            overlay.setdefault(name, {})[field] = filename
        drivers = channel.get("drivers") or []
        for driver_index, driver in enumerate(drivers):
            if not isinstance(driver, dict) or driver.get("measured_acoustics") is not None:
                continue
            for kind, filename in entry.items():
                if (kind.startswith(f"driver{driver_index}_")
                        and kind.endswith("_measured_acoustics") and isinstance(filename, str)):
                    path = _safe_bundle_member(assets, filename)
                    if verified_members is not None and filename not in verified_members:
                        raise ValueError(f"Measurement index references unhashed member: {filename!r}")
                    try:
                        blob_bytes = (
                            _read_verified_bundle_member_bytes(assets, filename, verified_members)
                            if verified_members is not None else path.read_bytes())
                        driver["measured_acoustics"] = json.loads(blob_bytes)
                        overlay.setdefault(name, {})[kind] = filename
                    except (ValueError, OSError):
                        if verified_members is not None:
                            raise
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
            path = _safe_bundle_member(assets, filename)
            if verified_members is not None and filename not in verified_members:
                raise ValueError(f"Measurement index references unhashed member: {filename!r}")
            if not path.is_file():
                if verified_members is not None:
                    raise ValueError(f"Missing measurement member: {filename!r}")
                continue
            try:
                metadata_filename = entry.get(f"{kind}_metadata")
                if metadata_filename is not None and not isinstance(metadata_filename, str):
                    if verified_members is not None:
                        raise ValueError(f"Invalid curve metadata member: {metadata_filename!r}")
                    metadata_filename = None
                driver["initial_curve"] = _read_indexed_curve(
                    assets, filename, metadata_filename, verified_members)
            except (ValueError, OSError):
                if verified_members is not None:
                    raise
                continue
            overlay.setdefault(name, {})[kind] = filename

    deployed = data.get("deployed_source_curves")
    if deployed is None:
        # Rust omits the key when empty (skip_serializing_if), so only
        # install it when at least one external curve will actually load.
        # An untracked empty dict would break payload verification, which
        # hashes the slim saved bytes.
        loadable = isinstance(deployed_index, dict) and any(
            isinstance(filename, str)
            and _safe_bundle_member(assets, filename).is_file()
            and (verified_members is None or filename in verified_members)
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
            path = _safe_bundle_member(assets, filename)
            if verified_members is not None and filename not in verified_members:
                raise ValueError(f"Measurement index references unhashed member: {filename!r}")
            if not path.is_file():
                if verified_members is not None:
                    raise ValueError(f"Missing measurement member: {filename!r}")
                continue
            metadata_index = index.get("deployed_source_curve_metadata")
            metadata_filename = metadata_index.get(name) if isinstance(metadata_index, dict) else None
            if metadata_filename is not None and not isinstance(metadata_filename, str):
                if verified_members is not None:
                    raise ValueError(f"Invalid deployed curve metadata member: {metadata_filename!r}")
                metadata_filename = None
            try:
                deployed[name] = _read_indexed_curve(
                    assets, filename, metadata_filename, verified_members)
            except (ValueError, OSError):
                if verified_members is not None:
                    raise
                continue
            overlay.setdefault("deployed_source_curves", {})[name] = filename
    return overlay


def load_roomeq_json(filepath: Path) -> dict:
    """Load and parse roomeq JSON output file."""
    filepath = Path(filepath)
    journal = filepath.with_name(filepath.name + ".autoeq-transaction.json")
    if journal.exists():
        raise RuntimeError(
            f"RoomEQ bundle publication is incomplete; recover it with the native bundle loader: {journal}"
        )
    export_journal = filepath.with_name(f".{filepath.name}.external-export-journal.json")
    if export_journal.exists():
        raise RuntimeError(
            f"RoomEQ native/export publication is incomplete; recover it with the native bundle loader: {export_journal}"
        )
    if not filepath.exists():
        print(f"Error: File not found: {filepath}")
        sys.exit(1)
    graph_bytes = _read_bounded_bytes(filepath, MAX_NATIVE_GRAPH_BYTES, "Native output graph")
    verified_members = _verify_artifact_bundle_manifest(filepath, graph_bytes)
    payload = json.loads(graph_bytes)
    marker = payload.get("artifact_bundle_schema_version")
    if marker is not None:
        if isinstance(marker, bool) or marker != 1:
            raise ValueError(f"Unsupported artifact bundle schema marker: {marker!r}")
        if verified_members is None:
            raise ValueError("Native output requires a missing artifact bundle manifest")
    assets = assets_dir_for(filepath)
    _verify_manifested_convolution_resources(payload, assets, verified_members)
    snapshot_owner = None
    if verified_members is not None:
        snapshot_owner, assets = _snapshot_verified_bundle_members(assets, verified_members)
    source_directory = assets if assets.is_dir() else filepath.resolve().parent
    data = RoomEqData(payload, source_directory)
    data._bundle_snapshot_owner = snapshot_owner
    data.assets_directory = assets
    # Re-inject slim-output measurement files so plots see the same curves
    # as legacy embedded outputs. Overlay keys are tracked for payload
    # verification, which strips them before hashing.
    try:
        data.measurement_index = _apply_measurement_overlay(
            data, filepath, verified_members, assets_override=assets)
    except (OSError, ValueError):
        if verified_members is not None:
            raise
        data.measurement_index = {}
    # Bind physical sidecars before plots replay a driver's plugin chain.
    # These private in-memory fields are never written back to the DSP JSON.
    for channel in (data.get("channels") or {}).values():
        for driver in channel.get("drivers") or []:
            for plugin in driver.get("plugins") or []:
                parameters = plugin.get("parameters") or {}
                if plugin.get("plugin_type") == "convolution" and parameters.get("room_eq_fir_placement") == "per_driver":
                    if verified_members is not None:
                        fir_bytes = _read_verified_bundle_member_bytes(
                            assets, parameters["ir_file"], verified_members)
                        rate, taps = read_fir_wav(fir_bytes)
                    else:
                        resolved = resolve_asset(filepath, parameters["ir_file"])
                        rate, taps = read_fir_wav(resolved)
                    plugin["parameters"] = FirParameters(parameters, rate, taps)
    return data
