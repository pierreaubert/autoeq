"""Target-curve loading and display alignment for RoomEQ reports."""

from __future__ import annotations

import bisect
import csv
import math
from pathlib import Path

from .dsp import _crossover_response


_FREQUENCY_COLUMNS = ("frequency", "freq", "frequency_hz", "freq_hz")
_SPL_COLUMNS = ("spl", "spl_db", "level", "level_db")


def _column(row: dict[str, str], names: tuple[str, ...]) -> str | None:
    normalized = {str(key).strip().lower(): value for key, value in row.items()}
    for name in names:
        if name in normalized:
            return normalized[name]
    return None


def _configured_target_path(data: dict, json_path: Path | None) -> Path | None:
    effective_config = (data.get("metadata") or {}).get("effective_config") or {}
    configured = effective_config.get("target_curve")
    if isinstance(configured, dict):
        configured = configured.get("path") or configured.get("file")
    if not isinstance(configured, str) or not configured.strip():
        return None

    path = Path(configured)
    if path.is_absolute():
        return path

    candidates = []
    if json_path is not None:
        candidates.append(Path(json_path).parent / path)
    candidates.append(path)
    return next((candidate for candidate in candidates if candidate.exists()), candidates[0])


def load_target_shape(data: dict, json_path: Path | None = None) -> dict | None:
    """Load the file-backed target stored in ``metadata.effective_config``."""
    path = _configured_target_path(data, json_path)
    if path is None:
        return None

    try:
        points: dict[float, float] = {}
        with path.open(newline="", encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                frequency_text = _column(row, _FREQUENCY_COLUMNS)
                spl_text = _column(row, _SPL_COLUMNS)
                if frequency_text is None or spl_text is None:
                    raise ValueError(
                        "target CSV needs frequency/freq and spl/spl_db columns"
                    )
                frequency = float(frequency_text)
                spl = float(spl_text)
                if frequency > 0.0 and math.isfinite(frequency) and math.isfinite(spl):
                    points[frequency] = spl
        if not points:
            raise ValueError("target CSV has no finite positive-frequency points")
    except (OSError, UnicodeError, ValueError) as error:
        print(f"Warning: Could not load target curve '{path}': {error}")
        return None

    frequencies = sorted(points)
    return {"freq": frequencies, "spl": [points[freq] for freq in frequencies]}


def _interpolate_log_space(target: dict, frequencies: list[float]) -> list[float]:
    source_freq = target["freq"]
    source_spl = target["spl"]
    source_log_freq = [math.log10(frequency) for frequency in source_freq]
    result = []

    for frequency in frequencies:
        if frequency <= source_freq[0]:
            result.append(source_spl[0])
            continue
        if frequency >= source_freq[-1]:
            result.append(source_spl[-1])
            continue

        upper = bisect.bisect_right(source_freq, frequency)
        lower = upper - 1
        position = (math.log10(frequency) - source_log_freq[lower]) / (
            source_log_freq[upper] - source_log_freq[lower]
        )
        result.append(
            source_spl[lower] + position * (source_spl[upper] - source_spl[lower])
        )

    return result


def align_target_to_curve(
    target: dict,
    reference: dict,
    min_freq: float = 20.0,
    max_freq: float = 20_000.0,
) -> dict | None:
    """Interpolate a relative target and align its level to a displayed curve.

    The relative-to-peak guard excludes crossover stopbands from level alignment.
    This is especially important for the LFE route, whose post-DSP response is
    intentionally low-passed.
    """
    frequencies = list(reference.get("freq") or [])
    reference_spl = list(reference.get("spl") or [])
    if not frequencies or len(frequencies) != len(reference_spl):
        return None

    target_spl = _interpolate_log_space(target, frequencies)
    band_levels = [
        level
        for frequency, level in zip(frequencies, reference_spl, strict=True)
        if min_freq <= frequency <= max_freq and math.isfinite(level)
    ]
    if not band_levels:
        return None
    passband_floor = max(band_levels) - 30.0

    offsets = [
        measured - desired
        for frequency, measured, desired in zip(
            frequencies, reference_spl, target_spl, strict=True
        )
        if min_freq <= frequency <= max_freq
        and math.isfinite(measured)
        and measured >= passband_floor
    ]
    if not offsets:
        return None

    offset = sum(offsets) / len(offsets)
    return {
        "freq": frequencies,
        "spl": [level + offset for level in target_spl],
    }


def _target_shape_for_channel(
    data: dict,
    channel_name: str,
    target: dict,
    reference: dict,
) -> dict:
    """Apply programme-route band limiting to a logical LFE target."""
    bass_management = ((data.get("metadata") or {}).get("bass_management") or {})
    graph = bass_management.get("routing_graph") or {}
    route = next(
        (
            candidate
            for candidate in graph.get("routes", [])
            if candidate.get("source_channel") == channel_name
            and candidate.get("route_kind") == "lfe_lowpass_to_sub"
        ),
        None,
    )
    if route is None:
        return target

    cutoff_hz = route.get("low_pass_hz")
    frequencies = list(reference.get("freq") or [])
    if not isinstance(cutoff_hz, (int, float)) or cutoff_hz <= 0.0 or not frequencies:
        return target

    effective_config = (data.get("metadata") or {}).get("effective_config") or {}
    sample_rate = float(effective_config.get("sample_rate", 48_000.0))
    target_spl = _interpolate_log_space(target, frequencies)
    response = _crossover_response(
        str(route.get("crossover_type") or "LR24"),
        "low",
        float(cutoff_hz),
        frequencies,
        sample_rate,
    )
    return {
        "freq": frequencies,
        "spl": [
            level + 20.0 * math.log10(max(abs(transfer), 1.0e-10))
            for level, transfer in zip(target_spl, response, strict=True)
        ],
    }


def build_target_overlay_curves(
    data: dict,
    reference_curves: dict[str, dict],
    json_path: Path | None = None,
) -> dict[str, dict]:
    """Build one level-aligned target overlay for each displayed channel curve."""
    target = load_target_shape(data, json_path)
    if target is None:
        return {}

    optimizer = (
        ((data.get("metadata") or {}).get("effective_config") or {}).get("optimizer")
        or {}
    )
    min_freq = float(optimizer.get("min_freq", 20.0))
    max_freq = float(optimizer.get("max_freq", 20_000.0))

    result = {}
    for channel_name, reference in reference_curves.items():
        channel_target = _target_shape_for_channel(data, channel_name, target, reference)
        aligned = align_target_to_curve(channel_target, reference, min_freq, max_freq)
        if aligned is not None:
            result[channel_name] = aligned
    return result
