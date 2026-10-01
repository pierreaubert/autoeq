"""Target-curve loading and display alignment for RoomEQ reports."""

from __future__ import annotations

import bisect
import csv
import math
from pathlib import Path

from .dsp import _crossover_response


_FREQUENCY_COLUMNS = ("frequency", "freq", "frequency_hz", "freq_hz")
_SPL_COLUMNS = ("spl", "spl_db", "level", "level_db")

#: Midband (Hz) over which a design-target level is matched to measured data.
#: Reports anchor every channel tab's target to the per-channel mean of the
#: measured L/R pair over this band so one global calibration offset applies.
TARGET_LEVEL_MATCH_BAND_HZ = (100.0, 10_000.0)


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


def band_mean_spl(
    curve: dict,
    min_freq: float,
    max_freq: float,
) -> float | None:
    """Arithmetic mean of finite SPL samples inside ``[min_freq, max_freq]``."""
    frequencies = curve.get("freq") or []
    levels = curve.get("spl") or []
    if len(frequencies) != len(levels):
        return None
    values = [
        level
        for frequency, level in zip(frequencies, levels, strict=True)
        if min_freq <= frequency <= max_freq and math.isfinite(level)
    ]
    if not values:
        return None
    return sum(values) / len(values)


def shift_target_to_reference_band_mean(
    target: dict,
    reference: dict,
    min_freq: float = TARGET_LEVEL_MATCH_BAND_HZ[0],
    max_freq: float = TARGET_LEVEL_MATCH_BAND_HZ[1],
) -> dict | None:
    """Interpolate ``target`` onto ``reference`` and match band means.

    Unlike :func:`align_target_to_curve` (per-channel optimizer-band fit with
    a relative-to-peak stopband guard), this applies the report-level rule:
    the target mean over ``[min_freq, max_freq]`` equals the reference mean
    over the same band. Returns ``None`` when either band is empty.
    """
    frequencies = list(reference.get("freq") or [])
    reference_spl = list(reference.get("spl") or [])
    if not frequencies or len(frequencies) != len(reference_spl):
        return None
    if not target.get("freq") or not target.get("spl"):
        return None
    target_spl = _interpolate_log_space(target, frequencies)
    reference_mean = band_mean_spl(
        {"freq": frequencies, "spl": reference_spl}, min_freq, max_freq
    )
    target_mean = band_mean_spl(
        {"freq": frequencies, "spl": target_spl}, min_freq, max_freq
    )
    if reference_mean is None or target_mean is None:
        return None
    offset = reference_mean - target_mean
    return {
        "freq": frequencies,
        "spl": [level + offset for level in target_spl],
    }


def global_target_offset_for_pair(
    reference_l: dict | None,
    reference_r: dict | None,
    shape: dict | None,
    min_freq: float = TARGET_LEVEL_MATCH_BAND_HZ[0],
    max_freq: float = TARGET_LEVEL_MATCH_BAND_HZ[1],
) -> float | None:
    """Single offset anchoring a target shape to measured per-channel level.

    The anchor is the mean of the L and R reference band means over
    ``[min_freq, max_freq]``: with no L/R pair (or an empty band) there is
    no anchor (``None``) and targets render as serialized. The one offset
    applies to every channel tab, preserving designed inter-channel target
    differences. Unlike an L+R *sum* anchor, per-channel overlays land at
    the level of the per-channel data instead of 3-6 dB above it.
    """
    if not isinstance(shape, dict) or not shape.get("freq") or not shape.get("spl"):
        return None
    if not isinstance(reference_l, dict) or not isinstance(reference_r, dict):
        return None
    mean_l = band_mean_spl(reference_l, min_freq, max_freq)
    mean_r = band_mean_spl(reference_r, min_freq, max_freq)
    if mean_l is None or mean_r is None:
        return None
    grid = list(reference_l.get("freq") or []) or list(reference_r.get("freq") or [])
    if not grid:
        return None
    shape_mean = band_mean_spl(
        {"freq": grid, "spl": _interpolate_log_space(shape, grid)},
        min_freq, max_freq,
    )
    if shape_mean is None:
        return None
    return (mean_l + mean_r) / 2.0 - shape_mean


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


def _target_alignment_band_for_channel(
    data: dict,
    channel_name: str,
    min_freq: float,
    max_freq: float,
) -> tuple[float, float]:
    """Match the routed optimizer's main-only target reference band."""
    bass_management = ((data.get("metadata") or {}).get("bass_management") or {})
    graph = bass_management.get("routing_graph") or {}
    route = next(
        (
            candidate
            for candidate in graph.get("routes", []) or []
            if candidate.get("source_channel") == channel_name
            and candidate.get("route_kind") == "redirected_bass_lowpass_to_sub"
            and candidate.get("low_pass_hz") is not None
        ),
        None,
    )
    if route is None:
        return min_freq, max_freq
    crossover_hz = float(route["low_pass_hz"])
    reference_min = max(min_freq, crossover_hz * 2.0)
    reference_max = min(max_freq, crossover_hz * 8.0, 2_000.0)
    if reference_min >= reference_max:
        return min_freq, max_freq
    return reference_min, reference_max


def _global_anchor_offset(
    data: dict, reference_curves: dict[str, dict]
) -> float | None:
    """Mean-based offset for absolute targets; None without an L/R anchor."""
    channels = data.get("channels") or {}
    shape = (channels.get("L") or {}).get("target_curve") or (
        channels.get("R") or {}
    ).get("target_curve")
    return global_target_offset_for_pair(
        reference_curves.get("L"), reference_curves.get("R"), shape
    )


def build_target_overlay_curves(
    data: dict,
    reference_curves: dict[str, dict],
    json_path: Path | None = None,
) -> dict[str, dict]:
    """Absolute targets share one global level anchor; legacy targets align per channel."""
    target = load_target_shape(data, json_path)

    optimizer = (
        ((data.get("metadata") or {}).get("effective_config") or {}).get("optimizer")
        or {}
    )
    min_freq = float(optimizer.get("min_freq", 20.0))
    max_freq = float(optimizer.get("max_freq", 20_000.0))

    # Serialized absolute targets carry the design shape but not the measured
    # level: shift every channel's overlay by the single L/R-anchored offset.
    anchor = _global_anchor_offset(data, reference_curves)

    result = {}
    for channel_name, reference in reference_curves.items():
        absolute_target = ((data.get("channels") or {}).get(channel_name) or {}).get("target_curve")
        if absolute_target and absolute_target.get("freq") and absolute_target.get("spl"):
            frequencies = list(reference.get("freq") or [])
            if frequencies:
                spl = _interpolate_log_space(absolute_target, frequencies)
                if anchor is not None:
                    spl = [level + anchor for level in spl]
                result[channel_name] = {"freq": frequencies, "spl": spl}
            continue
        if target is None:
            continue
        channel_target = _target_shape_for_channel(data, channel_name, target, reference)
        alignment_min, alignment_max = _target_alignment_band_for_channel(
            data, channel_name, min_freq, max_freq
        )
        aligned = align_target_to_curve(
            channel_target, reference, alignment_min, alignment_max
        )
        if aligned is not None:
            result[channel_name] = aligned
    return result
