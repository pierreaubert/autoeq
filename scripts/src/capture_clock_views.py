"""Clock evidence and fail-closed coherent rendering for capture manifests."""

from html import escape
import math
from typing import TypeGuard

COHERENT_LIMIT_US = 50.0


def _finite(value) -> TypeGuard[float | int]:
    return isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value)


def _known(value) -> TypeGuard[str]:
    return isinstance(value, str) and bool(value.strip()) and value.strip().lower() != "unknown"


def _mapping(value) -> dict:
    return value if isinstance(value, dict) else {}


def _configuration(data):
    if isinstance(data.get("speakers"), dict):
        return data
    metadata = _mapping(data.get("metadata"))
    return _mapping(metadata.get("effective_config"))


def _source(config, channel):
    mapping = _mapping(_mapping(config.get("system")).get("speakers"))
    key = mapping.get(channel, channel)
    return _mapping(config.get("speakers")).get(key) if isinstance(key, str) else None


def _capture_reason(source):
    if not isinstance(source, dict):
        return "capture provenance is unavailable", None
    provenance = _mapping(source.get("provenance"))
    capture = provenance.get("capture")
    if not isinstance(capture, dict):
        return "per-microphone clock bounds are unavailable", None
    measurements, takes = source.get("measurements"), capture.get("takes")
    if (
        not isinstance(measurements, list)
        or not isinstance(takes, list)
        or not 2 <= len(takes) <= 4
        or len(takes) != len(measurements)
    ):
        return "clock evidence does not identify every microphone", None
    geometry = capture.get("geometry")
    if geometry not in ("spread", "compact"):
        return "session geometry is unavailable", None
    ids, reference = set(), None
    for take in takes:
        if not isinstance(take, dict):
            return "invalid microphone clock record", None
        mic = take.get("microphone_id")
        prefix = f"{mic}: "
        if (
            not _known(mic) or mic in ids
            or not _known(take.get("device_id"))
            or not _known(take.get("calibration_id"))
        ):
            return prefix + "microphone, device, or calibration identity is invalid", None
        ids.add(mic)
        position = take.get("position_m")
        uncertainty = take.get("position_uncertainty_mm")
        if (
            not isinstance(position, list) or len(position) != 3
            or not all(_finite(x) for x in position)
            or not _finite(uncertainty) or uncertainty < 0
            or (geometry == "compact" and uncertainty > 1)
        ):
            return prefix + "microphone geometry is invalid", None
        if (
            not _finite(take.get("gain_db"))
            or take.get("calibration_orientation") not in ("on_axis", "ninety_degrees")
        ):
            return prefix + "gain or calibration orientation is invalid", None
        if take.get("correction_applied") != "resampled" or take.get("preserves_acoustic_delay") is not True:
            return prefix + "common-clock acoustic-delay evidence is unavailable", None
        if take.get("quality_passed") is not True:
            return prefix + "measurement quality is pending or failed", None
        skew = take.get("skew_ppm")
        if not _finite(take.get("offset_samples")) or not _finite(skew) or abs(skew) > 5000:
            return prefix + "clock fit is invalid", None
        bound = take.get("residual_uncertainty_us")
        if not _finite(bound) or not 0 <= bound < COHERENT_LIMIT_US:
            return prefix + "residual bound is unavailable or not below 50 µs", None
        declared = take.get("timing_reference_id")
        if not _known(declared) or (reference is not None and reference != declared):
            return prefix + "microphones do not share a timing reference", None
        reference = declared
    if reference != provenance.get("timing_reference_id"):
        return "source and microphone timing references disagree", None
    if provenance.get("capture_kind") not in ("stationary_ir", "direct_sound"):
        return "source contains magnitude-only acquisition evidence", None
    return None, reference


def coherent_clock_reason(data, channels):
    """Return a fallback reason unless every involved source passes its clock gate."""
    config = _configuration(data)
    reference, layout = None, None
    for channel in channels:
        source = _source(config, channel)
        if not isinstance(source, dict):
            return f"{channel}: capture provenance is unavailable"
        reason, current = _capture_reason(source)
        if reason:
            return f"{channel}: {reason}"
        capture = source["provenance"]["capture"]
        current_layout = (
            capture["geometry"],
            [(take["microphone_id"], take["device_id"], take["position_m"])
             for take in capture["takes"]],
        )
        if reference is not None and (reference != current or layout != current_layout):
            return "sources do not share the same timing reference and microphone layout"
        reference, layout = current, current_layout
    return None if reference is not None else "no capture sources are available"


def gated_lr_channel(data, left, right):
    """Synthesize a coherent pair only with timing evidence and finite measured phase."""
    from .dsp import synthesize_lr_channel

    reason = coherent_clock_reason(data, ("L", "R"))
    if reason is None:
        for key in ("initial_curve", "final_curve"):
            for channel in (left, right):
                curve = (channel or {}).get(key)
                if curve and (
                    not isinstance(curve.get("phase"), list)
                    or len(curve["phase"]) != len(curve.get("freq") or [])
                    or not all(_finite(x) for x in curve["phase"])
                ):
                    reason = "finite measured phase is unavailable"
    if reason is None:
        frequencies = [value for channel in (left, right)
                       for key in ("initial_curve", "final_curve")
                       for value in ((channel or {}).get(key) or {}).get("freq", [])]
        if not frequencies or any(not _finite(value) or value <= 0 for value in frequencies):
            reason = "finite positive frequency grid unavailable"
        else:
            config = _configuration(data)
            # Worst-case relative phase error adds the two source clock bounds.
            relative_us = sum(max(take["residual_uncertainty_us"]
                                  for take in _mapping(_source(config, name))["provenance"]["capture"]["takes"])
                              for name in ("L", "R"))
            if relative_us <= 0 or max(frequencies) > 25_000.0 / relative_us:
                reason = "relative clock uncertainty exceeds nine degrees over the plotted band"
    if reason:
        def magnitude_only(channel):
            if not channel:
                return None
            return {key: {"freq": curve.get("freq"), "spl": curve.get("spl")}
                    for key in ("initial_curve", "final_curve")
                    if (curve := channel.get(key))}
        left, right = magnitude_only(left), magnitude_only(right)
    return synthesize_lr_channel(left, right), reason


def capture_clock_qa_html(data):
    """Render original clock declarations, including failed fits and pending quality."""
    rows = []
    for source_name, source in _mapping(_configuration(data).get("speakers")).items():
        if not isinstance(source, dict):
            continue
        capture = _mapping(source.get("provenance")).get("capture")
        if not isinstance(capture, dict):
            continue
        reason, _ = _capture_reason(source)
        takes = capture.get("takes")
        for take in takes if isinstance(takes, list) and takes else [{}]:
            if not isinstance(take, dict):
                continue
            def number(name):
                value = take.get(name)
                return f"{value:.3f}" if _finite(value) else "unavailable"
            values = [
                source_name, capture.get("geometry", "unknown"),
                take.get("microphone_id", "unknown"), take.get("device_id", "unknown"),
                number("offset_samples"), number("skew_ppm"),
                number("residual_uncertainty_us"), take.get("correction_applied", "unknown"),
                reason or "Clock gate passed; phase validity still required",
            ]
            rows.append("<tr>" + "".join(f"<td>{escape(str(value))}</td>" for value in values) + "</tr>")
    if not rows:
        return ""
    headings = ("Source", "Geometry", "Microphone", "Device", "Offset (samples)", "Skew (ppm)", "Residual (µs)", "Correction", "Coherent-use status")
    return ('<section class="capture-clock-qa"><h2>Capture clock QA</h2>'
            '<p>Every participating microphone needs a finite residual bound below 50 µs, '
            'a common reference, and passed measurement quality. Missing evidence uses magnitude only. '
            'Bounds assume linear clocks and correctly isolated timing markers.</p><table class="summary-table"><thead><tr>'
            + "".join(f"<th>{heading}</th>" for heading in headings) + "</tr></thead><tbody>"
            + "".join(rows) + "</tbody></table></section>")
