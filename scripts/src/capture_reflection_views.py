"""Render captured arrival evidence without estimating new directions in the viewer."""

import math
from html import escape

from .capture_clock_views import _capture_reason, _configuration, _finite, _mapping, _source


def direction_reason(source, event):
    """Explain why an arrival cannot be drawn as one conditional direction."""
    reason, _ = _capture_reason(source)
    if reason:
        return reason
    capture = source["provenance"]["capture"]
    if capture.get("geometry") != "compact":
        return "array directions require compact geometry"
    if event.get("mirror_ambiguous") is not False:
        return "planar mirror ambiguity is unresolved"
    direction = event.get("direction")
    if (not isinstance(direction, list) or len(direction) != 3
            or not all(_finite(value) for value in direction)
            or abs(sum(value * value for value in direction) - 1.0) > 1e-3):
        return "finite unit direction unavailable"
    residual = event.get("residual_samples")
    if not _finite(residual) or not 0 <= residual <= 0.5:
        return "plane-wave fit residual unavailable or too large"
    band = event.get("band_hz")
    if (not isinstance(band, list) or len(band) != 2
            or not all(_finite(value) for value in band)
            or not 0 < band[0] < band[1] <= 3000):
        return "supported direction-analysis band unavailable"
    bounds = sorted((take["residual_uncertainty_us"] for take in capture["takes"]), reverse=True)
    relative_us = bounds[0] + bounds[1]
    if relative_us <= 0 or band[1] > 25000 / relative_us:
        return "relative timing uncertainty exceeds the direction-analysis band limit"
    return None


def capture_reflections_html(data):
    """Return per-channel arrival tables and gated azimuth/delay plots."""
    config = _configuration(data)
    channels = _mapping(data.get("channels")) or _mapping(config.get("speakers"))
    sections = []
    for channel in channels:
        source = _source(config, channel)
        capture = _mapping(_mapping(_mapping(source).get("provenance")).get("capture"))
        report = _mapping(capture.get("reflection_report"))
        if not report:
            continue
        rows, points = [], []
        direct = report.get("direct_sound")
        reflections = report.get("early_reflections")
        events = ([direct] if isinstance(direct, dict) else [])
        events += [event for event in (reflections if isinstance(reflections, list) else [])[:64]
                   if isinstance(event, dict)]
        for index, event in enumerate(events):
            relative = event.get("relative_ms")
            level = event.get("level_db")
            reason = direction_reason(source, event)
            expected_source = _mapping(_mapping(config.get("system")).get("speakers")).get(channel, channel)
            if report.get("source_id") != expected_source:
                reason = "arrival source identity does not match this channel"
            if not _finite(relative) or not 0 <= relative <= 80 or not _finite(level):
                reason = "finite early-arrival time/level unavailable"
            label = "Direct candidate" if event is direct else f"Reflection candidate {index + 1 - int(isinstance(direct, dict))}"
            status = reason or "Conditional measured direction"
            if reason is None and _finite(relative) and _finite(level):
                direction = event["direction"]
                horizontal = math.hypot(direction[0], direction[1])
                azimuth = math.atan2(direction[1], direction[0])
                elevation = math.asin(max(-1.0, min(1.0, direction[2])))
                status += f"; azimuth {math.degrees(azimuth):.1f}°, elevation {math.degrees(elevation):.1f}°"
                if horizontal <= 1e-6:
                    status = f"Conditional vertical arrival; azimuth undefined, elevation {math.degrees(elevation):.1f}°"
                if event is not direct and horizontal > 1e-6:
                    radius = relative / 80 * 110
                    x, y = 140 + radius * math.cos(azimuth), 140 - radius * math.sin(azimuth)
                    points.append(f'<circle class="capture-arrival-point" cx="{x:.2f}" cy="{y:.2f}" r="4" fill="#56b4e9">'
                                  f'<title>{escape(label)}: {relative:.2f} ms, {level:.1f} dB</title></circle>')
            issues = event.get("issues")
            if isinstance(issues, list):
                status += "; " + "; ".join(str(issue) for issue in issues)
            time_text = f"{relative:.2f}" if _finite(relative) else "—"
            level_text = f"{level:.1f}" if _finite(level) else "—"
            rows.append(f"<tr><td>{escape(label)}</td><td>{time_text}</td><td>{level_text}</td><td>{escape(status)}</td></tr>")
        plot = ""
        if points:
            rings = "".join(f'<circle cx="140" cy="140" r="{radius}" fill="none" stroke="currentColor" opacity="0.2"/>'
                            for radius in (27.5, 55, 82.5, 110))
            plot = ('<svg viewBox="0 0 280 280" width="280" height="280" role="img" aria-label="Conditional arrival azimuth and delay">'
                    + rings + '<text x="247" y="145">+x</text><text x="135" y="20">+y</text>'
                    + "".join(points) + '</svg><p>Radius: delay from direct candidate, 20 ms per ring. '
                    'Azimuth uses the recorded session axes. Mirror-ambiguous arrivals are omitted.</p>')
        issues = report.get("issues")
        notes = "; ".join(str(issue) for issue in issues) if isinstance(issues, list) else ""
        sections.append(f'<section class="capture-reflections"><h3>{escape(str(channel))}: early-arrival candidates</h3>'
                        f'<p>{escape(notes)}</p>' + plot
                        + '<table class="summary-table"><thead><tr><th>Event</th><th>Relative ms</th><th>Level dB</th><th>Evidence</th></tr></thead><tbody>'
                        + "".join(rows) + '</tbody></table></section>')
    if not sections:
        return ""
    return ('<section><h2>Measured arrival evidence</h2><p>Energy-based arrival candidates and conditional plane-wave directions. '
            'Microphone phase and source distance are not independently verified; fit residuals are not angular uncertainty bounds.</p>'
            + "".join(sections) + '</section>')
