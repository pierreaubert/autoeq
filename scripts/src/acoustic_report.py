"""Viewer-side acoustic helpers for the feat-report sections.

Only operations fully determined by the roomeq output JSON live here:
fractional-octave smoothing (reusing the viewer convention), band means,
IR envelopes, and arrival-time tables. Anything needing new measurement
analysis (RT60, reflection picking, waterfall/wavelet, early/late splits)
is a roomeq/math-audio requirement, not a viewer estimate.
"""

import math
from html import escape

from .dsp import smooth_octave

SPEED_OF_SOUND_M_S = 343.0

# Colour thresholds from reviews/feat-report.md (green, yellow, red).
# Each entry: (green_predicate, yellow_predicate); else red.
SUMMARY_THRESHOLDS = {
    "operational_room_response_pct": {"green": lambda v: v > 90.0,
                                      "yellow": lambda v: v >= 80.0},
    "t60_flatness_pct": {"green": lambda v: v > 80.0,
                          "yellow": lambda v: v >= 50.0},
    "notch_db": {"green": lambda v: v > -10.0, "yellow": lambda v: v >= -20.0},
    "early_late_ratio_db": {"green": lambda v: v > 3.0, "yellow": lambda v: v >= 0.0},
    "early_reflection_level_db": {"green": lambda v: v < -10.0,
                                  "yellow": lambda v: v <= -3.0},
}

SUB_NAMES = {"sub", "lfe", "sub1", "sub2"}
T60_OCTAVE_CENTERS_HZ = (63, 125, 250, 500, 1000, 2000, 4000, 8000, 16000)

# Default ±tolerance (seconds) for the Section 1 T60 flatness share when the
# operator declares none. ITU-R BS.1116-2 §8.2.3.1, Fig. 1 bounds the room
# reverberation time about its mean Tm at ±0.05 s across the 200 Hz–4 kHz
# midband (wider at the edges: +0.3 s leniency at 63 Hz, ±0.1 s at 8 kHz;
# 16 kHz lies above the figure's 8 kHz bound). The viewer applies the midband
# bound uniformly as a documented simplification; declare
# reporting.t60_flatness_tolerance_s to override it.
DEFAULT_T60_FLATNESS_TOLERANCE_S = 0.05


def band_mean(freq, spl, lo_hz, hi_hz):
    """Mean SPL of curve points inside [lo_hz, hi_hz]."""
    points = _valid_curve_points(freq, spl)
    if points is None:
        return None
    vals = [s for f, s in points if lo_hz <= f <= hi_hz]
    if not vals:
        return None
    try:
        mean = math.fsum(vals) / len(vals)
    except OverflowError:
        return None
    return mean if math.isfinite(mean) else None


def _valid_curve_points(freq, spl):
    """Reject malformed curve axes before pairing samples by frequency."""
    if (not isinstance(freq, (list, tuple)) or not isinstance(spl, (list, tuple))
            or not freq or len(freq) != len(spl)):
        return None
    points = []
    previous = 0.0
    for f, s in zip(freq, spl):
        if (isinstance(f, bool) or not isinstance(f, (int, float))
                or not math.isfinite(f) or f <= previous
                or isinstance(s, bool) or not isinstance(s, (int, float))
                or not math.isfinite(s)):
            return None
        points.append((f, s))
        previous = f
    return points


def smooth_curve(curve, octaves=1.0):
    """Return a 1-octave-smoothed copy of a {freq, spl} curve."""
    if not curve or not curve.get("freq") or not curve.get("spl"):
        return None
    return {
        "freq": list(curve["freq"]),
        "spl": smooth_octave(curve["freq"], curve["spl"], octaves),
    }


def is_sub_channel(name):
    return str(name).strip().lower() in SUB_NAMES or str(name).lower().startswith("sub")


def level_compensation(channels):
    """Proposed attenuation to align measured monitor band levels.

    Monitor reference band 0.5-3 kHz, sub band 30-80 Hz. Use the quietest
    monitor as reference so monitor proposals only attenuate. The JSON has no
    independent post-calibration measurement, so residuals remain unknown.
    """
    rows = []
    monitor_means = {}
    for name, ch in channels.items():
        if is_sub_channel(name):
            continue
        curve = (ch or {}).get("initial_curve")
        if not curve or not curve.get("freq"):
            continue
        mean = band_mean(curve["freq"], curve.get("spl"), 500.0, 3000.0)
        if mean is not None:
            monitor_means[name] = mean
    if not monitor_means:
        return []
    reference = min(monitor_means.values())
    for name, ch in channels.items():
        curve = (ch or {}).get("initial_curve")
        if not curve or not curve.get("freq"):
            rows.append({"speaker": str(name), "comp_db": None, "residual_db": None})
            continue
        if is_sub_channel(name):
            mean = band_mean(curve["freq"], curve.get("spl"), 30.0, 80.0)
            comp = (reference - mean) if mean is not None else None
        else:
            mean = monitor_means.get(name)
            comp = reference - mean if mean is not None else None
        rows.append({
            "speaker": str(name),
            "comp_db": comp,
            "residual_db": None,
        })
    return rows


def deepest_notch_db(channel_data):
    """Deepest final-curve notch below 300 Hz vs the 0.5-3 kHz mean."""
    curve = (channel_data or {}).get("final_curve")
    if not curve or not curve.get("freq") or not curve.get("spl"):
        return None
    points = _valid_curve_points(curve["freq"], curve["spl"])
    if points is None:
        return None
    ref = band_mean(curve["freq"], curve["spl"], 500.0, 3000.0)
    lows = [s for f, s in points if f < 300.0]
    if ref is None or not lows:
        return None
    return min(lows) - ref


def _pair_key(name):
    """Normalise a channel name to a symmetric-pair key, or None."""
    n = str(name).strip().upper()
    pairs = {
        "L": "L+R", "R": "L+R",
        "SL": "SL+SR", "SR": "SL+SR",
        "TFL": "TFL+TFR", "TFR": "TFL+TFR",
        "TBL": "TBL+TBR", "TBR": "TBL+TBR",
        "SSL": "SSL+SSR", "SSR": "SSL+SSR",
        "LBL": "LBL+LBR", "LBR": "LBL+LBR",
    }
    return pairs.get(n)


def symmetric_groups(channels):
    """Group channel names into symmetric monitor pairs.

    Returns (groups, unpaired): groups is a list of (label, [a, b]) with
    both members present; unpaired lists the rest (center, subs, solos).
    """
    buckets: dict[str, list[str]] = {}
    order: list[str] = []
    for name in channels:
        key = _pair_key(name)
        if key is None:
            continue
        if key not in buckets:
            buckets[key] = []
            order.append(key)
        buckets[key].append(str(name))
    groups = [(k, buckets[k]) for k in order if len(buckets[k]) == 2]
    grouped = {n for _, ms in groups for n in ms}
    unpaired = [str(n) for n in channels if str(n) not in grouped]
    return groups, unpaired


def pair_sum_difference(curve_a, curve_b):
    """Magnitude-sum and difference SPL curves for a symmetric pair.

    Curves must share the same frequency grid. Never combine values at
    different frequencies by array index.
    """
    if not curve_a or not curve_b:
        return None
    fa, sa = curve_a.get("freq"), curve_a.get("spl")
    fb, sb = curve_b.get("freq"), curve_b.get("spl")
    points_a = _valid_curve_points(fa, sa)
    points_b = _valid_curve_points(fb, sb)
    if (points_a is None or points_b is None or len(points_a) != len(points_b)
            or any(abs(a[0] - b[0]) > 1e-9 * max(1.0, abs(a[0]), abs(b[0]))
                   for a, b in zip(points_a, points_b))):
        return None
    lin_sum, lin_diff = [], []
    for (_, xa), (_, xb) in zip(points_a, points_b):
        try:
            pa = 10.0 ** (xa / 20.0)
            pb = 10.0 ** (xb / 20.0)
        except OverflowError:
            return None
        if not math.isfinite(pa) or not math.isfinite(pb):
            return None
        lin_sum.append(pa + pb)
        lin_diff.append(abs(pa - pb))
    if not all(math.isfinite(value) for value in (*lin_sum, *lin_diff)):
        return None
    def _to_spl(lin):
        return [20.0 * math.log10(v) if v > 1e-12 else -120.0 for v in lin]
    return {"freq": list(fa), "sum_spl": _to_spl(lin_sum), "diff_spl": _to_spl(lin_diff)}


def tof_table(metadata):
    """Extract time-of-flight rows from metadata.timing_diagnostics."""
    timing = (metadata or {}).get("timing_diagnostics") or {}
    rows = []
    for ch in timing.get("channels") or []:
        if not isinstance(ch, dict):
            continue
        rows.append({
            "name": str(ch.get("name", "?")),
            "before_ms": ch.get("measured_arrival_ms"),
            "applied_ms": ch.get("applied_delay_ms"),
            "after_ms": ch.get("final_arrival_ms"),
            "offset_ms": ch.get("final_offset_from_reference_ms"),
        })
    return rows


def t60_rows(channel_data):
    """Validate emitted measured-room octave fits before displaying them."""
    report = (channel_data or {}).get("t60_octaves")
    if not isinstance(report, dict) or report.get("basis") != "measured_room_ir":
        return None
    min_r2 = report.get("min_r2")
    if (isinstance(min_r2, bool) or not isinstance(min_r2, (int, float)) or not math.isfinite(min_r2)
            or not 0 < min_r2 <= 1):
        return None
    rows = report.get("bands")
    if not isinstance(rows, list) or len(rows) != len(T60_OCTAVE_CENTERS_HZ):
        return None
    validated = []
    for center, row in zip(T60_OCTAVE_CENTERS_HZ, rows):
        if not isinstance(row, dict) or row.get("centre_hz") != center:
            return None
        valid = row.get("valid")
        if not isinstance(valid, bool):
            return None
        r2 = row.get("r2")
        if (isinstance(r2, bool) or not isinstance(r2, (int, float))
                or not math.isfinite(r2) or not 0 <= r2 <= 1):
            return None
        fit = row.get("fit_range")
        value = row.get("t60_s")
        reason = row.get("reason")
        if valid:
            if (fit not in ("T20", "T30") or isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(value) or value <= 0 or r2 < min_r2):
                return None
            validated.append({"centre_hz": center, "t60_s": value,
                              "fit_range": fit, "r2": r2, "reason": ""})
        else:
            if not isinstance(reason, str) or not reason.strip():
                return None
            validated.append({"centre_hz": center, "t60_s": None,
                              "fit_range": None, "r2": r2, "reason": reason})
    return validated


def t60_flatness_tolerance_s(data):
    """(tolerance, declared) for the flatness share.

    The operator value wins when present and valid; otherwise the viewer
    applies the BS.1116 midband default (see
    DEFAULT_T60_FLATNESS_TOLERANCE_S) and reports it as undeclared. A
    malformed explicit value is treated as absent: roomeq structural
    validation already rejects such configs before any run, so a viewer-seen
    malformed value can only come from hand-edited JSON.
    """
    metadata = data.get("metadata") if isinstance(data, dict) else None
    tolerance = metadata.get("t60_flatness_tolerance_s") if isinstance(metadata, dict) else None
    if (isinstance(tolerance, bool) or not isinstance(tolerance, (int, float))
            or not math.isfinite(tolerance) or tolerance <= 0):
        return (DEFAULT_T60_FLATNESS_TOLERANCE_S, False)
    return (float(tolerance), True)


def t60_flatness_pct(data, channel_name):
    """Share of nine measured octave fits within ±tolerance of the room mean.

    The reference is the arithmetic mean of all nine fits from each complete
    measured channel. Incomplete channels do not define that reference, and
    their own percentage remains unavailable rather than gaining from a
    smaller denominator.
    """
    if not isinstance(data, dict):
        return None
    tolerance, _ = t60_flatness_tolerance_s(data)
    channels = data.get("channels")
    if not isinstance(channels, dict):
        return None
    complete = {}
    for name, channel in channels.items():
        rows = t60_rows(channel)
        if rows is not None and all(row["t60_s"] is not None for row in rows):
            complete[name] = [row["t60_s"] for row in rows]
    if channel_name not in complete:
        return None
    values = [value for row in complete.values() for value in row]
    try:
        mean = math.fsum(value / len(values) for value in values)
    except OverflowError:
        return None
    if not math.isfinite(mean):
        return None
    inside = sum(abs(value - mean) <= tolerance for value in complete[channel_name])
    return 100.0 * inside / len(complete[channel_name])


def t60_table_html(channel_data):
    """Render all nine bands, including reasons for unavailable estimates."""
    rows = t60_rows(channel_data)
    if rows is None:
        return ('<p class="epa-footer">Octave T60: pending roomeq field '
                't60_octaves from a measured room impulse response.</p>')
    parts = [
        '<div class="filters-section"><h3>Octave-band T60</h3>',
        '<p class="epa-footer">Measured room IR; T30 preferred, T20 fallback. '
        'Invalid bands are excluded from the curve.</p>',
        '<table class="epa-table"><thead><tr><th>Centre (Hz)</th><th>T60 (s)</th>'
        '<th>Fit</th><th>R²</th><th>Status</th></tr></thead><tbody>',
    ]
    for row in rows:
        value = f"{row['t60_s']:.3f}" if row["t60_s"] is not None else "n/a"
        fit = row["fit_range"] or "n/a"
        reason = escape(row["reason"] or "valid")
        parts.append(f"<tr><td>{row['centre_hz']}</td><td>{value}</td>"
                     f"<td>{fit}</td><td>{row['r2']:.3f}</td><td>{reason}</td></tr>")
    parts.append("</tbody></table></div>")
    return "".join(parts)


def early_late_ratio_db(channel_data):
    """Mean emitted early-minus-late dB over supported 1–8 kHz centers."""
    report = (channel_data or {}).get("early_late_curves")
    if (not isinstance(report, dict) or report.get("method") != "incoherent_band_energy"
            or report.get("reference") != "full_peak_band"
            or report.get("smoothing") != "third_octave"
            or report.get("split_ms") != 20.0):
        return None
    curves = [report.get(key) for key in ("full", "early", "late")]
    if not all(isinstance(curve, dict) for curve in curves):
        return None
    freq = curves[0].get("freq")
    if (not isinstance(freq, list) or len(freq) < 2
            or any(isinstance(f, bool) or not isinstance(f, (int, float))
                   or not math.isfinite(f) or f <= 0 for f in freq)
            or any(a >= b for a, b in zip(freq, freq[1:]))
            or freq[0] > 1000.0 or freq[-1] < 8000.0):
        return None
    for curve in curves:
        spl = curve.get("spl")
        if (curve.get("freq") != freq or not isinstance(spl, list) or len(spl) != len(freq)
                or any(isinstance(v, bool) or not isinstance(v, (int, float))
                       or not math.isfinite(v) for v in spl)):
            return None
    differences = [e - l for f, e, l in zip(freq, curves[1]["spl"], curves[2]["spl"])
                   if 1000.0 <= f <= 8000.0]
    if len(differences) < 2 or not all(math.isfinite(value) for value in differences):
        return None
    try:
        result = math.fsum(value / len(differences) for value in differences)
    except OverflowError:
        return None
    return result if math.isfinite(result) else None


def _validated_early_reflections(channel_data):
    """Return a measured-IR event report only when its six-column data is valid."""
    report = (channel_data or {}).get("early_reflections")
    if (not isinstance(report, dict)
            or report.get("basis") != "measured_room_ir"
            or report.get("method") != "bandlimited_early_reflection_table_v1"
            or report.get("band_hz") != [1000.0, 8000.0]
            or report.get("threshold_dbfs") != -15.0
            or not isinstance(report.get("direct_reference"), str)
            or not report["direct_reference"]):
        return None
    for side in ("pre", "post"):
        events = report.get(side)
        if not isinstance(events, list) or len(events) > 64:
            return None
        last_time = 0.0
        for event in events:
            if not isinstance(event, dict):
                return None
            gain, time, distance, dip = (event.get(key) for key in
                                         ("gain_dbfs", "time_ms", "distance_cm", "first_dip_hz"))
            ripple = event.get("ripple_db")
            if (any(isinstance(v, bool) or not isinstance(v, (int, float))
                    or not math.isfinite(v) for v in (gain, time, distance, dip))
                    or not -15.0 <= gain <= 0.0 or not last_time < time <= 15.0
                    or distance <= 0 or dip <= 0
                    or abs(distance - 34.3 * time) > max(0.1, 0.02 * 34.3 * time)
                    or abs(dip - 500.0 / time) > max(0.5, 0.02 * 500.0 / time)
                    or ripple is not None and (isinstance(ripple, bool)
                    or not isinstance(ripple, (int, float)) or not math.isfinite(ripple)
                    or ripple < 0)):
                return None
            last_time = time
    return report


def early_reflection_level_db(channel_data):
    """Return (strongest post-direct level, upper_bound) from emitted events."""
    report = _validated_early_reflections(channel_data)
    if report is None:
        return None
    post = report["post"]
    if not post:
        return -15.0, True
    return max(event["gain_dbfs"] for event in post), False


def early_reflections_html(channel_data, label):
    """Render emitted reflection candidates without estimating arrivals."""
    report = _validated_early_reflections(channel_data)
    if report is None:
        return ('<p class="epa-footer">1–8 kHz early reflections: pending '
                'roomeq field early_reflections from a measured room IR.</p>')
    parts = ['<div class="filters-section"><h3>1–8 kHz early reflections: '
             + escape(str(label)) + '</h3>',
             '<p class="epa-footer">Emitted band-limited candidates within 15 ms '
             'of each filtered direct peak. Levels are dB relative to that peak; '
             'the two-path dip and ripple are estimates, not confirmed acoustic '
             'nulls or audibility verdicts. Reference: '
             + escape(report["direct_reference"]) + '.</p>']
    for side, title, color in (("pre", "Before", "#2864b4"),
                               ("post", "After", "#c56816")):
        events = report[side]
        parts.append(f'<h4>{title} ({len(events)} candidates)</h4>')
        parts.append(f'<figure><figcaption>{title}: post-direct time (ms) vs '
                     'level (dB relative to direct). The direct peak is at 0 ms, 0 dB.'
                     '</figcaption><svg viewBox="0 0 800 230" role="img" '
                     'aria-label="Band-limited early-reflection levels">'
                     f'<circle cx="40" cy="20" r="4" fill="{color}"/>')
        for event in events:
            x = 40 + 720 * event["time_ms"] / 15.0
            y = 20 + 190 * (-event["gain_dbfs"]) / 15.0
            parts.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4" fill="{color}"/>')
        parts.append('</svg></figure><table class="epa-table"><thead><tr>'
                     '<th>Reflection</th><th>Gain (dB re direct)</th><th>Time (ms)</th>'
                     '<th>Extra path (cm)</th><th>First dip (Hz)</th>'
                     '<th>Comb ripple (dB p-p)</th></tr></thead><tbody>')
        for number, event in enumerate(events, 1):
            ripple = event["ripple_db"]
            ripple_text = f'{ripple:.2f}' if ripple is not None else 'unbounded'
            parts.append(f'<tr><td>{number}</td><td>{event["gain_dbfs"]:.2f}</td>'
                         f'<td>{event["time_ms"]:.2f}</td>'
                         f'<td>{event["distance_cm"]:.1f}</td>'
                         f'<td>{event["first_dip_hz"]:.1f}</td><td>{ripple_text}</td></tr>')
        parts.append('</tbody></table>')
    curve = (channel_data or {}).get("final_curve")
    freqs = curve.get("freq") if isinstance(curve, dict) else None
    levels = curve.get("spl") if isinstance(curve, dict) else None
    if (isinstance(freqs, list) and isinstance(levels, list)
            and 2 <= len(freqs) == len(levels) <= 65_536
            and all(isinstance(f, (int, float)) and not isinstance(f, bool)
                    and math.isfinite(f) and f > 0 for f in freqs)
            and all(isinstance(v, (int, float)) and not isinstance(v, bool)
                    and math.isfinite(v) for v in levels)
            and all(a < b for a, b in zip(freqs, freqs[1:]))):
        low, high = min(levels), max(levels)
        yspan = max(high - low, 1e-9)
        xspan = math.log(freqs[-1] / freqs[0])
        if math.isfinite(yspan) and math.isfinite(xspan) and xspan > 0:
            points = ' '.join(f'{40 + 720 * math.log(f/freqs[0])/xspan:.2f},'
                              f'{210 - 185 * (v-low)/yspan:.2f}'
                              for f, v in zip(freqs, levels))
            parts.append('<figure><figcaption>Final response magnitude with '
                         'post-correction two-path first-dip estimates. Vertical '
                         'lines mark candidate frequencies, not measured nulls.'
                         '</figcaption><svg viewBox="0 0 800 240" role="img" '
                         'aria-label="Final frequency response and reflection dip estimates">'
                         f'<polyline fill="none" stroke="#2864b4" points="{points}"/>')
            for event in report["post"]:
                dip = event["first_dip_hz"]
                if freqs[0] <= dip <= freqs[-1]:
                    x = 40 + 720 * math.log(dip / freqs[0]) / xspan
                    right = next(i for i, freq in enumerate(freqs) if freq >= dip)
                    if right == 0:
                        level = levels[0]
                    else:
                        left = right - 1
                        fraction = math.log(dip / freqs[left]) / math.log(freqs[right] / freqs[left])
                        level = levels[left] + fraction * (levels[right] - levels[left])
                    y = 210 - 185 * (level - low) / yspan
                    parts.append(f'<line x1="{x:.2f}" y1="210" x2="{x:.2f}" y2="{y:.2f}" '
                                 'stroke="#c56816" stroke-dasharray="4,3"/>')
            parts.append('</svg></figure>')
        else:
            parts.append('<p>First-dip overlay unavailable: invalid final response range.</p>')
    else:
        parts.append('<p>First-dip overlay unavailable: no valid final response curve.</p>')
    parts.append('</div>')
    return ''.join(parts)


def room_t60_rows(data):
    """Mean valid octave T60 estimates, retaining per-band speaker counts."""
    sources = [rows for channel in (data.get("channels") or {}).values()
               if (rows := t60_rows(channel)) is not None]
    if not sources:
        return None
    room = []
    for i, center in enumerate(T60_OCTAVE_CENTERS_HZ):
        values = [rows[i]["t60_s"] for rows in sources if rows[i]["t60_s"] is not None]
        room.append({"centre_hz": center,
                     "t60_s": sum(values) / len(values) if values else None,
                     "speaker_count": len(values)})
    return room


def notch_cell(notch):
    """Colour-coded HTML cell for the deepest-notch column."""
    if notch is None:
        return '<td style="background:#eee;color:#888">n/a</td>'
    th = SUMMARY_THRESHOLDS["notch_db"]
    color = "#2ecc71" if th["green"](notch) else "#f1c40f" if th["yellow"](notch) else "#e74c3c"
    return f'<td style="background:{color}33">{notch:+.1f}</td>'


def early_late_ratio_cell(channel_data):
    ratio = early_late_ratio_db(channel_data)
    if ratio is None:
        return pending_cell("early_late_curves")
    th = SUMMARY_THRESHOLDS["early_late_ratio_db"]
    color = "#2ecc71" if th["green"](ratio) else "#f1c40f" if th["yellow"](ratio) else "#e74c3c"
    return (f'<td style="background:{color}33" title="arithmetic mean of '
            f'early minus late dB at emitted 1–8 kHz third-octave centers">{ratio:+.1f}</td>')


def early_reflection_level_cell(channel_data):
    metric = early_reflection_level_db(channel_data)
    if metric is None:
        return pending_cell("early_reflections")
    level, upper_bound = metric
    th = SUMMARY_THRESHOLDS["early_reflection_level_db"]
    color = "#2ecc71" if th["green"](level) else "#f1c40f" if th["yellow"](level) else "#e74c3c"
    displayed = f'≤ {level:.1f}' if upper_bound else f'{level:+.1f}'
    return (f'<td style="background:{color}33" '
            f'title="strongest emitted post-correction 1–8 kHz reflection, '
            f'dB relative to direct">{displayed}</td>')


def t60_flatness_cell(data, channel_name):
    metric = t60_flatness_pct(data, channel_name)
    if metric is None:
        return pending_cell("nine valid measured t60_octaves fits")
    th = SUMMARY_THRESHOLDS["t60_flatness_pct"]
    color = "#2ecc71" if th["green"](metric) else "#f1c40f" if th["yellow"](metric) else "#e74c3c"
    tolerance, declared = t60_flatness_tolerance_s(data)
    source = (f"declared ±{tolerance:g} s"
              if declared else
              f"default ±{tolerance:g} s (ITU-R BS.1116-2 §8.2.3.1 Fig. 1 midband; "
              f"declare reporting.t60_flatness_tolerance_s to override)")
    return (f'<td style="background:{color}33" title="nine measured octave fits '
            f'within {escape(source)} of the complete-channel room mean\">{metric:.1f}</td>')


def pending_cell(field):
    return f'<td style="background:#eee;color:#888" title="needs roomeq field: {escape(field)}">pending</td>'


def operational_summaries_by_channel(data):
    """Index emitted req-R6 channel summaries by delivered output identity."""
    ledger = data.get("correction_decisions", {}) or {}
    summaries = ledger.get("channel_summaries", []) or []
    indexed = {}
    for entry in summaries:
        if not isinstance(entry, dict):
            continue
        channel = entry.get("channel")
        pct = entry.get("operational_response_pct")
        decided = entry.get("decided_equalize")
        delivered = entry.get("delivered_equalize")
        if (not isinstance(channel, str) or not channel
                or not isinstance(pct, (int, float)) or isinstance(pct, bool)
                or not math.isfinite(pct)):
            continue
        indexed[channel] = (float(pct), decided, delivered)
    return indexed


def operational_response_cell(summaries_by_channel, name):
    """Section 1 operational-response cell from the emitted ledger share."""
    hit = summaries_by_channel.get(name)
    if hit is None:
        return pending_cell("correction_decisions.channel_summaries")
    pct, decided, delivered = hit
    th = SUMMARY_THRESHOLDS["operational_room_response_pct"]
    color = "#2ecc71" if th["green"](pct) else "#f1c40f" if th["yellow"](pct) else "#e74c3c"
    detail = f"{delivered}/{decided} final EQ scope delivered" if (
        isinstance(decided, int) and isinstance(delivered, int)) else "final EQ scope share"
    return (f'<td style="background:{color}33" title="operational room response: {escape(detail)} '
            f"(share of decided final equalization records delivered: Applied/AlreadyAcceptable/Constrained; "
            f"empty scope stays pending, never 100%)\">{pct:.1f}</td>")


def summary_table_html(data):
    """Section 1 summary from emitted evidence, with pending unknown metrics."""
    channels = data.get("channels", {}) or {}
    summaries_by_channel = operational_summaries_by_channel(data)
    names = sorted(channels.keys())
    parts = [
        '<div class="filters-section">\n<h3>Section 1 — Results summary</h3>\n',
        '<p class="epa-footer">Colour thresholds per reviews/feat-report.md. '
        '"pending" cells name the roomeq field required '
        '(see reviews/req-roomeq-report.md).</p>\n',
        '<table class="epa-table"><thead><tr><th>Speaker</th>'
        "<th>Operational room response (%)</th>"
        "<th>Early reflection level (dB)</th>"
        "<th>Early vs late ratio (dB)</th>"
        "<th>T60 flatness in window (%)</th>"
        "<th>Deepest notch &lt; 300 Hz (dB)</th>"
        "</tr></thead><tbody>\n",
    ]
    for name in names:
        notch = deepest_notch_db(channels[name])
        parts.append(
            f"<tr><td>{escape(str(name))}</td>"
            f"{operational_response_cell(summaries_by_channel, name)}"
            f"{early_reflection_level_cell(channels[name])}"
            f"{early_late_ratio_cell(channels[name])}"
            f"{t60_flatness_cell(data, name)}"
            f"{notch_cell(notch)}</tr>\n"
        )
    parts.append("</tbody></table></div>\n")
    return "".join(parts)


def level_compensation_html(data):
    """Section 2 relative-level-compensation table."""
    rows = level_compensation(data.get("channels", {}) or {})
    if not rows:
        return ""
    parts = [
        '<div class="filters-section">\n<h3>Section 2 — Relative level compensation</h3>\n',
        '<p class="epa-footer">Monitor band 0.5–3 kHz, sub band 30–80 Hz. '
        "Reference: quietest monitor. These are proposed attenuation values; "
        "the post-calibration offset needs an independent measurement.</p>\n",
        '<table class="epa-table"><thead><tr><th>Speaker</th>'
        "<th>Level compensation (dB)</th>"
        "<th>Level offset after calibration (dB)</th>"
        "</tr></thead><tbody>\n",
    ]
    for r in rows:
        comp = f"{r['comp_db']:+.1f}" if r["comp_db"] is not None else "n/a"
        res = f"{r['residual_db']:+.1f}" if r["residual_db"] is not None else "n/a"
        parts.append(f"<tr><td>{escape(r['speaker'])}</td><td>{comp}</td><td>{res}</td></tr>\n")
    parts.append("</tbody></table></div>\n")
    return "".join(parts)


def tof_html(metadata):
    """Section 3 time-of-flight table (measured input and calculated output)."""
    rows = tof_table(metadata)
    if not rows:
        return ""
    parts = [
        '<div class="filters-section">\n<h3>Section 3 — Time of flight</h3>\n',
        '<p class="epa-footer">After-DSP arrival adds the deployed chain delay '
        'to the measured input arrival; it is not a post-playback capture.</p>',
        '<table class="epa-table"><thead><tr><th>Monitor</th>'
        "<th>Measured arrival before DSP (ms)</th>"
        "<th>Applied delay (ms)</th>"
        "<th>Calculated arrival after DSP (ms)</th>"
        "<th>Offset vs reference (ms)</th>"
        "</tr></thead><tbody>\n",
    ]
    for r in rows:
        def _f(v):
            return (f"{v:.3f}" if isinstance(v, (int, float))
                    and not isinstance(v, bool) and math.isfinite(v) else "n/a")
        parts.append(
            f"<tr><td>{escape(r['name'])}</td><td>{_f(r['before_ms'])}</td>"
            f"<td>{_f(r['applied_ms'])}</td><td>{_f(r['after_ms'])}</td>"
            f"<td>{_f(r['offset_ms'])}</td></tr>\n"
        )
    parts.append("</tbody></table></div>\n")
    return "".join(parts)
