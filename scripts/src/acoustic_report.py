"""Viewer-side acoustic helpers for the feat-report sections.

Only operations fully determined by the roomeq output JSON live here:
fractional-octave smoothing (reusing the viewer convention), band means,
IR envelopes, and arrival-time tables. Anything needing new measurement
analysis (RT60, reflection picking, waterfall/wavelet, early/late splits)
is a roomeq/math-audio requirement, not a viewer estimate.
"""

import math
from html import escape
from itertools import pairwise

from .dsp import smooth_octave

SPEED_OF_SOUND_M_S = 343.0

# Colour thresholds from reviews/feat-report.md (green, yellow, red).
# Each entry: (green_predicate, yellow_predicate); else red.
SUMMARY_THRESHOLDS = {
    "operational_room_response_pct": {"green": lambda v: v > 90.0,
                                      "yellow": lambda v: v >= 80.0},
    "t60_flatness_pct": {"green": lambda v: v > 80.0,
                          "yellow": lambda v: v >= 50.0},
    "t60_itu_pct": {"green": lambda v: v > 80.0,
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


def _predicted_band_mean(channel, lo_hz, hi_hz):
    """Band mean of the predicted post-DSP curve, or None when absent."""
    if not isinstance(channel, dict):
        return None
    curve = channel.get("final_curve")
    if not curve or not curve.get("freq"):
        return None
    return band_mean(curve["freq"], curve.get("spl"), lo_hz, hi_hz)


def level_compensation(channels):
    """Remaining downstream trim to align predicted monitor band levels.

    Monitor reference band 0.5-3 kHz, sub band 30-80 Hz. Predicted
    post-DSP band means (final_curve) already contain any in-chain level
    alignment, so the trim is measured against the quietest predicted
    monitor instead of the pre-DSP levels; without predicted curves it
    falls back to the pre-DSP proposal. Residual and balance include the
    shown trim, so the balance column verifies the loop closes near
    zero. Predictions only, not verification captures.
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
    reference_name = min(monitor_means, key=monitor_means.get)
    predicted_means = {name: _predicted_band_mean(channels[name], 500.0, 3000.0)
                       for name in monitor_means}
    predicted_reference = predicted_means.get(reference_name)
    for name, ch in channels.items():
        curve = (ch or {}).get("initial_curve")
        if not curve or not curve.get("freq"):
            rows.append({"speaker": str(name), "comp_db": None, "residual_db": None})
            continue
        if is_sub_channel(name):
            mean = band_mean(curve["freq"], curve.get("spl"), 30.0, 80.0)
            pre_comp = (reference - mean) if mean is not None else None
            predicted = _predicted_band_mean(ch, 30.0, 80.0)
        else:
            mean = monitor_means.get(name)
            pre_comp = reference - mean if mean is not None else None
            predicted = _predicted_band_mean(ch, 500.0, 3000.0)
        if predicted is not None and predicted_reference is not None:
            comp = predicted_reference - predicted
        else:
            comp = pre_comp
        landed = (predicted + comp) if predicted is not None and comp is not None else None
        residual = (landed - reference) if landed is not None else None
        rows.append({
            "speaker": str(name),
            "comp_db": comp,
            "residual_db": residual,
            "balance_db": (landed - predicted_reference
                           if landed is not None and predicted_reference is not None else None),
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


def response_landmarks(channel_data):
    """Peaks, notches and LF extension from the measured initial curve.

    Runs on a 1/3-octave-smoothed copy of ``initial_curve`` (Genelec GRADE
    §3.1/§3.5 style landmarks, computed viewer-side from the magnitude
    response only):

    - peaks/notches: local extrema with prominence >= 3 dB, merged within
      1/12 octave keeping the strongest; top 3 each by prominence;
    - lf_extension_hz: -6 dB point below the 30-200 Hz peak, scanning down
      from the peak; None when the curve never drops 6 dB (reported as
      "< 20 Hz" upstream) or the band is empty.

    Returns None when no usable initial curve exists.
    """
    curve = (channel_data or {}).get("initial_curve")
    if not curve or not curve.get("freq") or not curve.get("spl"):
        return None
    points = _valid_curve_points(curve["freq"], curve["spl"])
    if points is None or len(points) < 5:
        return None
    freq = [f for f, _ in points]
    smoothed = smooth_octave(freq, [s for _, s in points], 1.0 / 3.0)
    n = len(freq)

    def prominence(index, sign):
        """Prominence of an extremum at index (sign=+1 peak, -1 notch)."""
        level = smoothed[index] * sign
        left_base = level
        j = index - 1
        while j >= 0:
            v = smoothed[j] * sign
            if v > level:
                break
            left_base = min(left_base, v)
            j -= 1
        right_base = level
        j = index + 1
        while j < n:
            v = smoothed[j] * sign
            if v > level:
                break
            right_base = min(right_base, v)
            j += 1
        return level - max(left_base, right_base)

    extrema = []
    for i in range(1, n - 1):
        if not (20.0 <= freq[i] <= 20000.0):
            continue
        if smoothed[i] > smoothed[i - 1] and smoothed[i] >= smoothed[i + 1]:
            prom = prominence(i, 1.0)
            if prom >= 3.0:
                extrema.append(("peak", freq[i], smoothed[i], prom))
        elif smoothed[i] < smoothed[i - 1] and smoothed[i] <= smoothed[i + 1]:
            prom = prominence(i, -1.0)
            if prom >= 3.0:
                extrema.append(("notch", freq[i], smoothed[i], prom))

    def merged(kind):
        """Strongest-first extrema of one kind, 1/12 octave apart."""
        selected = []
        for _, f, level, prom in sorted(
            (e for e in extrema if e[0] == kind), key=lambda e: -e[3]
        ):
            if all(abs(math.log2(f / kept)) >= 1.0 / 12.0 for kept, _ in selected):
                selected.append((f, level))
            if len(selected) == 3:
                break
        return selected

    peaks = merged("peak")
    notches = merged("notch")

    lf_extension = None
    band = [(f, s) for f, s in zip(freq, smoothed) if 30.0 <= f <= 200.0]
    if band:
        peak_level = max(s for _, s in band)
        peak_freq = next(f for f, s in band if s == peak_level)
        # Scan down from the peak: the -6 dB point is the highest frequency
        # at or below the peak whose level has fallen 6 dB.
        below_peak = [(f, s) for f, s in zip(freq, smoothed) if f <= peak_freq]
        for f, s in reversed(below_peak):
            if s < peak_level - 6.0:
                lf_extension = f
                break
    return {"peaks": peaks, "notches": notches, "lf_extension_hz": lf_extension}


def landmarks_table_html(data):
    """Per-speaker peaks, notches and LF extension (Genelec GRADE §3.5/§3.3)."""
    channels = data.get("channels", {}) or {}
    if not channels:
        return ""
    parts = [
        '<div class="filters-section">\n<h3>Frequency landmarks — peaks, notches, LF extension</h3>\n',
        ('<p class="epa-footer">1/3-octave-smoothed measured response. '
        "Extrema need 3 dB prominence, 1/12 octave apart (top 3 each). "
        "LF extension is the −6 dB point below the 30–200 Hz peak.</p>\n"),
        ('<table class="epa-table"><thead><tr><th>Speaker</th>'
        "<th>LF extension (−6 dB)</th>"
        "<th>Strongest peaks (freq / level)</th>"
        "<th>Strongest notches (freq / level)</th>"
        "</tr></thead><tbody>\n"),
    ]
    for name in sorted(channels.keys()):
        marks = response_landmarks(channels[name])
        if marks is None:
            parts.append(
                f"<tr><td>{escape(str(name))}</td>"
                '<td style="color:#999">n/a</td>'
                '<td style="color:#999">n/a</td>'
                '<td style="color:#999">n/a</td></tr>\n'
            )
            continue
        lf = marks["lf_extension_hz"]
        lf_str = f"{lf:.0f} Hz" if lf is not None else "&lt; 20 Hz"
        peak_str = ("<br>".join(f"{f:.0f} Hz / {level:+.1f} dB" for f, level in marks["peaks"])
                    or "—")
        notch_str = ("<br>".join(f"{f:.0f} Hz / {level:+.1f} dB" for f, level in marks["notches"])
                     or "—")
        parts.append(
            f"<tr><td>{escape(str(name))}</td><td>{lf_str}</td>"
            f"<td>{peak_str}</td><td>{notch_str}</td></tr>\n"
        )
    parts.append("</tbody></table></div>\n")
    return "".join(parts)


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


def t60_itu_reference(data):
    """Reference for BS.1116-3 Figure 1, with explicit volume provenance."""
    metadata = data.get("metadata") or {}
    config = metadata.get("effective_config") or {}
    recording = config.get("recording_config") or {}
    dims = recording.get("room_dimensions")
    if isinstance(dims, dict):
        values = [dims.get(key) for key in ("length", "width", "height")]
        if all(isinstance(v, (int, float)) and not isinstance(v, bool)
               and math.isfinite(v) and v > 0 for v in values):
            volume = math.prod(values)
            if math.isfinite(volume) and volume > 0:
                return {"tm": 0.25 * (volume / 100.0) ** (1.0 / 3.0),
                        "volume_m3": volume,
                        "source": f"volume recommendation ({volume:g} m³)"}
    # Octave-band estimate of the specified midband mean. Require all five
    # centers for each contributing channel, never use the bass/16 kHz bands.
    means = []
    for channel in (data.get("channels") or {}).values():
        rows = t60_rows(channel)
        if rows is None:
            continue
        midband = [r["t60_s"] for r in rows if 200 <= r["centre_hz"] <= 4000]
        if len(midband) == 5 and all(v is not None for v in midband):
            means.append(math.fsum(v / 5 for v in midband))
    if not means:
        return None
    return {"tm": math.fsum(v / len(means) for v in means),
            "source": "measured midband mean; room volume unavailable"}


def t60_itu_note(reference):
    """Explain the recommendation without claiming full listening-room compliance."""
    source = (f'Tm = {reference["tm"]:.3f} s — {escape(reference["source"])}'
              if reference else 'limits unavailable: no room volume or complete midband fits')
    return ('<p class="epa-footer">ITU-R BS.1116-3 §8.2.3.1, Figure 1: '
            + source + '. Upper limit covers 63 Hz–8 kHz; lower limit starts at '
            '100 Hz. No extrapolated limits outside that range. A measured-mean '
            'reference checks relative decay shape only, not the volume-based '
            'recommendation. The summary’s fixed-window flatness score is a separate metric. '
            '<a href="https://www.itu.int/dms_pubrec/itu-r/rec/bs/R-REC-BS.1116-3-201502-I!!PDF-E.pdf">'
            'ITU recommendation</a>.</p>')


def t60_flatness_window(data):
    """Plot the same complete-channel reference and tolerance as the summary."""
    complete = [rows for channel in (data.get("channels") or {}).values()
                if (rows := t60_rows(channel)) is not None
                and all(row["t60_s"] is not None for row in rows)]
    values = [row["t60_s"] for rows in complete for row in rows]
    if not values:
        return None
    mean = math.fsum(value / len(values) for value in values)
    tolerance, _ = t60_flatness_tolerance_s(data)
    return max(0.0, mean - tolerance), mean + tolerance


def room_t60_table_html(data):
    """Keep octave values and speaker coverage next to the room-average graph."""
    rows = room_t60_rows(data)
    if not rows:
        return ""
    channels = [(name, bands) for name, channel in (data.get("channels") or {}).items()
                if (bands := t60_rows(channel)) is not None]
    parts = [('<h3>Octave-band T60 — room and speakers</h3><table class="epa-table">'
             '<thead><tr><th>Centre (Hz)</th><th>Room mean (s)</th><th>Valid speakers</th>')]
    parts.extend(f'<th>{escape(name)} T60 (s)</th>' for name, _ in channels)
    parts.append('</tr></thead><tbody>')
    for i, row in enumerate(rows):
        value = f'{row["t60_s"]:.3f}' if row["t60_s"] is not None else 'n/a'
        parts.append(f'<tr><td>{row["centre_hz"]}</td><td>{value}</td><td>{row["speaker_count"]}</td>')
        for _, bands in channels:
            value = f'{bands[i]["t60_s"]:.3f}' if bands[i]["t60_s"] is not None else 'n/a'
            parts.append(f'<td>{value}</td>')
        parts.append('</tr>')
    parts.append('</tbody></table>')
    return ''.join(parts)


def t60_table_html(channel_data):
    """Render all nine bands, including reasons for unavailable estimates."""
    rows = t60_rows(channel_data)
    if rows is None:
        return ('<p class="epa-footer">Octave T60: pending roomeq field '
                't60_octaves from a measured room impulse response.</p>')
    parts = [
        '<div class="filters-section"><h3>Octave-band T60</h3>',
        ('<p class="epa-footer">Measured room IR; T30 preferred, T20 fallback. '
        'Invalid bands are excluded from the curve.</p>'),
        ('<table class="epa-table"><thead><tr><th>Centre (Hz)</th><th>T60 (s)</th>'
        '<th>Fit</th><th>R²</th><th>Status</th></tr></thead><tbody>'),
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
            or any(a >= b for a, b in pairwise(freq))
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
    """Describe and tabulate emitted candidates; plots are standard figure sections."""
    report = _validated_early_reflections(channel_data)
    if report is None:
        return ('<p class="epa-footer">1–8 kHz early reflections: pending '
                'roomeq field early_reflections from a measured room IR.</p>')
    parts = ['<div class="filters-section"><h3>1–8 kHz early reflections: '
             + escape(str(label)) + '</h3>',
             '<p>Candidate search: first 15 ms after the filtered direct peak, '
             'at or above −15 dB relative to direct sound. Before uses the measured '
             'room IR; after is predicted through the saved DSP chain, not a new measurement. '
             'Reference: ' + escape(report["direct_reference"]) + '.</p>',
             ('<p>The exported data contains candidate peaks, not a continuous '
             '1–8 kHz energy-time curve. Two-path first-dip and ripple values are '
             'estimates, not confirmed acoustic nulls.</p>')]
    for side, title in (("pre", "Before"), ("post", "Predicted after EQ")):
        events = report[side]
        parts.append(f'<h4>{title} ({len(events)} candidates)</h4>')
        if not events:
            parts.append('<p>No candidates were emitted at or above −15 dB '
                         'within the 15 ms search window. This does not mean '
                         'the room has no reflections.</p>')
            continue
        parts.append('<table class="epa-table"><thead><tr>'
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
    parts.append('</div>')
    return ''.join(parts)


def early_reflection_figures(channel_data, label, tab=None):
    """Plot exported candidate stems and frequency responses, without inventing an ETC."""
    from .wasm_report import annotation, axis, figure, series

    report = _validated_early_reflections(channel_data)
    if report is None:
        return []
    figures = []
    for side, title, color in (("pre", "Before", "#2864b4"),
                               ("post", "Predicted after EQ", "#c56816")):
        events = report[side]
        if not events:
            continue
        xs, ys, labels = [], [], []
        for number, event in enumerate(events, 1):
            time, gain = event["time_ms"], event["gain_dbfs"]
            xs.extend([time, time, time])
            ys.extend([-40, gain, None])
            labels.append(annotation(time, gain + 1, str(number)))
        figures.append(figure(
            f"{label}: {title} — early-reflection candidates (not a continuous ETC)",
            axis("Time after direct peak (ms)", "linear", 0, 15),
            axis("Level relative to direct sound (dB)", "linear", -40, 3),
            series_list=[
                series(title, xs, ys, color=color),
                series("Direct reference", [0, 0], [-40, 0], color="#555555"),
                series("Detection threshold (−15 dB)", [0, 15], [-15, -15],
                       color="#888888", dash="dash"),
            ], annotations=labels, tab=tab))
    curves = []
    for key, name, color in (("initial_curve", "Measured before EQ", "#2864b4"),
                              ("final_curve", "Predicted after EQ", "#c56816")):
        curve = (channel_data or {}).get(key)
        if not isinstance(curve, dict):
            continue
        freqs, levels = curve.get("freq"), curve.get("spl")
        if (not isinstance(freqs, list) or not isinstance(levels, list)
                or not 2 <= len(freqs) == len(levels) <= 65_536
                or not all(isinstance(f, (int, float)) and not isinstance(f, bool)
                           and math.isfinite(f) and f > 0 for f in freqs)
                or not all(isinstance(v, (int, float)) and not isinstance(v, bool)
                           and math.isfinite(v) for v in levels)
                or not all(a < b for a, b in pairwise(freqs))):
            continue
        curves.append(series(name, freqs, levels, color=color))
    if curves:
        # Same 50 dB convention as every other SPL-vs-frequency plot: the
        # top lands on a multiple of 5 dB so the 1 dB / 5 dB grid aligns.
        hi = math.ceil(max(max(s["y"]) for s in curves) / 5) * 5
        lo = hi - 50.0
        for number, event in enumerate(report["post"], 1):
            dip = event["first_dip_hz"]
            curves.append(series(f"Candidate {number}: {dip:.1f} Hz (estimated dip)",
                                 [dip, dip], [lo, hi], color="#888888", dash="dot"))
        figures.append(figure(
            f"{label}: response and estimated reflection dips (not measured nulls)",
            axis("Frequency (Hz)", "log", 20, 20000),
            axis("SPL (dB)", "linear", lo, hi),
            series_list=curves, tab=tab))
    return figures


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


def t60_itu_pct(data, channel_name):
    """Share of eight octave centers within the volume-based ITU envelope."""
    reference = t60_itu_reference(data)
    if reference is None or reference.get("volume_m3") is None:
        return None
    rows = t60_rows((data.get("channels") or {}).get(channel_name, {}))
    if rows is None:
        return None
    bands = [row for row in rows if 63 <= row["centre_hz"] <= 8000]
    if len(bands) != 8 or any(row["t60_s"] is None for row in bands):
        return None
    tm = reference["tm"]
    inside = 0
    for row in bands:
        hz, value = row["centre_hz"], row["t60_s"]
        # Figure 1's bass upper line is straight on a logarithmic x axis.
        upper = tm + (0.30 - 0.25 * math.log(hz / 63) / math.log(200 / 63)
                      if hz < 200 else 0.05 if hz <= 4000 else 0.10)
        lower = None if hz < 100 else tm - (0.05 if hz <= 4000 else 0.10)
        # Inclusive boundaries; absorb only floating-point arithmetic noise.
        inside += value <= upper + 1e-12 and (lower is None or value >= lower - 1e-12)
    return 100.0 * inside / 8


def t60_itu_cell(data, channel_name):
    metric = t60_itu_pct(data, channel_name)
    if metric is None:
        return ('<td style="background:#eee;color:#666" title="Requires valid '
                'recording_config.room_dimensions and all eight measured octave '
                'T60 fits from 63 Hz to 8 kHz">Not assessed</td>')
    th = SUMMARY_THRESHOLDS["t60_itu_pct"]
    color = "#2ecc71" if th["green"](metric) else "#f1c40f" if th["yellow"](metric) else "#e74c3c"
    return (f'<td style="background:{color}33" '
            f'title="{round(metric * 8 / 100)}/8 octave centers within ITU-R '
            'BS.1116-3 Figure 1 limits around the volume recommendation; '
            '63 Hz has only an upper bound; 4 kHz uses the midband limits; '
            f'16 kHz is excluded. Not overall room compliance.">{metric:.1f}</td>')


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


_DELIVERED_STATUSES = {"applied", "already_acceptable", "constrained"}


def _derive_summaries_from_decisions(decisions):
    """Derive req-R6 channel summaries from legacy decision records.

    Mirrors the normative rule in
    ``crates/roomeq-model/src/decision_ledger.rs`` (``operational_summaries``):
    scope is final-claim records (stage ``final`` with a non-empty
    ``final_graph_identity``) with action ``equalize``, grouped by
    ``physical_output``; records superseded by another record are excluded
    from both counts; delivered scope carries status ``applied``,
    ``already_acceptable`` or ``constrained``. Channels without decided
    final equalization scope stay absent (rendered as pending, never 100%).
    """
    superseded = set()
    for record in decisions:
        if not isinstance(record, dict):
            continue
        ids = record.get("supersedes_ids") or []
        if isinstance(ids, list):
            superseded.update(i for i in ids if isinstance(i, str))
    decided: dict[str, int] = {}
    delivered: dict[str, int] = {}
    for record in decisions:
        if not isinstance(record, dict):
            continue
        if record.get("stage") != "final":
            continue
        identity = record.get("final_graph_identity")
        if not isinstance(identity, str) or not identity.strip():
            continue
        if record.get("action") != "equalize":
            continue
        decision_id = record.get("decision_id")
        if isinstance(decision_id, str) and decision_id in superseded:
            continue
        channel = record.get("physical_output") or record.get("logical_input")
        if not isinstance(channel, str) or not channel:
            continue
        decided[channel] = decided.get(channel, 0) + 1
        if record.get("status") in _DELIVERED_STATUSES:
            delivered[channel] = delivered.get(channel, 0) + 1
    indexed = {}
    for channel, decided_count in decided.items():
        delivered_count = delivered.get(channel, 0)
        if delivered_count > decided_count:
            continue
        indexed[channel] = (
            100.0 * delivered_count / decided_count,
            decided_count,
            delivered_count,
        )
    return indexed


def operational_summaries_by_channel(data):
    """Index emitted req-R6 channel summaries by delivered output identity.

    Uses the engine-emitted ``channel_summaries`` when present; legacy
    outputs carry only ``decisions``, from which the same shares are
    derived with the normative ledger rule.
    """
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
    if indexed:
        return indexed
    decisions = ledger.get("decisions", []) or []
    if isinstance(decisions, list) and decisions:
        return _derive_summaries_from_decisions(decisions)
    return indexed


def operational_response_cell(summaries_by_channel, name):
    """Section 1 operational-response cell from the emitted ledger share."""
    hit = summaries_by_channel.get(name)
    if hit is None:
        return ('<td style="background:#eee;color:#666" '
                'title="No final EQ assessment is present in the saved decision ledger">'
                'Not assessed</td>')
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
        '<div class="filters-section">\n<h3>Results summary</h3>\n',
        ('<p class="epa-footer">Acoustic values use the imported room impulse response. '
        'Operational response is the share of final EQ decisions confirmed as delivered; '
        '“Not assessed” means the saved ledger has no final EQ assessment. '
        'Missing acoustic values require a measured room IR and valid analysis.</p>\n'),
        ('<table class="epa-table"><thead><tr><th>Speaker</th>'
        "<th>Operational room response (%)</th>"
        "<th>Early reflection level (dB)</th>"
        "<th>Early vs late ratio (dB)</th>"
        "<th>T60 flatness in window (%)</th>"
        "<th>T60 within ITU recommendation (%)</th>"
        "<th>Deepest notch &lt; 300 Hz (dB)</th>"
        "</tr></thead><tbody>\n"),
    ]
    for name in names:
        notch = deepest_notch_db(channels[name])
        parts.append(
            f"<tr><td>{escape(str(name))}</td>"
            f"{operational_response_cell(summaries_by_channel, name)}"
            f"{early_reflection_level_cell(channels[name])}"
            f"{early_late_ratio_cell(channels[name])}"
            f"{t60_flatness_cell(data, name)}"
            f"{t60_itu_cell(data, name)}"
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
        '<div class="filters-section">\n<h3>Relative level compensation</h3>\n',
        ('<p class="epa-footer">Monitor band 0.5–3 kHz, sub band 30–80 Hz. '
        "Compensation is the attenuation still needed downstream, measured on the "
        "predicted post-DSP curves so any in-chain level alignment is not applied "
        "twice. The post-DSP columns include that trim: the offset is the predicted "
        "landing level versus the quietest monitor before correction (it includes EQ, "
        "time alignment and headroom attenuation; a negative value does not by itself "
        "indicate channel imbalance), and the balance column verifies the remaining "
        "spread against the quietest monitor after DSP. "
        "These are predictions, not verification captures.</p>\n"),
        ('<table class="epa-table"><thead><tr><th>Speaker</th>'
        "<th>Level compensation (dB)</th>"
        "<th>Post-DSP offset from original reference (dB)</th>"
        "<th>Post-DSP balance vs reference monitor (dB)</th>"
        "</tr></thead><tbody>\n"),
    ]
    for r in rows:
        comp = f"{r['comp_db']:+.1f}" if r["comp_db"] is not None else "n/a"
        res = f"{r['residual_db']:+.1f}" if r["residual_db"] is not None else "n/a"
        balance = f"{r['balance_db']:+.1f}" if r.get('balance_db') is not None else "n/a"
        parts.append(f"<tr><td>{escape(r['speaker'])}</td><td>{comp}</td><td>{res}</td><td>{balance}</td></tr>\n")
    parts.append("</tbody></table></div>\n")
    return "".join(parts)


def tof_html(metadata):
    """Section 3 time-of-flight table (measured input and calculated output)."""
    rows = tof_table(metadata)
    if not rows:
        return ""
    parts = [
        '<div class="filters-section">\n<h3>Time of flight</h3>\n',
        ('<p class="epa-footer">After-DSP arrival adds the deployed chain delay '
        'to the measured input arrival; it is not a post-playback capture.</p>'),
        ('<table class="epa-table"><thead><tr><th>Monitor</th>'
        "<th>Measured arrival before DSP (ms)</th>"
        "<th>Applied delay (ms)</th>"
        "<th>Calculated arrival after DSP (ms)</th>"
        "<th>Offset vs reference (ms)</th>"
        "</tr></thead><tbody>\n"),
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
