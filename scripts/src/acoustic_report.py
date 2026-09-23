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
    "notch_db": {"green": lambda v: v > -10.0, "yellow": lambda v: v >= -20.0},
}

SUB_NAMES = {"sub", "lfe", "sub1", "sub2"}


def _finite(values):
    return [v for v in values if isinstance(v, (int, float)) and math.isfinite(v)]


def band_mean(freq, spl, lo_hz, hi_hz):
    """Mean SPL of curve points inside [lo_hz, hi_hz]."""
    vals = [s for f, s in zip(freq, spl) if lo_hz <= f <= hi_hz]
    vals = _finite(vals)
    if not vals:
        return None
    return sum(vals) / len(vals)


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
    """Relative level compensation per feat-report Section 2.

    Monitor reference band 0.5-3 kHz, sub band 30-80 Hz. The reference is
    the loudest monitor; each monitor compensation is its band mean minus
    the reference (so the loudest is 0.0). The sub is set to the monitor
    reference level. Residual after calibration is 0.0 by construction.
    """
    rows = []
    monitor_means = {}
    for name, ch in channels.items():
        if is_sub_channel(name):
            continue
        curve = (ch or {}).get("initial_curve")
        if not curve or not curve.get("freq"):
            continue
        mean = band_mean(curve["freq"], curve["spl"], 500.0, 3000.0)
        if mean is not None:
            monitor_means[name] = mean
    if not monitor_means:
        return []
    reference = max(monitor_means.values())
    for name, ch in channels.items():
        curve = (ch or {}).get("initial_curve")
        if not curve or not curve.get("freq"):
            rows.append({"speaker": str(name), "comp_db": None, "residual_db": None})
            continue
        if is_sub_channel(name):
            mean = band_mean(curve["freq"], curve["spl"], 30.0, 80.0)
            comp = (mean - reference) if mean is not None else None
        else:
            comp = monitor_means[name] - reference
        rows.append({
            "speaker": str(name),
            "comp_db": comp,
            "residual_db": 0.0 if comp is not None else None,
        })
    return rows


def deepest_notch_db(channel_data):
    """Deepest final-curve notch below 300 Hz vs the 0.5-3 kHz mean."""
    curve = (channel_data or {}).get("final_curve")
    if not curve or not curve.get("freq") or not curve.get("spl"):
        return None
    ref = band_mean(curve["freq"], curve["spl"], 500.0, 3000.0)
    lows = [s for f, s in zip(curve["freq"], curve["spl"]) if f < 300.0]
    lows = _finite(lows)
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

    Both curves are resampled by index onto curve A's grid; callers must
    only pass pairs sharing the optimizer grid (roomeq emits one shared
    grid per run). Returns None when grids differ in length.
    """
    if not curve_a or not curve_b:
        return None
    fa, sa = curve_a.get("freq"), curve_a.get("spl")
    fb, sb = curve_b.get("freq"), curve_b.get("spl")
    if not fa or not sb or len(fa) != len(fb) or len(sa) != len(sb):
        return None
    lin_sum, lin_diff = [], []
    for xa, xb in zip(sa, sb):
        pa = 10.0 ** (xa / 20.0)
        pb = 10.0 ** (xb / 20.0)
        lin_sum.append(pa + pb)
        lin_diff.append(abs(pa - pb))
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


def notch_cell(notch):
    """Colour-coded HTML cell for the deepest-notch column."""
    if notch is None:
        return '<td style="background:#eee;color:#888">n/a</td>'
    th = SUMMARY_THRESHOLDS["notch_db"]
    color = "#2ecc71" if th["green"](notch) else "#f1c40f" if th["yellow"](notch) else "#e74c3c"
    return f'<td style="background:{color}33">{notch:+.1f}</td>'


def pending_cell(field):
    return f'<td style="background:#eee;color:#888" title="needs roomeq field: {escape(field)}">pending</td>'


def summary_table_html(data):
    """Section 1 summary shell: live notch column, pending metric columns."""
    channels = data.get("channels", {}) or {}
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
            f"{pending_cell('summary.operational_room_response_pct')}"
            f"{pending_cell('early_reflections')}"
            f"{pending_cell('early_late_curves')}"
            f"{pending_cell('t60_octaves')}"
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
        "Reference: loudest monitor.</p>\n",
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
    """Section 3 time-of-flight table (before/after/applied)."""
    rows = tof_table(metadata)
    if not rows:
        return ""
    parts = [
        '<div class="filters-section">\n<h3>Section 3 — Time of flight</h3>\n',
        '<table class="epa-table"><thead><tr><th>Monitor</th>'
        "<th>Arrival before DSP (ms)</th>"
        "<th>Applied delay (ms)</th>"
        "<th>Arrival after DSP (ms)</th>"
        "<th>Offset vs reference (ms)</th>"
        "</tr></thead><tbody>\n",
    ]
    for r in rows:
        def _f(v):
            return f"{v:.3f}" if isinstance(v, (int, float)) and math.isfinite(v) else "n/a"
        parts.append(
            f"<tr><td>{escape(r['name'])}</td><td>{_f(r['before_ms'])}</td>"
            f"<td>{_f(r['applied_ms'])}</td><td>{_f(r['after_ms'])}</td>"
            f"<td>{_f(r['offset_ms'])}</td></tr>\n"
        )
    parts.append("</tbody></table></div>\n")
    return "".join(parts)
