"""Render measured-IR full STI results; all acoustic computation lives in math-rir."""

import math
from html import escape

OCTAVES = [125, 250, 500, 1000, 2000, 4000, 8000]
MODULATIONS = [0.63, 0.8, 1, 1.25, 1.6, 2, 2.5, 3.15, 4, 5, 6.3, 8, 10, 12.5]
METHOD = "iec_60268_16_2020_indirect_ir_only_v1"


def _finite(value, low, high):
    return (not isinstance(value, bool) and isinstance(value, (int, float))
            and math.isfinite(value) and low <= value <= high)


def speech_transmission_html(channel_data, label):
    """Return a REW-style modulation table for a native measured room capture."""
    title = f'<h3>Speech Transmission Index — {escape(str(label))}</h3>'
    report = (channel_data or {}).get("speech_transmission")
    valid = isinstance(report, dict)
    if valid:
        matrix = report.get("modulation_transfer")
        mti = report.get("mti")
        valid = (report.get("method") == METHOD
                 and report.get("basis") == "measured_room_ir"
                 and report.get("octave_centers_hz") == OCTAVES
                 and report.get("modulation_frequencies_hz") == MODULATIONS
                 and _finite(report.get("sti"), 0, 1)
                 and _finite(report.get("sample_rate_hz"), 16000 * math.sqrt(2) / 0.99, 192000)
                 and _finite(report.get("duration_s"), 0, float("inf"))
                 and report.get("duration_s", 0) > 0
                 and isinstance(mti, list) and len(mti) == 7
                 and all(_finite(value, 0, 1) for value in mti)
                 and isinstance(matrix, list) and len(matrix) == 14
                 and all(isinstance(row, list) and len(row) == 7
                         and all(_finite(value, 0, 1) for value in row)
                         for row in matrix))
    if not valid:
        return (title + '<p class="epa-footer">STI unavailable: requires a native measured '
                'room impulse response covering all seven octaves (125 Hz–8 kHz), '
                'with usable band energy and its complete decay. Synthesized EQ '
                'impulses do not supply this result.</p>')
    sti = report["sti"]
    rating = ("Bad" if sti < 0.3 else "Poor" if sti < 0.45 else "Fair"
              if sti < 0.6 else "Good" if sti < 0.75 else "Excellent")
    parts = [title, f'<p><strong>Full STI (IR-only): {sti:.3f} — {rating}</strong></p>',
             '<p>Indirect method using the IEC 60268-16:2020 full-STI model: '
             '14 modulation frequencies in seven octave bands. Native capture: '
             f'{report["sample_rate_hz"]:g} Hz, {report["duration_s"]:.3f} s.</p>',
             '<p class="epa-footer">No operational speech/noise levels, auditory masking '
             'or hearing-threshold corrections are applied. Arbitrary IR amplitude '
             'does not establish calibrated SPL. This result describes the measured '
             'capture, not a post-EQ prediction or IEC-certified instrument result. '
             'Verify full decay, measurement SNR and linear time-invariant behavior.</p>',
             '<div style="overflow-x:auto"><table class="summary-table">',
             '<caption>Modulation transfer m-values by octave band</caption>',
             '<thead><tr><th scope="col">Modulation (Hz)</th>']
    parts.extend(f'<th scope="col">{frequency:g} Hz</th>' for frequency in OCTAVES)
    parts.append('</tr></thead><tbody>')
    for frequency, row in zip(MODULATIONS, report["modulation_transfer"]):
        parts.append(f'<tr><th scope="row">{frequency:g}</th>')
        parts.extend(f'<td>{value:.3f}</td>' for value in row)
        parts.append('</tr>')
    parts.append('<tr><th scope="row">MTI</th>')
    parts.extend(f'<td><strong>{value:.3f}</strong></td>' for value in report["mti"])
    parts.append('</tr></tbody></table></div>')
    warnings = report.get("warnings") or []
    if isinstance(warnings, list):
        parts.extend(f'<p class="epa-footer">{escape(warning)}</p>'
                     for warning in warnings if isinstance(warning, str))
    return "".join(parts)
