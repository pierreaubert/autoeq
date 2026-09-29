"""Render integrity-bound imported capture diagnostics, not a new playback verdict."""

from html import escape
import json
import math
from pathlib import Path

from .payload_binding import ALGORITHM, payload_digest
from .wasm_report import axis, series, figure, embedded_figure, grid_figure


def _trace_svg(view, prefix, units="raw sample units", label=None):
    times, before, after = (view[key] for key in ("times_ms", f"pre_{prefix}", f"post_{prefix}"))
    if not all(isinstance(values, list) for values in (times, before, after)):
        raise ValueError("missing raw trace arrays")
    if not 2 <= len(times) == len(before) == len(after) <= 65_536:
        raise ValueError("unaligned or oversized raw traces")
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
           for values in (times, before, after) for v in values):
        raise ValueError("invalid raw trace value")
    if times[0] != 0 or any(a >= b for a, b in zip(times, times[1:])):
        raise ValueError("invalid common time axis")
    lo, hi = min(before + after), max(before + after)
    span = max(hi - lo, 1e-12)
    if not math.isfinite(span):
        raise ValueError("trace range overflow")

    return embedded_figure(figure(label or prefix.upper(), axis("Time (ms)", vmin=0, vmax=times[-1]),
        axis(units), series_list=[series("Baseline", times, before, color="#2864b4"),
                                 series("Candidate", times, after, color="#c56816")]))


def _noise_html(noise, graph):
    analysis = noise["analysis"]
    spectrum = analysis["spectrum"]
    if (analysis["method"] != "periodic_hann_welch_pressure_v1"
            or spectrum["provenance"]["graph_identity"] != graph
            or noise["evidence_kind"] not in ("synthetic_noise_capture", "operator_declared_recorded_noise_capture")):
        raise ValueError("noise graph, evidence kind, or analysis method mismatch")
    def finite(value):
        return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
    freqs, powers = analysis["bin_freqs_hz"], analysis["pressure_psd_pa2_per_hz"]
    if (not isinstance(freqs, list) or not isinstance(powers, list)
            or not 1 <= len(freqs) == len(powers) <= 32768
            or any(not finite(f) or f <= 0 for f in freqs)
            or any(a >= b for a, b in zip(freqs, freqs[1:]))
            or any(not finite(p) or p < 0 for p in powers)):
        raise ValueError("invalid calibrated noise PSD")
    parts = ['<h3>Calibrated ambient noise</h3>',
             f'<p>{escape(noise["evidence_kind"])}; {escape(noise["conditions"])}</p>',
             f'<p>{escape(analysis["scope"])}</p>',
             f'<p>Observation: {escape(str(analysis["duration_seconds"]))} s; '
             f'bin spacing: {escape(str(analysis["bin_spacing_hz"]))} Hz; '
             f'Hann ENBW: {escape(str(analysis["window_enbw_hz"]))} Hz. '
             'Bin spacing is not resolving power. No self-noise subtraction.</p>',
             '<details><summary>Numeric calibration and acquisition limitations</summary><pre>'
             + escape(json.dumps(analysis["calibration"], indent=2)) + '</pre></details>']
    levels = [10 * (math.log10(p) - math.log10((20e-6) ** 2)) if p > 0 else None for p in powers]
    positive = [level for level in levels if level is not None]
    if positive:
        parts.append(embedded_figure(figure("Calibrated ambient pressure PSD",
            axis("Frequency (Hz)", "log"), axis("dB re (20 µPa)²/Hz"),
            series_list=[series("Pressure PSD", freqs, levels, color="#2864b4")])))
    else:
        parts.append('<p>No finite noise SPL: all estimated supported powers are zero. This is not proof of a noiseless room.</p>')
    centers, db = spectrum["freqs"], spectrum["noise_spl_db"]
    edges, bins = analysis["octave_edges_hz"], analysis["octave_bin_centers_hz"]
    if (not all(isinstance(v, list) for v in (centers, db, edges, bins))
            or not len(centers) == len(db) == len(edges) == len(bins) <= 10
            or any(not finite(v) for v in centers + db)):
        raise ValueError("invalid noise octave arrays")
    parts.append('<table><caption>FFT-bin-summed nominal octave levels, unweighted dB SPL re 20 µPa; not certified octave filters</caption>'
                 '<tr><th>Center (Hz)</th><th>Nominal band (Hz)</th><th>Summed bin centers (Hz)</th><th>dB SPL</th></tr>')
    for center, level, edge, extent in zip(centers, db, edges, bins):
        if (center <= 0 or not all(isinstance(v, list) and len(v) == 2 and all(finite(x) for x in v)
                                   and 0 < v[0] < v[1] for v in (edge, extent))
                or not edge[0] <= extent[0] <= extent[1] < edge[1]):
            raise ValueError("invalid noise octave support")
        parts.append(f'<tr><td>{center:g}</td><td>{edge[0]:.4g}–{edge[1]:.4g}</td>'
                     f'<td>{extent[0]:.4g}–{extent[1]:.4g}</td><td>{level:.4g}</td></tr>')
    parts.append('</table>')
    for band, reason in sorted(analysis["unavailable_bands"].items()):
        parts.append(f'<p>Noise {escape(band)} Hz unavailable: {escape(reason)}</p>')
    return ''.join(parts)


def _octave_t60_html(view):
    """Render bound raw-capture octave fits without filling invalid bands."""
    centers = [63, 125, 250, 500, 1000, 2000, 4000, 8000, 16000]
    def finite(value):
        return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
    band = view.get("valid_band_hz")
    threshold = view.get("min_r2")
    if (view.get("method") != "octave_schroeder_t30_t20_v1"
            or not finite(threshold) or not 0 < threshold <= 1
            or not isinstance(band, list) or len(band) != 2
            or not all(finite(v) for v in band) or not 0 < band[0] < band[1]):
        raise ValueError("invalid octave T60 analysis contract")
    for key in ("pre", "post"):
        rows = view.get(key)
        if not isinstance(rows, list) or len(rows) != len(centers):
            raise ValueError("incomplete octave T60 bands")
        for center, row in zip(centers, rows):
            if not isinstance(row, dict) or row.get("centre_hz") != center:
                raise ValueError("invalid octave T60 center")
            valid, value, r2, reason = (row.get(name) for name in ("valid", "t60_s", "r2", "reason"))
            if not isinstance(valid, bool) or not finite(r2) or not 0 <= r2 <= 1:
                raise ValueError("invalid octave T60 fit quality")
            if valid:
                if (not finite(value) or value <= 0 or row.get("fit_range") not in ("T20", "T30")
                        or r2 < threshold or reason):
                    raise ValueError("invalid accepted octave T60 fit")
            elif value is not None or not isinstance(reason, str) or not reason:
                raise ValueError("invalid rejected octave T60 fit")
    parts = ['<h3>Capture IR octave T60 diagnostic</h3>',
             '<p>' + escape(view["scope"]) + '</p>',
             f'<p>Declared supported band: {band[0]:g}–{band[1]:g} Hz; '
             f'minimum fit R²: {threshold:g}. No invalid or EDT-only band is plotted; '
             'gaps are unavailable fits.</p>']
    values = [row["t60_s"] for key in ("pre", "post") for row in view[key] if row["valid"]]
    if values:
        from .figures import create_t60_octaves_figure
        plot = create_t60_octaves_figure("Baseline", view["pre"])
        other = create_t60_octaves_figure("Candidate", view["post"])
        if plot and other:
            other["figure"]["series"][0].update(name="Candidate", color="#c56816")
            plot["figure"]["series"][0]["name"] = "Baseline"
            plot["figure"]["series"].extend(other["figure"]["series"])
        if plot or other:
            parts.append(embedded_figure(plot or other))
    parts.append('<table><caption>Per-band T30/T20 estimates from matched IR captures</caption>'
                 '<tr><th>Hz</th><th>Baseline T60 (s)</th><th>Candidate T60 (s)</th>'
                 '<th>Baseline fit</th><th>Candidate fit</th></tr>')
    for center, pre, post in zip(centers, view["pre"], view["post"]):
        def cell(row):
            return f'{row["t60_s"]:.3f}' if row["valid"] else 'n/a: ' + escape(row["reason"])
        parts.append(f'<tr><td>{center}</td><td>{cell(pre)}</td><td>{cell(post)}</td>'
                     f'<td>{escape(str(pre.get("fit_range") or "n/a"))}; R²={pre["r2"]:.3f}</td>'
                     f'<td>{escape(str(post.get("fit_range") or "n/a"))}; R²={post["r2"]:.3f}</td></tr>')
    parts.append('</table>')
    return ''.join(parts)


def _room_mean_t60_html(groups):
    """Average accepted capture fits across distinct sources in one seat."""
    centers = [63, 125, 250, 500, 1000, 2000, 4000, 8000, 16000]
    parts = []
    for (seat, kind, _baseline, _stimulus, sample_rate, _settings, _band), captures in groups.items():
        sources = {source for source, _ in captures}
        if len(sources) < 2 or len(sources) != len(captures):
            continue
        parts.append(f'<h3>Capture room mean octave T60: {escape(seat)}</h3>'
                     f'<p>{escape(kind)}; {len(sources)} distinct sources with matching '
                     f'declared settings, {sample_rate:g} Hz sample rate, and capture band '
                     f'({escape(", ".join(sorted(sources)))}). Means include accepted T30/T20 '
                     'fits only. Counts vary by band. This advisory average does not '
                     'establish a passive-room damping change.</p>')
        averages = {}
        for side in ("pre", "post"):
            averages[side] = []
            for index in range(len(centers)):
                valid = [(source, view[side][index]["t60_s"]) for source, view in captures
                         if view[side][index]["valid"]]
                averages[side].append((math.fsum(value / len(valid) for _, value in valid), len(valid))
                                      if valid else (None, 0))
        values = [value for side in ("pre", "post") for value, _ in averages[side]
                  if value is not None]
        if values:
            from .figures import create_t60_octaves_figure
            plot = None
            for side, color in (("pre", "#2864b4"), ("post", "#c56816")):
                rows = [{"centre_hz": c, "t60_s": v} for c, (v, _) in zip(centers, averages[side])]
                candidate = create_t60_octaves_figure("Capture room mean", rows)
                if candidate:
                    candidate["figure"]["series"][0].update(name=side, color=color)
                    if plot is None: plot = candidate
                    else: plot["figure"]["series"].extend(candidate["figure"]["series"])
            if plot:
                parts.append(embedded_figure(plot))
        parts.append('<table><caption>Accepted source fits; n is the number of contributing '
                     'sources</caption><tr><th>Hz</th><th>Baseline mean (s)</th>'
                     '<th>Baseline n</th><th>Candidate mean (s)</th><th>Candidate n</th></tr>')
        for index, center in enumerate(centers):
            pre, pre_n = averages["pre"][index]
            post, post_n = averages["post"][index]
            pre_text = f'{pre:.3f}' if pre is not None else 'n/a'
            post_text = f'{post:.3f}' if post is not None else 'n/a'
            parts.append(f'<tr><td>{center}</td><td>{pre_text}</td><td>{pre_n}</td>'
                         f'<td>{post_text}</td><td>{post_n}</td></tr>')
        parts.append('</table>')
    return ''.join(parts)


def _early_reflections_html(view):
    """Render emitted band-limited reflection candidates from matched IRs."""
    def finite(value):
        return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
    if (view.get("method") != "bandlimited_early_reflection_table_v1"
            or view.get("band_hz") != [1000.0, 8000.0]
            or view.get("threshold_dbfs") != -15.0
            or not all(isinstance(view.get(key), int) and view[key] >= 0
                       for key in ("pre_direct_sample", "post_direct_sample"))):
        raise ValueError("invalid early reflection analysis contract")
    parts = ['<h3>Early reflection candidates (1–8 kHz)</h3>',
             '<p>' + escape(view["scope"]) + '</p>',
             '<p>' + escape(view["direct_reference"]) + '; '
             'reflections below −15 dB relative to direct are omitted. '
             f'Filtered direct samples: baseline {view["pre_direct_sample"]}, '
             f'candidate {view["post_direct_sample"]}; table times are relative '
             'to each direct peak.</p>']
    for key, label, color in (("pre", "Baseline", "#2864b4"),
                              ("post", "Candidate", "#c56816")):
        events = view.get(key)
        if not isinstance(events, list) or len(events) > 64:
            raise ValueError("invalid early reflection event list")
        last_time = 0.0
        for event in events:
            if not isinstance(event, dict):
                raise ValueError("invalid early reflection event")
            gain, time, distance, first_dip = (event.get(field) for field in
                ("gain_dbfs", "time_ms", "distance_cm", "first_dip_hz"))
            ripple = event.get("ripple_db")
            if (not all(finite(v) for v in (gain, time, distance, first_dip))
                    or not last_time < time <= 15.0 or gain < -15.0
                    or distance <= 0 or first_dip <= 0
                    or ripple is not None and (not finite(ripple) or ripple < 0)):
                raise ValueError("invalid early reflection values")
            last_time = time
        parts.append(f'<h4>{label} ({len(events)} candidates)</h4>')
        if events:
            xs, ys = [], []
            for event in events:
                xs.extend([event["time_ms"]] * 3)
                ys.extend([-40, event["gain_dbfs"], None])
            parts.append(embedded_figure(figure(label + ": early-reflection candidates",
                axis("Time after direct peak (ms)", vmin=0, vmax=15),
                axis("Level relative to direct (dB)", vmin=-40, vmax=3),
                series_list=[series(label, xs, ys, color=color)])))
        parts.append('<table><tr><th>Reflection</th><th>Gain (dB re direct)</th>'
                     '<th>Time (ms)</th><th>Extra path (cm)</th>'
                     '<th>First dip (Hz)</th><th>Comb ripple (dB p-p)</th></tr>')
        for number, event in enumerate(events, 1):
            ripple = f'{event["ripple_db"]:.2f}' if event["ripple_db"] is not None else 'unbounded'
            parts.append(f'<tr><td>{number}</td><td>{event["gain_dbfs"]:.2f}</td>'
                         f'<td>{event["time_ms"]:.2f}</td><td>{event["distance_cm"]:.1f}</td>'
                         f'<td>{event["first_dip_hz"]:.1f}</td><td>{ripple}</td></tr>')
        parts.append('</table>')
    return ''.join(parts)


def _early_late_html(view):
    """Render capture-bound band energies on each capture's shared reference."""
    def finite(value):
        return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)

    band = view.get("valid_band_hz")
    if (view.get("method") != "incoherent_band_energy"
            or view.get("reference") != "full_peak_band"
            or view.get("smoothing") != "third_octave"
            or view.get("split_ms") != 20.0
            or view.get("direct_reference") not in
            ("broadband envelope peak", "120 Hz lowpass envelope peak")
            or not isinstance(band, list) or len(band) != 2
            or not all(finite(value) for value in band) or not 0 < band[0] < band[1]
            or not isinstance(view.get("scope"), str) or not view["scope"]):
        raise ValueError("invalid capture early/late analysis contract")
    parts = ['<h3>Captured IR early and late energy</h3>',
             '<p>' + escape(view["scope"]) + '</p>',
             f'<p>20 ms split from {escape(view["direct_reference"])}; full, early, and '
             'late use one peak-band reference within each capture. Baseline and candidate '
             'levels have separate references.</p>']
    common_freq = None
    for side, label in (("pre", "Baseline"), ("post", "Candidate")):
        record = view.get(side)
        if (not isinstance(record, dict) or not isinstance(record.get("direct_sample"), int)
                or isinstance(record["direct_sample"], bool) or record["direct_sample"] < 0):
            raise ValueError("invalid capture early/late direct sample")
        curves = [record.get(key) for key in ("full", "early", "late")]
        if not all(isinstance(curve, dict) for curve in curves):
            raise ValueError("missing capture early/late curves")
        freq = curves[0].get("freq")
        if (not isinstance(freq, list) or not 2 <= len(freq) <= 21
                or any(not finite(value) or value <= 0 for value in freq)
                or any(a >= b for a, b in zip(freq, freq[1:]))
                or any(f / 2 ** (1/6) < band[0] - 1e-6
                       or f * 2 ** (1/6) > band[1] + 1e-6 for f in freq)
                or common_freq is not None and freq != common_freq):
            raise ValueError("invalid capture early/late band support")
        common_freq = freq
        for curve in curves:
            values = curve.get("spl")
            if (curve.get("freq") != freq or not isinstance(values, list)
                    or len(values) != len(freq)
                    or any(not finite(value) or not -120.01 <= value <= 0.01
                           for value in values)):
                raise ValueError("invalid capture early/late levels")
        parts.append(f'<h4>{label}</h4><p>Direct sample: {record["direct_sample"]}.</p>')
        parts.append(embedded_figure(figure(label + ": early vs late energy",
            axis("Frequency (Hz)", "log"), axis("Level vs full peak band (dB)"),
            series_list=[series(name, freq, curve["spl"], color=color)
                         for name, curve, color in zip(("Full", "Early", "Late"), curves,
                                                      ("#202020", "#2864b4", "#c56816"))])))
        centers = [(f, e, l) for f, e, l in zip(freq, curves[1]["spl"], curves[2]["spl"])
                   if 1_000 <= f <= 8_000]
        if freq[0] <= 1_000 and freq[-1] >= 8_000 and len(centers) >= 2:
            ratio = math.fsum((e - l) / len(centers) for _, e, l in centers)
            if math.isfinite(ratio):
                parts.append(f'<p>1–8 kHz mean early minus late: {ratio:+.2f} dB '
                             '(arithmetic mean of third-octave differences).</p>')
        else:
            parts.append('<p>1–8 kHz early/late ratio unavailable: full band support is missing.</p>')
    return ''.join(parts)


def resonance_color(freq):
    """Stable frequency colour shared by surfaces, decay curves and mode tables."""
    import colorsys
    ratio = max(0, min(1, math.log(max(freq, 10) / 10) / math.log(30)))
    rgb = colorsys.hsv_to_rgb(0.5 + ratio / 3, 0.65, 0.8)
    return "#" + "".join(f"{round(v * 255):02x}" for v in rgb)


def worst_resonances(resonances):
    """Longest fitted decays first, then strongest 60 ms peak; at most two."""
    return sorted(resonances, key=lambda r: (
        r["decay_time_s"] is not None, r["decay_time_s"] or 0, r["level_db"]),
        reverse=True)[:2]


def resonance_summary_html(data):
    """All exported pre-correction resonance candidates, grouped by speaker."""
    entries = []
    for name, channel in (data.get("channels") or {}).items():
        waterfall, decays = channel.get("waterfall"), channel.get("resonance_decays")
        if not optimization_waterfall_html(waterfall, decays):
            entries.append((name, None))
        else:
            entries.append((name, decays["decays"]))
    if not any(modes is not None for _, modes in entries):
        return ""
    count = max(1, max(len(modes or []) for _, modes in entries))
    parts = ['<h2>Room resonance summary — all speakers</h2><p>Pre-correction detections. '
             'Each cell gives frequency, 60 ms level relative to that speaker’s full-grid peak, '
             'and fitted decay time. Colours identify frequency, not severity. These candidates '
             'are not a geometric identification of room modes.</p><table><tr><th>Speaker</th>']
    parts.extend(f'<th>Mode {i + 1}</th>' for i in range(count))
    parts.append('</tr>')
    for name, modes in entries:
        parts.append(f'<tr><th>{escape(name)}</th>')
        if not modes:
            message = 'Unavailable' if modes is None else 'No candidates detected'
            parts.append(f'<td colspan="{count}">{message}</td>')
        else:
            for mode in sorted(modes, key=lambda r: r["freq_hz"]):
                decay = f'{mode["decay_time_s"]:.3f} s' if mode["decay_time_s"] is not None else 'fit unavailable'
                color = resonance_color(mode["freq_hz"])
                parts.append(f'<td style="background:{color}33;border-top:4px solid {color}">'
                             f'{mode["freq_hz"]:.1f} Hz<br>{mode["level_db"]:.2f} dB<br>{decay}</td>')
            parts.extend('<td>—</td>' for _ in range(count-len(modes)))
        parts.append('</tr>')
    parts.append('</table>')
    return ''.join(parts)


def _waterfall_wireframe_figure(times, freqs, rows, color, caption, aria_label, resonances=()):
    """Use a filled d3rs surface, retaining the function name for existing callers."""
    highlights = [(min(range(len(freqs)), key=lambda i: abs(freqs[i]-r["freq_hz"])),
                   f'{r["freq_hz"]:.1f} Hz', resonance_color(r["freq_hz"]))
                  for r in worst_resonances(resonances)]
    return embedded_figure(grid_figure(aria_label, freqs, times, rows,
                                      surface=True, highlights=highlights))


def _resonance_table_figures(resonances, freqs, times, rows, label, color):
    """Keep all detections in the table; plot the two longest fitted decays."""
    parts = ['<table><caption>Detected 60 ms peaks and fitted 20–200 ms decay</caption>'
             '<tr><th>Frequency (Hz)</th><th>Level (dB re own grid peak)</th>'
             '<th>Fitted decay (s)</th></tr>']
    for r in resonances:
        decay = f'{r["decay_time_s"]:.3g}' if r["decay_time_s"] is not None else 'unavailable'
        parts.append(f'<tr><td>{r["freq_hz"]:.3g}</td><td>{r["level_db"]:.2f}</td><td>{decay}</td></tr>')
    parts.append('</table>')
    curves = []
    for r in worst_resonances(resonances):
        closest = min(range(len(freqs)), key=lambda i: abs(freqs[i]-r["freq_hz"]))
        name = f'{r["freq_hz"]:.1f} Hz'
        if r["decay_time_s"] is not None:
            name += f' — T60 {r["decay_time_s"]:.3f} s'
        curves.append(series(name, times, [row[closest] for row in rows],
                             color=resonance_color(r["freq_hz"])))
    if curves:
        parts.append('<p>Highlighted: up to two longest fitted decays, then strongest 60 ms '
                     'peak. Curves retain the full-grid peak reference; no fitted intercept '
                     'is exported, so no extrapolated fit line is invented.</p>')
        parts.append(embedded_figure(figure(label + ": decay of highlighted resonances",
            axis("Time from broadband peak (ms)", vmin=0, vmax=500),
            axis("Level relative to full grid peak (dB)", vmin=-60, vmax=0),
            series_list=curves)))
    return ''.join(parts)


def _wavelet_heatmap_figure(freqs, times, rows, caption, aria_label):
    """Frequency runs horizontally, time vertically; show the early 15 ms."""
    # Stored CWT rows are frequency-major; renderer grids are time-major.
    section = grid_figure(aria_label, freqs, times, [list(row) for row in zip(*rows)])
    early_count = sum(-1 <= t <= 15 for t in times)
    return (f'<p>Three-cycle wavelet; first 15 ms ({early_count} exported time samples). '
            'Log-frequency x-axis, time y-axis. '
            'Colours retain the full-grid peak reference. Display cells reflect the '
            'exported sampling; zoom cannot restore omitted samples.</p>' + embedded_figure(section))


def _waterfall_html(view):
    """Render a bounded, separately normalized capture STFT diagnostic."""
    def finite(value):
        return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)

    band = view.get("valid_band_hz")
    if (view.get("method") != "hann_stft_waterfall_v1"
            or view.get("reference") != "each_full_grid_peak"
            or view.get("window_ms") != 32.0 or view.get("hop_ms") != 2.0
            or view.get("post_ms") != 500.0 or not isinstance(band, list) or len(band) != 2
            or not all(finite(v) for v in band) or not 0 < band[0] < band[1]):
        raise ValueError("invalid capture waterfall contract")
    parts = ['<h3>Captured IR waterfall and resonance decay</h3>',
             '<p>' + escape(view["scope"]) + '</p>',
             f'<p>Declared supported band: {band[0]:g}–{band[1]:g} Hz. '
             'Each grid is referenced to its own full-grid peak; colors and levels '
             'do not compare absolute output between captures.</p>']
    for side, label, color in (("pre", "Baseline", "#2864b4"),
                               ("post", "Candidate", "#c56816")):
        record = view.get(side)
        grid = record.get("grid") if isinstance(record, dict) else None
        if not isinstance(grid, dict):
            raise ValueError("missing capture waterfall grid")
        times, freqs, rows = (grid.get(key) for key in ("times_ms", "freqs_hz", "mags_db"))
        if (not all(isinstance(item, list) for item in (times, freqs, rows))
                or not 2 <= len(times) <= 100 or not 2 <= len(freqs) <= 64
                or len(rows) != len(times) or not all(isinstance(row, list)
                    and len(row) == len(freqs) for row in rows)
                or any(not finite(v) for v in times + freqs)
                or any(a >= b for a, b in zip(times, times[1:]))
                or any(a >= b for a, b in zip(freqs, freqs[1:]))
                or times[0] < -5.01 or times[-1] > 500.01
                or any(not band[0] <= f <= band[1] for f in freqs)
                or any(not finite(v) or v < -100.01 or v > 0.01 for row in rows for v in row)):
            raise ValueError("invalid capture waterfall samples")
        resonances = record.get("resonances")
        if not isinstance(resonances, list) or len(resonances) > 64:
            raise ValueError("invalid capture resonance list")
        for resonance in resonances:
            if not isinstance(resonance, dict):
                raise ValueError("invalid capture resonance")
            frequency, level, decay = (resonance.get(key) for key in
                                       ("freq_hz", "level_db", "decay_time_s"))
            if (not finite(frequency) or not freqs[0] <= frequency <= freqs[-1]
                    or not finite(level) or not -100.01 <= level <= 0.01
                    or decay is not None and (not finite(decay) or decay <= 0)):
                raise ValueError("invalid capture resonance values")
        parts.append(f'<h4>{label}</h4>')
        parts.append(_waterfall_wireframe_figure(
            times, freqs, rows, color,
            f'{label} waterfall: log frequency (Hz), time from broadband absolute '
            'peak (ms), relative STFT level (dB). Oblique grid is a visual projection.',
            'Captured IR waterfall', resonances))
        parts.append(_resonance_table_figures(resonances, freqs, times, rows, label, color))
    return ''.join(parts)


def _wavelet_html(view):
    """Render the bounded three-cycle capture CWT heatmap."""
    def finite(value):
        return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)

    band = view.get("valid_band_hz")
    if (view.get("method") != "complex_morlet_three_cycle_v1"
            or view.get("reference") != "each_full_grid_peak"
            or view.get("cycles") != 3.0 or view.get("freqs_per_octave") != 6.0
            or view.get("hop_ms") != 1.0 or view.get("display_range_db") != [-30.0, 0.0]
            or not isinstance(band, list) or len(band) != 2
            or not all(finite(v) for v in band) or not 0 < band[0] < band[1]):
        raise ValueError("invalid capture wavelet contract")
    parts = ['<h3>Captured IR three-cycle wavelet</h3>',
             '<p>' + escape(view["scope"]) + '</p>',
             '<p>Blue = −30 dB, red = 0 dB relative to each capture’s own full '
             'wavelet-grid peak. Colors do not compare absolute output levels.</p>']
    for side, label in (("pre", "Baseline"), ("post", "Candidate")):
        grid = view.get(side)
        if not isinstance(grid, dict):
            raise ValueError("missing capture wavelet grid")
        freqs, times, rows = (grid.get(key) for key in ("freqs_hz", "times_ms", "mags_db"))
        if (not all(isinstance(item, list) for item in (freqs, times, rows))
                or not 2 <= len(freqs) <= 64 or not 2 <= len(times) <= 100
                or len(rows) != len(freqs)
                or not all(isinstance(row, list) and len(row) == len(times) for row in rows)
                or any(not finite(v) for v in freqs + times)
                or any(a >= b for a, b in zip(freqs, freqs[1:]))
                or any(a >= b for a, b in zip(times, times[1:]))
                or any(not band[0] <= f <= band[1] for f in freqs)
                or times[0] < -5.01 or times[-1] > 500.01
                or any(not finite(v) or v < -30.01 or v > 0.01 for row in rows for v in row)):
            raise ValueError("invalid capture wavelet samples")
        parts.append(_wavelet_heatmap_figure(
            freqs, times, rows,
            f'{label}: time relative to broadband absolute peak (ms), log center '
            'frequency (Hz), and relative wavelet level (dB).',
            'Captured three-cycle wavelet heatmap'))
    return ''.join(parts)


def _optimization_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _optimization_band(band):
    return (isinstance(band, list) and len(band) == 2
            and all(_optimization_number(v) for v in band) and 0 < band[0] < band[1])


def _optimization_grid(times, freqs, rows, mag_floor_db):
    """Shared single-grid checks: bounded, strictly increasing axes, own-peak levels."""
    if not all(isinstance(item, list) for item in (times, freqs, rows)):
        return False
    if (not 2 <= len(times) <= 800 or not 2 <= len(freqs) <= 2049
            or any(not _optimization_number(v) for v in times + freqs)
            or any(a >= b for a, b in zip(times, times[1:]))
            or any(a >= b for a, b in zip(freqs, freqs[1:]))
            or times[0] < -5.01 or times[-1] > 500.01):
        return False
    if len(rows) != len(times):
        return False
    return all(isinstance(row, list) and len(row) == len(freqs)
               and all(_optimization_number(v) and mag_floor_db <= v <= 0.01 for v in row)
               for row in rows)


def optimization_waterfall_html(waterfall, decays):
    """Render a single-sided optimization waterfall plus resonance decays.

    Returns "" when the optimization contract does not hold; the caller
    keeps the pending placeholder instead of rendering a partial grid.
    """
    if not isinstance(waterfall, dict) or not isinstance(decays, dict):
        return ""
    if (waterfall.get("basis") != "measured_room_ir"
            or waterfall.get("method") != "hann_stft_waterfall_v1"
            or waterfall.get("reference") != "full_grid_peak"
            or waterfall.get("window_ms") != 32.0 or waterfall.get("hop_ms") != 2.0
            or waterfall.get("post_ms") != 500.0
            or not _optimization_band(waterfall.get("valid_band_hz"))):
        return ""
    band = waterfall.get("valid_band_hz")
    times, freqs, rows = (waterfall.get(key) for key in ("times_ms", "freqs_hz", "mags_db"))
    if (not _optimization_grid(times, freqs, rows, -100.01)
            or any(not band[0] <= f <= band[1] for f in freqs)
            or any(not _optimization_number(v) for v in times + freqs)):
        return ""
    resonances = decays.get("decays")
    if (decays.get("basis") != "measured_room_ir"
            or decays.get("method") != "hann_stft_waterfall_v1"
            or decays.get("reference") != "full_grid_peak"
            or decays.get("slice_ms") != 60.0
            or not isinstance(resonances, list) or len(resonances) > 64):
        return ""
    for resonance in resonances:
        if not isinstance(resonance, dict):
            return ""
        frequency, level, decay = (resonance.get(key) for key in
                                   ("freq_hz", "level_db", "decay_time_s"))
        if (not _optimization_number(frequency) or not freqs[0] <= frequency <= freqs[-1]
                or not _optimization_number(level) or not -100.01 <= level <= 0.01
                or decay is not None and (not _optimization_number(decay) or decay <= 0)):
            return ""
    scope = waterfall.get("scope")
    if not isinstance(scope, str) or not scope:
        return ""
    parts = ['<h3>Measured room IR waterfall and resonance decay</h3>',
             '<p>' + escape(scope) + '</p>',
             f'<p>Supported band: {band[0]:g}–{band[1]:g} Hz. The grid is referenced '
             'to its own full-grid peak; colors and levels do not compare absolute '
             'output between channels.</p>',
             '<h4>Pre-correction</h4>']
    parts.append(_waterfall_wireframe_figure(
        times, freqs, rows, "#2864b4",
        'Pre-correction waterfall: log frequency (Hz), time from broadband absolute '
        'peak (ms), relative STFT level (dB). Oblique grid is a visual projection.',
        'Measured room IR waterfall', resonances))
    parts.append(_resonance_table_figures(resonances, freqs, times, rows,
                                          "Pre-correction", "#2864b4"))
    return ''.join(parts)


def optimization_wavelet_html(wavelet):
    """Render a single-sided optimization three-cycle wavelet heatmap.

    Returns "" when the optimization contract does not hold; the caller
    keeps the pending placeholder instead of rendering a partial heatmap.
    """
    if not isinstance(wavelet, dict):
        return ""
    if (wavelet.get("basis") != "measured_room_ir"
            or wavelet.get("method") != "complex_morlet_three_cycle_v1"
            or wavelet.get("reference") != "full_grid_peak"
            or wavelet.get("cycles") != 3.0 or wavelet.get("freqs_per_octave") not in (6.0, 48.0)
            or wavelet.get("hop_ms") not in (0.1, 1.0) or wavelet.get("display_range_db") != [-30.0, 0.0]
            or not _optimization_band(wavelet.get("valid_band_hz"))):
        return ""
    band = wavelet.get("valid_band_hz")
    freqs, times, rows = (wavelet.get(key) for key in ("freqs_hz", "times_ms", "mags_db"))
    if (not isinstance(freqs, list) or not isinstance(times, list) or not isinstance(rows, list)
            or not 2 <= len(freqs) <= 2049 or not 2 <= len(times) <= 800
            or len(rows) != len(freqs)
            or not all(isinstance(row, list) and len(row) == len(times) for row in rows)
            or any(not _optimization_number(v) for v in freqs + times)
            or any(a >= b for a, b in zip(freqs, freqs[1:]))
            or any(a >= b for a, b in zip(times, times[1:]))
            or any(not band[0] <= f <= band[1] for f in freqs)
            or times[0] < -5.01 or times[-1] > 500.01
            or any(not _optimization_number(v) or v < -30.01 or v > 0.01
                   for row in rows for v in row)):
        return ""
    return ('<h3>Measured room IR three-cycle wavelet</h3>'
            '<p>Blue = −30 dB, red = 0 dB relative to the channel\u2019s own full '
            'wavelet-grid peak. Colors do not compare absolute output levels.</p>'
            '<p>Frequency runs left to right; time runs upward. Dense exports use '
            '0.1 ms early-time sampling through 15 ms and 1 ms later sampling; '
            'the recorded time coordinates remain authoritative.</p>'
            + _wavelet_heatmap_figure(
                freqs, times, rows,
                'Pre-correction: time relative to broadband absolute peak (ms), log '
                'center frequency (Hz), and relative wavelet level (dB).',
                'Measured room IR three-cycle wavelet heatmap'))


def capture_views_html(report):
    """Reject stale view bindings and preserve synthetic/acquisition distinctions."""
    parts = ['<section class="capture-views"><h1>Capture acceptance diagnostics</h1>',
             '<p>These plots do not establish safety, improved room damping, or listener benefit. '
             'Hashes check artifact consistency, not operator declarations or hardware authenticity.</p>']
    try:
        status = report.get("status")
        if status is not None:
            if status not in ("accepted", "unchanged", "rejected", "insufficient_evidence"):
                raise ValueError("unsupported verification status")
            count = report.get("verified_captures")
            exit_code = report.get("exit_code")
            if (isinstance(count, bool) or not isinstance(count, int) or count < 0
                    or isinstance(exit_code, bool) or not isinstance(exit_code, int)):
                raise ValueError("invalid verification capture count or exit code")
            parts.append(f'<p>Recorded capture verification status: <strong>{escape(status)}</strong>; '
                         f'{count} pre-result capture check(s); process exit code {exit_code}. '
                         'This is the report’s recorded verdict, not a new validation by the viewer.</p>')
            detail = report.get("detail")
            if detail is not None:
                if not isinstance(detail, str):
                    raise ValueError("invalid verification detail")
                parts.append(f'<p>Recorded detail: {escape(detail)}</p>')
        else:
            parts.append('<p>Recorded capture verification status unavailable in this legacy report.</p>')
        graph = report["graph_id"]
        comparisons = report["comparisons"]
        if not isinstance(graph, str) or not graph or not isinstance(comparisons, list) or not comparisons:
            raise ValueError("missing comparison identity or entries")
        t60_groups = {}
        for comparison in comparisons:
            views = comparison.get("capture_views")
            if views is None:
                parts.append('<p>Unavailable: legacy comparison has no capture views.</p>')
                continue
            binding = comparison.get("capture_views_binding") or {}
            if (binding.get("algorithm") != ALGORITHM or binding.get("graph_identity") != graph
                    or payload_digest(views, graph) != binding.get("sha256")
                    or views["candidate_graph_id"] != graph
                    or views["source"] != comparison["source"] or views["seat"] != comparison["seat"]):
                raise ValueError("capture view payload or graph/source/seat binding changed")
            kind = views["evidence_kind"]
            if kind not in ("unavailable", "synthetic_capture_pair", "operator_declared_recorded_capture_pair"):
                raise ValueError("unsupported capture evidence kind")
            if not all(isinstance(views.get(key), str) for key in
                       ("source", "seat", "baseline_graph_id", "scope")):
                raise ValueError("invalid capture identity or scope")
            parts.append(f'<h2>{escape(views["source"])} / {escape(views["seat"])}</h2>')
            parts.append(f'<p>Matched IR evidence: {escape(kind)}; baseline {escape(views["baseline_graph_id"])}; '
                         f'candidate {escape(graph)}</p><p>{escape(views["scope"])}</p>')
            view = views.get("ir_step")
            if view is not None:
                if view["provenance"]["graph_identity"] != graph:
                    raise ValueError("IR graph identity mismatch")
                parts.append(f'<p>Common sample-zero reference: {escape(view["common_reference"])}</p>')
                parts.append(_trace_svg(view, "ir"))
                parts.append(_trace_svg(view, "step"))
            etc = views.get("etc")
            if etc is not None:
                if etc["provenance"]["graph_identity"] != graph or etc["method"] != "hann_analytic_octave_linear_v1":
                    raise ValueError("ETC graph identity or analysis method mismatch")
                if (etc["window_ms"] != [0.0, 40.0] or len(etc["bands"]) != 4
                        or [band["center_hz"] for band in etc["bands"]] != [500.0, 1000.0, 2000.0, 4000.0]):
                    raise ValueError("ETC band/window contract mismatch")
                parts.append(f'<h3>Matched octave ETC</h3><p>{escape(etc["scope"])}</p>')
                parts.append(f'<p>Analysis half-support by band (ms): {escape(str(etc["filter_half_support_ms"]))}; '
                             f'display floor: {escape(str(etc["display_floor_db"]))} dB.</p>')
                for band in etc["bands"]:
                    trace = {"times_ms": band["times_ms"], "pre_etc": band["pre_db"], "post_etc": band["post_db"]}
                    parts.append(_trace_svg(trace, "etc", "dB relative to the same baseline-band peak",
                                            f'ETC {band["center_hz"]:g} Hz'))
            decay = views.get("decay")
            if decay is not None:
                if (decay["provenance"]["graph_identity"] != graph
                        or decay["method"] != "finite_window_octave_schroeder_v1"):
                    raise ValueError("decay graph identity or analysis method mismatch")
                if not isinstance(decay["bands"], list) or not decay["bands"]:
                    raise ValueError("missing decay bands")
                parts.append(f'<h3>Matched octave decay</h3><p>{escape(decay["scope"])}</p>')
                parts.append('<p>Finite-window T20 slope extrapolation, not passive-room RT. '
                             'Noise-window and fit budgets are operator declarations: '
                             + escape(str(decay["settings"])) + '</p>')
                for band in decay["bands"]:
                    center = band["center_hz"]
                    if isinstance(center, bool) or center not in (63, 125, 250, 500, 1000, 2000, 4000):
                        raise ValueError("unsupported decay octave")
                    trace = {"times_ms": band["times_ms"], "pre_decay": band["pre_db"], "post_decay": band["post_db"]}
                    parts.append(_trace_svg(trace, "decay", "dB relative to the same baseline-band integrated energy",
                                            f'Common-reference decay {center:g} Hz'))
                    if band["post_normalized_db"] is not None:
                        trace.update(pre_decay=band["pre_normalized_db"], post_decay=band["post_normalized_db"])
                        parts.append(_trace_svg(trace, "decay", "dB relative to each capture’s own initial band energy",
                                                f'Normalized decay {center:g} Hz'))
                    else:
                        parts.append('<p>Candidate normalized decay unavailable: no nonzero energy reference.</p>')
                    for prefix in ("pre", "post"):
                        fit = band[f"{prefix}_fit"]
                        if fit is None:
                            parts.append(f'<p>{prefix} fit unavailable: {escape(band[f"{prefix}_fit_unavailable"] or "unspecified")}</p>')
                        else:
                            seconds, r_squared = fit["extrapolated_60_db_seconds"], fit["r_squared"]
                            if (any(isinstance(v, bool) or not isinstance(v, (float, int)) or not math.isfinite(v)
                                    for v in (seconds, r_squared)) or seconds <= 0 or not 0 <= r_squared <= 1):
                                raise ValueError("invalid decay fit")
                            parts.append(f'<p>{prefix} finite-window extrapolation: {seconds:.4g} s; R²={r_squared:.4g}; '
                                         f'fit diagnostics: {escape(str(fit))}</p>')
                for band, reason in sorted(decay.get("unavailable_bands", {}).items()):
                    parts.append(f'<p>Decay {escape(str(band))} Hz unavailable: {escape(reason)}</p>')
            octave_t60 = views.get("octave_t60")
            if octave_t60 is not None:
                parts.append(_octave_t60_html(octave_t60))
                settings, band = views.get("settings"), views.get("valid_band_hz")
                stimulus = views.get("stimulus_hash")
                sample_rate = views.get("sample_rate_hz")
                if (kind != "unavailable" and isinstance(settings, dict) and isinstance(band, list)
                        and len(band) == 2 and all(isinstance(v, (int, float))
                        and not isinstance(v, bool) and math.isfinite(v) for v in band)
                        and 0 < band[0] < band[1] and band == octave_t60["valid_band_hz"]
                        and isinstance(stimulus, str) and stimulus
                        and isinstance(sample_rate, (int, float)) and not isinstance(sample_rate, bool)
                        and math.isfinite(sample_rate) and sample_rate > 0):
                    group_key = (views["seat"], kind, views["baseline_graph_id"], stimulus,
                                 sample_rate, json.dumps(settings, sort_keys=True, allow_nan=False),
                                 tuple(band))
                    t60_groups.setdefault(group_key, []).append((views["source"], octave_t60))
            reflections = views.get("early_reflections")
            if reflections is not None:
                parts.append(_early_reflections_html(reflections))
            early_late = views.get("early_late_curves")
            if early_late is not None:
                parts.append(_early_late_html(early_late))
            waterfall = views.get("waterfall")
            if waterfall is not None:
                parts.append(_waterfall_html(waterfall))
            wavelet = views.get("wavelet")
            if wavelet is not None:
                parts.append(_wavelet_html(wavelet))
            noise = views.get("ambient_noise")
            if noise is not None:
                parts.append(_noise_html(noise, graph))
            parts.append('<ul>')
            for key, reason in sorted(views["unavailable"].items()):
                parts.append(f'<li>{escape(key)} unavailable: {escape(reason)}</li>')
            parts.append('</ul>')
        parts.append(_room_mean_t60_html(t60_groups))
        return "".join(parts) + '</section>'
    except (KeyError, ValueError, TypeError, AttributeError, OverflowError) as error:
        return '<section class="capture-views"><h1>Capture acceptance diagnostics</h1><p>Unavailable: ' + escape(str(error)) + '</p></section>'


def create_capture_report(source, destination):
    """Write a separate HTML diagnostic report without overwriting input or output."""
    report = json.loads(Path(source).read_text(encoding="utf-8"))
    content = '<!doctype html><html><meta charset="utf-8"><title>Capture diagnostics</title><body>' + capture_views_html(report) + '</body></html>'
    with Path(destination).open("x", encoding="utf-8") as handle:
        handle.write(content)
