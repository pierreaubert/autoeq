"""Render integrity-bound imported capture diagnostics, not a new playback verdict."""

from html import escape
import json
import math
from pathlib import Path

from .payload_binding import ALGORITHM, payload_digest


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

    def points(values):
        return " ".join(f"{40 + 720 * t / times[-1]:.3f},{20 + 180 * (hi - v) / span:.3f}"
                        for t, v in zip(times, values))

    return (f'<figure><figcaption>{escape(label or prefix.upper())}: {escape(units)}; blue baseline, orange candidate; '
            f'0–{times[-1]:.4g} ms; shared amplitude range {lo:.4g}–{hi:.4g}.</figcaption>'
            '<svg viewBox="0 0 800 220" role="img" aria-label="Matched capture traces">'
            f'<polyline fill="none" stroke="#2864b4" points="{points(before)}"/>'
            f'<polyline fill="none" stroke="#c56816" points="{points(after)}"/></svg></figure>')


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
        lo, hi = min(positive), max(positive)
        xspan, yspan = max(math.log(freqs[-1] / freqs[0]), 1e-12), max(hi - lo, 1e-12)
        parts.append('<figure><figcaption>One-sided pressure PSD: dB re (20 µPa)²/Hz; '
                     f'{freqs[0]:.4g}–{freqs[-1]:.4g} Hz, logarithmic frequency; '
                     f'{lo:.4g}–{hi:.4g} dB. Zero-power bins have no finite level.</figcaption>'
                     '<svg viewBox="0 0 800 220" role="img" aria-label="Calibrated ambient pressure PSD">')
        points = []
        for frequency, level in zip(freqs + [freqs[-1]], levels + [None]):
            if level is None:
                if points:
                    parts.append('<polyline fill="none" stroke="#2864b4" points="' + ' '.join(points) + '"/>')
                    points = []
            else:
                points.append(f'{40 + 720 * math.log(frequency / freqs[0]) / xspan:.3f},{20 + 180 * (hi - level) / yspan:.3f}')
        parts.append('</svg></figure>')
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


def capture_views_html(report):
    """Reject stale view bindings and preserve synthetic/acquisition distinctions."""
    parts = ['<section class="capture-views"><h1>Capture acceptance diagnostics</h1>',
             '<p>These plots do not establish safety, improved room damping, or listener benefit. '
             'Hashes check artifact consistency, not operator declarations or hardware authenticity.</p>']
    try:
        graph = report["graph_id"]
        comparisons = report["comparisons"]
        if not isinstance(graph, str) or not graph or not isinstance(comparisons, list) or not comparisons:
            raise ValueError("missing comparison identity or entries")
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
            noise = views.get("ambient_noise")
            if noise is not None:
                parts.append(_noise_html(noise, graph))
            parts.append('<ul>')
            for key, reason in sorted(views["unavailable"].items()):
                parts.append(f'<li>{escape(key)} unavailable: {escape(reason)}</li>')
            parts.append('</ul>')
        return "".join(parts) + '</section>'
    except (KeyError, ValueError, TypeError, AttributeError, OverflowError) as error:
        return '<section class="capture-views"><h1>Capture acceptance diagnostics</h1><p>Unavailable: ' + escape(str(error)) + '</p></section>'


def create_capture_report(source, destination):
    """Write a separate HTML diagnostic report without overwriting input or output."""
    report = json.loads(Path(source).read_text(encoding="utf-8"))
    content = '<!doctype html><html><meta charset="utf-8"><title>Capture diagnostics</title><body>' + capture_views_html(report) + '</body></html>'
    with Path(destination).open("x", encoding="utf-8") as handle:
        handle.write(content)
