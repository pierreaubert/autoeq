"""Render bound acceptance diagnostics without promoting predictions into captures."""

from html import escape
import math

from payload_binding import ALGORITHM, payload_digest, verify_payload_binding


def _magnitude_svg(view, label):
    """Use common log-frequency and dB axes for retained pre/post traces."""
    freq, pre, post = (view.get(key) for key in ("freqs", "pre_db", "post_db"))
    if not all(isinstance(values, list) for values in (freq, pre, post)):
        raise ValueError("magnitude traces are missing")
    if not 2 <= len(freq) == len(pre) == len(post) <= 100_000:
        raise ValueError("magnitude traces are empty, unaligned, or oversized")
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
           for values in (freq, pre, post) for value in values):
        raise ValueError("magnitude traces contain invalid values")
    if freq[0] <= 0 or any(a >= b for a, b in zip(freq, freq[1:])):
        raise ValueError("magnitude frequency axis is invalid")
    lo, hi = min(pre + post), max(pre + post)
    span = max(hi - lo, 1.0)
    if not math.isfinite(span):
        raise ValueError("magnitude range overflow")
    from wasm_report import axis, series, figure, embedded_figure
    return embedded_figure(figure(label, axis("Frequency (Hz)", "log"),
        axis("Magnitude (dB)"), series_list=[
            series("Before", freq, pre, color="#2864b4"),
            series("Retained prediction after", freq, post, color="#c56816")]))


def waveform_status_html(data, label=""):
    """Display recorded waveform availability, not verified acoustic acceptance."""
    title = "Waveform-view availability" + (f" — {label}" if label else "")
    start = f'<section class="waveform-status"><h2>{escape(title)}</h2>'

    def unavailable(reason):
        return start + f'<p>Unavailable: {escape(reason)}.</p></section>'

    if not isinstance(data, dict):
        return unavailable("malformed report data")
    metadata = data.get("metadata", {})
    if metadata is None:  # Legacy reports may omit metadata entirely.
        return ""
    if not isinstance(metadata, dict):
        return unavailable("malformed waveform metadata")
    outcomes = metadata.get("stage_outcomes", [])
    if not isinstance(outcomes, list) or any(not isinstance(stage, dict) for stage in outcomes):
        return unavailable("malformed stage outcomes")
    stages = [stage for stage in outcomes if stage.get("stage") == "waveform_views"]
    if not stages:
        return ""
    if len(stages) != 1:
        return unavailable("conflicting waveform diagnostics")
    channels = data.get("channels")
    if not isinstance(channels, dict) or any(
            not isinstance(name, str) or not isinstance(chain, dict)
            for name, chain in channels.items()):
        return unavailable("malformed waveform channel inventory")
    checks = stages[0].get("checks")
    if not isinstance(checks, list) or any(not isinstance(check, dict) for check in checks):
        return unavailable("malformed waveform checks")
    expected = {f"{view}:{channel}" for channel in channels for view in ("pre_ir", "post_ir")}
    ids = [check.get("id") for check in checks]
    if (any(not isinstance(identifier, str) for identifier in ids)
            or len(ids) != len(expected) or set(ids) != expected):
        return unavailable("waveform checks do not cover each channel and view exactly once")
    rows = []
    for check in sorted(checks, key=lambda check: check["id"]):
        view, _, channel = check["id"].partition(":")
        available = channels[channel].get(view) is not None
        passed = check.get("passed")
        if not isinstance(passed, bool) or available != passed:
            state = "Unavailable: recorded status and saved waveform disagree"
        elif passed:
            measured = (view == "pre_ir" and
                        (channels[channel].get("t60_octaves") or {}).get("basis") == "measured_room_ir")
            state = ("Measured room impulse response imported" if measured else
                     "Calculated impulse response saved — prediction after DSP" if view == "post_ir" else
                     "Impulse response reconstructed from the input frequency response")
        else:
            state = "Unavailable: " + str(check.get("diagnostic") or "no reason recorded")
        rows.append("<tr>" + "".join(f"<td>{escape(value)}</td>" for value in (channel, view, state)) + "</tr>")
    return (start + '<p>An impulse response shows amplitude over time. Before DSP uses the imported '
            'room IR when supplied; otherwise it is reconstructed from the input response. '
            'After DSP is a calculated prediction, not a new microphone measurement. '
            'Available means the waveform was saved and can be plotted.</p>'
            '<table><thead><tr><th>Channel</th><th>View</th><th>Status / reason</th></tr></thead><tbody>'
            + "".join(rows) + '</tbody></table></section>')


def acceptance_views_html(data, label=""):
    """Render verified production diagnostics or a precise unavailable state."""
    heading = "Acceptance diagnostics" + (f" — {label}" if label else "")
    start = f'<section class="acceptance-views"><h2>{escape(heading)}</h2>'
    try:
        evidence = (data.get("correction_decisions") or {}).get("acceptance_evidence")
        if evidence is None:
            return start + '<p>Unavailable: no production acceptance views were recorded (legacy or unfinalized output).</p></section>'
        verified, reason, graph_identity = verify_payload_binding(data)
        if not verified:
            raise ValueError(reason)
        if evidence.get("version") != "acceptance-evidence-v1":
            raise ValueError("unsupported acceptance-evidence version")
        payload, binding = evidence.get("payload"), evidence.get("binding") or {}
        if (binding.get("algorithm") != ALGORITHM or binding.get("graph_identity") != graph_identity
                or payload_digest(payload, graph_identity) != binding.get("sha256")):
            raise ValueError("acceptance-view payload or graph binding changed")
        channels = payload.get("channels")
        if not isinstance(channels, dict) or set(channels) != set(data.get("channels") or {}):
            raise ValueError("acceptance diagnostics do not cover the delivered channel set")
        parts = [start, '<p>Retained predictions and serialized DSP controls—not recorded playback, '
                 'physical safety certification, or demonstrated listener benefit. '
                 'Existing measurement conditioning is retained; additional smoothing cannot recover raw detail.</p>',
                 f'<p>Graph: {escape(graph_identity)}</p>']
        for channel, entry in sorted(channels.items()):
            if entry.get("source_id") != channel or entry.get("evidence_kind") != "retained_prediction":
                raise ValueError("unsupported source or evidence classification")
            parts.append(f'<h3>{escape(channel)}</h3>')
            parts.append(f'<p>Seat provenance: {escape(str(entry.get("seat_provenance", "unavailable")))}</p>')
            bundle = entry["bundle"]
            settings = bundle["settings"]
            parts.append(f'<p>Rate: {escape(str(settings["sample_rate_hz"]))} Hz; '
                         f'levels: {escape(settings["normalization"])}; '
                         f'window: {escape(settings["window"])}</p>')
            for key, title, view in [
                ("magnitude", "General magnitude (additional 1/12-octave smoothing)", bundle.get("magnitude")),
                ("fine_magnitude", "Fine magnitude (additional 1/24-octave smoothing)", entry.get("fine_magnitude")),
            ]:
                if view is not None:
                    if view.get("provenance", {}).get("graph_identity") != graph_identity:
                        raise ValueError(f"{key} graph provenance mismatch")
                    parts.append(_magnitude_svg(view, title))
            disposition = bundle.get("disposition")
            if disposition is not None:
                if disposition.get("graph_identity") != graph_identity:
                    raise ValueError("DSP disposition graph identity mismatch")
                parts.append('<p>Serialized DSP disposition (not calibrated output capability):</p><ul>')
                for controls in disposition.get("channels", []):
                    parts.append('<li>' + escape(f'{controls["channel"]}: gain {controls["gain_db"]} dB; '
                                                 f'delay {controls["delay_ms"]} ms; '
                                                 f'inverted {controls["inverted"]}') + '</li>')
                parts.append('</ul>')
            parts.append('<p>Unavailable evidence:</p><ul>')
            for view, reason in sorted(entry["unavailable"].items()):
                parts.append(f'<li>{escape(view)}: {escape(str(reason))}</li>')
            parts.append('</ul><p>No complete acoustic/headroom acceptance is established by these diagnostics.</p>')
        return "".join(parts) + '</section>'
    except (ValueError, TypeError, KeyError, AttributeError, OverflowError, RecursionError) as error:
        return start + f'<p>Unavailable: {escape(str(error))}</p></section>'
