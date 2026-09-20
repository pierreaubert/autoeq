"""Explain recorded RoomEQ decisions without inferring causes from response curves."""

from html import escape
import math


def _band(value):
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    if not all(isinstance(v, (int, float)) and not isinstance(v, bool)
               and math.isfinite(v) and v > 0 for v in value):
        return None
    return tuple(value) if value[0] < value[1] else None


def _band_label(value):
    return f"{value[0]:g}–{value[1]:g} Hz"


def correction_explanation_html(data: dict, label: str = "") -> str:
    """Render final acceptance separately from scope and historical decisions.

    Missing evidence stays unknown. A configured correction band is permission
    to shape the response, not proof that every frequency was changed. Likewise,
    a stage's applied status cannot override final rejection or reversion.
    """
    metadata = data.get("metadata") or {}
    acceptance = metadata.get("correction_acceptance") or {}
    quality = acceptance.get("acoustic_quality") or {}
    optimizer = ((metadata.get("effective_config") or {}).get("optimizer") or {})
    policy = optimizer.get("correction_band") or {}
    requested = _band([policy.get("min_hz"), policy.get("max_hz")])
    recorded = _band(quality.get("correction_band_hz"))
    observation = _band([optimizer.get("min_freq"), optimizer.get("max_freq")])
    if optimizer.get("correction_band") is None:
        # Without an explicit narrower policy, RoomEQ uses min_freq..max_freq.
        requested = observation
    observation = _band(quality.get("evaluated_band_hz")) or observation
    rows = []

    def row(scope, band, action, reason):
        rows.append("<tr>" + "".join(
            f'<td style="padding:8px;border-bottom:1px solid #ddd;vertical-align:top">{escape(str(cell))}</td>'
            for cell in (scope, band, action, reason)
        ) + "</tr>")

    final_messages = {
        "accepted": "The final correction passed the recorded acceptance checks. This is not a listening-benefit claim.",
        "unchanged": "The final outcome is unchanged; earlier correction attempts are not proof of delivered correction.",
        "rejected": "The final correction was rejected. Earlier applied stages do not establish an accepted result.",
        "insufficient_evidence": "The available evidence was insufficient to accept the final correction.",
    }
    outcome = acceptance.get("outcome")
    if outcome == "accepted" and not (
        acceptance.get("accepted") is True and acceptance.get("decision") == "accepted"
    ):
        final_message = "The acceptance record is incomplete or inconsistent; final correction is not verified."
    else:
        final_message = final_messages.get(outcome, "No final correction decision was recorded.")
    row("System", "All assessed frequencies", "Final outcome", final_message)

    # Prefer realized scorecard scope, preserving the requested scope separately
    # when it differs. Neither scope alone proves a change was delivered.
    correction = recorded or requested
    if correction:
        row("System", _band_label(correction), "Correction scope",
            "Recorded active-correction band." if recorded else
            "The configuration permits response shaping in this band; it does not prove that every frequency was corrected.")
        if requested and recorded and requested != recorded:
            row("System", _band_label(requested), "Requested scope",
                "The requested correction band differs from the recorded active-correction band shown above.")
        if observation:
            for outside in ((observation[0], min(correction[0], observation[1])),
                            (max(correction[1], observation[0]), observation[1])):
                if _band(outside):
                    row("System", _band_label(outside), "Outside correction scope",
                        "Excluded from the active correction band. Routing, crossovers, gain and filter tails can still affect playback here.")
    else:
        row("System", "Frequency range unavailable", "Scope not recorded",
            "This output does not provide an explicit correction band. It is not inferred from the response curves.")

    for seat in quality.get("final_seats") or []:
        scope = f"{seat.get('logical_input', 'Unknown input')} / {seat.get('partition', 'unknown partition')} / seat {seat.get('seat_index', '?')}"
        for value in seat.get("unassessed_bands_hz") or []:
            band = _band(value)
            if band:
                row(scope, _band_label(band), "Not assessed",
                    "Final playback was not assessed in this band. This does not establish that correction was absent or unnecessary.")

    veto_reasons = {
        "SubJnd": "Below the configured level-difference proxy threshold.",
        "SubErbWidth": "Below the configured affected-width proxy threshold.",
        "HighQAboveGuard": "Exceeds the configured high-frequency Q guard.",
        "ModeProximityBan": "Flagged by the recorded mode-proximity rule.",
        "Audible": "Passed the configured veto rules; this is not proof of audibility.",
    }
    for channel, verdicts in sorted((metadata.get("audibility_veto") or {}).items()):
        for verdict in verdicts:
            frequency = verdict.get("center_hz")
            if not isinstance(frequency, (int, float)) or not math.isfinite(frequency) or frequency <= 0:
                continue
            decision = verdict.get("decision")
            action = "Removal nominated" if decision == "Remove" else "Recorded veto decision"
            action += " (stage history)" if verdict.get("enforced") else " (advisory)"
            assessment = verdict.get("acceptance") or {}
            reason = veto_reasons.get(verdict.get("reason"), f"Recorded reason: {verdict.get('reason', 'unknown')}.")
            reason += f" Confidence: {assessment.get('confidence', 'unknown')}."
            reason += " This filter-center record does not establish a frequency interval or the final exported filter set."
            row(channel, f"{frequency:g} Hz (filter center)", action, reason)

    details = []
    for violation in acceptance.get("violations") or []:
        details.append(f"Final acceptance constraint: {violation}")
    for stage in acceptance.get("reverted_stages") or []:
        details.append(f"Reverted stage: {stage}")
    for stage in metadata.get("stage_outcomes") or []:
        name = stage.get("stage", "unknown")
        status = stage.get("status", "unknown")
        for advisory in stage.get("advisories") or []:
            details.append(f"Stage {name} ({status}): {advisory}")
        for check in stage.get("checks") or []:
            if check.get("passed") is False:
                details.append(f"Stage {name} ({status}), {check.get('id', 'unknown check')}: {check.get('diagnostic') or 'Check failed; no explanation recorded.'}")
        if status in ("skipped", "degraded", "failed") and not stage.get("advisories") and not any(
            check.get("passed") is False for check in stage.get("checks") or []
        ):
            details.append(f"Stage {name}: {status}; no reason recorded.")
    diagnostics = ""
    if details:
        diagnostics = ('<details><summary>Recorded constraints and stage history</summary>'
                       '<p>Stage history may describe attempts that were later reverted. The final outcome above takes precedence.</p><ul>'
                       + "".join(f"<li>{escape(item)}</li>" for item in details)
                       + "</ul></details>")
    title = "Why this correction?" + (f" — {label}" if label else "")
    return (
        '<section class="correction-explanation" style="background:#fff;border:1px solid #ddd;border-radius:8px;padding:20px;margin:16px 0">'
        f"<h2>{escape(title)}</h2>"
        "<p>Recorded reasons for correction scope, limitations and the final outcome.</p>"
        '<div style="overflow-x:auto"><table style="width:100%;border-collapse:collapse;text-align:left">'
        "<thead><tr><th>Channel / scope</th><th>Frequency</th><th>Decision</th><th>Why</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table></div>"
        f"{diagnostics}"
        "<p>Detailed reasons for individual peaks and dips are unavailable unless recorded above. "
        "An unchanged response alone does not identify a cancellation, poor measurement, or an already acceptable result.</p>"
        "</section>\n"
    )
