"""Explain recorded RoomEQ decisions without inferring causes from response curves."""

from html import escape
import json
import math
from .payload_binding import verify_payload_binding


def _band(value):
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    if not all(isinstance(v, (int, float)) and not isinstance(v, bool)
               and math.isfinite(v) and v > 0 for v in value):
        return None
    return tuple(value) if value[0] < value[1] else None


def _band_label(value):
    return f"{value[0]:g}–{value[1]:g} Hz"


def _measurement_conditioning_history(stage):
    """Describe producer receipts without upgrading them into capture validation."""
    details = []
    names = {
        "source_overlap_alignment": "Aligned source grids within shared support",
        "source_spatial_power_rms": "Averaged spatial magnitudes in the power domain (no averaged phase)",
        "source_coherent_pressure_mean": "Computed a declared coherent pressure mean",
        "roomeq_dense_grid_conditioning": "Executed the dense-grid conditioning policy",
    }
    for check in stage.get("checks") or []:
        if not isinstance(check, dict) or check.get("passed") is not True:
            continue
        try:
            payload = json.loads(check.get("diagnostic") or "null")
        except (TypeError, ValueError):
            continue
        if not isinstance(payload, dict) or not isinstance(payload.get("receipt"), dict):
            continue
        entries = payload["receipt"].get("entries")
        if not isinstance(entries, list):
            continue
        channel = payload.get("channel", "unknown channel")
        if payload.get("source_snapshot_binding") == "verified_parsed_curve_snapshot":
            details.append(f"Measurement preparation for {channel}: native response identities match the frozen input snapshot in source order; this does not authenticate the original recording.")
        if not entries:
            details.append(f"Measurement preparation for {channel}: no numerical loading change recorded; this does not describe later processing.")
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            operation = entry.get("operation", "unknown operation")
            description = names.get(operation, f"Recorded operation: {operation}")
            parameters = entry.get("parameters")
            parameters = parameters if isinstance(parameters, dict) else {}
            band = _band(parameters.get("declared_valid_band_hz"))
            if band:
                description += f"; declared usable band {_band_label(band)}"
            if "input_bins" in parameters and "output_bins" in parameters:
                description += f"; {parameters['input_bins']} → {parameters['output_bins']} bins"
            details.append(f"Measurement preparation for {channel}: {description}.")
    if details:
        details.append("Measurement preparation is partial recorded history, not proof of capture/calibration validity or complete conditioning lineage.")
    return details


_K4_STATUS_TEXT = {
    "applied": "Applied",
    "already_acceptable": "Already acceptable (no correction needed)",
    "insufficient_evidence": "Insufficient evidence (no correction applied)",
    "outside_scope": "Outside scope (no correction applied)",
    "constrained": "Constrained (partial correction only; remainder withheld)",
    "reverted": "Reverted (attempted correction rolled back; not delivered)",
    "unresolved": "Unresolved (no decision reached)",
    "advisory": "Advisory nomination only (nothing applied or removed)",
}

_K4_ACTION_TEXT = {
    "equalize": "Equalize",
    "phase_correct": "Phase correction",
    "gain_adjust": "Gain adjustment",
    "reroute": "Reroute",
    "prune": "Prune",
}

_K4_REASON_TEXT = {
    "serialized_final_gain": "Gain present in the delivered DSP",
    "final_electrical_headroom": "Attenuation for the configured digital headroom ceiling and any declared physical-drive limits",
    "final_channel_level_alignment": "Final channel-level alignment",
    "branch_gain_not_acoustic_benefit_or_net_parallel_gain": (
        "Scalar gain on this input/output branch; not net parallel-path gain or measured acoustic benefit"
    ),
    "joint_array_proposal_reverted": "Joint subwoofer gain/delay proposal reverted",
    "joint_shared_eq_proposal_reverted": "Shared subwoofer EQ proposal reverted",
    "stage_history_not_final_route_acceptance": (
        "This is stage history, not approval of later routing or recorded playback"
    ),
}


def _finite_number(value):
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)) and math.isfinite(value):
        return value
    return None


def _format_quantity(value, unit):
    number = _finite_number(value)
    if number is None:
        return None
    text = f"{number:g} {unit}".strip()
    return text


def _quantities_text(items):
    parts = []
    for item in items or []:
        if not isinstance(item, dict):
            continue
        text = _format_quantity(item.get("value"), str(item.get("unit") or ""))
        name = item.get("name")
        if text is None or not isinstance(name, str) or not name.strip():
            continue
        parts.append(f"{name.strip()}: {text}")
    return "; ".join(parts)


def _k4_frequency_text(record: dict) -> str:
    band = _band(record.get("frequency_band_hz"))
    if band is not None:
        return _band_label(band)
    center = _finite_number(record.get("filter_center_hz"))
    if center is not None and center > 0:
        return f"{center:g} Hz (filter center)"
    return "Frequency unavailable"


def _k4_scope_text(record: dict) -> str:
    logical = record.get("logical_input") or "Unknown input"
    physical = record.get("physical_output") or "unknown output"
    seats = [str(seat) for seat in (record.get("seat_refs") or []) if str(seat).strip()]
    scope = f"{logical} / {physical}"
    if seats:
        scope += f" / seat {', '.join(seats)}"
    measurements = [str(ref) for ref in (record.get("measurement_refs") or []) if str(ref).strip()]
    if measurements:
        scope += f" (measured: {', '.join(measurements)})"
    return scope


def _k4_decision_text(record: dict) -> str:
    action = _K4_ACTION_TEXT.get(record.get("action"), "Recorded action unavailable")
    status = _K4_STATUS_TEXT.get(record.get("status"), "Recorded status unavailable")
    stage = record.get("stage")
    if stage == "final":
        return f"{action} — {status} (final)"
    return f"{action} — {status} (provisional; not a delivery claim)"


def _k4_reason_text(record: dict) -> str:
    codes = [str(code) for code in (record.get("reason_codes") or []) if str(code).strip()]
    labels = [_K4_REASON_TEXT.get(code, code) for code in codes]
    reason = f"Recorded reason: {', '.join(labels)}." if codes else "No reason recorded."
    confidence = record.get("confidence")
    reason += f" Confidence: {confidence}." if isinstance(confidence, str) and confidence else " Confidence: unknown."
    observed = _quantities_text(record.get("observed"))
    if observed:
        reason += f" Observed: {observed}."
    limits = _quantities_text(record.get("limits"))
    if limits:
        reason += f" Limit: {limits}."
    evidence = [str(ref) for ref in (record.get("evidence_refs") or []) if str(ref).strip()]
    reason += f" Evidence: {', '.join(evidence)}." if evidence else " Evidence: unavailable."
    for key, template in (("related_decision_ids", "Linked record: {}. Both the applied partial correction and its constrained remainder stay visible."),
                          ("supersedes_ids", "Supersedes: {}. The superseded attempt is not delivered correction.")):
        linked = [str(item) for item in (record.get(key) or []) if str(item).strip()]
        if linked:
            reason += " " + template.format(", ".join(linked))
    return reason


_K4_SET_STATUSES = {
    "applied": {"applied", "already_acceptable", "constrained"},
    "reverted": {"reverted"},
    "withheld": {"insufficient_evidence", "outside_scope", "unresolved", "advisory"},
}

_K4_SET_TEXT = {
    "applied": "applied delivery claims",
    "reverted": "reverted (rolled back; not delivered)",
    "provisional": "provisional history (not delivery claims)",
    "withheld": "withheld (final rows without delivery)",
}


def _k4_sets_summary(records: list, delivery_blocked: bool = False) -> str:
    """Group decision records into acceptance sets for the top summary.

    Applied delivery claims, reversions, provisional history, and withheld
    rows each render with counts, IDs, and recorded reasons. Malformed
    entries join provisional history as unverified, never as delivery.
    """
    sets: dict[str, list[str]] = {"applied": [], "reverted": [], "provisional": [], "withheld": []}
    reasons: dict[str, list[str]] = {"applied": [], "reverted": [], "provisional": [], "withheld": []}
    for record in records:
        if not isinstance(record, dict):
            sets["provisional"].append("unreadable entry")
            continue
        decision_id = record.get("decision_id")
        label = decision_id.strip() if isinstance(decision_id, str) and decision_id.strip() else "missing decision ID"
        codes = [str(code) for code in (record.get("reason_codes") or []) if str(code).strip()]
        if record.get("stage") != "final":
            bucket = "provisional"
        else:
            status = record.get("status")
            bucket = next((name for name, members in _K4_SET_STATUSES.items() if status in members), "withheld")
            if delivery_blocked and bucket == "applied":
                bucket = "withheld"
        sets[bucket].append(label)
        for code in codes:
            if code not in reasons[bucket]:
                reasons[bucket].append(code)
    parts = []
    for bucket in ("applied", "reverted", "provisional", "withheld"):
        ids = ", ".join(sets[bucket]) if sets[bucket] else "none"
        why = f" Reasons: {', '.join(reasons[bucket])}." if reasons[bucket] else " No reason recorded."
        parts.append(f"{len(sets[bucket])} {_K4_SET_TEXT[bucket]} ({ids}).{why}")
    return "Final acceptance sets: " + " ".join(parts)


def _k4_bucket_ids(records: list, bucket: str) -> list[str]:
    """Decision IDs in one acceptance set (applied/reverted/provisional/withheld)."""
    ids = []
    for record in records:
        if not isinstance(record, dict):
            if bucket == "provisional":
                ids.append("unreadable entry")
            continue
        decision_id = record.get("decision_id")
        label = decision_id.strip() if isinstance(decision_id, str) and decision_id.strip() else "missing decision ID"
        if record.get("stage") != "final":
            record_bucket = "provisional"
        else:
            status = record.get("status")
            record_bucket = next((name for name, members in _K4_SET_STATUSES.items() if status in members), "withheld")
        if record_bucket == bucket:
            ids.append(label)
    return ids


def _k4_section_html(data: dict) -> tuple[str, list[str]]:
    """Render K4 final decision records; provisional records stay history.

    Returns the section HTML and extra history lines for the details block.
    One row per record: rows are never merged, so per-seat differences and
    gaps stay visible. Malformed records render as unverified, never crash.
    """
    ledger = data.get("correction_decisions")
    if not isinstance(ledger, dict):
        return "", []
    decisions = ledger.get("decisions")
    if not isinstance(decisions, list) or not decisions:
        return "", []
    metadata = data.get("metadata") or {}
    acceptance = metadata.get("correction_acceptance") or {}
    delivery_blocked = (acceptance.get("decision") == "identity_fallback"
                        or acceptance.get("outcome") == "rejected")
    delivered = metadata.get("delivered_graph_identity")
    if not (isinstance(delivered, str) and delivered.strip()):
        delivered = None
    finals = [record for record in decisions
              if isinstance(record, dict) and record.get("stage") == "final"]
    identities = {str(record.get("final_graph_identity")).strip() for record in finals
                  if isinstance(record.get("final_graph_identity"), str)
                  and str(record.get("final_graph_identity")).strip()}
    disagree = len(identities) > 1
    payload_verified, payload_reason, payload_identity = verify_payload_binding(data)
    binding_invalid = (not payload_verified or disagree
                       or any(record.get("final_graph_identity") != payload_identity for record in finals)
                       or (delivered is not None and delivered != payload_identity))
    rows = []
    history = []
    for record in decisions:
        if not isinstance(record, dict):
            history.append("Unverified decision record: unreadable entry; not delivered correction.")
            continue
        decision_id = record.get("decision_id")
        if not (isinstance(decision_id, str) and decision_id.strip()):
            history.append("Unverified decision record: missing decision ID; not delivered correction.")
            continue
        scope = _k4_scope_text(record)
        frequency = _k4_frequency_text(record)
        decision = _k4_decision_text(record)
        reason = _k4_reason_text(record)
        if record.get("stage") != "final":
            history.append(
                f"Provisional record {decision_id} ({scope}; {frequency}; {decision}): {reason} "
                "Final reconciled records above take precedence.")
            continue
        identity = record.get("final_graph_identity")
        if delivery_blocked and record.get("status") in _K4_SET_STATUSES["applied"]:
            decision += " (not delivered correction)"
            reason += " Final rejection or identity fallback overrides this candidate record."
        verified = (payload_verified and identity == payload_identity
                    and isinstance(identity, str) and identity.strip()
                    and not disagree
                    and (delivered is None or identity.strip() == delivered.strip()))
        if not verified:
            decision += " (unverified; not a delivery claim)"
            reason += f" {payload_reason}"
            if not (isinstance(identity, str) and identity.strip()):
                reason += " Unverified: final record without a delivered-graph identity is not a delivery claim."
            elif disagree:
                reason += (" Unverified: final records disagree on the delivered-graph identity; "
                           "the explanation cannot be bound to one delivered graph.")
            elif (delivered is not None and identity != delivered) or payload_verified:
                reason += (" Unverified: record identity does not match the recorded delivered graph; "
                           "not displayed as applied.")
        rows.append("<tr>" + "".join(
            f'<td style="padding:8px;border-bottom:1px solid #ddd;vertical-align:top">{escape(str(cell))}</td>'
            for cell in (scope, frequency, decision, reason)
        ) + "</tr>")
    if not rows and not history:
        return "", []
    banner = (f"<p><strong>{escape(payload_reason)}</strong></p>")
    if disagree:
        banner += ("<p><strong>Final records disagree on the delivered-graph identity; "
                  "this explanation is unverified and nothing below is shown as delivered.</strong></p>")
    summary = f"<p><strong>{escape(_k4_sets_summary(decisions, delivery_blocked or binding_invalid))}</strong></p>"
    table = ('<h3>Recorded final correction decisions</h3>' + summary + banner)
    if rows:
        table += ('<div style="overflow-x:auto"><table style="width:100%;border-collapse:collapse;text-align:left">'
                  "<thead><tr><th>Channel / output / seat</th><th>Frequency</th><th>Decision</th><th>Why</th></tr></thead>"
                  f"<tbody>{''.join(rows)}</tbody></table></div>")
    else:
        table += ("<p>No final reconciled decisions recorded; provisional history below is not delivered correction.</p>")
    return table, history


def _verification_html(metadata: dict) -> str:
    """Render predicted/backend/acoustic and small-signal/dynamic checks separately."""
    comparisons = metadata.get("playback_comparisons") or metadata.get("verification") or []
    if not isinstance(comparisons, list) or not comparisons:
        return ""
    kind_text = {
        "stationary_ir": "acoustic capture",
        "spatial_magnitude": "acoustic magnitude capture (no timing reference)",
        "direct_sound": "acoustic direct-sound capture",
        "simulated_backend": "exported-backend simulation (not an acoustic recording)",
        "unknown": "uncharacterized capture",
    }
    items = []
    for comparison in comparisons:
        if not isinstance(comparison, dict):
            continue
        kind = kind_text.get(comparison.get("capture_kind"), "uncharacterized capture")
        state = comparison.get("processing_state") or "unstated signal level"
        baseline = comparison.get("baseline_graph_identity") or "identity missing"
        candidate = comparison.get("candidate_graph_identity") or "identity missing"
        stimulus = comparison.get("stimulus_hash") or "stimulus hash missing"
        assessed = (isinstance(comparison.get("baseline_graph_identity"), str)
                    and comparison.get("baseline_graph_identity").strip()
                    and isinstance(comparison.get("candidate_graph_identity"), str)
                    and comparison.get("candidate_graph_identity").strip()
                    and comparison.get("baseline_graph_identity") != comparison.get("candidate_graph_identity")
                    and isinstance(comparison.get("stimulus_hash"), str)
                    and comparison.get("stimulus_hash").strip())
        verdict = "recorded" if assessed else "unassessed: insufficient evidence, no promotion"
        source = comparison.get("source_id") or "unstated source"
        seats = comparison.get("seat_ids") or []
        seat_text = f" Seats: {', '.join(str(seat) for seat in seats)}." if seats else ""
        items.append(
            f"Playback check ({kind}; {state}): {verdict}. "
            f"Source {source}.{seat_text} Baseline {baseline}; candidate {candidate}; stimulus {stimulus}.")
    listening = metadata.get("listening_evidence")
    if isinstance(listening, dict):
        result = listening.get("result") or "unassessed"
        protocol = listening.get("protocol_id") or "unstated protocol"
        items.append(
            f"Recorded listening outcome: {result} (protocol {protocol}). "
            "A modeled perceptual score is not a listener result; an inconclusive or "
            "unassessed outcome is not listening benefit.")
    if not items:
        return ""
    return ('<h3>Recorded playback and listening verification</h3><ul>'
            + "".join(f"<li>{escape(item)}</li>" for item in items) + "</ul>")


def _output_loss_html(quality: dict) -> str:
    """Surface raw (unnormalized) output-loss evidence beside normalized plots."""
    entries = quality.get("useful_output") or []
    if not isinstance(entries, list) or not entries:
        return ""
    rows = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        scope = (f"{entry.get('logical_input') or 'Unknown input'} / "
                 f"{entry.get('partition') or 'unknown partition'} / seat {entry.get('seat_index', '?')}")
        mean = _format_quantity(entry.get("mean_level_change_db"), "dB")
        unexplained = _format_quantity(entry.get("unexplained_loss_rms_db"), "dB")
        worst = _format_quantity(entry.get("worst_unexplained_loss_db"), "dB")
        worst_hz = _finite_number(entry.get("worst_loss_frequency_hz"))
        detail = (f"Mean level change: {mean or 'unknown'}. "
                  f"Unexplained loss (RMS): {unexplained or 'unknown'}.")
        if worst is not None:
            detail += f" Worst unexplained loss: {worst}"
            detail += f" at {worst_hz:g} Hz." if worst_hz is not None else "."
        detail += " Raw output evidence; display normalization cannot hide this loss."
        rows.append("<tr>" + "".join(
            f'<td style="padding:8px;border-bottom:1px solid #ddd;vertical-align:top">{escape(str(cell))}</td>'
            for cell in (scope, detail)
        ) + "</tr>")
    if not rows:
        return ""
    return ('<h3>Recorded raw output loss (unnormalized)</h3>'
            '<div style="overflow-x:auto"><table style="width:100%;border-collapse:collapse;text-align:left">'
            "<thead><tr><th>Seat</th><th>Recorded loss</th></tr></thead>"
            f"<tbody>{''.join(rows)}</tbody></table></div>")


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
    if outcome in ("accepted", "unchanged") and data.get("correction_decisions"):
        verified, binding_reason, _ = verify_payload_binding(data)
        if not verified:
            final_message = f"Recorded outcome: {outcome}; not verified for this payload. {binding_reason}"
    # Final acceptance is authoritative even when stale candidate records
    # remain in a legacy ledger. A fallback cannot supersede itself.
    if acceptance.get("decision") == "identity_fallback":
        final_message = (
            "The final outcome is unchanged: it fell back to identity; no applied correction records "
            "can override that outcome. The fallback withheld correction; candidate "
            "records below are not delivered correction."
        )
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

    k4_section, k4_history = _k4_section_html(data)
    if not k4_section:
        row("System", "Individual decisions unavailable", "Reason unavailable",
            "No versioned decision ledger was recorded with this output. "
            "Detailed reasons for individual peaks and dips are unavailable.")
    verification_section = _verification_html(metadata)
    output_loss_section = _output_loss_html(quality)

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
    details.extend(k4_history)
    for violation in acceptance.get("violations") or []:
        details.append(f"Final acceptance constraint: {violation}")
    for stage in acceptance.get("reverted_stages") or []:
        details.append(f"Reverted stage: {stage}")
    for stage in metadata.get("stage_outcomes") or []:
        name = stage.get("stage", "unknown")
        status = stage.get("status", "unknown")
        if name == "measurement_input_conditioning":
            details.extend(_measurement_conditioning_history(stage))
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
        f"{k4_section}"
        f"{verification_section}"
        f"{output_loss_section}"
        f"{diagnostics}"
        "<p>Detailed reasons for individual peaks and dips are unavailable unless recorded above. "
        "An unchanged response alone does not identify a cancellation, poor measurement, or an already acceptable result.</p>"
        "</section>\n"
    )
