"""HTML report generation for roomeq visualization."""

import datetime
import math
import re
from html import escape
from pathlib import Path

from . import wasm_report
from .acceptance_views import acceptance_views_html, waveform_status_html
from .acoustic_report import (
    early_reflection_figures,
    early_reflections_html,
    landmarks_table_html,
    level_compensation_html,
    response_landmarks,
    room_t60_rows,
    room_t60_table_html,
    smooth_curve,
    summary_table_html,
    symmetric_groups,
    t60_itu_note,
    t60_itu_reference,
    t60_rows,
    t60_table_html,
    tof_html,
)
from .capture_clock_views import capture_clock_qa_html, gated_lr_channel
from .capture_reflection_views import capture_reflections_html
from .capture_views import (
    capture_views_html,
    optimization_waterfall_html,
    optimization_wavelet_html,
    resonance_summary_html,
)
from .correction_explanation import correction_explanation_html
from .data_extract import (
    channel_has_eq,
    clip_curve_to_measured_band,
    display_channel_entries,
    driver_display_names,
    extract_eq_passes,
    get_channel_sort_key,
    get_plottable_drivers,
)
from .dsp import (
    build_post_dsp_source_curves,
    driver_destination_route,
    kautz_sections,
    per_driver_chain_plugins,
    per_driver_corrected_curve,
    per_driver_effective_eq,
    split_driver_eq_plugins,
    sum_driver_initial_curves,
    symmetric_complex_sum,
)
from .figures import (
    _mode_color,
    _mode_label,
    add_channel_response_overlays,
    create_bass_management_headroom_figure,
    create_bass_management_routing_figure,
    create_channel_figure,
    create_combined_figure,
    create_comparison_eq_overlay_figure,
    create_comparison_group_delay_figure,
    create_comparison_ir_figure,
    create_comparison_overlay_figure,
    create_comparison_phase_figure,
    create_comparison_zoomed_figure,
    create_early_late_figure,
    create_eq_figure,
    create_ir_figure,
    create_mode_subplots_figure,
    create_multipass_eq_figure,
    create_score_comparison_figure,
    create_symmetric_pair_figure,
    create_t60_octaves_figure,
)
from .loaders import RoomEqData
from .payload_binding import verify_payload_binding
from .signal_flow import signal_flow_sections
from .target_overlay import (
    TARGET_LEVEL_MATCH_BAND_HZ,
    build_target_overlay_curves,
    global_target_offset_for_pair,
    shift_target_to_reference_band_mean,
)


def _emit_html(sections: list[dict], html: str | None, tab: str | None = None,
             flat: bool = False) -> None:
    """Append an HTML fragment as a raw-HTML section (skips empties)."""
    if html:
        sections.append(wasm_report.html_section(html, tab=tab, flat=flat))


def _emit_fig(sections: list[dict], fig: dict | list | None) -> None:
    """Append a figure section (or list of sections); skips None entries."""
    if fig is None:
        return
    if isinstance(fig, list):
        sections.extend(s for s in fig if s is not None)
    else:
        sections.append(fig)

# Synthetic channel name used for the complex L+R sum tab in the
# comparison report. Picked so it cannot collide with a real recording
# channel id (those are short upper-case codes like L / R / LFE / Cs).
LR_SUM_CHANNEL = "L+R"


def _has_redirected_bass_route(data: dict, source_name: str) -> bool:
    """Return whether ``source_name`` feeds the physical subwoofer output."""
    bass_management = (data.get("metadata", {}) or {}).get("bass_management", {}) or {}
    graph = bass_management.get("routing_graph", {}) or {}
    return any(
        route.get("source_channel") == source_name
        and route.get("route_kind") == "redirected_bass_lowpass_to_sub"
        for route in (graph.get("routes", []) or [])
    )


def _comparison_source_label(channel_name: str, includes_redirected_bass: bool) -> str:
    """Describe comparison curves as logical sources, not physical outputs."""
    label = f"Logical source {channel_name}"
    if includes_redirected_bass:
        label += f" (physical {channel_name} + redirected bass)"
    return label


def _channel_display_final_curve(
    channel_name: str,
    physical_sub: str,
    channel_data: dict,
    post_dsp_curves: dict[str, dict],
) -> dict | None:
    """Select the logical response rendered on an individual channel tab.

    The physical-sub ``final_curve`` is the aggregate bass bus. It is not the
    response produced when only the LFE logical input is driven and may contain
    cancellations from unrelated redirected-main routes.
    """
    if channel_name == physical_sub:
        return post_dsp_curves.get(channel_name) or channel_data.get("final_curve")
    return channel_data.get("final_curve")


def _driver_shaping_summary_html(
    data: dict, channel_name: str, driver_index: int
) -> str:
    """Render the per-sub DSP chain summary for a driver tab.

    Lists the driver alignment (gain / delay / low-pass) plus the
    destination-matched route transfer, i.e. everything the tab's EQ
    plot combines on top of the shared channel EQ filters.
    """
    channel = (data.get("channels") or {}).get(channel_name) or {}
    drivers = get_plottable_drivers(channel)
    if driver_index < 0 or driver_index >= len(drivers):
        return ""
    driver = drivers[driver_index]

    gain_db = 0.0
    delay_ms = 0.0
    inverted = False
    low_pass_hz: float | None = None
    low_pass_type: str | None = None
    for plugin in driver.get("plugins") or []:
        if not isinstance(plugin, dict):
            continue
        params = plugin.get("parameters") or {}
        kind = str(plugin.get("plugin_type", "")).lower()
        if kind == "gain":
            try:
                gain_db += float(params.get("gain_db", 0.0))
            except (TypeError, ValueError):
                pass
            inverted = inverted or bool(params.get("invert", False))
        elif kind == "delay":
            try:
                delay_ms += float(params.get("delay_ms", 0.0))
            except (TypeError, ValueError):
                pass
        elif kind == "crossover" and str(params.get("output", "")).lower().startswith(
            "low"
        ):
            try:
                candidate = float(params.get("frequency", 0.0))
            except (TypeError, ValueError):
                continue
            if candidate > 0.0:
                low_pass_hz = candidate
                raw_type = params.get("type")
                low_pass_type = str(raw_type) if raw_type is not None else None

    parts = [f"driver gain {gain_db:+.1f} dB"]
    if inverted:
        parts.append("polarity inverted")
    if abs(delay_ms) > 1e-9:
        parts.append(f"delay {delay_ms:.3f} ms")
    if low_pass_hz is not None:
        lp_label = f" {low_pass_type}" if low_pass_type else ""
        parts.append(f"low-pass{lp_label} @ {low_pass_hz:.1f} Hz")

    route = driver_destination_route(data, str(driver.get("name")))
    route_gain: float | None = None
    route_lp: float | None = None
    if route is not None and route.get("route_kind") == "lfe_lowpass_to_sub":
        try:
            route_gain = float(route.get("gain_db", 0.0))
        except (TypeError, ValueError):
            route_gain = None
        low = route.get("low_pass_hz")
        try:
            route_lp = float(low) if low is not None else None
        except (TypeError, ValueError):
            route_lp = None
    if route_gain is not None:
        parts.append(f"LFE route gain {route_gain:+.1f} dB")
    if route_lp is not None:
        parts.append(f"LFE route low-pass @ {route_lp:.1f} Hz")

    return (
        '<div class="filters-section">\n'
        "    <h3>Sub DSP Chain</h3>\n"
        f'    <div class="filter-list">{escape("; ".join(parts))}</div>\n'
        "</div>\n"
    )


def _format_eq_filter_line(filt: dict, number: int) -> str:
    """Render one numbered EQ filter row."""
    if filt.get("topology") == "kautz_filter":
        # Parameter display only; the transfer evaluator validates actual Nyquist.
        sections = kautz_sections(filt, math.inf)
        entries = "; ".join(
            f"section {i}: pole={pole:.1f} Hz, Q={q:.2f}, weight={gain:+.6g} (linear)"
            for i, (pole, q, gain) in enumerate(sections, 1)
        )
        return f"Filter {number}: KAUTZ bank (unity dry path); {entries}<br>\n"
    filter_type = filt.get("filter_type", "peak")
    if filt.get("topology") == "warped_biquad":
        filter_type = f"WARPED {filter_type}"
    freq = filt.get("freq", 0)
    q = filt.get("q", 1)
    gain = filt.get("db_gain", 0)
    return (
        f"Filter {number}: {escape(str(filter_type).upper())} @ {freq:.1f} Hz, "
        f"Q={q:.2f}, Gain={gain:+.1f} dB<br>\n"
    )


def _eq_passes_list_html(passes: list[dict]) -> str:
    """Render EQ passes, grouped by pass label when labels are present.

    Numbering restarts at 1 within each rendered group; callers splitting
    filters by origin render one group per origin.
    """
    parts: list[str] = []
    has_labeled = any(p["label"] for p in passes)
    if has_labeled:
        for p in passes:
            parts.append(
                f'\n                    <h4 style="color: {p["color"]}; '
                'margin-bottom: 5px;">'
                f"{p['display_name']}</h4>\n"
                '                    <div class="filter-list" '
                'style="margin-bottom: 10px;">\n'
            )
            for j, filt in enumerate(p["filters"], 1):
                parts.append(_format_eq_filter_line(filt, j))
            parts.append("                    </div>\n")
    else:
        parts.append('\n                    <div class="filter-list">\n')
        number = 0
        for p in passes:
            for filt in p["filters"]:
                number += 1
                parts.append(_format_eq_filter_line(filt, number))
        parts.append("                    </div>\n")
    return "".join(parts)


def _driver_eq_filters_html(
    data: dict, channel_name: str, driver_index: int, id_prefix: str
) -> str:
    """Render a driver tab's EQ filters split by ownership.

    A driver tab's deployed chain merges the physical driver's own EQ with
    the shared channel input-chain EQ. Listing that merge as one flat
    1..N+M list hides which PEQ lives where, so each origin gets its own
    inner tab (styled like the outer channel tabs) with a 1..N numbering.
    Returns an empty string when neither origin carries EQ filters,
    mirroring the legacy empty-EQ behavior.
    """
    split = split_driver_eq_plugins(data, channel_name, driver_index)
    if split is None:
        return ""
    driver_plugins, shared_plugins = split
    names = driver_display_names(data, channel_name)
    if 0 <= driver_index < len(names):
        driver_name = names[driver_index]
    else:
        drivers = get_plottable_drivers(
            (data.get("channels") or {}).get(channel_name) or {}
        )
        driver_name = (
            str(drivers[driver_index].get("name") or driver_index)
            if 0 <= driver_index < len(drivers)
            else str(driver_index)
        )
    driver_passes = extract_eq_passes({"plugins": driver_plugins})
    shared_passes = extract_eq_passes({"plugins": shared_plugins})
    driver_count = sum(len(p["filters"]) for p in driver_passes)
    shared_count = sum(len(p["filters"]) for p in shared_passes)
    if driver_count == 0 and shared_count == 0:
        return ""
    total = driver_count + shared_count
    safe_driver = escape(driver_name)
    safe_channel = escape(str(channel_name))
    parts = [
        ('\n                <div class="filters-section">\n'
        "                    <h3>EQ Filters</h3>\n"
        '                    <p class="epa-footer">'
        f"{total} PEQ filter(s) total: {driver_count} driver + "
        f"{shared_count} shared channel. Numbering restarts in each tab; "
        "the EQ plot shows the combined response.</p>\n")
    ]
    if driver_count > 0 and shared_count > 0:
        drv_id = f"{id_prefix}_drv"
        sh_id = f"{id_prefix}_sh"
        parts.append(
            '                    <div class="eq-tabs">\n'
            '                        <div class="eq-tab-header">\n'
            '                            <button class="eq-tab-btn active" '
            f'onclick="openEqTab(event, \'{drv_id}\')">'
            f"Driver: {safe_driver} ({driver_count})</button>\n"
            '                            <button class="eq-tab-btn" '
            f'onclick="openEqTab(event, \'{sh_id}\')">'
            f"Shared channel {safe_channel} ({shared_count})</button>\n"
            "                        </div>\n"
            f'                        <div id="{drv_id}" class="eq-tab-panel active">\n'
            f"{_eq_passes_list_html(driver_passes)}"
            "                        </div>\n"
            f'                        <div id="{sh_id}" class="eq-tab-panel">\n'
            f"{_eq_passes_list_html(shared_passes)}"
            "                        </div>\n"
            "                    </div>\n"
        )
    else:
        origin = f"Driver: {safe_driver}" if driver_count > 0 else f"Shared channel {safe_channel}"
        passes = driver_passes if driver_count > 0 else shared_passes
        parts.append(
            f"                    <h4>{origin} ({total})</h4>\n"
            f"{_eq_passes_list_html(passes)}"
        )
    parts.append("                </div>\n")
    return "".join(parts)


def _eq_filter_table_html(passes: list[dict]) -> str:
    """Render EQ passes as compact tables (one per pass when labeled)."""
    parts: list[str] = []
    has_labeled = any(p["label"] for p in passes)
    for p in passes:
        if not p["filters"]:
            continue
        if has_labeled:
            parts.append(
                f'            <h4 style="color: {p["color"]}; margin-bottom: 5px;">'
                f"{p['display_name']}</h4>\n"
            )
        parts.append(
            '            <table class="bm-table">\n'
            "                <thead><tr><th>#</th><th>Type</th><th>Freq</th>"
            "<th>Q</th><th>Gain</th></tr></thead>\n"
            "                <tbody>\n"
        )
        for j, filt in enumerate(p["filters"], 1):
            if filt.get("topology") in ("kautz_filter", "warped_biquad"):
                parts.append(
                    f'<tr><td>{j}</td><td colspan="4">'
                    + _format_eq_filter_line(filt, j) + '</td></tr>\n'
                )
                continue
            filter_type = str(filt.get("filter_type", "peak")).upper()
            freq = filt.get("freq", 0)
            q = filt.get("q", 1)
            gain = filt.get("db_gain", 0)
            freq_str = f"{freq:.1f} Hz" if isinstance(freq, (int, float)) else "-"
            q_str = f"{q:.2f}" if isinstance(q, (int, float)) else "-"
            gain_str = f"{gain:+.1f} dB" if isinstance(gain, (int, float)) else "-"
            parts.append(
                f"                    <tr><td>{j}</td><td>{filter_type}</td>"
                f"<td>{freq_str}</td><td>{q_str}</td><td>{gain_str}</td></tr>\n"
            )
        parts.append("                </tbody>\n            </table>\n")
    return "".join(parts)


def _gain_plugins_html(data: dict) -> str:
    """Expose level trims that are not visible in the EQ-only overview."""
    owners = [("Global", data.get("global_plugins") or [])]
    channels = data.get("channels") or {}
    for name in sorted(channels, key=get_channel_sort_key):
        channel = channels[name] or {}
        owners.append((f"Channel {name}", channel.get("plugins") or []))
        for index, driver in enumerate(channel.get("drivers") or []):
            label = driver.get("name") or str(index)
            owners.append((f"Channel {name} / driver {label}", driver.get("plugins") or []))
    rows = []
    for owner, plugins in owners:
        for index, plugin in enumerate(plugins):
            if plugin.get("plugin_type") != "gain":
                continue
            params = plugin.get("parameters") or {}
            gain = params.get("gain_db", 0.0)
            gain_label = f"{gain:+.3f} dB" if isinstance(gain, (int, float)) else "invalid"
            values = [owner, str(index + 1), str(params.get("room_eq_stage", "unspecified")),
                      str(params.get("label", "unlabelled gain")), gain_label]
            rows.append("<tr>" + "".join(f"<td>{escape(value)}</td>" for value in values) + "</tr>")
    if not rows:
        return ""
    return (
        '<div class="plot-container"><h2>Level Gains and Safety Attenuation</h2>'
        '<p class="epa-footer">The EQ-only overview does not show these gain plugins. '
        'They affect the post-DSP response even when the EQ trace is flat. '
        'Entries are listed in plugin order within each owner; they are not a summed '
        'input-to-output gain. Routing gains and crossover responses also affect playback. '
        'Route-owned entries describe routing and must not be counted twice.</p>'
        '<table class="bm-table"><thead><tr><th>Owner</th><th>Plugin #</th>'
        '<th>Stage</th><th>Purpose</th><th>Gain</th></tr></thead><tbody>'
        + "".join(rows) + '</tbody></table></div>'
    )


def _all_eq_filters_html(data: dict) -> str:
    """Render every EQ filter list in one summary section.

    Mirrors the per-tab filter sections (driver tabs split by origin with a
    1..N numbering) so the whole correction is visible without switching
    tabs. Channels render as a button set: clicking a channel button shows
    that channel's EQ tables. Returns an empty string when no channel
    carries EQ filters.
    """
    channels = data.get("channels") or {}
    panels: list[tuple[str, int, str]] = []
    for entry in display_channel_entries(data):
        channel_name = entry["channel"]
        driver_index = entry["driver"]
        safe_label = escape(str(entry["label"]))
        groups: list[tuple[str, list[dict]]] = []
        if driver_index is not None:
            split = split_driver_eq_plugins(data, channel_name, driver_index)
            if split is None:
                continue
            names = driver_display_names(data, channel_name)
            driver_name = (
                names[driver_index]
                if 0 <= driver_index < len(names)
                else str(driver_index)
            )
            driver_passes = extract_eq_passes({"plugins": split[0]})
            shared_passes = extract_eq_passes({"plugins": split[1]})
            if any(p["filters"] for p in driver_passes):
                groups.append((f"Driver: {driver_name}", driver_passes))
            if any(p["filters"] for p in shared_passes):
                groups.append((f"Shared channel {channel_name}", shared_passes))
        else:
            passes = extract_eq_passes(channels.get(channel_name) or {})
            if any(p["filters"] for p in passes):
                groups.append((f"Channel {channel_name}", passes))
        if not groups:
            continue
        total = sum(sum(len(p["filters"]) for p in passes) for _, passes in groups)
        body = [f"            <h3>{safe_label}</h3>\n"]
        for origin, passes in groups:
            count = sum(len(p["filters"]) for p in passes)
            body.append(f"            <h4>{escape(origin)} ({count})</h4>\n")
            body.append(_eq_filter_table_html(passes))
        panels.append((safe_label, total, "".join(body)))
    if not panels:
        return ""
    parts = [
        ('        <div class="plot-container">\n'
        "            <h2>All EQ Filters</h2>\n"
        '            <p class="epa-footer">Complete PEQ listing for every '
        "channel and driver (same data and numbering as the per-tab "
        "sections below). Select a channel to show its filters.</p>\n"
        '            <div class="eq-tabs">\n'
        '                <div class="eq-tab-header">\n'),
    ]
    for index, (label, total, _) in enumerate(panels):
        active = " active" if index == 0 else ""
        parts.append(
            '                    <button class="eq-tab-btn'
            + active + '" '
            f'onclick="openEqTab(event, \'alleq_{index}\')">'
            f"{label} ({total})</button>\n"
        )
    parts.append("                </div>\n")
    for index, (_, _, body) in enumerate(panels):
        active = " active" if index == 0 else ""
        parts.append(
            f'                <div id="alleq_{index}" class="eq-tab-panel{active}">\n'
            f"{body}"
            "                </div>\n"
        )
    parts.append("            </div>\n        </div>\n")
    return "".join(parts)


def _crossover_config_html(data: dict) -> str:
    """Render the crossover configuration in one summary section.

    Covers the deployed crossover/band-split DSP plugins (per channel and
    per driver) plus the routing-graph crossover cutoffs, so the full
    crossover setup is visible without opening individual tabs. Returns an
    empty string when no crossover configuration exists.
    """
    channels = data.get("channels") or {}
    if not isinstance(channels, dict):
        return ""
    plugin_rows: list[str] = []
    for name in sorted(channels.keys(), key=get_channel_sort_key):
        channel = channels[name] or {}
        owners: list[tuple[str, list[dict]]] = [(f"Channel {name}", channel.get("plugins") or [])]
        names = driver_display_names(data, name)
        for index, driver in enumerate(get_plottable_drivers(channel)):
            label = (
                names[index]
                if 0 <= index < len(names)
                else str((driver or {}).get("name") or index)
            )
            owners.append((f"Driver {label}", (driver or {}).get("plugins") or []))
        for owner, plugins in owners:
            for plugin in plugins:
                if not isinstance(plugin, dict):
                    continue
                kind = str(plugin.get("plugin_type", "")).lower()
                if kind not in ("crossover", "band_split"):
                    continue
                params = plugin.get("parameters") or {}
                freq = params.get("frequency")
                freq_str = f"{freq:.1f} Hz" if isinstance(freq, (int, float)) else "-"
                if kind == "band_split":
                    type_str = str(params.get("type", "band_split"))
                    output_str = "-"
                else:
                    type_str = str(params.get("type", "-"))
                    output_str = str(params.get("output", "-"))
                stage = params.get("room_eq_stage")
                stage_str = str(stage) if stage is not None else "-"
                plugin_rows.append(
                    f"                    <tr><td>{escape(str(owner))}</td>"
                    f"<td>{escape(type_str)}</td><td>{escape(output_str)}</td>"
                    f"<td>{freq_str}</td><td>{escape(stage_str)}</td></tr>\n"
                )
    route_rows: list[str] = []
    bass_management = (data.get("metadata") or {}).get("bass_management", {}) or {}
    graph = bass_management.get("routing_graph", {}) or {}
    for route in graph.get("routes", []) or []:
        if not isinstance(route, dict):
            continue
        low = route.get("low_pass_hz")
        high = route.get("high_pass_hz")
        if low is None and high is None:
            continue
        low_str = f"{low:.1f} Hz" if isinstance(low, (int, float)) else "-"
        high_str = f"{high:.1f} Hz" if isinstance(high, (int, float)) else "-"
        crossover_type = route.get("crossover_type")
        type_str = str(crossover_type) if crossover_type is not None else "-"
        route_rows.append(
            f"                    <tr><td>{escape(str(route.get('source_channel', '-')))} "
            f"\u2192 {escape(str(route.get('destination', '-')))}</td>"
            f"<td>{escape(str(route.get('route_kind', '-')))}</td>"
            f"<td>{escape(type_str)}</td><td>{low_str}</td><td>{high_str}</td></tr>\n"
        )
    if not plugin_rows and not route_rows:
        return ""
    parts = [
        ('        <div class="plot-container">\n'
        "            <h2>Crossover Configuration</h2>\n")
    ]
    if plugin_rows:
        parts.append(
            "            <h3>Deployed Crossover DSP</h3>\n"
            '            <table class="bm-table">\n'
            "                <thead><tr><th>Owner</th><th>Type</th><th>Output</th>"
            "<th>Frequency</th><th>Stage</th></tr></thead>\n"
            "                <tbody>\n"
            + "".join(plugin_rows)
            + "                </tbody>\n            </table>\n"
        )
    if route_rows:
        parts.append(
            "            <h3>Routing-Graph Crossovers</h3>\n"
            '            <table class="bm-table">\n'
            "                <thead><tr><th>Route</th><th>Kind</th><th>Type</th>"
            "<th>Low-pass</th><th>High-pass</th></tr></thead>\n"
            "                <tbody>\n"
            + "".join(route_rows)
            + "                </tbody>\n            </table>\n"
        )
    parts.append("        </div>\n")
    return "".join(parts)


# Ordered EPA fields with their display labels and formatting rules.
# Used by the pre/post tables in both single-mode and comparison reports.
_EPA_FIELDS: list[tuple[str, str, str]] = [
    ("preference", "Preference", "{:+.2f}"),
    ("evaluation", "Evaluation", "{:+.2f}"),
    ("potency", "Potency", "{:+.2f}"),
    ("activity", "Activity", "{:+.2f}"),
    ("sharpness_acum", "Sharpness (acum)", "{:+.2f}"),
    ("roughness", "Roughness", "{:+.3f}"),
    ("total_loudness_sone", "Total loudness (sone)", "{:+.2f}"),
    ("loudness_balance", "Loudness balance", "{:+.3f}"),
]

# Fields where a larger value is the desired outcome (pre → post).
# Used to colour the delta column green/red.
_EPA_HIGHER_IS_BETTER: set[str] = {
    "preference",
    "evaluation",
    "total_loudness_sone",
    "loudness_balance",
}


def _format_delta(field: str, pre: float | None, post: float | None) -> str:
    """Return an HTML-colored span with the post-pre delta for an EPA field."""
    if pre is None or post is None:
        return '<span style="color:#999">-</span>'
    delta = post - pre
    if abs(delta) < 1e-9:
        return '<span style="color:#999">=</span>'
    higher_is_better = field in _EPA_HIGHER_IS_BETTER
    improved = (delta > 0) == higher_is_better
    color = "#2ecc71" if improved else "#e74c3c"
    return f'<span style="color:{color};font-weight:600">{delta:+.3f}</span>'


def _epa_channel_table_html(channel_epa: dict | None) -> str:
    """Render the pre/post EPA comparison table for a single channel.

    `channel_epa` is the `epa_per_channel[<channel>]` dict holding `pre` and
    `post` EpaScore objects. Returns an empty string if the data is missing
    or malformed so the caller can just append it unconditionally.
    """
    if not channel_epa:
        return ""
    pre = channel_epa.get("pre") or {}
    post = channel_epa.get("post") or {}
    if not pre and not post:
        return ""

    rows: list[str] = []
    for field, label, fmt in _EPA_FIELDS:
        pre_val = pre.get(field)
        post_val = post.get(field)
        pre_str = fmt.format(pre_val) if isinstance(pre_val, (int, float)) else "-"
        post_str = fmt.format(post_val) if isinstance(post_val, (int, float)) else "-"
        delta_html = _format_delta(field, pre_val, post_val)
        rows.append(
            f"                        <tr><td>{label}</td>"
            f"<td>{pre_str}</td><td>{post_str}</td>"
            f"<td>{delta_html}</td></tr>\n"
        )

    return (
        '                <div class="epa-section">\n'
        '                    <h3>EPA Psychoacoustic Scores</h3>\n'
        '                    <table class="epa-table">\n'
        "                        <thead><tr>"
        "<th>Metric</th><th>Before EQ</th><th>After EQ</th><th>Δ</th>"
        "</tr></thead>\n"
        "                        <tbody>\n"
        + "".join(rows)
        + "                        </tbody>\n"
        "                    </table>\n"
        '                    <p class="epa-footer">Higher is better for '
        "Preference / Evaluation / Total loudness / Loudness balance; "
        "lower is better for Activity / Sharpness deviation / Roughness. "
        "Δ cells are green when the change improves the metric.</p>\n"
        "                </div>\n"
    )


def _epa_comparison_table_html(
    ch_name: str,
    mode_datasets: list[tuple[str, dict]],
) -> str:
    """Render a multi-mode EPA comparison table for a single channel.

    Shows one column per processing mode with its post-EQ EPA values, plus
    a single "Before EQ" column taken from the first mode that has pre
    data (the input measurement is the same across modes in a typical
    comparison run).
    """
    # Collect (mode_name, pre_dict, post_dict) triples; skip modes without data.
    entries: list[tuple[str, dict, dict]] = []
    for mode_name, data in mode_datasets:
        ch_epa = (data.get("metadata") or {}).get("epa_per_channel", {}).get(ch_name)
        if not ch_epa:
            continue
        pre = ch_epa.get("pre") or {}
        post = ch_epa.get("post") or {}
        if not pre and not post:
            continue
        entries.append((mode_name, pre, post))

    if not entries:
        return ""

    # Use the first mode's pre-EQ values as the "Before EQ" baseline.
    _, baseline_pre, _ = entries[0]

    mode_header_cells = "".join(
        f"<th>{_mode_label(name)} (post)</th>" for name, _, _ in entries
    )
    header_row = f"<tr><th>Metric</th><th>Before EQ</th>{mode_header_cells}</tr>"

    body_rows: list[str] = []
    for field, label, fmt in _EPA_FIELDS:
        pre_val = baseline_pre.get(field)
        pre_str = fmt.format(pre_val) if isinstance(pre_val, (int, float)) else "-"
        mode_cells: list[str] = []
        for _, _, post in entries:
            post_val = post.get(field)
            if isinstance(post_val, (int, float)):
                post_str = fmt.format(post_val)
                if isinstance(pre_val, (int, float)):
                    delta_span = _format_delta(field, pre_val, post_val)
                    cell = f'<td>{post_str} <span style="font-size:0.85em">{delta_span}</span></td>'
                else:
                    cell = f"<td>{post_str}</td>"
            else:
                cell = '<td style="color:#999">-</td>'
            mode_cells.append(cell)
        body_rows.append(
            f"<tr><td>{label}</td><td>{pre_str}</td>{''.join(mode_cells)}</tr>"
        )

    return (
        '<div class="epa-section">\n'
        '    <h3>EPA Psychoacoustic Scores (by mode)</h3>\n'
        '    <table class="epa-table">\n'
        f"        <thead>{header_row}</thead>\n"
        "        <tbody>\n            "
        + "\n            ".join(body_rows)
        + "\n        </tbody>\n"
        "    </table>\n"
        '    <p class="epa-footer">Higher is better for Preference / '
        "Evaluation / Total loudness / Loudness balance; lower is better "
        "for Activity / Sharpness deviation / Roughness. Δ colours show "
        "whether each mode improved the metric versus the pre-EQ baseline.</p>\n"
        "</div>\n"
    )


def _epa_summary_pref(metadata: dict) -> tuple[float | None, float | None]:
    """Return the (pre, post) preference averaged across all channels.

    Returns `(None, None)` if `metadata.epa_per_channel` is absent or empty.
    """
    per_channel = metadata.get("epa_per_channel") or {}
    if not per_channel:
        return (None, None)
    pre_vals: list[float] = []
    post_vals: list[float] = []
    for entry in per_channel.values():
        pre = (entry or {}).get("pre") or {}
        post = (entry or {}).get("post") or {}
        if isinstance(pre.get("preference"), (int, float)):
            pre_vals.append(pre["preference"])
        if isinstance(post.get("preference"), (int, float)):
            post_vals.append(post["preference"])
    pre_avg = sum(pre_vals) / len(pre_vals) if pre_vals else None
    post_avg = sum(post_vals) / len(post_vals) if post_vals else None
    return (pre_avg, post_avg)


def _fmt_db(value: object, suffix: str = " dB") -> str:
    return f"{value:+.2f}{suffix}" if isinstance(value, (int, float)) else "-"


def _fmt_hz(value: object) -> str:
    return f"{value:.1f} Hz" if isinstance(value, (int, float)) else "-"


def _fmt_ms(value: object) -> str:
    return f"{value:.3f} ms" if isinstance(value, (int, float)) else "-"


def _eq_filter_counts(data: dict) -> list[int]:
    """Count emitted sections, including driver-local entries, per channel.

    Kautz banks contribute every basis section, including zero weights. Shared
    channel entries count once, not once per destination or driver. Route-owned
    descriptive markers are excluded consistently with physical-driver replay.
    """
    channels = data.get("channels") or {}
    if isinstance(channels, dict):
        channel_values = channels.values()
    elif isinstance(channels, list):
        channel_values = channels
    else:
        return []

    counts: list[int] = []
    for channel in channel_values:
        if not isinstance(channel, dict):
            continue
        total = 0
        owners = [channel] + [driver for driver in channel.get("drivers") or []
                              if isinstance(driver, dict)]
        plugins = [plugin for owner in owners for plugin in owner.get("plugins") or []]
        for plugin in plugins:
            if not isinstance(plugin, dict) or plugin.get("plugin_type") != "eq":
                continue
            params = plugin.get("parameters") or {}
            if params.get("room_eq_stage") == "route_owned":
                continue
            filters = params.get("filters") or plugin.get("filters") or []
            if isinstance(filters, list):
                total += sum(
                    len(kautz_sections(filt, math.inf))
                    if filt.get("topology") == "kautz_filter" else 1
                    for filt in filters
                )
        counts.append(total)
    return counts


def _eq_filter_summary_html(data: dict) -> str:
    counts = _eq_filter_counts(data)
    if not counts:
        return '<td style="color:#999">-</td>'
    if min(counts) == max(counts):
        return f"<td>{counts[0]}</td>"
    avg = sum(counts) / len(counts)
    return f"<td>{min(counts)}-{max(counts)} (avg {avg:.1f})</td>"


def _gd_summary_cells_html(metadata: dict) -> str:
    """Render comparison-table cells for metadata.group_delay."""
    gd = metadata.get("group_delay") or {}
    if not gd:
        return (
            '<td style="color:#999">-</td>'
            '<td style="color:#999">-</td>'
            '<td style="color:#999">-</td>'
            '<td style="color:#999">-</td>'
            '<td style="color:#999">-</td>'
        )

    advisory = str(gd.get("advisory", "-"))
    advisory_color = "#2ecc71" if advisory == "success" else "#b9770e"
    applied = gd.get("applied")
    applied_str = "yes" if applied is True else "no" if applied is False else "-"
    applied_color = "#2ecc71" if applied is True else "#999"
    pre = gd.get("sum_gd_pre_rms_ms")
    post = gd.get("sum_gd_post_rms_ms")
    rms_str = f"{_fmt_ms(pre)} → {_fmt_ms(post)}"
    improvement = gd.get("improvement_db")
    improvement_str = f"{improvement:+.2f} dB" if isinstance(improvement, (int, float)) else "-"
    improvement_color = (
        "#2ecc71" if isinstance(improvement, (int, float)) and improvement >= 0.0 else "#e74c3c"
    )
    ap_counts = gd.get("per_channel_ap_count") or []
    ap_total = sum(v for v in ap_counts if isinstance(v, int))
    mean_coh = gd.get("mean_coherence")
    coh_str = f", coh={mean_coh:.2f}" if isinstance(mean_coh, (int, float)) else ""
    ap_str = f"{ap_total}{coh_str}"

    return (
        f'<td style="color:{advisory_color};font-weight:600">{escape(advisory)}</td>'
        f'<td style="color:{applied_color};font-weight:600">{applied_str}</td>'
        f"<td>{rms_str}</td>"
        f'<td style="color:{improvement_color};font-weight:600">{improvement_str}</td>'
        f"<td>{ap_str}</td>"
    )


def _mixed_phase_summary_html(metadata: dict, channels: dict | list) -> str:
    """Render per-channel mixed-phase and FIR temporal evidence."""
    mixed_phase = metadata.get("mixed_phase_per_channel") or {}
    if isinstance(channels, dict):
        channel_map = channels
    elif isinstance(channels, list):
        channel_map = {
            str(channel.get("channel")): channel
            for channel in channels
            if isinstance(channel, dict) and channel.get("channel")
        }
    else:
        channel_map = {}

    temporal_by_channel = {
        name: (channel or {}).get("fir_temporal_masking")
        for name, channel in channel_map.items()
        if isinstance(channel, dict) and (channel or {}).get("fir_temporal_masking")
    }
    perceptual = metadata.get("perceptual_metrics") or {}
    has_global_fir_metrics = any(
        isinstance(perceptual.get(field), (int, float))
        for field in (
            "fir_pre_ringing_audible_db",
            "fir_post_ringing_audible_db",
            "fir_temporal_masking_penalty",
        )
    )
    if not mixed_phase and not temporal_by_channel and not has_global_fir_metrics:
        return ""

    def phase_range(report: dict) -> str:
        minimum = report.get("residual_excess_phase_min_deg")
        maximum = report.get("residual_excess_phase_max_deg")
        if isinstance(minimum, (int, float)) and isinstance(maximum, (int, float)):
            return f"{minimum:+.2f}° to {maximum:+.2f}°"
        return "-"

    def phase_rms(report: dict) -> str:
        value = report.get("residual_excess_phase_rms_deg")
        return f"{value:.2f}°" if isinstance(value, (int, float)) else "-"

    def ringing_pair(masking: dict, peak_field: str, audible_field: str) -> str:
        peak = masking.get(peak_field)
        audible = masking.get(audible_field)
        if isinstance(peak, (int, float)) and isinstance(audible, (int, float)):
            return f"{peak:+.2f} / {audible:+.2f} dB"
        if isinstance(peak, (int, float)):
            return f"{peak:+.2f} / - dB"
        if isinstance(audible, (int, float)):
            return f"- / {audible:+.2f} dB"
        return "-"

    rows: list[str] = []
    channel_names = sorted(
        set(mixed_phase) | set(temporal_by_channel), key=get_channel_sort_key
    )
    for channel_name in channel_names:
        phase = mixed_phase.get(channel_name) or {}
        masking = temporal_by_channel.get(channel_name) or {}
        taps = phase.get("fir_taps")
        taps_str = str(taps) if isinstance(taps, int) else "-"
        penalty = masking.get("penalty")
        penalty_str = f"{penalty:.3f}" if isinstance(penalty, (int, float)) else "-"
        rows.append(
            "<tr>"
            f"<td>{escape(str(channel_name))}</td>"
            f"<td>{_fmt_ms(phase.get('estimated_delay_ms'))}</td>"
            f"<td>{taps_str}</td>"
            f"<td>{phase_range(phase)}</td>"
            f"<td>{phase_rms(phase)}</td>"
            f"<td>{_fmt_ms(masking.get('main_time_ms'))}</td>"
            f"<td>{ringing_pair(masking, 'pre_ringing_peak_db', 'pre_ringing_audible_db')}</td>"
            f"<td>{ringing_pair(masking, 'post_ringing_peak_db', 'post_ringing_audible_db')}</td>"
            f"<td>{penalty_str}</td>"
            "</tr>\n"
        )

    pre = perceptual.get("fir_pre_ringing_audible_db")
    post = perceptual.get("fir_post_ringing_audible_db")
    penalty = perceptual.get("fir_temporal_masking_penalty")
    summary_parts: list[str] = []
    if isinstance(pre, (int, float)) or isinstance(post, (int, float)):
        summary_parts.append(
            "Worst audible pre/post ringing: "
            f"{pre:+.2f} dB" if isinstance(pre, (int, float)) else "Worst audible pre/post ringing: -"
        )
        summary_parts[-1] += (
            f" / {post:+.2f} dB" if isinstance(post, (int, float)) else " / -"
        )
    if isinstance(penalty, (int, float)):
        summary_parts.append(f"worst temporal masking penalty: {penalty:.3f}")
    summary_html = (
        f'<p class="epa-footer">{escape("; ".join(summary_parts))}.</p>\n'
        if summary_parts
        else ""
    )

    table_html = ""
    if rows:
        table_html = (
            '<table class="bm-table">\n'
            "<thead><tr><th>Channel</th><th>Estimated delay</th><th>FIR taps</th>"
            "<th>Residual phase range</th><th>Residual RMS</th><th>Main impulse</th>"
            "<th>Pre peak / audible</th><th>Post peak / audible</th><th>Penalty</th>"
            "</tr></thead>\n<tbody>\n"
            + "".join(rows)
            + "</tbody>\n</table>\n"
        )
    return (
        '<div class="filters-section mixed-phase-section">\n'
        "<h3>Mixed-Phase and FIR Timing</h3>\n"
        + table_html
        + summary_html
        + "</div>\n"
    )


def _bass_management_summary_html(report: dict) -> str:
    if not report:
        return ""

    routing = report.get("routing_graph") or {}
    routes = routing.get("routes") or []
    route_count = len(routes)
    physical_outputs = sorted(
        {
            str(route.get("destination"))
            for route in routes
            if route.get("route_kind")
            in {"redirected_bass_lowpass_to_sub", "lfe_lowpass_to_sub"}
            and route.get("destination")
        }
    )
    advisories: list[str] = []
    advisory = report.get("advisory")
    if advisory:
        advisories.append(str(advisory))
    advisories.extend(str(item) for item in routing.get("advisories") or [])
    advisories = sorted({item for item in advisories if item and item != "ok"})
    advisory_html = (
        f"<br><span style=\"color:#b36b00\">{escape('; '.join(advisories))}</span>"
        if advisories
        else ""
    )

    items = [
        ("Enabled", "yes" if report.get("enabled") else "no"),
        ("Crossover", f"{escape(str(report.get('crossover_type', '-')))} @ {_fmt_hz(report.get('crossover_frequency_hz'))}"),
        ("LFE gain", _fmt_db(report.get("lfe_playback_gain_db"))),
        ("Shared sub gain", _fmt_db(report.get("applied_sub_gain_db"))),
        ("Physical bass outputs", ", ".join(escape(name) for name in physical_outputs) or escape(str(report.get("physical_sub_output", "-")))),
        ("Route count", str(route_count)),
        ("Graph mode", "route branches" if route_count else "linear / none"),
    ]
    cells = "".join(
        f'<div class="metadata-item"><span class="metadata-label">{label}:</span> '
        f'<span class="metadata-value">{value}</span></div>\n'
        for label, value in items
    )
    return (
        '<div class="metadata bass-management-section">\n'
        "<h2>Bass Management</h2>\n"
        f'<div class="metadata-grid">{cells}</div>\n'
        f"{advisory_html}\n"
        "</div>\n"
    )


def _bass_management_groups_table_html(report: dict) -> str:
    groups = report.get("groups") or []
    if not groups:
        groups = ((report.get("optimization") or {}).get("group_results") or [])
    if not groups:
        return ""

    rows = []
    for group in groups:
        advisories = ", ".join(
            str(item) for item in group.get("advisories", []) if item != "ok"
        )
        rows.append(
            "<tr>"
            f"<td>{escape(str(group.get('group_id', '-')))}</td>"
            f"<td>{escape(', '.join(str(role) for role in group.get('roles', [])))}</td>"
            f"<td>{escape(str(group.get('crossover_type', '-')))}</td>"
            f"<td>{_fmt_hz(group.get('selected_crossover_hz'))}</td>"
            f"<td>{_fmt_ms(group.get('main_delay_ms'))}</td>"
            f"<td>{_fmt_ms(group.get('bass_route_delay_ms'))}</td>"
            f"<td>{'yes' if group.get('polarity_inverted') else 'no'}</td>"
            f"<td>{_fmt_db(group.get('trim_db'))}</td>"
            f"<td>{escape(advisories) if advisories else '-'}</td>"
            "</tr>\n"
        )

    return (
        '<div class="filters-section bass-management-section">\n'
        "<h3>Per-Speaker-Group Crossovers</h3>\n"
        '<table class="bm-table">\n'
        "<thead><tr><th>Group</th><th>Roles</th><th>Type</th><th>Selected XO</th>"
        "<th>Main delay</th><th>Bass delay</th><th>Invert bass</th><th>Trim</th><th>Advisories</th></tr></thead>\n"
        f"<tbody>{''.join(rows)}</tbody>\n"
        "</table>\n"
        "</div>\n"
    )


def _bass_management_sub_outputs_table_html(report: dict) -> str:
    outputs = report.get("sub_outputs") or []
    if not outputs:
        outputs = ((report.get("optimization") or {}).get("sub_output_results") or [])
    if not outputs:
        return ""

    rows = []
    for output in outputs:
        rows.append(
            "<tr>"
            f"<td>{escape(str(output.get('output_role', '-')))}</td>"
            f"<td>{escape(str(output.get('strategy_source', '-')))}</td>"
            f"<td>{_fmt_db(output.get('gain_db'))}</td>"
            f"<td>{_fmt_ms(output.get('delay_ms'))}</td>"
            f"<td>{'yes' if output.get('polarity_inverted') else 'no'}</td>"
            f"<td>{_fmt_db(output.get('headroom_contribution_db'))}</td>"
            "</tr>\n"
        )

    return (
        '<div class="filters-section bass-management-section">\n'
        "<h3>Physical Bass Outputs</h3>\n"
        '<table class="bm-table">\n'
        "<thead><tr><th>Output</th><th>Strategy</th><th>Gain</th><th>Delay</th>"
        "<th>Invert</th><th>Headroom contribution</th></tr></thead>\n"
        f"<tbody>{''.join(rows)}</tbody>\n"
        "</table>\n"
        "</div>\n"
    )


# Verdict severities shared by the three playback-status boxes.
_STATUS_OK = "#2ecc71"
_STATUS_WARN = "#f1c40f"
_STATUS_BAD = "#e74c3c"

# Top-level report buckets rendered as centered shell tabs. DSP analysis
# carries the correction rationale and the signal flow; Acoustics analysis
# carries every measured-room section; the psychoacoustic bucket is EPA only.
BUCKET_DSP = "DSP analysis"
BUCKET_ACOUSTICS = "Acoustics analysis"
BUCKET_PSYCHOACOUSTIC = "Psychoacoustic report"


def _workspace_roomeq_version() -> str | None:
    """Version of the `roomeq` binary built from this checkout.

    The report viewer ships in the same workspace, so
    `[workspace.package] version` in the root `Cargo.toml` is the
    report-generator version. DSP outputs predate producer stamping and
    carry no tool version; see `_roomeq_version`.
    """
    try:
        text = (wasm_report.REPO_ROOT / "Cargo.toml").read_text(encoding="utf-8")
    except OSError:
        return None
    package = text.split("[workspace.package]", 1)
    if len(package) != 2:
        return None
    match = re.search(r'(?m)^\s*version\s*=\s*"([^"]+)"',
                      package[1].split("[", 1)[0])
    if not match:
        return None
    return match.group(1).strip() or None


def _roomeq_version(data: dict) -> str | None:
    """RoomEQ version for the header: stamped producer version if present.

    Falls back to this checkout's workspace version, which is exact when
    the report is rendered from the same checkout that produced the DSP.
    """
    metadata = data.get("metadata") or {}
    for key in ("producer_version", "roomeq_version", "tool_version"):
        value = metadata.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    value = data.get("roomeq_version")
    if isinstance(value, str) and value.strip():
        return value.strip()
    return _workspace_roomeq_version()


def _report_provenance(datasets: list[tuple[str, dict]]) -> dict:
    """Header provenance: RoomEQ version, DSP data date, report render date."""
    versions, stamps = [], []
    for _, data in datasets:
        version = _roomeq_version(data)
        if version and version not in versions:
            versions.append(version)
        stamp = (data.get("metadata") or {}).get("timestamp")
        if isinstance(stamp, str) and stamp.strip() and stamp not in stamps:
            stamps.append(stamp.strip())
    provenance: dict = {}
    if versions:
        provenance["roomeq_version"] = " / ".join(versions)
    if stamps:
        provenance["data_timestamp"] = " / ".join(stamps)
    provenance["generated_at"] = (
        datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))
    return provenance


def _status_box_html(title: str, color: str, paragraphs: list[str]) -> str:
    """One verdict box; titles and paragraphs are escaped HTML."""
    return (f'<section class="playback-status-box" role="note" '
            f'style="flex:1;min-width:260px;border:2px solid {color};padding:16px">'
            f'<h2>{escape(title)}</h2>'
            + "".join(f"<p>{escape(detail)}</p>" for detail in paragraphs)
            + "</section>\n")


def _playback_status_html(metadata: dict, label: str = "", *, data: dict | None = None) -> str:
    """Three-box verdict: recorded eligibility, payload binding, playback approval.

    Each box carries its own severity so a stale payload no longer hides a
    recorded acceptance (or vice versa): eligibility reports the recorded
    acceptance verdict, binding reports whether the delivered bytes still
    match the decision ledger, and playback combines both into the final
    go/no-go plus its conditional input assumptions.
    """
    acceptance = metadata.get("correction_acceptance") or {}
    outcome = acceptance.get("outcome")
    eligible = outcome == "unchanged" or (
        outcome == "accepted" and acceptance.get("accepted") is True
        and acceptance.get("decision") == "accepted"
    )
    verified, binding_reason, _ = verify_payload_binding(data or {})
    approved = eligible and verified
    prefix = f"{label} — " if label else ""

    # Box 1: what RoomEQ recorded about this graph (green only on a clean,
    # self-consistent acceptance; yellow when there is no usable verdict).
    eligibility_title = prefix + "Saved DSP playback eligibility: " + str(outcome or "unverified")
    violations = [str(item) for item in acceptance.get("violations", [])
                  if isinstance(item, str) and item.strip()]
    if eligible and outcome in ("accepted", "unchanged"):
        eligibility_color = _STATUS_OK
        eligibility = [f"Recorded verdict: {outcome}."]
    elif outcome in ("accepted", "unchanged"):
        eligibility_color = _STATUS_BAD
        eligibility = [("Recorded verdict is contradictory: outcome "
                       f"'{outcome}' without matching accepted/decision flags.")]
    elif outcome == "rejected":
        eligibility_color = _STATUS_BAD
        eligibility = ["Recorded verdict: rejected."]
    elif outcome is None:
        eligibility_color = _STATUS_WARN
        eligibility = ["No recorded acceptance verdict (legacy or unverified output)."]
    else:
        eligibility_color = _STATUS_WARN
        eligibility = [f"Recorded verdict: {outcome} — no playback approval."]
    if violations:
        eligibility.append("Recorded violations: " + "; ".join(violations) + ".")

    # Box 2: whether the delivered bytes still match the recorded decisions.
    binding_title = prefix + ("Delivered payload: verified" if verified
                              else "Delivered payload: not verified")
    binding_color = _STATUS_OK if verified else _STATUS_WARN

    # Box 3: the final go/no-go with its conditional input assumptions.
    playback_title = prefix + "Playback approval"
    playback: list[str] = []
    if not approved:
        playback.append("Not approved for playback. These curves are diagnostic only.")
    else:
        playback.append("Approved for playback under the declared contract.")
        playback.append("Recorded validation is evidence for the saved graph and its declared playback contract, "
                       "not an independent replay by this report.")
    policy = ((metadata.get("effective_config") or {}).get("optimizer") or {}).get("finalization") or {}
    peaks = {"default": policy.get("default_input_peak", 1.0),
             **(policy.get("input_peak_limits") or {})}
    reduced = {name: peak for name, peak in peaks.items()
               if isinstance(peak, (int, float)) and math.isfinite(peak) and 0 < peak < 1}
    if reduced:
        values = ", ".join(f"{name}: {20 * math.log10(peak):.2f} dBFS"
                           for name, peak in sorted(reduced.items()))
        playback.append("Conditional on enforced input-peak ceilings (" + values + "). "
                       "This report does not enforce them; full-scale input safety is not established.")
    limited = [check["id"].split(":", 1)[1]
               for stage in metadata.get("stage_outcomes", [])
               for check in stage.get("checks", [])
               if check.get("id", "").startswith("runtime_limiter_physical_output:")]
    if limited:
        playback.append("Runtime limiter required on physical outputs: " + ", ".join(limited) + ". "
                       "Response curves describe small-signal playback below limiting. Loud bass peaks "
                       "may be reduced dynamically. Protection is sample-peak, not a true-peak or "
                       "loudspeaker-excursion guarantee. Do not bypass the limiter or add downstream gain.")
    playback_color = _STATUS_BAD if not approved else (
        _STATUS_WARN if reduced or limited else _STATUS_OK)

    return ('<section class="playback-status" role="group" aria-label="Playback verdict" '
            'style="display:flex;gap:12px;flex-wrap:wrap;margin:16px 0;border:0;padding:0">'
            + _status_box_html(eligibility_title, eligibility_color, eligibility)
            + _status_box_html(binding_title, binding_color, [binding_reason])
            + _status_box_html(playback_title, playback_color, playback)
            + "</section>\n")


def create_html_report(
    data: dict,
    output_path: Path,
    output_json_path: Path | None = None,
    smoothed_octaves: float = 1.0,
    capture_verification: dict | None = None,
) -> None:
    """Create an HTML report with all channel plots.

    Args:
        data: Output JSON data (roomeq result)
        output_path: Path to write HTML report
        output_json_path: Path to output JSON (for resolving relative paths)
        smoothed_octaves: Fractional-octave smoothing for the Section 2
            smoothed-response overlays (feat-report asks for 1 octave).
        capture_verification: Optional matched verification report. Its graph
            identity must match the verified saved optimization payload.
    """
    if output_json_path is not None and not isinstance(data, RoomEqData):
        data = RoomEqData(data, output_json_path.resolve().parent)
    channels_dict = data.get("channels", {})
    metadata = data.get("metadata", {})

    # Sort channels by classical order
    sorted_channel_names = sorted(channels_dict.keys(), key=get_channel_sort_key)
    channels = [(name, channels_dict[name]) for name in sorted_channel_names]

    page_title = "Room EQ"

    # Build HTML content
    # Sections in document order; the shell renders them and builds the
    # per-channel tab bar from the section `tab` fields.
    sections: list[dict] = []
    _emit_html(sections, _playback_status_html(metadata, data=data), flat=True)
    status_sections = sections
    why, overview, speakers, timing, symmetric, time_domain, epa = ([] for _ in range(7))
    landmark_figures = []
    sections = why
    _emit_html(sections, correction_explanation_html(data))
    _emit_html(sections, acceptance_views_html(data))
    _emit_html(sections, waveform_status_html(data))
    _emit_html(overview, summary_table_html(data))
    for name, channel in channels:
        marks = response_landmarks(channel)
        curve = smooth_curve(channel.get("initial_curve"), 1 / 3)
        if marks and curve:
            annotations = [wasm_report.annotation(f, level, f"{kind}: {f:.0f} Hz")
                           for kind in ("peaks", "notches") for f, level in marks[kind]]
            lines = ([wasm_report.vline(marks["lf_extension_hz"], "#777", dash="dash")]
                     if marks["lf_extension_hz"] is not None else [])
            _emit_fig(landmark_figures, wasm_report.figure(
                f"Frequency landmarks — {name} (1/3 octave)",
                wasm_report.axis("Frequency (Hz)", "log", 20, 20000),
                wasm_report.axis("Measured level (dB)"),
                series_list=[wasm_report.series(name, curve["freq"], curve["spl"])],
                annotations=annotations, vlines=lines, tab=name))


    _emit_html(time_domain, capture_clock_qa_html(data))
    _emit_html(time_domain, capture_reflections_html(data))
    mixed_phase_html = _mixed_phase_summary_html(metadata, channels_dict)
    if mixed_phase_html:
        _emit_html(sections, mixed_phase_html)

    # Combined plot
    sections = overview
    _emit_fig(sections, create_combined_figure(data, output_json_path))

    # Single-screen summaries: the full PEQ listing and the crossover
    # configuration, so nothing requires switching per-channel tabs.
    _emit_html(sections, _gain_plugins_html(data))
    _emit_html(sections, _crossover_config_html(data))
    sections = time_domain
    room_decay = room_t60_rows(data)
    decay_reference = t60_itu_reference(data)
    if room_decay:
        _emit_html(sections, room_t60_table_html(data))
        room_decay_fig = create_t60_octaves_figure("Room mean", room_decay, itu_reference=decay_reference)
        if room_decay_fig:
            _emit_fig(sections, room_decay_fig)
            _emit_html(sections,
                '<p class="epa-footer">Arithmetic mean of valid measured-room T60 '
                'estimates at each octave. Speaker coverage varies by band; invalid '
                'fits are excluded.</p>\n' + t60_itu_note(decay_reference)
            )
        _emit_html(sections, 
            '<p class="epa-footer">Room T60 contributing speakers by octave: '
            + ", ".join(f"{row['centre_hz']} Hz: {row['speaker_count']}"
                        for row in room_decay)
            + '</p>\n'
        )

    # Time of flight (feat-report Section 3): table only. The before/after
    # bar charts were removed; the table carries both arrival columns.
    sections = timing
    _emit_html(sections, tof_html(metadata))

    # Symmetric-monitor summing, magnitude domain (feat-report Section 2).
    sections = symmetric
    # The complex pressure sum needs phase data roomeq does not emit yet.
    pair_groups, unpaired = symmetric_groups(channels_dict)
    if pair_groups or unpaired:
        # Bare paragraphs: the group summary already titles this section and
        # the shell wraps html sections in its own card.
        symmetric_head = (
            '<p>Rust aligns final curves within their shared frequency range. '
            'The absolute sum assumes equal phase; it is an upper bound, not a prediction '
            'of interference from simultaneous playback. The complex sum is the coherent '
            'pressure sum from measured phase; the sum difference between them is the '
            'cancellation loss.</p>'
        )
        if unpaired:
            symmetric_head += (
                '<p>Unpaired channels (not summed): '
                + ", ".join(escape(name) for name in unpaired)
                + '</p>'
            )
        _emit_html(sections, symmetric_head)
    for label, members in pair_groups:
        group_start = len(sections)
        combo = (getattr(data, "symmetric_pairs", None) or {}).get(label)
        if combo is None:
            _emit_html(sections,
                '<p>Symmetric pair ' + escape(label)
                + ': no Rust pair export in this output bundle. Regenerate with the updated RoomEQ exporter.</p>', tab=label
            )
            continue
        abs_freq = list(combo["freq"])
        complex_spl = symmetric_complex_sum(
            (channels_dict.get(members[0]) or {}).get("final_curve"),
            (channels_dict.get(members[1]) or {}).get("final_curve"),
            abs_freq,
        )
        _emit_fig(sections, create_symmetric_pair_figure(
            label, abs_freq, list(combo["sum_spl"]), complex_spl
        ))
        if complex_spl is None:
            _emit_html(sections,
                '<p class="epa-footer">Symmetric pair ' + escape(label)
                + ': complex sum unavailable without finite measured phase on both '
                + escape(members[0]) + ' and ' + escape(members[1])
                + ' final curves; showing the absolute (magnitude) sum only.</p>',
                tab=label,
            )
        for name in members:
            for landmark in landmark_figures:
                if landmark.get("tab") == name:
                    sections.append({**landmark, "tab": label})
        for section in sections[group_start:]:
            section["tab"] = label

    # Bass-management routing/headroom section. This is driven by the
    # route-level #14 schema, not the deprecated single matrix summary.
    bass_management = metadata.get("bass_management") or {}
    sections = overview
    if bass_management:
        _emit_html(sections, _bass_management_summary_html(bass_management))
        routing_fig = create_bass_management_routing_figure(data)
        headroom_fig = create_bass_management_headroom_figure(data)
        _emit_fig(sections, routing_fig)
        _emit_fig(sections, headroom_fig)
        _emit_html(sections, _bass_management_groups_table_html(bass_management))
        _emit_html(sections, _bass_management_sub_outputs_table_html(bass_management))

    # Reconstruct each logical input after bass management once. For a main
    # channel this is the crossover-aware complex sum of its high-passed main
    # output and its own low-passed route through the physical subwoofer.
    post_dsp_curves = build_post_dsp_source_curves(data)
    bass_report = metadata.get("bass_management") or {}
    physical_sub = (
        bass_report.get("physical_sub_output")
        or bass_report.get("lfe_channel")
        or "LFE"
    )
    target_references = dict(post_dsp_curves)
    target_curves = build_target_overlay_curves(
        data, target_references, output_json_path
    )

    # Individual channel sections in tabs. Multi-driver channels (e.g.
    # two subwoofers on one LFE bus) expand into one tab per physical
    # driver so each subwoofer gets its own plots and filter details.
    # Individual channel sections in shell tabs (the shell builds the tab
    # bar from the section `tab` fields, so no tab buttons are emitted).
    tab_entries = display_channel_entries(data)

    for i, entry in enumerate(tab_entries):
        sections = speakers
        channel_name = entry["channel"]
        driver_index = entry["driver"]
        tab_label = entry["label"]
        safe_label = escape(tab_label)
        channel_data = channels_dict[channel_name]
        is_driver_tab = driver_index is not None

        if is_driver_tab:
            # Physical sub output: its own measurement replayed through
            # its full deployed chain (shared EQ plus per-sub
            # gain/crossover/route).
            drivers = get_plottable_drivers(channel_data)
            driver = (
                drivers[driver_index]
                if 0 <= driver_index < len(drivers)
                else {}
            )
            chain = per_driver_chain_plugins(data, channel_name, driver_index) or []
            eq_source: dict = {"plugins": chain}
            # Raw measurements stop at the recorded band; the stored curve
            # continues past it for DSP replay and must not be drawn as data.
            initial_curve = clip_curve_to_measured_band(driver.get("initial_curve"), driver)
            final_curve = clip_curve_to_measured_band(
                per_driver_corrected_curve(data, channel_name, driver_index), driver
            )
            passes = extract_eq_passes(eq_source)
            # The tab EQ plot appears only when EQ exists (mirroring the
            # legacy empty-EQ behavior); otherwise the tab keeps its
            # response plots without an EQ section.
            eq_response_view = (
                per_driver_effective_eq(data, channel_name, driver_index)
                if channel_has_eq(channel_data)
                else None
            )
            target_view = None
            lfe_plus_channel = None
            acoustic_data = driver.get("measured_acoustics") or {}
            ir_pre = acoustic_data.get("pre_ir") or driver.get("pre_ir")
            ir_post = driver.get("post_ir")
            caption_html = (
                f'<p class="epa-footer">Physical driver output of {escape(channel_name)}: '
                "its measurement through this driver's own DSP chain "
                "(shared EQ + driver gain/crossover/route). The EQ plot below "
                "shows this total driver shaping.</p>\n"
            )
        else:
            acoustic_data = channel_data
            initial_curve = channel_data.get("initial_curve")
            if channel_data.get("drivers"):
                # A multi-driver aggregate initial is level-relative optimizer
                # state; the summed driver measurements are the acoustic baseline
                # matching the logical-input corrected curve.
                driver_baseline = sum_driver_initial_curves(channel_data)
                if driver_baseline is not None:
                    initial_curve = driver_baseline
            final_curve = _channel_display_final_curve(
                channel_name, physical_sub, channel_data, post_dsp_curves
            )
            eq_source = channel_data
            passes = extract_eq_passes(channel_data)
            eq_response_view = channel_data.get("eq_response")
            target_view = target_curves.get(channel_name)
            lfe_plus_channel = None
            if channel_name != physical_sub and _has_redirected_bass_route(data, channel_name):
                lfe_plus_channel = post_dsp_curves.get(channel_name)
            ir_pre = channel_data.get("pre_ir")
            ir_post = channel_data.get("post_ir")
            caption_html = ""

        # Extract EQ filters (grouped by pass for 3-pass pipeline)
        eq_filters = []
        for p in passes:
            eq_filters.extend(p["filters"])

        _emit_html(sections,
            f"<h2>Channel: {safe_label}</h2>\n{caption_html}",
            tab=tab_label,
        )

        # Full range plot
        fig_full = create_channel_figure(
            tab_label, initial_curve, final_curve, " (Full Range)",
            tab=tab_label,
        )
        add_channel_response_overlays(
            fig_full,
            tab_label,
            target_view,
            lfe_plus_channel,
        )
        _emit_fig(sections, fig_full)
        _emit_html(sections, landmarks_table_html({"channels": {tab_label: {
            **channel_data, "initial_curve": initial_curve}}}), tab=tab_label)

        # EQ response plot (uses per-pass breakdown when 3-pass labels are present)
        fig_eq = create_multipass_eq_figure(
            tab_label, eq_source, eq_response_view,
            sample_rate=float(data.get("sample_rate", 48_000.0)),
            tab=tab_label,
        )
        if fig_eq is None:
            fig_eq = create_eq_figure(
                tab_label, eq_filters, eq_response_view,
                sample_rate=float(data.get("sample_rate", 48_000.0)),
                tab=tab_label,
            )
        if fig_eq:
            fig_full["figure"]["y2"] = wasm_report.axis("EQ gain (dB)", "linear", -25.0, 25.0)
            for trace in fig_eq["figure"]["series"]:
                fig_full["figure"]["series"].append({**trace, "y_axis": 1})

        # IR waveform plot
        sections = time_domain
        if is_driver_tab and acoustic_data:
            rate = acoustic_data.get("sample_rate_hz")
            _emit_html(sections,
                f'<p>Independent measured driver IR: {escape(tab_label)}; '
                f'native sample rate {escape(str(rate))} Hz. This is not the summed '
                'parent-speaker response. No measured post-DSP capture is implied.</p>',
                tab=tab_label)
            if isinstance(rate, (int, float)) and rate <= 16000:
                _emit_html(sections,
                    '<p>1–8 kHz reflection analysis is unavailable at this native '
                    'sample rate. T60 bands beyond Nyquist are marked unavailable.</p>',
                    tab=tab_label)
        _emit_fig(sections, create_ir_figure(
            tab_label,
            ir_pre,
            ir_post,
            tab=tab_label,
        ))

        if not is_driver_tab or acoustic_data:
            _emit_html(sections, early_reflections_html(acoustic_data, tab_label),
                       tab=tab_label)
            _emit_fig(sections, early_reflection_figures(acoustic_data, tab_label, tab=tab_label))
            early_late = acoustic_data.get("early_late_curves")
            early_late_fig = create_early_late_figure(
                tab_label, early_late, tab=tab_label
            )
            if early_late_fig:
                _emit_fig(sections, early_late_fig)
                _emit_html(sections,
                    '<p class="epa-footer">Third-octave early and late energy contributions '
                    'share the full curve\'s peak-band reference. Full is an incoherent '
                    'energy sum, not a coherent pressure response.</p>\n',
                    tab=tab_label,
                )
            else:
                _emit_html(sections,
                    '<p class="epa-footer">Early vs late sound: pending '
                    'roomeq field early_late_curves with a shared level reference.</p>\n',
                    tab=tab_label,
                )

            # Channel tabs reuse the room-level T60 block above (its table
            # already carries per-speaker columns). Only driver tabs keep
            # their own T60: driver measured-acoustics fits exist nowhere else.
            if is_driver_tab:
                t60 = t60_rows(acoustic_data)
                t60_fig = create_t60_octaves_figure(tab_label, t60, tab=tab_label, itu_reference=decay_reference)
                _emit_html(sections, t60_table_html(acoustic_data), tab=tab_label)
                _emit_fig(sections, t60_fig)
                if t60_fig:
                    _emit_html(sections, t60_itu_note(decay_reference), tab=tab_label)
            waterfall_html = optimization_waterfall_html(
                acoustic_data.get("waterfall"), acoustic_data.get("resonance_decays"))
            if waterfall_html:
                _emit_html(sections, waterfall_html, tab=tab_label)
            else:
                _emit_html(sections,
                    '<p class="epa-footer">Waterfall and resonance decay: pending '
                    'roomeq fields waterfall and resonance_decays from a measured '
                    'room impulse response.</p>\n',
                    tab=tab_label,
                )
            wavelet_html = optimization_wavelet_html(acoustic_data.get("wavelet"))
            if wavelet_html:
                _emit_html(sections, wavelet_html, tab=tab_label)
            else:
                _emit_html(sections,
                    '<p class="epa-footer">Three-cycle wavelet: pending roomeq '
                    'field wavelet from a measured room impulse response.</p>\n',
                    tab=tab_label,
                )

        # EPA psychoacoustic scores (pre/post) for this channel
        sections = epa
        epa_per_channel = metadata.get("epa_per_channel") or {}
        epa_html = _epa_channel_table_html(epa_per_channel.get(channel_name))
        if epa_html:
            _emit_html(sections, epa_html, tab=tab_label)

        # Filter details (grouped by pass when 3-pass labels are present).
        sections = speakers
        # Driver tabs merge the driver EQ with the shared channel EQ, so
        # they render per-origin inner tabs instead of one flat list.
        has_labeled = any(p["label"] for p in passes)
        if is_driver_tab:
            _emit_html(sections,
                _driver_eq_filters_html(data, channel_name, driver_index, f"eq_{i}"),
                tab=tab_label,
            )
        elif has_labeled and passes:
            filter_parts = [
                ('<div class="filters-section">\n'
                '    <h3>EQ Filters (3-Pass Pipeline)</h3>\n')
            ]
            for p in passes:
                filter_parts.append(
                    f'<h4 style="color: {p["color"]}; margin-bottom: 5px;">'
                    f"{p['display_name']}</h4>\n"
                    '<div class="filter-list" style="margin-bottom: 10px;">\n'
                )
                for j, f in enumerate(p["filters"], 1):
                    filter_parts.append(_format_eq_filter_line(f, j))
                filter_parts.append("</div>\n")
            filter_parts.append("</div>\n")
            _emit_html(sections, "".join(filter_parts), tab=tab_label)
        elif eq_filters:
            filter_parts = [
                ('<div class="filters-section">\n'
                '    <h3>EQ Filters</h3>\n'
                '    <div class="filter-list">\n')
            ]
            for j, f in enumerate(eq_filters, 1):
                filter_parts.append(_format_eq_filter_line(f, j))
            filter_parts.append("</div>\n</div>\n")
            _emit_html(sections, "".join(filter_parts), tab=tab_label)

        if is_driver_tab:
            shaping_html = _driver_shaping_summary_html(data, channel_name, driver_index)
            if shaping_html:
                _emit_html(sections, shaping_html, tab=tab_label)

    if capture_verification is not None:
        sections = why
        verified, reason, graph_identity = verify_payload_binding(data)
        capture_graph = (capture_verification.get("graph_id")
                         if isinstance(capture_verification, dict) else None)
        if not verified:
            _emit_html(sections, 
                '<section class="capture-views"><h1>Capture acceptance diagnostics</h1>'
                '<p>Unavailable: saved optimization graph binding is invalid: '
                + escape(reason) + '</p></section>\n'
            )
        elif capture_graph != graph_identity:
            _emit_html(sections, 
                '<section class="capture-views"><h1>Capture acceptance diagnostics</h1>'
                '<p>Unavailable: verification candidate graph does not match the saved '
                'optimization graph.</p></section>\n'
            )
        else:
            _emit_html(sections, capture_views_html(capture_verification))

    # Assemble the self-contained HTML+WASM report.
    time_domain.insert(0, wasm_report.html_section(
        '<p>Arrival timing and applied delays are in Time of Flight. '
        'The diagnostics below describe the measured room impulse responses.</p>'))
    epa.insert(0, wasm_report.html_section(
        '<p>EPA scores are model-based psychoacoustic estimates before and after DSP. '
        'They are not listening-test results or proof of improved audibility. '
        'Compare each metric using its stated direction and units.</p>'))
    modes = resonance_summary_html(data)
    if modes:
        time_domain.append({"kind": "html", "html": modes, "tab": None, "footer": True})
    # The relative-level table details per-speaker levels, so it leads the
    # Details tab even though it is computed from the whole output.
    _relative_levels = level_compensation_html(data)
    if _relative_levels:
        speakers.insert(0, wasm_report.html_section(_relative_levels))

    sections = list(status_sections)
    for bucket, subhead, contents in (
        (BUCKET_DSP, "Why this correction?", why),
        (BUCKET_ACOUSTICS, "Summary", overview),
        (BUCKET_ACOUSTICS, "Details per speaker", speakers),
        (BUCKET_ACOUSTICS, "Time of Flight", timing),
        (BUCKET_ACOUSTICS, "Symmetric monitors", symmetric),
        (BUCKET_ACOUSTICS, "Time domain analysis", time_domain),
        (BUCKET_PSYCHOACOUSTIC, "Section 6: EPA scores", epa),
        (BUCKET_DSP, "DSP signal flow", signal_flow_sections(data)),
    ):
        if not contents:
            contents.append(wasm_report.html_section("<p>No data available in this output.</p>"))
        sections.extend({**section, "bucket": bucket, "subhead": subhead}
                        for section in contents)
    payload = wasm_report.payload(page_title, sections,
                                  _report_provenance([("", data)]))
    wasm_report.write_report(output_path, page_title, payload)

    print(f"HTML report written to: {output_path}")


def create_comparison_html_report(
    mode_datasets: list[tuple[str, dict]],
    output_path: Path,
) -> None:
    """Create an HTML report comparing multiple processing modes.

    Args:
        mode_datasets: List of (mode_name, roomeq_output_data) tuples.
        output_path: Path to write HTML report.
    """
    # Compare what reaches the microphone, including each source channel's
    # redirected-bass route. Plotting an isolated high-passed main below its
    # crossover creates a false SPL collapse and makes different crossover
    # choices look like different RoomEQ magnitude targets.
    comparison_channels: dict[int, dict[str, dict]] = {}
    for _, data in mode_datasets:
        routed_curves = build_post_dsp_source_curves(data)
        comparison_channels[id(data)] = {
            name: {
                **channel,
                "final_curve": routed_curves.get(name, channel.get("final_curve")),
            }
            for name, channel in (data.get("channels", {}) or {}).items()
        }

    # Collect all channel names across modes (union)
    all_channel_names: set[str] = set()
    for _, data in mode_datasets:
        all_channel_names.update(comparison_channels[id(data)].keys())
    sorted_channels = sorted(all_channel_names, key=get_channel_sort_key)

    # If both L and R are present in at least one mode, append a
    # synthetic "L+R" tab whose curves are the complex (coherent) sum
    # of the per-mode L and R curves. The tab is only emitted when at
    # least one mode actually has both channels — otherwise the sum
    # would be undefined and the tab would be empty.
    has_lr_pair = any(
        "L" in comparison_channels[id(data)] and "R" in comparison_channels[id(data)]
        for _, data in mode_datasets
    )
    if has_lr_pair:
        sorted_channels.append(LR_SUM_CHANNEL)

    mode_names = [name for name, _ in mode_datasets]
    page_title = f"RoomEQ Mode Comparison: {', '.join(_mode_label(n) for n in mode_names)}"

    # Sections in document order; the shell renders them and builds the
    # per-channel tab bar from the section `tab` fields.
    sections: list[dict] = []

    for mode_name, data in mode_datasets:
        _emit_html(sections, _playback_status_html(data.get("metadata") or {}, mode_name, data=data),
                   flat=True)
        _emit_html(sections, capture_clock_qa_html(data))
        _emit_html(sections, capture_reflections_html(data))
        _emit_html(sections, correction_explanation_html(data, mode_name))
        _emit_html(sections, acceptance_views_html(data, mode_name))
        _emit_html(sections, waveform_status_html(data, mode_name))

    # --- Summary table ---
    summary_parts = ['<h2>Summary</h2>\n<table class="summary-table">\n<thead><tr>']
    has_gd_summary = any(
        (data.get("metadata") or {}).get("group_delay") for _, data in mode_datasets
    )
    has_auto_summary = any("_auto" in mode_name for mode_name, _ in mode_datasets)
    summary_parts.append(
        "<th>Mode</th><th>Loss</th>"
        '<th title="Flat loss before EQ (always computed via compute_flat_loss, regardless of which objective was minimized)">Pre flat-loss</th>'
        '<th title="Flat loss after EQ (always computed via compute_flat_loss, regardless of which objective was minimized)">Post flat-loss</th>'
        "<th>Improvement</th>"
        "<th>EPA Pref (pre)</th><th>EPA Pref (post)</th><th>EPA Δ</th>"
    )
    if has_auto_summary:
        summary_parts.append(
            '<th title="Emitted EQ sections per channel, including driver-local sections; '
            'Kautz banks count all basis sections and shared entries count once; '
            'ranges show min-max across channels">EQ sections</th>'
        )
    if has_gd_summary:
        summary_parts.append(
            '<th title="Group-delay optimization advisory">GD advisory</th>'
            '<th title="Whether GD controls were inserted into exported DSP">GD applied</th>'
            '<th title="Summed in-band GD RMS before and after GD optimization">GD RMS</th>'
            '<th title="GD RMS improvement, 20*log10(pre/post)">GD Δ</th>'
            '<th title="Total emitted all-pass filters; coh is mean in-band coherence">GD AP/coh</th>'
        )
    summary_parts.append("</tr></thead>\n<tbody>\n")

    mode_scores: list[tuple[str, float, float]] = []
    loss_types: list[str | None] = []
    for mode_name, data in mode_datasets:
        meta = data.get("metadata", {})
        pre = meta.get("pre_score", 0)
        post = meta.get("post_score", 0)
        improv = ((pre - post) / pre * 100) if pre > 0 else 0
        mode_scores.append((mode_name, pre, post))
        color = _mode_color(mode_name)

        loss_type = meta.get("loss_type")
        loss_types.append(loss_type)
        loss_cell = f"<td>{loss_type}</td>" if loss_type else '<td style="color:#999">-</td>'

        epa_pre_avg, epa_post_avg = _epa_summary_pref(meta)
        if epa_pre_avg is not None and epa_post_avg is not None:
            epa_delta = epa_post_avg - epa_pre_avg
            epa_color = "#2ecc71" if epa_delta >= 0 else "#e74c3c"
            epa_cells = (
                f"<td>{epa_pre_avg:.2f}</td><td>{epa_post_avg:.2f}</td>"
                f'<td style="color:{epa_color};font-weight:600">{epa_delta:+.2f}</td>'
            )
        else:
            epa_cells = '<td style="color:#999">-</td><td style="color:#999">-</td><td style="color:#999">-</td>'

        summary_parts.append(
            f'<tr><td style="color:{color};font-weight:600">{_mode_label(mode_name)}</td>'
            f"{loss_cell}"
            f"<td>{pre:.4f}</td><td>{post:.4f}</td>"
            f'<td class="improvement">{improv:.1f}%</td>'
            f"{epa_cells}"
            f"{_eq_filter_summary_html(data) if has_auto_summary else ''}"
            f"{_gd_summary_cells_html(meta) if has_gd_summary else ''}</tr>\n"
        )
    summary_parts.append("</tbody></table>\n")
    _emit_html(sections, "".join(summary_parts))

    # When the report mixes loss functions, clarify what the Pre/Post
    # flat-loss columns actually measure — readers might otherwise
    # assume an EPA-loss run's "score" reflects the EPA composite.
    distinct_losses = sorted({lt for lt in loss_types if lt})
    if len(distinct_losses) >= 2:
        _emit_html(sections, 
            '<p style="color:#555;font-size:0.9em;margin-top:10px;">'
            "ℹ This report mixes runs that minimized different loss "
            f"functions ({', '.join(distinct_losses)}). The "
            "<strong>Pre flat-loss</strong> and <strong>Post flat-loss</strong> "
            "columns are <em>always</em> computed via the flat-loss helper "
            "(<code>compute_flat_loss</code> over <code>[min_freq, max_freq]</code>), "
            "regardless of which objective each mode actually minimized — "
            "this keeps every mode on the same scale so the columns answer "
            "<em>“how flat is the response?”</em> for all modes. For "
            "<em>perceptual</em> outcomes across loss types, the "
            "<strong>EPA Pref</strong> columns are the right place to look."
            "</p>\n"
        )
    if has_gd_summary:
        _emit_html(sections, 
            '<p style="color:#555;font-size:0.9em;margin-top:10px;">'
            "GD columns come from <code>metadata.group_delay</code>. "
            "A non-success advisory is expected for recordings that do not "
            "carry coherence or independent sweep realisations; it means the "
            "production safety gates downgraded or skipped the requested GD path."
            "</p>\n"
        )
    if has_auto_summary:
        _emit_html(sections, 
            '<p style="color:#555;font-size:0.9em;margin-top:10px;">'
            "Auto columns show emitted EQ filter counts from the output JSON. "
            "Resolved automatic Q and gain bounds are logged by roomeq during the run "
            "but are not currently persisted in comparison JSON."
            "</p>\n"
        )

    # Score bar chart (passes loss_types so the chart can label/warn correctly)
    _emit_fig(sections, create_score_comparison_figure(mode_scores, loss_types))

    # Global design-target level anchor. Serialized absolute targets carry
    # the design shape but not the measured level, so match the main (L/R)
    # target shape to the measured per-channel mean of the L/R pair over
    # 100 Hz - 10 kHz once. The single offset applies to every channel tab,
    # preserving designed inter-channel target differences. Without an L/R
    # pair there is no anchor and targets render as serialized.
    _match_lo, _match_hi = TARGET_LEVEL_MATCH_BAND_HZ
    global_target_offset: float | None = None
    main_target_shape: dict | None = None
    for _, anchor_data in mode_datasets:
        anchor_channels = comparison_channels[id(anchor_data)]
        anchor_l = anchor_channels.get("L") or {}
        anchor_r = anchor_channels.get("R") or {}
        ref_l = anchor_l.get("final_curve") or anchor_l.get("initial_curve")
        ref_r = anchor_r.get("final_curve") or anchor_r.get("initial_curve")
        anchor_shape = anchor_l.get("target_curve") or anchor_r.get("target_curve")
        offset = global_target_offset_for_pair(
            ref_l, ref_r, anchor_shape, _match_lo, _match_hi
        )
        if offset is None:
            continue
        global_target_offset = offset
        main_target_shape = anchor_shape
        break

    # --- Per-channel tabs ---
    # Per-channel sections in shell tabs (tab bar built by the shell).
    for i, ch_name in enumerate(sorted_channels):
        is_lr = (ch_name == LR_SUM_CHANNEL)
        includes_redirected_bass = not is_lr and any(
            _has_redirected_bass_route(data, ch_name) for _, data in mode_datasets
        )
        source_label = _comparison_source_label(ch_name, includes_redirected_bass)
        if is_lr:
            _emit_html(sections,
                "<h2>Logical source L+R</h2>\n"
                "<p>Complex sums require valid per-microphone clock evidence and measured phase. "
                "Otherwise the plot uses a magnitude-only power sum, which is not a coherent pressure prediction.</p>\n",
                tab=ch_name,
            )
        else:
            _emit_html(sections, f"<h2>{source_label}</h2>\n", tab=ch_name)
            if includes_redirected_bass:
                _emit_html(sections,
                    '<p style="color:#666;font-size:0.9em;margin:-10px 0 15px 0;">'
                    "The response traces are the coherent microphone prediction for this "
                    "input source: its high-passed physical speaker output plus its "
                    "low-passed route through the physical subwoofer. They are not the "
                    f"isolated {ch_name} loudspeaker output."
                    "</p>\n",
                    tab=ch_name,
                )

        # Build mode_data for this channel. The synthetic L+R channel
        # is computed on the fly from each mode's L and R curves; all
        # other channels are read straight out of the JSON.
        mode_data: list[tuple[str, dict]] = []
        for mode_name, data in mode_datasets:
            channels = comparison_channels[id(data)]
            if is_lr:
                lr, fallback_reason = gated_lr_channel(data, channels.get("L"), channels.get("R"))
                if fallback_reason:
                    _emit_html(sections, f"<p><strong>{escape(str(mode_name))}: magnitude-only fallback.</strong> {escape(fallback_reason)}</p>\n", tab=ch_name)
                if lr:
                    mode_data.append((mode_name, lr))
            else:
                ch_data = channels.get(ch_name, {})
                if ch_data:
                    mode_data.append((mode_name, ch_data))

        if not mode_data:
            _emit_html(sections, "<p>No data for this channel.</p>\n", tab=ch_name)
            continue

        # Shared design target: first mode carrying a serialized absolute
        # target wins (compared modes normally share one fixture target).
        # No fallback flat line — it would misstate the design slope.
        # The synthetic L+R channel carries no target of its own, so it
        # reuses the main (L/R) target shape interpolated onto its grid.
        # Every tab's target is then shifted by the global L+R level anchor
        # so the drawn target mean matches the measured midband level.
        comparison_target = None
        if is_lr and main_target_shape is not None:
            lr_grid = None
            for _, ch_data in mode_data:
                lr_grid = (ch_data or {}).get("final_curve") or (ch_data or {}).get(
                    "initial_curve"
                )
                if lr_grid:
                    break
            if lr_grid is not None:
                comparison_target = shift_target_to_reference_band_mean(
                    main_target_shape, lr_grid, _match_lo, _match_hi
                )
        else:
            for _, ch_data in mode_data:
                candidate = (ch_data or {}).get("target_curve")
                if (
                    isinstance(candidate, dict)
                    and candidate.get("freq")
                    and candidate.get("spl")
                ):
                    comparison_target = {
                        "freq": list(candidate["freq"]),
                        "spl": list(candidate["spl"]),
                    }
                    break
        if (
            comparison_target is not None
            and not is_lr
            and global_target_offset is not None
        ):
            comparison_target = {
                "freq": list(comparison_target["freq"]),
                "spl": [
                    level + global_target_offset
                    for level in comparison_target["spl"]
                ],
            }

        # 1. Overlay plot (full range) + Zoomed (bass)
        _emit_fig(sections, create_comparison_overlay_figure(
            source_label, mode_data, target_curve=comparison_target,
            tab=ch_name,
        ))
        _emit_fig(sections, create_comparison_zoomed_figure(
            source_label, mode_data, tab=ch_name
        ))

        # 2. Phase before/after per mode
        _emit_fig(sections, create_comparison_phase_figure(
            ch_name, mode_data, tab=ch_name
        ))

        # 3. Group delay before/after per mode
        _emit_fig(sections, create_comparison_group_delay_figure(
            ch_name, mode_data, tab=ch_name
        ))

        # 4. Impulse response before/after per mode
        _emit_fig(sections, create_comparison_ir_figure(
            ch_name, mode_data, tab=ch_name
        ))

        # 5. Per-mode detail (one section per mode; was a subplot grid)
        _emit_fig(sections, create_mode_subplots_figure(
            ch_name, mode_data, target_curve=comparison_target, tab=ch_name
        ))

        # 6. EQ response overlay
        _emit_fig(sections, create_comparison_eq_overlay_figure(
            ch_name, mode_data,
            sample_rates={name: float(output.get("sample_rate", 48_000.0))
                          for name, output in mode_datasets},
            tab=ch_name,
        ))

        # 7. EPA psychoacoustic scores per mode
        epa_html = _epa_comparison_table_html(ch_name, mode_datasets)
        if epa_html:
            _emit_html(sections, epa_html, tab=ch_name)

    payload = wasm_report.payload(page_title, sections,
                                  _report_provenance(mode_datasets))
    wasm_report.write_report(output_path, page_title, payload)

    print(f"Comparison report written to: {output_path}")
