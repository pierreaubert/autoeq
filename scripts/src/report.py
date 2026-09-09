"""HTML report generation for roomeq visualization."""

from html import escape
from pathlib import Path

from .figures import (
    create_channel_figure,
    add_channel_response_overlays,
    create_zoomed_figure,
    create_eq_figure,
    create_multipass_eq_figure,
    create_ir_figure,
    create_combined_figure,
    create_bass_management_routing_figure,
    create_bass_management_headroom_figure,
    create_comparison_overlay_figure,
    create_comparison_zoomed_figure,
    create_comparison_eq_overlay_figure,
    create_comparison_phase_figure,
    create_comparison_group_delay_figure,
    create_comparison_ir_figure,
    create_mode_subplots_figure,
    create_score_comparison_figure,
    _mode_label,
    _mode_color,
)
from .data_extract import (
    channel_has_eq,
    display_channel_entries,
    driver_display_names,
    extract_eq_passes,
    get_channel_sort_key,
    get_plottable_drivers,
)
from .dsp import (
    build_post_dsp_source_curves,
    driver_destination_route,
    per_driver_chain_plugins,
    per_driver_corrected_curve,
    per_driver_effective_eq,
    split_driver_eq_plugins,
    sum_driver_initial_curves,
    synthesize_lr_channel,
)
from .target_overlay import build_target_overlay_curves

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
    filter_type = filt.get("filter_type", "peak")
    freq = filt.get("freq", 0)
    q = filt.get("q", 1)
    gain = filt.get("db_gain", 0)
    return (
        f"Filter {number}: {filter_type.upper()} @ {freq:.1f} Hz, "
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
        '\n                <div class="filters-section">\n'
        "                    <h3>EQ Filters</h3>\n"
        '                    <p class="epa-footer">'
        f"{total} PEQ filter(s) total: {driver_count} driver + "
        f"{shared_count} shared channel. Numbering restarts in each tab; "
        "the EQ plot shows the combined response.</p>\n"
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


def _all_eq_filters_html(data: dict) -> str:
    """Render every EQ filter list in one summary section.

    Mirrors the per-tab filter sections (driver tabs split by origin with a
    1..N numbering) so the whole correction is visible without switching
    tabs. Returns an empty string when no channel carries EQ filters.
    """
    channels = data.get("channels") or {}
    blocks: list[str] = []
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
        blocks.append(f"            <h3>{safe_label}</h3>\n")
        for origin, passes in groups:
            count = sum(len(p["filters"]) for p in passes)
            blocks.append(f"            <h4>{escape(origin)} ({count})</h4>\n")
            blocks.append(_eq_filter_table_html(passes))
    if not blocks:
        return ""
    return (
        '        <div class="plot-container">\n'
        "            <h2>All EQ Filters</h2>\n"
        '            <p class="epa-footer">Complete PEQ listing for every '
        "channel and driver (same data and numbering as the per-tab "
        "sections below).</p>\n"
        + "".join(blocks)
        + "        </div>\n"
    )


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
        '        <div class="plot-container">\n'
        "            <h2>Crossover Configuration</h2>\n"
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
        for plugin in channel.get("plugins") or []:
            if not isinstance(plugin, dict) or plugin.get("plugin_type") != "eq":
                continue
            params = plugin.get("parameters") or {}
            filters = params.get("filters") or plugin.get("filters") or []
            if isinstance(filters, list):
                total += len(filters)
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


def create_html_report(
    data: dict,
    output_path: Path,
    output_json_path: Path | None = None,
) -> None:
    """Create an HTML report with all channel plots.

    Args:
        data: Output JSON data (roomeq result)
        output_path: Path to write HTML report
        output_json_path: Path to output JSON (for resolving relative paths)
    """
    channels_dict = data.get("channels", {})
    metadata = data.get("metadata", {})
    version = data.get("version", "unknown")

    # Sort channels by classical order
    sorted_channel_names = sorted(channels_dict.keys(), key=get_channel_sort_key)
    channels = [(name, channels_dict[name]) for name in sorted_channel_names]

    # Short name for title: parent_dir/filename
    if output_json_path:
        short_name = f"{output_json_path.parent.name}/{output_json_path.name}"
    else:
        short_name = ""
    page_title = f"RoomEQ Results - {short_name}" if short_name else "RoomEQ Results"

    # Build HTML content
    html_parts = [
        "<!DOCTYPE html>\n"
        "<html>\n"
        "<head>\n"
        '    <meta charset="utf-8">\n'
        f"    <title>{page_title}</title>\n"
        '    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>\n'
        """    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0;
            padding: 20px;
            background: #f5f5f5;
        }
        .container {
            max-width: 1400px;
            margin: 0 auto;
        }
        h1 {
            color: #333;
            border-bottom: 2px solid #4a90d9;
            padding-bottom: 10px;
        }
        h2 {
            color: #444;
            margin-top: 30px;
        }
        .metadata {
            background: white;
            padding: 15px 20px;
            border-radius: 8px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .metadata h2 {
            margin-top: 0;
            color: #555;
            font-size: 1.1em;
        }
        .metadata-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 10px;
        }
        .metadata-item {
            padding: 5px 0;
        }
        .metadata-label {
            font-weight: 600;
            color: #666;
        }
        .metadata-value {
            color: #333;
        }
        .improvement {
            color: #2ecc71;
            font-weight: bold;
        }
        .plot-container {
            background: white;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .plot-row {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-bottom: 20px;
        }
        @media (max-width: 1000px) {
            .plot-row {
                grid-template-columns: 1fr;
            }
        }
        .filters-section {
            background: #fdfdfd;
            padding: 15px 20px;
            border-radius: 8px;
            margin-top: 20px;
            border: 1px solid #eee;
        }
        .filters-section h3 {
            margin-top: 0;
            color: #555;
        }
        .filter-list {
            font-family: monospace;
            font-size: 0.9em;
            background: #f8f8f8;
            padding: 10px;
            border-radius: 4px;
            overflow-x: auto;
        }
        .channel-section {
            padding: 10px 0;
        }
        .epa-section {
            background: #fafafa;
            padding: 15px 20px;
            border-radius: 8px;
            margin-top: 20px;
            border: 1px solid #eee;
        }
        .epa-section h3 {
            margin-top: 0;
            color: #555;
        }
        .epa-table {
            width: 100%;
            border-collapse: collapse;
            font-size: 0.95em;
        }
        .epa-table th,
        .epa-table td {
            padding: 6px 10px;
            border: 1px solid #e4e4e4;
            text-align: right;
        }
        .epa-table th:first-child,
        .epa-table td:first-child {
            text-align: left;
        }
        .epa-table th {
            background: #f1f1f1;
            font-weight: 600;
            color: #555;
        }
        .epa-table tbody tr:nth-child(odd) {
            background: #fdfdfd;
        }
        .epa-footer {
            margin: 10px 0 0;
            color: #888;
            font-size: 0.8em;
            font-style: italic;
        }
        .bass-management-section {
            border-left: 4px solid #2ecc71;
        }
        .bm-table {
            width: 100%;
            border-collapse: collapse;
            font-size: 0.92em;
        }
        .bm-table th,
        .bm-table td {
            padding: 7px 9px;
            border: 1px solid #e4e4e4;
            text-align: left;
            vertical-align: top;
        }
        .bm-table th {
            background: #f1f1f1;
            font-weight: 600;
            color: #555;
        }
        .bm-table tbody tr:nth-child(odd) {
            background: #fdfdfd;
        }

        /* Tabs styles */
        .tabs-container {
            margin-top: 30px;
        }
        .tab-header {
            display: flex;
            flex-wrap: wrap;
            background: #e0e0e0;
            padding: 10px 10px 0;
            border-radius: 8px 8px 0 0;
            gap: 2px;
        }
        .tab-btn {
            padding: 10px 20px;
            border: none;
            background: #d0d0d0;
            cursor: pointer;
            border-radius: 5px 5px 0 0;
            font-weight: 600;
            color: #666;
            transition: all 0.2s;
        }
        .tab-btn:hover {
            background: #c0c0c0;
        }
        .tab-btn.active {
            background: white;
            color: #4a90d9;
            border-top: 3px solid #4a90d9;
        }
        .tab-content {
            display: none;
            background: white;
            padding: 20px;
            border-radius: 0 0 8px 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .tab-content.active {
            display: block;
        }
        /* Inner EQ tabs: per-origin filter lists inside driver tabs.
           Scoped classes so they never clash with the outer channel tabs. */
        .eq-tabs {
            margin-top: 10px;
        }
        .eq-tab-header {
            display: flex;
            flex-wrap: wrap;
            background: #ececec;
            padding: 6px 6px 0;
            border-radius: 6px 6px 0 0;
            gap: 2px;
        }
        .eq-tab-btn {
            padding: 6px 14px;
            border: none;
            background: #d8d8d8;
            cursor: pointer;
            border-radius: 4px 4px 0 0;
            font-weight: 600;
            font-size: 0.85em;
            color: #666;
            transition: all 0.2s;
        }
        .eq-tab-btn:hover {
            background: #c8c8c8;
        }
        .eq-tab-btn.active {
            background: #f8f8f8;
            color: #4a90d9;
            border-top: 2px solid #4a90d9;
        }
        .eq-tab-panel {
            display: none;
        }
        .eq-tab-panel.active {
            display: block;
        }
    </style>
    <script>
        function openEqTab(evt, panelId) {
            var container = evt.currentTarget.closest(".eq-tabs");
            if (!container) {
                return;
            }
            var panels = container.querySelectorAll(".eq-tab-panel");
            for (var i = 0; i < panels.length; i++) {
                panels[i].classList.remove("active");
            }
            var btns = container.querySelectorAll(".eq-tab-btn");
            for (var j = 0; j < btns.length; j++) {
                btns[j].classList.remove("active");
            }
            document.getElementById(panelId).classList.add("active");
            evt.currentTarget.classList.add("active");
        }
        function openChannel(evt, channelId) {
            var i, tabcontent, tablinks;
            tabcontent = document.getElementsByClassName("tab-content");
            for (i = 0; i < tabcontent.length; i++) {
                tabcontent[i].classList.remove("active");
            }
            tablinks = document.getElementsByClassName("tab-btn");
            for (i = 0; i < tablinks.length; i++) {
                tablinks[i].classList.remove("active");
            }
            document.getElementById(channelId).classList.add("active");
            evt.currentTarget.classList.add("active");
            
            // Trigger resize to fix Plotly plots in the newly visible tab
            window.dispatchEvent(new Event('resize'));
        }
    </script>
</head>
<body>
    <div class="container">
"""
        f"        <h1>{page_title}</h1>\n"
    ]

    # Metadata section
    if metadata:
        pre_score = metadata.get("pre_score", 0)
        post_score = metadata.get("post_score", 0)
        improvement = pre_score - post_score if pre_score and post_score else 0

        epa_pre_avg, epa_post_avg = _epa_summary_pref(metadata)
        if epa_pre_avg is not None and epa_post_avg is not None:
            epa_delta = epa_post_avg - epa_pre_avg
            epa_color = "#2ecc71" if epa_delta >= 0 else "#e74c3c"
            epa_summary_html = (
                '                <div class="metadata-item">\n'
                '                    <span class="metadata-label">EPA Preference (avg):</span>\n'
                f'                    <span class="metadata-value">{epa_pre_avg:.2f} → {epa_post_avg:.2f} '
                f'<span style="color:{epa_color};font-weight:600">({epa_delta:+.2f})</span></span>\n'
                "                </div>\n"
            )
        else:
            epa_summary_html = ""

        html_parts.append(
            f"""
        <div class="metadata">
            <h2>Optimization Summary</h2>
            <div class="metadata-grid">
                <div class="metadata-item">
                    <span class="metadata-label">Version:</span>
                    <span class="metadata-value">{version}</span>
                </div>
                <div class="metadata-item">
                    <span class="metadata-label">Algorithm:</span>
                    <span class="metadata-value">{metadata.get('algorithm', 'N/A')}</span>
                </div>
                <div class="metadata-item">
                    <span class="metadata-label">Loss function:</span>
                    <span class="metadata-value">{metadata.get('loss_type', 'N/A')}</span>
                </div>
                <div class="metadata-item">
                    <span class="metadata-label">Iterations:</span>
                    <span class="metadata-value">{metadata.get('iterations', 'N/A')}</span>
                </div>
                <div class="metadata-item">
                    <span class="metadata-label">Score Before:</span>
                    <span class="metadata-value">{pre_score:.2f}</span>
                </div>
                <div class="metadata-item">
                    <span class="metadata-label">Score After:</span>
                    <span class="metadata-value">{post_score:.2f}</span>
                </div>
                <div class="metadata-item">
                    <span class="metadata-label">Improvement:</span>
                    <span class="metadata-value improvement">{improvement:.2f}</span>
                </div>
{epa_summary_html}                <div class="metadata-item">
                    <span class="metadata-label">Timestamp:</span>
                    <span class="metadata-value">{metadata.get('timestamp', 'N/A')}</span>
                </div>
            </div>
        </div>
"""
        )

    mixed_phase_html = _mixed_phase_summary_html(metadata, channels_dict)
    if mixed_phase_html:
        html_parts.append(mixed_phase_html)

    # Combined plot
    combined_fig = create_combined_figure(data, output_json_path)
    combined_html = combined_fig.to_html(full_html=False, include_plotlyjs=False)
    html_parts.append(
        f"""
        <div class="plot-container">
            <h2>All Channels Overview</h2>
            {combined_html}
        </div>
"""
    )

    # Single-screen summaries: the full PEQ listing and the crossover
    # configuration, so nothing requires switching per-channel tabs.
    html_parts.append(_all_eq_filters_html(data))
    html_parts.append(_crossover_config_html(data))

    # Bass-management routing/headroom section. This is driven by the
    # route-level #14 schema, not the deprecated single matrix summary.
    bass_management = metadata.get("bass_management") or {}
    if bass_management:
        html_parts.append(_bass_management_summary_html(bass_management))
        routing_fig = create_bass_management_routing_figure(data)
        headroom_fig = create_bass_management_headroom_figure(data)
        if routing_fig or headroom_fig:
            html_parts.append('<div class="plot-row">\n')
            if routing_fig:
                html_parts.append(
                    f'<div class="plot-container">{routing_fig.to_html(full_html=False, include_plotlyjs=False)}</div>\n'
                )
            if headroom_fig:
                html_parts.append(
                    f'<div class="plot-container">{headroom_fig.to_html(full_html=False, include_plotlyjs=False)}</div>\n'
                )
            html_parts.append("</div>\n")
        html_parts.append(_bass_management_groups_table_html(bass_management))
        html_parts.append(_bass_management_sub_outputs_table_html(bass_management))

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
    tab_entries = display_channel_entries(data)
    html_parts.append('<div class="tabs-container">\n')
    html_parts.append('    <div class="tab-header">\n')
    for i, entry in enumerate(tab_entries):
        active_class = " active" if i == 0 else ""
        safe_id = f"channel_{i}"
        html_parts.append(f'        <button class="tab-btn{active_class}" onclick="openChannel(event, \'{safe_id}\')">{escape(entry["label"])}</button>\n')
    html_parts.append('    </div>\n')

    for i, entry in enumerate(tab_entries):
        channel_name = entry["channel"]
        driver_index = entry["driver"]
        tab_label = entry["label"]
        safe_label = escape(tab_label)
        channel_data = channels_dict[channel_name]
        is_driver_tab = driver_index is not None
        active_class = " active" if i == 0 else ""
        safe_id = f"channel_{i}"

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
            initial_curve = driver.get("initial_curve")
            final_curve = per_driver_corrected_curve(data, channel_name, driver_index)
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
            ir_pre = driver.get("pre_ir")
            ir_post = driver.get("post_ir")
            caption_html = (
                f'<p class="epa-footer">Physical sub output of {escape(channel_name)}: '
                "its measurement through this sub's own DSP chain "
                "(shared EQ + per-sub gain/crossover/route). The EQ plot below "
                "shows this total per-sub shaping.</p>\n"
            )
        else:
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

        html_parts.append(
            f"""
        <div id="{safe_id}" class="tab-content{active_class}">
            <div class="channel-section">
                <h2>Channel: {safe_label}</h2>
{caption_html}"""
        )

        # Full range plot
        fig_full = create_channel_figure(
            tab_label, initial_curve, final_curve, " (Full Range)"
        )
        add_channel_response_overlays(
            fig_full,
            tab_label,
            target_view,
            lfe_plus_channel,
        )
        full_html = fig_full.to_html(full_html=False, include_plotlyjs=False)

        # Zoomed plot (20-1200 Hz)
        fig_zoom = create_zoomed_figure(tab_label, initial_curve, final_curve)
        add_channel_response_overlays(
            fig_zoom,
            tab_label,
            target_view,
            lfe_plus_channel,
        )
        zoom_html = fig_zoom.to_html(full_html=False, include_plotlyjs=False)

        html_parts.append(
            f"""
                <div class="plot-row">
                    <div class="plot-container">
                        {full_html}
                    </div>
                    <div class="plot-container">
                        {zoom_html}
                    </div>
                </div>
"""
        )

        # EQ response plot (uses per-pass breakdown when 3-pass labels are present)
        fig_eq = create_multipass_eq_figure(
            tab_label, eq_source, eq_response_view
        )
        if fig_eq is None:
            fig_eq = create_eq_figure(tab_label, eq_filters, eq_response_view)
        if fig_eq:
            eq_html = fig_eq.to_html(full_html=False, include_plotlyjs=False)
            html_parts.append(
                f"""
                <div class="plot-container">
                    {eq_html}
                </div>
"""
            )

        # IR waveform plot
        fig_ir = create_ir_figure(
            tab_label,
            ir_pre,
            ir_post,
        )
        if fig_ir:
            ir_html = fig_ir.to_html(full_html=False, include_plotlyjs=False)
            html_parts.append(
                f"""
                <div class="plot-container">
                    {ir_html}
                </div>
"""
            )

        # EPA psychoacoustic scores (pre/post) for this channel
        epa_per_channel = metadata.get("epa_per_channel") or {}
        epa_html = _epa_channel_table_html(epa_per_channel.get(channel_name))
        if epa_html:
            html_parts.append(epa_html)

        # Filter details (grouped by pass when 3-pass labels are present).
        # Driver tabs merge the driver EQ with the shared channel EQ, so
        # they render per-origin inner tabs instead of one flat list.
        has_labeled = any(p["label"] for p in passes)
        if is_driver_tab:
            html_parts.append(
                _driver_eq_filters_html(data, channel_name, driver_index, f"eq_{i}")
            )
        elif has_labeled and passes:
            html_parts.append(
                """
                <div class="filters-section">
                    <h3>EQ Filters (3-Pass Pipeline)</h3>
"""
            )
            for p in passes:
                html_parts.append(
                    f"""
                    <h4 style="color: {p['color']}; margin-bottom: 5px;">{p['display_name']}</h4>
                    <div class="filter-list" style="margin-bottom: 10px;">
"""
                )
                for j, f in enumerate(p["filters"], 1):
                    filter_type = f.get("filter_type", "peak")
                    freq = f.get("freq", 0)
                    q = f.get("q", 1)
                    gain = f.get("db_gain", 0)
                    html_parts.append(
                        f"Filter {j}: {filter_type.upper()} @ {freq:.1f} Hz, Q={q:.2f}, Gain={gain:+.1f} dB<br>\n"
                    )
                html_parts.append("                    </div>\n")
            html_parts.append("                </div>\n")
        elif eq_filters:
            html_parts.append(
                """
                <div class="filters-section">
                    <h3>EQ Filters</h3>
                    <div class="filter-list">
"""
            )
            for j, f in enumerate(eq_filters, 1):
                filter_type = f.get("filter_type", "peak")
                freq = f.get("freq", 0)
                q = f.get("q", 1)
                gain = f.get("db_gain", 0)
                html_parts.append(
                    f"Filter {j}: {filter_type.upper()} @ {freq:.1f} Hz, Q={q:.2f}, Gain={gain:+.1f} dB<br>\n"
                )
            html_parts.append(
                """
                    </div>
                </div>
"""
            )

        if is_driver_tab:
            shaping_html = _driver_shaping_summary_html(data, channel_name, driver_index)
            if shaping_html:
                html_parts.append(shaping_html)

        html_parts.append(
            """
            </div>
        </div>
"""
        )

    html_parts.append('</div><!-- tabs-container -->\n')

    # Close HTML
    html_parts.append(
        """
    </div>
</body>
</html>
"""
    )

    # Write output
    with open(output_path, "w") as f:
        f.write("".join(html_parts))

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

    html_parts = [
        "<!DOCTYPE html>\n<html>\n<head>\n"
        '    <meta charset="utf-8">\n'
        f"    <title>{page_title}</title>\n"
        '    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>\n'
        """    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
               margin: 0; padding: 20px; background: #f5f5f5; }
        .container { max-width: 1400px; margin: 0 auto; }
        h1 { color: #333; border-bottom: 2px solid #4a90d9; padding-bottom: 10px; }
        h2 { color: #444; margin-top: 30px; }
        .summary-table { width: 100%; border-collapse: collapse; margin: 15px 0; }
        .summary-table th, .summary-table td { padding: 8px 12px; border: 1px solid #ddd; text-align: center; }
        .summary-table th { background: #f0f0f0; font-weight: 600; color: #555; }
        .improvement { color: #2ecc71; font-weight: bold; }
        .plot-container { background: white; padding: 15px; border-radius: 8px;
                         margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .plot-row { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 20px; }
        @media (max-width: 1000px) { .plot-row { grid-template-columns: 1fr; } }
        .tabs-container { margin-top: 30px; }
        .tab-header { display: flex; flex-wrap: wrap; background: #e0e0e0;
                     padding: 10px 10px 0; border-radius: 8px 8px 0 0; gap: 2px; }
        .tab-btn { padding: 10px 20px; border: none; background: #d0d0d0; cursor: pointer;
                  border-radius: 5px 5px 0 0; font-weight: 600; color: #666; transition: all 0.2s; }
        .tab-btn:hover { background: #c0c0c0; }
        .tab-btn.active { background: white; color: #4a90d9; border-top: 3px solid #4a90d9; }
        .tab-content { display: none; background: white; padding: 20px;
                      border-radius: 0 0 8px 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .tab-content.active { display: block; }
        .epa-section { background: #fafafa; padding: 15px 20px; border-radius: 8px;
                      margin-top: 20px; border: 1px solid #eee; }
        .epa-section h3 { margin-top: 0; color: #555; }
        .epa-table { width: 100%; border-collapse: collapse; font-size: 0.95em; }
        .epa-table th, .epa-table td { padding: 6px 10px; border: 1px solid #e4e4e4; text-align: right; }
        .epa-table th:first-child, .epa-table td:first-child { text-align: left; }
        .epa-table th { background: #f1f1f1; font-weight: 600; color: #555; }
        .epa-table tbody tr:nth-child(odd) { background: #fdfdfd; }
        .epa-footer { margin: 10px 0 0; color: #888; font-size: 0.8em; font-style: italic; }
    </style>
    <script>
        function openChannel(evt, channelId) {
            var tabcontent = document.getElementsByClassName("tab-content");
            for (var i = 0; i < tabcontent.length; i++) tabcontent[i].classList.remove("active");
            var tablinks = document.getElementsByClassName("tab-btn");
            for (var i = 0; i < tablinks.length; i++) tablinks[i].classList.remove("active");
            document.getElementById(channelId).classList.add("active");
            evt.currentTarget.classList.add("active");
            window.dispatchEvent(new Event('resize'));
        }
    </script>
</head>
<body>
    <div class="container">
"""
        f"        <h1>{page_title}</h1>\n"
    ]

    # --- Summary table ---
    html_parts.append('<div class="plot-container">\n<h2>Summary</h2>\n')
    html_parts.append('<table class="summary-table">\n<thead><tr>')
    has_gd_summary = any(
        (data.get("metadata") or {}).get("group_delay") for _, data in mode_datasets
    )
    has_auto_summary = any("_auto" in mode_name for mode_name, _ in mode_datasets)
    html_parts.append(
        "<th>Mode</th><th>Loss</th>"
        '<th title="Flat loss before EQ (always computed via compute_flat_loss, regardless of which objective was minimized)">Pre flat-loss</th>'
        '<th title="Flat loss after EQ (always computed via compute_flat_loss, regardless of which objective was minimized)">Post flat-loss</th>'
        "<th>Improvement</th>"
        "<th>EPA Pref (pre)</th><th>EPA Pref (post)</th><th>EPA Δ</th>"
    )
    if has_auto_summary:
        html_parts.append(
            '<th title="Per-channel count of emitted EQ filters; ranges show min-max across channels">EQ filters</th>'
        )
    if has_gd_summary:
        html_parts.append(
            '<th title="Group-delay optimization advisory">GD advisory</th>'
            '<th title="Whether GD controls were inserted into exported DSP">GD applied</th>'
            '<th title="Summed in-band GD RMS before and after GD optimization">GD RMS</th>'
            '<th title="GD RMS improvement, 20*log10(pre/post)">GD Δ</th>'
            '<th title="Total emitted all-pass filters; coh is mean in-band coherence">GD AP/coh</th>'
        )
    html_parts.append("</tr></thead>\n<tbody>\n")

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

        html_parts.append(
            f'<tr><td style="color:{color};font-weight:600">{_mode_label(mode_name)}</td>'
            f"{loss_cell}"
            f"<td>{pre:.4f}</td><td>{post:.4f}</td>"
            f'<td class="improvement">{improv:.1f}%</td>'
            f"{epa_cells}"
            f"{_eq_filter_summary_html(data) if has_auto_summary else ''}"
            f"{_gd_summary_cells_html(meta) if has_gd_summary else ''}</tr>\n"
        )
    html_parts.append("</tbody></table>\n")

    # When the report mixes loss functions, clarify what the Pre/Post
    # flat-loss columns actually measure — readers might otherwise
    # assume an EPA-loss run's "score" reflects the EPA composite.
    distinct_losses = sorted({lt for lt in loss_types if lt})
    if len(distinct_losses) >= 2:
        html_parts.append(
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
        html_parts.append(
            '<p style="color:#555;font-size:0.9em;margin-top:10px;">'
            "GD columns come from <code>metadata.group_delay</code>. "
            "A non-success advisory is expected for recordings that do not "
            "carry coherence or independent sweep realisations; it means the "
            "production safety gates downgraded or skipped the requested GD path."
            "</p>\n"
        )
    if has_auto_summary:
        html_parts.append(
            '<p style="color:#555;font-size:0.9em;margin-top:10px;">'
            "Auto columns show emitted EQ filter counts from the output JSON. "
            "Resolved automatic Q and gain bounds are logged by roomeq during the run "
            "but are not currently persisted in comparison JSON."
            "</p>\n"
        )

    # Score bar chart (passes loss_types so the chart can label/warn correctly)
    score_fig = create_score_comparison_figure(mode_scores, loss_types)
    html_parts.append(score_fig.to_html(full_html=False, include_plotlyjs=False))
    html_parts.append("</div>\n")

    # --- Per-channel tabs ---
    html_parts.append('<div class="tabs-container">\n<div class="tab-header">\n')
    for i, ch_name in enumerate(sorted_channels):
        active = " active" if i == 0 else ""
        html_parts.append(
            f'    <button class="tab-btn{active}" '
            f"""onclick="openChannel(event, 'ch_{i}')">{ch_name}</button>\n"""
        )
    html_parts.append("</div>\n")

    for i, ch_name in enumerate(sorted_channels):
        active = " active" if i == 0 else ""
        html_parts.append(f'<div id="ch_{i}" class="tab-content{active}">\n')
        is_lr = (ch_name == LR_SUM_CHANNEL)
        includes_redirected_bass = not is_lr and any(
            _has_redirected_bass_route(data, ch_name) for _, data in mode_datasets
        )
        source_label = _comparison_source_label(ch_name, includes_redirected_bass)
        if is_lr:
            html_parts.append(
                "<h2>Logical source L+R (complex sum)</h2>\n"
                '<p style="color:#666;font-size:0.9em;margin:-10px 0 15px 0;">'
                "Coherent (phase-aware) sum of the L and R frequency "
                "responses, computed per mode from the channel curves "
                "in the JSON. Useful for spotting room-mode coupling and "
                "centre-channel coloration that is invisible when L and R "
                "are inspected separately."
                "</p>\n"
            )
        else:
            html_parts.append(f"<h2>{source_label}</h2>\n")
            if includes_redirected_bass:
                html_parts.append(
                    '<p style="color:#666;font-size:0.9em;margin:-10px 0 15px 0;">'
                    "The response traces are the coherent microphone prediction for this "
                    "input source: its high-passed physical speaker output plus its "
                    "low-passed route through the physical subwoofer. They are not the "
                    f"isolated {ch_name} loudspeaker output."
                    "</p>\n"
                )

        # Build mode_data for this channel. The synthetic L+R channel
        # is computed on the fly from each mode's L and R curves; all
        # other channels are read straight out of the JSON.
        mode_data: list[tuple[str, dict]] = []
        for mode_name, data in mode_datasets:
            channels = comparison_channels[id(data)]
            if is_lr:
                lr = synthesize_lr_channel(channels.get("L"), channels.get("R"))
                if lr:
                    mode_data.append((mode_name, lr))
            else:
                ch_data = channels.get(ch_name, {})
                if ch_data:
                    mode_data.append((mode_name, ch_data))

        if not mode_data:
            html_parts.append("<p>No data for this channel.</p>\n</div>\n")
            continue

        # 1. Overlay plot (full range) + Zoomed (bass)
        fig_overlay = create_comparison_overlay_figure(source_label, mode_data)
        fig_zoom = create_comparison_zoomed_figure(source_label, mode_data)
        html_parts.append('<div class="plot-row">\n')
        html_parts.append(f'<div class="plot-container">{fig_overlay.to_html(full_html=False, include_plotlyjs=False)}</div>\n')
        html_parts.append(f'<div class="plot-container">{fig_zoom.to_html(full_html=False, include_plotlyjs=False)}</div>\n')
        html_parts.append("</div>\n")

        # 2. Phase before/after per mode
        fig_phase = create_comparison_phase_figure(ch_name, mode_data)
        if fig_phase:
            html_parts.append(f'<div class="plot-container">{fig_phase.to_html(full_html=False, include_plotlyjs=False)}</div>\n')

        # 3. Group delay before/after per mode
        fig_gd = create_comparison_group_delay_figure(ch_name, mode_data)
        if fig_gd:
            html_parts.append(f'<div class="plot-container">{fig_gd.to_html(full_html=False, include_plotlyjs=False)}</div>\n')

        # 4. Impulse response before/after per mode
        fig_ir = create_comparison_ir_figure(ch_name, mode_data)
        if fig_ir:
            html_parts.append(f'<div class="plot-container">{fig_ir.to_html(full_html=False, include_plotlyjs=False)}</div>\n')

        # 5. Per-mode subplots
        fig_subplots = create_mode_subplots_figure(ch_name, mode_data)
        html_parts.append(f'<div class="plot-container">{fig_subplots.to_html(full_html=False, include_plotlyjs=False)}</div>\n')

        # 6. EQ response overlay
        fig_eq = create_comparison_eq_overlay_figure(ch_name, mode_data)
        if fig_eq:
            html_parts.append(f'<div class="plot-container">{fig_eq.to_html(full_html=False, include_plotlyjs=False)}</div>\n')

        # 7. EPA psychoacoustic scores per mode
        epa_html = _epa_comparison_table_html(ch_name, mode_datasets)
        if epa_html:
            html_parts.append(epa_html)

        html_parts.append("</div>\n")

    html_parts.append("</div>\n</div>\n</body>\n</html>\n")

    with open(output_path, "w") as f:
        f.write("".join(html_parts))

    print(f"Comparison report written to: {output_path}")
