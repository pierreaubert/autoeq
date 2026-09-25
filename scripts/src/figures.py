"""Schema section builders for roomeq visualization.

Ported from Plotly: the same data logic, colors, titles, and default
smoothing, emitting ``autoeq-report-data-v1`` render-only section dicts
(see ``crates/autoeq-report-wasm/SCHEMA.md``) assembled by
:mod:`scripts.src.wasm_report`. Plotly-only interactivity (smoothing
dropdowns, Linear/dB toggles, subplot grids, hover customdata, marker
symbols) is folded into static equivalents documented per builder.
"""

import math

from . import DEFAULT_SMOOTHING
from .dsp import (
    smooth_octave,
    resample_spl_onto_grid,
    compute_eq_response,
    compute_group_delay,
    compute_group_delay_from_ir,
    generate_freq_points,
    build_post_dsp_source_curves,
    per_driver_effective_eq,
    wrap_phase,
)
from .data_extract import (
    channel_has_eq,
    clip_curve_to_measured_band,
    compute_y_range,
    compute_average_spl_in_range,
    driver_display_names,
    extract_eq_passes,
    get_all_crossover_frequencies,
    get_channel_sort_key,
    get_plottable_drivers,
)
from .target_overlay import build_target_overlay_curves
from .wasm_report import (
    axis,
    series,
    hline,
    vline,
    figure,
    bar_chart,
    sankey_chart,
)

def _align_final_to_initial_grid(
    final_curve: dict | None, freq_data: list[float] | None
) -> tuple[list[float] | None, list[float] | None]:
    """Return ``(plot_freq, spl)`` for an After-EQ trace sharing ``freq_data``.

    Routed outputs legitimately mix grids (driver-measurement baselines vs.
    deployed replay grids); resampling keeps the trace and the smoothing
    dropdown on one x-axis instead of failing on length mismatches.
    """
    if not final_curve:
        return None, None
    if freq_data is None:
        freq_data = final_curve["freq"]
    spl_raw = resample_spl_onto_grid(
        final_curve["freq"], final_curve["spl"], freq_data
    )
    return freq_data, spl_raw


def create_channel_figure(
    channel_name: str,
    initial_curve: dict | None,
    final_curve: dict | None,
    title_suffix: str = "",
    tab=None,
):
    """Build a channel Before/After EQ section (default smoothing only)."""
    freq_data = None
    series_list = []

    # Add initial curve (before EQ)
    if initial_curve:
        freq_data = initial_curve["freq"]
        spl_raw = initial_curve["spl"]
        spl_smoothed = smooth_octave(freq_data, spl_raw, DEFAULT_SMOOTHING)
        series_list.append(
            series(
                "Before EQ",
                freq_data,
                spl_smoothed,
                color="rgba(255, 100, 100, 0.8)",
                width=2,
            )
        )

    # Add final curve (after EQ)
    plot_freq, spl_raw = _align_final_to_initial_grid(final_curve, freq_data)
    if final_curve:
        if freq_data is None:
            freq_data = plot_freq
        spl_smoothed = smooth_octave(freq_data, spl_raw, DEFAULT_SMOOTHING)
        series_list.append(
            series(
                "After EQ",
                freq_data,
                spl_smoothed,
                color="rgba(100, 200, 100, 0.9)",
                width=2,
            )
        )

    # Compute dynamic y-range
    y_min, y_max = compute_y_range([initial_curve, final_curve])

    return figure(
        f"Channel: {channel_name}{title_suffix}",
        axis("Frequency (Hz)", "log", 20, 20000),
        axis("SPL (dB)", "linear", y_min, y_max),
        series_list=series_list,
        tab=tab,
    )


def add_channel_response_overlays(
    fig,
    channel_name: str,
    target_curve: dict | None,
    lfe_plus_channel_curve: dict | None,
):
    """Append Target/LFE series to a channel figure section dict."""
    if target_curve and target_curve.get("freq") and target_curve.get("spl"):
        fig["figure"]["series"].append(
            series(
                "Target",
                target_curve["freq"],
                target_curve["spl"],
                color="rgba(40, 40, 40, 0.9)",
                width=2,
                dash="dot",
            )
        )

    if (
        lfe_plus_channel_curve
        and lfe_plus_channel_curve.get("freq")
        and lfe_plus_channel_curve.get("spl")
    ):
        frequencies = lfe_plus_channel_curve["freq"]
        spl = smooth_octave(
            frequencies,
            lfe_plus_channel_curve["spl"],
            DEFAULT_SMOOTHING,
        )
        fig["figure"]["series"].append(
            series(
                f"LFE + {channel_name}",
                frequencies,
                spl,
                color="rgba(148, 103, 189, 0.95)",
                width=2.5,
            )
        )

    return fig


def create_zoomed_figure(
    channel_name: str,
    initial_curve: dict | None,
    final_curve: dict | None,
    min_freq: float = 20.0,
    max_freq: float = 1200.0,
    y_range: float = 10.0,
    tab=None,
):
    """Build a zoomed channel section (20-1200Hz, centered y-axis)."""
    # Compute average SPL for centering (use final curve if available, else initial)
    ref_curve = final_curve if final_curve else initial_curve
    avg_spl = (
        compute_average_spl_in_range(ref_curve, min_freq, max_freq)
        if ref_curve
        else 0.0
    )

    freq_data = None
    series_list = []

    # Add initial curve (before EQ)
    if initial_curve:
        freq_data = initial_curve["freq"]
        spl_raw = initial_curve["spl"]
        spl_smoothed = smooth_octave(freq_data, spl_raw, DEFAULT_SMOOTHING)
        series_list.append(
            series(
                "Before EQ",
                freq_data,
                spl_smoothed,
                color="rgba(255, 100, 100, 0.8)",
                width=2,
            )
        )

    # Add final curve (after EQ)
    plot_freq, spl_raw = _align_final_to_initial_grid(final_curve, freq_data)
    if final_curve:
        if freq_data is None:
            freq_data = plot_freq
        spl_smoothed = smooth_octave(freq_data, spl_raw, DEFAULT_SMOOTHING)
        series_list.append(
            series(
                "After EQ",
                freq_data,
                spl_smoothed,
                color="rgba(100, 200, 100, 0.9)",
                width=2,
            )
        )

    # Add target line at average
    series_list.append(
        series(
            f"Average ({avg_spl:.1f} dB)",
            [min_freq, max_freq],
            [avg_spl, avg_spl],
            color="rgba(150, 150, 150, 0.5)",
            width=1,
            dash="dash",
        )
    )

    return figure(
        f"Channel: {channel_name} (Zoom {int(min_freq)}-{int(max_freq)} Hz)",
        axis("Frequency (Hz)", "log", min_freq, max_freq),
        axis("SPL (dB)", "linear", avg_spl - y_range, avg_spl + y_range),
        series_list=series_list,
        tab=tab,
    )


def create_eq_figure(
    channel_name: str,
    eq_filters: list[dict],
    eq_response_data: dict | None = None,
    sample_rate: float = 48_000.0,
    tab=None,
):
    """Build an EQ frequency-response section.

    Args:
        channel_name: Name of the channel.
        eq_filters: List of EQ filter dicts (for individual filter decomposition).
        eq_response_data: Optional pre-computed EQ response from JSON output
            (with 'freq' and 'spl' keys). When provided, used for the combined
            EQ curve instead of recomputing from biquad filters.
    """
    if not eq_filters and not eq_response_data:
        return None

    # Use pre-computed EQ response from JSON if available, otherwise compute from filters
    if eq_response_data and "freq" in eq_response_data and "spl" in eq_response_data:
        freq_points = eq_response_data["freq"]
        eq_response = eq_response_data["spl"]
    else:
        freq_points = generate_freq_points(20.0, min(20000.0, sample_rate / 2), 500)
        freq_points = [min(f, sample_rate / 2) for f in freq_points]
        eq_response = compute_eq_response(eq_filters, freq_points, sample_rate)

    if not eq_response:
        return None

    series_list = []

    # Add combined EQ response
    series_list.append(
        series(
            "Combined EQ",
            freq_points,
            eq_response,
            color="rgba(100, 100, 255, 0.9)",
            width=2,
        )
    )

    # Add individual filter responses
    colors = [
        "rgba(255, 150, 150, 0.6)",
        "rgba(150, 255, 150, 0.6)",
        "rgba(150, 150, 255, 0.6)",
        "rgba(255, 255, 150, 0.6)",
        "rgba(255, 150, 255, 0.6)",
        "rgba(150, 255, 255, 0.6)",
        "rgba(200, 200, 200, 0.6)",
    ]

    for i, filt in enumerate(eq_filters):
        single_response = compute_eq_response([filt], freq_points, sample_rate)
        freq = filt.get("freq", 0)
        gain = filt.get("db_gain", 0)
        filter_type = filt.get("filter_type", "peak")
        if filt.get("topology") == "kautz_filter":
            filter_label = "KAUTZ bank (unity + weighted basis)"
        else:
            prefix = "WARPED " if filt.get("topology") == "warped_biquad" else ""
            filter_label = f"{prefix}{filter_type.upper()} {freq:.0f}Hz {gain:+.1f}dB"

        series_list.append(
            series(
                filter_label,
                freq_points,
                single_response,
                color=colors[i % len(colors)],
                width=1,
                dash="dot",
            )
        )

    # Add 0 dB reference line
    series_list.append(
        series(
            "0 dB",
            [freq_points[0], freq_points[-1]],
            [0, 0],
            color="rgba(150, 150, 150, 0.5)",
            width=1,
            dash="dash",
        )
    )

    # Compute y-range from EQ response
    y_limit: int | None = None
    y_min = -15
    y_max = 15

    if eq_response:
        is_lfe = "lfe" in channel_name.lower()
        if is_lfe:
            lfe_spl = []
            for f, s in zip(freq_points, eq_response):
                if 20 <= f <= 200:
                    lfe_spl.append(s)
            if lfe_spl:
                max_abs = max(abs(min(lfe_spl)), abs(max(lfe_spl)))
                y_limit = max(10, math.ceil(max_abs / 5) * 5 + 5)
            else:
                y_limit = 15
        else:
            eq_max = max(eq_response)
            eq_min = min(eq_response)
            y_max = math.ceil(eq_max / 5) * 5
            y_min = math.floor(eq_min / 5) * 5
            y_min = max(y_min, y_max - 50)
            y_limit = None
    else:
        y_limit = 15

    # Compute y_range for plot
    if y_limit is not None:
        plot_y_range = [-y_limit, y_limit]
    else:
        plot_y_range = [y_min, y_max]

    return figure(
        f"EQ Response: {channel_name}",
        axis("Frequency (Hz)", "log", 20, 20000),
        axis("Gain (dB)", "linear", plot_y_range[0], plot_y_range[1]),
        series_list=series_list,
        tab=tab,
    )


def create_multipass_eq_figure(
    channel_name: str,
    channel_data: dict,
    eq_response_data: dict | None = None,
    sample_rate: float = 48_000.0,
    tab=None,
):
    """Build a per-pass EQ-responses section for the 3-pass pipeline.

    When labeled passes exist (cea2034_speaker_correction, user_preference),
    shows each pass as a distinct colored curve plus the combined response.
    Falls back to the standard create_eq_figure for single-pass configs.

    Args:
        channel_name: Name of the channel.
        channel_data: Full channel data dict from roomeq output.
        eq_response_data: Optional pre-computed combined EQ response.
    """
    passes = extract_eq_passes(channel_data)

    if not passes:
        return None

    # If only one unlabeled pass, fall back to standard EQ figure
    has_labeled = any(p["label"] for p in passes)
    if not has_labeled:
        all_filters = []
        for p in passes:
            all_filters.extend(p["filters"])
        return create_eq_figure(
            channel_name, all_filters, eq_response_data, sample_rate, tab=tab
        )

    freq_points = generate_freq_points(20.0, min(20000.0, sample_rate / 2), 500)
    freq_points = [min(f, sample_rate / 2) for f in freq_points]
    series_list = []

    # Collect all filters for the combined response
    all_filters = []
    for p in passes:
        all_filters.extend(p["filters"])

    # Combined EQ response (from pre-computed data or calculated)
    if eq_response_data and "freq" in eq_response_data and "spl" in eq_response_data:
        combined_freq = eq_response_data["freq"]
        combined_response = eq_response_data["spl"]
    else:
        combined_freq = freq_points
        combined_response = compute_eq_response(all_filters, freq_points, sample_rate)

    if combined_response:
        series_list.append(
            series(
                "Combined (all passes)",
                combined_freq,
                combined_response,
                color="rgba(50, 50, 50, 0.9)",
                width=2.5,
            )
        )

    # Per-pass responses
    for p in passes:
        pass_response = compute_eq_response(p["filters"], freq_points, sample_rate)
        if pass_response:
            series_list.append(
                series(
                    p["display_name"],
                    freq_points,
                    pass_response,
                    color=p["color"],
                    width=2,
                )
            )

    # 0 dB reference
    series_list.append(
        series(
            "0 dB",
            [freq_points[0], freq_points[-1]],
            [0, 0],
            color="rgba(150, 150, 150, 0.5)",
            width=1,
            dash="dash",
        )
    )

    # Y-range
    y_min, y_max = -15, 15
    if combined_response:
        eq_max = max(combined_response)
        eq_min = min(combined_response)
        y_max = math.ceil(eq_max / 5) * 5
        y_min = math.floor(eq_min / 5) * 5
        y_min = max(y_min, y_max - 50)

    return figure(
        f"EQ Response (3-Pass): {channel_name}",
        axis("Frequency (Hz)", "log", 20, 20000),
        axis("Gain (dB)", "linear", y_min, y_max),
        series_list=series_list,
        tab=tab,
    )


_IR_FLOOR_DB = -80.0


def _to_db(amplitude: list[float]) -> list[float]:
    """Convert linear amplitude to dB, floored at _IR_FLOOR_DB."""
    return [
        max(20.0 * math.log10(abs(a)), _IR_FLOOR_DB) if abs(a) > 1e-10 else _IR_FLOOR_DB
        for a in amplitude
    ]


def create_ir_figure(
    channel_name: str,
    pre_ir: dict | None,
    post_ir: dict | None,
    display_ms: float = 100.0,
    tab=None,
):
    """Build a pre-/post-correction impulse-response section (Linear default).

    Args:
        channel_name: Name of the channel.
        pre_ir: Dict with 'time_ms' and 'amplitude' keys (before correction).
        post_ir: Dict with 'time_ms' and 'amplitude' keys (after correction).
        display_ms: Initial x-axis range in milliseconds (default 100 ms).
    """
    if not pre_ir and not post_ir:
        return None

    series_list = []

    if pre_ir:
        series_list.append(
            series(
                "Before EQ",
                pre_ir["time_ms"],
                pre_ir["amplitude"],
                color="rgba(255, 100, 100, 0.8)",
                width=1,
            )
        )

    if post_ir:
        series_list.append(
            series(
                "After EQ",
                post_ir["time_ms"],
                post_ir["amplitude"],
                color="rgba(100, 200, 100, 0.9)",
                width=1,
            )
        )

    # 0 reference line
    ref_lines = [hline(0, "rgba(150, 150, 150, 0.4)", width=1, dash="dash")]

    return figure(
        f"Impulse Response: {channel_name}",
        axis("Time (ms)", "linear", 0, display_ms),
        axis("Amplitude (normalized)", "linear", -1.1, 1.1),
        series_list=series_list,
        hlines=ref_lines,
        tab=tab,
    )
def _get_driver_initial_curves(channel_data: dict) -> list[tuple[str, dict]] | None:
    """Extract per-driver initial curves from a channel's driver chains.

    Returns:
        List of (driver_name, curve_data) tuples, or None if no per-driver curves exist.
        Each curve_data has "freq" and "spl" keys.
    """
    drivers = channel_data.get("drivers", [])
    if not drivers:
        return None

    result = []
    for driver in drivers:
        initial_curve = driver.get("initial_curve")
        if initial_curve and "freq" in initial_curve and "spl" in initial_curve:
            name = driver.get("name", f"driver_{driver.get('index', '?')}")
            # Raw measurements stop at the recorded band; the stored curve
            # continues past it for DSP replay and must not be drawn as data.
            initial_curve = clip_curve_to_measured_band(initial_curve, driver)
            result.append((name, initial_curve))

    return result if result else None


def _driver_trace_kwargs(driver_index: int, color: str, point_count: int) -> dict:
    """Return line-style kwargs distinguishing drivers within one channel.

    Render-only port of the old trace-style helper: marker symbols are dropped, so
    every driver is a plain solid line in the channel color (width 2).
    Drivers stay distinguishable via their ``Original:`` / ``EQ:`` series
    names. ``point_count`` is kept for signature compatibility.
    """
    return {"color": color, "width": 2.0, "dash": "solid"}


def create_combined_figure(data: dict, json_path=None, tab=None) -> dict:
    """Create a combined overview figure section (original + EQ + corrected).

    Args:
        data: Output JSON data (roomeq result with correction filters)
        json_path: Path to output JSON (used to resolve relative target files)
        tab: Tab label passed into the emitted section.
    """
    channels_dict = data.get("channels", {})

    x_axis = axis("Frequency (Hz)", "log", 20, 20000)
    if not channels_dict:
        print("Warning: No channels found in the JSON file")
        return figure(
            "Combined Overview",
            x_axis,
            axis("SPL (dB)", "linear", -20, 30),
            series_list=[],
            tab=tab,
        )

    # Sort channels by classical order
    sorted_channel_names = sorted(channels_dict.keys(), key=get_channel_sort_key)
    channels = [(name, channels_dict[name]) for name in sorted_channel_names]

    # Color palette for channels
    channel_colors = [
        "rgba(31, 119, 180, 0.9)",  # blue
        "rgba(255, 127, 14, 0.9)",  # orange
        "rgba(44, 160, 44, 0.9)",  # green
        "rgba(214, 39, 40, 0.9)",  # red
        "rgba(148, 103, 189, 0.9)",  # purple
        "rgba(140, 86, 75, 0.9)",  # brown
        "rgba(227, 119, 194, 0.9)",  # pink
        "rgba(127, 127, 127, 0.9)",  # gray
        "rgba(188, 189, 34, 0.9)",  # olive
        "rgba(23, 190, 207, 0.9)",  # cyan
    ]

    # Generate frequency points for EQ response
    freq_points = generate_freq_points(20.0, 20000.0, 500)

    # Collect all curves for y-range computation
    all_initial_curves: list[dict | None] = []
    all_corrected_curves: list[dict | None] = []

    # Per-driver data for channels that are speaker groups
    per_driver_initial: dict[str, list[tuple[str, dict]]] = {}

    post_dsp_curves = build_post_dsp_source_curves(data)
    target_curves = build_target_overlay_curves(data, post_dsp_curves, json_path)
    for channel_name, channel_data in channels:
        driver_curves = _get_driver_initial_curves(channel_data)
        if driver_curves:
            per_driver_initial[channel_name] = driver_curves
            for _, dcurve in driver_curves:
                all_initial_curves.append(dcurve)
        else:
            all_initial_curves.append(channel_data.get("initial_curve"))

        final_curve = post_dsp_curves.get(channel_name)
        if final_curve and "freq" in final_curve and "spl" in final_curve:
            all_corrected_curves.append(final_curve)
        target_curve = target_curves.get(channel_name)
        if target_curve and "freq" in target_curve and "spl" in target_curve:
            all_corrected_curves.append(target_curve)

    # Compute y-ranges
    spl_y_min, spl_y_max = compute_y_range(
        all_initial_curves + all_corrected_curves
    )

    # EQ traces (row 2 of the original). Multi-driver channels (e.g. two
    # subwoofers sharing one LFE bus) expand into one effective per-driver
    # shaping curve so each physical sub is visible instead of a single
    # collapsed channel trace. Entries are (trace name, freq, spl, color
    # index, driver index or None).
    eq_row_traces: list[tuple[str, list, list, int, int | None]] = []
    for channel_index, (channel_name, channel_data) in enumerate(channels):
        drivers = get_plottable_drivers(channel_data)
        if drivers and channel_has_eq(channel_data):
            labels = driver_display_names(data, channel_name)
            expanded = False
            for driver_index in range(len(drivers)):
                effective = per_driver_effective_eq(data, channel_name, driver_index)
                if (
                    effective
                    and effective.get("freq")
                    and effective.get("spl")
                ):
                    if driver_index < len(labels):
                        label = labels[driver_index]
                    else:
                        label = f"{channel_name}/{driver_index}"
                    eq_row_traces.append(
                        (
                            f"EQ: {label}",
                            effective["freq"],
                            effective["spl"],
                            channel_index,
                            driver_index,
                        )
                    )
                    expanded = True
            if expanded:
                continue
        eq_response_data = channel_data.get("eq_response")
        if (
            eq_response_data
            and "freq" in eq_response_data
            and "spl" in eq_response_data
        ):
            eq_row_traces.append(
                (
                    f"EQ: {channel_name}",
                    eq_response_data["freq"],
                    eq_response_data["spl"],
                    channel_index,
                    None,
                )
            )
        else:
            plugins = channel_data.get("plugins", [])
            eq_filters = []
            for plugin in plugins:
                if plugin.get("plugin_type") == "eq":
                    filters = plugin.get("parameters", {}).get("filters", [])
                    eq_filters.extend(filters)
            if eq_filters:
                eq_resp = compute_eq_response(eq_filters, freq_points)
                eq_row_traces.append(
                    (
                        f"EQ: {channel_name}",
                        freq_points,
                        eq_resp,
                        channel_index,
                        None,
                    )
                )

    # Compute EQ y-range
    all_eq_values: list[float] = []
    for _, _, eq_spl_values, _, _ in eq_row_traces:
        all_eq_values.extend(eq_spl_values)

    if all_eq_values:
        eq_y_upper = min(20, math.ceil(max(all_eq_values) / 5) * 5 + 5)
        eq_y_lower = max(-20, math.floor(min(all_eq_values) / 5) * 5 - 5)
    else:
        eq_y_upper = 15
        eq_y_lower = -15

    spl_series: list[dict] = []
    eq_series: list[dict] = []

    # --- Original curves (default smoothing only; no dropdown) ---
    for i, (channel_name, channel_data) in enumerate(channels):
        color = channel_colors[i % len(channel_colors)]

        if channel_name in per_driver_initial:
            for d_idx, (driver_name, dcurve) in enumerate(
                per_driver_initial[channel_name]
            ):
                spl_smoothed = smooth_octave(
                    dcurve["freq"], dcurve["spl"], DEFAULT_SMOOTHING
                )
                style = _driver_trace_kwargs(d_idx, color, len(spl_smoothed))
                spl_series.append(
                    series(
                        f"Original: {channel_name}/{driver_name}",
                        dcurve["freq"],
                        spl_smoothed,
                        color=style["color"],
                        width=style["width"],
                        dash=style["dash"],
                    )
                )
        else:
            initial_curve = channel_data.get("initial_curve")
            if initial_curve:
                spl_smoothed = smooth_octave(
                    initial_curve["freq"],
                    initial_curve["spl"],
                    DEFAULT_SMOOTHING,
                )
                spl_series.append(
                    series(
                        f"Original: {channel_name}",
                        initial_curve["freq"],
                        spl_smoothed,
                        color=color,
                        width=2.0,
                    )
                )

    # --- EQ responses on the secondary Gain axis (unsmoothed) ---
    for trace_name, eq_freq, eq_spl, color_index, _driver_index in eq_row_traces:
        color = channel_colors[color_index % len(channel_colors)]
        eq_series.append(
            series(
                trace_name,
                eq_freq,
                eq_spl,
                color=color,
                width=2.0,
                y_axis=1,
            )
        )

    # --- One microphone-predicted post-DSP curve per input (smoothed) ---
    for i, (channel_name, channel_data) in enumerate(channels):
        color = channel_colors[i % len(channel_colors)]
        final_curve = post_dsp_curves.get(channel_name)

        if final_curve and "freq" in final_curve and "spl" in final_curve:
            freq = final_curve["freq"]
            spl_smoothed = smooth_octave(freq, final_curve["spl"], DEFAULT_SMOOTHING)
            spl_series.append(
                series(
                    f"Corrected: {channel_name}",
                    freq,
                    spl_smoothed,
                    color=color,
                    width=2.0,
                )
            )

    # Target overlays share each channel's color and level alignment. They
    # are deliberately not smoothed.
    for i, (channel_name, _) in enumerate(channels):
        target_curve = target_curves.get(channel_name)
        if not target_curve:
            continue
        color = channel_colors[i % len(channel_colors)]
        spl_series.append(
            series(
                f"Target: {channel_name}",
                target_curve["freq"],
                target_curve["spl"],
                color=color,
                width=2.0,
                dash="dot",
            )
        )

    # --- Crossover vertical lines (one per frequency; spanned all 3 rows) ---
    crossover_freqs = get_all_crossover_frequencies(data)
    vlines = []
    for xover_freq in crossover_freqs:
        freq_label = (
            f"{xover_freq / 1000:.1f}k" if xover_freq >= 1000 else f"{xover_freq:.0f}"
        )
        vlines.append(
            vline(
                xover_freq,
                "rgba(180, 80, 180, 0.7)",
                dash="dashdot",
                width=1.5,
                label=f"Xover {freq_label} Hz",
            )
        )

    # --- 0 dB reference for the EQ gain axis ---
    # A schema hline always reads the primary (SPL) axis, so the EQ zero
    # line is a two-point series pinned to y2 instead.
    eq_series.append(
        series(
            "0 dB",
            [freq_points[0], freq_points[-1]],
            [0, 0],
            color="rgba(150, 150, 150, 0.5)",
            width=1,
            dash="dash",
            y_axis=1,
        )
    )

    return figure(
        "Combined Overview",
        x_axis,
        axis("SPL (dB)", "linear", spl_y_min, spl_y_max),
        series_list=spl_series + eq_series,
        vlines=vlines,
        y2=axis("Gain (dB)", "linear", eq_y_lower, eq_y_upper),
        tab=tab,
    )
def _bass_management_report(data: dict) -> dict:
    metadata = data.get("metadata") or {}
    return metadata.get("bass_management") or {}


def _route_display_name(route_kind: str) -> str:
    names = {
        "main_highpass_to_self": "Main HP",
        "redirected_bass_lowpass_to_sub": "Redirected LP",
        "lfe_lowpass_to_sub": "LFE LP",
    }
    return names.get(route_kind, route_kind.replace("_", " ").title())


def _route_color(route_kind: str, alpha: float = 0.58) -> str:
    colors = {
        "main_highpass_to_self": f"rgba(74, 144, 217, {alpha})",
        "redirected_bass_lowpass_to_sub": f"rgba(46, 204, 113, {alpha})",
        "lfe_lowpass_to_sub": f"rgba(231, 126, 34, {alpha})",
    }
    return colors.get(route_kind, f"rgba(127, 140, 141, {alpha})")


def _driver_link_color(alpha: float = 0.58) -> str:
    return f"rgba(148, 103, 189, {alpha})"


def _driver_alignment(driver: dict) -> tuple[float, float, bool]:
    """Extract the (gain_db, delay_ms, inverted) alignment of a sub driver."""
    gain_db = 0.0
    delay_ms = 0.0
    inverted = False
    for plugin in driver.get("plugins", []) or []:
        if not isinstance(plugin, dict):
            continue
        params = plugin.get("parameters", {}) or {}
        plugin_type = str(plugin.get("plugin_type", "")).lower()
        if plugin_type == "gain":
            try:
                gain_db += float(params.get("gain_db", 0.0))
            except (TypeError, ValueError):
                pass
            inverted = inverted or bool(params.get("invert", False))
        elif plugin_type == "delay":
            try:
                delay_ms += float(params.get("delay_ms", 0.0))
            except (TypeError, ValueError):
                pass
    return gain_db, delay_ms, inverted


def _driver_low_pass(driver: dict) -> tuple[float | None, str | None]:
    """Extract the deployed per-driver low-pass (LP_i) of a sub driver, if any.

    Reads the last low-pass ``crossover`` plugin in the driver chain, matching
    serial DSP order. Returns ``(frequency_hz, crossover_type)`` or
    ``(None, None)`` when the driver carries no low-pass (legacy
    single-crossover chains).
    """
    frequency: float | None = None
    crossover_type: str | None = None
    for plugin in driver.get("plugins", []) or []:
        if not isinstance(plugin, dict):
            continue
        if str(plugin.get("plugin_type", "")).lower() != "crossover":
            continue
        params = plugin.get("parameters", {}) or {}
        if str(params.get("output", "")).lower() not in ("low", "lowpass", "lp"):
            continue
        try:
            candidate = float(params.get("frequency", 0.0))
        except (TypeError, ValueError):
            continue
        if candidate > 0.0:
            frequency = candidate
            raw_type = params.get("type")
            crossover_type = str(raw_type) if raw_type is not None else None
    return frequency, crossover_type


def create_bass_management_routing_figure(data: dict, tab=None):
    """Create a Sankey section from route-level bass-management metadata.

    The #14 routed schema may emit several branches per source channel:
    a high-passed self-route plus one low-passed route per physical bass
    output.

    A physical bass output may itself fan out to several sub drivers (e.g.
    a 2.2 rig with two drivers under ``out: LFE``). One link per driver is
    drawn from the bus node so the split and each driver's alignment
    gain/delay stay visible instead of collapsing into a single output.

    Sankey customdata hover text is dropped (nodes/links only).
    Returns None when there are no routes.
    """
    report = _bass_management_report(data)
    routing_graph = report.get("routing_graph") or {}
    routes = routing_graph.get("routes") or []
    if not routes:
        return None

    node_index: dict[str, int] = {}
    labels: list[str] = []

    def add_node(prefix: str, name: str) -> int:
        key = f"{prefix}:{name}"
        if key not in node_index:
            node_index[key] = len(labels)
            labels.append(f"{prefix}: {name}")
        return node_index[key]

    sources: list[int] = []
    targets: list[int] = []
    values: list[float] = []
    colors: list[str] = []

    for route in routes:
        source = str(route.get("source_channel", "?"))
        destination = str(route.get("destination", "?"))
        route_kind = str(route.get("route_kind", "route"))
        gain_linear = route.get("gain_linear") or route.get("matrix_gain") or 1.0
        try:
            value = max(abs(float(gain_linear)), 0.05)
        except (TypeError, ValueError):
            value = 1.0

        sources.append(add_node("in", source))
        targets.append(add_node("out", destination))
        values.append(value)
        colors.append(_route_color(route_kind))

    # Multi-driver fan-out from each physical bass output bus.
    channels_dict = data.get("channels", {}) or {}
    for channel_name in sorted(channels_dict, key=get_channel_sort_key):
        channel_data = channels_dict[channel_name] or {}
        drivers = channel_data.get("drivers") or []
        if not drivers or f"out:{channel_name}" not in node_index:
            continue
        bus_node = node_index[f"out:{channel_name}"]
        for index, driver in enumerate(drivers):
            if not isinstance(driver, dict):
                continue
            driver_name = str(driver.get("name", f"driver_{index}"))
            gain_db, _delay_ms, _inverted = _driver_alignment(driver)
            try:
                value = max(10.0 ** (float(gain_db) / 20.0), 0.05)
            except (TypeError, ValueError):
                value = 1.0
            sources.append(bus_node)
            targets.append(add_node("sub", driver_name))
            values.append(value)
            colors.append(_driver_link_color())

    links = [
        (source, target, value, color)
        for source, target, value, color in zip(sources, targets, values, colors)
    ]
    return sankey_chart(
        "Bass Management Routing Graph", labels, links, tab=tab
    )


def _coerce_bar_values(entries) -> list[float]:
    """Coerce per-output bar values to floats (missing -> 0.0)."""
    coerced = []
    for entry in entries:
        if entry is None:
            coerced.append(0.0)
            continue
        try:
            coerced.append(float(entry))
        except (TypeError, ValueError):
            coerced.append(0.0)
    return coerced


def create_bass_management_headroom_figure(data: dict, tab=None):
    """Create a per-physical-output bass-bus headroom bar-chart section.

    Returns None when there is no per-output headroom simulation.
    """
    report = _bass_management_report(data)
    simulation = report.get("headroom_simulation") or {}
    per_output = simulation.get("per_output") or []
    if not per_output:
        return None

    outputs = [entry.get("output_role", f"out {i + 1}") for i, entry in enumerate(per_output)]
    rms = _coerce_bar_values([entry.get("rms_bus_gain_db") for entry in per_output])
    peak = _coerce_bar_values([entry.get("coherent_peak_gain_db") for entry in per_output])
    lfe = _coerce_bar_values([entry.get("lfe_contribution_db") for entry in per_output])
    headroom_margin = simulation.get("headroom_margin_db")

    hlines = []
    if isinstance(headroom_margin, (int, float)):
        hlines.append(
            hline(
                headroom_margin,
                "rgba(40, 40, 40, 0.7)",
                dash="dash",
                width=2.0,
                label=f"headroom limit {headroom_margin:+.1f} dB",
            )
        )

    groups = [
        ("RMS programme gain", rms, "rgba(74, 144, 217, 0.75)"),
        ("Coherent peak gain", peak, "rgba(231, 76, 60, 0.78)"),
        ("LFE contribution", lfe, "rgba(230, 126, 34, 0.72)"),
    ]
    return bar_chart(
        "Bass Bus Headroom Simulation",
        outputs,
        groups,
        ylabel="Gain vs programme reference (dB)",
        hlines=hlines,
        tab=tab,
    )
# ============================================================================
# Mode color/label tables (copied verbatim from figures.py)
# ============================================================================

# Color scheme: distinct colours from the matplotlib tab20 family so
# each processing/loss/auto/GD scenario stands out on busy overlay plots.
MODE_COLORS: dict[str, str] = {
    "iir": "#1f77b4",              # blue
    "iir_epa": "#ff7f0e",          # orange
    "fir": "#2ca02c",              # green
    "fir_epa": "#d62728",          # red
    "hybrid": "#9467bd",           # purple
    "hybrid_epa": "#8c564b",       # brown
    "mixed_phase": "#e377c2",      # pink
    "mixed_phase_epa": "#17becf",  # cyan
    "iir_auto_filters": "#393b79",       # indigo
    "iir_auto_bounds": "#637939",        # olive green
    "iir_auto_all": "#8c6d31",           # ochre
    "mixed_phase_auto_all": "#843c39",   # maroon
    "iir_gd_safety_gate": "#ad494a",     # muted red
    "iir_gd_delay_only": "#bcbd22",        # olive
    "iir_gd_fixed_allpass": "#7f7f7f",     # gray
    "iir_gd_adaptive_allpass": "#1f9e89",  # teal
    "fir_gd_phase_linear": "#ff9896",      # salmon
    "mixed_phase_gd": "#c5b0d5",           # lavender
}

# Display names: every mode includes its loss function in the label so
# legends, summary tables, and per-mode subplot titles always show
# which objective the run minimized at a glance.
MODE_DISPLAY_NAMES: dict[str, str] = {
    "iir": "IIR (flat)",
    "iir_epa": "IIR (EPA)",
    "fir": "FIR (flat)",
    "fir_epa": "FIR (EPA)",
    "hybrid": "Hybrid (flat)",
    "hybrid_epa": "Hybrid (EPA)",
    "mixed_phase": "MixedPhase (flat)",
    "mixed_phase_epa": "MixedPhase (EPA)",
    "iir_auto_filters": "IIR auto filters",
    "iir_auto_bounds": "IIR auto bounds",
    "iir_auto_all": "IIR auto all",
    "mixed_phase_auto_all": "MixedPhase auto all",
    "iir_gd_safety_gate": "IIR GD safety gate",
    "iir_gd_delay_only": "IIR GD delay-only",
    "iir_gd_fixed_allpass": "IIR GD fixed AP",
    "iir_gd_adaptive_allpass": "IIR GD adaptive AP",
    "fir_gd_phase_linear": "FIR GD target",
    "mixed_phase_gd": "MixedPhase + GD",
}
def _mode_color(mode_name: str) -> str:
    return MODE_COLORS.get(mode_name, "#888888")


def _mode_label(mode_name: str) -> str:
    return MODE_DISPLAY_NAMES.get(mode_name, mode_name)


def create_comparison_overlay_figure(
    channel_name: str,
    mode_data: list[tuple[str, dict]],
    title_suffix: str = "",
    target_curve: dict | None = None,
    tab=None,
):
    """Overlay final curves from multiple modes on the same plot.

    ``target_curve`` is the shared design target (freq/spl); compared modes
    normally share one fixture target. When absent no target is drawn —
    a fabricated flat line would misstate the design slope.
    """
    series_list = []

    initial_curve = None
    for _, ch_data in mode_data:
        initial_curve = ch_data.get("initial_curve")
        if initial_curve:
            break

    all_curves: list[dict | None] = [initial_curve]

    if initial_curve:
        spl_smoothed = smooth_octave(
            initial_curve["freq"], initial_curve["spl"], DEFAULT_SMOOTHING
        )
        series_list.append(series(
            "Before EQ",
            initial_curve["freq"], spl_smoothed,
            color="rgba(200, 200, 200, 0.6)", width=2,
        ))

    for mode_name, ch_data in mode_data:
        final_curve = ch_data.get("final_curve")
        all_curves.append(final_curve)
        if final_curve:
            spl_smoothed = smooth_octave(
                final_curve["freq"], final_curve["spl"], DEFAULT_SMOOTHING
            )
            series_list.append(series(
                _mode_label(mode_name),
                final_curve["freq"], spl_smoothed,
                color=_mode_color(mode_name), width=2,
            ))

    if target_curve and target_curve.get("freq") and target_curve.get("spl"):
        series_list.append(series(
            "Target",
            target_curve["freq"], target_curve["spl"],
            color="rgba(40, 40, 40, 0.9)", width=2, dash="dot",
        ))
        all_curves.append(target_curve)

    y_min, y_max = compute_y_range(all_curves)

    return figure(
        f"{channel_name}: Mode Comparison{title_suffix}",
        axis("Frequency (Hz)", "log", 20, 20000),
        axis("SPL (dB)", "linear", y_min, y_max),
        series_list=series_list,
        tab=tab,
    )


def create_comparison_zoomed_figure(
    channel_name: str,
    mode_data: list[tuple[str, dict]],
    min_freq: float = 20.0,
    max_freq: float = 500.0,
    y_half_range: float = 12.0,
    tab=None,
):
    """Zoomed overlay of final curves (bass region) from multiple modes."""
    series_list = []

    initial_curve = None
    for _, ch_data in mode_data:
        initial_curve = ch_data.get("initial_curve")
        if initial_curve:
            break

    ref_curve = None
    for _, ch_data in mode_data:
        ref_curve = ch_data.get("final_curve")
        if ref_curve:
            break
    if ref_curve is None:
        ref_curve = initial_curve

    avg_spl = compute_average_spl_in_range(ref_curve, min_freq, max_freq) if ref_curve else 0.0

    if initial_curve:
        spl_smoothed = smooth_octave(initial_curve["freq"], initial_curve["spl"], DEFAULT_SMOOTHING)
        series_list.append(series(
            "Before EQ",
            initial_curve["freq"], spl_smoothed,
            color="rgba(200, 200, 200, 0.6)", width=2,
        ))

    for mode_name, ch_data in mode_data:
        final_curve = ch_data.get("final_curve")
        if final_curve:
            spl_smoothed = smooth_octave(final_curve["freq"], final_curve["spl"], DEFAULT_SMOOTHING)
            series_list.append(series(
                _mode_label(mode_name),
                final_curve["freq"], spl_smoothed,
                color=_mode_color(mode_name), width=2,
            ))

    return figure(
        f"{channel_name}: Bass ({int(min_freq)}-{int(max_freq)} Hz)",
        axis("Frequency (Hz)", "log", min_freq, max_freq),
        axis("SPL (dB)", "linear",
             avg_spl - y_half_range, avg_spl + y_half_range),
        series_list=series_list,
        tab=tab,
    )


def create_comparison_eq_overlay_figure(
    channel_name: str,
    mode_data: list[tuple[str, dict]],
    *,
    sample_rates: dict[str, float] | None = None,
    tab=None,
):
    """Overlay EQ response curves from multiple modes."""
    series_list = []
    has_data = False
    freq_points = generate_freq_points(20.0, 20000.0, 500)

    for mode_name, ch_data in mode_data:
        eq_response_data = ch_data.get("eq_response")
        if eq_response_data and "freq" in eq_response_data and "spl" in eq_response_data:
            eq_freq = eq_response_data["freq"]
            eq_spl = eq_response_data["spl"]
        else:
            plugins = ch_data.get("plugins", [])
            eq_filters = []
            for plugin in plugins:
                if plugin.get("plugin_type") == "eq":
                    filters = plugin.get("parameters", {}).get("filters", [])
                    eq_filters.extend(filters)
            sample_rate = float((sample_rates or {}).get(mode_name, 48_000.0))
            if not math.isfinite(sample_rate) or sample_rate <= 40.0:
                raise ValueError("EQ comparison sample rate must support frequencies above 20 Hz")
            eq_freq = [min(f, sample_rate / 2) for f in generate_freq_points(
                20.0, min(20000.0, sample_rate / 2), 500)] if eq_filters else None
            eq_spl = compute_eq_response(eq_filters, eq_freq, sample_rate) if eq_filters else None

        if eq_spl:
            has_data = True
            series_list.append(series(
                _mode_label(mode_name),
                eq_freq, eq_spl,
                color=_mode_color(mode_name), width=2,
            ))

    if not has_data:
        return None

    series_list.append(series(
        "0 dB",
        [freq_points[0], freq_points[-1]], [0, 0],
        color="rgba(150, 150, 150, 0.5)", width=1, dash="dash",
    ))

    return figure(
        f"EQ Response Comparison: {channel_name}",
        axis("Frequency (Hz)", "log", 20, 20000),
        axis("Gain (dB)", "linear", -15, 15),
        series_list=series_list,
        tab=tab,
    )


def create_mode_subplots_figure(
    channel_name: str,
    mode_data: list[tuple[str, dict]],
    target_curve: dict | None = None,
    tab=None,
):
    """One before/after figure section per mode (subplot grid becomes a list).

    ``target_curve`` is drawn dotted in every cell when given; otherwise no
    reference line is drawn (a flat placeholder would misstate the slope).
    """
    all_curves: list[dict | None] = []
    for _, ch_data in mode_data:
        all_curves.append(ch_data.get("initial_curve"))
        all_curves.append(ch_data.get("final_curve"))
    if target_curve:
        all_curves.append(target_curve)
    y_min, y_max = compute_y_range(all_curves)

    sections = []
    for mode_name, ch_data in mode_data:
        initial_curve = ch_data.get("initial_curve")
        final_curve = ch_data.get("final_curve")
        color = _mode_color(mode_name)
        cell_series = []

        if initial_curve:
            spl_sm = smooth_octave(initial_curve["freq"], initial_curve["spl"], DEFAULT_SMOOTHING)
            cell_series.append(series(
                "Before EQ",
                initial_curve["freq"], spl_sm,
                color="rgba(200, 200, 200, 0.6)", width=1.5,
            ))

        if final_curve:
            spl_sm = smooth_octave(final_curve["freq"], final_curve["spl"], DEFAULT_SMOOTHING)
            cell_series.append(series(
                _mode_label(mode_name),
                final_curve["freq"], spl_sm,
                color=color, width=2,
            ))

        if target_curve and target_curve.get("freq") and target_curve.get("spl"):
            cell_series.append(series(
                "Target",
                target_curve["freq"], target_curve["spl"],
                color="rgba(40, 40, 40, 0.9)", width=1.5, dash="dot",
            ))

        sections.append(figure(
            f"{_mode_label(mode_name)}: {channel_name} Per-Mode Detail",
            axis("Frequency (Hz)", "log", 20, 20000),
            axis("SPL (dB)", "linear", y_min, y_max),
            series_list=cell_series,
            tab=tab,
        ))
    return sections
def create_comparison_phase_figure(
    channel_name: str,
    mode_data: list[tuple[str, dict]],
    tab=None,
) -> list | None:
    """One figure section per mode showing phase before/after."""
    has_phase = False
    for _, ch_data in mode_data:
        for key in ("initial_curve", "final_curve"):
            curve = ch_data.get(key)
            if curve and curve.get("phase"):
                has_phase = True
                break
        if has_phase:
            break
    if not has_phase:
        return None

    sections = []
    for mode_name, ch_data in mode_data:
        initial_curve = ch_data.get("initial_curve")
        final_curve = ch_data.get("final_curve")
        color = _mode_color(mode_name)
        s_list = []
        if initial_curve and initial_curve.get("phase"):
            phase_sm = wrap_phase(initial_curve["phase"])
            s_list.append(series(
                "Before EQ", initial_curve["freq"], phase_sm,
                color="rgba(200, 200, 200, 0.6)", width=1.5,
            ))
        if final_curve and final_curve.get("phase"):
            phase_sm = wrap_phase(final_curve["phase"])
            s_list.append(series(
                _mode_label(mode_name), final_curve["freq"], phase_sm,
                color=color, width=2.0,
            ))
        sections.append(figure(
            f"{channel_name}: Phase Before / After EQ — "
            f"{_mode_label(mode_name)}",
            axis("Frequency (Hz)", "log", 20, 20000),
            axis("Phase (°)", "linear", None, None),
            series_list=s_list, tab=tab,
        ))
    return sections


def create_comparison_group_delay_figure(
    channel_name: str,
    mode_data: list[tuple[str, dict]],
    tab=None,
) -> list | None:
    """One figure section per mode showing group delay before/after.

    Group delay is computed from unwrapped phase: GD = -d(phase)/d(omega).
    """
    has_phase = False
    for _, ch_data in mode_data:
        if ch_data.get("pre_ir") or ch_data.get("post_ir"):
            has_phase = True
            break
        for key in ("initial_curve", "final_curve"):
            curve = ch_data.get(key)
            if curve and curve.get("phase"):
                has_phase = True
                break
        if has_phase:
            break
    if not has_phase:
        return None

    # First pass: compute smoothed GD traces and a shared y-range from
    # all GD data (2nd/98th percentiles plus a 15% margin, as before).
    per_mode: list[tuple[str, str, list | None, list | None,
                          list | None, list | None]] = []
    all_gd: list[list[float]] = []
    for mode_name, ch_data in mode_data:
        initial_curve = ch_data.get("initial_curve")
        final_curve = ch_data.get("final_curve")
        before_freq: list | None = None
        before_gd: list | None = None
        after_freq: list | None = None
        after_gd: list | None = None
        if ch_data.get("pre_ir") or (initial_curve and initial_curve.get("phase")):
            gd_freq, gd_ms = compute_group_delay_from_ir(ch_data.get("pre_ir"))
            if not gd_freq and initial_curve and initial_curve.get("phase"):
                gd_freq, gd_ms = compute_group_delay(
                    initial_curve["freq"], initial_curve["phase"]
                )
            if gd_freq:
                before_freq = list(gd_freq)
                before_gd = smooth_octave(gd_freq, gd_ms, 1.0 / 3.0)
                all_gd.append(before_gd)
        if ch_data.get("post_ir") or (final_curve and final_curve.get("phase")):
            gd_freq, gd_ms = compute_group_delay_from_ir(ch_data.get("post_ir"))
            if not gd_freq and final_curve and final_curve.get("phase"):
                gd_freq, gd_ms = compute_group_delay(
                    final_curve["freq"], final_curve["phase"]
                )
            if gd_freq:
                after_freq = list(gd_freq)
                after_gd = smooth_octave(gd_freq, gd_ms, 1.0 / 3.0)
                all_gd.append(after_gd)
        per_mode.append((mode_name, _mode_color(mode_name),
                         before_freq, before_gd, after_freq, after_gd))

    y_lo: float | None = None
    y_hi: float | None = None
    if all_gd:
        flat = [v for gd in all_gd for v in gd]
        sorted_vals = sorted(flat)
        n = len(sorted_vals)
        lo = sorted_vals[max(0, int(n * 0.02))]
        hi = sorted_vals[min(n - 1, int(n * 0.98))]
        margin = (hi - lo) * 0.15
        y_lo = lo - margin
        y_hi = hi + margin

    sections = []
    for mode_name, color, before_freq, before_gd, after_freq, after_gd in per_mode:
        s_list = []
        if before_freq:
            s_list.append(series(
                "Before EQ", before_freq, before_gd,
                color="rgba(200, 200, 200, 0.6)", width=1.5,
            ))
        if after_freq:
            s_list.append(series(
                _mode_label(mode_name), after_freq, after_gd,
                color=color, width=2.0,
            ))
        sections.append(figure(
            f"{channel_name}: Group Delay Before / After EQ — "
            f"{_mode_label(mode_name)}",
            axis("Frequency (Hz)", "log", 20, 20000),
            axis("Group Delay (ms)", "linear", y_lo, y_hi),
            series_list=s_list, tab=tab,
        ))
    return sections


def create_comparison_ir_figure(
    channel_name: str,
    mode_data: list[tuple[str, dict]],
    display_ms: float = 100.0,
    tab=None,
) -> list | None:
    """One figure section per mode showing impulse response before/after."""
    has_ir = False
    for _, ch_data in mode_data:
        if ch_data.get("pre_ir") or ch_data.get("post_ir"):
            has_ir = True
            break
    if not has_ir:
        return None

    sections = []
    for mode_name, ch_data in mode_data:
        pre_ir = ch_data.get("pre_ir")
        post_ir = ch_data.get("post_ir")
        color = _mode_color(mode_name)
        s_list = []
        if pre_ir:
            s_list.append(series(
                "Before EQ", pre_ir["time_ms"], pre_ir["amplitude"],
                color="rgba(200, 200, 200, 0.6)", width=1.0,
            ))
        if post_ir:
            s_list.append(series(
                _mode_label(mode_name), post_ir["time_ms"],
                post_ir["amplitude"], color=color, width=1.0,
            ))
        sections.append(figure(
            f"{channel_name}: Impulse Response Before / After EQ — "
            f"{_mode_label(mode_name)}",
            axis("Time (ms)", "linear", 0, display_ms),
            axis("Amplitude", "linear", -1.1, 1.1),
            series_list=s_list, tab=tab,
        ))
    return sections


def create_score_comparison_figure(
    mode_scores: list[tuple[str, float, float]],
    loss_types: list[str | None] | None = None,
    tab=None,
) -> dict:
    """Bar section comparing pre/post flat-loss values across modes.

    Note: even when a mode minimized a non-flat loss (e.g. EPA), the
    `pre_score` / `post_score` numbers in the JSON metadata are *always*
    computed via the flat-loss helper in
    `crate::roomeq::workflows::compute_flat_loss`. That makes every bar
    directly comparable on a single shared y-axis regardless of which
    objective each mode was actually optimizing.

    The bar values therefore answer the question "did this mode flatten
    the response well?", not "how well did this mode minimize its own
    objective?". For perceptual comparison across loss functions, look
    at the EPA Pref columns of the summary table instead.
    """
    mode_names = [_mode_label(n) for n, _, _ in mode_scores]
    pre_scores = [pre for _, pre, _ in mode_scores]
    post_scores = [post for _, _, post in mode_scores]

    # If multiple loss functions were used, readers need to know the bars
    # still measure the same underlying flat loss (otherwise they might
    # assume the EPA bars are EPA-loss values). The bar schema has no
    # annotation slot, so the note is folded into the title.
    distinct_losses = (
        sorted({lt for lt in loss_types if lt}) if loss_types else []
    )
    title_text = "Flat-loss before / after EQ (lower is better)"
    if len(distinct_losses) >= 2:
        title_text += (
            " — Note: bars show the flat-loss metric for every run, "
            "even ones that optimized EPA — this is so all modes "
            "stay on a single comparable scale. For perceptual "
            "comparison see the EPA Pref columns above."
        )

    return bar_chart(
        title_text,
        mode_names,
        [("Before EQ", pre_scores, "rgba(200, 200, 200, 0.7)"),
         ("After EQ", post_scores, None)],
        ylabel="Flat loss (lower is better)",
        tab=tab,
    )


def create_smoothed_figure(
    channel_name: str,
    initial_curve: dict | None,
    final_curve: dict | None,
    octaves: float = 1.0,
    tab=None,
) -> dict | None:
    """Per-speaker 1-octave smoothed Before/After overlay (feat-report 2b)."""
    if not initial_curve and not final_curve:
        return None
    s_list = []
    curves = (("Before EQ (1-oct smoothed)", initial_curve, "rgba(255, 100, 100, 0.8)"),
              ("After EQ (1-oct smoothed)", final_curve, "rgba(100, 200, 100, 0.9)"))
    for label, curve, color in curves:
        if not curve or not curve.get("freq") or not curve.get("spl"):
            continue
        s_list.append(series(
            label, curve["freq"],
            smooth_octave(curve["freq"], curve["spl"], octaves),
            color=color, width=2.0,
        ))
    y_min, y_max = compute_y_range([initial_curve, final_curve])
    return figure(
        f"Smoothed response (1 oct): {channel_name}",
        axis("Frequency (Hz)", "log", 20, 20000),
        axis("SPL (dB)", "linear", y_min, y_max),
        series_list=s_list, tab=tab,
    )


def create_tof_figure(
    tof_rows: list[dict],
    after: bool = False,
    tab=None,
) -> dict | None:
    """Time-of-flight bar section before (measured) or after (calculated) DSP."""
    key = "after_ms" if after else "before_ms"
    kept = []
    for row in tof_rows:
        value = row.get(key)
        if (isinstance(value, (int, float)) and not isinstance(value, bool)
                and math.isfinite(value)):
            kept.append((row["name"], float(value)))
    if not kept:
        return None
    title = ("Calculated arrival after DSP" if after
             else "Measured arrival before DSP")
    # The bar schema has no null values: rows without a measurement are
    # dropped (Plotly drew a gap there, i.e. no bar either).
    names = [name for name, _ in kept]
    plot_values = [value for _, value in kept]
    return bar_chart(
        title,
        names,
        [(title, plot_values,
          "rgba(100, 200, 100, 0.8)" if after else "rgba(255, 100, 100, 0.8)")],
        ylabel="Arrival (ms)",
        legend=False,
        tab=tab,
    )


def create_symmetric_pair_figure(
    label: str,
    freq: list[float],
    sum_spl: list[float],
    diff_spl: list[float],
    tab=None,
) -> dict:
    """Symmetric-pair magnitude sum + difference (feat-report 2d, viewer part)."""
    s_list = [
        series(f"{label} magnitude sum", freq, sum_spl,
               color="rgba(74, 144, 217, 0.9)", width=2.0),
        series(f"{label} |magnitude difference|", freq, diff_spl,
               color="rgba(255, 150, 50, 0.9)", width=2.0, dash="dash"),
    ]
    all_spl = list(sum_spl) + list(diff_spl)
    finite = [v for v in all_spl if math.isfinite(v)]
    pad = (max(finite) - min(finite)) * 0.1 if finite else 5.0
    y_range = (min(finite) - pad, max(finite) + pad) if finite else (0.0, 1.0)
    return figure(
        f"Symmetric pair: {label} (magnitude domain; "
        "complex sum pending roomeq field)",
        axis("Frequency (Hz)", "log", 20, 20000),
        axis("SPL (dB)", "linear", y_range[0], y_range[1]),
        series_list=s_list, tab=tab,
    )


def create_early_late_figure(
    label: str,
    report: dict | None,
    tab=None,
) -> dict | None:
    """Render emitted third-octave energy contributions on a shared reference."""
    if not isinstance(report, dict) or report.get("method") != "incoherent_band_energy":
        return None
    if report.get("reference") != "full_peak_band":
        return None
    if report.get("smoothing") != "third_octave" or report.get("split_ms") != 20.0:
        return None
    curves = [report.get(key) for key in ("full", "early", "late")]
    if not all(isinstance(curve, dict) for curve in curves):
        return None
    frequency = curves[0].get("freq")
    if not isinstance(frequency, list) or not frequency:
        return None
    if not all(isinstance(f, (int, float)) and math.isfinite(f) and f > 0 for f in frequency):
        return None
    for curve in curves:
        spl = curve.get("spl")
        if (curve.get("freq") != frequency or not isinstance(spl, list)
                or len(spl) != len(frequency)
                or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in spl)):
            return None
    s_list = []
    for name, curve, color in zip(("Full", "Early", "Late"), curves,
                                  ("#333333", "#3089c5", "#e38836")):
        s_list.append(series(name, frequency, curve["spl"],
                             color=color, width=2.0))
    return figure(
        f"{label}: early vs late band energy (20 ms split)",
        axis("Frequency (Hz)", "log", 20, 20000),
        axis("Level vs full peak band (dB)", "linear", None, None),
        series_list=s_list, tab=tab,
    )


def create_t60_octaves_figure(
    label: str,
    rows: list[dict] | None,
    tab=None,
) -> dict | None:
    """Section with only valid measured-room octave T60 estimates."""
    if not rows or not any(row["t60_s"] is not None for row in rows):
        return None
    # None y entries pass through series() as gaps (connectgaps=False before).
    return figure(
        f"{label}: measured octave-band T60",
        axis("Octave centre (Hz)", "log", None, None),
        axis("T60 (s)", "linear", 0, None),
        series_list=[series(
            "T60",
            [row["centre_hz"] for row in rows],
            [row["t60_s"] for row in rows],
            color="#3089c5", width=2.0,
        )],
        tab=tab,
    )
