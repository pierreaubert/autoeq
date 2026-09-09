"""Validate reported physical crossover routes against requested matrix axes."""
import math


def verify_crossover(row):
    axes = row["requested_axes"]
    bass = row["bass_management"]
    if axes["topology"] == 0:
        assert bass is None, "stereo-only row unexpectedly acquired bass routing"
        return "not_applicable"
    assert bass and bass["enabled"], "requested routed topology has no bass management"
    optimization = bass["optimization"]
    expected_type = "LR48" if axes["crossover"] == 2 else "LR24"
    assert optimization["crossover_type"] == expected_type
    automatic = axes["crossover"] == 1
    assert optimization["crossover_range_hz"] == ([120.0, 220.0] if automatic else None)
    if axes["phase"] == 1:
        assert optimization["phase_available"] is True
        assert optimization["applied"] is True, "phase-capable crossover selection did not execute"
    else:
        assert optimization["phase_available"] is False
        assert optimization["applied"] is False
        assert "missing_phase_crossover_alignment_skipped" in optimization["advisories"]
    graph = bass["routing_graph"]
    groups = {group["group_id"]: group for group in bass["groups"]}
    assert groups, "no physical crossover groups were reported"
    expected_sources = set(graph["input_channels"]) - {bass["lfe_channel"]}
    seen = {"main_highpass_to_self": set(), "redirected_bass_lowpass_to_sub": set()}
    for route in graph["routes"]:
        kind = route["route_kind"]
        if kind not in seen:
            continue
        source = route["source_channel"]
        assert source not in seen[kind], "duplicate physical branch in single-sub matrix"
        seen[kind].add(source)
        group = groups[route["group_id"]]
        frequency = group["selected_crossover_hz"]
        assert math.isfinite(frequency)
        assert (120.0 <= frequency <= 220.0) if automatic else frequency == 160.0
        assert group["crossover_type"] == expected_type
        assert route["crossover_type"] == expected_type
        if kind == "main_highpass_to_self":
            assert route["destination"] == source
            assert route["high_pass_hz"] == frequency
        else:
            assert route["destination"] == bass["physical_sub_output"]
            assert route["low_pass_hz"] == frequency
    assert all(sources == expected_sources for sources in seen.values()), "missing main or sub branch"
    if automatic and axes["phase"] != 1:
        return "unsupported_missing_phase"
    return "automatic_routes_verified" if automatic else "fixed_routes_verified"
