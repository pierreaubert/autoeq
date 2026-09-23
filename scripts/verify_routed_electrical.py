#!/usr/bin/env python3
"""Independently replay supported routed PEQ/LR24/LR48 steady-sine output peaks.

This diagnostic checks serialized RoomEQ output, not an optimizer prediction.
It is intentionally limited to gain, delay, peak EQ, bound FIR convolution,
and LR24/LR48 routes. It does
not certify transient/true peak, nonlinear limiting, physical driver demand,
native backend playback, or acoustic summation.

CLI exit 0 means a bound graph matches its reported sampled peaks and all
reported digital limits; exit 2 means the bound replay matches a truthful
over-limit refusal. Invalid, stale, or unsupported artifacts exit nonzero.
"""

from __future__ import annotations

import argparse
from bisect import bisect_left
import cmath
import json
import math
from pathlib import Path

from src.loaders import RoomEqData, read_fir_wav
from src.payload_binding import verify_payload_binding


def _biquad(coefficients: tuple[float, ...], frequency: float, rate: float) -> complex:
    b0, b1, b2, a0, a1, a2 = coefficients
    z1 = cmath.exp(-2j * math.pi * frequency / rate)
    z2 = z1 * z1
    return (b0 + b1 * z1 + b2 * z2) / (a0 + a1 * z1 + a2 * z2)


def _peak(filter_config: dict, frequency: float, rate: float) -> complex:
    if filter_config.get("filter_type") != "peak":
        raise ValueError(f"unsupported EQ filter: {filter_config.get('filter_type')}")
    center = float(filter_config["freq"])
    quality = float(filter_config["q"])
    gain = float(filter_config["db_gain"])
    if not (0 < center < rate / 2 and quality > 0 and all(map(math.isfinite, (center, quality, gain)))):
        raise ValueError("invalid peak EQ parameters")
    omega = 2 * math.pi * center / rate
    alpha = math.sin(omega) / (2 * quality)
    amplitude = 10 ** (gain / 40)
    cosine = math.cos(omega)
    return _biquad(
        (
            1 + alpha * amplitude,
            -2 * cosine,
            1 - alpha * amplitude,
            1 + alpha / amplitude,
            -2 * cosine,
            1 - alpha / amplitude,
        ),
        frequency,
        rate,
    )


def _lr24(frequency: float, center: float, rate: float, high: bool) -> complex:
    if not (0 < center < rate / 2):
        raise ValueError("invalid LR24 crossover frequency")
    omega = 2 * math.pi * center / rate
    cosine = math.cos(omega)
    alpha = math.sin(omega) / math.sqrt(2)
    if high:
        b0, b1, b2 = (1 + cosine) / 2, -(1 + cosine), (1 + cosine) / 2
    else:
        b0, b1, b2 = (1 - cosine) / 2, 1 - cosine, (1 - cosine) / 2
    section = _biquad((b0, b1, b2, 1 + alpha, -2 * cosine, 1 - alpha), frequency, rate)
    return section * section


def _lr48(frequency: float, center: float, rate: float, high: bool) -> complex:
    """Two cascaded fourth-order Butterworth filters, each with distinct Qs."""
    if not (0 < center < rate / 2):
        raise ValueError("invalid LR48 crossover frequency")
    omega = 2 * math.pi * center / rate
    cosine = math.cos(omega)
    if high:
        b0, b1, b2 = (1 + cosine) / 2, -(1 + cosine), (1 + cosine) / 2
    else:
        b0, b1, b2 = (1 - cosine) / 2, 1 - cosine, (1 - cosine) / 2
    transfer = 1 + 0j
    for index in (0, 1, 0, 1):
        quality = 1 / (2 * math.sin(math.pi / 4 * (index + 0.5)))
        alpha = math.sin(omega) / (2 * quality)
        transfer *= _biquad(
            (b0, b1, b2, 1 + alpha, -2 * cosine, 1 - alpha),
            frequency,
            rate,
        )
    return transfer


class _ConvolutionReplay:
    def __init__(self, source_directory: Path | None):
        self.source_directory = source_directory
        self.taps: dict[str, list[float]] = {}
        self.responses: dict[tuple[str, float], complex] = {}

    def response(self, reference: str, frequency: float, rate: float) -> complex:
        if self.source_directory is None:
            raise ValueError("unsupported serial plugin: convolution without saved resource location")
        if not isinstance(reference, str) or not reference:
            raise ValueError("convolution has no IR file reference")
        if reference not in self.taps:
            path = Path(reference)
            if not path.is_absolute():
                path = self.source_directory / path
            resource_rate, taps = read_fir_wav(path)
            if resource_rate != round(rate):
                raise ValueError("convolution sidecar sample rate differs from reported rate")
            self.taps[reference] = taps
        key = (reference, frequency)
        if key not in self.responses:
            z = cmath.exp(-2j * math.pi * frequency / rate)
            power = 1 + 0j
            transfer = 0j
            for tap in self.taps[reference]:
                transfer += tap * power
                power *= z
            self.responses[key] = transfer
        return self.responses[key]


def _plugin(
    plugin: dict, frequency: float, rate: float, convolution: _ConvolutionReplay
) -> complex:
    kind = plugin["plugin_type"]
    parameters = plugin["parameters"]
    if kind == "gain":
        gain = float(parameters["gain_db"])
        if not math.isfinite(gain):
            raise ValueError("nonfinite gain")
        return 10 ** (gain / 20) * (-1 if parameters.get("invert", False) else 1)
    if kind == "delay":
        delay = float(parameters["delay_ms"])
        if not math.isfinite(delay) or delay < 0:
            raise ValueError("invalid delay")
        return cmath.exp(-2j * math.pi * frequency * delay / 1000)
    if kind == "eq":
        transfer = 1 + 0j
        for filter_config in parameters["filters"]:
            transfer *= _peak(filter_config, frequency, rate)
        return transfer
    if kind == "convolution":
        mix = float(parameters.get("mix", 1.0))
        gain = float(parameters.get("gain_db", 0.0))
        if not math.isfinite(mix) or not math.isfinite(gain):
            raise ValueError("invalid convolution mix or gain")
        mix = max(0.0, min(1.0, mix))
        wet = convolution.response(parameters.get("ir_file"), frequency, rate)
        return (1 - mix) + wet * (mix * 10 ** (gain / 20))
    raise ValueError(f"unsupported serial plugin: {kind}")


def _selected_plugins(channel: dict, owner: str) -> list[dict]:
    plugins = channel.get("plugins", [])
    for plugin in plugins:
        stage = plugin["parameters"].get("room_eq_stage")
        if stage not in ("pre_route", "post_route", "route_owned"):
            raise ValueError(f"unresolved plugin ownership: {plugin['plugin_type']}")
        if stage == "route_owned" and plugin["plugin_type"] not in ("gain", "delay", "crossover"):
            raise ValueError("unsupported route-owned plugin")
    return [plugin for plugin in plugins if plugin["parameters"].get("room_eq_stage") == owner]


def _pre_plugins(channels: dict, route: dict) -> list[dict]:
    source = route["source_channel"]
    if source in channels:
        return _selected_plugins(channels[source], "pre_route")
    if (
        source == "LFE"
        and route.get("route_kind") == "lfe_lowpass_to_sub"
        and route.get("pre_chain_channel") == "LFE"
    ):
        # The production physical-routing resolver admits a declared virtual
        # LFE input with an empty pre-route DSP chain.
        return []
    raise ValueError(f"unsupported virtual or missing input channel: {source}")


def _post_plugins(channels: dict, routing: dict, destination: str) -> list[dict]:
    for name in routing["physical_sub_outputs"]:
        parent = channels.get(name)
        if not parent:
            continue
        for driver in parent.get("drivers") or []:
            if driver["name"] != destination:
                continue
            plugins = _selected_plugins(parent, "post_route")
            for plugin in driver["plugins"]:
                if plugin["parameters"].get("room_eq_stage") != "post_route":
                    raise ValueError("unresolved physical driver plugin ownership")
                if plugin["plugin_type"] in ("gain", "delay"):
                    if plugin["plugin_type"] == "gain" and plugin["parameters"].get("room_eq_correction_gain") is True:
                        plugins.append(plugin)
                    if plugin["plugin_type"] == "delay" and plugin["parameters"].get("room_eq_correction_delay") is True:
                        plugins.append(plugin)
                    continue  # Otherwise baked into the bass route.
                if plugin["plugin_type"] not in ("eq", "convolution"):
                    raise ValueError(f"unsupported physical driver plugin: {plugin['plugin_type']}")
                plugins.append(plugin)
            return plugins
    return _selected_plugins(channels[destination], "post_route")


def _route(route: dict, frequency: float, rate: float) -> complex:
    crossover_type = route["crossover_type"]
    if crossover_type not in ("LR24", "LR48"):
        raise ValueError(f"unsupported crossover: {route['crossover_type']}")
    low, high = route.get("low_pass_hz"), route.get("high_pass_hz")
    if (low is None) == (high is None):
        raise ValueError("route must specify exactly one crossover pass")
    gain = float(route["gain_db"])
    gain_linear = float(route["gain_linear"])
    matrix_gain = float(route["matrix_gain"])
    delay = float(route["delay_ms"])
    if (
        not all(map(math.isfinite, (gain, gain_linear, matrix_gain, delay)))
        or gain_linear <= 0
        or delay < 0
        or abs(20 * math.log10(gain_linear) - gain) > 1e-6
    ):
        raise ValueError("invalid or contradictory route gain/delay")
    transfer = 10 ** (gain / 20) * (-1 if route.get("polarity_inverted") else 1)
    transfer *= cmath.exp(-2j * math.pi * frequency * delay / 1000)
    crossover = _lr24 if crossover_type == "LR24" else _lr48
    return transfer * crossover(frequency, float(high if high is not None else low), rate, high is not None)


def _independent_paths(channels: dict) -> list[tuple[str, str, list[dict]]]:
    """Expand unbranched channels and explicitly indexed physical drivers."""
    if not channels:
        raise ValueError("independent electrical graph has no channels")
    paths = []
    for name, channel in channels.items():
        if not name or channel.get("channel", name) != name:
            raise ValueError("independent channel has inconsistent identity")
        common = channel.get("plugins", [])
        drivers = channel.get("drivers") or []
        if not drivers:
            output = json.dumps(["channel", name], separators=(",", ":"), ensure_ascii=False)
            paths.append((name, output, common))
            continue
        indices = set()
        names = set()
        for driver in drivers:
            index, driver_name = driver["index"], driver["name"]
            if not driver_name or index in indices or driver_name in names:
                raise ValueError("independent drivers require unique indices and names")
            indices.add(index)
            names.add(driver_name)
            output = json.dumps(
                ["driver", name, index, driver_name],
                separators=(",", ":"),
                ensure_ascii=False,
            )
            paths.append((name, output, common + driver["plugins"]))
    return paths


def output_complex_transfers(
    graph: dict,
    frequencies: list[float],
    rate: float,
    source_directory: Path | None = None,
) -> dict[str, dict[str, list[complex]]]:
    """Replay each source-to-output complex transfer before input-peak aggregation."""
    routing = (graph.get("metadata", {}).get("bass_management") or {}).get("routing_graph")
    channels = graph["channels"]
    global_plugins = graph.get("global_plugins", [])
    if routing is None:
        if global_plugins:
            raise ValueError("unsupported global processing on independent graph")
        independent = _independent_paths(channels)
        input_names = list(channels)
        paths = [(source, destination, None, plugins) for source, destination, plugins in independent]
        output_names = list(dict.fromkeys(destination for _, destination, _ in independent))
    else:
        if not isinstance(routing, dict):
            raise ValueError("unsupported bass-management routing declaration")
        if len(global_plugins) != 1 or global_plugins[0]["plugin_type"] != "matrix":
            raise ValueError("unsupported global routing")
        input_names = routing["input_channels"]
        paths = []
        for route in routing["routes"]:
            source, destination = route["source_channel"], route["destination"]
            if source not in input_names or destination not in routing["output_channels"]:
                raise ValueError("route endpoint is not declared")
            for field, names, name in (
                ("source_index", routing["input_channels"], source),
                ("destination_index", routing["output_channels"], destination),
            ):
                if field in route:
                    index = route[field]
                    if (
                        not isinstance(index, int)
                        or isinstance(index, bool)
                        or index < 0
                        or index >= len(names)
                        or names[index] != name
                    ):
                        raise ValueError("route endpoint index disagrees with declared identity")
            pre = _pre_plugins(channels, route)
            post = _post_plugins(channels, routing, destination)
            paths.append((source, destination, route, pre + post))
        output_names = routing["output_channels"]
    if len(frequencies) < 2 or any(not math.isfinite(f) or f < 0 or f > rate / 2 for f in frequencies):
        raise ValueError("invalid frequency grid")
    if any(left >= right for left, right in zip(frequencies, frequencies[1:])):
        raise ValueError("frequency grid is not strictly increasing")
    if not input_names or not output_names:
        raise ValueError("electrical graph has no declared input or output")
    outputs = {
        destination: {source: [] for source in input_names}
        for destination in output_names
    }
    convolution = _ConvolutionReplay(source_directory)
    for frequency in frequencies:
        grouped: dict[tuple[str, str], complex] = {}
        for source, destination, route, plugins in paths:
            transfer = _route(route, frequency, rate) if route is not None else 1 + 0j
            for plugin in plugins:
                transfer *= _plugin(plugin, frequency, rate, convolution)
            key = (destination, source)
            grouped[key] = grouped.get(key, 0j) + transfer
        for destination, sources in outputs.items():
            for source, transfers in sources.items():
                transfers.append(grouped.get((destination, source), 0j))
    return outputs


def output_amplitudes(
    graph: dict,
    frequencies: list[float],
    input_peaks: dict[str, float],
    rate: float,
    source_directory: Path | None = None,
) -> dict[str, list[float]]:
    """Sum coherent same-input routes, then independently bound input phases."""
    if any(not math.isfinite(value) or value < 0 for value in input_peaks.values()):
        raise ValueError("invalid input peak")
    routing = (graph.get("metadata", {}).get("bass_management") or {}).get("routing_graph")
    if routing is not None and not isinstance(routing, dict):
        raise ValueError("unsupported bass-management routing declaration")
    expected_inputs = set(graph["channels"] if routing is None else routing["input_channels"])
    if set(input_peaks) != expected_inputs:
        kind = "independent channels" if routing is None else "routed inputs"
        raise ValueError(f"input peak policy does not match {kind}")
    transfers = output_complex_transfers(graph, frequencies, rate, source_directory)
    return {
        destination: [
            sum(abs(source_transfer[index]) * input_peaks[source]
                for source, source_transfer in per_source.items())
            for index in range(len(frequencies))
        ]
        for destination, per_source in transfers.items()
    }


def _final_electrical_stage(graph: dict) -> dict:
    stages = [
        stage for stage in graph["metadata"]["stage_outcomes"]
        if stage["stage"] == "final_graph_sampled_electrical_headroom"
    ]
    if len(stages) != 1:
        raise ValueError("expected exactly one final electrical stage")
    return stages[0]


def verify_artifact(
    artifact: dict, tolerance: float = 2e-5, require_binding: bool = False
) -> dict[str, float]:
    graph = artifact.get("after_final_seat_validation", artifact)
    if require_binding:
        verified, reason, _ = verify_payload_binding(graph)
        if not verified:
            raise ValueError(f"serialized graph payload binding failed: {reason}")
    stage = _final_electrical_stage(graph)
    reported = {}
    for check in stage["checks"]:
        if not check["id"].startswith("sampled_physical_output:"):
            continue
        data = json.loads(check["diagnostic"])
        output = check["id"].split(":", 1)[1]
        if data["output"] != output or output in reported:
            raise ValueError("duplicate or mismatched physical-output report")
        if check["kind"] != "safety":
            raise ValueError("physical-output check is not a safety check")
        observed = float(check["observed"])
        limit = float(check["limit"])
        peak = float(data["peak_amplitude"])
        if not all(map(math.isfinite, (observed, limit, peak))) or limit <= 0:
            raise ValueError("invalid physical-output check")
        if not math.isclose(observed, peak, rel_tol=1e-10, abs_tol=1e-10):
            raise ValueError("physical-output observation differs from diagnostic")
        if check["passed"] != (peak <= limit * 10 ** (1e-6 / 20)):
            raise ValueError("physical-output verdict differs from numeric limit")
        reported[data["output"]] = data
    if not reported:
        raise ValueError("no sampled physical-output reports")
    rates = {float(data["sample_rate_hz"]) for data in reported.values()}
    if len(rates) != 1 or not all(math.isfinite(rate) and rate > 0 for rate in rates):
        raise ValueError("inconsistent or invalid reported sample rate")
    rate = rates.pop()
    if "sample_rate_hz" in artifact and float(artifact["sample_rate_hz"]) != rate:
        raise ValueError("artifact and reported sample rates differ")
    input_peaks = next(iter(reported.values()))["input_peak_limits"]
    if any(data["input_peak_limits"] != input_peaks for data in reported.values()):
        # Distinct outputs may have different input subsets; merge and check overlaps.
        input_peaks = {}
        for data in reported.values():
            for name, peak in data["input_peak_limits"].items():
                if name in input_peaks and input_peaks[name] != peak:
                    raise ValueError("inconsistent reported input peaks")
                input_peaks[name] = peak
    frequencies = [rate * 0.5 * index / 8192 for index in range(8193)]
    routing = (graph.get("metadata", {}).get("bass_management") or {}).get("routing_graph")
    channels = graph["channels"]
    path_plugins = []
    if routing is None:
        if graph.get("global_plugins"):
            raise ValueError("unsupported global processing on independent graph")
        independent = _independent_paths(channels)
        sources_by_output = {destination: {source} for source, destination, _ in independent}
        for _, _, plugins in independent:
            path_plugins.extend(plugins)
    else:
        if not isinstance(routing, dict):
            raise ValueError("unsupported bass-management routing declaration")
        sources_by_output = {name: set() for name in routing["output_channels"]}
        for route in routing["routes"]:
            if route["destination"] not in sources_by_output:
                raise ValueError("route destination is not declared")
            sources_by_output[route["destination"]].add(route["source_channel"])
            for key in ("high_pass_hz", "low_pass_hz"):
                if route.get(key) is not None:
                    frequencies.append(float(route[key]))
            path_plugins.extend(_pre_plugins(channels, route))
            path_plugins.extend(_post_plugins(channels, routing, route["destination"]))
    for plugin in path_plugins:
        for value in [plugin["parameters"], *plugin["parameters"].get("filters", [])]:
            for key in ("freq", "frequency"):
                if key in value and 0 < float(value[key]) < rate / 2:
                    frequencies.append(float(value[key]))
    frequencies = sorted(set(frequencies))
    if any(data["grid_points"] != len(frequencies) for data in reported.values()):
        raise ValueError("reported and replayed frequency grids differ")
    for output, data in reported.items():
        inputs = data["inputs"]
        if (
            output not in sources_by_output
            or not isinstance(inputs, list)
            or len(inputs) != len(set(inputs))
            or set(inputs) != sources_by_output[output]
            or set(data["input_peak_limits"]) != sources_by_output[output]
        ):
            raise ValueError(f"{output} reported input provenance differs from routes")
        if data["evaluated_band_hz"] != [frequencies[0], frequencies[-1]]:
            raise ValueError(f"{output} reported evaluated band differs from replay grid")
    source_directory = getattr(graph, "source_directory", None) or getattr(
        artifact, "source_directory", None
    )
    amplitudes = output_amplitudes(graph, frequencies, input_peaks, rate, source_directory)
    if set(amplitudes) != set(reported):
        raise ValueError("reported and replayed physical outputs differ")
    peaks = {name: max(values) for name, values in amplitudes.items()}
    for name, peak in peaks.items():
        if not math.isclose(peak, reported[name]["peak_amplitude"], rel_tol=tolerance, abs_tol=tolerance):
            raise ValueError(f"{name} independent peak {peak:.9f} differs from reported {reported[name]['peak_amplitude']:.9f}")
        reported_frequency = float(reported[name]["peak_frequency_hz"])
        index = bisect_left(frequencies, reported_frequency)
        if (
            index == len(frequencies)
            or not math.isclose(frequencies[index], reported_frequency, abs_tol=1e-6)
            or not math.isclose(amplitudes[name][index], peak, rel_tol=1e-9, abs_tol=1e-12)
        ):
            raise ValueError(f"{name} reported peak frequency has no replayed peak")
    return peaks


def assess_bound_artifact(artifact: dict) -> dict:
    """Separate a matching electrical refusal from a within-limit result."""
    peaks = verify_artifact(artifact, require_binding=True)
    graph = artifact.get("after_final_seat_validation", artifact)
    stage = _final_electrical_stage(graph)
    over_limit = sorted(
        check["id"].split(":", 1)[1]
        for check in stage["checks"]
        if check["id"].startswith("sampled_physical_output:") and not check["passed"]
    )
    return {
        "status": (
            "independent_sampled_electrical_over_limit"
            if over_limit else "independent_sampled_electrical_match"
        ),
        "peaks": peaks,
        "over_limit_outputs": over_limit,
        "scope": "sampled_steady_sine_reported_digital_limits_only",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path, help="RoomEQ JSON or canonical canary artifact")
    args = parser.parse_args()
    artifact = RoomEqData(json.loads(args.artifact.read_text()), args.artifact.resolve().parent)
    result = assess_bound_artifact(artifact)
    print(json.dumps(result, sort_keys=True))
    if result["over_limit_outputs"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
