"""DSP computation functions for biquad filters and smoothing."""

import math
import sys

import numpy as np


def smooth_octave(freq: list[float], spl: list[float], octave_fraction: float) -> list[float]:
    """
    Apply octave smoothing to frequency response data.

    Args:
        freq: Frequency points in Hz
        spl: SPL values in dB
        octave_fraction: Smoothing width in octaves (e.g., 1/3 for 1/3 octave smoothing)

    Returns:
        Smoothed SPL values
    """
    if not freq or not spl or octave_fraction is None:
        return spl

    n = len(freq)
    smoothed = []

    for i in range(n):
        f_center = freq[i]
        if f_center <= 0:
            smoothed.append(spl[i])
            continue

        # Calculate frequency range for this octave fraction
        # For 1/N octave, the bandwidth is 2^(1/N) ratio
        ratio = 2 ** (octave_fraction / 2)
        f_low = f_center / ratio
        f_high = f_center * ratio

        # Find all points within the smoothing window
        values = []
        weights = []

        for j in range(n):
            if f_low <= freq[j] <= f_high:
                # Use triangular weighting (closer to center = more weight)
                log_dist = abs(math.log10(freq[j]) - math.log10(f_center))
                log_half_width = math.log10(ratio)
                weight = 1.0 - (log_dist / log_half_width) if log_half_width > 0 else 1.0
                values.append(spl[j])
                weights.append(max(0, weight))

        if values and sum(weights) > 0:
            # Weighted average
            smoothed.append(sum(v * w for v, w in zip(values, weights)) / sum(weights))
        else:
            smoothed.append(spl[i])

    return smoothed


_FILTER_TYPE_ALIASES = {
    "lowpass": "lowpass",
    "lp": "lowpass",
    "highpass": "highpass",
    "hp": "highpass",
    "highpassvariableq": "highpassvariableq",
    "hpq": "highpassvariableq",
    "bandpass": "bandpass",
    "bp": "bandpass",
    "peak": "peak",
    "peaking": "peak",
    "peakdip": "peak",
    "parametric": "peak",
    "pk": "peak",
    "notch": "notch",
    "bandstop": "notch",
    "no": "notch",
    "lowshelf": "lowshelf",
    "ls": "lowshelf",
    "highshelf": "highshelf",
    "hs": "highshelf",
    "allpass": "allpass",
    "ap": "allpass",
    "lowshelforf": "lowshelforf",
    "lso": "lowshelforf",
    "highshelforf": "highshelforf",
    "hso": "highshelforf",
    "peakmatched": "peakmatched",
    "pkm": "peakmatched",
}


def _canonical_filter_type(filter_type: object) -> str:
    key = "".join(character for character in str(filter_type).lower() if character.isalnum())
    try:
        return _FILTER_TYPE_ALIASES[key]
    except KeyError as error:
        raise ValueError(f"unsupported biquad filter type {filter_type!r}") from error


def biquad_coefficients(
    filter_type: object,
    freq: float,
    sample_rate: float = 48_000.0,
    q: float = 1.0,
    db_gain: float = 0.0,
) -> tuple[float, float, float, float, float]:
    """Return Rust-canonical normalized ``(a1, a2, b0, b1, b2)`` coefficients.

    This mirrors ``math_audio_iir_fir::Biquad::constants`` for every
    ``BiquadFilterType`` used by RoomEQ. Keeping coefficient generation in one
    helper also prevents the plotted aggregate and per-filter curves from
    drifting apart.
    """
    kind = _canonical_filter_type(filter_type)
    sample_rate = float(sample_rate)
    if not math.isfinite(sample_rate) or sample_rate <= 0.0:
        sample_rate = 48_000.0
    nyquist = sample_rate / 2.0
    margin = nyquist * math.sqrt(sys.float_info.epsilon)
    freq = float(freq)
    if not math.isfinite(freq) or freq <= 0.0:
        freq = margin
    elif freq >= nyquist:
        freq = nyquist - margin

    q = float(q)
    if not math.isfinite(q):
        raise ValueError(f"biquad Q must be finite, got {q!r}")
    if q == 0.0:
        if kind == "notch":
            q = 30.0
        elif kind in {"bandpass", "highpass", "lowpass"}:
            q = 1.0 / math.sqrt(2.0)
        elif kind in {"lowshelf", "highshelf", "lowshelforf", "highshelforf"}:
            q = 1.0668676536332304
    if q <= 0.0:
        q = 1.0e-2

    db_gain = float(db_gain)
    if not math.isfinite(db_gain):
        db_gain = 0.0
    amplitude = 10.0 ** (db_gain / 40.0)
    omega = 2.0 * math.pi * freq / sample_rate
    sine = math.sin(omega)
    cosine = math.cos(omega)
    alpha = sine / (2.0 * q)
    beta = math.sqrt(2.0 * amplitude)

    if kind == "lowpass":
        b0, b1, b2 = (1.0 - cosine) / 2.0, 1.0 - cosine, (1.0 - cosine) / 2.0
        a0, a1, a2 = 1.0 + alpha, -2.0 * cosine, 1.0 - alpha
    elif kind in {"highpass", "highpassvariableq"}:
        b0, b1, b2 = (1.0 + cosine) / 2.0, -(1.0 + cosine), (1.0 + cosine) / 2.0
        a0, a1, a2 = 1.0 + alpha, -2.0 * cosine, 1.0 - alpha
    elif kind == "bandpass":
        b0, b1, b2 = alpha, 0.0, -alpha
        a0, a1, a2 = 1.0 + alpha, -2.0 * cosine, 1.0 - alpha
    elif kind == "notch":
        b0, b1, b2 = 1.0, -2.0 * cosine, 1.0
        a0, a1, a2 = 1.0 + alpha, -2.0 * cosine, 1.0 - alpha
    elif kind == "peak":
        b0, b1, b2 = 1.0 + alpha * amplitude, -2.0 * cosine, 1.0 - alpha * amplitude
        a0, a1, a2 = 1.0 + alpha / amplitude, -2.0 * cosine, 1.0 - alpha / amplitude
    elif kind == "lowshelf":
        b0 = amplitude * ((amplitude + 1.0) - (amplitude - 1.0) * cosine + beta * sine)
        b1 = 2.0 * amplitude * ((amplitude - 1.0) - (amplitude + 1.0) * cosine)
        b2 = amplitude * ((amplitude + 1.0) - (amplitude - 1.0) * cosine - beta * sine)
        a0 = (amplitude + 1.0) + (amplitude - 1.0) * cosine + beta * sine
        a1 = -2.0 * ((amplitude - 1.0) + (amplitude + 1.0) * cosine)
        a2 = (amplitude + 1.0) + (amplitude - 1.0) * cosine - beta * sine
    elif kind == "highshelf":
        b0 = amplitude * ((amplitude + 1.0) + (amplitude - 1.0) * cosine + beta * sine)
        b1 = -2.0 * amplitude * ((amplitude - 1.0) + (amplitude + 1.0) * cosine)
        b2 = amplitude * ((amplitude + 1.0) + (amplitude - 1.0) * cosine - beta * sine)
        a0 = (amplitude + 1.0) - (amplitude - 1.0) * cosine + beta * sine
        a1 = 2.0 * ((amplitude - 1.0) - (amplitude + 1.0) * cosine)
        a2 = (amplitude + 1.0) - (amplitude - 1.0) * cosine - beta * sine
    elif kind == "allpass":
        b0, b1, b2 = 1.0 - alpha, -2.0 * cosine, 1.0 + alpha
        a0, a1, a2 = 1.0 + alpha, -2.0 * cosine, 1.0 - alpha
    elif kind in {"lowshelforf", "highshelforf"}:
        linear_gain = 10.0 ** (db_gain / 20.0)
        if kind == "lowshelforf":
            a0 = (amplitude + 1.0) + (amplitude - 1.0) * cosine + beta * sine
            a1 = -2.0 * ((amplitude - 1.0) + (amplitude + 1.0) * cosine)
            a2 = (amplitude + 1.0) + (amplitude - 1.0) * cosine - beta * sine
            sum_b = linear_gain * (a0 + a1 + a2)
            diff_b = a0 - a1 + a2
        else:
            a0 = (amplitude + 1.0) - (amplitude - 1.0) * cosine + beta * sine
            a1 = 2.0 * ((amplitude - 1.0) - (amplitude + 1.0) * cosine)
            a2 = (amplitude + 1.0) - (amplitude - 1.0) * cosine - beta * sine
            sum_b = a0 + a1 + a2
            diff_b = linear_gain * (a0 - a1 + a2)

        b1 = (sum_b - diff_b) / 2.0
        p = (sum_b + diff_b) / 2.0
        cosine_2w = math.cos(2.0 * omega)
        sine_2w = math.sin(2.0 * omega)
        denominator_real = a0 + a1 * cosine + a2 * cosine_2w
        denominator_imag = -a1 * sine - a2 * sine_2w
        target = linear_gain * (denominator_real**2 + denominator_imag**2)
        known = (
            p**2 / 2.0
            + b1**2
            + 2.0 * b1 * p * cosine
            + p**2 / 2.0 * cosine_2w
        )
        d_coefficient = 0.5 - 0.5 * cosine_2w
        d_squared = (target - known) / d_coefficient if abs(d_coefficient) > 1.0e-15 else 0.0
        d_value = math.sqrt(d_squared) if d_squared >= 0.0 else 0.0
        d_signed = d_value if linear_gain >= 1.0 else -d_value
        b0, b2 = (p + d_signed) / 2.0, (p - d_signed) / 2.0
    else:  # peakmatched
        linear_gain = 10.0 ** (db_gain / 20.0)
        radius = math.exp(-(omega / q) / 2.0)
        a0, a1, a2 = 1.0, -2.0 * radius * cosine, radius**2
        sum_b = 1.0 + a1 + a2
        diff_b = 1.0 - a1 + a2
        b1 = (sum_b - diff_b) / 2.0
        p = (sum_b + diff_b) / 2.0
        cosine_2w = math.cos(2.0 * omega)
        sine_2w = math.sin(2.0 * omega)
        denominator_real = 1.0 + a1 * cosine + a2 * cosine_2w
        denominator_imag = -a1 * sine - a2 * sine_2w
        target = linear_gain**2 * (denominator_real**2 + denominator_imag**2)
        known = (
            p**2 / 2.0
            + b1**2
            + 2.0 * b1 * p * cosine
            + p**2 / 2.0 * cosine_2w
        )
        d_coefficient = 0.5 - 0.5 * cosine_2w
        d_squared = (target - known) / d_coefficient if abs(d_coefficient) > 1.0e-15 else 0.0
        d_value = math.sqrt(d_squared) if d_squared >= 0.0 else 0.0
        d_signed = d_value if linear_gain >= 1.0 else -d_value
        b0, b2 = (p + d_signed) / 2.0, (p - d_signed) / 2.0

    if abs(a0) < 1.0e-15:
        return (0.0, 0.0, 1.0, 0.0, 0.0)
    return (a1 / a0, a2 / a0, b0 / a0, b1 / a0, b2 / a0)


def compute_eq_response(
    filters: list[dict], freq_points: list[float], sample_rate: float = 48_000.0
) -> list[float]:
    """Compute the combined EQ response using Rust-canonical biquad math."""
    if not filters or not freq_points:
        return []

    combined_db = [0.0] * len(freq_points)
    for filt in filters:
        coefficients = biquad_coefficients(
            filt.get("filter_type", "peak"),
            filt.get("freq", filt.get("frequency", 1_000.0)),
            sample_rate,
            filt.get("q", 1.0),
            filt.get("db_gain", filt.get("gain", 0.0)),
        )
        a1, a2, b0, b1, b2 = coefficients
        for index, frequency in enumerate(freq_points):
            if frequency <= 0.0:
                continue
            omega = 2.0 * math.pi * frequency / sample_rate
            cosine, sine = math.cos(omega), math.sin(omega)
            cosine_2w, sine_2w = math.cos(2.0 * omega), math.sin(2.0 * omega)
            numerator = complex(b0 + b1 * cosine + b2 * cosine_2w, -b1 * sine - b2 * sine_2w)
            denominator = complex(1.0 + a1 * cosine + a2 * cosine_2w, -a1 * sine - a2 * sine_2w)
            denominator_magnitude = abs(denominator)
            if denominator_magnitude > 1.0e-10:
                magnitude = max(abs(numerator) / denominator_magnitude, 1.0e-10)
                combined_db[index] += 20.0 * math.log10(magnitude)

    return combined_db


def _biquad_complex_response(
    coefficients: tuple[float, float, float, float, float],
    frequency: float,
    sample_rate: float,
) -> complex:
    """Return the complex response of one normalized biquad."""
    a1, a2, b0, b1, b2 = coefficients
    omega = 2.0 * math.pi * frequency / sample_rate
    z1 = complex(math.cos(omega), -math.sin(omega))
    z2 = z1 * z1
    denominator = 1.0 + a1 * z1 + a2 * z2
    if abs(denominator) <= 1.0e-15:
        return 0j
    return (b0 + b1 * z1 + b2 * z2) / denominator


def _crossover_response(
    crossover_type: str,
    output: str,
    frequency_hz: float,
    frequencies: list[float],
    sample_rate: float,
) -> list[complex]:
    """Return a RoomEQ crossover transfer function on ``frequencies``."""
    kind = "".join(ch for ch in crossover_type.lower() if ch.isalnum())
    output_kind = "lowpass" if output.lower().startswith("low") else "highpass"

    # Linkwitz-Riley filters are cascaded identical Butterworth sections.
    if kind in {"lr24", "lr4"}:
        section_qs = [1.0 / math.sqrt(2.0)] * 2
    elif kind in {"lr48", "lr8"}:
        section_qs = [0.541196100146197, 1.306562964876377] * 2
    elif kind in {"butterworth12", "bw12"}:
        section_qs = [1.0 / math.sqrt(2.0)]
    elif kind in {"butterworth24", "bw24"}:
        section_qs = [0.541196100146197, 1.306562964876377]
    else:
        raise ValueError(f"unsupported crossover type: {crossover_type!r}")

    sections = [
        biquad_coefficients(output_kind, frequency_hz, sample_rate, q, 0.0)
        for q in section_qs
    ]
    response: list[complex] = []
    for frequency in frequencies:
        value = 1.0 + 0.0j
        for section in sections:
            value *= _biquad_complex_response(section, frequency, sample_rate)
        response.append(value)
    return response


def apply_plugins_to_curve(
    curve: dict | None,
    plugins: list[dict],
    sample_rate: float = 48_000.0,
) -> dict | None:
    """Apply gain/EQ/crossover/delay plugins to an acoustic response curve.

    Magnitude and phase are both propagated. This is used by report generation
    to predict what a microphone sees after the exported DSP chain.
    """
    if not curve:
        return None
    frequencies = curve.get("freq") or []
    spl = curve.get("spl") or []
    if not frequencies or len(frequencies) != len(spl):
        return None

    transfer = [1.0 + 0.0j for _ in frequencies]
    for plugin in plugins:
        plugin_type = str(plugin.get("plugin_type", "")).lower()
        parameters = plugin.get("parameters", {}) or {}
        if plugin_type == "gain":
            gain = 10.0 ** (float(parameters.get("gain_db", 0.0)) / 20.0)
            if parameters.get("invert", False):
                gain = -gain
            transfer = [value * gain for value in transfer]
        elif plugin_type == "delay":
            delay_seconds = float(parameters.get("delay_ms", 0.0)) / 1000.0
            transfer = [
                value
                * complex(
                    math.cos(-2.0 * math.pi * frequency * delay_seconds),
                    math.sin(-2.0 * math.pi * frequency * delay_seconds),
                )
                for value, frequency in zip(transfer, frequencies)
            ]
        elif plugin_type == "eq":
            for filt in parameters.get("filters", []):
                coefficients = biquad_coefficients(
                    filt.get("filter_type", "peak"),
                    filt.get("freq", filt.get("frequency", 1_000.0)),
                    sample_rate,
                    filt.get("q", 1.0),
                    filt.get("db_gain", filt.get("gain", 0.0)),
                )
                transfer = [
                    value
                    * _biquad_complex_response(coefficients, frequency, sample_rate)
                    for value, frequency in zip(transfer, frequencies)
                ]
        elif plugin_type == "crossover":
            crossover = _crossover_response(
                str(parameters.get("type", "LR24")),
                str(parameters.get("output", "low")),
                float(parameters.get("frequency", 80.0)),
                frequencies,
                sample_rate,
            )
            transfer = [value * xover for value, xover in zip(transfer, crossover)]

    phase = curve.get("phase")
    has_phase = isinstance(phase, list) and len(phase) == len(frequencies)
    result_spl: list[float] = []
    result_phase: list[float] = []
    for index, value in enumerate(transfer):
        # Match Rust's acoustic-domain floor after the complete transfer is
        # applied. Flooring the dimensionless transfer first lets a deeply
        # attenuated input fall below the canonical -240 dB output floor.
        acoustic_magnitude = 10.0 ** (float(spl[index]) / 20.0) * abs(value)
        result_spl.append(20.0 * math.log10(max(acoustic_magnitude, 1.0e-12)))
        if has_phase:
            result_phase.append(float(phase[index]) + math.degrees(math.atan2(value.imag, value.real)))

    result: dict = {"freq": list(frequencies), "spl": result_spl}
    if has_phase:
        result["phase"] = result_phase
    return result


def replay_serialized_output(data: dict) -> dict[str, dict]:
    """Replay serialized channel chains without consuming reported curves."""
    sample_rate = float(data.get("sample_rate", 48_000.0) or 48_000.0)
    replayed = {}
    for name, channel in (data.get("channels", {}) or {}).items():
        curve = channel.get("initial_curve") or channel.get("raw_curve")
        if curve and channel.get("plugins") is not None:
            replayed[name] = apply_plugins_to_curve(curve, channel["plugins"], sample_rate)
    return replayed


def build_post_dsp_source_curves(data: dict) -> dict[str, dict]:
    """Build one microphone-predicted post-DSP curve per input channel.

    For a bass-managed main, the result is the coherent acoustic sum of its
    emitted high-pass main branch and only that input's low-pass sub branch.
    The LFE result contains only its own route. The aggregate emitted LFE
    ``final_curve`` is deliberately not reused because it represents the whole
    bass bus and would duplicate unrelated source channels.
    """
    deployed = data.get("deployed_source_curves") or {}
    if deployed:
        return deployed

    channels = data.get("channels", {}) or {}
    bass_management = (data.get("metadata", {}) or {}).get("bass_management", {}) or {}
    graph = bass_management.get("routing_graph", {}) or {}
    routes = graph.get("routes", []) or []
    input_trim_db = graph.get("input_trim_db", {}) or {}
    if not routes:
        return {
            name: channel["final_curve"]
            for name, channel in channels.items()
            if channel.get("final_curve")
        }

    sub_name = bass_management.get("physical_sub_output") or bass_management.get("lfe_channel") or "LFE"
    sub_channel = channels.get(sub_name, {})
    sub_initial = sub_channel.get("initial_curve")
    if not sub_initial:
        return {
            name: channel["final_curve"]
            for name, channel in channels.items()
            if channel.get("final_curve")
        }

    # Only destination post-route processing belongs to every signal emitted
    # by the physical sub. The LFE chain's pre-route processing belongs to its
    # logical input and must not leak into redirected L/R/etc. branches.
    sub_post_route_plugins = [
        plugin
        for plugin in sub_channel.get("plugins", [])
        if (plugin.get("parameters", {}) or {}).get("room_eq_stage") == "post_route"
    ]
    sample_rate = float(data.get("sample_rate", 48_000.0) or 48_000.0)

    routed_by_source: dict[str, dict] = {}
    for route in routes:
        route_kind = route.get("route_kind")
        if route_kind not in {"redirected_bass_lowpass_to_sub", "lfe_lowpass_to_sub"}:
            continue
        low_pass_hz = route.get("low_pass_hz")
        if low_pass_hz is None:
            continue
        source_name = str(route.get("source_channel"))
        source_plugins = (channels.get(source_name, {}) or {}).get("plugins", [])
        source_pre_route_plugins = [
            plugin
            for plugin in source_plugins
            if (plugin.get("parameters", {}) or {}).get("room_eq_stage")
            == "pre_route"
            and not (
                plugin.get("plugin_type") == "gain"
                and (plugin.get("parameters", {}) or {}).get("label")
                == "post_dsp_input_level_alignment"
            )
        ]
        # The canonical routing graph owns this labeled calibration trim.
        # Excluding its serialized plugin above keeps Rust and Python from
        # applying it twice while every other pre-route transfer remains common
        # to the main and redirected-sub branches.
        route_input_trim_db = float(input_trim_db.get(source_name, 0.0))
        route_plugins = [
            *source_pre_route_plugins,
            {
                "plugin_type": "crossover",
                "parameters": {
                    "type": route.get("crossover_type", "LR24"),
                    "output": "low",
                    "frequency": low_pass_hz,
                },
            },
            {
                "plugin_type": "gain",
                "parameters": {
                    "gain_db": float(route.get("gain_db", 0.0))
                    + route_input_trim_db,
                    "invert": bool(route.get("polarity_inverted", False)),
                },
            },
            {
                "plugin_type": "delay",
                "parameters": {"delay_ms": route.get("delay_ms", 0.0)},
            },
            *sub_post_route_plugins,
        ]
        routed = apply_plugins_to_curve(sub_initial, route_plugins, sample_rate)
        if routed:
            routed_by_source[source_name] = routed

    results: dict[str, dict] = {}
    for name, channel in channels.items():
        routed = routed_by_source.get(name)
        if name == sub_name:
            if routed:
                results[name] = routed
        elif routed:
            combined = complex_sum_curves(channel.get("final_curve"), routed)
            if combined:
                results[name] = combined
        elif channel.get("final_curve"):
            results[name] = channel["final_curve"]
    return results


def unwrap_phase(phase_deg: list[float]) -> list[float]:
    """Unwrap phase in degrees to remove discontinuities.

    Handles arbitrarily large jumps by rounding the correction to the
    nearest multiple of 360 degrees.
    """
    if not phase_deg:
        return phase_deg
    unwrapped = [phase_deg[0]]
    for i in range(1, len(phase_deg)):
        diff = phase_deg[i] - unwrapped[-1]
        # Round to nearest multiple of 360 to remove wrapping
        correction = round(diff / 360.0) * 360.0
        unwrapped.append(phase_deg[i] - correction)
    return unwrapped


def wrap_phase(phase_deg: list[float]) -> list[float]:
    """Wrap phase to the JSON/report convention ``[-180, 180)``."""
    return [((float(value) + 180.0) % 360.0) - 180.0 for value in phase_deg]


def compute_group_delay_from_ir(
    ir: dict | None,
    min_freq: float = 20.0,
    max_freq: float = 20_000.0,
    point_count: int = 512,
) -> tuple[list[float], list[float]]:
    """Derive group delay from an impulse-response waveform.

    The serialized curve phase is intentionally wrapped for display and a
    sparse log-frequency phase curve cannot be unwrapped reliably when the
    acoustic propagation delay spans multiple turns between points.  The IR
    retains that timing, so its dense FFT phase is the authoritative source
    for the group-delay plot.
    """
    if not ir:
        return [], []
    time_ms = np.asarray(ir.get("time_ms", []), dtype=float)
    amplitude = np.asarray(ir.get("amplitude", []), dtype=float)
    if time_ms.size < 4 or amplitude.size != time_ms.size:
        return [], []

    steps_ms = np.diff(time_ms)
    finite_steps = steps_ms[np.isfinite(steps_ms) & (steps_ms > 0.0)]
    if finite_steps.size == 0:
        return [], []
    sample_rate = 1000.0 / float(np.median(finite_steps))
    nyquist = sample_rate / 2.0
    upper = min(float(max_freq), nyquist)
    lower = max(float(min_freq), sample_rate / amplitude.size)
    if not math.isfinite(sample_rate) or lower >= upper:
        return [], []

    spectrum = np.fft.rfft(amplitude)
    fft_freq = np.fft.rfftfreq(amplitude.size, d=1.0 / sample_rate)
    magnitude = np.abs(spectrum)
    peak_magnitude = float(np.max(magnitude))
    if not math.isfinite(peak_magnitude) or peak_magnitude <= 1.0e-12:
        return [], []

    phase = np.unwrap(np.angle(spectrum))
    output_freq = np.geomspace(lower, upper, max(8, int(point_count)))
    output_phase = np.interp(output_freq, fft_freq, phase)
    omega = 2.0 * np.pi * output_freq
    group_delay_ms = -np.gradient(output_phase, omega) * 1000.0
    output_magnitude = np.interp(output_freq, fft_freq, magnitude)
    valid = (
        np.isfinite(group_delay_ms)
        & np.isfinite(output_magnitude)
        & (output_magnitude >= peak_magnitude * 1.0e-8)
    )
    return output_freq[valid].tolist(), group_delay_ms[valid].tolist()


def compute_group_delay(
    freq: list[float], phase_deg: list[float],
) -> tuple[list[float], list[float]]:
    """Compute group delay from frequency and phase data.

    Group delay = -d(phase)/d(omega), where omega = 2*pi*f.
    Phase is unwrapped before differentiation.

    Returns:
        (freq_out, gd_ms): Frequency points and group delay in milliseconds.
        Output has len(freq)-1 points (centered between input points).
    """
    if len(freq) < 2 or len(phase_deg) < 2:
        return [], []

    unwrapped = unwrap_phase(phase_deg)

    freq_out: list[float] = []
    gd_ms: list[float] = []
    for i in range(len(freq) - 1):
        f0, f1 = freq[i], freq[i + 1]
        if f0 <= 0 or f1 <= 0 or f1 == f0:
            continue
        omega0 = 2 * math.pi * f0
        omega1 = 2 * math.pi * f1
        dphi = math.radians(unwrapped[i + 1] - unwrapped[i])
        domega = omega1 - omega0
        gd_s = -dphi / domega
        freq_out.append(math.sqrt(f0 * f1))  # geometric mean
        gd_ms.append(gd_s * 1000.0)

    return freq_out, gd_ms


def generate_freq_points(min_freq: float = 20.0, max_freq: float = 20000.0, n_points: int = 200) -> list[float]:
    """Generate logarithmically spaced frequency points."""
    log_min = math.log10(min_freq)
    log_max = math.log10(max_freq)
    return [10 ** (log_min + (log_max - log_min) * i / (n_points - 1)) for i in range(n_points)]


def complex_sum_curves(l_curve: dict | None, r_curve: dict | None) -> dict | None:
    """Coherent (complex) sum of two frequency-response curves.

    Each curve is a dict with `freq` (Hz), `spl` (dB), and optionally
    `phase` (degrees). Both curves must share the same frequency grid.

    When both curves have phase data the sum is computed in the complex
    plane: each band is converted to a complex amplitude
    `10**(spl/20) * exp(j * phase_rad)`, the two phasors are added, and
    the magnitude is converted back to dB. The output curve also
    carries the resulting `phase` so downstream phase / group-delay
    plots keep working.

    When phase is missing on either side, the function falls back to an
    incoherent power sum: `10*log10(10**(L/10) + 10**(R/10))`. This
    yields the same result the ear would hear for two uncorrelated
    sources but does not preserve phase information, so the output
    curve has no `phase` field.

    Returns `None` if either input is missing or the frequency grids
    do not match (in which case the caller should skip rendering the
    L+R panel rather than silently producing wrong data).
    """
    if not l_curve or not r_curve:
        return None
    l_freq = l_curve.get("freq") or []
    r_freq = r_curve.get("freq") or []
    l_spl = l_curve.get("spl") or []
    r_spl = r_curve.get("spl") or []
    if not l_freq or not r_freq or not l_spl or not r_spl:
        return None
    if len(l_freq) != len(r_freq) or len(l_spl) != len(r_spl):
        return None
    # Sanity check: grids should be element-wise equal in this pipeline
    # (both channels are interpolated onto the same logarithmic grid by
    # the optimizer). Bail out if they aren't to avoid silent garbage.
    for lf, rf in zip(l_freq, r_freq):
        if abs(lf - rf) > max(1e-6, lf * 1e-9):
            return None

    l_phase = l_curve.get("phase")
    r_phase = r_curve.get("phase")
    if (
        l_phase is not None
        and r_phase is not None
        and len(l_phase) == len(l_freq)
        and len(r_phase) == len(r_freq)
    ):
        # Local references so type-checkers can narrow Optional → list.
        l_phase_arr: list[float] = l_phase
        r_phase_arr: list[float] = r_phase
        spl_out: list[float] = []
        phase_out: list[float] = []
        for i in range(len(l_freq)):
            mag_l = 10.0 ** (l_spl[i] / 20.0)
            mag_r = 10.0 ** (r_spl[i] / 20.0)
            phi_l = math.radians(l_phase_arr[i])
            phi_r = math.radians(r_phase_arr[i])
            re = mag_l * math.cos(phi_l) + mag_r * math.cos(phi_r)
            im = mag_l * math.sin(phi_l) + mag_r * math.sin(phi_r)
            mag_sum = math.sqrt(re * re + im * im)
            if mag_sum <= 0.0:
                spl_out.append(-200.0)
                phase_out.append(0.0)
            else:
                spl_out.append(20.0 * math.log10(mag_sum))
                phase_out.append(math.degrees(math.atan2(im, re)))
        return {"freq": list(l_freq), "spl": spl_out, "phase": phase_out}

    # Phase missing — fall back to incoherent power sum.
    spl_out = [
        10.0 * math.log10(10.0 ** (l_spl[i] / 10.0) + 10.0 ** (r_spl[i] / 10.0))
        for i in range(len(l_freq))
    ]
    return {"freq": list(l_freq), "spl": spl_out}


def synthesize_lr_channel(l_ch_data: dict | None, r_ch_data: dict | None) -> dict | None:
    """Build a synthetic 'L+R' channel dict from the L and R channel data.

    Sums `initial_curve` and `final_curve` via `complex_sum_curves`.
    Drops `eq_response` (per-channel EQ filter responses are not
    summable in any meaningful way) and any per-channel impulse-response
    blobs (they would need to be summed in the time domain, out of scope
    here).

    Returns `None` if either side is missing both curves.
    """
    if not l_ch_data or not r_ch_data:
        return None
    initial = complex_sum_curves(
        l_ch_data.get("initial_curve"), r_ch_data.get("initial_curve")
    )
    final = complex_sum_curves(
        l_ch_data.get("final_curve"), r_ch_data.get("final_curve")
    )
    if not initial and not final:
        return None
    out: dict = {"channel": "L+R", "plugins": []}
    if initial:
        out["initial_curve"] = initial
    if final:
        out["final_curve"] = final
    return out
