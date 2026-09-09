"""Functions for extracting data from roomeq output JSON."""

import math


# Channel order mapping for sorting (classical order)
CHANNEL_ORDER_MAP = {
    "L": 10, "LEFT": 10,
    "R": 20, "RIGHT": 20,
    "C": 30, "CENTER": 30,
    "LFE": 40, "SUB": 40, "SUBWOOFER": 40, "LFE1": 41, "LFE2": 42,
    "SL": 50, "SURROUND LEFT": 50, "LS": 50,
    "SR": 60, "SURROUND RIGHT": 60, "RS": 60,
    "SBL": 70, "SURROUND BACK LEFT": 70, "LBS": 70, "LB": 70,
    "SBR": 80, "SURROUND BACK RIGHT": 80, "RBS": 80, "RB": 80,
    "FHL": 90, "FRONT HEIGHT LEFT": 90,
    "FHR": 100, "FRONT HEIGHT RIGHT": 100,
    "BHL": 110, "BACK HEIGHT LEFT": 110,
    "BHR": 120, "BACK HEIGHT RIGHT": 120,
}


def get_channel_sort_key(channel_name: str) -> tuple[int, str]:
    """Get sort key for a channel name."""
    name_upper = channel_name.upper()
    # Try exact match
    if name_upper in CHANNEL_ORDER_MAP:
        return (CHANNEL_ORDER_MAP[name_upper], channel_name)

    # Try to see if it starts with one of the keys (e.g. "L (tweeter)")
    for key, order in CHANNEL_ORDER_MAP.items():
        if name_upper.startswith(key) and (len(name_upper) == len(key) or not name_upper[len(key)].isalnum()):
            return (order, channel_name)

    # Default: large number to put unknown at the end
    return (1000, channel_name)


def compute_y_range(curves: list[dict | None]) -> tuple[float, float]:
    """Compute y-axis range from curve data: 50 dB span, max rounded up to next multiple of 5."""
    all_spl = []
    for curve in curves:
        if curve and "spl" in curve:
            all_spl.extend(curve["spl"])

    if not all_spl:
        return (-20, 30)

    upper = math.ceil(max(all_spl) / 5) * 5
    return (upper - 50, upper)


def compute_average_spl_in_range(
    curve: dict, min_freq: float = 20.0, max_freq: float = 1200.0
) -> float:
    """Compute average SPL in a frequency range."""
    if not curve or "freq" not in curve or "spl" not in curve:
        return 0.0

    freq = curve["freq"]
    spl = curve["spl"]

    values_in_range = [
        s for f, s in zip(freq, spl) if min_freq <= f <= max_freq
    ]

    if not values_in_range:
        return 0.0

    return sum(values_in_range) / len(values_in_range)


def extract_crossover_frequencies(channel_data: dict) -> list[float]:
    """
    Extract crossover frequencies from a channel's plugin configuration.

    Looks for:
    - "crossover" plugins in driver chains (active crossovers)
    - "band_split" plugins in main chain (mixed mode crossovers)

    Returns:
        Sorted list of unique crossover frequencies in Hz
    """
    crossover_freqs = set()

    # Check main plugins for band_split and crossover
    plugins = channel_data.get("plugins", [])
    for plugin in plugins:
        if plugin.get("plugin_type") == "band_split":
            freq = plugin.get("parameters", {}).get("frequency")
            if freq:
                crossover_freqs.add(float(freq))
        elif plugin.get("plugin_type") == "crossover":
            freq = plugin.get("parameters", {}).get("frequency")
            if freq:
                crossover_freqs.add(float(freq))

    # Check driver chains for crossover plugins
    drivers = channel_data.get("drivers", [])
    for driver in drivers:
        driver_plugins = driver.get("plugins", [])
        for plugin in driver_plugins:
            if plugin.get("plugin_type") == "crossover":
                freq = plugin.get("parameters", {}).get("frequency")
                if freq:
                    crossover_freqs.add(float(freq))

    return sorted(crossover_freqs)


def get_plottable_drivers(channel_data: dict | None) -> list[dict]:
    """Return driver dicts that carry their own measured initial curve.

    A channel with such drivers (multi-sub LFE, multi-driver speakers) is
    displayed per driver: Row 1 of the overview already expands them, and
    the EQ row plus the per-channel tabs follow the same rule so a
    two-subwoofer setup shows one EQ and one tab per subwoofer instead of
    a single collapsed LFE entry.
    """
    if not isinstance(channel_data, dict):
        return []
    drivers = channel_data.get("drivers") or []
    plottable = []
    for driver in drivers:
        if not isinstance(driver, dict):
            continue
        initial = driver.get("initial_curve") or {}
        if initial.get("freq") and initial.get("spl"):
            plottable.append(driver)
    return plottable


def _configured_sub_names(data: dict, channel_name: str, driver_count: int) -> list[str]:
    """Resolve friendly per-driver names from the effective config.

    For a bass-managed channel the logical channel (e.g. ``LFE``) maps to
    a speaker group (e.g. ``subs``) via ``system.speakers``, and the group
    lists its physical subwoofers by name (e.g. ``Left Sub``). Returns []
    when the config is absent or does not match ``driver_count`` so callers
    fall back to the serialized driver names.
    """
    if driver_count <= 0:
        return []
    effective = ((data.get("metadata") or {}).get("effective_config") or {})
    if not isinstance(effective, dict):
        return []
    system_speakers = ((effective.get("system") or {}).get("speakers") or {})
    speaker_key = system_speakers.get(channel_name) if isinstance(system_speakers, dict) else None
    speakers = effective.get("speakers") or {}
    group = speakers.get(speaker_key) if isinstance(speakers, dict) else None
    subwoofers = (group or {}).get("subwoofers") or [] if isinstance(group, dict) else []
    names = [
        str(entry.get("name"))
        for entry in subwoofers
        if isinstance(entry, dict) and entry.get("name")
    ]
    return names if len(names) == driver_count else []


def driver_display_names(data: dict, channel_name: str) -> list[str]:
    """Return one display name per plottable driver of ``channel_name``."""
    channel = (data.get("channels") or {}).get(channel_name) or {}
    drivers = get_plottable_drivers(channel)
    friendly = _configured_sub_names(data, channel_name, len(drivers))
    if friendly:
        return friendly
    names = []
    for index, driver in enumerate(drivers):
        raw = driver.get("name") or driver.get("index", index)
        names.append(str(raw))
    return names


def channel_has_eq(channel_data: dict | None) -> bool:
    """Return whether a channel carries any EQ (channel or driver level).

    Covers stored ``eq_response`` curves as well as EQ filter plugins on
    the channel chain or any driver chain. The overview EQ row only
    expands multi-driver channels per sub when this holds, so runs
    without EQ keep their legacy (empty) EQ row instead of showing
    crossover-only shaping labeled as EQ.
    """
    if not isinstance(channel_data, dict):
        return False
    if (channel_data.get("eq_response") or {}).get("spl"):
        return True
    chains = [channel_data.get("plugins") or []]
    for driver in channel_data.get("drivers") or []:
        if isinstance(driver, dict):
            chains.append(driver.get("plugins") or [])
    for plugins in chains:
        for plugin in plugins:
            if not isinstance(plugin, dict):
                continue
            if plugin.get("plugin_type") != "eq":
                continue
            if (plugin.get("parameters") or {}).get("filters"):
                return True
    return False


def display_channel_entries(data: dict) -> list[dict]:
    """Expand multi-driver channels into per-driver display entries.

    Returns ordered ``{"label", "channel", "driver"}`` dicts where
    ``driver`` is the driver index or `None` for whole-channel entries.
    Channels without plottable drivers yield a single entry labeled with
    the channel name, preserving legacy behavior.
    """
    channels = data.get("channels") or {}
    entries: list[dict] = []
    for name in sorted(channels.keys(), key=get_channel_sort_key):
        channel = channels[name]
        drivers = get_plottable_drivers(channel)
        if not drivers:
            entries.append({"label": name, "channel": name, "driver": None})
            continue
        for index, label in enumerate(driver_display_names(data, name)):
            entries.append({"label": label, "channel": name, "driver": index})
    return entries


def extract_eq_passes(channel_data: dict) -> list[dict]:
    """
    Extract EQ plugins from a channel, grouped by pass label.

    The 3-pass pipeline labels EQ plugins as:
    - "cea2034_speaker_correction" (Pass 1)
    - "room_eq_correction" (Pass 2, or unlabeled)
    - "user_preference" (Pass 3)

    Unlabeled EQ plugins are assigned to "Room EQ".

    Returns:
        List of dicts with keys: label, display_name, filters, color
    """
    plugins = channel_data.get("plugins", [])

    PASS_DISPLAY_NAMES = {
        "cea2034_speaker_correction": "Pass 1: Speaker Correction (CEA2034)",
        "room_eq_correction": "Pass 2: Room EQ",
        "user_preference": "Pass 3: User Preference",
    }

    PASS_COLORS = {
        "cea2034_speaker_correction": "rgba(255, 165, 0, 0.9)",   # orange
        "room_eq_correction": "rgba(100, 100, 255, 0.9)",         # blue
        "user_preference": "rgba(180, 100, 255, 0.9)",            # purple
    }

    passes: list[dict] = []

    for plugin in plugins:
        if plugin.get("plugin_type") != "eq":
            continue

        params = plugin.get("parameters", {})
        label = params.get("label", "")
        filters = params.get("filters", [])

        if not filters:
            continue

        display_name = PASS_DISPLAY_NAMES.get(label, "Room EQ")
        color = PASS_COLORS.get(label, "rgba(100, 100, 255, 0.9)")

        passes.append({
            "label": label,
            "display_name": display_name,
            "filters": filters,
            "color": color,
        })

    return passes


def get_all_crossover_frequencies(data: dict) -> list[float]:
    """
    Extract all unique crossover frequencies from all channels.

    Returns:
        Sorted list of unique crossover frequencies in Hz
    """
    all_freqs = set()
    channels = data.get("channels", {})

    for channel_data in channels.values():
        freqs = extract_crossover_frequencies(channel_data)
        all_freqs.update(freqs)

    return sorted(all_freqs)
