#!/usr/bin/env python3
"""Check a selected public RoomEQ delay against its saved complex DSP graph."""

import argparse
import cmath
import copy
import json
import math
from pathlib import Path
import re

from src.payload_binding import verify_payload_binding
from verify_routed_electrical import output_complex_transfers


TRIAL_ID = re.compile(r"^joint_drive_delay_(.+)_(\d+)_to_(-?\d+(?:\.\d+)?)$")
GAIN_DELAY_TRIAL_ID = re.compile(
    r"^joint_drive_gain_delay_(.+)_(\d+)_to_(-?\d+(?:\.\d+)?)"
    r"__joint_drive_delay_(.+)_(\d+)_to_(-?\d+(?:\.\d+)?)$"
)
FREQUENCIES_HZ = (40.0, 80.0, 120.0)


def check_selected_delay(data: dict, rate: float = 48_000.0) -> dict:
    """Validate the selected correction in the final bound independent graph."""
    verified, reason, identity = verify_payload_binding(data)
    if not verified:
        raise ValueError(f"selected graph payload binding failed: {reason}")
    if (data.get("metadata", {}).get("bass_management") or {}).get("routing_graph"):
        raise ValueError("this selected-delay contract requires an independent driver graph")
    channels = data["channels"]
    if len(channels) != 1 or data.get("global_plugins"):
        raise ValueError("selected-delay fixture has unexpected channel/global topology")
    channel_name, channel = next(iter(channels.items()))
    drivers = sorted(channel.get("drivers") or [], key=lambda driver: driver["index"])
    if len(drivers) < 2 or [driver["index"] for driver in drivers] != list(range(len(drivers))):
        raise ValueError("selected-delay fixture has invalid physical-driver indices")
    emitted_ms = []
    added_ms = []
    emitted_gain_db = []
    correction_gain = []
    for driver in drivers:
        emitted = added = 0.0
        gain_db = 0.0
        has_correction_gain = False
        for plugin in driver["plugins"]:
            if plugin["plugin_type"] == "gain":
                parameters = plugin["parameters"]
                gain = float(parameters["gain_db"])
                if not math.isfinite(gain):
                    raise ValueError("selected graph has invalid physical gain")
                gain_db += gain
                if parameters.get("label") == "joint_physical_drive_refinement":
                    if (parameters.get("room_eq_correction_gain") is not True
                            or parameters.get("room_eq_stage") != "post_route"):
                        raise ValueError("selected correction gain has invalid ownership")
                    has_correction_gain = True
                continue
            if plugin["plugin_type"] != "delay":
                continue
            parameters = plugin["parameters"]
            delay = float(parameters["delay_ms"])
            if not math.isfinite(delay) or delay < 0:
                raise ValueError("selected graph has invalid physical delay")
            emitted += delay
            if parameters.get("label") == "joint_physical_drive_delay_refinement":
                if parameters.get("room_eq_correction_delay") is not True or parameters.get("room_eq_stage") != "post_route":
                    raise ValueError("selected correction delay has invalid ownership")
                added += delay
        emitted_ms.append(emitted)
        added_ms.append(added)
        if not math.isfinite(gain_db):
            raise ValueError("selected graph has invalid physical gain sum")
        emitted_gain_db.append(gain_db)
        correction_gain.append(has_correction_gain)
    if not any(added_ms):
        raise ValueError("selected graph contains no correction-owned physical delay")

    stages = data["metadata"]["stage_outcomes"]
    objective = [stage for stage in stages if stage["stage"] == "final_candidate_objective"]
    if len(objective) != 1:
        raise ValueError("selected graph needs one final candidate objective")
    score = float(objective[0]["checks"][0]["observed"])
    if not math.isfinite(score):
        raise ValueError("selected graph has invalid candidate score")
    matching = []
    for stage in stages:
        for trial in stage.get("checks") or []:
            trial_id = trial.get("id", "")
            parsed = TRIAL_ID.fullmatch(trial_id)
            combined = GAIN_DELAY_TRIAL_ID.fullmatch(trial_id) if not parsed else None
            if (not parsed and not combined) or not trial["passed"] or trial.get("observed") is None:
                continue
            if combined:
                gain_name, gain_index_text, gain_target_text, name, index_text, target_text = combined.groups()
            else:
                name, index_text, target_text = parsed.groups()
            index = int(index_text)
            target_ms = float(target_text)
            if name != channel_name or index <= 0 or index >= len(drivers):
                continue
            if combined and (
                gain_name != channel_name
                or int(gain_index_text) != index
                or not correction_gain[index]
                or abs(emitted_gain_db[index] - float(gain_target_text)) >= 1e-9
            ):
                continue
            if (abs(float(trial["observed"]) - score) < 1e-9
                    and abs(emitted_ms[index] - emitted_ms[0] - target_ms) < 1e-9):
                matching.append((trial["id"], index, target_ms))
    if len(matching) != 1:
        raise ValueError("selected delay trial does not uniquely match emitted physical timing")
    trial_id, selected_index, target_ms = matching[0]

    uncorrected = copy.deepcopy(data)
    for driver in uncorrected["channels"][channel_name]["drivers"]:
        driver["plugins"] = [plugin for plugin in driver["plugins"]
                             if plugin["parameters"].get("label") != "joint_physical_drive_delay_refinement"]
    frequencies = list(FREQUENCIES_HZ)
    corrected = output_complex_transfers(data, frequencies, rate)
    baseline = output_complex_transfers(uncorrected, frequencies, rate)
    max_error = 0.0
    source = channel_name
    for driver in drivers:
        output = json.dumps(["driver", channel_name, driver["index"], driver["name"]],
                            separators=(",", ":"), ensure_ascii=False)
        for frequency, actual, before in zip(frequencies, corrected[output][source], baseline[output][source]):
            if abs(before) < 1e-12:
                raise ValueError("selected complex replay lacks a usable source transfer")
            expected = cmath.exp(-2j * math.pi * frequency * added_ms[driver["index"]] / 1000)
            max_error = max(max_error, abs(actual / before - expected))
    if max_error >= 1e-10:
        raise ValueError(f"selected complex delay replay differs from emitted control: {max_error}")
    return {
        "status": "selected_delay_complex_replay_verified",
        "payload_identity": identity,
        "selected_trial": trial_id,
        "selected_driver_index": selected_index,
        "target_relative_ms": target_ms,
        "frequencies_hz": frequencies,
        "max_complex_error": max_error,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path, help="Finalized selected-output JSON")
    arguments = parser.parse_args()
    print(json.dumps(check_selected_delay(json.loads(arguments.artifact.read_text())), indent=2))


if __name__ == "__main__":
    main()
