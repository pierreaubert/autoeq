#!/usr/bin/env python3
"""Check a measured native graph and its owned FIR bytes; print an audit row.

Acoustic replay belongs to Rust. This independent artifact check prevents a
successful process exit or a later mode's overwritten WAV from passing the audit.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path


def finite_tree(value, location="result"):
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"nonfinite value at {location}")
    if isinstance(value, dict):
        for key, child in value.items():
            finite_tree(child, f"{location}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            finite_tree(child, f"{location}[{index}]")


def inspect(path):
    path = Path(path)
    data = json.loads(path.read_text())
    finite_tree(data)
    metadata = data["metadata"]
    electrical = [stage for stage in metadata.get("stage_outcomes", [])
                  if stage["stage"] == "final_graph_sampled_electrical_headroom"]
    if len(electrical) != 1:
        raise ValueError("missing or ambiguous final electrical assessment")
    optimizer = (metadata.get("effective_config") or {}).get("optimizer") or {}
    policy = optimizer.get("finalization") or {}
    if policy.get("default_input_peak", 1.0) == 1.0 and not policy.get("input_peak_limits"):
        if any(not check["passed"] for check in electrical[0].get("checks", [])):
            raise ValueError("delivered graph exceeds unit-peak electrical safety")
    report = metadata["correction_acceptance"]
    outcome = report["outcome"]
    if outcome not in {"accepted", "unchanged", "rejected", "insufficient_evidence"}:
        raise ValueError(f"invalid outcome: {outcome}")
    if outcome == "accepted" and (
        not report["accepted"] or report["decision"] != "accepted"
    ):
        raise ValueError("accepted outcome contradicts detailed decision")
    metrics = report["metrics"]
    quality = report.get("acoustic_quality") or {}
    if outcome == "accepted" and not quality.get("final_seats"):
        raise ValueError("accepted correction has no physical-seat replay evidence")
    if outcome == "accepted" and metrics["improvement_db"] <= 1e-6:
        raise ValueError("accepted correction has no measured improvement")
    if report["decision"] == "identity_fallback":
        quality = report.get("acoustic_quality") or {}
        for seat in quality.get("final_seats", []):
            if abs(seat["improvement_db"]) > 1e-6:
                raise ValueError("identity fallback retains nonidentity seat metrics")

    manifest = json.loads(path.with_suffix(".manifest.json").read_text())
    if manifest["status"] != "complete":
        raise ValueError("native graph manifest is incomplete")
    for asset in manifest["assets_owned"]:
        if not Path(asset).is_file():
            raise ValueError(f"missing owned artifact: {asset}")
    inventory = metadata.get("final_convolution_sha256") or {}
    convolution_count = 0
    for channel in data["channels"].values():
        for chain in [channel, *(channel.get("drivers") or [])]:
            for plugin in chain.get("plugins", []):
                if plugin["plugin_type"] != "convolution":
                    continue
                convolution_count += 1
                reference = plugin["parameters"]["ir_file"]
                expected = inventory.get(reference)
                if not expected:
                    raise ValueError(f"unbound convolution: {reference}")
                asset = Path(reference)
                if not asset.is_absolute():
                    asset = path.parent / asset
                actual = hashlib.sha256(asset.read_bytes()).hexdigest()
                if actual != expected:
                    raise ValueError(f"convolution bytes changed: {asset}")
    scored = bool(quality.get("final_seats"))
    return {
        "result": str(path),
        "outcome": outcome,
        "before_rms_db": metrics["pre_target_weighted_rms_db"] if scored else None,
        "after_rms_db": metrics["post_target_weighted_rms_db"] if scored else None,
        "final_seats": len(quality.get("final_seats", [])),
        "observation_band_hz": quality.get("evaluated_band_hz"),
        "requested_observation_band_hz": [optimizer.get("min_freq"), optimizer.get("max_freq")],
        "convolutions_verified": convolution_count,
        "violations": report.get("violations", []),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", nargs="+", type=Path)
    args = parser.parse_args()
    for path in args.results:
        print(json.dumps(inspect(path), allow_nan=False))


if __name__ == "__main__":
    main()
