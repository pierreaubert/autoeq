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

from src.payload_binding import ALGORITHM, payload_digest


def finite_tree(value, location="result"):
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"nonfinite value at {location}")
    if isinstance(value, dict):
        for key, child in value.items():
            finite_tree(child, f"{location}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            finite_tree(child, f"{location}[{index}]")


def is_subwoofer_group(speaker):
    """Match the untagged SpeakerConfig variants in their deserialization order."""
    if not isinstance(speaker, dict) or "name" not in speaker:
        return False
    # These speaker variants precede subwoofer groups in the Rust enum.
    if ({"primary", "support"} <= speaker.keys()
            or "drivers" in speaker or "measurements" in speaker):
        return False
    return "subwoofers" in speaker or {"front", "rear"} <= speaker.keys()


def iter_delay_observations(data):
    """Yield (location, delay_ms) for every serialized physical delay."""
    owners = [("global_plugins", data.get("global_plugins") or [])]
    for name, chain in (data.get("channels") or {}).items():
        owners.append((f"channels.{name}", chain.get("plugins") or []))
        for driver in chain.get("drivers") or []:
            owners.append((f"channels.{name}.drivers.{driver.get('name', '?')}",
                           driver.get("plugins") or []))
    for location, plugins in owners:
        for index, plugin in enumerate(plugins):
            if plugin.get("plugin_type") != "delay":
                continue
            parameters = plugin.get("parameters") or {}
            if "delay_ms" in parameters:
                yield f"{location}.plugins[{index}].delay_ms", parameters["delay_ms"]
    graph = ((data.get("metadata") or {}).get("bass_management") or {}).get("routing_graph") or {}
    for index, route in enumerate(graph.get("routes", [])):
        if "delay_ms" in route:
            yield f"routing_graph.routes[{index}].delay_ms", route["delay_ms"]


def validate_causal_delays(data):
    """Reject negative physical delays: a real-time block cannot pre-play.

    Relative advances are a valid optimization variable, but the final
    graph must carry the compiled causal realization (nonnegative delays
    plus a serialized common offset). Any negative delay_ms is an
    unresolved causal delay, not a host detail.
    """
    for location, delay_ms in iter_delay_observations(data):
        if not isinstance(delay_ms, (int, float)) or isinstance(delay_ms, bool):
            raise ValueError(f"non-numeric delay at {location}")
        if delay_ms < 0.0:
            raise ValueError(f"unresolved causal delay {delay_ms} ms at {location}")


def validate_payload_binding(data):
    """Reject decisions whose graph no longer matches their binding.

    Any post-finalization change to a gain, FIR, delay, or route breaks
    the digest; the earlier approval stays invalid until the graph is
    re-finalized. Graphs without a binding predate the ledger and are
    audited on their other evidence only. Resource-byte verification
    stays in this script's convolution section below; this check covers
    the digest and final-identity agreement only.
    """
    ledger = data.get("correction_decisions") or {}
    binding = ledger.get("payload_binding")
    if not isinstance(binding, dict):
        return
    if ledger.get("ledger_version") != "1.0.0" or binding.get("algorithm") != ALGORITHM:
        raise ValueError("payload binding has an unsupported ledger or algorithm version")
    identity = binding.get("graph_identity")
    if not isinstance(identity, str) or not identity.strip():
        raise ValueError("payload binding has no graph identity")
    payload = {key: value for key, value in data.items() if key != "correction_decisions"}
    if payload_digest(payload, identity) != binding.get("sha256"):
        raise ValueError("delivered payload changed; recorded decisions are stale")
    for record in ledger.get("decisions") or []:
        if record.get("stage") == "final" and record.get("final_graph_identity") != identity:
            raise ValueError(
                f"final decision '{record.get('decision_id')}' does not match "
                "the delivered payload binding"
            )


def validate_runtime_limiters(data, checks, policy):
    """Require terminal native sub protection, not just a passing verdict flag."""
    graph = ((data.get("metadata") or {}).get("bass_management") or {}).get("routing_graph") or {}
    subs = {route["destination"] for route in graph.get("routes", [])
            if route.get("route_kind") in {"redirected_bass_lowpass_to_sub", "lfe_lowpass_to_sub"}}
    expected = {
        "threshold_db": min(policy.get("output_ceiling_dbfs", 0.0), -1.0),
        "release_ms": 100.0, "lookahead_ms": 5.0, "soft": False,
        "true_peak": False, "isp_mode": False, "dual_release": False,
        "mix": 1.0, "feed_forward": True, "link_amount": 1.0,
        "label": "room_eq_sub_output_limiter", "room_eq_stage": "post_route",
    }
    found = set()
    owners = [("global", data.get("global_plugins") or [], False)]
    for name, chain in data.get("channels", {}).items():
        drivers = chain.get("drivers") or []
        owners.append((name, chain.get("plugins") or [], not drivers))
        owners.extend((driver["name"], driver.get("plugins") or [], True) for driver in drivers)
    for name, plugins, physical in owners:
        for index, plugin in enumerate(plugins):
            if plugin.get("plugin_type") != "limiter":
                continue
            if (not policy.get("subwoofer_limiter") or not physical or name not in subs
                    or index != len(plugins) - 1 or plugin.get("parameters") != expected
                    or name in found):
                raise ValueError("invalid terminal runtime sub-output limiter")
            found.add(name)
    proved = {check["id"].split(":", 1)[1] for check in checks
              if check.get("id", "").startswith("runtime_limiter_physical_output:")}
    if found != proved or (policy.get("subwoofer_limiter") and found != subs):
        raise ValueError("runtime limiter evidence does not match protected sub outputs")


def inspect(path):
    path = Path(path)
    data = json.loads(path.read_text())
    finite_tree(data)
    validate_causal_delays(data)
    validate_payload_binding(data)
    metadata = data["metadata"]
    electrical = [stage for stage in metadata.get("stage_outcomes", [])
                  if stage["stage"] == "final_graph_sampled_electrical_headroom"]
    if len(electrical) != 1:
        raise ValueError("missing or ambiguous final electrical assessment")
    optimizer = (metadata.get("effective_config") or {}).get("optimizer") or {}
    electrical_checks = electrical[0].get("checks", [])
    if not electrical_checks or any(not check["passed"] for check in electrical_checks):
        raise ValueError("delivered graph lacks passing electrical safety for its configured input budgets")
    validate_runtime_limiters(data, electrical_checks, optimizer.get("finalization") or {})
    report = metadata["correction_acceptance"]
    outcome = report["outcome"]
    if outcome not in {"accepted", "unchanged", "rejected", "insufficient_evidence"}:
        raise ValueError(f"invalid outcome: {outcome}")
    if outcome in {"rejected", "insufficient_evidence"}:
        raise ValueError(f"graph is not approved for playback: {outcome}")
    if outcome == "accepted" and (
        not report["accepted"] or report["decision"] != "accepted"
    ):
        raise ValueError("accepted outcome contradicts detailed decision")
    metrics = report["metrics"]
    quality = report.get("acoustic_quality") or {}
    if outcome == "accepted" and not quality.get("final_seats"):
        raise ValueError("accepted correction has no physical-seat replay evidence")
    if outcome == "accepted":
        def seat_key(seat):
            return (seat["partition"], seat["logical_input"], seat["seat_index"])

        seats = [seat_key(seat) for seat in quality["final_seats"]]
        output_evidence = quality.get("useful_output") or []
        output_seats = [seat_key(output) for output in output_evidence]
        if (len(set(seats)) != len(seats)
                or len(set(output_seats)) != len(output_seats)
                or set(output_seats) != set(seats)):
            raise ValueError("accepted correction lacks unique useful-output evidence for every seat")
        budget = (optimizer.get("finalization") or {}).get("max_useful_output_loss_db", 3.0)
        config = metadata.get("effective_config") or {}
        system = config.get("system") or {}
        sub_outputs = (system.get("subwoofers") or {}).get("outputs") or []
        sub_names = {str(value).lower() for item in sub_outputs
                     for value in [item.get("id", ""), item.get("speaker", "")] if value}
        for output in output_evidence:
            name = output["logical_input"]
            key = (system.get("speakers") or {}).get(name, name)
            subwoofer = (name.lower() == "lfe" or name.lower().startswith("sub")
                         or name.lower() in sub_names or key.lower() in sub_names
                         or is_subwoofer_group((config.get("speakers") or {}).get(key)))
            loss = max(output["unexplained_loss_rms_db"],
                       output.get("bass_unexplained_loss_rms_db") or 0.0)
            if not subwoofer and loss > budget:
                raise ValueError(f"useful-output loss {loss:.3f} dB exceeds {budget:.3f} dB budget")
    if (metadata.get("bass_management") or {}).get("routing_graph"):
        alignment = [stage for stage in metadata.get("stage_outcomes", [])
                     if stage["stage"] == "final_delivered_channel_alignment"]
        if len(alignment) != 1 or not alignment[0].get("checks"):
            raise ValueError("routed graph has no terminal channel-alignment evidence")
        if any(not check["passed"] for check in alignment[0]["checks"]):
            raise ValueError("delivered routed graph has incorrect channel levels")
    if outcome == "accepted" and metrics["improvement_db"] <= 1e-6:
        raise ValueError("accepted correction has no measured improvement")
    if report["decision"] == "identity_fallback":
        quality = report.get("acoustic_quality") or {}
        for seat in quality.get("final_seats", []):
            if abs(seat["improvement_db"]) > 1e-6:
                raise ValueError("identity fallback retains nonidentity seat metrics")

    manifest_path = path.parent / f"{path.stem}_files" / "manifest.json"
    if not manifest_path.is_file():
        # Legacy layout: manifest next to the native graph.
        manifest_path = path.with_suffix(".manifest.json")
    manifest = json.loads(manifest_path.read_text())
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
                    candidates = [
                        path.parent / asset,
                        path.parent / f"{path.stem}_files" / asset,
                    ]
                    asset = next(
                        (candidate for candidate in candidates if candidate.is_file()),
                        candidates[0],
                    )
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
        "requested_correction_band_hz": [optimizer.get("min_freq"), optimizer.get("max_freq")],
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
