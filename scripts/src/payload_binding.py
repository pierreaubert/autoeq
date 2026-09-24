"""Independently verify RoomEQ payload digests and referenced FIR bytes.

The codec is specified in roomeq-model/src/payload_binding.rs. This validates
artifact consistency, not signatures, acoustic safety, or listener benefit.
"""

import hashlib
import math
from pathlib import Path
import struct

ALGORITHM = "sha256-json-typed-v1"


def payload_digest(payload, graph_identity):
    """Hash typed JSON without depending on either language's float formatting."""
    digest = hashlib.sha256(ALGORITHM.encode("utf-8") + b"\0")

    def encode(value):
        if value is None:
            digest.update(b"n")
        elif value is True:
            digest.update(b"t")
        elif value is False:
            digest.update(b"f")
        elif isinstance(value, int):
            if not -(2**63) <= value < 2**64:
                raise ValueError("integer outside the payload codec's domain")
            digest.update(b"i" + str(value).encode("ascii") + b";")
        elif isinstance(value, float):
            if not math.isfinite(value):
                raise ValueError("nonfinite payload number")
            digest.update(b"d" + struct.pack(">d", value))
        elif isinstance(value, str):
            raw = value.encode("utf-8")
            digest.update(b"s" + struct.pack(">Q", len(raw)) + raw)
        elif isinstance(value, list):
            digest.update(b"a" + struct.pack(">Q", len(value)))
            for item in value:
                encode(item)
        elif isinstance(value, dict):
            if any(not isinstance(key, str) for key in value):
                raise ValueError("payload keys must be strings")
            digest.update(b"o" + struct.pack(">Q", len(value)))
            for key in sorted(value):
                encode(key)
                encode(value[key])
        else:
            raise ValueError("unsupported payload value")

    encode(graph_identity)
    encode(payload)
    return digest.hexdigest()


def _convolution_references(data):
    plugins = list(data.get("global_plugins") or [])
    for channel in (data.get("channels") or {}).values():
        plugins.extend(channel.get("plugins") or [])
        for driver in channel.get("drivers") or []:
            plugins.extend(driver.get("plugins") or [])
    references = set()
    for plugin in plugins:
        if plugin.get("plugin_type") == "convolution":
            reference = (plugin.get("parameters") or {}).get("ir_file")
            if not isinstance(reference, str) or not reference.strip():
                raise ValueError("convolution resource has no file reference")
            references.add(reference)
    return references


def verify_payload_binding(data):
    """Return (verified, reason, bound graph identity); never trust a stored pass."""
    try:
        ledger = data.get("correction_decisions") or {}
        binding = ledger.get("payload_binding")
        if not isinstance(binding, dict):
            return False, "Recomputable payload binding unavailable (legacy or unbound output).", None
        if ledger.get("ledger_version") != "1.0.0" or binding.get("algorithm") != ALGORITHM:
            return False, "Unsupported ledger or payload binding version.", None
        identity = binding.get("graph_identity")
        if not isinstance(identity, str) or not identity.strip():
            return False, "Payload binding has no graph identity.", None
        # Loader-only replay caches are attributes, not serialized dict keys.
        # Measurement overlays re-injected from `<stem>_files/` are also
        # excluded: the ledger binds the slim saved bytes, and the overlay
        # index records exactly which fields came from external files.
        payload = {key: value for key, value in data.items() if key != "correction_decisions"}
        overlay = getattr(data, "measurement_index", None)
        if isinstance(overlay, dict) and overlay:
            import copy

            payload = copy.deepcopy(payload)
            for channel_name, fields in overlay.items():
                if channel_name == "deployed_source_curves":
                    deployed = payload.get("deployed_source_curves")
                    if isinstance(deployed, dict) and isinstance(fields, dict):
                        for name in fields:
                            deployed.pop(name, None)
                    continue
                if not isinstance(fields, dict):
                    continue
                channel = (payload.get("channels") or {}).get(channel_name)
                if not isinstance(channel, dict):
                    continue
                for field in fields:
                    if field in channel:
                        channel.pop(field, None)
                    elif field.endswith("_initial_curve"):
                        for driver in channel.get("drivers") or []:
                            if isinstance(driver, dict):
                                driver.pop("initial_curve", None)
        if payload_digest(payload, identity) != binding.get("sha256"):
            return False, "Delivered payload changed; recorded decisions are stale.", identity
        decisions = ledger.get("decisions") or []
        if not isinstance(decisions, list):
            return False, "Malformed decision ledger.", identity
        seen = set()
        for record in decisions:
            if not isinstance(record, dict):
                return False, "Malformed decision record.", identity
            decision_id = record.get("decision_id")
            if not isinstance(decision_id, str) or not decision_id.strip() or decision_id in seen:
                return False, "Missing or duplicate decision ID.", identity
            seen.add(decision_id)
            if record.get("stage") == "final" and record.get("final_graph_identity") != identity:
                return False, "Final decision identity does not match the delivered payload binding.", identity
        references = _convolution_references(data)
        inventory = (data.get("metadata") or {}).get("final_convolution_sha256")
        if inventory is not None and (not isinstance(inventory, dict) or set(inventory) != references):
            return False, "Convolution resource inventory disagrees with delivered graph.", identity
        if references:
            if not isinstance(inventory, dict):
                return False, "Convolution resource-byte binding unavailable.", identity
            source_directory = getattr(data, "source_directory", None)
            if source_directory is None:
                return False, "Convolution resource location unavailable; load the saved result file.", identity
            for reference in sorted(references):
                expected = inventory.get(reference)
                if not isinstance(expected, str) or len(expected) != 64:
                    return False, f"Convolution resource is unbound: {reference}.", identity
                path = Path(reference)
                if not path.is_absolute():
                    path = source_directory / path
                if not path.is_file():
                    return False, f"Convolution resource is missing or not a regular file: {reference}.", identity
                digest = hashlib.sha256()
                with path.open("rb") as handle:
                    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                        digest.update(chunk)
                if digest.hexdigest() != expected:
                    return False, f"Convolution resource bytes changed: {reference}.", identity
        return True, "Delivered payload and referenced resource bytes verified (not acoustic verification).", identity
    except (ValueError, TypeError, AttributeError, OverflowError, OSError, RecursionError) as error:
        return False, f"Payload/resource verification unavailable: {error}.", None
