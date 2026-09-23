"""Artifact consistency controls; synthetic fixtures do not establish acoustics."""

import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from scripts.src.correction_explanation import correction_explanation_html
from scripts.src.loaders import FirParameters, RoomEqData, load_roomeq_json
from scripts.src.payload_binding import ALGORITHM, payload_digest, verify_payload_binding
from scripts.test_correction_explanation import _record


def bound_fixture():
    data = {"channels": {"L": {"plugins": [{"plugin_type": "gain", "parameters": {"gain_db": 1.0}}]}}}
    bind(data)
    return data


def bind(data):
    payload = {k: v for k, v in data.items() if k != "correction_decisions"}
    data["correction_decisions"] = {"ledger_version": "1.0.0", "decisions": [_record()],
        "payload_binding": {"algorithm": ALGORITHM, "graph_identity": "graph-final-1",
                            "sha256": payload_digest(payload, "graph-final-1")}}


class PayloadBindingTests(unittest.TestCase):
    def test_playback_banner_cannot_reuse_stale_approval(self):
        from scripts.src.report import _playback_status_html
        data = bound_fixture()
        data["metadata"] = {"correction_acceptance": {
            "outcome": "accepted", "accepted": True, "decision": "accepted"}}
        bind(data)
        valid = _playback_status_html(data["metadata"], data=data)
        self.assertIn("#2ecc71", valid)
        data["channels"]["L"]["plugins"][0]["parameters"]["gain_db"] = 12.0
        stale = _playback_status_html(data["metadata"], data=data)
        self.assertNotIn("#2ecc71", stale)
        self.assertIn("Not approved for playback", stale)
        self.assertIn("Delivered payload changed", stale)

    def test_payload_and_final_identity_mutations_are_detected(self):
        data = bound_fixture()
        self.assertTrue(verify_payload_binding(data)[0])
        self.assertIn("1 applied delivery claims", correction_explanation_html(data))
        for change in ("gain", "route", "metadata", "identity", "digest", "algorithm", "absence"):
            changed = copy.deepcopy(data)
            if change == "gain":
                changed["channels"]["L"]["plugins"][0]["parameters"]["gain_db"] = 12.0
            elif change == "route":
                changed["global_plugins"] = [{"plugin_type": "matrix", "parameters": {}}]
            elif change == "metadata":
                changed["metadata"] = {"delivered_graph_identity": "graph-final-1"}
            elif change == "identity":
                changed["correction_decisions"]["decisions"][0]["final_graph_identity"] = "stale"
            elif change == "absence":
                del changed["correction_decisions"]["payload_binding"]
            else:
                changed["correction_decisions"]["payload_binding"]["sha256" if change == "digest" else "algorithm"] = "wrong"
            with self.subTest(change=change):
                self.assertFalse(verify_payload_binding(changed)[0])
                html = correction_explanation_html(changed)
                self.assertNotIn("1 applied delivery claims", html)
                self.assertIn("unverified; not a delivery claim", html)

    def test_codec_retains_numeric_types_and_order(self):
        value = {"unicode": "é𝄞", "small": 1e-20, "large": 2**64 - 1,
                 "nested": [None, True, False, -0.0, 1.0, -7]}
        original = payload_digest(value, "graph-1")
        self.assertEqual(original, "ea5287ff38c712e5de32deb7ce8ecdc89c8d9309eb30dd35e2c6616c7a75c029")
        self.assertEqual(original, payload_digest(dict(reversed(list(value.items()))), "graph-1"))
        self.assertEqual(original, payload_digest(json.loads(json.dumps(value)), "graph-1"))
        for index, replacement in [(3, 0.0), (4, 1), (1, 1)]:
            changed = copy.deepcopy(value)
            changed["nested"][index] = replacement
            self.assertNotEqual(original, payload_digest(changed, "graph-1"))
        with self.assertRaises(ValueError):
            payload_digest({"nan": float("nan")}, "graph-1")

    def test_resource_bytes_are_rechecked_without_cached_approval(self):
        with tempfile.TemporaryDirectory(dir="/Volumes/home_tmp/tmp") as directory:
            root = Path(directory)
            resource = root / "impulse.bin"
            resource.write_bytes(b"fixture impulse bytes")
            data = bound_fixture()
            data["channels"]["L"]["plugins"] = [{"plugin_type": "convolution", "parameters": {"ir_file": resource.name}}]
            data["metadata"] = {"final_convolution_sha256": {
                resource.name: hashlib.sha256(resource.read_bytes()).hexdigest()}}
            bind(data)
            self.assertFalse(verify_payload_binding(data)[0], "unresolved resource directory is not proof")
            path = root / "result.json"
            path.write_text(json.dumps(data), encoding="utf-8")
            loaded = load_roomeq_json(path)
            self.assertTrue(verify_payload_binding(loaded)[0])
            self.assertEqual(json.loads(json.dumps(loaded)), data)
            resource.write_bytes(b"different impulse bytes")
            self.assertIn("resource bytes changed", verify_payload_binding(loaded)[1])
            self.assertNotIn("1 applied delivery claims", correction_explanation_html(loaded))
            resource.unlink()
            self.assertFalse(verify_payload_binding(loaded)[0])

    def test_all_convolution_locations_and_inventory_are_checked(self):
        for location in ("global", "channel", "driver"):
            data = bound_fixture()
            plugin = {"plugin_type": "convolution", "parameters": {"ir_file": "unbound.wav"}}
            if location == "global":
                data["global_plugins"] = [plugin]
            elif location == "channel":
                data["channels"]["L"]["plugins"] = [plugin]
            else:
                data["channels"]["L"]["drivers"] = [{"name": "way", "plugins": [plugin]}]
            bind(data)
            with self.subTest(location=location):
                self.assertIn("resource-byte binding unavailable", verify_payload_binding(data)[1])
            data["metadata"] = {"final_convolution_sha256": {"different.wav": "0" * 64}}
            bind(data)
            self.assertIn("inventory disagrees", verify_payload_binding(data)[1])

    def test_replay_cache_does_not_change_serialized_payload(self):
        data = bound_fixture()
        before = json.dumps(data)
        parameters = data["channels"]["L"]["plugins"][0]["parameters"]
        data["channels"]["L"]["plugins"][0]["parameters"] = FirParameters(parameters, 48000, [1.0])
        self.assertEqual(before, json.dumps(data))
        self.assertTrue(verify_payload_binding(data)[0])
        self.assertEqual(data["channels"]["L"]["plugins"][0]["parameters"].get("_fir_taps"), [1.0])


if __name__ == "__main__":
    unittest.main()
