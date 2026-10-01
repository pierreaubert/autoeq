"""Artifact consistency controls; synthetic fixtures do not establish acoustics."""

import copy
import hashlib
import json
import re
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


def status_box_colors(html):
    """Map each playback-status box title to its border color."""
    boxes = {}
    for match in re.finditer(
            r'<section class="playback-status-box"[^>]*border:2px solid (#[0-9a-f]{6})[^>]*>'
            r"\s*<h2>(.*?)</h2>", html):
        boxes[match.group(2)] = match.group(1)
    return boxes


class PayloadBindingTests(unittest.TestCase):
    def test_playback_banner_cannot_reuse_stale_approval(self):
        from scripts.src.report import _playback_status_html
        data = bound_fixture()
        data["metadata"] = {"correction_acceptance": {
            "outcome": "accepted", "accepted": True, "decision": "accepted"}}
        bind(data)
        valid = _playback_status_html(data["metadata"], data=data)
        self.assertEqual(status_box_colors(valid), {
            "Saved DSP playback eligibility: accepted": "#2ecc71",
            "Delivered payload: verified": "#2ecc71",
            "Playback approval": "#2ecc71",
        })
        data["channels"]["L"]["plugins"][0]["parameters"]["gain_db"] = 12.0
        stale = _playback_status_html(data["metadata"], data=data)
        boxes = status_box_colors(stale)
        # The recorded acceptance stays visible (green eligibility box),
        # but the stale payload (yellow) blocks playback (red): no green
        # may read as a playback go-ahead.
        self.assertEqual(boxes["Saved DSP playback eligibility: accepted"], "#2ecc71")
        self.assertEqual(boxes["Delivered payload: not verified"], "#f1c40f")
        self.assertEqual(boxes["Playback approval"], "#e74c3c")
        self.assertIn("Not approved for playback", stale)
        self.assertNotIn("Approved for playback", stale)
        self.assertIn("Delivered payload changed", stale)
        self.assertIn("re-finalize before playback", stale)

    def test_playback_banner_reports_each_verdict_separately(self):
        from scripts.src.report import _playback_status_html
        cases = [
            # (acceptance, bound payload, expected box colors)
            ({"outcome": "rejected", "violations": ["no_safe_candidate"]}, True, {
                "Saved DSP playback eligibility: rejected": "#e74c3c",
                "Delivered payload: verified": "#2ecc71",
                "Playback approval": "#e74c3c",
            }),
            ({}, False, {
                "Saved DSP playback eligibility: unverified": "#f1c40f",
                "Delivered payload: not verified": "#f1c40f",
                "Playback approval": "#e74c3c",
            }),
            ({"outcome": "accepted"}, False, {
                "Saved DSP playback eligibility: accepted": "#e74c3c",
                "Delivered payload: not verified": "#f1c40f",
                "Playback approval": "#e74c3c",
            }),
        ]
        for acceptance, bound, expected in cases:
            with self.subTest(acceptance=acceptance, bound=bound):
                data = bound_fixture() if bound else {"channels": {}}
                data["metadata"] = {"correction_acceptance": dict(acceptance)}
                if bound:
                    bind(data)
                html = _playback_status_html(data["metadata"], data=data)
                self.assertEqual(status_box_colors(html), expected)
                self.assertIn("Not approved for playback", html)
        rejected = _playback_status_html(
            {"correction_acceptance": {"outcome": "rejected",
                                      "violations": ["no_safe_candidate"]}},
            data={"channels": {}})
        self.assertIn("Recorded violations: no_safe_candidate.", rejected)

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

    def test_emission_timestamp_does_not_change_digest(self):
        early = {"channels": {"L": 1.0}, "metadata": {"timestamp": "2026-01-01T00:00:00+00:00"}}
        late = {"channels": {"L": 1.0}, "metadata": {"timestamp": "2026-12-31T23:59:59+00:00"}}
        self.assertEqual(payload_digest(early, "graph-1"), payload_digest(late, "graph-1"))
        # The caller's payload keeps its timestamp; only the hash input is normalized.
        self.assertIn("timestamp", early["metadata"])
        data = {"channels": {"L": {"plugins": []}}, "metadata": {"timestamp": "2026-01-01T00:00:00+00:00"}}
        bind(data)
        data["metadata"]["timestamp"] = "2026-12-31T23:59:59+00:00"
        self.assertTrue(verify_payload_binding(data)[0])

    def test_resource_bytes_are_rechecked_without_cached_approval(self):
        preferred = Path("/Volumes/home_tmp/tmp")
        temp_root = preferred if preferred.is_dir() else Path(__file__).resolve().parents[1] / "target/qa/payload-binding-tmp"
        try:
            temp_root.mkdir(parents=True, exist_ok=True)
            probe = tempfile.TemporaryDirectory(dir=temp_root)
            probe.cleanup()
        except OSError:
            # Preferred roots may exist without being writable (foreign
            # mounts, sandboxes); the system temp dir keeps the
            # resource-byte contract testable anywhere.
            temp_root = Path(tempfile.gettempdir())
        with tempfile.TemporaryDirectory(dir=temp_root) as directory:
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
