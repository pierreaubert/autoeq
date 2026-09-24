"""Slim-bundle loader regressions: external curves re-inject transparently."""

import json
import tempfile
import unittest
from pathlib import Path

from scripts.src.loaders import load_roomeq_json
from scripts.src.payload_binding import ALGORITHM, payload_digest, verify_payload_binding


def _bind_slim(payload):
    body = {k: v for k, v in payload.items() if k != "correction_decisions"}
    payload["correction_decisions"] = {
        "ledger_version": "1.0.0",
        "decisions": [],
        "payload_binding": {
            "algorithm": ALGORITHM,
            "graph_identity": "graph-bundle-1",
            "sha256": payload_digest(body, "graph-bundle-1"),
        },
    }


class BundleLoaderTests(unittest.TestCase):
    def _write_slim(self, root: Path):
        slim = {
            "version": "1",
            "channels": {"L": {"channel": "L", "plugins": []}},
            "deployed_source_curves": {},
        }
        _bind_slim(slim)
        (root / "dsp.json").write_text(json.dumps(slim))
        assets = root / "dsp_files"
        assets.mkdir()
        (assets / "L__initial.csv").write_text("freq,spl,phase\n100,80.0,0.0\n1000,81.0,10.0\n")
        (assets / "L__final.csv").write_text("freq,spl\n100,79.0\n1000,80.0\n")
        (assets / "L__pre_ir.csv").write_text("time_ms,amplitude\n0.0,1.0\n0.1,0.5\n")
        (assets / "deployed__L.csv").write_text("freq,spl\n100,79.5\n1000,80.5\n")
        (assets / "measurements_index.json").write_text(json.dumps({
            "channels": {"L": {
                "initial_curve": "L__initial.csv",
                "final_curve": "L__final.csv",
                "pre_ir": "L__pre_ir.csv",
            }},
            "deployed_source_curves": {"L": "deployed__L.csv"},
        }))
        return root / "dsp.json"

    def test_external_curves_reinject_with_phase_and_ir(self):
        with tempfile.TemporaryDirectory() as directory:
            data = load_roomeq_json(self._write_slim(Path(directory)))
            self.assertEqual(data["channels"]["L"]["initial_curve"]["spl"], [80.0, 81.0])
            self.assertEqual(data["channels"]["L"]["initial_curve"]["phase"], [0.0, 10.0])
            self.assertEqual(data["channels"]["L"]["final_curve"]["spl"], [79.0, 80.0])
            self.assertNotIn("phase", data["channels"]["L"]["final_curve"])
            self.assertEqual(data["channels"]["L"]["pre_ir"]["amplitude"], [1.0, 0.5])
            self.assertEqual(data["deployed_source_curves"]["L"]["spl"], [79.5, 80.5])

    def test_slim_binding_verifies_through_the_overlay(self):
        with tempfile.TemporaryDirectory() as directory:
            data = load_roomeq_json(self._write_slim(Path(directory)))
            valid, reason, _ = verify_payload_binding(data)
            self.assertTrue(valid, reason)

    def test_legacy_embedded_output_loads_without_overlay(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "legacy.json"
            path.write_text(json.dumps({
                "channels": {"L": {
                    "channel": "L",
                    "plugins": [],
                    "initial_curve": {"freq": [100.0], "spl": [80.0]},
                }},
            }))
            data = load_roomeq_json(path)
            self.assertEqual(
                data["channels"]["L"]["initial_curve"], {"freq": [100.0], "spl": [80.0]})
            self.assertEqual(data.measurement_index, {})


if __name__ == "__main__":
    unittest.main()
