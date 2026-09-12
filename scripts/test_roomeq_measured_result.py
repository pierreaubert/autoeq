"""Artifact regressions for the measured RoomEQ harness."""

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from check_roomeq_measured_result import inspect


class MeasuredArtifactTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "dsp-fir.json"
        self.wav = self.path.parent / "left.wav"
        self.wav.write_bytes(b"owned first-mode FIR bytes")
        self.data = {
            "channels": {"L": {"plugins": [{
                "plugin_type": "convolution", "parameters": {"ir_file": "left.wav"},
            }]}},
            "metadata": {
                "stage_outcomes": [{"stage": "final_graph_sampled_electrical_headroom",
                                    "checks": [{"passed": True}]}],
                "correction_acceptance": {
                    "accepted": True, "decision": "accepted", "outcome": "accepted",
                    "metrics": {"improvement_db": 1.0,
                                "pre_target_weighted_rms_db": 2.0,
                                "post_target_weighted_rms_db": 1.0},
                    "acoustic_quality": {"final_seats": [{"improvement_db": 1.0}]},
                },
                "final_convolution_sha256": {
                    "left.wav": hashlib.sha256(self.wav.read_bytes()).hexdigest(),
                },
            },
        }
        self.path.with_suffix(".manifest.json").write_text(json.dumps({
            "status": "complete", "assets_owned": [str(self.path), str(self.wav)],
        }))

    def save(self):
        self.path.write_text(json.dumps(self.data))

    def test_later_mode_cannot_overwrite_an_earlier_fir(self):
        self.save()
        self.assertEqual(inspect(self.path)["convolutions_verified"], 1)
        self.wav.write_bytes(b"later mode silently overwrote the shared filename")
        with self.assertRaisesRegex(ValueError, "convolution bytes changed"):
            inspect(self.path)

    def test_identity_cannot_retain_rejected_candidate_metrics(self):
        report = self.data["metadata"]["correction_acceptance"]
        report.update(accepted=False, decision="identity_fallback", outcome="unchanged")
        report["acoustic_quality"] = {"final_seats": [{"improvement_db": -9.8}]}
        self.save()
        with self.assertRaisesRegex(ValueError, "nonidentity seat metrics"):
            inspect(self.path)

    def test_process_success_is_not_proof_of_improvement(self):
        self.data["metadata"]["correction_acceptance"]["metrics"]["improvement_db"] = -1.0
        self.save()
        with self.assertRaisesRegex(ValueError, "no measured improvement"):
            inspect(self.path)

    def test_rejected_fallback_still_requires_electrical_safety(self):
        report = self.data["metadata"]["correction_acceptance"]
        report.update(accepted=False, decision="rejected", outcome="rejected")
        self.data["metadata"]["stage_outcomes"][0]["checks"][0]["passed"] = False
        self.save()
        with self.assertRaisesRegex(ValueError, "electrical safety"):
            inspect(self.path)


if __name__ == "__main__":
    unittest.main()
