"""Artifact regressions for the measured RoomEQ harness."""

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from check_roomeq_measured_result import inspect, validate_runtime_limiters


class MeasuredArtifactTests(unittest.TestCase):
    def test_runtime_limiter_evidence_requires_real_terminal_protection(self):
        params = {"threshold_db": -1.0, "release_ms": 100.0, "lookahead_ms": 5.0,
                  "soft": False, "true_peak": False, "isp_mode": False,
                  "dual_release": False, "mix": 1.0, "feed_forward": True,
                  "link_amount": 1.0, "label": "room_eq_sub_output_limiter",
                  "room_eq_stage": "post_route"}
        plugin = {"plugin_type": "limiter", "parameters": params}
        data = {"channels": {"Sub1": {"plugins": [plugin]}}, "metadata": {
            "bass_management": {"routing_graph": {"routes": [{
                "destination": "Sub1", "route_kind": "lfe_lowpass_to_sub"}]}}}}
        checks = [{"id": "runtime_limiter_physical_output:Sub1", "passed": True}]
        policy = {"subwoofer_limiter": True}
        validate_runtime_limiters(data, checks, policy)
        for bad_params in [{**params, "mix": 0.5}, {**params, "threshold_db": 0.0},
                           {**params, "room_eq_stage": "pre_route"}]:
            plugin["parameters"] = bad_params
            with self.assertRaisesRegex(ValueError, "runtime sub-output limiter"):
                validate_runtime_limiters(data, checks, policy)
        plugin["parameters"] = params
        data["channels"]["Sub1"]["plugins"].append({"plugin_type": "gain"})
        with self.assertRaises(ValueError):
            validate_runtime_limiters(data, checks, policy)
        data["channels"]["Sub1"]["plugins"] = []
        with self.assertRaisesRegex(ValueError, "evidence"):
            validate_runtime_limiters(data, checks, policy)

    def test_named_subwoofer_group_loss_is_exempt_with_or_without_alias(self):
        for group in [
            {"name": "array", "subwoofers": []},
            {"name": "array", "front": [], "rear": []},
            {"name": "array", "front": {}, "rear": {}, "separation_meters": 1.0},
        ]:
            for alias in [False, True]:
                with self.subTest(group=group, alias=alias):
                    config = {"speakers": {"array" if alias else "L": group}}
                    if alias:
                        config["system"] = {"speakers": {"L": "array"}}
                    self.data["metadata"]["effective_config"] = config
                    output = self.data["metadata"]["correction_acceptance"]["acoustic_quality"]["useful_output"][0]
                    output["unexplained_loss_rms_db"] = 10.0
                    output["bass_unexplained_loss_rms_db"] = 10.0
                    self.save()
                    self.assertEqual(inspect(self.path)["outcome"], "accepted")

    def test_main_group_cannot_claim_subwoofer_loss_exemption(self):
        for fields in [{"drivers": []}, {"measurements": []}, {"primary": {}, "support": {}}]:
            with self.subTest(fields=fields):
                # Unknown extra fields must not change untagged variant precedence.
                speaker = {"name": "main", "subwoofers": [], "front": [], "rear": [], **fields}
                self.data["metadata"]["effective_config"] = {"speakers": {"L": speaker}}
                output = self.data["metadata"]["correction_acceptance"]["acoustic_quality"]["useful_output"][0]
                output["unexplained_loss_rms_db"] = 10.0
                self.save()
                with self.assertRaisesRegex(ValueError, "useful-output loss"):
                    inspect(self.path)

    def test_subwoofer_loss_does_not_use_main_spl_budget(self):
        quality = self.data["metadata"]["correction_acceptance"]["acoustic_quality"]
        for name in ["LFE", "Sub1", "L", "R", "SL", "TFL"]:
            with self.subTest(name=name):
                quality["final_seats"][0]["logical_input"] = name
                quality["useful_output"][0].update(logical_input=name, unexplained_loss_rms_db=10.0)
                self.save()
                if name in ["LFE", "Sub1"]:
                    self.assertEqual(inspect(self.path)["outcome"], "accepted")
                else:
                    with self.assertRaisesRegex(ValueError, "useful-output loss"):
                        inspect(self.path)

    def test_correction_band_is_not_reported_as_observation_band(self):
        self.data["metadata"]["effective_config"] = {
            "optimizer": {"min_freq": 40.0, "max_freq": 200.0}
        }
        self.data["metadata"]["correction_acceptance"]["acoustic_quality"]["evaluated_band_hz"] = [80.0, 16000.0]
        self.save()
        row = inspect(self.path)
        self.assertEqual(row["requested_correction_band_hz"], [40.0, 200.0])
        self.assertEqual(row["observation_band_hz"], [80.0, 16000.0])
        self.assertNotIn("requested_observation_band_hz", row)

    def test_accepted_verdict_cannot_hide_excessive_useful_output_loss(self):
        output = self.data["metadata"]["correction_acceptance"]["acoustic_quality"]["useful_output"][0]
        for field in ["unexplained_loss_rms_db", "bass_unexplained_loss_rms_db"]:
            with self.subTest(field=field):
                output[field] = 4.0
                self.save()
                with self.assertRaisesRegex(ValueError, "useful-output loss"):
                    inspect(self.path)
                self.data["metadata"]["effective_config"] = {
                    "optimizer": {"finalization": {"max_useful_output_loss_db": 5.0}}
                }
                self.save()
                self.assertEqual(inspect(self.path)["outcome"], "accepted")
                output[field] = 0.0
                del self.data["metadata"]["effective_config"]

    def test_every_seat_needs_unique_useful_output_evidence(self):
        quality = self.data["metadata"]["correction_acceptance"]["acoustic_quality"]
        original = quality["useful_output"][0]
        for evidence in [[], [original, original], [dict(original, seat_index=1)]]:
            with self.subTest(evidence=evidence):
                quality["useful_output"] = evidence
                self.save()
                with self.assertRaisesRegex(ValueError, "useful-output evidence"):
                    inspect(self.path)

    def test_explicit_input_budget_does_not_bypass_electrical_checks(self):
        self.data["metadata"]["effective_config"] = {
            "optimizer": {"finalization": {"default_input_peak": 0.125}}
        }
        self.data["metadata"]["stage_outcomes"][0]["checks"][0]["passed"] = False
        self.save()
        with self.assertRaisesRegex(ValueError, "electrical safety"):
            inspect(self.path)

    def test_empty_electrical_evidence_is_not_success(self):
        self.data["metadata"]["stage_outcomes"][0]["checks"] = []
        self.save()
        with self.assertRaisesRegex(ValueError, "electrical safety"):
            inspect(self.path)

    def test_rejected_acoustics_cannot_pass_with_safe_electrical_output(self):
        self.data["metadata"]["correction_acceptance"].update(
            outcome="rejected", accepted=False, decision="rejected"
        )
        self.save()
        with self.assertRaisesRegex(ValueError, "not approved for playback"):
            inspect(self.path)

    def test_routed_graph_requires_terminal_alignment_not_earlier_success(self):
        self.data["metadata"]["bass_management"] = {"routing_graph": {"input_channels": ["L", "R"]}}
        self.data["metadata"]["stage_outcomes"].append({
            "stage": "final_channel_level_alignment",
            "checks": [{"passed": True}],
        })
        self.save()
        with self.assertRaisesRegex(ValueError, "terminal channel-alignment"):
            inspect(self.path)

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
                    "acoustic_quality": {
                        "final_seats": [{"improvement_db": 1.0, "partition": "training",
                                         "logical_input": "L", "seat_index": 0}],
                        "useful_output": [{"partition": "training", "logical_input": "L",
                                           "seat_index": 0, "unexplained_loss_rms_db": 0.0}],
                    },
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
