"""Regression tests for complete, truthful measured-mode comparison evidence."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

SCRIPT = Path(__file__).with_name("roomeq_suite_report.py")
SPEC = importlib.util.spec_from_file_location("suite_report", SCRIPT)
suite = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(suite)


class MeasuredModeComparisonTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def result(self, mode, metric=1.0):
        return {"metadata": {
            "correction_acceptance": {"outcome": "accepted", "metrics": {
                "pre_target_weighted_rms_db": metric,
                "post_target_weighted_rms_db": 0.5,
                "improvement_db": 0.5}},
            "effective_config": {"version": "3.0.0", "speakers": {"L": {"measurement": "L.csv"}},
                "system": {"bass_management": {"lfe_playback_gain_db": 0.0}},
                "target_curve": "target.csv", "recording_config": {"playback_sample_rate": 48000, "recording_sample_rate": 48000}, "optimizer": {
                    "processing_mode": mode,
                    "finalization": {"subwoofer_limiter": False, "output_ceiling_dbfs": -1.0}}}}}

    def save(self, mode, result):
        folder = self.root / "room" / mode
        folder.mkdir(parents=True, exist_ok=True)
        (folder / f"dsp-{mode}.json").write_text(json.dumps(result))

    def compare(self):
        return suite.compare_scenario(self.root, "room", ["iir", "fir"])

    def test_equal_controls_allow_mode_design_differences(self):
        self.save("iir", self.result("low_latency"))
        fir = self.result("phase_linear")
        fir["metadata"]["effective_config"]["optimizer"]["fir"] = {"taps": 4096}
        self.save("fir", fir)
        report = self.compare()
        self.assertEqual(report["verdict"], "pass")
        self.assertTrue(report["pre_metrics_equal"])
        self.assertTrue(report["shared_playback_controls_equal"])
        self.assertTrue(report["effective_config_diffs"])
        self.assertFalse(any("explicit intentional" in n for n in report["notes"]))

    def test_every_requested_mode_needs_finite_pre_metric(self):
        self.save("iir", self.result("iir"))
        for invalid in [None, True, "1.0", -1.0, 10**400, float("nan"), float("inf"), -float("inf")]:
            with self.subTest(invalid=invalid):
                self.save("fir", self.result("fir", invalid))
                report = self.compare()
                self.assertEqual(report["verdict"], "fail")
                self.assertIsNone(report["pre_metrics_equal"])
                self.assertIsNone(report["pre_metrics"].get("fir"))

    def test_missing_mode_and_invalid_json_object_fail(self):
        self.save("iir", self.result("iir"))
        self.assertEqual(self.compare()["verdict"], "fail")
        self.save("fir", [self.result("fir")])
        self.assertEqual(self.compare()["verdict"], "fail")

    def test_missing_or_empty_effective_config_does_not_pass(self):
        self.save("iir", self.result("iir"))
        for invalid in [None, {}, [], "unknown", {"version": "3.0.0", "speakers": {}}, {"optimizer": {"finalization": {}}}]:
            with self.subTest(invalid=invalid):
                result = self.result("fir")
                result["metadata"]["effective_config"] = invalid
                self.save("fir", result)
                self.assertEqual(self.compare()["verdict"], "fail")

    def test_every_requested_mode_is_required_even_with_two_present(self):
        self.save("iir", self.result("iir"))
        self.save("fir", self.result("fir"))
        report = suite.compare_scenario(self.root, "room", ["iir", "fir", "mixed"])
        self.assertEqual(report["verdict"], "fail")
        self.assertFalse(report["mode_results_complete"])

    def test_shared_control_changes_fail_despite_equal_pre_metrics(self):
        mutations = [
            lambda c: c["optimizer"]["finalization"].update(subwoofer_limiter=True),
            lambda c: c["optimizer"]["finalization"].update(output_ceiling_dbfs=-3.0),
            lambda c: c["recording_config"].update(playback_sample_rate=44100),
            lambda c: c["system"]["bass_management"].update(lfe_playback_gain_db=10.0),
            lambda c: c.update(target_curve="other.csv"),
            lambda c: c["speakers"]["L"].update(measurement="other.csv"),
        ]
        self.save("iir", self.result("iir"))
        for mutate in mutations:
            result = self.result("fir")
            mutate(result["metadata"]["effective_config"])
            self.save("fir", result)
            report = self.compare()
            self.assertTrue(report["pre_metrics_equal"])
            self.assertEqual(report["verdict"], "fail")
            self.assertFalse(report["shared_playback_controls_equal"])
            self.assertTrue(report["shared_playback_control_diffs"])

    def test_missing_metadata_and_metrics_do_not_crash_or_pass(self):
        self.save("iir", self.result("iir"))
        for result in [{}, {"metadata": []}, {"metadata": {"correction_acceptance": []}},
                       {"metadata": {"correction_acceptance": {"metrics": []}}}]:
            with self.subTest(result=result):
                self.save("fir", result)
                self.assertEqual(self.compare()["verdict"], "fail")

    def test_duplicate_requested_modes_cannot_count_as_comparison(self):
        self.save("iir", self.result("iir"))
        report = suite.compare_scenario(self.root, "room", ["iir", "iir"])
        self.assertEqual(report["verdict"], "fail")

    def test_cli_check_refuses_policy_mismatch_and_preserves_diffs(self):
        self.save("iir", self.result("iir"))
        fir = self.result("fir")
        fir["metadata"]["effective_config"]["optimizer"]["finalization"]["subwoofer_limiter"] = True
        self.save("fir", fir)
        args = [sys.executable, str(SCRIPT), str(self.root), "--scenarios", "room", "--modes", "iir fir", "--check"]
        run = subprocess.run(args, capture_output=True, text=True, timeout=10)
        self.assertEqual(run.returncode, 2, run.stdout + run.stderr)
        report = json.loads((self.root / "comparison.json").read_text())["scenarios"]["room"]
        self.assertTrue(report["pre_metrics_equal"])
        self.assertIn("optimizer.finalization.subwoofer_limiter", [d["path"] for d in report["shared_playback_control_diffs"]])
        # Report-only execution remains available and still writes the failure.
        report_only = subprocess.run(args[:-1], capture_output=True, text=True, timeout=10)
        self.assertEqual(report_only.returncode, 0, report_only.stderr)
        self.assertEqual(json.loads((self.root / "comparison.json").read_text())["scenarios"]["room"]["verdict"], "fail")

    def test_cli_malformed_metadata_reports_error_without_crashing(self):
        self.save("iir", self.result("iir"))
        self.save("fir", {"metadata": ["invalid"]})
        run = subprocess.run([sys.executable, str(SCRIPT), str(self.root), "--scenarios", "room", "--modes", "iir fir", "--check"], capture_output=True, text=True, timeout=10)
        self.assertEqual(run.returncode, 2, run.stdout + run.stderr)
        self.assertNotIn("Traceback", run.stderr)
        summary = json.loads((self.root / "summary.json").read_text())
        self.assertEqual(summary["combinations"]["room/fir"]["status"], "error")

    def test_cli_check_accepts_complete_equal_controls(self):
        self.save("iir", self.result("iir"))
        self.save("fir", self.result("fir"))
        run = subprocess.run([sys.executable, str(SCRIPT), str(self.root), "--scenarios", "room", "--modes", "iir fir", "--check"], capture_output=True, text=True, timeout=10)
        self.assertEqual(run.returncode, 0, run.stdout + run.stderr)


if __name__ == "__main__":
    unittest.main()
