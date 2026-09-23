"""Clock provenance gates must control plots, not just their status labels."""

import copy
import math
import tempfile
import unittest
from pathlib import Path

from scripts.src.capture_clock_views import (
    capture_clock_qa_html, coherent_clock_reason, gated_lr_channel,
)
from scripts.src.report import create_comparison_html_report


def fixture():
    takes = [{
        "microphone_id": f"mic-{index}", "device_id": f"usb-{index}",
        "calibration_id": f"cal-{index}", "gain_db": 0.0,
        "calibration_orientation": "on_axis", "position_m": [index * 0.06, 0.0, 0.0],
        "position_uncertainty_mm": 0.5, "offset_samples": 123.25, "skew_ppm": -80.0,
        "residual_uncertainty_us": 12.0, "correction_applied": "resampled",
        "preserves_acoustic_delay": True, "quality_passed": True,
        "timing_reference_id": "fixed-emitter",
    } for index in range(2)]
    source = {"measurements": ["seat-a.csv", "seat-b.csv"], "provenance": {
        "capture_kind": "stationary_ir", "timing_reference_id": "fixed-emitter",
        "capture": {"geometry": "compact", "takes": takes},
    }}
    channels = {name: {"initial_curve": {"freq": [100, 200, 500], "spl": [0, 0, 0], "phase": [phase] * 3},
                       "final_curve": {"freq": [100, 200, 500], "spl": [0, 0, 0], "phase": [phase] * 3},
                       "plugins": []} for name, phase in [("L", 0), ("R", 180)]}
    return {"channels": channels, "metadata": {"effective_config": {
        "speakers": {"L": copy.deepcopy(source), "R": copy.deepcopy(source)},
    }}}


class CaptureClockViewsTests(unittest.TestCase):
    def test_high_frequency_phase_falls_back_despite_valid_clock_labels(self):
        data = fixture()
        for channel in data["channels"].values():
            for key in ("initial_curve", "final_curve"):
                channel[key]["freq"] = [100, 1000, 20000]
        curve, reason = gated_lr_channel(data, data["channels"]["L"], data["channels"]["R"])
        assert reason is not None and curve is not None
        self.assertIn("relative clock uncertainty", reason)
        self.assertNotIn("phase", curve["final_curve"])
        self.assertAlmostEqual(curve["final_curve"]["spl"][0], 10 * math.log10(2))

    def test_valid_clock_evidence_allows_actual_complex_cancellation(self):
        data = fixture()
        curve, reason = gated_lr_channel(data, data["channels"]["L"], data["channels"]["R"])
        assert curve is not None
        self.assertIsNone(reason)
        self.assertLess(curve["final_curve"]["spl"][0], -250)
        self.assertIn("phase", curve["final_curve"])

    def test_one_bad_microphone_switches_to_power_sum_and_removes_phase(self):
        for bound in (None, float("nan"), float("inf"), -1, 50, True):
            with self.subTest(bound=bound):
                data = fixture()
                source = data["metadata"]["effective_config"]["speakers"]["R"]
                source["provenance"]["capture"]["takes"][1]["residual_uncertainty_us"] = bound
                original_phase = copy.deepcopy(data["channels"]["R"]["final_curve"]["phase"])
                curve, reason = gated_lr_channel(data, data["channels"]["L"], data["channels"]["R"])
                assert curve is not None and reason is not None
                self.assertIn("mic-1", reason)
                self.assertAlmostEqual(curve["final_curve"]["spl"][0], 10 * math.log10(2))
                self.assertNotIn("phase", curve["final_curve"])
                self.assertEqual(data["channels"]["R"]["final_curve"]["phase"], original_phase)

    def test_missing_quality_reference_fit_or_geometry_never_passes(self):
        for key, value in [("quality_passed", False), ("offset_samples", None),
                           ("skew_ppm", 5001), ("preserves_acoustic_delay", False),
                           ("correction_applied", "none"), ("timing_reference_id", "other"),
                           ("position_uncertainty_mm", 2), ("calibration_id", "")]:
            data = fixture()
            source = data["metadata"]["effective_config"]["speakers"]["R"]
            source["provenance"]["capture"]["takes"][0][key] = value
            self.assertIsNotNone(coherent_clock_reason(data, ("L", "R")), key)
        self.assertIsNotNone(coherent_clock_reason({}, ("L", "R")))

    def test_incompatible_source_layout_or_missing_phase_uses_magnitude(self):
        data = fixture()
        source = data["metadata"]["effective_config"]["speakers"]["R"]
        source["provenance"]["capture"]["takes"].reverse()
        self.assertIn("layout", coherent_clock_reason(data, ("L", "R")) or "")
        data = fixture()
        del data["channels"]["R"]["final_curve"]["phase"]
        curve, reason = gated_lr_channel(data, data["channels"]["L"], data["channels"]["R"])
        assert curve is not None and reason is not None
        self.assertIn("phase", reason)
        self.assertNotIn("phase", curve["initial_curve"])
        self.assertNotIn("phase", curve["final_curve"])

    def test_report_exposes_failed_clock_and_escapes_device_labels(self):
        data = fixture()
        take = data["metadata"]["effective_config"]["speakers"]["R"]["provenance"]["capture"]["takes"][1]
        take["device_id"] = "<script>device</script>"
        take["quality_passed"] = False
        take["residual_uncertainty_us"] = None
        html = capture_clock_qa_html(data)
        self.assertIn("Capture clock QA", html)
        self.assertIn("compact", html)
        self.assertIn("unavailable", html)
        self.assertIn("&lt;script&gt;", html)
        self.assertNotIn("<script>device", html)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "comparison.html"
            create_comparison_html_report([("capture", data)], path)
            report = path.read_text()
        self.assertIn("Capture clock QA", report)
        self.assertIn("magnitude-only fallback", report)
        self.assertIn("measurement quality is pending or failed", report)
        self.assertNotIn("Logical source L+R (complex sum)", report)

    def test_malformed_metadata_is_a_visible_fallback(self):
        for metadata in (None, [], "invalid", {"effective_config": []}):
            self.assertIsNotNone(coherent_clock_reason({"metadata": metadata}, ("L", "R")))
        data = fixture()
        source = data["metadata"]["effective_config"]["speakers"]["R"]
        source["provenance"]["capture"]["takes"] = "invalid"
        self.assertIn("every microphone", coherent_clock_reason(data, ("L", "R")) or "")
        self.assertIn("Capture clock QA", capture_clock_qa_html(data))
        self.assertIn("clock evidence does not identify every microphone", capture_clock_qa_html(data))

    def test_logical_channel_mapping_finds_captured_sources(self):
        data = fixture()
        config = data["metadata"]["effective_config"]
        config["speakers"]["left-source"] = config["speakers"].pop("L")
        config["speakers"]["right-source"] = config["speakers"].pop("R")
        config["system"] = {"speakers": {"L": "left-source", "R": "right-source"}}
        self.assertIsNone(coherent_clock_reason(data, ("L", "R")))


if __name__ == "__main__":
    unittest.main()
