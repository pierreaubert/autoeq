"""Saved waveform reasons are visible without implying acoustic acceptance."""

import unittest
from copy import deepcopy
import tempfile
from pathlib import Path

from scripts.src.acceptance_views import waveform_status_html


AVAILABLE_STATES = (
    "Measured room impulse response imported",
    "Calculated impulse response saved — prediction after DSP",
    "Impulse response reconstructed from the input frequency response",
)


class WaveformStatusTests(unittest.TestCase):
    def test_main_and_comparison_reports_include_saved_reasons(self):
        from scripts.src.report import create_html_report, create_comparison_html_report
        from scripts.test_figures import two_sub_overview_data
        data = two_sub_overview_data()
        checks = []
        for channel, chain in data["channels"].items():
            for view in ("pre_ir", "post_ir"):
                chain.pop(view, None)
                checks.append({"id": f"{view}:{channel}", "passed": False,
                               "diagnostic": "driver <woofer>: phase unavailable"})
        data.setdefault("metadata", {})["stage_outcomes"] = [
            {"stage": "waveform_views", "checks": checks}]
        with tempfile.TemporaryDirectory() as directory:
            main = Path(directory) / "report.html"
            comparison = Path(directory) / "comparison.html"
            create_html_report(data, main, None)
            create_comparison_html_report([("iir", data)], comparison)
            for output in (main, comparison):
                html = output.read_text()
                self.assertIn("Waveform-view availability", html)
                self.assertIn("driver &lt;woofer&gt;: phase unavailable", html)

    def fixture(self):
        return {"channels": {"L": {"pre_ir": {"amplitude": [1.0]}}},
                "metadata": {"stage_outcomes": [{"stage": "waveform_views", "checks": [
                    {"id": "pre_ir:L", "passed": True},
                    {"id": "post_ir:L", "passed": False,
                     "diagnostic": "driver <woofer>: phase unavailable"}]}]}}

    def test_reasons_are_visible_and_escaped(self):
        html = waveform_status_html(self.fixture(), "<mode>")
        self.assertIn("Impulse response reconstructed from the input frequency response", html)
        self.assertIn("Unavailable: driver &lt;woofer&gt;: phase unavailable", html)
        self.assertIn("&lt;mode&gt;", html)
        self.assertIn("not a new microphone measurement", html)
        self.assertNotIn("<woofer>", html)

    def test_passed_views_use_measured_or_calculated_states(self):
        data = self.fixture()
        data["channels"]["L"]["t60_octaves"] = {"basis": "measured_room_ir"}
        data["channels"]["L"]["post_ir"] = {"amplitude": [1.0]}
        for check in data["metadata"]["stage_outcomes"][0]["checks"]:
            check["passed"] = True
            check.pop("diagnostic", None)
        html = waveform_status_html(data)
        self.assertIn("Measured room impulse response imported", html)
        self.assertIn("Calculated impulse response saved — prediction after DSP", html)

    def test_inconsistent_status_is_not_promoted(self):
        data = self.fixture()
        data["channels"]["L"].pop("pre_ir")
        self.assertIn("recorded status and saved waveform disagree", waveform_status_html(data))
        data["metadata"]["stage_outcomes"] *= 2
        self.assertIn("conflicting waveform diagnostics", waveform_status_html(data))
        self.assertEqual(waveform_status_html({}), "")

    def test_malformed_diagnostics_do_not_crash_or_promote(self):
        for metadata in ([], "invalid", {"stage_outcomes": None},
                         {"stage_outcomes": {}}, {"stage_outcomes": [None]},
                         {"stage_outcomes": [{"stage": "waveform_views", "checks": [None]}]}):
            with self.subTest(metadata=metadata):
                data = self.fixture()
                data["metadata"] = metadata
                html = waveform_status_html(data)
                self.assertIn("Unavailable:", html)
                for state in AVAILABLE_STATES:
                    self.assertNotIn(state, html)

    def test_checks_cover_exact_channel_view_set(self):
        for fault in ("missing", "duplicate", "unknown_channel", "unknown_view", "bad_channel"):
            with self.subTest(fault=fault):
                data = self.fixture()
                checks = data["metadata"]["stage_outcomes"][0]["checks"]
                if fault == "missing":
                    checks.pop()
                elif fault == "duplicate":
                    checks.append(deepcopy(checks[0]))
                elif fault == "unknown_channel":
                    checks[0]["id"] = "pre_ir:absent"
                elif fault == "unknown_view":
                    checks[0]["id"] = "other:L"
                else:
                    data["channels"]["L"] = []
                html = waveform_status_html(data)
                self.assertIn("Unavailable:", html)
                for state in AVAILABLE_STATES:
                    self.assertNotIn(state, html)

    def test_channel_names_with_colons_are_preserved(self):
        data = self.fixture()
        data["channels"]["L:woofer"] = data["channels"].pop("L")
        for check in data["metadata"]["stage_outcomes"][0]["checks"]:
            check["id"] += ":woofer"
        html = waveform_status_html(data)
        self.assertIn("Impulse response reconstructed from the input frequency response", html)
        self.assertIn("L:woofer", html)
