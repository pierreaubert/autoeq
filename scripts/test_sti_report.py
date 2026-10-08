"""STI report contract and rendering integration regressions."""
import importlib.util
import json
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_renderer(directory):
    spec = importlib.util.spec_from_file_location("sti_renderer", ROOT / directory / "speech_transmission.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fixture():
    return {
        "method": "iec_60268_16_2020_indirect_ir_only_v1",
        "basis": "measured_room_ir", "sample_rate_hz": 48000,
        "duration_s": 2.0, "sti": 0.5,
        "octave_centers_hz": [125, 250, 500, 1000, 2000, 4000, 8000],
        "modulation_frequencies_hz": [0.63, 0.8, 1, 1.25, 1.6, 2, 2.5, 3.15, 4, 5, 6.3, 8, 10, 12.5],
        "modulation_transfer": [[0.5] * 7 for _ in range(14)],
        "mti": [0.5] * 7, "warnings": [],
    }


class StiReportTests(unittest.TestCase):
    def test_both_reporters_render_full_table_and_assumptions(self):
        for directory in ["ui/display-roomeq", "scripts/src"]:
            renderer = load_renderer(directory)
            report = fixture()
            report["warnings"] = ["Capture < 1.6 s & verify decay"]
            html = renderer.speech_transmission_html({"speech_transmission": report}, "L <speaker>")
            self.assertIn("Full STI (IR-only): 0.500 — Fair", html)
            self.assertIn("8000 Hz", html)
            self.assertEqual(html.count('<th scope="row">'), 15)
            self.assertIn("auditory masking", html)
            self.assertIn("L &lt;speaker&gt;", html)
            self.assertIn("Capture &lt; 1.6 s &amp; verify decay", html)

    def test_missing_malformed_or_synthesized_data_stays_unavailable(self):
        renderer = load_renderer("ui/display-roomeq")
        for field, value in [("basis", "synthesized"), ("method", "unknown"),
                             ("sample_rate_hz", 16000), ("sti", float("nan")),
                             ("mti", [0.5] * 6), ("modulation_transfer", [[0.5] * 7]),
                             ("duration_s", 0)]:
            report = fixture()
            report[field] = value
            html = renderer.speech_transmission_html({"speech_transmission": report}, "L")
            self.assertIn("STI unavailable", html)
            self.assertNotIn("<table", html)
        self.assertIn("STI unavailable", renderer.speech_transmission_html({}, "L"))
        report = fixture()
        report["modulation_transfer"][0][0] = float("inf")
        self.assertIn("STI unavailable", renderer.speech_transmission_html({"speech_transmission": report}, "L"))

    def test_legacy_full_report_contains_sti_section(self):
        from scripts.src.loaders import RoomEqData
        from scripts.src.report import create_html_report
        import tempfile
        data = RoomEqData({"version": "1", "channels": {
            "L": {"channel": "L", "plugins": [], "speech_transmission": fixture()}
        }}, ROOT)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "report.html"
            create_html_report(data, output, None)
            html = output.read_text()
        self.assertIn("Full STI (IR-only): 0.500", html)

    def test_current_full_report_contains_sti_section(self):
        payload = {"version": "1", "channels": {
            "L": {"channel": "L", "plugins": [], "speech_transmission": fixture()}
        }}
        code = """
import json, sys, tempfile
from pathlib import Path
from loaders import RoomEqData
from report import create_html_report
payload = json.load(sys.stdin)
with tempfile.TemporaryDirectory() as directory:
    output = Path(directory) / "report.html"
    create_html_report(RoomEqData(payload, Path(directory)), output, None)
    assert "Full STI (IR-only): 0.500" in output.read_text()
"""
        result = subprocess.run([sys.executable, "-c", code], input=json.dumps(payload),
                                cwd=ROOT / "ui/display-roomeq", text=True,
                                capture_output=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
