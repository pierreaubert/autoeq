"""Independent source-to-payload baseline for comparison orchestration."""

import json
import math
import os
import shlex
import subprocess
from pathlib import Path
import unittest
from unittest.mock import patch

from scripts.src import report


FIXTURE = Path(__file__).parent / "tests/fixtures/comparison_payload_python_goldens.json"


class ComparisonPayloadPythonGoldenTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
        if cls.fixture.get("schema_version") != 1:
            raise AssertionError("unsupported comparison payload golden schema")
        # This fixture records the source-checkout dev producer. Optimized
        # profiles can differ by a few floating-point bits, so select the
        # reference build explicitly and keep the exact payload assertion.
        built = subprocess.run(
            [*shlex.split(os.environ.get("CARGO", "cargo")), "build", "--offline",
             "-p", "autoeq-report-wasm", "--bin", "report-producer", "--profile", "dev",
             "--message-format=json-render-diagnostics"],
            cwd=FIXTURE.parents[3], text=True, capture_output=True, check=False,
        )
        if built.returncode:
            raise RuntimeError("Reference report producer build failed: " + built.stderr)
        producers = []
        for line in built.stdout.splitlines():
            event = json.loads(line)
            if (event.get("reason") == "compiler-artifact"
                    and event["target"]["name"] == "report-producer" and event.get("executable")):
                producers.append(Path(event["executable"]))
        if len(producers) != 1 or not producers[0].is_file():
            raise RuntimeError("Cargo did not identify the reference report producer")
        cls.reference_producer = str(producers[0])

    def capture_payload(self, case):
        captured = {}

        def writer(path, title, payload, *args, **kwargs):
            captured.update(title=title, payload=payload)

        expected_provenance = case["payload"].get("provenance")
        with (
            patch.dict(os.environ, {"AUTOEQ_REPORT_PRODUCER": self.reference_producer}),
            patch(
                "scripts.src.native_dataset_adapter._prepared_header_provenance",
                return_value=expected_provenance,
            ),
            patch("scripts.src.report.wasm_report.write_report", side_effect=writer) as write,
        ):
            report.create_comparison_html_report(
                [(name, data) for name, data in case["modes"]],
                Path("comparison-payload-golden.html"),
            )
        write.assert_called_once()
        return captured

    def test_full_payload_matches_pre_orchestration_python_golden(self):
        for case in self.fixture["cases"]:
            with self.subTest(case=case["name"]):
                actual = self.capture_payload(case)
                self.assertEqual(actual["title"], case["title"])
                self.assertEqual(actual["payload"], case["payload"])
                json.dumps(actual["payload"], ensure_ascii=False, allow_nan=False)

    def test_golden_covers_comparison_source_and_target_decisions(self):
        case = self.fixture["cases"][0]
        actual = self.capture_payload(case)
        payload = actual["payload"]
        sections = payload["sections"]
        html = "\n".join(section.get("html", "") for section in sections)
        tabs = {section.get("tab") for section in sections if section.get("tab")}

        self.assertEqual(tabs, {"C", "L", "R", "LFE", "L+R"})
        self.assertIn("finite measured phase is unavailable", html)
        self.assertIn("capture provenance is unavailable", html)
        self.assertIn("mixes runs that minimized different loss functions", html)
        self.assertFalse(any("bucket" in section for section in sections))

        by_title = {
            section["figure"]["title"]: section["figure"]
            for section in sections
            if section.get("kind") == "figure"
        }
        left = by_title["Logical source L: Mode Comparison"]
        left_target = next(series for series in left["series"] if series["name"] == "Target")
        self.assertEqual(left_target["y"], [72.0] * 5)

        lr = by_title["Logical source L+R: Mode Comparison"]
        lr_series = {series["name"]: series for series in lr["series"]}
        # The first mode has known phase and clock evidence, so it uses the
        # coherent pressure sum. The second mode is downgraded to a power sum.
        coherent_db = lr_series["iir_auto"]["y"][0]
        magnitude_db = lr_series["fir_manual"]["y"][0]
        self.assertTrue(math.isclose(coherent_db, 76.07803782087734, abs_tol=1e-12))
        self.assertTrue(math.isclose(magnitude_db, 73.1244260279434, abs_tol=1e-12))

        lfe = by_title["Logical source LFE: Mode Comparison"]
        lfe_auto = next(series for series in lfe["series"] if series["name"] == "iir_auto")
        # The mode input has two separate LFE driver measurements and a route;
        # this value locks the current acoustic sum rather than the level-relative
        # aggregate channel curve.
        self.assertTrue(math.isclose(lfe_auto["y"][0], 66.03697511944506, abs_tol=1e-12))

        modes = {name: data for name, data in case["modes"]}
        self.assertEqual([modes[name]["sample_rate"] for name in modes], [96_000, 44_100, 48_000])
        self.assertEqual(len(modes["iir_auto"]["channels"]["LFE"]["drivers"]), 2)

        eq = by_title["EQ Response Comparison: L"]
        eq_series = {series["name"]: series for series in eq["series"]}
        self.assertEqual(eq_series["iir_auto"]["x"][-1], 20_000.000000000004)
        self.assertNotEqual(eq_series["iir_auto"]["y"][-1], eq_series["fir_manual"]["y"][-1])


if __name__ == "__main__":
    unittest.main()
