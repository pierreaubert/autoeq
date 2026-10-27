"""Exercise measured-runner failure isolation without running optimization."""

import json
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


class MeasuredRunnerTests(unittest.TestCase):
    def test_display_failure_does_not_skip_audit_or_report_badroom(self):
        self.exercise_runner(False)

    def test_native_refusal_still_inspects_hidden_diagnostic_graph(self):
        self.exercise_runner(True)

    def test_successful_stages_cannot_hide_failed_comparison_evidence(self):
        self.exercise_runner(False, report_only=True)

    def exercise_runner(self, native_failure, report_only=False):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            scripts = root / "scripts"
            scripts.mkdir()
            source = Path(__file__).parent
            for name in ("test_roomeq_measured.sh", "roomeq_suite_report.py", "epa_evidence.py"):
                shutil.copy(source / name, scripts / name)
            scenario = "2.0_gen8340a_p1_badroom"
            config = root / "input" / scenario
            config.mkdir(parents=True)
            for name in ("recordings.json", "optimiser-iir.json"):
                (config / name).write_text("{}")
            binary = root / "target/release/roomeq"
            binary.parent.mkdir(parents=True)
            binary.write_text("#!/usr/bin/env python3\n"
                              "import json, os, pathlib, sys\n"
                              "if '--dry-run' in sys.argv:\n"
                              " pathlib.Path('target/release/roomeq').write_text('#!/bin/sh\\nexit 99\\n')\n"
                              " sys.exit(0)\n"
                              "p = pathlib.Path(sys.argv[sys.argv.index('--output') + 1])\n"
                              "failed = os.environ.get('FIXTURE_NATIVE_REFUSAL') == '1'\n"
                              "if failed:\n"
                              " p = p.parent / '.roomeq-attempt-fixture' / p.name\n"
                              " p.parent.mkdir()\n"
                              "p.write_text(json.dumps({'metadata': {'correction_acceptance': "
                              "{'outcome': 'rejected' if failed else 'unchanged', 'metrics': {}}}}))\n"
                              "sys.exit(3 if failed else 0)\n")
            binary.chmod(0o755)
            producer = binary.with_name("report-producer")
            producer.write_text("#!/bin/sh\nexit 0\n")
            producer.chmod(0o755)
            expected_binary_hash = hashlib.sha256(binary.read_bytes()).hexdigest()
            (root / "ui").mkdir()
            (root / "ui/display-roomeq").write_text(
                "import sys\n" + ("sys.exit(0)\n" if report_only else
                "print('fixture display failure', file=sys.stderr)\nsys.exit(7)\n"))
            (scripts / "check_roomeq_measured_result.py").write_text(
                "import pathlib, sys\nassert pathlib.Path(sys.argv[1]).is_file()\n"
                "print('{\"audit_attempted\": true}')\n")
            environment = {**os.environ, "CARGO": "true", "PYTHON": "python3",
                           "CARGO_TARGET_DIR": str(root / "target"),
                           "AUTOEQ_REPORT_PRODUCER": str(producer),
                           "SCENARIOS": scenario, "MODES": "iir", "IN": "input",
                           "OUT": "output", "FIXTURE_NATIVE_REFUSAL": str(int(native_failure))}
            run = subprocess.run(["bash", str(scripts / "test_roomeq_measured.sh")],
                                 env=environment, capture_output=True, text=True, timeout=30)
            self.assertEqual(run.returncode, 1, run.stdout + run.stderr)
            folder = root / "output" / scenario / "iir"
            self.assertTrue((folder / "audit.jsonl").is_file(), run.stdout + run.stderr)
            self.assertEqual(json.loads((folder / "audit.jsonl").read_text()),
                             {"audit_attempted": True})
            if report_only:
                self.assertIn("comparison evidence check failed", run.stderr)
                self.assertIn("CHECK FAIL", run.stdout)
            else:
                self.assertIn("fixture display failure", (folder / "display.log").read_text())
            self.assertEqual(json.loads((folder / "stages.json").read_text()),
                             {"roomeq": 3 if native_failure else 0,
                              "display": 0 if report_only else 7, "audit": 0})
            summary = json.loads((root / "output/summary.json").read_text())
            self.assertEqual(summary["scenarios"], [scenario])
            self.assertEqual(summary["counts"], {"error" if native_failure else "unchanged": 1})
            if native_failure:
                self.assertFalse((folder / "dsp-iir.json").exists())
                receipt = json.loads((folder / "diagnostic-artifact.json").read_text())
                self.assertEqual(receipt["scope"], "not_approved_for_playback")
                self.assertEqual(receipt["sha256"], hashlib.sha256(
                    (folder / receipt["path"]).read_bytes()).hexdigest())
                self.assertEqual(summary["combinations"][f"{scenario}/iir"]["diagnostic_artifact"], receipt)
            manifest = json.loads((root / "output/manifest.json").read_text())
            self.assertEqual(manifest["suite_exit"], 1)
            self.assertIsNotNone(manifest["finished_utc"])
            self.assertEqual(manifest["comparison_check"], {"requested": True, "exit": 2})
            frozen_binary = root / manifest["roomeq_bin"]
            self.assertNotEqual(frozen_binary, binary)
            self.assertEqual(manifest["roomeq_sha256"], expected_binary_hash)
            self.assertEqual(hashlib.sha256(frozen_binary.read_bytes()).hexdigest(),
                             expected_binary_hash)
            self.assertEqual(frozen_binary.stat().st_mode & 0o222, 0)
            self.assertIn("exit 99", binary.read_text())


if __name__ == "__main__":
    unittest.main()
