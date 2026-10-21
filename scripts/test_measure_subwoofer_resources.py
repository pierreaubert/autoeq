import copy
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from scripts import measure_subwoofer_resources as suite


class ResourceEvidenceTests(unittest.TestCase):
    def sample(self):
        return {"process": {"wall_ms": 2, "user_cpu_ms": 1, "system_cpu_ms": 0, "peak_rss_bytes": 100},
                "resources": {"wall_ms": 1, "allocations": 1, "bytes_allocated": 8,
                              "reallocations": 0, "deallocations": 0, "bytes_deallocated": 0},
                "quality": {"decision": "identity_fallback"}}

    def test_missing_quality_or_measurements_and_budget_overruns_fail(self):
        for section, key, value in [
            ("process", "peak_rss_bytes", 0), ("process", "peak_rss_bytes", 1001),
            ("process", "wall_ms", 11), ("process", "user_cpu_ms", float("nan")),
            ("resources", "allocations", 0), ("resources", "bytes_allocated", -1),
            ("resources", "wall_ms", float("inf")),
        ]:
            with self.subTest(section=section, key=key, value=value):
                sample = copy.deepcopy(self.sample())
                sample[section][key] = value
                with self.assertRaises(ValueError):
                    suite.validate_sample(sample, 10, 1000)
        sample = self.sample()
        sample["quality"] = {}
        with self.assertRaises(ValueError):
            suite.validate_sample(sample, 10, 1000)

    def test_failure_replaces_previous_success_and_retains_partial_samples(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "evidence.json"
            output.write_text('{"status":"passed"}')
            executable = Path(tmp) / "binary"
            executable.write_text("fixture")
            ok = subprocess.CompletedProcess([], 0, json.dumps(self.sample()), "")
            failed = subprocess.CompletedProcess([], 1, "", "known-solvable fixture reverted")
            with patch.object(suite.platform, "platform", return_value="test-platform"), \
                    patch.object(suite.subprocess, "run", side_effect=[ok, failed]):
                with self.assertRaisesRegex(ValueError, "fixture reverted"):
                    suite.run_suite(executable, output, 2, [suite.WORKLOADS[0]])
            report = json.loads(output.read_text())
            self.assertEqual(report["status"], "failed")
            self.assertEqual(len(report["samples"]), 1)
            self.assertEqual(report["active"]["repetition"], 1)

    def test_one_fresh_worker_per_sample_and_complete_matrix(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "report.json"
            executable = Path(tmp) / "binary"
            executable.write_text("fixture")
            ok = subprocess.CompletedProcess([], 0, json.dumps(self.sample()), "")
            with patch.object(suite.platform, "platform", return_value="test-platform"), \
                    patch.object(suite.subprocess, "run", return_value=ok) as run:
                report = suite.run_suite(executable, output, 2, suite.WORKLOADS[:2])
            self.assertEqual(report["status"], "passed")
            self.assertEqual(len(report["samples"]), 4)
            self.assertEqual(run.call_count, 4)
            self.assertTrue(all(call.args[0][2] == "--worker" for call in run.call_args_list))

    def test_missing_binary_invalidates_cached_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "report.json"
            output.write_text('{"status":"passed"}')
            with self.assertRaises(FileNotFoundError):
                suite.run_suite(Path(tmp) / "missing", output)
            self.assertEqual(json.loads(output.read_text())["status"], "failed")

    def test_build_failure_invalidates_cached_success_before_any_sample(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "report.json"
            output.write_text('{"status":"passed"}')
            with patch.object(suite.sys, "argv", ["resources", "--output", str(output)]), \
                    patch.object(suite.subprocess, "run", side_effect=subprocess.CalledProcessError(101, "cargo")):
                with self.assertRaises(subprocess.CalledProcessError):
                    suite.main()
            report = json.loads(output.read_text())
            self.assertEqual(report["status"], "failed")
            self.assertEqual(report["stage"], "build")
            self.assertEqual(report["samples"], [])


if __name__ == "__main__":
    unittest.main()
