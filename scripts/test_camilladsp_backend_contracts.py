#!/usr/bin/env python3
"""Exercise required-backend launcher failures without executing DSP or Cargo."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location(
    "backend_contracts", Path(__file__).with_name("run_camilladsp_backend_contracts.py")
)
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)


class RequiredBackendTests(unittest.TestCase):
    def run_launcher(self, output="", returncode=0, missing=False, error=None):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            results = [subprocess.CompletedProcess([], 0, "CamillaDSP test\n")]
            results.append(error or subprocess.CompletedProcess([], returncode, output))
            with patch.object(launcher, "__file__", str(root / "scripts/launcher.py")), \
                 patch.object(launcher.shutil, "which", return_value=None if missing else "/backend"), \
                 patch.object(launcher.subprocess, "run", side_effect=results), \
                 contextlib.redirect_stdout(io.StringIO()):
                if missing or error or returncode or output != self.success:
                    with self.assertRaises(Exception):
                        launcher.main()
                else:
                    launcher.main()
            record = json.loads((root / "target/qa/camilladsp-backend-contracts.json").read_text())
            self.assertEqual(record["status"], "passed" if output == self.success and not (missing or error or returncode) else "failed")
            return record

    success = "".join(f"test {name} ... ok\n" for name in sorted(launcher.REQUIRED_TESTS)) + "test result: ok. 7 passed; 0 failed; 0 ignored; 100 filtered out\ntest result: ok. 1 passed; 0 failed; 0 ignored; 600 filtered out\n"

    def test_success(self):
        record = self.run_launcher(self.success)
        self.assertEqual(record["tests_passed"], 8)
        self.assertIn("roomeq-workflow", record["command"])

    def test_missing_workflow_owner(self):
        owner = "room_optimization::gd::tests::tool_contract_camilladsp_fractional_gd_matches_exported_response"
        self.run_launcher(self.success.replace(f"test {owner} ... ok\n", "test unrelated ... ok\n"))

    def test_missing_second_summary(self):
        self.run_launcher(self.success.rsplit("test result:", 1)[0])

    def test_second_suite_failed(self):
        self.run_launcher(self.success.replace("ok. 1 passed; 0 failed", "FAILED. 0 passed; 1 failed"))

    def test_second_suite_ignored(self):
        self.run_launcher(self.success.replace("1 passed; 0 failed; 0 ignored", "0 passed; 0 failed; 1 ignored"))

    def test_missing_backend(self):
        self.run_launcher(missing=True)

    def test_zero_tests(self):
        self.run_launcher("test result: ok. 0 passed; 0 failed; 0 ignored\n")

    def test_too_few_tests(self):
        self.run_launcher("test result: ok. 5 passed; 0 failed; 0 ignored\n")

    def test_ignored_tests(self):
        self.run_launcher("test result: ok. 7 passed; 0 failed; 1 ignored\n")

    def test_silent_skip(self):
        self.run_launcher(self.success + "skipping optional PCM backend\n")

    def test_nonzero_exit(self):
        self.run_launcher(self.success, returncode=1)

    def test_missing_summary(self):
        self.run_launcher("compilation succeeded\n")

    def test_count_without_required_owners(self):
        self.run_launcher("test result: ok. 7 passed; 0 failed; 0 ignored\n")

    def test_missing_physical_sub_contract(self):
        physical = "tests::realized_transfer::tool_contract_camilladsp_multisub_coherent_peak_at_all_rates"
        self.run_launcher(self.success.replace(f"test {physical} ... ok\n", "test unrelated ... ok\n"))

    def test_timeout(self):
        self.run_launcher(error=subprocess.TimeoutExpired("cargo", 180))


if __name__ == "__main__":
    unittest.main()
