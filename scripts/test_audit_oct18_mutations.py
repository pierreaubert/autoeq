from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import audit_oct18_mutations as suite


class AuditMutationTests(unittest.TestCase):
    def test_mutants_cannot_replace_normal_test_executables(self):
        import json
        import subprocess
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "logs"
            file, _, name, before, _ = suite.CASES["C1"]
            def snapshot(destination):
                source = destination / file
                source.parent.mkdir(parents=True)
                source.write_text(before)
                return "snapshot"
            passing = f"test {name} ... ok\ntest result: ok. 1 passed; 0 failed; 0 ignored;"
            failing = f"test {name} ... FAILED\ntest result: FAILED. 0 passed; 1 failed; 0 ignored;"
            with patch.dict(suite.os.environ, {"CARGO_TARGET_DIR": str(Path(tmp) / "normal")}), \
                    patch.object(suite, "source_snapshot", side_effect=snapshot), \
                    patch.object(suite.subprocess, "run", side_effect=[
                        subprocess.CompletedProcess([], 0, passing, ""),
                        subprocess.CompletedProcess([], 101, failing, ""),
                    ]) as run:
                suite.run(output, ["C1"])
            expected = str((output / "cargo-target").resolve())
            for call in run.call_args_list:
                self.assertEqual(call.kwargs["env"]["CARGO_TARGET_DIR"], expected)
            self.assertEqual(json.loads((output / "report.json").read_text())["target_dir"], expected)

    def test_each_mutation_has_one_current_source_site_and_changes_it(self):
        for case, (file, _, _, before, after) in suite.CASES.items():
            with self.subTest(case=case):
                self.assertEqual((suite.ROOT / file).read_text().count(before), 1)
                self.assertNotEqual(before, after)

    def test_compile_errors_zero_tests_and_ignored_tests_cannot_count_as_caught(self):
        for log in ["error: could not compile", "test result: ok. 0 passed; 0 failed; 0 ignored;",
                    "test selected ... ignored\ntest result: ok. 0 passed; 0 failed; 1 ignored;",
                    "test wrong ... FAILED\ntest result: FAILED. 0 passed; 1 failed; 0 ignored;"]:
            with self.subTest(log=log), self.assertRaises(ValueError):
                suite.require_execution(log, "selected", False)
        suite.require_execution("test selected ... FAILED\ntest result: FAILED. 0 passed; 1 failed; 0 ignored;",
                                "selected", False)

    def test_failed_snapshot_invalidates_old_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "logs"
            output.mkdir()
            (output / "report.json").write_text('{"status":"passed"}')
            with patch.object(suite, "source_snapshot", side_effect=ValueError("snapshot failed")):
                with self.assertRaisesRegex(ValueError, "snapshot failed"):
                    suite.run(output, ["C1"])
            import json
            self.assertEqual(json.loads((output / "report.json").read_text())["status"], "failed")


if __name__ == "__main__":
    unittest.main()
