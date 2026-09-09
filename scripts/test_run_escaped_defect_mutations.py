import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

# The shared mutation runner is also usable from its historical script entrypoint.
import sys
from scripts import escaped_defect_ownership
with patch.dict(sys.modules, {"escaped_defect_ownership": escaped_defect_ownership}):
    from scripts import run_escaped_defect_mutations as suite


class MutationSuiteTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.registry = self.root / "registry.json"
        self.registry.write_text(json.dumps({"defects": [
            {"id": "first", "mutant_fixture": "first"},
            {"id": "second", "mutant_fixture": "second"},
        ]}))
        self.evidence = self.root / "report.json"
        self.artifacts = [self.root / "first.json", self.root / "second.json"]
        for artifact in self.artifacts:
            artifact.write_text('{"test_fixture":true}')

    def test_all_registered_mutants_are_run_and_indexed(self):
        with patch.object(suite, "ROOT", self.root), patch.object(suite, "check", return_value=2), \
             patch.object(suite, "run_mutant", side_effect=self.artifacts) as run:
            report = suite.execute(self.registry, self.evidence)
        self.assertEqual(run.call_count, 2)
        self.assertEqual(report["status"], "passed")
        self.assertEqual([x["defect_id"] for x in report["mutants"]], ["first", "second"])
        self.assertTrue(all(len(x["evidence_sha256"]) == 64 for x in report["mutants"]))

    def test_failure_retains_partial_run_and_replaces_cached_success(self):
        self.evidence.write_text('{"status":"passed"}')
        with patch.object(suite, "ROOT", self.root), patch.object(suite, "check", return_value=2), \
             patch.object(suite, "run_mutant", side_effect=[self.artifacts[0], ValueError("survivor")]):
            with self.assertRaisesRegex(ValueError, "survivor"):
                suite.execute(self.registry, self.evidence)
        report = json.loads(self.evidence.read_text())
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["active_defect"], "second")
        self.assertEqual(len(report["mutants"]), 1)

    def test_invalid_ownership_runs_no_mutants(self):
        with patch.object(suite, "check", side_effect=ValueError("ownership")), \
             patch.object(suite, "run_mutant") as run:
            with self.assertRaisesRegex(ValueError, "ownership"):
                suite.execute(self.registry, self.evidence)
        run.assert_not_called()
        self.assertEqual(json.loads(self.evidence.read_text())["status"], "failed")

    def test_incomplete_execution_count_cannot_pass(self):
        with patch.object(suite, "ROOT", self.root), patch.object(suite, "check", return_value=3), \
             patch.object(suite, "run_mutant", side_effect=self.artifacts):
            with self.assertRaisesRegex(ValueError, "execution count"):
                suite.execute(self.registry, self.evidence)
        self.assertEqual(json.loads(self.evidence.read_text())["status"], "failed")

    def test_missing_per_mutant_evidence_cannot_pass(self):
        with patch.object(suite, "ROOT", self.root), patch.object(suite, "check", return_value=2), \
             patch.object(suite, "run_mutant", return_value=self.root / "missing.json"):
            with self.assertRaises(OSError):
                suite.execute(self.registry, self.evidence)
        report = json.loads(self.evidence.read_text())
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["mutants"], [])


if __name__ == "__main__":
    unittest.main()
