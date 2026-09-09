"""Negative ownership checks, without substituting fixture names for discovery."""
import copy
import json
import runpy
import sys
from pathlib import Path
import tempfile
import subprocess
import unittest
from unittest.mock import patch

from scripts import escaped_defect_ownership as owner
from scripts.escaped_defect_mutation import require_caught, require_single_mutant, require_regression_log


class OwnershipChecks(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        fixture = self.root / "fixture"
        fixture.mkdir()
        (fixture / "check.py").write_text("# discovery fixture only; not mutation proof\n")
        self.item = {
            "id": "defect", "stage": "replay", "invariant": "gain once",
            "regression_test": {"package": "pkg", "target": "pkg", "name": "tests::gain_once"},
            "pr_recipe": "regressions", "mutant_fixture": "fixture",
        }
        (fixture / "manifest.json").write_text(json.dumps({
            "runner": "check.py", "id": self.item["id"],
            "regression_test": self.item["regression_test"],
        }))
        self.tests = {("pkg", "pkg", "tests::gain_once")}
        self.recipes = {"regressions": {}}

    def errors(self, item=None, reachable=None):
        return owner.validate([item or self.item], self.tests, self.recipes,
                              {"regressions"} if reachable is None else reachable, self.root)

    def test_exact_resolved_identity_is_accepted_as_ownership_only(self):
        self.assertEqual(self.errors(), [])

    def test_cli_runs_exact_regressions_and_retains_success(self):
        registry = self.root / "registry.json"
        registry.write_text(json.dumps({"defects": [self.item]}))
        evidence = self.root / "evidence.json"
        execution = {"test": self.item["regression_test"], "executed": 1, "passed": 1}
        with patch.dict(sys.modules, {"escaped_defect_ownership": owner}), \
             patch.object(sys, "argv", ["check", str(registry), "--execute", "--evidence", str(evidence)]), \
             patch.object(owner, "check", return_value=1), \
             patch.object(owner, "execute_regression", return_value=execution) as execute:
            runpy.run_path(str(owner.ROOT / "scripts/check_escaped_defects.py"), run_name="__main__")
        execute.assert_called_once_with(self.item["regression_test"])
        report = json.loads(evidence.read_text())
        self.assertEqual(report["status"], "passed")
        self.assertEqual(report["executions"][0]["defect_id"], "defect")
        self.assertEqual(len(report["registry_sha256"]), 64)

    def test_cli_invalidates_cached_success_when_ownership_fails(self):
        registry = self.root / "registry.json"
        registry.write_text(json.dumps({"defects": [self.item]}))
        evidence = self.root / "evidence.json"
        evidence.write_text('{"status":"passed"}')
        with patch.dict(sys.modules, {"escaped_defect_ownership": owner}), \
             patch.object(sys, "argv", ["check", str(registry), "--execute", "--evidence", str(evidence)]), \
             patch.object(owner, "check", side_effect=ValueError("missing regression")), \
             patch.object(owner, "execute_regression") as execute:
            with self.assertRaises(SystemExit) as raised:
                runpy.run_path(str(owner.ROOT / "scripts/check_escaped_defects.py"), run_name="__main__")
        self.assertEqual(raised.exception.code, 1)
        execute.assert_not_called()
        report = json.loads(evidence.read_text())
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["executions"], [])

    def test_nonexistent_renamed_and_substring_test_fail(self):
        for name in ["tests::missing", "tests::renamed", "gain_once", ""]:
            item = copy.deepcopy(self.item)
            item["regression_test"]["name"] = name
            self.assertTrue(self.errors(item), name)

    def test_wrong_package_and_target_fail(self):
        for field in ["package", "target"]:
            item = copy.deepcopy(self.item)
            item["regression_test"][field] = "wrong"
            self.assertTrue(self.errors(item))

    def test_missing_and_unreachable_recipe_fail(self):
        item = copy.deepcopy(self.item)
        item["pr_recipe"] = "nonexistent"
        self.assertTrue(self.errors(item))
        self.assertTrue(self.errors(reachable=set()))

    def test_readme_and_empty_manifest_are_not_executable_fixtures(self):
        (self.root / "fixture/manifest.json").write_text("{}")
        self.assertTrue(self.errors())

    def test_mutant_fixture_cannot_substitute_another_regression(self):
        manifest_path = self.root / "fixture/manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["regression_test"]["name"] = "tests::unrelated"
        manifest_path.write_text(json.dumps(manifest))
        self.assertTrue(any("does not match" in error for error in self.errors()))
        (self.root / "fixture/manifest.json").unlink()
        (self.root / "fixture/README.md").write_text("claimed mutation")
        self.assertTrue(self.errors())

    def test_zero_inventory_and_ignored_tests_fail(self):
        with patch.object(owner, "command_json", return_value={"rust-suites": {}}):
            with self.assertRaisesRegex(ValueError, "zero"):
                owner.discover_tests({"pkg"})
        suite = {"kind": "lib", "package-name": "pkg", "binary-name": "pkg",
                 "testcases": {"tests::gain_once": {"ignored": True, "filter-match": {"status": "matches"}}}}
        with patch.object(owner, "command_json", return_value={"rust-suites": {"pkg": suite}}):
            with self.assertRaisesRegex(ValueError, "zero"):
                owner.discover_tests({"pkg"})

    def test_recipe_reachability_uses_commands_and_dependencies(self):
        recipes = {"ci": {"dependencies": [{"recipe": "contracts"}], "body": [["just regressions"]]},
                   "contracts": {}, "regressions": {}, "unused": {}}
        self.assertEqual(owner.reachable_recipes(recipes, {"ci"}), {"ci", "contracts", "regressions"})
        workflow = self.root / "ci.yml"
        workflow.write_text("steps:\n  - name: just unused\n    run: |\n      # just unused\n      just ci\n")
        self.assertEqual(owner.workflow_roots(workflow), {"ci"})

    def test_execution_rejects_empty_filtered_ignored_and_failed_results(self):
        for code, summary in [
            (0, "test result: ok. 0 passed; 0 failed; 0 ignored;"),
            (0, "test result: ok. 0 passed; 0 failed; 1 ignored;"),
            (1, "test result: FAILED. 0 passed; 1 failed; 0 ignored;"),
            (0, "test result: ok. 2 passed; 0 failed; 0 ignored;"),
        ]:
            with patch.object(owner, "discover_tests", return_value=self.tests), patch.object(
                owner.subprocess, "run", return_value=subprocess.CompletedProcess([], code, summary, "")
            ):
                with self.assertRaisesRegex(ValueError, "exactly one"):
                    owner.execute_regression(self.item["regression_test"])

    def test_execution_preserves_exact_test_identity(self):
        with patch.object(owner, "discover_tests", return_value=self.tests), patch.object(
            owner.subprocess, "run", return_value=subprocess.CompletedProcess(
                [], 0, "test result: ok. 1 passed; 0 failed; 0 ignored;", ""
            )
        ) as run:
            evidence = owner.execute_regression(self.item["regression_test"])
            self.assertEqual(evidence["executed"], 1)
            command = run.call_args.args[0]
            self.assertIn("tests::gain_once", command)
            self.assertIn("--exact", command)

    def test_survivor_timeout_unviable_empty_and_failed_baseline_are_not_kills(self):
        success = {"total_mutants": 1, "caught": 1, "missed": 0,
                   "timeout": 0, "unviable": 0,
                   "outcomes": [{"scenario": "Baseline", "summary": "Success"},
                                {"scenario": {"Mutant": {}}, "summary": "CaughtMutant",
                                 "phase_results": [{"phase": "Build", "process_status": "Success"},
                                                   {"phase": "Test", "process_status": {"Failure": 101}}]}]}
        require_caught(success)
        for field in ["missed", "timeout", "unviable", "caught", "total_mutants"]:
            broken = copy.deepcopy(success)
            broken[field] = 1 - broken[field]
            with self.assertRaises(ValueError):
                require_caught(broken)
        success["outcomes"][0]["summary"] = "Failure"
        with self.assertRaisesRegex(ValueError, "baseline"):
            require_caught(success)

    def test_mutant_discovery_rejects_empty_and_broadened_selection(self):
        require_single_mutant([{"name": "selected"}])
        for discovery in [[], [{}, {}], {"name": "selected"}, None]:
            with self.assertRaisesRegex(ValueError, "exactly one"):
                require_single_mutant(discovery)

    def test_caught_status_must_name_the_registered_failed_test(self):
        good = "test tests::gain_once ... FAILED\ntest result: FAILED. 0 passed; 1 failed; 0 ignored;"
        require_regression_log(good, "tests::gain_once", "FAILED")
        for bad in [
            good.replace("tests::gain_once", "tests::unrelated"),
            good.replace(" ... FAILED", " ... ignored"),
            good + "\ntest tests::unrelated ... FAILED",
            "test result: FAILED. 0 passed; 1 failed; 0 ignored;",
            good.replace("1 failed", "2 failed"),
        ]:
            with self.assertRaises(ValueError):
                require_regression_log(bad, "tests::gain_once", "FAILED")

    def test_baseline_log_requires_exact_nonignored_execution(self):
        good = "test result: ok. 0 passed; 0 failed; 0 ignored;\ntest tests::gain_once ... ok\ntest result: ok. 1 passed; 0 failed; 0 ignored;"
        require_regression_log(good, "tests::gain_once", "ok")
        with self.assertRaises(ValueError):
            require_regression_log(good, "tests::gain_once", "FAILED")


if __name__ == "__main__":
    unittest.main()
