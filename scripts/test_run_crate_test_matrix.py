"""Contract tests for fail-closed AutoEQ QA matrix accounting."""

import pathlib
import json
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from scripts import run_crate_test_matrix as matrix

QA_TESTS = pathlib.Path(__file__).resolve().parents[1] / "crates/autoeq-qa/tests"
sys.path.insert(0, str(QA_TESTS))
from qa_support.metrics import maximum_error_metrics, require_finite_numbers
from py05_heldout_contract import (
    _validate_fixture as validate_py05_fixture,
    _validate_output_grid as validate_py05_output_grid,
)
from h02_scorer_sign import _h02_error_metrics


class PythonQaResultTests(unittest.TestCase):
    CASE = {"id": "PY01", "tolerance_kind": "rel", "tolerance_value": 1e-4}

    @staticmethod
    def record(**updates):
        result = {
            "QA_RESULT": True,
            "case": "PY01",
            "pass": True,
            "max_rel_error": 2e-5,
            "max_abs_error": 0.0,
            "tolerance": 1e-4,
            "tolerance_kind": "rel",
            "provenance": "wolfram-engine-15.0.0",
        }
        result.update(updates)
        return result

    def test_one_matching_record_passes(self):
        stdout = json.dumps(self.record()) + "\n"
        status, result = matrix.parse_python_qa_result(self.CASE, stdout, 0)
        self.assertEqual(status, "passed")
        self.assertEqual(result["case"], "PY01")

    def test_duplicate_records_fail(self):
        record = json.dumps(self.record())
        status, result = matrix.parse_python_qa_result(self.CASE, record + "\n" + record, 0)
        self.assertEqual(status, "failed")
        self.assertIsNone(result)

    def test_wrong_case_identity_fails(self):
        stdout = json.dumps(self.record(case="PY02")) + "\n"
        status, result = matrix.parse_python_qa_result(self.CASE, stdout, 0)
        self.assertEqual(status, "failed")
        self.assertEqual(result["case"], "PY02")

    def test_wrong_tolerance_value_or_error_fails(self):
        wrong_tolerance = json.dumps(self.record(tolerance=1e-3))
        status, _ = matrix.parse_python_qa_result(self.CASE, wrong_tolerance, 0)
        self.assertEqual(status, "failed")

        exceeded_error = json.dumps(self.record(max_rel_error=2e-4))
        status, _ = matrix.parse_python_qa_result(self.CASE, exceeded_error, 0)
        self.assertEqual(status, "failed")

    def test_both_error_metrics_are_required_and_finite(self):
        for field in ("max_abs_error", "max_rel_error"):
            missing = self.record()
            del missing[field]
            status, _ = matrix.parse_python_qa_result(
                self.CASE, json.dumps(missing), 0
            )
            self.assertEqual(status, "failed")

            for invalid in (float("nan"), float("inf"), True):
                status, _ = matrix.parse_python_qa_result(
                    self.CASE, json.dumps(self.record(**{field: invalid})), 0
                )
                self.assertEqual(status, "failed")

    def test_duplicate_json_key_is_rejected(self):
        stdout = (
            '{"QA_RESULT":true,"case":"PY01","case":"PY01",'
            '"pass":true,"max_rel_error":0,"max_abs_error":0,'
            '"tolerance":0.0001,"tolerance_kind":"rel",'
            '"provenance":"wolfram-engine-15.0.0"}\n'
        )
        status, _ = matrix.parse_python_qa_result(self.CASE, stdout, 0)
        self.assertEqual(status, "failed")


class PythonQaErrorMetricTests(unittest.TestCase):
    def test_zero_reference_uses_documented_unit_denominator(self):
        self.assertEqual(maximum_error_metrics([(2.5e-6, 0.0)]), (2.5e-6, 2.5e-6))

    def test_nonfinite_comparison_or_fixture_values_are_rejected(self):
        for pair in ((float("nan"), 0.0), (0.0, float("inf"))):
            with self.subTest(pair=pair), self.assertRaises(ValueError):
                maximum_error_metrics([pair])
        with self.assertRaises(ValueError):
            require_finite_numbers([80.0, float("nan")], "fixture")

    def test_py05_rejects_nonfinite_input_arrays(self):
        fixture = {
            "grid_hz": [100.0],
            "base_spl_db": [float("nan")],
            "base_phase_deg": [0.0],
            "heldout_spl_db": [81.0],
            "heldout_phase_deg": [1.0],
        }
        with self.assertRaisesRegex(ValueError, "not finite"):
            validate_py05_fixture(fixture)

    def test_py05_rejects_displaced_frequency_rows(self):
        self.assertEqual(validate_py05_output_grid([100.0], [100.0]), [(100.0, 100.0)])
        with self.assertRaisesRegex(ValueError, "frequency grid differs"):
            validate_py05_output_grid([100.001], [100.0])

    def test_h02_metrics_include_coefficients_and_complex_response(self):
        max_abs, max_rel = _h02_error_metrics(
            [1.0001, 1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0, 1.0],
            [(1.01 + 0j, 1.0 + 0j)] * 4
            + [(1.0 + 0j, 1.0 + 0j)] * 4,
        )
        self.assertAlmostEqual(max_abs, 0.01)
        self.assertAlmostEqual(max_rel, 0.01)

    def test_h02_preserves_zero_complex_reference_refusal(self):
        with self.assertRaisesRegex(ValueError, "zero reference and nonzero error"):
            _h02_error_metrics(
                [1.0] * 5, [1.0] * 5, [(1e-12 + 0j, 0j)]
            )


class RustQaTargetTests(unittest.TestCase):
    CASE = {
        "id": "R01",
        "test": "wolfram_r01",
        "tolerance_kind": "abs",
        "tolerance_value": 1e-4,
    }

    @staticmethod
    def record(**updates):
        result = {
            "case": "R01",
            "pass": True,
            "max_rel_error": 0.0,
            "max_abs_error": 2e-5,
            "tolerance": 1e-4,
            "tolerance_kind": "abs",
            "provenance": "checked-in-golden",
            "_cargo_target": "wolfram_r01",
        }
        result.update(updates)
        return result

    def test_nocapture_result_prefix_is_parsed_after_libtest_text(self):
        output = (
            "Running tests/wolfram_r01.rs (target/debug/deps/wolfram_r01-abc)\n"
            "test comparison ... QA_RESULT: "
            + json.dumps(self.record())
        )
        records, errors = matrix.parse_rust_qa_records(output)
        self.assertEqual(errors, [])
        validated, errors = matrix.validate_rust_qa_records(
            [self.CASE], records, errors
        )
        self.assertEqual(errors, [])
        self.assertEqual(validated["R01"]["status"], "passed")
        self.assertEqual(validated["R01"]["record_count"], 1)

    def test_duplicate_json_key_is_rejected(self):
        output = (
            "Running tests/wolfram_r01.rs (target/debug/deps/wolfram_r01-abc)\n"
            'QA_RESULT: {"case":"R01","case":"R01"}\n'
        )
        records, errors = matrix.parse_rust_qa_records(output)
        self.assertEqual(records, [])
        self.assertEqual(len(errors), 1)

    def test_missing_duplicate_and_wrong_value_records_fail(self):
        validated, _ = matrix.validate_rust_qa_records([self.CASE], [], [])
        self.assertEqual(validated["R01"]["status"], "failed")
        self.assertEqual(validated["R01"]["record_count"], 0)

        record = self.record()
        validated, _ = matrix.validate_rust_qa_records(
            [self.CASE], [record, record], []
        )
        self.assertEqual(validated["R01"]["status"], "failed")
        self.assertEqual(validated["R01"]["record_count"], 2)

        validated, _ = matrix.validate_rust_qa_records(
            [self.CASE], [self.record(tolerance=1e-3)], []
        )
        self.assertEqual(validated["R01"]["status"], "failed")

    def test_observed_error_over_declared_tolerance_fails(self):
        self.assertFalse(
            matrix.rust_qa_record_matches(
                self.CASE, self.record(max_abs_error=2e-4)
            )
        )
        self.assertFalse(
            matrix.rust_qa_record_matches(
                self.CASE, self.record(_cargo_target="negative_controls")
            )
        )
        self.assertFalse(
            matrix.rust_qa_record_matches(
                self.CASE, self.record(provenance="untrusted"), "checked-in-golden"
            )
        )
        self.assertTrue(
            matrix.rust_qa_record_matches(
                self.CASE, self.record(provenance="live-engine"), "live-engine"
            )
        )

    def test_missing_target_summary_fails_coverage_gate(self):
        targets = ["manifest_case", "auxiliary_case"]
        results = {"manifest_case": {"passed": 1, "failed": 0, "ignored": 0, "summary_count": 1}}
        reports = matrix.rust_target_reports(targets, results)
        self.assertTrue(reports[0]["executed"])
        self.assertEqual(reports[1]["status"], "not_run")
        self.assertFalse(matrix.rust_target_coverage_satisfied(reports, len(targets)))

    def test_ignored_only_target_is_not_executed_coverage(self):
        output = (
            "Running tests/negative_controls.rs (target/debug/deps/negative_controls)\n"
            "running 0 tests\n\n"
            "test result: ok. 0 passed; 0 failed; 2 ignored; 0 measured; 0 filtered out; finished in 0.00s\n"
        )
        results = matrix.parse_rust_case_results(output)
        reports = matrix.rust_target_reports(["negative_controls"], results)
        self.assertTrue(reports[0]["reported"])
        self.assertFalse(reports[0]["executed"])
        self.assertEqual(reports[0]["status"], "not_run")
        self.assertFalse(matrix.rust_target_coverage_satisfied(reports, 1))

    def test_numerical_case_requires_result_even_when_target_summary_passes(self):
        self.assertEqual(matrix.rust_case_status("passed", "failed", 0), "failed")
        self.assertEqual(matrix.rust_case_status("passed", "passed", 1), "passed")
        self.assertEqual(matrix.rust_case_status("not_run", "failed", 0), "not_run")

    def test_zero_test_target_is_not_executed_coverage(self):
        output = (
            "Running tests/empty.rs (target/debug/deps/empty)\n"
            "running 0 tests\n\n"
            "test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s\n"
        )
        result = matrix.parse_rust_case_results(output)
        report = matrix.rust_target_reports(["empty"], result)[0]
        self.assertFalse(report["executed"])
        self.assertEqual(report["status"], "not_run")


class CargoMatrixTests(unittest.TestCase):
    def test_unknown_feature_in_matrix_command_is_rejected(self):
        policy = {"focused_tests": {"demo": "cargo test -p demo --lib --features invented"}}
        packages = {
            "demo": {
                "features": {"valid": []},
                "targets": [{"kind": ["lib"]}],
            }
        }
        errors = matrix.validate_package_matrix(policy, packages)
        self.assertTrue(any("unknown Cargo features" in error for error in errors), errors)

    def test_duplicate_policy_keys_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = pathlib.Path(temporary_directory) / "policy.json"
            path.write_text(
                '{"focused_tests": {}, "focused_tests": {"core": "cargo test"}}',
                encoding="utf-8",
            )

            with self.assertRaisesRegex(
                ValueError, "duplicate JSON object key: focused_tests"
            ):
                matrix.load_policy(path)


class ProvenanceTests(unittest.TestCase):
    def test_full_metadata_uses_locked_offline_resolution(self):
        response = mock.Mock(
            returncode=0,
            stdout='{"packages": [], "workspace_members": []}',
            stderr="",
        )
        with mock.patch.object(matrix.subprocess, "run", return_value=response) as run:
            self.assertEqual(matrix.full_cargo_metadata()["packages"], [])

        command = run.call_args.args[0]
        self.assertEqual(
            command[1:5], ["metadata", "--format-version", "1", "--locked"]
        )
        self.assertIn("--offline", command)
        self.assertNotIn("--no-deps", command)

    def test_resolved_local_packages_include_transitive_path_sources_only(self):
        metadata = {
            "workspace_members": ["path+file:///repo#autoeq@1.0.0"],
            "packages": [
                {
                    "id": "path+file:///repo#autoeq@1.0.0",
                    "name": "autoeq",
                    "version": "1.0.0",
                    "source": None,
                },
                {
                    "id": "path+file:///external/gpui#gpui-profiler@0.2.0",
                    "name": "gpui-profiler",
                    "version": "0.2.0",
                    "source": None,
                },
                {
                    "id": "registry+https://example.invalid#serde@1.0.0",
                    "name": "serde",
                    "version": "1.0.0",
                    "source": "registry+https://example.invalid",
                },
            ],
        }

        packages = matrix.resolved_external_local_packages(metadata)

        self.assertEqual([package["name"] for package in packages], ["gpui-profiler"])

    def test_local_dependency_snapshot_binds_manifest_and_repository_source(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = pathlib.Path(temporary_directory)
            package_dir = root / "gpui-profiler"
            package_dir.mkdir()
            manifest = package_dir / "Cargo.toml"
            manifest.write_text(
                '[package]\nname = "gpui-profiler"\nversion = "0.2.0"\n',
                encoding="utf-8",
            )
            source = package_dir / "src.rs"
            source.write_text("pub fn record() {}\n", encoding="utf-8")
            metadata = {
                "workspace_members": [],
                "packages": [
                    {
                        "id": "path+file:///external/gpui#gpui-profiler@0.2.0",
                        "name": "gpui-profiler",
                        "version": "0.2.0",
                        "source": None,
                        "manifest_path": str(manifest),
                    }
                ],
            }
            previous_root = matrix.REPO_ROOT
            matrix.REPO_ROOT = root / "workspace"
            matrix.REPO_ROOT.mkdir()
            try:
                repositories = matrix.local_path_dependency_provenance(metadata)
            finally:
                matrix.REPO_ROOT = previous_root

        self.assertEqual(len(repositories), 1)
        self.assertTrue(repositories[0]["available"])
        self.assertEqual(repositories[0]["packages"][0]["version"], "0.2.0")
        self.assertEqual(
            repositories[0]["packages"][0]["manifest_path"], "Cargo.toml"
        )
        self.assertEqual(repositories[0]["working_tree"]["file_count"], 2)

    def test_missing_resolved_local_path_dependency_fails_closed(self):
        metadata = {
            "workspace_members": [],
            "packages": [
                {
                    "id": "path+file:///external/missing#missing@0.1.0",
                    "name": "missing",
                    "version": "0.1.0",
                    "source": None,
                    "manifest_path": "/external/missing/Cargo.toml",
                }
            ],
        }

        with self.assertRaisesRegex(
            RuntimeError, "resolved local Cargo package is unavailable"
        ):
            matrix.local_path_dependency_provenance(metadata)

    def test_source_digest_tracks_working_files_but_skips_target_outputs(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = pathlib.Path(temporary_directory)
            subprocess.run(["git", "init", "-q"], cwd=root, check=True)
            (root / ".gitignore").write_text("target/\n", encoding="utf-8")
            source = root / "source.rs"
            source.write_text("fn first() {}\n", encoding="utf-8")
            subprocess.run(["git", "add", ".gitignore", "source.rs"], cwd=root, check=True)

            baseline = matrix.working_tree_snapshot(root)
            (root / "target").mkdir()
            (root / "target" / "generated.json").write_text("{}", encoding="utf-8")
            with_ignored_output = matrix.working_tree_snapshot(root)
            source.write_text("fn changed() {}\n", encoding="utf-8")
            changed_source = matrix.working_tree_snapshot(root)

        self.assertEqual(baseline["sha256"], with_ignored_output["sha256"])
        self.assertNotEqual(baseline["sha256"], changed_source["sha256"])

    def test_inventory_hashes_bind_manifest_oracle_golden_and_test_file(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = pathlib.Path(temporary_directory)
            manifest = root / "manifest.toml"
            script = root / "oracle.wls"
            golden = root / "golden.json"
            test_file = root / "case.rs"
            for path, text in (
                (manifest, "case = []\n"),
                (script, "Print[1]\n"),
                (golden, "{}\n"),
                (test_file, "#[test] fn case() {}\n"),
            ):
                path.write_text(text, encoding="utf-8")
            inventory = {
                "manifest": manifest.name,
                "manifest_sha256": matrix.sha256_file(manifest),
                "cases": [
                    {
                        "id": "C01",
                        "script": script.name,
                        "script_sha256": matrix.sha256_file(script),
                        "golden": golden.name,
                        "golden_sha256": matrix.sha256_file(golden),
                        "test_path": test_file.name,
                        "test_sha256": matrix.sha256_file(test_file),
                    }
                ],
            }
            previous_root = matrix.REPO_ROOT
            matrix.REPO_ROOT = root
            try:
                self.assertEqual(matrix.verify_inventory_input_hashes(inventory), [])
                test_file.write_text("#[test] fn altered() {}\n", encoding="utf-8")
                errors = matrix.verify_inventory_input_hashes(inventory)
            finally:
                matrix.REPO_ROOT = previous_root

        self.assertIn("QA case C01 input changed after inventory: case.rs", errors)

    def test_environment_requirement_mismatch_fails_provenance(self):
        state = {
            "git_head": "base",
            "working_tree": {"sha256": "same"},
            "local_path_repositories": [],
        }

        result = matrix.provenance_report(
            state, state, {"python": {"requirements_match": False}}
        )

        self.assertFalse(result["python_requirements_match"])
        self.assertFalse(result["passed"])

    def test_source_change_during_run_fails_provenance(self):
        before = {
            "git_head": "base",
            "working_tree": {"sha256": "before"},
            "local_path_repositories": [],
        }
        after = {
            "git_head": "base",
            "working_tree": {"sha256": "after"},
            "local_path_repositories": [],
        }
        environment = {"python": {"requirements_match": True}}

        result = matrix.provenance_report(before, after, environment)

        self.assertFalse(result["unchanged_during_run"])
        self.assertFalse(result["passed"])

    def test_unavailable_local_path_dependency_fails_provenance(self):
        state = {
            "git_head": "base",
            "working_tree": {"sha256": "same"},
            "local_path_repositories": [
                {"path_from_repository": "../math-audio", "available": False}
            ],
        }

        result = matrix.provenance_report(
            state, state, {"python": {"requirements_match": True}}
        )

        self.assertFalse(result["local_path_repositories_available"])
        self.assertFalse(result["passed"])


if __name__ == "__main__":
    unittest.main()
