"""Contract tests for strict and partial optimizer-matrix analysis."""

from __future__ import annotations

import json
from pathlib import Path
import struct
import tempfile
import unittest

from scripts import analyze_optimizer_benchmark_matrix as analyzer
from scripts import run_optimizer_benchmark_matrix as runner


def bits(value: float) -> int:
    return struct.unpack(">Q", struct.pack(">d", value))[0]


def dump_json(path: Path, value: object, *, canonical: bool = False) -> bytes:
    if canonical:
        payload = analyzer.canonical_json_bytes(value) + b"\n"
    else:
        payload = (json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + "\n").encode()
    path.write_bytes(payload)
    return payload


class AnalyzerFixture:
    def __init__(self, root: Path, *, selected_count: int = 2,
                 outcomes: tuple[str, ...] = ("completed", "timed_out")):
        self.root = root
        self.run = root / "run"
        self.cells_dir = self.run / "cells"
        self.cells_dir.mkdir(parents=True)
        self.binary = root / "optimizer-benchmark"
        self.binary.write_bytes(b"test executable identity\n")
        self.repository = root.resolve()
        self.environment_sha = "e" * 64
        self.specs = [self._spec(index) for index in range(2)]
        inventory = {
            "schema": runner.INVENTORY_SCHEMA,
            "expected_cell_count": len(self.specs),
            "spec_inventory_sha256": analyzer.sha256_bytes(analyzer.canonical_json_bytes(self.specs)),
            "cells": self.specs,
        }
        inventory_bytes = dump_json(self.run / "cell-spec-list.stdout.json", inventory, canonical=True)
        (self.run / "cell-spec-list.stderr.log").write_bytes(b"")
        self.inventory = inventory
        self.inventory_bytes = inventory_bytes
        self.receipts: list[dict[str, object]] = []
        self.results: list[dict[str, object]] = []
        for index, spec in enumerate(self.specs[:selected_count]):
            outcome = outcomes[index]
            result = self._result(spec, outcome)
            self.results.append(result)
            self.receipts.append(self._write_cell(index, spec, result))
        selected_ids = [spec["cell_id"] for spec in self.specs[:selected_count]]
        attempted_ids = [receipt["cell_id"] for receipt in self.receipts]
        valid_ids = [receipt["cell_id"] for receipt in self.receipts if receipt["runner_status"] == "result_recorded"]
        failures = [receipt["cell_id"] for receipt in self.receipts if receipt["runner_status"] != "result_recorded"]
        missing = [cell_id for cell_id in selected_ids if cell_id not in attempted_ids]
        unresolved = [cell_id for cell_id in selected_ids if cell_id not in valid_ids]
        all_complete = selected_ids == [spec["cell_id"] for spec in self.specs] and len(valid_ids) == len(self.specs)
        outcomes_count: dict[str, int] = {}
        for result in self.results:
            outcomes_count[str(result["outcome"])] = outcomes_count.get(str(result["outcome"]), 0) + 1
        self.matrix = {
            "schema": runner.RUN_SCHEMA,
            "status": "recorded" if not failures and len(attempted_ids) == len(selected_ids) else "incomplete",
            "started_at_utc": "2026-10-03T00:00:00Z",
            "finished_at_utc": "2026-10-03T00:00:01Z",
            "repository": str(self.repository),
            "binary_path": str(self.binary.resolve()),
            "binary_sha256": analyzer.sha256_file(self.binary),
            "cell_spec_list_command": [str(self.binary.resolve()), "--list-cell-specs"],
            "cell_spec_list_exit_code": 0,
            "cell_spec_list_watchdog_timed_out": False,
            "cell_spec_list_stdout_sha256": analyzer.sha256_bytes(inventory_bytes),
            "cell_spec_list_stderr_sha256": analyzer.sha256_file(self.run / "cell-spec-list.stderr.log"),
            "cell_spec_list_elapsed_millis": 1,
            "spec_inventory_sha256": inventory["spec_inventory_sha256"],
            "expected_cell_count": len(self.specs),
            "selected_cell_count": len(selected_ids),
            "selected_cell_ids": selected_ids,
            "planned_cell_ids": list(selected_ids),
            "attempted_cell_ids": attempted_ids,
            "valid_result_cell_ids": valid_ids,
            "runner_failed_cell_ids": failures,
            "missing_cell_ids": missing,
            "unresolved_result_cell_ids": unresolved,
            "completed_cell_count": len(self.receipts),
            "runner_failure_count": len(failures),
            "optimizer_outcome_counts": outcomes_count,
            "complete_matrix": all_complete,
            "environment_sha256": self.environment_sha,
            "environment_overrides": {},
            "cells": self.receipts,
        }
        dump_json(self.run / "matrix-run.json", self.matrix)

    def _spec(self, index: int) -> dict[str, object]:
        return {
            "schema": "autoeq.optimizer_benchmark_cell/v1",
            "cell_id": f"ordinary:case-{index}:autoeq:cobra:seed1:cap4",
            "manifest_sha256": "a" * 64,
            "fixture_sha256": "b" * 64,
            "optimizer_version": "test-version",
            "case_id": f"case-{index}",
            "backend": "autoeq:cobra",
            "seed": 1,
            "purpose": "ordinary",
            "root_search_budget": 4,
            "stage_search_budget": 4,
            "cooperative_deadline_millis": 10_000,
            "process_watchdog_millis": 30_000,
            "filter_count": 1,
            "population_size": 8,
            "sample_rate_hz_bits": bits(48_000.0),
            "frequency_min_hz_bits": bits(20.0),
            "frequency_max_hz_bits": bits(20_000.0),
            "min_q_bits": bits(0.5),
            "max_q_bits": bits(6.0),
            "min_gain_db_bits": bits(-6.0),
            "max_gain_db_bits": bits(6.0),
            "multi_strategy": "weighted_sum_equal_weights",
            "local_refiner": "autoeq:cobyla",
            "bo_ehvi": False,
        }

    @staticmethod
    def _counter(budget: int, search: int, validation: int = 0,
                 *, deadline: bool = False, cancelled: bool = False) -> dict[str, object]:
        return {
            "evaluation_budget": budget,
            "evaluations_started": search,
            "evaluations_completed": search,
            "evaluations_failed": 0,
            "evaluations_refused": 0,
            "evaluations_in_flight": 0,
            "component_evaluations_started": search * 2,
            "component_evaluations_completed": search * 2,
            "validation_evaluations_started": validation,
            "validation_evaluations_completed": validation,
            "validation_evaluations_failed": 0,
            "validation_evaluations_refused": 0,
            "validation_evaluations_in_flight": 0,
            "validation_component_evaluations_started": validation * 2,
            "validation_component_evaluations_completed": validation * 2,
            "cancellation_requested": cancelled,
            "deadline_reached": deadline,
            "budget_exhausted": search >= budget,
        }

    def _result(self, spec: dict[str, object], outcome: str) -> dict[str, object]:
        completed = outcome == "completed"
        search = 4 if completed else 2
        root = self._counter(4, search, validation=1 if completed else 0,
                             deadline=outcome == "timed_out")
        stage = self._counter(4, search)
        metrics = [
            {"measurement_id": "left", "baseline_loss": 2.0, "final_loss": 1.0},
            {"measurement_id": "right", "baseline_loss": 3.0, "final_loss": 2.0},
        ] if completed else []
        heldout = [
            {"measurement_id": "heldout_left", "baseline_loss": 2.5, "final_loss": 1.5},
            {"measurement_id": "heldout_right", "baseline_loss": 3.5, "final_loss": 2.5},
        ] if completed else []
        return {
            "schema": runner.RESULT_SCHEMA,
            "matrix_expected_cell_count": 2,
            "matrix_spec_inventory_sha256": self.inventory["spec_inventory_sha256"],
            "cell_id": spec["cell_id"],
            "spec_sha256": analyzer.sha256_bytes(analyzer.canonical_json_bytes(spec)),
            "spec": spec,
            "executable_path": str(self.binary.resolve()),
            "executable_sha256": analyzer.sha256_file(self.binary),
            "runtime_os": "test-os",
            "runtime_arch": "test-arch",
            "outcome": outcome,
            "status": "controlled test pipeline",
            "refusal": "cooperative deadline reached" if outcome == "timed_out" else None,
            "optimizer_loss_after_engine_normalization": 0.75 if completed else None,
            "parameters_log10_hz_q_gain_db": [3.0, 1.0, -1.0] if completed else None,
            "active_filter_count": 1 if completed else None,
            "training_source_metrics": metrics,
            "held_out_source_metrics": heldout,
            "comparison_loss": 2.5 if completed else None,
            "comparison_measurements_expected": 2 if completed else 0,
            "comparison_measurements_available": 2 if completed else 0,
            "realized": {
                "active_filter_count": 1,
                "filter_parameters_hz_q_gain_db": [[1000.0, 1.0, -1.0]],
                "max_abs_gain_db": 1.0,
                "max_q": 1.0,
                "transfer_db_min": -1.0,
                "transfer_db_max": 0.5,
                "transfer_db_rms": 0.4,
                "maximum_bound_violation": 0.0,
            } if completed else None,
            "stage_evidence": [{
                "dispatch": {"kind": "backend_invoked"},
                "profile": {
                    "dispatch_algorithm": "autoeq:cobra",
                    "resolved_backend": "autoeq:cobra",
                    "effective_evaluation_limit": 4,
                    "stage_evaluation_budget": 4,
                    "parameter_dimension": 3,
                    "lower_bounds": [1.0, 0.5, -6.0],
                    "upper_bounds": [4.0, 6.0, 6.0],
                    "native_budget_profile": None,
                },
                "run_counters_at_dispatch_return": root,
                "stage_counters_at_dispatch_return": stage,
                "evidence": {"termination": "timed_out" if outcome == "timed_out" else "completed"},
            }],
            "root_counters": root,
            "input_source_hashes": {"fixtures/case.csv": "c" * 64},
            "source_normalization_reference_hz": 425.0,
            "source_metric_score_calls": 8 if completed else 0,
            "callback_invocations": 0,
            "engine_elapsed_millis": 10.0,
            "elapsed_millis": 12.0,
        }

    def _write_cell(self, index: int, spec: dict[str, object], result: dict[str, object]) -> dict[str, object]:
        cell_dir = self.cells_dir / f"{index + 1:04d}-test-cell-{index}"
        cell_dir.mkdir()
        spec_path = cell_dir / "cell-spec.json"
        spec_bytes = dump_json(spec_path, spec, canonical=True)
        stdout_path = cell_dir / "stdout.log"
        stderr_path = cell_dir / "stderr.log"
        stdout_path.write_bytes(b"optimizer output\n")
        stderr_path.write_bytes(b"")
        result_path = cell_dir / "cell-result.json"
        result_bytes = json.dumps(result, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode()
        result_path.write_bytes(result_bytes)
        receipt = {
            "schema": "autoeq.optimizer_benchmark_cell_process/v1",
            "cell_id": spec["cell_id"],
            "spec_sha256": analyzer.sha256_bytes(analyzer.canonical_json_bytes(spec)),
            "inventory_sha256": self.inventory["spec_inventory_sha256"],
            "binary_path": str(self.binary.resolve()),
            "binary_sha256": analyzer.sha256_file(self.binary),
            "environment_sha256": self.environment_sha,
            "command": [str(self.binary.resolve()), "--cell-spec", str(spec_path.resolve()), "--output", str((cell_dir / "cell-result.partial.json").resolve())],
            "working_directory": str(self.repository),
            "cell_directory": str(cell_dir.resolve()),
            "spec_path": str(spec_path.resolve()),
            "partial_result_path": str((cell_dir / "cell-result.partial.json").resolve()),
            "partial_result_retained": False,
            "result_path": str(result_path.resolve()),
            "result_sha256": analyzer.sha256_bytes(result_bytes),
            "stdout_path": str(stdout_path.resolve()),
            "stdout_sha256": analyzer.sha256_file(stdout_path),
            "stderr_path": str(stderr_path.resolve()),
            "stderr_sha256": analyzer.sha256_file(stderr_path),
            "runner_status": "result_recorded",
            "optimizer_outcome": result["outcome"],
            "exit_code": 0,
            "watchdog_timed_out": False,
            "started_at_utc": "2026-10-03T00:00:00Z",
            "finished_at_utc": "2026-10-03T00:00:00.010Z",
            "elapsed_millis": 10,
            "error": None,
        }
        dump_json(cell_dir / "cell-process.json", receipt)
        return receipt

    def refresh_after_result_edit(self, index: int, result: dict[str, object]) -> None:
        receipt = self.receipts[index]
        result_path = Path(str(receipt["result_path"]))
        result_bytes = json.dumps(result, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode()
        result_path.write_bytes(result_bytes)
        receipt["result_sha256"] = analyzer.sha256_bytes(result_bytes)
        dump_json(Path(str(receipt["cell_directory"])) / "cell-process.json", receipt)
        self.matrix["cells"] = self.receipts
        dump_json(self.run / "matrix-run.json", self.matrix)


class OptimizerBenchmarkAnalyzerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="autoeq-matrix-analysis-")
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_full_run_accepts_typed_timeout_but_excludes_it_from_quality(self) -> None:
        fixture = AnalyzerFixture(self.root)
        output = self.root / "analysis"
        self.assertEqual(analyzer.main(["--run", str(fixture.run), "--output", str(output), "--require-complete"]), 0)
        report = json.loads((output / "matrix-analysis.json").read_bytes())
        self.assertTrue(report["strict_gate_pass"])
        self.assertEqual(report["independently_verified_result_count"], 2)
        self.assertEqual(report["verified_outcome_counts"], {"completed": 1, "timed_out": 1})
        self.assertEqual(report["quality_eligible_completed_count"], 1)
        completed_row = next(row for row in report["groups_by_backend_case_purpose_cap"] if row["case_id"] == "case-0")
        timeout_row = next(row for row in report["groups_by_backend_case_purpose_cap"] if row["case_id"] == "case-1")
        self.assertEqual(completed_row["quality_eligible_completed"], 1)
        self.assertEqual(timeout_row["outcomes"].get("timed_out"), 1)

    def test_partial_default_is_explicit_and_strict_mode_fails(self) -> None:
        fixture = AnalyzerFixture(self.root, selected_count=1, outcomes=("completed",))
        partial_output = self.root / "partial"
        self.assertEqual(analyzer.main(["--run", str(fixture.run), "--output", str(partial_output)]), 0)
        partial = json.loads((partial_output / "matrix-analysis.json").read_bytes())
        self.assertFalse(partial["matrix_execution_complete_recomputed"])
        self.assertEqual(partial["unselected_inventory_cell_ids"], [fixture.specs[1]["cell_id"]])
        self.assertEqual(partial["missing_cell_ids_recomputed"], [])
        self.assertFalse(partial["strict_gate_pass"])
        strict_output = self.root / "strict"
        self.assertEqual(analyzer.main(["--run", str(fixture.run), "--output", str(strict_output), "--require-complete"]), 1)
        strict = json.loads((strict_output / "matrix-analysis.json").read_bytes())
        self.assertIn("incomplete_matrix", {item["code"] for item in strict["strict_gate_failures"]})

    def test_tampered_result_fails_receipt_integrity(self) -> None:
        fixture = AnalyzerFixture(self.root, selected_count=1, outcomes=("completed",))
        receipt = fixture.receipts[0]
        Path(str(receipt["result_path"])).write_bytes(b"{}\n")
        output = self.root / "tamper-analysis"
        self.assertEqual(analyzer.main(["--run", str(fixture.run), "--output", str(output), "--require-complete"]), 1)
        report = json.loads((output / "matrix-analysis.json").read_bytes())
        self.assertIn("result_integrity", {item["code"] for item in report["strict_gate_failures"]})
        self.assertEqual(report["quality_eligible_completed_count"], 0)

    def test_missing_comparison_and_duplicate_metric_id_are_quality_excluded(self) -> None:
        for mutation, code in (("missing_loss", "comparison_loss"), ("duplicate_metric", "metric_duplicate")):
            with self.subTest(mutation=mutation):
                case_root = self.root / mutation
                case_root.mkdir()
                fixture = AnalyzerFixture(case_root, selected_count=1, outcomes=("completed",))
                result = fixture.results[0]
                if mutation == "missing_loss":
                    result["comparison_loss"] = None
                else:
                    rows = result["held_out_source_metrics"]
                    rows[1]["measurement_id"] = rows[0]["measurement_id"]
                fixture.refresh_after_result_edit(0, result)
                output = case_root / "analysis"
                self.assertEqual(analyzer.main(["--run", str(fixture.run), "--output", str(output), "--require-complete"]), 1)
                report = json.loads((output / "matrix-analysis.json").read_bytes())
                self.assertEqual(report["quality_eligible_completed_count"], 0)
                self.assertEqual(report["quality_excluded_completed_count"], 1)
                self.assertIn(code, {item["code"] for item in report["strict_gate_failures"]})

    def test_out_of_bounds_filter_is_not_in_quality_distribution(self) -> None:
        fixture = AnalyzerFixture(self.root, selected_count=1, outcomes=("completed",))
        result = fixture.results[0]
        result["realized"]["filter_parameters_hz_q_gain_db"][0][0] = 22_000.0
        fixture.refresh_after_result_edit(0, result)
        output = self.root / "infeasible"
        self.assertEqual(analyzer.main(["--run", str(fixture.run), "--output", str(output), "--require-complete"]), 1)
        report = json.loads((output / "matrix-analysis.json").read_bytes())
        self.assertEqual(report["quality_eligible_completed_count"], 0)
        self.assertEqual(report["quality_excluded_completed_count"], 1)
        self.assertIn("realized_frequency", {item["code"] for item in report["strict_gate_failures"]})

    def test_counter_cap_violation_fails_even_when_result_hash_is_updated(self) -> None:
        fixture = AnalyzerFixture(self.root, selected_count=1, outcomes=("completed",))
        result = fixture.results[0]
        result["root_counters"]["evaluations_started"] = 5
        result["root_counters"]["evaluations_completed"] = 5
        stage = result["stage_evidence"][0]
        stage["run_counters_at_dispatch_return"]["evaluations_started"] = 5
        stage["run_counters_at_dispatch_return"]["evaluations_completed"] = 5
        stage["stage_counters_at_dispatch_return"]["evaluations_started"] = 5
        stage["stage_counters_at_dispatch_return"]["evaluations_completed"] = 5
        fixture.refresh_after_result_edit(0, result)
        output = self.root / "overbudget"
        self.assertEqual(analyzer.main(["--run", str(fixture.run), "--output", str(output), "--require-complete"]), 1)
        report = json.loads((output / "matrix-analysis.json").read_bytes())
        self.assertIn("counter_budget", {item["code"] for item in report["strict_gate_failures"]})

    def test_completed_cell_requires_profile_and_stage_counters(self) -> None:
        mutations = (
            ("missing_stage", lambda result: result.update(stage_evidence=[]), "stage_missing"),
            (
                "missing_profile",
                lambda result: result["stage_evidence"][0].pop("profile"),
                "profile_missing",
            ),
            (
                "missing_stage_counters",
                lambda result: result["stage_evidence"][0].pop("stage_counters_at_dispatch_return"),
                "stage_counters_missing",
            ),
        )
        for name, mutate, expected_code in mutations:
            with self.subTest(mutation=name):
                case_root = self.root / name
                case_root.mkdir()
                fixture = AnalyzerFixture(case_root, selected_count=1, outcomes=("completed",))
                result = fixture.results[0]
                mutate(result)
                fixture.refresh_after_result_edit(0, result)
                output = case_root / "analysis"
                self.assertEqual(
                    analyzer.main(["--run", str(fixture.run), "--output", str(output), "--require-complete"]),
                    1,
                )
                report = json.loads((output / "matrix-analysis.json").read_bytes())
                self.assertIn(
                    expected_code,
                    {issue["code"] for issue in report["integrity_and_validation_problems"]},
                )

    def test_malformed_root_and_stage_containers_are_reported_not_raised(self) -> None:
        cases = (
            ({"root_counters": []}, {"counter_schema"}),
            ({"stage_evidence": None}, {"stage_schema"}),
            ({"root_counters": [], "stage_evidence": None}, {"counter_schema", "stage_schema"}),
        )
        for index, (mutation, expected_codes) in enumerate(cases):
            with self.subTest(fields=list(mutation)):
                case_root = self.root / str(index)
                case_root.mkdir()
                fixture = AnalyzerFixture(case_root, selected_count=1, outcomes=("completed",))
                result = fixture.results[0]
                result.update(mutation)
                fixture.refresh_after_result_edit(0, result)
                output = case_root / "malformed-counters"
                self.assertEqual(
                    analyzer.main(["--run", str(fixture.run), "--output", str(output), "--require-complete"]),
                    1,
                )
                report = json.loads((output / "matrix-analysis.json").read_bytes())
                codes = {issue["code"] for issue in report["integrity_and_validation_problems"]}
                self.assertTrue(expected_codes <= codes)
                self.assertFalse(report["strict_gate_pass"])
                self.assertEqual(report["quality_eligible_completed_count"], 0)

    def test_malformed_observer_count_retains_failed_report(self) -> None:
        class ObserverFixture(AnalyzerFixture):
            def _spec(self, index: int) -> dict[str, object]:
                spec = super()._spec(index)
                spec["purpose"] = "observer_stop"
                return spec

        for index, count in enumerate((None, "1", [], True)):
            with self.subTest(count=count):
                case_root = self.root / str(index)
                case_root.mkdir()
                fixture = ObserverFixture(case_root, outcomes=("observer_stopped", "observer_stopped"))
                for row, result in enumerate(fixture.results):
                    result["root_counters"]["cancellation_requested"] = True
                    result["callback_invocations"] = 1
                    fixture.refresh_after_result_edit(row, result)
                valid_output = case_root / "valid"
                self.assertEqual(analyzer.main([
                    "--run", str(fixture.run), "--output", str(valid_output), "--require-complete"
                ]), 0)
                result = fixture.results[0]
                result["callback_invocations"] = count
                fixture.refresh_after_result_edit(0, result)
                output = case_root / "invalid"
                self.assertEqual(analyzer.main([
                    "--run", str(fixture.run), "--output", str(output), "--require-complete"
                ]), 1)
                report = json.loads((output / "matrix-analysis.json").read_bytes())
                codes = {issue["code"] for issue in report["integrity_and_validation_problems"]}
                self.assertIn("result_metrics", codes)
                self.assertIn("stop_evidence", codes)
                self.assertFalse(report["strict_gate_pass"])

    def test_duplicate_inventory_ids_and_nonfinite_json_are_refused(self) -> None:
        duplicate_root = self.root / "duplicate"
        duplicate_root.mkdir()
        fixture = AnalyzerFixture(duplicate_root)
        inventory = fixture.inventory
        inventory["cells"][1]["cell_id"] = inventory["cells"][0]["cell_id"]
        inventory["spec_inventory_sha256"] = analyzer.sha256_bytes(analyzer.canonical_json_bytes(inventory["cells"]))
        inventory_bytes = dump_json(fixture.run / "cell-spec-list.stdout.json", inventory, canonical=True)
        fixture.matrix["cell_spec_list_stdout_sha256"] = analyzer.sha256_bytes(inventory_bytes)
        fixture.matrix["spec_inventory_sha256"] = inventory["spec_inventory_sha256"]
        dump_json(fixture.run / "matrix-run.json", fixture.matrix)
        duplicate_out = duplicate_root / "duplicate-output"
        self.assertEqual(analyzer.main(["--run", str(fixture.run), "--output", str(duplicate_out), "--require-complete"]), 1)
        self.assertTrue((duplicate_out / "analysis-error.json").is_file())
        with self.assertRaises(analyzer.AnalysisInputError):
            analyzer.parse_json(b'{"x":1,"x":2}', "duplicate test")
        with self.assertRaises(analyzer.AnalysisInputError):
            analyzer.parse_json(b'{"x":1e999}', "non-finite test")

    def test_existing_output_is_never_overwritten(self) -> None:
        fixture = AnalyzerFixture(self.root)
        output = self.root / "existing"
        output.mkdir()
        marker = output / "keep.txt"
        marker.write_text("original")
        self.assertEqual(analyzer.main(["--run", str(fixture.run), "--output", str(output), "--require-complete"]), 1)
        self.assertEqual(marker.read_text(), "original")
        self.assertFalse((output / "analysis-error.json").exists())


if __name__ == "__main__":
    unittest.main()
