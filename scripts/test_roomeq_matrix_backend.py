#!/usr/bin/env python3
"""Failure challenges for durable matrix-backend evidence validation."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import run_roomeq_matrix_backend as runner


class EvidenceTests(unittest.TestCase):
    def test_help_and_unknown_arguments_do_not_start_or_replace_evidence(self):
        artifact = self.root / "target/qa/roomeq-matrix-backend.json"
        artifact.write_text("existing-evidence")
        for arguments, exit_code in [(["--help"], 0), (["--unknown"], 2)]:
            with self.subTest(arguments=arguments), \
                 patch.object(runner.subprocess, "Popen") as process, \
                 contextlib.redirect_stdout(io.StringIO()), \
                 contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as stopped:
                    runner.main(arguments)
                self.assertEqual(stopped.exception.code, exit_code)
                process.assert_not_called()
                self.assertEqual(artifact.read_text(), "existing-evidence")

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.directory = self.root / "target/qa/roomeq-parameter-bundles/row-00-test"
        self.directory.mkdir(parents=True)
        self.patch = patch.object(runner, "ROOT", self.root)
        self.patch.start()
        self.addCleanup(self.patch.stop)
        graph = {"channels": {"L": {"plugins": []}}}
        (self.directory / "graph.json").write_text(json.dumps(graph))
        (self.directory / "request.json").write_text("{}")
        (self.directory / "backend-camilladsp.yaml").write_text("fixture")
        self.row = {"row": 0, "sample_rate_hz": 48000, "requested_axes": {},
                    "unexecuted_axes": ["crossover"], "replay_bundle": {
                        "directory": str(self.directory), "request": "request.json",
                        "selected_output": "graph.json"}}
        self.proof = {"status": "passed", "run_id": "fresh", "row": 0,
                      "sample_rate_hz": 48000, "requested_axes": {},
                      "unexecuted_axes": ["crossover"], "input_channels": ["L"],
                      "output_channels": ["L"],
                      "frequencies_hz": [20 * 1000 ** (i / 48) for i in range(49)],
                      "comparisons": [{"input": "L", "output": "L", "frequency_count": 49,
                                       "max_absolute_complex_error": 0.0}],
                      "artifact_sha256": {name: runner.digest(self.directory / name)
                                          for name in ["graph.json", "request.json"]},
                      "yaml_sha256": runner.digest(self.directory / "backend-camilladsp.yaml")}

        self.proof["comparisons"][0]["complex_samples"] = [
            {"actual": [0.0, 0.0], "expected": [0.0, 0.0]} for _ in range(49)
        ]

    def verify(self):
        (self.directory / "backend-complex-transfer.json").write_text(json.dumps(self.proof))
        return runner.validate_row(self.row, "fresh")

    def test_complete_fresh_evidence(self):
        self.assertEqual(self.verify()["status"], "passed")

    def test_stale_run(self):
        self.proof["run_id"] = "old"
        with self.assertRaises(ValueError): self.verify()

    def test_running_not_passed(self):
        self.proof["status"] = "running"
        with self.assertRaises(ValueError): self.verify()

    def test_changed_graph(self):
        (self.directory / "graph.json").write_text('{"channels": {}}')
        with self.assertRaises(ValueError): self.verify()

    def test_missing_path(self):
        self.proof["comparisons"] = []
        with self.assertRaises(ValueError): self.verify()

    def test_missing_hash(self):
        del self.proof["artifact_sha256"]["request.json"]
        with self.assertRaises(ValueError): self.verify()

    def test_changed_yaml(self):
        (self.directory / "backend-camilladsp.yaml").write_text("changed")
        with self.assertRaises(ValueError): self.verify()

    def test_nonfinite_error(self):
        self.proof["comparisons"][0]["max_absolute_complex_error"] = float("nan")
        with self.assertRaises(ValueError): self.verify()

    def test_finite_out_of_tolerance_transfer_is_rejected(self):
        comparison = self.proof["comparisons"][0]
        comparison["complex_samples"] = [{"actual": [1.0, 0.0], "expected": [0.0, 0.0]} for _ in range(49)]
        comparison["max_absolute_complex_error"] = 1.0
        with self.assertRaises(ValueError): self.verify()

    def test_relative_tolerance_is_preserved(self):
        comparison = self.proof["comparisons"][0]
        comparison["complex_samples"] = [{"actual": [100.5, 0.0], "expected": [100.0, 0.0]} for _ in range(49)]
        comparison["max_absolute_complex_error"] = 0.5
        self.assertEqual(self.verify()["status"], "passed")

    def test_polarity_and_phase_faults_fail_even_with_equal_magnitude(self):
        for actual in [[-1.0, 0.0], [0.0, 1.0]]:
            with self.subTest(actual=actual):
                comparison = self.proof["comparisons"][0]
                comparison["complex_samples"] = [
                    {"actual": actual, "expected": [1.0, 0.0]} for _ in range(49)
                ]
                comparison["max_absolute_complex_error"] = runner.math.hypot(actual[0] - 1.0, actual[1])
                with self.assertRaisesRegex(ValueError, "exceeds tolerance"):
                    self.verify()

    def test_absolute_tolerance_boundary(self):
        comparison = self.proof["comparisons"][0]
        comparison["complex_samples"] = [
            {"actual": [0.0, 0.0001], "expected": [0.0, 0.0]} for _ in range(49)
        ]
        comparison["max_absolute_complex_error"] = 0.0001
        self.assertEqual(self.verify()["status"], "passed")

    def test_missing_complex_samples(self):
        del self.proof["comparisons"][0]["complex_samples"]
        with self.assertRaises(ValueError): self.verify()

    def test_error_summary_must_match_samples(self):
        self.proof["comparisons"][0]["max_absolute_complex_error"] = 0.001
        with self.assertRaises(ValueError): self.verify()

    def test_duplicate_frequency(self):
        self.proof["frequencies_hz"][1] = 20
        with self.assertRaises(ValueError): self.verify()

    def test_missing_backend_is_failure(self):
        matrix = self.root / "target/qa/roomeq-parameter-matrix.json"
        matrix.write_text(json.dumps([{"row": i} for i in range(16)]))
        with patch.object(runner.shutil, "which", return_value=None):
            with self.assertRaisesRegex(RuntimeError, "binary is required"):
                runner.main([])
        record = json.loads((matrix.parent / "roomeq-matrix-backend.json").read_text())
        self.assertEqual(record["status"], "failed")

    def test_partial_matrix_is_failure(self):
        matrix = self.root / "target/qa/roomeq-parameter-matrix.json"
        matrix.write_text(json.dumps({"status": "running", "completed_rows": []}))
        with self.assertRaisesRegex(ValueError, "completed ordered"):
            runner.main([])
        record = json.loads((matrix.parent / "roomeq-matrix-backend.json").read_text())
        self.assertEqual(record["status"], "failed")


if __name__ == "__main__":
    unittest.main()
