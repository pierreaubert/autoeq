"""Failure-path contract tests for the bounded optimizer matrix runner."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from scripts import run_optimizer_benchmark_matrix as matrix


class OptimizerBenchmarkMatrixTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="autoeq-matrix-runner-")
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def inventory(self, count: int = matrix.EXPECTED_CELLS) -> dict[str, object]:
        cells = [
            {
                "schema": "autoeq.optimizer_benchmark_cell/v1",
                "cell_id": f"cell-{index:03d}",
                "process_watchdog_millis": 1_000,
            }
            for index in range(count)
        ]
        return {
            "schema": matrix.INVENTORY_SCHEMA,
            "expected_cell_count": count,
            "spec_inventory_sha256": matrix.sha256_bytes(
                matrix.canonical_json_bytes(cells)
            ),
            "cells": cells,
        }

    def binary(self, mode: str = "valid", watchdog_millis: int = 1_000) -> Path:
        inventory = self.inventory()
        for spec in inventory["cells"]:
            spec["process_watchdog_millis"] = watchdog_millis
        inventory["spec_inventory_sha256"] = matrix.sha256_bytes(
            matrix.canonical_json_bytes(inventory["cells"])
        )
        inventory_literal = repr(json.dumps(inventory, separators=(",", ":")))
        source = f'''#!/usr/bin/env python3
import hashlib
import json
import os
from pathlib import Path
import sys
import time

MODE = {mode!r}
INVENTORY_TEXT = {inventory_literal}

if sys.argv[1:] == ["--list-cell-specs"]:
    sys.stdout.write(INVENTORY_TEXT)
    raise SystemExit(0)

spec_path = Path(sys.argv[sys.argv.index("--cell-spec") + 1])
result_path = Path(sys.argv[sys.argv.index("--output") + 1])
spec = json.loads(spec_path.read_bytes())
if MODE == "hang":
    Path(os.environ["MATRIX_TEST_PID_FILE"]).write_text(str(os.getpid()))
    time.sleep(30)
    raise SystemExit(0)
if MODE == "exit-failure":
    sys.stderr.write("deliberate child failure\\n")
    raise SystemExit(7)
if MODE == "invalid-json":
    result_path.write_bytes(b"{{invalid")
    raise SystemExit(0)

spec_bytes = json.dumps(spec, ensure_ascii=False, separators=(",", ":")).encode()
binary_path = Path(__file__).resolve()
result = {{
    "schema": {matrix.RESULT_SCHEMA!r},
    "matrix_expected_cell_count": {matrix.EXPECTED_CELLS},
    "matrix_spec_inventory_sha256": {inventory["spec_inventory_sha256"]!r},
    "cell_id": spec["cell_id"],
    "spec_sha256": hashlib.sha256(spec_bytes).hexdigest(),
    "spec": spec,
    "executable_path": str(binary_path),
    "executable_sha256": hashlib.sha256(binary_path.read_bytes()).hexdigest(),
    "outcome": "completed",
}}
if MODE == "identity-mismatch":
    result["spec_sha256"] = "0" * 64
result_path.write_text(json.dumps(result, separators=(",", ":")))
'''
        path = self.root / "optimizer-benchmark"
        path.write_text(source)
        path.chmod(0o755)
        return path

    def run_args(
        self,
        binary: Path,
        output_name: str,
        cell_ids: list[str] | None = None,
    ) -> argparse.Namespace:
        return argparse.Namespace(
            binary=binary,
            output=self.root / output_name,
            cell_id=cell_ids or [],
        )

    @staticmethod
    def run_record(output: Path) -> dict[str, object]:
        return json.loads((output / "matrix-run.json").read_bytes())

    def test_watchdog_kills_and_reaps_child_and_records_unresolved_cell(self):
        if os.name != "posix":
            self.skipTest("process-group assertion requires POSIX")
        binary = self.binary("hang", watchdog_millis=250)
        pid_path = self.root / "child.pid"
        with mock.patch.dict(os.environ, {"MATRIX_TEST_PID_FILE": str(pid_path)}):
            exit_code = matrix.run(
                self.run_args(binary, "hang-run", ["cell-000"])
            )
        self.assertEqual(exit_code, 1)
        child_pid = int(pid_path.read_text())
        with self.assertRaises(ProcessLookupError):
            os.kill(child_pid, 0)

        output = self.root / "hang-run"
        receipt = self.run_record(output)
        self.assertEqual(receipt["status"], "incomplete")
        self.assertEqual(receipt["missing_cell_ids"], [])
        self.assertEqual(receipt["unresolved_result_cell_ids"], ["cell-000"])
        cell = receipt["cells"][0]
        self.assertEqual(cell["runner_status"], "failed")
        process = json.loads(
            (Path(cell["cell_directory"]) / "cell-process.json").read_bytes()
        )
        self.assertTrue(process["watchdog_timed_out"])
        self.assertLess(process["exit_code"], 0)

    @unittest.skipUnless(os.name == "posix", "real SIGINT requires POSIX")
    def test_sigint_reaps_active_child_and_persists_partial_run(self):
        binary = self.binary("hang", watchdog_millis=30_000)
        pid_path = self.root / "interrupt-child.pid"
        output = self.root / "interrupt-run"
        process = subprocess.Popen(
            [sys.executable, str(Path(matrix.__file__).resolve()),
             "--binary", str(binary), "--output", str(output),
             "--cell-id", "cell-000", "--cell-id", "cell-001"],
            env={**os.environ, "MATRIX_TEST_PID_FILE": str(pid_path)},
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
        child_pid = None
        try:
            deadline = time.monotonic() + 10
            while time.monotonic() < deadline:
                if pid_path.exists():
                    value = pid_path.read_text().strip()
                    if value:
                        child_pid = int(value)
                        break
                if process.poll() is not None:
                    break
                time.sleep(0.01)
            self.assertIsNotNone(child_pid, "benchmark child must be running before SIGINT")
            os.kill(child_pid, 0)
            process.send_signal(signal.SIGINT)
            stdout, stderr = process.communicate(timeout=10)
            self.assertEqual(process.returncode, 130, (stdout, stderr))
            with self.assertRaises(ProcessLookupError):
                os.kill(child_pid, 0)
            receipt = self.run_record(output)
            self.assertEqual(receipt["status"], "interrupted")
            self.assertFalse(receipt["complete_matrix"])
            self.assertEqual(receipt["attempted_cell_ids"], ["cell-000"])
            self.assertEqual(receipt["missing_cell_ids"], ["cell-001"])
            self.assertEqual(receipt["unresolved_result_cell_ids"], ["cell-000", "cell-001"])
            self.assertEqual(receipt["cells"][0]["runner_status"], "interrupted")
            cell_receipt = json.loads((
                Path(receipt["cells"][0]["cell_directory"]) / "cell-process.json"
            ).read_bytes())
            self.assertFalse(cell_receipt["watchdog_timed_out"])
            self.assertIsNone(cell_receipt["result_sha256"])
        finally:
            if process.poll() is None:
                process.kill()
                process.communicate(timeout=10)
            if child_pid is not None:
                try:
                    os.killpg(child_pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass

    def test_nonzero_exit_and_invalid_json_keep_failure_artifacts(self):
        for mode, expected_code in (("exit-failure", 7), ("invalid-json", 0)):
            with self.subTest(mode=mode):
                binary = self.binary(mode)
                exit_code = matrix.run(
                    self.run_args(binary, f"{mode}-run", ["cell-000"])
                )
                self.assertEqual(exit_code, 1)
                receipt = self.run_record(self.root / f"{mode}-run")
                self.assertEqual(receipt["status"], "incomplete")
                self.assertEqual(receipt["valid_result_cell_ids"], [])
                self.assertEqual(receipt["unresolved_result_cell_ids"], ["cell-000"])
                cell = receipt["cells"][0]
                self.assertEqual(cell["runner_status"], "failed")
                cell_dir = Path(cell["cell_directory"])
                process = json.loads((cell_dir / "cell-process.json").read_bytes())
                self.assertEqual(process["exit_code"], expected_code)
                self.assertTrue((cell_dir / "cell-spec.json").is_file())
                self.assertTrue((cell_dir / "stdout.log").is_file())
                self.assertTrue((cell_dir / "stderr.log").is_file())
                if mode == "invalid-json":
                    self.assertEqual(
                        (cell_dir / "cell-result.partial.json").read_bytes(),
                        b"{invalid",
                    )
                    self.assertFalse((cell_dir / "cell-result.json").exists())

    def test_result_identity_mismatch_is_refused_and_partial_json_is_kept(self):
        binary = self.binary("identity-mismatch")
        self.assertEqual(
            matrix.run(self.run_args(binary, "identity-run", ["cell-000"])), 1
        )
        receipt = self.run_record(self.root / "identity-run")
        cell = receipt["cells"][0]
        self.assertEqual(cell["runner_status"], "failed")
        cell_dir = Path(cell["cell_directory"])
        process = json.loads((cell_dir / "cell-process.json").read_bytes())
        self.assertIn("identity or outcome", process["error"])
        self.assertTrue((cell_dir / "cell-result.partial.json").is_file())
        self.assertFalse((cell_dir / "cell-result.json").exists())

    def test_valid_subset_binds_executable_and_full_spec_inventory(self):
        binary = self.binary()
        output = self.root / "valid-run"
        self.assertEqual(
            matrix.run(self.run_args(binary, "valid-run", ["cell-000"])), 0
        )

        receipt = self.run_record(output)
        self.assertEqual(receipt["status"], "recorded")
        self.assertFalse(receipt["complete_matrix"])
        inventory_bytes = (output / "cell-spec-list.stdout.json").read_bytes()
        inventory, specs = matrix.validate_inventory(inventory_bytes)
        self.assertEqual(len(specs), matrix.EXPECTED_CELLS)
        self.assertEqual(
            receipt["cell_spec_list_stdout_sha256"],
            matrix.sha256_bytes(inventory_bytes),
        )
        self.assertEqual(
            receipt["spec_inventory_sha256"], inventory["spec_inventory_sha256"]
        )

        cell = receipt["cells"][0]
        cell_dir = Path(cell["cell_directory"])
        result = json.loads((cell_dir / "cell-result.json").read_bytes())
        process = json.loads((cell_dir / "cell-process.json").read_bytes())
        self.assertEqual(result["spec"], specs[0])
        self.assertEqual(result["executable_sha256"], matrix.sha256_file(binary))
        self.assertEqual(result["executable_sha256"], receipt["binary_sha256"])
        self.assertEqual(
            process["result_sha256"],
            matrix.sha256_file(cell_dir / "cell-result.json"),
        )

    def test_interrupted_run_keeps_planned_attempted_and_missing_ids(self):
        binary = self.binary()
        selected = ["cell-000", "cell-001", "cell-002"]
        first_receipt = {
            "cell_id": "cell-000",
            "runner_status": "result_recorded",
            "optimizer_outcome": "completed",
            "cell_directory": "test-cell-000",
        }
        with mock.patch.object(
            matrix,
            "run_one_cell",
            side_effect=[first_receipt, KeyboardInterrupt],
        ):
            exit_code = matrix.run(self.run_args(binary, "partial-run", selected))

        self.assertEqual(exit_code, 130)
        receipt = self.run_record(self.root / "partial-run")
        self.assertEqual(receipt["status"], "interrupted")
        self.assertEqual(receipt["planned_cell_ids"], selected)
        self.assertEqual(receipt["attempted_cell_ids"], ["cell-000"])
        self.assertEqual(receipt["missing_cell_ids"], ["cell-001", "cell-002"])
        self.assertEqual(
            receipt["unresolved_result_cell_ids"], ["cell-001", "cell-002"]
        )
        self.assertFalse(receipt["complete_matrix"])

    def test_existing_output_directory_is_not_overwritten(self):
        binary = self.binary()
        output = self.root / "existing-run"
        output.mkdir()
        sentinel = output / "keep.txt"
        sentinel.write_text("original evidence")
        with self.assertRaises(FileExistsError):
            matrix.run(self.run_args(binary, "existing-run", ["cell-000"]))
        self.assertEqual(sentinel.read_text(), "original evidence")
        self.assertFalse((output / "matrix-run.json").exists())

    def test_inventory_rejects_duplicate_json_keys(self):
        payload = (
            b'{"schema":"autoeq.optimizer_benchmark_cell_inventory/v1",'
            b'"schema":"autoeq.optimizer_benchmark_cell_inventory/v1"}'
        )
        with self.assertRaisesRegex(ValueError, "duplicate JSON object key"):
            matrix.validate_inventory(payload)


if __name__ == "__main__":
    unittest.main()
