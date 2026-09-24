"""Fast checks for selected-delay contract artifact reporting."""

import contextlib
import io
import json
import unittest
from unittest.mock import patch

import run_selected_delay_contract


class SelectedDelayRunnerTests(unittest.TestCase):
    def test_temporary_artifact_is_not_reported_as_retained(self):
        output = io.StringIO()
        with (
            patch("sys.argv", ["run_selected_delay_contract.py"]),
            patch.object(
                run_selected_delay_contract,
                "SCRATCH_ROOT",
                run_selected_delay_contract.Path(__file__).resolve().parents[1],
            ),
            patch.object(
                run_selected_delay_contract.tempfile,
                "TemporaryDirectory",
                return_value=contextlib.nullcontext("/Volumes/home_tmp/tmp/test-only"),
            ),
            patch.object(
                run_selected_delay_contract,
                "run",
                return_value={"status": "verified", "artifact": "/Volumes/home_tmp/tmp/test-only/selected-output.json"},
            ),
            contextlib.redirect_stdout(output),
        ):
            run_selected_delay_contract.main()

        report = json.loads(output.getvalue())
        self.assertEqual(report["status"], "verified")
        self.assertIs(report["artifact_retained"], False)
        self.assertNotIn("artifact", report)

    def test_explicit_artifact_is_reported_as_retained(self):
        output = io.StringIO()
        artifact = "/Volumes/home_tmp/tmp/selected-output-test-only.json"
        with (
            patch("sys.argv", ["run_selected_delay_contract.py", "--artifact", artifact]),
            patch.object(
                run_selected_delay_contract,
                "run",
                return_value={"status": "verified", "artifact": artifact},
            ),
            contextlib.redirect_stdout(output),
        ):
            run_selected_delay_contract.main()

        report = json.loads(output.getvalue())
        self.assertEqual(report["artifact"], artifact)
        self.assertIs(report["artifact_retained"], True)

    def test_combined_option_selects_public_combined_producer(self):
        output = io.StringIO()
        artifact = "/Volumes/home_tmp/tmp/combined-output-test-only.json"
        with (
            patch("sys.argv", ["run_selected_delay_contract.py", "--combined", "--artifact", artifact]),
            patch.object(
                run_selected_delay_contract,
                "run",
                return_value={"status": "verified", "artifact": artifact},
            ) as producer,
            contextlib.redirect_stdout(output),
        ):
            run_selected_delay_contract.main()

        producer.assert_called_once_with(run_selected_delay_contract.Path(artifact), combined=True)
        self.assertIs(json.loads(output.getvalue())["artifact_retained"], True)


if __name__ == "__main__":
    unittest.main()
