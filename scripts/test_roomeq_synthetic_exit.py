#!/usr/bin/env python3
"""Shell-level exit contract checks against a freshly built release QA binary.

Build with: cargo build -p roomeq-qa --release --bin roomeq-qa-synthetic
Run with: python3 scripts/test_roomeq_synthetic_exit.py
"""

from pathlib import Path
from copy import deepcopy
import json
import math
import os
import shlex
import subprocess
import struct
import unittest
from parameter_crossover_contract import verify_crossover
from parameter_signal_contract import verify_signal_axes
from src.payload_binding import verify_payload_binding

ROOT = Path(__file__).resolve().parents[1]
TARGET = Path(os.environ.get("CARGO_TARGET_DIR", ROOT / "target"))
if not TARGET.is_absolute():
    TARGET = ROOT / TARGET
BINARY = TARGET / "release" / ("roomeq-qa-synthetic.exe" if os.name == "nt" else "roomeq-qa-synthetic")
# Progress observers preserve adaptive selection and its existing per-pass
# budgets. A single full-scale 5.1 refusal row can exceed 15 minutes, and the
# reduced-level matrix contains 17 rows (parallel workers by default, serial
# with --jobs 1). This only bounds subprocess runtime; it does not relax any
# acoustic or safety assertion.
MATRIX_TIMEOUT_SECONDS = 3 * 60 * 60


def build_qa_binary():
    """Build the owning target once; an existing file alone may be stale."""
    selected = os.environ.get("AUTOEQ_QA_SYNTHETIC")
    if selected:
        executable = Path(selected).resolve()
        if not executable.is_file():
            raise RuntimeError(f"Selected synthetic QA executable is missing: {executable}")
        return executable
    cargo = shlex.split(os.environ.get("CARGO", "cargo"))
    metadata = json.loads(subprocess.check_output(
        [*cargo, "metadata", "--offline", "--no-deps", "--format-version", "1"], cwd=ROOT))
    owners = [(package, target) for package in metadata["packages"]
              if package["id"] in metadata["workspace_members"] for target in package["targets"]
              if target["name"] == "roomeq-qa-synthetic" and "bin" in target["kind"]]
    if len(owners) != 1:
        raise RuntimeError("Synthetic QA target must have exactly one Cargo owner")
    package, target = owners[0]
    command = [*cargo, "build", "--offline", "--release", "-p", package["name"],
               "--bin", target["name"], "--message-format=json-render-diagnostics"]
    required = target.get("required-features", [])
    if required:
        command += ["--features", ",".join(required)]
    built = subprocess.run(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                           text=True, check=False)
    if built.returncode:
        raise RuntimeError("Synthetic QA build failed: " + built.stderr)
    executables = []
    for line in built.stdout.splitlines():
        event = json.loads(line)
        if (event.get("reason") == "compiler-artifact" and event["target"]["name"] == target["name"]
                and event.get("executable")):
            executables.append(Path(event["executable"]))
    if len(executables) != 1 or not executables[0].is_file():
        raise RuntimeError("Cargo did not identify the synthetic QA executable")
    return executables[0]


class SavedOutput(dict):
    """Retain the sidecar directory when checking a saved RoomEQ payload."""



class SyntheticExitContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        global BINARY
        BINARY = build_qa_binary()

    def assert_hybrid_selection(self, row):
        stages = [stage for stage in row["stage_outcomes"]
                  if stage["stage"] == "hybrid_best_part_selection"]
        self.assertEqual(len(stages), 1, row["row"])
        stage = stages[0]
        self.assertEqual(stage["status"], "applied", row["row"])
        selected = [value.removeprefix("selected=")
                    for value in stage["advisories"] if value.startswith("selected=")]
        self.assertEqual(len(selected), 1)
        self.assertIn(selected[0], {"iir", "fir", "hybrid"})
        self.assertEqual(len(stage["checks"]), 2)
        parts = {}
        initial = []
        for check in stage["checks"]:
            name = check["id"].removeprefix("hybrid_best_part_")
            self.assertIn(name, {"iir", "fir"})
            details = json.loads(check["diagnostic"])
            self.assertEqual(details["score_basis"], "configured_reference_training_mean")
            self.assertTrue(details["original_hybrid_safety_budgets"])
            parts[name] = details["part_post_mean_db"]
            initial.append(details["hybrid_post_mean_db"])
        self.assertEqual(set(parts), {"iir", "fir"})
        self.assertAlmostEqual(initial[0], initial[1], places=9)
        rows = row["selected_acceptance"]["acoustic_quality"]["training"]["configured_baseline"]["residuals"]
        post = sum(value["score"]["post_weighted_rms_db"] for value in rows) / len(rows)
        self.assertTrue(all(math.isfinite(value) and value >= 0
                            for value in [post, *initial, *parts.values()]))
        expected = initial[0] if selected[0] == "hybrid" else parts[selected[0]]
        self.assertAlmostEqual(post, expected, places=9)
        self.assertLessEqual(post, min(initial[0], *parts.values()) + 1e-9)
        return selected[0]

    def run_qa(self, *args, timeout_seconds=60):
        self.assertTrue(BINARY.is_file(), "Build the release QA binary first")
        return subprocess.run(
            [str(BINARY), *args], cwd=ROOT, capture_output=True, text=True,
            timeout=timeout_seconds, check=False,
        )

    def test_successful_subrunner_exits_zero(self):
        result = self.run_qa("--release-gates")
        self.assertIn("demo expectations hold", result.stdout)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_invalid_execution_request_exits_nonzero(self):
        result = self.run_qa("--mode", "nonexistent-mode-for-exit-contract")
        self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("mode filter", result.stderr)

    def test_parameter_smoke_matrix_exits_zero_and_records_scope(self):
        result = self.run_qa("--parameter-matrix-reduced-level", timeout_seconds=MATRIX_TIMEOUT_SECONDS)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("finite-output smoke rows passed", result.stdout)
        records = json.loads((ROOT / "target/qa/roomeq-parameter-matrix.json").read_text())
        self.assertTrue(records)
        self.assertEqual({row["sample_rate_hz"] for row in records}, {44100, 48000, 96000})
        verified_rates = {row["sample_rate_hz"] for row in records
                          if verify_signal_axes(row) == "biquad_design_rate_verified"}
        self.assertEqual(verified_rates, {44100, 48000, 96000})
        design_row = next(row for row in records
                          if verify_signal_axes(row) == "biquad_design_rate_verified")
        for fault_kind in ("runtime_rate", "design_rate", "phase"):
            fault = deepcopy(design_row)
            if fault_kind == "runtime_rate":
                fault["sample_rate_hz"] += 1
            elif fault_kind == "design_rate":
                rates = next(values for values in fault["delivered_biquad_sample_rates_hz"].values() if values)
                rates[0] += 1
            else:
                measurement = next(iter(fault["effective_config"]["measurements"].values()))
                measurement["has_phase"] = not measurement["has_phase"]
            with self.assertRaises(AssertionError, msg=fault_kind):
                verify_signal_axes(fault)
        crossover_statuses = {verify_crossover(row) for row in records}
        self.assertEqual(crossover_statuses, {
            "not_applicable", "fixed_routes_verified", "automatic_routes_verified",
            "unsupported_missing_phase",
        })
        # Challenge actual emitted row evidence, not only hand-built fixtures.
        routed = next(row for row in records if row["requested_axes"]["topology"] != 0)
        fault = deepcopy(routed)
        routes = fault["bass_management"]["routing_graph"]["routes"]
        routes.remove(next(route for route in routes if route["route_kind"] == "redirected_bass_lowpass_to_sub"))
        with self.assertRaises(AssertionError):
            verify_crossover(fault)
        applied_fir_modes = set()
        safe_fallback_rows = set()
        for row in records:
            self.assertEqual(row["input_scenario"], "reduced_level_processing")
            limits = row["effective_config"]["optimizer"]["finalization"]
            self.assertEqual(limits["default_input_peak"], 0.1)
            self.assertEqual(limits["max_attenuation_db"], 12.0)
            self.assertEqual(limits["output_ceiling_dbfs"], 0.0)
            self.assertEqual(limits["input_peak_limits"], {})
            self.assertEqual(row["outcome_scope"], "finite_output_smoke")
            bundle = row["replay_bundle"]
            directory = (ROOT / bundle["directory"]).resolve()
            self.assertTrue(directory.is_relative_to((ROOT / "target/qa/roomeq-parameter-bundles").resolve()))
            request = json.loads((directory / bundle["request"]).read_text())
            output = json.loads((directory / bundle["selected_output"]).read_text())
            # QA seed metadata must not leave the selected DSP output unbound.
            saved_output = SavedOutput(output)
            saved_output.source_directory = directory
            binding_verified, binding_reason, _ = verify_payload_binding(saved_output)
            self.assertTrue(binding_verified, binding_reason)
            self.assertEqual(request["row"], row["row"])
            self.assertEqual(request["sample_rate_hz"], row["sample_rate_hz"])
            self.assertEqual(request["requested_axes"], row["requested_axes"])
            self.assertEqual(request["input_scenario"], "reduced_level_processing")
            self.assertEqual(request["inputs"]["configuration_without_speakers"]["optimizer"]["finalization"], limits)
            self.assertEqual(set(request["inputs"]["single_speaker_measurements"]),
                             set(row["effective_config"]["measurements"]))
            declared = request["inputs"]["declared_measurement_sources"]
            if row["requested_axes"]["phase"] == 1:
                self.assertEqual(set(declared), set(row["effective_config"]["measurements"]))
                for source in declared.values():
                    self.assertEqual(source["provenance"]["capture_kind"], "stationary_ir")
                    self.assertEqual(source["provenance"]["timing_reference_id"],
                                     "qa-analytic-common-clock")
                    self.assertIn("phase_deg", source["inline"])
            else:
                self.assertEqual(declared, {})
                self.assertEqual(set(output["channels"]), set(row["delivered_fir_taps"]))
            for name, chain in output["channels"].items():
                convolutions = [plugin for plugin in chain["plugins"]
                                if plugin["plugin_type"] == "convolution"]
                if row["delivered_fir_taps"][name] is not None:
                    self.assertTrue(convolutions, name)
                for plugin in convolutions:
                    sidecar = (directory / plugin["parameters"]["ir_file"]).resolve()
                    self.assertTrue(sidecar.is_relative_to(directory), str(sidecar))
                    payload = sidecar.read_bytes()
                    self.assertEqual(payload[:4], b"RIFF")
                    self.assertEqual(payload[8:12], b"WAVE")
                    offset, rate, block_align, data_size = 12, None, None, None
                    while offset + 8 <= len(payload):
                        tag, size = struct.unpack_from("<4sI", payload, offset)
                        offset += 8
                        self.assertLessEqual(offset + size, len(payload))
                        if tag == b"fmt ":
                            _, _, rate, _, block_align = struct.unpack_from("<HHIIH", payload, offset)
                        elif tag == b"data":
                            data_size = size
                        offset += size + size % 2
                    self.assertEqual(rate, row["sample_rate_hz"])
                    self.assertIsNotNone(data_size)
                    self.assertIsNotNone(block_align)
                    self.assertGreater(block_align, 0)
                    self.assertEqual(data_size // block_align, row["delivered_fir_taps"][name])
            crossover = verify_crossover(row)
            statuses = {"fixed_routes_verified": "fixed_routes_emitted",
                        "automatic_routes_verified": "automatic_selection_applied",
                        "unsupported_missing_phase": "unsupported_missing_phase",
                        "not_applicable": "not_applicable"}
            self.assertEqual(row["crossover_execution"], statuses[crossover])
            self.assertEqual(row["unexecuted_axes"],
                             ["crossover"] if crossover == "unsupported_missing_phase" else [])
            self.assertEqual(row["not_applicable_axes"],
                             ["crossover"] if crossover == "not_applicable" else [])
            self.assertIn("optimizer", row["effective_config"])
            self.assertIn("requested_axes", row)
            axes = row["requested_axes"]
            self.assertEqual(row["effective_config"]["optimizer"]["processing_mode"],
                             ["low_latency", "phase_linear", "hybrid"][axes["mode"]])
            measurements = row["effective_config"]["measurements"]
            self.assertEqual(len(measurements), [2, 3, 6][axes["topology"]])
            if axes["topology"]:
                self.assertEqual(row["home_cinema_layout"]["bed_channels"], [0, 2, 5][axes["topology"]])
                self.assertTrue(row["bass_management"]["enabled"])
                self.assertEqual(len(row["bass_management"]["signal_flow"]),
                                 len(row["bass_management"]["routing_graph"]["input_channels"]))
            for measurement in measurements.values():
                self.assertEqual(measurement["has_phase"], axes["phase"] == 1)
            if axes["measurement_shape"] == 1:
                self.assertEqual(len({measurement["grid_points"] for measurement in measurements.values()}), len(measurements))
            if axes["measurement_shape"] == 2:
                for measurement in measurements.values():
                    self.assertAlmostEqual(measurement["band_hz"][0], 100.0)
                    self.assertAlmostEqual(measurement["band_hz"][1], 6000.0)
            if row["fir_duration_applicable"]:
                taps = row["effective_config"]["optimizer"]["fir"]["taps"]
                duration = taps * 1000 / row["sample_rate_hz"]
                self.assertLessEqual(abs(duration - row["requested_fir_duration_ms"]), 500 / row["sample_rate_hz"] + 1e-9)
            acceptance = row["selected_acceptance"] or {}
            decision = acceptance.get("decision")
            delivered = row["delivered_fir_taps"]
            self.assertTrue(delivered)
            if decision in {"identity_fallback", "safe_fallback"}:
                safe_fallback_rows.add(row["row"])
                self.assertIs(acceptance.get("accepted"), False, row["row"])
                self.assertTrue(acceptance.get("violations"), row["row"])
                self.assertTrue(all(value is None for value in delivered.values()), row["row"])
            else:
                self.assertEqual(decision, "accepted", row["row"])
                winner = self.assert_hybrid_selection(row) if row["mode"] == "Hybrid" else None
                if winner == "iir":
                    self.assertTrue(all(value is None for value in delivered.values()), row["row"])
                elif row["fir_duration_applicable"]:
                    self.assertEqual(set(delivered.values()), {taps}, row["row"])
                    applied_fir_modes.add(row["mode"])
                else:
                    self.assertTrue(all(value is None for value in delivered.values()), row["row"])
        self.assertEqual(applied_fir_modes, {"PhaseLinear", "Hybrid"})
        self.assertTrue(safe_fallback_rows, "matrix must report declined correction candidates")

    def test_full_scale_parameter_cases_refuse_for_structural_headroom(self):
        result = self.run_qa("--parameter-matrix-refusals", timeout_seconds=MATRIX_TIMEOUT_SECONDS)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        rows = json.loads((ROOT / "target/qa/roomeq-parameter-refusals.json").read_text())
        self.assertEqual({row["row"] for row in rows}, {0, 5, 6, 11, 12, 14, 15, 16})
        for row in rows:
            self.assertEqual(row["input_scenario"], "full_scale_safety_refusal")
            self.assertEqual(row["processing_status"], "refused")
            self.assertEqual(row["outcome_scope"], "expected_safety_refusal")
            self.assertEqual(row["assertion_status"], "passed")
            self.assertEqual(row["finalization"]["default_input_peak"], 1.0)
            self.assertEqual(row["finalization"]["max_attenuation_db"], 12.0)
            self.assertEqual(row["finalization"]["output_ceiling_dbfs"], -24.0)
            required = row["required_attenuation_db_by_seed"]
            self.assertEqual(len(required), 5)
            self.assertTrue(all(value > 12.0 for value in required))
            bundle = row["replay_bundle"]
            directory = (ROOT / bundle["directory"]).resolve()
            self.assertTrue(directory.is_relative_to((ROOT / "target/qa/roomeq-parameter-bundles").resolve()))
            self.assertFalse((directory / "selected-output.json").exists())
            self.assertEqual(json.loads((directory / bundle["refusal"]).read_text()), row)
            request = json.loads((directory / bundle["request"]).read_text())
            self.assertEqual(request["input_scenario"], row["input_scenario"])
            self.assertEqual(request["inputs"]["configuration_without_speakers"]["optimizer"]["finalization"], row["finalization"])


if __name__ == "__main__":
    unittest.main()
