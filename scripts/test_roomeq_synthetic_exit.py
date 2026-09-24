#!/usr/bin/env python3
"""Shell-level exit contract checks against a freshly built release QA binary.

Build with: cargo build --release --features qa --bin roomeq-qa-synthetic
Run with: python3 scripts/test_roomeq_synthetic_exit.py
"""

from pathlib import Path
from copy import deepcopy
import json
import subprocess
import struct
import unittest
from parameter_crossover_contract import verify_crossover
from parameter_signal_contract import verify_signal_axes
from src.payload_binding import verify_payload_binding

ROOT = Path(__file__).resolve().parents[1]
BINARY = ROOT / "target/release/roomeq-qa-synthetic"
# Progress observers preserve adaptive selection and its existing per-pass
# budgets. A single full-scale 5.1 refusal row can exceed 15 minutes, and the
# reduced-level matrix contains 16 serial rows. This only bounds subprocess
# runtime; it does not relax any acoustic or safety assertion.
MATRIX_TIMEOUT_SECONDS = 3 * 60 * 60


class SavedOutput(dict):
    """Retain the sidecar directory when checking a saved RoomEQ payload."""



class SyntheticExitContract(unittest.TestCase):
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
            if decision == "identity_fallback":
                safe_fallback_rows.add(row["row"])
                self.assertTrue(acceptance.get("violations"), row["row"])
                self.assertTrue(all(value is None for value in delivered.values()), row["row"])
            else:
                self.assertEqual(decision, "accepted", row["row"])
                if row["fir_duration_applicable"]:
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
        self.assertEqual({row["row"] for row in rows}, {0, 4, 6, 9, 10, 15})
        for row in rows:
            self.assertEqual(row["input_scenario"], "full_scale_safety_refusal")
            self.assertEqual(row["processing_status"], "refused")
            self.assertEqual(row["outcome_scope"], "expected_safety_refusal")
            self.assertEqual(row["assertion_status"], "passed")
            self.assertEqual(row["finalization"]["default_input_peak"], 1.0)
            self.assertEqual(row["finalization"]["max_attenuation_db"], 12.0)
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
