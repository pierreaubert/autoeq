"""Focused contract tests for multirate inventory identity and grouping."""

from __future__ import annotations

import struct
import unittest

from scripts import analyze_optimizer_benchmark_matrix as analyzer
from scripts import run_optimizer_benchmark_matrix as runner


def bits(value: float) -> int:
    return struct.unpack(">Q", struct.pack(">d", value))[0]


def rate_canary_cells() -> list[dict[str, object]]:
    cases = sorted(runner.RATE_CANARY_CASE_IDS)
    rates = runner.RATE_CANARY_SAMPLE_RATES_HZ
    backends = sorted(runner.RATE_CANARY_REGISTERED_BACKENDS)
    cells: list[dict[str, object]] = []

    def add(case: str, rate: int, backend: str, purpose: str,
            root_budget: int, stage_budget: int) -> None:
        cells.append({
            "schema": runner.RATE_CANARY_SPEC_SCHEMA,
            "cell_id": f"rate{rate}hz:{purpose}:{case}:{backend}",
            "process_watchdog_millis": 30_000,
            "case_id": case,
            "sample_rate_hz_bits": bits(float(rate)),
            "frequency_min_hz_bits": bits(20.0),
            "frequency_max_hz_bits": bits(float(rate) / 2.0 - 1.0),
            "backend": backend,
            "purpose": purpose,
            "seed": 42,
            "root_search_budget": root_budget,
            "stage_search_budget": stage_budget,
        })

    for rate in rates:
        for case in cases:
            for backend in backends:
                add(case, rate, backend, "ordinary", 128, 128)
            add(case, rate, "autoeq:de", "adaptive", 512, 128)
            add(case, rate, "autoeq:de", "refinement", 512, 256)
            for backend in ("autoeq:nsga2", "autoeq:nsga3", "autoeq:bo"):
                add(case, rate, backend, "pareto_front", 512, 512)
        add("analytic_headphone_peq", rate, "autoeq:cobra", "observer_stop", 128, 128)
        add("analytic_headphone_peq", rate, "autoeq:cobyla", "observer_unsupported", 128, 128)
    return cells


def validate_cells(cells: list[dict[str, object]]) -> None:
    inventory = {
        "schema": runner.RATE_CANARY_INVENTORY_SCHEMA,
        "expected_cell_count": len(cells),
        "spec_inventory_sha256": runner.sha256_bytes(runner.canonical_json_bytes(cells)),
        "sample_rate_scope": "digital_filter_realization_hz; measurement_capture_rate_not_asserted",
        "sample_rates_hz": list(runner.RATE_CANARY_SAMPLE_RATES_HZ),
        "cells": cells,
    }
    runner.validate_inventory(runner.canonical_json_bytes(inventory), rate_canary=True)


def analyzer_cells() -> list[dict[str, object]]:
    cells = rate_canary_cells()
    for cell in cells:
        cell.update({
            "manifest_sha256": "a" * 64,
            "fixture_sha256": "b" * 64,
            "filter_count": 5,
            "cooperative_deadline_millis": 10_000,
            "population_size": 30,
            "optimizer_version": "test-version",
            "multi_strategy": "average",
            "local_refiner": "none",
            "bo_ehvi": False,
            "min_q_bits": bits(0.1),
            "max_q_bits": bits(10.0),
            "min_gain_db_bits": bits(-10.0),
            "max_gain_db_bits": bits(10.0),
        })
    return cells


class OptimizerRateCanaryTests(unittest.TestCase):
    def test_exact_fixed_profile_is_accepted(self) -> None:
        cells = rate_canary_cells()
        self.assertEqual(len(cells), 112)
        validate_cells(cells)

    def test_analyzer_forwards_rate_canary_inventory_validation(self) -> None:
        cells = analyzer_cells()
        inventory = {
            "schema": runner.RATE_CANARY_INVENTORY_SCHEMA,
            "expected_cell_count": len(cells),
            "spec_inventory_sha256": runner.sha256_bytes(runner.canonical_json_bytes(cells)),
            "sample_rate_scope": "digital_filter_realization_hz; measurement_capture_rate_not_asserted",
            "sample_rates_hz": list(runner.RATE_CANARY_SAMPLE_RATES_HZ),
            "cells": cells,
        }
        parsed, parsed_cells = analyzer.validate_inventory(
            runner.canonical_json_bytes(inventory), rate_canary=True
        )
        self.assertEqual(parsed["schema"], runner.RATE_CANARY_INVENTORY_SCHEMA)
        self.assertEqual(len(parsed_cells), runner.RATE_CANARY_EXPECTED_CELLS)

    def test_consistent_case_replacement_is_rejected(self) -> None:
        cells = rate_canary_cells()
        for cell in cells:
            if cell["case_id"] == "measured_stereo_8361a":
                cell["case_id"] = "unregistered_case"
        with self.assertRaisesRegex(ValueError, "three fixed cases"):
            validate_cells(cells)

    def test_consistent_backend_replacement_is_rejected(self) -> None:
        cells = rate_canary_cells()
        for cell in cells:
            if cell["purpose"] == "ordinary" and cell["backend"] == "mh:firefly":
                cell["backend"] = "mh:unregistered"
        with self.assertRaisesRegex(ValueError, "registered 13-backend plan"):
            validate_cells(cells)

    def test_legacy_grouping_does_not_decode_rate_bits(self) -> None:
        self.assertIsNone(analyzer.sample_rate_for_group({"cell_id": "legacy"}, rate_canary=False))

    def test_rate_grouping_rejects_malformed_bit_fields(self) -> None:
        for spec in (
            {"cell_id": "missing"},
            {"cell_id": "bool", "sample_rate_hz_bits": True},
            {"cell_id": "overflow", "sample_rate_hz_bits": 1 << 64},
        ):
            with self.subTest(spec=spec), self.assertRaises(analyzer.AnalysisInputError):
                analyzer.sample_rate_for_group(spec, rate_canary=True)

    def test_rate_grouping_separates_realization_rates(self) -> None:
        low = analyzer.sample_rate_for_group(
            {"cell_id": "low", "sample_rate_hz_bits": bits(44_100.0)}, rate_canary=True
        )
        high = analyzer.sample_rate_for_group(
            {"cell_id": "high", "sample_rate_hz_bits": bits(96_000.0)}, rate_canary=True
        )
        self.assertEqual((low, high), (44_100.0, 96_000.0))


if __name__ == "__main__":
    unittest.main()
