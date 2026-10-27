"""Regression tests for complete, truthful measured-mode comparison evidence."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from argparse import Namespace

SCRIPT = Path(__file__).with_name("roomeq_suite_report.py")
SPEC = importlib.util.spec_from_file_location("suite_report", SCRIPT)
suite = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(suite)


class MeasuredModeComparisonTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def result(self, mode, metric=1.0):
        return {"metadata": {
            "correction_acceptance": {"outcome": "accepted", "metrics": {
                "pre_target_weighted_rms_db": metric,
                "post_target_weighted_rms_db": 0.5,
                "improvement_db": 0.5}},
            "effective_config": {"version": "3.0.0", "speakers": {"L": {"measurement": "L.csv"}},
                "system": {"bass_management": {"lfe_playback_gain_db": 0.0}},
                "target_curve": "target.csv", "recording_config": {"playback_sample_rate": 48000, "recording_sample_rate": 48000}, "optimizer": {
                    "processing_mode": mode,
                    "finalization": {"subwoofer_limiter": False, "output_ceiling_dbfs": -1.0}}}}}

    def save(self, mode, result):
        folder = self.root / "room" / mode
        folder.mkdir(parents=True, exist_ok=True)
        (folder / f"dsp-{mode}.json").write_text(json.dumps(result))

    def compare(self):
        return suite.compare_scenario(self.root, "room", ["iir", "fir"])

    def frozen_result(self, mode, native_pre=1.0):
        result = self.result(mode, native_pre)
        result["metadata"]["correction_acceptance"]["acoustic_quality"] = {
            "training": {"curve_count": 1, "frozen_grid_residual": [{
                "logical_input": "L", "curve_index": 0,
                "frequency_grid_sha256": "a" * 64, "baseline_samples_sha256": "b" * 64,
                "target_samples_sha256": "c" * 64, "sample_count": 3,
                "evaluated_band_hz": [20.0, 20000.0], "normalized_to_pre": True,
                "level_offset_db": 80.0, "post_reference_gain_db": 0.0,
                "pre_weighted_rms_db": 1.0, "post_weighted_rms_db": 0.5,
                "improvement_db": 0.5, "post_covers_baseline_grid": True,
            }]}}
        return result

    def save_configured_parts(self, scores, hybrid_outcome="accepted"):
        for mode, values in scores.items():
            result = self.frozen_result(mode)
            acceptance = result["metadata"]["correction_acceptance"]
            if mode == "mixed":
                acceptance["outcome"] = hybrid_outcome
            training = acceptance["acoustic_quality"]["training"]
            rows = []
            for index, post in enumerate(values):
                row = dict(training["frozen_grid_residual"][0])
                row.update(curve_index=index, post_weighted_rms_db=post, improvement_db=1.0-post)
                rows.append(row)
            training.update(curve_count=len(rows), frozen_grid_residual=rows,
                configured_baseline={"method": "configured_routing_unprocessed_captures_v1",
                    "playback_graph_sha256": "f"*64, "unavailable": [],
                    "residuals": [{"seat_index": index, "physical_outputs": ["L"], "score": row}
                                  for index, row in enumerate(rows)]})
            self.save(mode, result)

    def compare_parts(self):
        return suite.compare_scenario(self.root, "room", ["iir", "fir", "mixed"])

    def test_accepted_hybrid_must_match_best_part_mean_even_with_better_median(self):
        self.save_configured_parts({"iir": [0.2, 0.2, 0.8], "fir": [0.3]*3,
                                    "mixed": [0.0, 0.1, 1.0]})
        report = self.compare_parts()
        check = report["hybrid_best_part"]
        self.assertTrue(report["frozen_grid_basis_equal"])
        self.assertLess(report["frozen_grid_scores"]["mixed"]["post_weighted_rms_median_db"], 0.3)
        self.assertEqual(report["verdict"], "fail")
        self.assertEqual(check["best_part"], "fir")
        self.assertAlmostEqual(check["signed_regression_db"], 1.1/3-0.3)
        self.assertFalse(check["passed"])
        self.assertTrue(check["required_for_accepted_hybrid"])

    def test_hybrid_part_ties_use_only_numerical_tolerance(self):
        for difference, verdict in [(0.0, "pass"), (0.5e-9, "pass"), (2e-9, "fail"), (0.04, "fail")]:
            with self.subTest(difference=difference):
                self.save_configured_parts({"iir": [0.3], "fir": [0.4], "mixed": [0.3+difference]})
                report = self.compare_parts()
                self.assertEqual(report["verdict"], verdict)
                self.assertEqual(report["hybrid_best_part"]["best_part"], "iir")

    def test_hybrid_best_part_requires_configured_supported_matching_evidence(self):
        self.save_configured_parts({"iir": [0.5], "fir": [0.4], "mixed": [0.3]})
        self.assertEqual(self.compare_parts()["verdict"], "pass")
        path = self.root / "room/fir/dsp-fir.json"
        original = json.loads(path.read_text())
        for mutation in ("missing_configured", "unavailable", "different_raw", "rejected", "invalid_metadata"):
            with self.subTest(mutation=mutation):
                result = json.loads(json.dumps(original))
                acceptance = result["metadata"]["correction_acceptance"]
                training = acceptance["acoustic_quality"]["training"]
                if mutation == "missing_configured":
                    del training["configured_baseline"]
                elif mutation == "unavailable":
                    training["configured_baseline"]["unavailable"] = ["missing seat phase"]
                elif mutation == "different_raw":
                    training["configured_baseline"]["residuals"][0]["score"]["baseline_samples_sha256"] = "e"*64
                elif mutation == "rejected":
                    acceptance["outcome"] = "rejected"
                else:
                    result["metadata"] = "invalid"
                self.save("fir", result)
                report = self.compare_parts()
                self.assertEqual(report["verdict"], "fail")
                self.assertEqual(report["hybrid_best_part"]["status"], "unavailable")
                self.assertIsNone(report["hybrid_best_part"]["passed"])
        self.save("fir", original)
        (self.root / "room/fir/dsp-fir.json").unlink()
        self.assertEqual(self.compare_parts()["verdict"], "fail")

    def test_flagged_hybrid_fallback_reports_gap_without_claiming_correction(self):
        self.save_configured_parts({"iir": [0.3], "fir": [0.4], "mixed": [0.5]},
                                    hybrid_outcome="safe_fallback")
        report = self.compare_parts()
        self.assertEqual(report["verdict"], "pass")
        check = report["hybrid_best_part"]
        self.assertEqual(check["status"], "fallback_diagnostic")
        self.assertFalse(check["required_for_accepted_hybrid"])
        self.assertFalse(check["passed"])
        self.assertAlmostEqual(check["signed_regression_db"], 0.2)

    def test_fixed_grid_comparison_can_differ_from_native_safety_metrics(self):
        self.save("iir", self.frozen_result("iir", 1.0))
        self.save("fir", self.frozen_result("fir", 1.2))
        report = self.compare()
        self.assertFalse(report["pre_metrics_equal"])
        self.assertTrue(report["frozen_grid_basis_equal"])
        self.assertEqual(report["verdict"], "pass")
        self.assertEqual(report["frozen_grid_scores"]["fir"]["improvement_db"], 0.5)

    def test_summary_uses_frozen_mean_and_retains_native_and_training_scores(self):
        self.save_configured_parts({"iir": [0.5, 1.0, 1.5], "fir": [0.5, 1.0, 1.5]})
        for mode, native_pre in [("iir", 5.7), ("fir", 5.4)]:
            path = self.root / "room" / mode / f"dsp-{mode}.json"
            data = json.loads(path.read_text())
            metadata = data["metadata"]
            metadata.update(pre_score=5.8, post_score=3.2)
            acceptance = metadata["correction_acceptance"]
            acceptance["metrics"].update(pre_target_weighted_rms_db=native_pre,
                post_target_weighted_rms_db=native_pre-0.04, improvement_db=0.04)
            rows = acceptance["acoustic_quality"]["training"]["configured_baseline"]["residuals"]
            for row, pre in zip(rows, [1.0, 2.0, 9.0]):
                score = row["score"]
                score.update(pre_weighted_rms_db=pre, improvement_db=pre-score["post_weighted_rms_db"])
            self.save(mode, data)
            row = suite.combo_status(self.root, "room", mode)
            self.assertEqual(row["score_basis"], "configured_reference_training_mean")
            self.assertEqual(row["pre_target_weighted_rms_db"], 4.0)
            self.assertEqual(row["post_target_weighted_rms_db"], 1.0)
            self.assertEqual(row["improvement_db"], 3.0)
            self.assertEqual(row["native_marginal_metrics"]["pre_target_weighted_rms_db"], native_pre)
            self.assertEqual(row["native_marginal_metrics"]["improvement_db"], 0.04)
            self.assertEqual(row["optimizer_training_scores"], {"pre_score": 5.8, "post_score": 3.2})
        comparison = self.compare()
        self.assertEqual(comparison["verdict"], "pass")
        self.assertFalse(comparison["pre_metrics_equal"])
        self.assertEqual(comparison["pre_metric_basis"], "native_marginal_safety_diagnostic")
        self.assertTrue(comparison["comparison_pre_metrics_equal"])
        self.assertEqual(comparison["comparison_pre_metrics"], {"iir": 4.0, "fir": 4.0})

    def test_legacy_expectation_keeps_native_basis_and_report_source_receipt(self):
        expected = {"combinations": {}}
        for mode in ["iir", "fir"]:
            result = self.frozen_result(mode)
            result["metadata"]["correction_acceptance"]["metrics"]["improvement_db"] = 0.04
            self.save(mode, result)
            expected["combinations"][f"room/{mode}"] = {
                "status": "accepted", "outcome": "accepted", "improvement_db": 0.04}
        path = self.root / "expect.json"
        path.write_text(json.dumps(expected))
        run = subprocess.run([sys.executable, str(SCRIPT), str(self.root), "--scenarios", "room",
                              "--modes", "iir fir", "--expect", str(path), "--check"],
                             capture_output=True, text=True, timeout=10)
        self.assertEqual(run.returncode, 0, run.stdout+run.stderr)
        for filename in ["summary.json", "comparison.json"]:
            report = json.loads((self.root / filename).read_text())
            self.assertEqual(report["generated_by_sha256"], suite.REPORT_SOURCE_SHA256)
        summary = json.loads((self.root / "summary.json").read_text())
        self.assertEqual(summary["combinations"]["room/fir"]["improvement_db"], 0.5)
        expected["combinations"]["room/fir"]["score_basis"] = "unknown"
        path.write_text(json.dumps(expected))
        run = subprocess.run([sys.executable, str(SCRIPT), str(self.root), "--scenarios", "room",
                              "--modes", "iir fir", "--expect", str(path)],
                             capture_output=True, text=True, timeout=10)
        self.assertEqual(run.returncode, 2, run.stdout+run.stderr)
        self.assertIn("score_basis unknown", run.stdout)

    def test_relative_target_constant_level_uses_recorded_canonical_basis(self):
        for mode, removed in [("iir", 0.0), ("fir", 60.0)]:
            result = self.frozen_result(mode)
            row = result["metadata"]["correction_acceptance"]["acoustic_quality"]["training"]["frozen_grid_residual"][0]
            row["target_samples_sha256"] = ("c" if mode == "iir" else "d") * 64
            row["level_offset_db"] = 80.0 - removed
            row["relative_target_reference"] = {
                "method": "first_baseline_frequency_target_level_nanodb_v1",
                "target_level_removed_db": removed, "target_shape_sha256": "e" * 64,
                "level_offset_db": 80.0,
            }
            self.save(mode, result)
        self.assertEqual(self.compare()["verdict"], "pass")
        for key, value in [("target_shape_sha256", "f" * 64), ("level_offset_db", 81.0),
                           ("target_level_removed_db", True), ("method", "unknown")]:
            with self.subTest(key=key):
                result = json.loads((self.root / "room/fir/dsp-fir.json").read_text())
                row = result["metadata"]["correction_acceptance"]["acoustic_quality"]["training"]["frozen_grid_residual"][0]
                row["relative_target_reference"][key] = value
                self.save("fir", result)
                self.assertEqual(self.compare()["verdict"], "fail")
                row["relative_target_reference"][key] = {
                    "target_shape_sha256": "e" * 64, "level_offset_db": 80.0,
                    "target_level_removed_db": 60.0,
                    "method": "first_baseline_frequency_target_level_nanodb_v1",
                }[key]
                self.save("fir", result)
        result = json.loads((self.root / "room/fir/dsp-fir.json").read_text())
        row = result["metadata"]["correction_acceptance"]["acoustic_quality"]["training"]["frozen_grid_residual"][0]
        row["normalized_to_pre"] = False
        self.save("fir", result)
        self.assertEqual(self.compare()["verdict"], "fail")

    def test_configured_reference_cannot_inherit_or_waive_candidate_baseline(self):
        for mode, raw_hash in [("iir", "b" * 64), ("fir", "d" * 64)]:
            result = self.frozen_result(mode)
            partition = result["metadata"]["correction_acceptance"]["acoustic_quality"]["training"]
            score = dict(partition["frozen_grid_residual"][0])
            partition["frozen_grid_residual"][0]["baseline_samples_sha256"] = raw_hash
            partition["configured_baseline"] = {
                "method": "configured_routing_unprocessed_captures_v1", "playback_graph_sha256": "f" * 64,
                "residuals": [{"seat_index": 0, "physical_outputs": ["L"], "score": score}],
                "unavailable": [],
            }
            self.save(mode, result)
        self.assertEqual(self.compare()["verdict"], "pass")
        result = json.loads((self.root / "room/fir/dsp-fir.json").read_text())
        configured = result["metadata"]["correction_acceptance"]["acoustic_quality"]["training"]["configured_baseline"]
        for key, value in [("unavailable", ["missing physical phase"]),
                           ("playback_graph_sha256", "e" * 64), ("method", "unknown")]:
            with self.subTest(key=key):
                original = configured[key]
                configured[key] = value
                self.save("fir", result)
                self.assertEqual(self.compare()["verdict"], "fail")
                configured[key] = original
        configured["residuals"][0]["score"]["baseline_samples_sha256"] = "a" * 64
        self.save("fir", result)
        self.assertEqual(self.compare()["verdict"], "fail")

    def test_matching_scalar_pre_does_not_hide_different_frozen_bases(self):
        for key, value in [("baseline_samples_sha256", "d" * 64),
                           ("frequency_grid_sha256", "d" * 64),
                           ("target_samples_sha256", "d" * 64),
                           ("level_offset_db", 81.0), ("post_reference_gain_db", -1.0)]:
            with self.subTest(key=key):
                self.save("iir", self.frozen_result("iir"))
                fir = self.frozen_result("fir")
                fir["metadata"]["correction_acceptance"]["acoustic_quality"]["training"]["frozen_grid_residual"][0][key] = value
                self.save("fir", fir)
                report = self.compare()
                self.assertTrue(report["pre_metrics_equal"])
                self.assertFalse(report["frozen_grid_basis_equal"])
                self.assertEqual(report["verdict"], "fail")

    def test_fixed_grid_evidence_does_not_waive_missing_native_metric(self):
        self.save("iir", self.frozen_result("iir"))
        self.save("fir", self.frozen_result("fir", None))
        report = self.compare()
        self.assertTrue(report["frozen_grid_basis_equal"])
        self.assertIsNone(report["pre_metrics_equal"])
        self.assertEqual(report["verdict"], "fail")

    def test_incomplete_frozen_support_cannot_fall_back_to_equal_scalars(self):
        self.save("iir", self.frozen_result("iir"))
        for key, value in [("post_covers_baseline_grid", False),
                           ("post_weighted_rms_db", None), ("improvement_db", 0.7),
                           ("sample_count", True), ("curve_index", True),
                           ("frequency_grid_sha256", "malformed")]:
            with self.subTest(key=key):
                fir = self.frozen_result("fir")
                fir["metadata"]["correction_acceptance"]["acoustic_quality"]["training"]["frozen_grid_residual"][0][key] = value
                self.save("fir", fir)
                self.assertEqual(self.compare()["verdict"], "fail")
        self.save("fir", self.result("fir"))
        self.assertEqual(self.compare()["verdict"], "fail")
        for mode in ["iir", "fir"]:
            malformed = self.frozen_result(mode)
            malformed["metadata"]["correction_acceptance"]["acoustic_quality"]["training"]["frozen_grid_residual"] = []
            self.save(mode, malformed)
        self.assertEqual(self.compare()["verdict"], "fail")

    def test_saved_manifest_preserves_badroom_and_failed_combinations(self):
        scenarios = ["room", "2.0_gen8340a_p1_badroom"]
        (self.root / "manifest.json").write_text(json.dumps({
            "scenarios": scenarios, "modes": ["iir", "fir", "mixed"]}))
        with patch.dict(os.environ, {}, clear=True):
            run = subprocess.run([sys.executable, str(SCRIPT), str(self.root)],
                                 capture_output=True, text=True, timeout=10)
        self.assertEqual(run.returncode, 0, run.stderr)
        summary = json.loads((self.root / "summary.json").read_text())
        self.assertEqual(summary["scenarios"], scenarios)
        self.assertEqual(len(summary["combinations"]), 6)
        self.assertEqual(summary["counts"], {"error": 6})

    def test_explicit_matrix_overrides_manifest(self):
        (self.root / "manifest.json").write_text(json.dumps({
            "scenarios": ["saved"], "modes": ["mixed"]}))
        args = Namespace(out_dir=self.root, scenarios="chosen", modes="fir")
        self.assertEqual(suite.expected_matrix(args), (["chosen"], ["fir"]))

    def test_default_matrix_includes_all_23_scenarios(self):
        args = Namespace(out_dir=self.root, scenarios="", modes="")
        with patch.dict(os.environ, {}, clear=True):
            scenarios, modes = suite.expected_matrix(args)
        self.assertEqual(len(scenarios), 23)
        self.assertEqual(len(set(scenarios)), 23)
        self.assertTrue({"2.0_t7v_2024", "2.0_t7v_2026"}.issubset(scenarios))
        self.assertEqual(len([s for s in scenarios if "badroom" in s]), 6)
        self.assertEqual(len(modes), 4)

    def test_safe_fallback_remains_a_distinct_terminal_outcome(self):
        result = self.result("fir")
        result["metadata"]["correction_acceptance"]["outcome"] = "safe_fallback"
        self.save("fir", result)
        row = suite.combo_status(self.root, "room", "fir")
        self.assertEqual(row["status"], "safe_fallback")
        self.assertEqual(row["outcome"], "safe_fallback")

    def test_comparison_retains_level_matched_epa_and_pre_ringing(self):
        result = self.result("fir")
        receipt = {"method": "log_frequency_pressure_energy_1_2_khz_v1",
                   "reference_band_hz": [1000.0, 2000.0], "frequency_grid_sha256": "a" * 64,
                   "sample_count": 128, "pre_reference_db": 80.0, "post_reference_db": 68.0,
                   "listening_level_phon": 70.0, "pre": {"preference": 5.0},
                   "post": {"preference": 5.0}, "preference_delta": 0.0}
        result["metadata"]["epa_per_channel"] = {"L": {"level_matched": receipt}}
        result["metadata"]["epa_multichannel"] = {"level_matched": receipt}
        result["metadata"]["correction_acceptance"]["acoustic_quality"] = {
            "temporal": {"pre_ringing_audible_db": -35.0}}
        self.save("fir", result)
        row = suite.combo_status(self.root, "room", "fir")
        self.assertEqual(row["status"], "accepted")
        self.assertEqual(row["pre_ringing_audible_db"], -35.0)
        self.assertEqual(row["level_matched_epa"]["per_channel"]["L"]["preference_delta"], 0.0)
        self.assertEqual(row["level_matched_epa"]["multichannel"]["post_reference_db"], 68.0)
        result["metadata"]["epa_per_channel"]["L"]["level_matched"]["post"] = "invalid"
        self.save("fir", result)
        self.assertEqual(suite.combo_status(self.root, "room", "fir")["status"], "error")

    def test_stage_failures_remain_separate_from_playback_outcome(self):
        self.save("iir", self.result("iir"))
        (self.root / "room/iir/stages.json").write_text(json.dumps({
            "roomeq": 0, "display": 1, "audit": 0}))
        entry = suite.combo_status(self.root, "room", "iir")
        self.assertEqual(entry["status"], "accepted")
        self.assertEqual(entry["stages"], {"roomeq": 0, "display": 1, "audit": 0})

    def test_cli_check_rejects_failed_audit_despite_approved_playback(self):
        for mode in ["iir", "fir"]:
            self.save(mode, self.result(mode))
        (self.root / "room/iir/stages.json").write_text(json.dumps({
            "roomeq": 0, "display": 0, "audit": 1}))
        run = subprocess.run([sys.executable, str(SCRIPT), str(self.root),
                              "--scenarios", "room", "--modes", "iir fir", "--check"],
                             capture_output=True, text=True, timeout=10)
        self.assertEqual(run.returncode, 2, run.stdout + run.stderr)
        self.assertIn("1 combinations with stage failures", run.stdout)

    def test_equal_controls_allow_mode_design_differences(self):
        self.save("iir", self.result("low_latency"))
        fir = self.result("phase_linear")
        fir["metadata"]["effective_config"]["optimizer"]["fir"] = {"taps": 4096}
        self.save("fir", fir)
        report = self.compare()
        self.assertEqual(report["verdict"], "pass")
        self.assertTrue(report["pre_metrics_equal"])
        self.assertTrue(report["shared_playback_controls_equal"])
        self.assertTrue(report["effective_config_diffs"])
        self.assertFalse(any("explicit intentional" in n for n in report["notes"]))

    def test_every_requested_mode_needs_finite_pre_metric(self):
        self.save("iir", self.result("iir"))
        for invalid in [None, True, "1.0", -1.0, 10**400, float("nan"), float("inf"), -float("inf")]:
            with self.subTest(invalid=invalid):
                self.save("fir", self.result("fir", invalid))
                report = self.compare()
                self.assertEqual(report["verdict"], "fail")
                self.assertIsNone(report["pre_metrics_equal"])
                self.assertIsNone(report["pre_metrics"].get("fir"))

    def test_missing_mode_and_invalid_json_object_fail(self):
        self.save("iir", self.result("iir"))
        self.assertEqual(self.compare()["verdict"], "fail")
        self.save("fir", [self.result("fir")])
        self.assertEqual(self.compare()["verdict"], "fail")

    def test_missing_or_empty_effective_config_does_not_pass(self):
        self.save("iir", self.result("iir"))
        for invalid in [None, {}, [], "unknown", {"version": "3.0.0", "speakers": {}}, {"optimizer": {"finalization": {}}}]:
            with self.subTest(invalid=invalid):
                result = self.result("fir")
                result["metadata"]["effective_config"] = invalid
                self.save("fir", result)
                self.assertEqual(self.compare()["verdict"], "fail")

    def test_every_requested_mode_is_required_even_with_two_present(self):
        self.save("iir", self.result("iir"))
        self.save("fir", self.result("fir"))
        report = suite.compare_scenario(self.root, "room", ["iir", "fir", "mixed"])
        self.assertEqual(report["verdict"], "fail")
        self.assertFalse(report["mode_results_complete"])

    def test_shared_control_changes_fail_despite_equal_pre_metrics(self):
        mutations = [
            lambda c: c["optimizer"]["finalization"].update(subwoofer_limiter=True),
            lambda c: c["optimizer"]["finalization"].update(output_ceiling_dbfs=-3.0),
            lambda c: c["recording_config"].update(playback_sample_rate=44100),
            lambda c: c["system"]["bass_management"].update(lfe_playback_gain_db=10.0),
            lambda c: c.update(target_curve="other.csv"),
            lambda c: c["speakers"]["L"].update(measurement="other.csv"),
        ]
        self.save("iir", self.result("iir"))
        for mutate in mutations:
            result = self.result("fir")
            mutate(result["metadata"]["effective_config"])
            self.save("fir", result)
            report = self.compare()
            self.assertTrue(report["pre_metrics_equal"])
            self.assertEqual(report["verdict"], "fail")
            self.assertFalse(report["shared_playback_controls_equal"])
            self.assertTrue(report["shared_playback_control_diffs"])

    def test_missing_metadata_and_metrics_do_not_crash_or_pass(self):
        self.save("iir", self.result("iir"))
        for result in [{}, {"metadata": []}, {"metadata": {"correction_acceptance": []}},
                       {"metadata": {"correction_acceptance": {"metrics": []}}}]:
            with self.subTest(result=result):
                self.save("fir", result)
                self.assertEqual(self.compare()["verdict"], "fail")

    def test_duplicate_requested_modes_cannot_count_as_comparison(self):
        self.save("iir", self.result("iir"))
        report = suite.compare_scenario(self.root, "room", ["iir", "iir"])
        self.assertEqual(report["verdict"], "fail")

    def test_cli_check_refuses_policy_mismatch_and_preserves_diffs(self):
        self.save("iir", self.result("iir"))
        fir = self.result("fir")
        fir["metadata"]["effective_config"]["optimizer"]["finalization"]["subwoofer_limiter"] = True
        self.save("fir", fir)
        args = [sys.executable, str(SCRIPT), str(self.root), "--scenarios", "room", "--modes", "iir fir", "--check"]
        run = subprocess.run(args, capture_output=True, text=True, timeout=10)
        self.assertEqual(run.returncode, 2, run.stdout + run.stderr)
        report = json.loads((self.root / "comparison.json").read_text())["scenarios"]["room"]
        self.assertTrue(report["pre_metrics_equal"])
        self.assertIn("optimizer.finalization.subwoofer_limiter", [d["path"] for d in report["shared_playback_control_diffs"]])
        # Report-only execution remains available and still writes the failure.
        report_only = subprocess.run(args[:-1], capture_output=True, text=True, timeout=10)
        self.assertEqual(report_only.returncode, 0, report_only.stderr)
        self.assertEqual(json.loads((self.root / "comparison.json").read_text())["scenarios"]["room"]["verdict"], "fail")

    def test_cli_malformed_metadata_reports_error_without_crashing(self):
        self.save("iir", self.result("iir"))
        self.save("fir", {"metadata": ["invalid"]})
        run = subprocess.run([sys.executable, str(SCRIPT), str(self.root), "--scenarios", "room", "--modes", "iir fir", "--check"], capture_output=True, text=True, timeout=10)
        self.assertEqual(run.returncode, 2, run.stdout + run.stderr)
        self.assertNotIn("Traceback", run.stderr)
        summary = json.loads((self.root / "summary.json").read_text())
        self.assertEqual(summary["combinations"]["room/fir"]["status"], "error")

    def test_cli_check_accepts_complete_equal_controls(self):
        self.save("iir", self.result("iir"))
        self.save("fir", self.result("fir"))
        run = subprocess.run([sys.executable, str(SCRIPT), str(self.root), "--scenarios", "room", "--modes", "iir fir", "--check"], capture_output=True, text=True, timeout=10)
        self.assertEqual(run.returncode, 0, run.stdout + run.stderr)

    def test_cli_check_enforces_hybrid_best_part_and_writes_numeric_receipt(self):
        args = [sys.executable, str(SCRIPT), str(self.root), "--scenarios", "room",
                "--modes", "iir fir mixed", "--check"]
        for hybrid_score, expected_exit in [(0.5, 2), (0.3, 0)]:
            with self.subTest(hybrid_score=hybrid_score):
                self.save_configured_parts({"iir": [0.4], "fir": [0.3],
                                            "mixed": [hybrid_score]})
                run = subprocess.run(args, capture_output=True, text=True, timeout=10)
                self.assertEqual(run.returncode, expected_exit, run.stdout + run.stderr)
                report = json.loads((self.root / "comparison.json").read_text())["scenarios"]["room"]
                receipt = report["hybrid_best_part"]
                self.assertTrue(receipt["required_for_accepted_hybrid"])
                self.assertEqual(receipt["best_part"], "fir")
                self.assertAlmostEqual(receipt["signed_regression_db"], hybrid_score - 0.3)
                self.assertEqual(receipt["passed"], expected_exit == 0)


if __name__ == "__main__":
    unittest.main()
