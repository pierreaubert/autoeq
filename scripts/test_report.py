#!/usr/bin/env python3

import copy
import json
import math
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from scripts.src.acoustic_report import (
    band_mean,
    deepest_notch_db,
    early_late_ratio_db,
    early_reflection_figures,
    early_reflection_level_db,
    early_reflections_html,
    landmarks_table_html,
    level_compensation,
    operational_summaries_by_channel,
    pair_sum_difference,
    response_landmarks,
    room_t60_rows,
    summary_table_html,
    symmetric_groups,
    t60_flatness_pct,
    t60_flatness_tolerance_s,
    t60_rows,
    t60_table_html,
    tof_html,
    tof_table,
)
from scripts.src.data_extract import display_channel_entries
from scripts.src.dsp import split_driver_eq_plugins
from scripts.src.figures import (
    create_comparison_zoomed_figure,
    create_tof_figure,
    create_zoomed_figure,
)
from scripts.src.loaders import RoomEqData
from scripts.src.payload_binding import ALGORITHM, payload_digest
from scripts.src.report import (
    BUCKET_ACOUSTICS,
    BUCKET_DSP,
    BUCKET_PSYCHOACOUSTIC,
    _all_eq_filters_html,
    _channel_display_final_curve,
    _comparison_source_label,
    _crossover_config_html,
    _driver_eq_filters_html,
    _driver_shaping_summary_html,
    _gain_plugins_html,
    _has_redirected_bass_route,
    _mixed_phase_summary_html,
    _playback_status_html,
    _report_provenance,
    _roomeq_version,
    _workspace_roomeq_version,
    create_comparison_html_report,
    create_html_report,
)
from scripts.src.target_overlay import (
    band_mean_spl,
    shift_target_to_reference_band_mean,
)
from scripts.src.wasm_report import grid_figure
from scripts.test_capture_views import fixture as capture_verification_fixture
from scripts.test_figures import two_sub_overview_data


def _driver_eq_split_data():
    """Two-sub fixture with driver-level EQ on the first sub only."""
    data = copy.deepcopy(two_sub_overview_data())
    driver_eq = [
        {"filter_type": "peak", "freq": 55.0, "q": 2.0, "db_gain": 1.5},
        {"filter_type": "peak", "freq": 77.0, "q": 3.0, "db_gain": -2.5},
    ]
    first_sub = data["channels"]["LFE"]["drivers"][0]
    first_sub["plugins"].append(
        {"plugin_type": "eq", "parameters": {"filters": driver_eq}}
    )
    # A graph-owned stage must not leak into either filter list.
    first_sub["plugins"].append(
        {
            "plugin_type": "eq",
            "parameters": {
                "room_eq_stage": "route_owned",
                "filters": [
                    {
                        "filter_type": "peak",
                        "freq": 90.0,
                        "q": 1.0,
                        "db_gain": 9.0,
                    }
                ],
            },
        }
    )
    return data


class BoundCaptureReportTests(unittest.TestCase):
    def _bound_output(self):
        data = _driver_eq_split_data()
        graph = "candidate"
        data["correction_decisions"] = {
            "ledger_version": "1.0.0", "decisions": [],
            "payload_binding": {
                "algorithm": ALGORITHM, "graph_identity": graph,
                "sha256": payload_digest(data, graph),
            },
        }
        return data

    def test_matching_capture_verification_joins_saved_report(self):
        data = self._bound_output()
        capture = capture_verification_fixture()
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            result_path = directory / "result.json"
            capture_path = directory / "capture.json"
            html_path = directory / "report.html"
            result_path.write_text(json.dumps(data), encoding="utf-8")
            capture_path.write_text(json.dumps(capture), encoding="utf-8")
            command = [sys.executable, str(Path(__file__).resolve().parent.parent / "ui" / "display-roomeq"),
                       str(result_path), "-o", str(html_path),
                       "--capture-verification", str(capture_path)]
            run = subprocess.run(command, capture_output=True, text=True, check=False)
            self.assertEqual(run.returncode, 0, run.stderr)
            html = html_path.read_text(encoding="utf-8")
            self.assertIn("Capture acceptance diagnostics", html)
            self.assertIn("Matched IR evidence: synthetic_capture_pair", html)
            self.assertIn("&lt;source&gt; / seat", html)

    def test_mismatched_or_tampered_graph_cannot_display_capture_views(self):
        data = self._bound_output()
        capture = capture_verification_fixture()
        with tempfile.TemporaryDirectory() as directory:
            html_path = Path(directory) / "report.html"
            capture["graph_id"] = "other"
            create_html_report(data, html_path, capture_verification=capture)
            html = html_path.read_text(encoding="utf-8")
            self.assertIn("verification candidate graph does not match", html)
            self.assertNotIn("Matched IR evidence", html)
            capture["graph_id"] = "candidate"
            data["channels"]["LFE"]["plugins"].append(
                {"plugin_type": "gain", "parameters": {"gain_db": 1.0}}
            )
            create_html_report(data, html_path, capture_verification=capture)
            html = html_path.read_text(encoding="utf-8")
            self.assertIn("saved optimization graph binding is invalid", html)
            self.assertNotIn("Matched IR evidence", html)


class MixedPhaseReportTests(unittest.TestCase):
    def test_mixed_phase_and_temporal_masking_are_rendered_per_channel(self):
        metadata = {
            "mixed_phase_per_channel": {
                "L": {
                    "estimated_delay_ms": 4.25,
                    "fir_taps": 1024,
                    "residual_excess_phase_min_deg": -45.0,
                    "residual_excess_phase_max_deg": 18.5,
                    "residual_excess_phase_rms_deg": 23.75,
                }
            },
            "perceptual_metrics": {
                "fir_pre_ringing_audible_db": -46.0,
                "fir_post_ringing_audible_db": -52.0,
                "fir_temporal_masking_penalty": 0.125,
            },
        }
        channels = {
            "L": {
                "fir_temporal_masking": {
                    "main_index": 512,
                    "main_time_ms": 10.667,
                    "pre_ringing_peak_db": -31.5,
                    "post_ringing_peak_db": -36.25,
                    "pre_ringing_audible_db": -48.75,
                    "post_ringing_audible_db": -55.5,
                    "penalty": 0.1,
                }
            }
        }

        html = _mixed_phase_summary_html(metadata, channels)

        for expected in (
            "Mixed-Phase and FIR Timing",
            "4.250 ms",
            "1024",
            "-45.00° to +18.50°",
            "23.75°",
            "10.667 ms",
            "-31.50 / -48.75 dB",
            "-36.25 / -55.50 dB",
            "Worst audible pre/post ringing: -46.00 dB / -52.00 dB",
        ):
            self.assertIn(expected, html)

    def test_absent_mixed_phase_and_fir_metrics_emit_no_section(self):
        self.assertEqual(_mixed_phase_summary_html({}, {}), "")


class ComparisonSourceLabelTests(unittest.TestCase):
    def test_redirected_bass_source_is_not_labelled_as_physical_channel(self):
        data = {
            "metadata": {
                "bass_management": {
                    "routing_graph": {
                        "routes": [
                            {
                                "source_channel": "L",
                                "route_kind": "redirected_bass_lowpass_to_sub",
                            }
                        ]
                    }
                }
            }
        }

        self.assertTrue(_has_redirected_bass_route(data, "L"))
        self.assertFalse(_has_redirected_bass_route(data, "R"))
        self.assertEqual(
            _comparison_source_label("L", True),
            "Logical source L (physical L + redirected bass)",
        )
        self.assertEqual(_comparison_source_label("LFE", False), "Logical source LFE")


class ChannelDisplayCurveTests(unittest.TestCase):
    def test_lfe_tab_uses_logical_input_curve_not_aggregate_bass_bus(self):
        aggregate = {"freq": [20.0], "spl": [-60.0]}
        logical_lfe = {"freq": [20.0], "spl": [-48.0]}
        channel = {"final_curve": aggregate}

        self.assertIs(
            _channel_display_final_curve(
                "LFE", "LFE", channel, {"LFE": logical_lfe}
            ),
            logical_lfe,
        )

    def test_main_tab_keeps_main_only_curve_for_separate_routed_overlay(self):
        main_only = {"freq": [80.0], "spl": [-55.0]}
        routed_sum = {"freq": [80.0], "spl": [-52.0]}
        channel = {"final_curve": main_only}

        self.assertIs(
            _channel_display_final_curve("L", "LFE", channel, {"L": routed_sum}),
            main_only,
        )


class SubDriverTabTests(unittest.TestCase):
    def test_native_driver_acoustics_are_rendered_in_the_driver_tab(self):
        data = two_sub_overview_data()
        data["channels"]["LFE"]["drivers"][0]["measured_acoustics"] = {
            "sample_rate_hz": 3000.0,
            "pre_ir": {"time_ms": [0.0, 1.0, 2.0], "amplitude": [1.0, 0.5, 0.1]},
            "t60_octaves": {"basis": "measured_room_ir", "min_r2": 0.9, "bands": []},
        }
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "report.html"
            create_html_report(data, output, None)
            html = output.read_text(encoding="utf-8")
        self.assertIn("Independent measured driver IR: Left Sub", html)
        self.assertIn("native sample rate 3000.0 Hz", html)
        self.assertIn("T60 bands beyond Nyquist are marked unavailable", html)

    def test_two_subs_expand_into_per_sub_tabs(self):
        entries = display_channel_entries(two_sub_overview_data())

        self.assertEqual(
            [(entry["label"], entry["channel"], entry["driver"]) for entry in entries],
            [
                ("L", "L", None),
                ("R", "R", None),
                ("Left Sub", "LFE", 0),
                ("Right Sub", "LFE", 1),
            ],
        )

    def test_driver_names_fall_back_without_effective_config(self):
        data = two_sub_overview_data()
        del data["metadata"]["effective_config"]

        entries = display_channel_entries(data)

        self.assertEqual(
            [entry["label"] for entry in entries],
            ["L", "R", "Two subs_1", "Two subs_2"],
        )

    def test_channels_without_drivers_keep_single_tab(self):
        entries = display_channel_entries(
            {"channels": {"L": {}, "R": {}, "LFE": {}}}
        )

        self.assertEqual(
            [(entry["label"], entry["driver"]) for entry in entries],
            [("L", None), ("R", None), ("LFE", None)],
        )

    def test_driver_shaping_summary_lists_chain(self):
        data = two_sub_overview_data()

        first = _driver_shaping_summary_html(data, "LFE", 0)
        second = _driver_shaping_summary_html(data, "LFE", 1)

        for expected in (
            "driver gain +0.0 dB",
            "low-pass LR24 @ 80.0 Hz",
            "LFE route gain +14.0 dB",
            "LFE route low-pass @ 120.0 Hz",
        ):
            self.assertIn(expected, first)
        for expected in (
            "driver gain -4.0 dB",
            "low-pass LR24 @ 90.0 Hz",
            "LFE route gain +10.0 dB",
        ):
            self.assertIn(expected, second)
        self.assertEqual(_driver_shaping_summary_html(data, "LFE", 7), "")

    def test_driver_shaping_summary_preserves_sankey_hover_alignment(self):
        # The old Sankey hover showed each driver's plugin alignment
        # ("gain: +6.00 dB", "delay: 2.500 ms"); the per-driver table keeps it.
        data = {
            "channels": {
                "LFE": {
                    "drivers": [
                        {
                            "name": "sub_1",
                            "initial_curve": {"freq": [20.0, 40.0],
                                              "spl": [70.0, 70.0]},
                            "plugins": [
                                {"plugin_type": "gain",
                                 "parameters": {"gain_db": 6.0}},
                                {"plugin_type": "delay",
                                 "parameters": {"delay_ms": 2.5}},
                            ],
                        },
                    ],
                },
            },
        }

        summary = _driver_shaping_summary_html(data, "LFE", 0)

        self.assertIn("driver gain +6.0 dB", summary)
        self.assertIn("delay 2.500 ms", summary)

    def test_html_report_renders_per_sub_tabs(self):
        data = two_sub_overview_data()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "report.html"
            create_html_report(data, output, None)
            html = output.read_text(encoding="utf-8")

        # Per-channel tabs are shell tabs now (no tab buttons in the file):
        # sections carry the tab label in the payload.
        self.assertIn('"tab": "Left Sub"', html)
        self.assertIn('"tab": "Right Sub"', html)
        self.assertNotIn('"tab": "LFE"', html)
        self.assertIn("<h2>Channel: Left Sub</h2>", html)
        self.assertIn("Sub DSP Chain", html)
        self.assertIn("EQ: Left Sub", html)
        self.assertIn("EQ: Right Sub", html)


class DriverEqFilterSplitTests(unittest.TestCase):
    def test_split_counts_driver_vs_shared_filters(self):
        data = _driver_eq_split_data()

        split = split_driver_eq_plugins(data, "LFE", 0)
        assert split is not None
        driver_plugins, shared_plugins = split

        driver_count = sum(
            len(plugin["parameters"]["filters"]) for plugin in driver_plugins
        )
        shared_count = sum(
            len(plugin["parameters"]["filters"]) for plugin in shared_plugins
        )
        self.assertEqual(driver_count, 2)
        self.assertEqual(shared_count, 1)

    def test_split_excludes_route_owned_stages(self):
        data = _driver_eq_split_data()

        split = split_driver_eq_plugins(data, "LFE", 0)
        assert split is not None
        driver_plugins, _ = split

        freqs = [
            filt["freq"]
            for plugin in driver_plugins
            for filt in plugin["parameters"]["filters"]
        ]
        self.assertNotIn(90.0, freqs)

    def test_split_rejects_invalid_driver(self):
        data = _driver_eq_split_data()

        self.assertIsNone(split_driver_eq_plugins(data, "LFE", 7))
        self.assertIsNone(split_driver_eq_plugins(data, "L", 0))
        self.assertEqual(_driver_eq_filters_html(data, "LFE", 7, "eq_9"), "")

    def test_driver_tab_renders_per_origin_inner_tabs(self):
        data = _driver_eq_split_data()

        html = _driver_eq_filters_html(data, "LFE", 0, "eq_2")

        self.assertIn("Driver: Left Sub (2)", html)
        self.assertIn("Shared channel LFE (1)", html)
        self.assertIn("eq_2_drv", html)
        self.assertIn("eq_2_sh", html)
        self.assertIn("openEqTab", html)
        # Numbering restarts in each tab: Filter 1 appears once per tab.
        self.assertEqual(html.count("Filter 1:"), 2)
        self.assertEqual(html.count("Filter 2:"), 1)
        # The driver tab panel precedes the shared channel panel.
        self.assertLess(html.index("55.0 Hz"), html.index("40.0 Hz"))

    def test_single_origin_renders_without_inner_tabs(self):
        data = _driver_eq_split_data()

        html = _driver_eq_filters_html(data, "LFE", 1, "eq_3")

        self.assertIn("Shared channel LFE (1)", html)
        self.assertNotIn("eq-tab-btn", html)
        self.assertIn("40.0 Hz", html)

    def test_no_eq_renders_no_section(self):
        data = two_sub_overview_data()
        data["channels"]["LFE"]["plugins"] = []

        self.assertEqual(_driver_eq_filters_html(data, "LFE", 0, "eq_3"), "")

    def test_html_report_splits_driver_tab_filters(self):
        data = _driver_eq_split_data()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "report.html"
            create_html_report(data, output, None)
            html = output.read_text(encoding="utf-8")

        self.assertIn("Driver: Left Sub (2)", html)
        self.assertIn("Shared channel LFE (1)", html)
        # Nested driver-tab switcher lives in the shell; the content keeps
        # the tab hooks the shell wires up.
        self.assertIn("openEqTab", html)
        self.assertIn("eq-tab-btn", html)


class SummarySectionTests(unittest.TestCase):
    def test_runtime_limiter_status_discloses_small_signal_response(self):
        html = _playback_status_html({
            "correction_acceptance": {"outcome": "accepted", "accepted": True, "decision": "accepted"},
            "stage_outcomes": [{"checks": [{"id": "runtime_limiter_physical_output:Sub1"}]}],
        })
        self.assertIn("Runtime limiter required", html)
        self.assertIn("small-signal", html)
        self.assertIn("Do not bypass", html)
        self.assertIn("Sub1", html)

    def test_gain_summary_exposes_safety_and_driver_trims_without_eq(self):
        def gain(value, stage, label):
            return {"plugin_type": "gain", "parameters": {
                "gain_db": value, "room_eq_stage": stage, "label": label}}

        data = {"global_plugins": [gain(-1.0, "pre_route", "global trim")], "channels": {
            "L": {"plugins": [gain(-13.11163, "pre_route", "final_electrical_headroom"),
                              gain(-9.42897, "post_route", "level alignment")]},
            "LFE": {"drivers": [{"name": "woofer<script>", "plugins": [
                gain(-3.56015, "post_route", "safety<script>")]}]},
        }}
        html = _gain_plugins_html(data)
        for expected in ["Global", "Channel L", "-13.112 dB", "-9.429 dB", "-3.560 dB",
                         "pre_route", "post_route", "final_electrical_headroom",
                         "woofer&lt;script&gt;", "safety&lt;script&gt;", "not a summed"]:
            self.assertIn(expected, html)
        self.assertNotIn("<script>", html)
        self.assertLess(html.index("-13.112 dB"), html.index("-9.429 dB"))

    def test_gain_summary_empty_without_gain_plugins(self):
        self.assertEqual(_gain_plugins_html({"channels": {"L": {"plugins": []}}}), "")

    def test_comparison_report_discloses_each_playback_verdict(self):
        rejected = _driver_eq_split_data()
        rejected.setdefault("metadata", {})["correction_acceptance"] = {"outcome": "rejected"}
        conditional = copy.deepcopy(rejected)
        conditional["metadata"]["correction_acceptance"] = {
            "outcome": "accepted", "accepted": True, "decision": "accepted",
        }
        conditional["metadata"]["effective_config"] = {
            "optimizer": {"finalization": {"default_input_peak": 0.125}}
        }
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "comparison.html"
            create_comparison_html_report([("strict", rejected), ("conditional", conditional)], output)
            html = output.read_text(encoding="utf-8")
        self.assertIn("strict — Saved DSP playback eligibility: rejected", html)
        self.assertIn("conditional — Saved DSP playback eligibility: accepted", html)
        self.assertIn("Not approved for playback", html)
        self.assertIn("Conditional on enforced input-peak ceilings", html)
        self.assertIn("Why this correction? — strict", html)
        self.assertIn("Why this correction? — conditional", html)
        self.assertLess(html.index("Why this correction? — conditional"), html.index("<h2>Summary</h2>"))

    def test_playback_status_rejects_missing_or_failed_verdicts(self):
        for outcome in [None, "rejected", "insufficient_evidence", "accepted"]:
            with self.subTest(outcome=outcome):
                html = _playback_status_html({"correction_acceptance": {"outcome": outcome}})
                self.assertIn("Not approved for playback", html)

    def test_playback_status_discloses_reduced_input_contract(self):
        metadata = {
            "correction_acceptance": {"outcome": "accepted", "accepted": True, "decision": "accepted"},
            "effective_config": {"optimizer": {"finalization": {
                "default_input_peak": 10 ** (-18 / 20),
                "input_peak_limits": {"L<script>": 0.5},
            }}},
        }
        from scripts.src.payload_binding import ALGORITHM, payload_digest
        data = {"metadata": metadata, "channels": {}}
        data["correction_decisions"] = {"ledger_version": "1.0.0", "decisions": [],
            "payload_binding": {"algorithm": ALGORITHM, "graph_identity": "test-graph",
                                "sha256": payload_digest(data, "test-graph")}}
        html = _playback_status_html(metadata, "IIR<script>", data=data)
        self.assertIn("default: -18.00 dBFS", html)
        self.assertIn("L&lt;script&gt;: -6.02 dBFS", html)
        self.assertIn("IIR&lt;script&gt;", html)
        self.assertNotIn("<script>", html)
        self.assertIn("full-scale input safety is not established", html)
        self.assertNotIn("Not approved for playback", html)

    def test_all_eq_filters_lists_every_origin(self):
        html = _all_eq_filters_html(_driver_eq_split_data())

        self.assertIn("<h2>All EQ Filters</h2>", html)
        self.assertIn("Driver: Left Sub (2)", html)
        self.assertIn("Shared channel LFE (1)", html)
        self.assertIn("<td>55.0 Hz</td>", html)
        self.assertIn("<td>40.0 Hz</td>", html)

    def test_all_eq_filters_skips_channels_without_eq(self):
        data = two_sub_overview_data()
        data["channels"]["LFE"]["plugins"] = []

        html = _all_eq_filters_html(data)

        self.assertEqual(html, "")

    def test_all_eq_filters_renders_channel_buttonset(self):
        html = _all_eq_filters_html(_driver_eq_split_data())

        self.assertIn('<div class="eq-tabs">', html)
        self.assertIn("openEqTab(event, 'alleq_0')", html)
        self.assertIn('id="alleq_0" class="eq-tab-panel active"', html)
        self.assertIn('id="alleq_1" class="eq-tab-panel"', html)
        # Every channel panel keeps its filter tables.
        self.assertIn("Driver: Left Sub (2)", html)
        self.assertIn("Shared channel LFE (1)", html)

    def test_operational_share_derives_from_legacy_final_decisions(self):
        def decision(channel, status, decision_id, supersedes=()):
            return {
                "decision_id": decision_id,
                "stage": "final",
                "action": "equalize",
                "status": status,
                "logical_input": channel,
                "physical_output": channel,
                "final_graph_identity": "graph-1",
                "supersedes_ids": list(supersedes),
            }

        data = {"correction_decisions": {"decisions": [
            decision("L", "applied", "d-1"),
            decision("L", "unresolved", "d-2"),
            decision("R", "constrained", "d-3"),
            # Provisional history never enters the scope.
            dict(decision("R", "applied", "d-4"), stage="provisional"),
            # A superseded record leaves both counts.
            decision("R", "applied", "d-5", supersedes=["d-3"]),
        ]}}
        summaries = operational_summaries_by_channel(data)

        self.assertEqual(summaries["L"], (50.0, 2, 1))
        self.assertEqual(summaries["R"], (100.0, 1, 1))

    def test_operational_share_stays_pending_without_final_scope(self):
        data = {"correction_decisions": {"decisions": [{
            "decision_id": "d-1", "stage": "provisional", "action": "equalize",
            "status": "unresolved", "physical_output": "L",
            "final_graph_identity": "",
        }]}}
        self.assertEqual(operational_summaries_by_channel(data), {})
        html = summary_table_html({"channels": {"L": {}}, **data})
        self.assertIn("Not assessed", html)

    def test_level_residual_predicts_from_final_curve(self):
        init_l, init_r = _stereo_curves([1.0, 1.0, 1.0, 1.0], [-1.0, -1.0, -1.0, -1.0])
        rows = level_compensation({
            "L": {"initial_curve": init_l,
                  "final_curve": {"freq": [100.0, 1000.0, 2000.0, 3000.0],
                                  "spl": [0.0, 0.0, 0.0, 0.0]}},
            "R": {"initial_curve": init_r},
        })
        by_name = {r["speaker"]: r for r in rows}
        # Without R's predicted curve there is no predicted reference, so the
        # trim falls back to the pre-DSP proposal (-2.0) and the residual is
        # the landing level with that trim: 0.0 - 2.0 - (-1.0).
        self.assertAlmostEqual(by_name["L"]["residual_db"], -1.0)
        self.assertIsNone(by_name["R"]["residual_db"])
        self.assertIsNone(by_name["L"]["balance_db"])

    def test_level_compensation_is_remaining_trim_against_predicted_curves(self):
        # L measures 2 dB louder than R before DSP; the delivered chain
        # already aligned them in-chain, so no downstream trim remains.
        init_l, init_r = _stereo_curves([83.0, 83.0, 83.0, 83.0],
                                        [81.0, 81.0, 81.0, 81.0])
        post_l, post_r = _stereo_curves([78.0, 78.0, 78.0, 78.0],
                                        [78.0, 78.0, 78.0, 78.0])
        rows = level_compensation({
            "L": {"initial_curve": init_l, "final_curve": post_l},
            "R": {"initial_curve": init_r, "final_curve": post_r},
        })
        by_name = {r["speaker"]: r for r in rows}
        self.assertAlmostEqual(by_name["L"]["comp_db"], 0.0)
        self.assertAlmostEqual(by_name["R"]["comp_db"], 0.0)
        self.assertAlmostEqual(by_name["L"]["balance_db"], 0.0)
        self.assertAlmostEqual(by_name["L"]["residual_db"],
                               by_name["R"]["residual_db"])

    def test_level_compensation_closes_predicted_spread(self):
        # The delivered chain left 0.5 dB of predicted spread: the trim
        # covers exactly that remainder and both monitors land together.
        init_l, init_r = _stereo_curves([83.5, 83.5, 83.5, 83.5],
                                        [83.0, 83.0, 83.0, 83.0])
        post_l, post_r = _stereo_curves([78.5, 78.5, 78.5, 78.5],
                                        [78.0, 78.0, 78.0, 78.0])
        rows = level_compensation({
            "L": {"initial_curve": init_l, "final_curve": post_l},
            "R": {"initial_curve": init_r, "final_curve": post_r},
        })
        by_name = {r["speaker"]: r for r in rows}
        self.assertAlmostEqual(by_name["L"]["comp_db"], -0.5)
        self.assertAlmostEqual(by_name["R"]["comp_db"], 0.0)
        self.assertAlmostEqual(by_name["L"]["balance_db"], 0.0)
        self.assertAlmostEqual(by_name["R"]["balance_db"], 0.0)
        self.assertAlmostEqual(by_name["L"]["residual_db"],
                               by_name["R"]["residual_db"])
        self.assertAlmostEqual(by_name["L"]["residual_db"], -5.0)

    def test_landmarks_report_peaks_and_lf_extension(self):
        freq = [20.0, 30.0, 40.0, 60.0, 100.0, 200.0, 400.0, 1000.0, 2000.0]
        spl = [-20.0, -8.0, -2.0, 0.0, 0.0, 0.0, 0.0, 8.0, 0.0]
        marks = response_landmarks(
            {"initial_curve": {"freq": list(freq), "spl": list(spl)}})
        assert marks is not None
        self.assertEqual(marks["lf_extension_hz"], 30.0)
        self.assertTrue(marks["peaks"])
        html = landmarks_table_html(
            {"channels": {"L": {"initial_curve": {"freq": list(freq),
                                                  "spl": list(spl)}}}})
        self.assertIn("Frequency landmarks", html)
        self.assertIn("30 Hz", html)
        self.assertIsNone(response_landmarks({}))

    def test_crossover_config_lists_plugins_and_routes(self):
        html = _crossover_config_html(two_sub_overview_data())

        self.assertIn("<h2>Crossover Configuration</h2>", html)
        self.assertIn("Deployed Crossover DSP", html)
        self.assertIn("Driver Left Sub", html)
        self.assertIn("LR24", html)
        self.assertIn("80.0 Hz", html)
        self.assertIn("Routing-Graph Crossovers", html)
        self.assertIn("120.0 Hz", html)

    def test_crossover_config_empty_without_crossovers(self):
        self.assertEqual(_crossover_config_html({"channels": {"L": {}}}), "")
        self.assertEqual(_crossover_config_html({}), "")

    def test_k4_final_decisions_render_ahead_of_summary_in_both_modes(self):
        data = _driver_eq_split_data()
        data["correction_decisions"] = {"ledger_version": "1.0.0", "decisions": [{
            "decision_id": "d-1", "ledger_version": "1.0.0", "stage": "final",
            "logical_input": "LFE", "physical_output": "Sub1",
            "measurement_refs": ["meas-1"], "seat_refs": ["seat-a"],
            "frequency_band_hz": [40.0, 120.0], "action": "equalize",
            "status": "applied", "reason_codes": ["within_limits"],
            "observed": [{"name": "post_p95_abs_residual_db", "value": 3.0, "unit": "db"}],
            "limits": [{"name": "max_post_p95_abs_residual_db", "value": 6.0, "unit": "db"}],
            "evidence_refs": ["ev-1"], "confidence": "moderate",
            "final_graph_identity": "graph-final-1",
        }]}
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "report.html"
            create_html_report(data, output, None)
            html = output.read_text(encoding="utf-8")
        self.assertIn("Recorded final correction decisions", html)
        self.assertLess(html.index("Recorded final correction decisions"),
                        html.index('"subhead": "Summary"'))
        self.assertLess(html.index("Recorded final correction decisions"),
                        html.index('"title": "Combined Overview — Before EQ"'))
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "comparison.html"
            create_comparison_html_report([("iir", data), ("fir", data)], output)
            html = output.read_text(encoding="utf-8")
        self.assertEqual(html.count("Recorded final correction decisions"), 2)
        self.assertLess(html.index("Recorded final correction decisions"),
                        html.index("<h2>Summary</h2>"))

    def test_html_report_contains_summary_sections(self):
        data = _driver_eq_split_data()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "report.html"
            create_html_report(data, output, None)
            html = output.read_text(encoding="utf-8")

        self.assertIn("EQ Filters", html)
        self.assertNotIn("<h2>All EQ Filters</h2>", html)
        self.assertIn("<h2>Crossover Configuration</h2>", html)
        self.assertIn("Not approved for playback", html)
        self.assertLess(html.index("Why this correction?"), html.index('"subhead": "Summary"'))
        self.assertLess(html.index("Why this correction?"), html.index('"title": "Combined Overview — Before EQ"'))
        # Summaries precede the per-channel tabs (first non-null section tab).
        self.assertLess(
            html.index("<h2>Crossover Configuration</h2>"),
            html.index('"tab": "'),
        )


def _stereo_curves(spl_l, spl_r, freq=None):
    freq = freq or [100.0, 1000.0, 2000.0, 3000.0]
    return (
        {"freq": list(freq), "spl": list(spl_l)},
        {"freq": list(freq), "spl": list(spl_r)},
    )


def report_payload(html):
    """Decode the embedded report payload (HTML inside is JSON-escaped on disk)."""
    match = re.search(
        r'<script id="report-payload" type="application/json">(.*?)</script>',
        html, re.DOTALL,
    )
    assert match is not None, "report payload script tag missing"
    return json.loads(match.group(1))


class AcousticReportTests(unittest.TestCase):
    def test_itu_summary_column_counts_eight_bands_and_requires_volume(self):
        from scripts.src.acoustic_report import t60_itu_cell, t60_itu_pct
        bands = [{"centre_hz": hz, "t60_s": 0.25, "fit_range": "T30",
                  "r2": 0.96, "valid": True, "reason": ""}
                 for hz in [63, 125, 250, 500, 1000, 2000, 4000, 8000, 16000]]
        bands[-1]["t60_s"] = 10.0  # Outside the recommendation's coverage.
        data = {"metadata": {"effective_config": {"recording_config": {
                    "room_dimensions": {"length": 5, "width": 4, "height": 5}}}},
                "channels": {"L": {"t60_octaves": {"basis": "measured_room_ir",
                            "min_r2": 0.9, "bands": bands}}}}
        self.assertEqual(t60_itu_pct(data, "L"), 100.0)
        bands[0]["t60_s"] = 0.01  # No lower bound at 63 Hz.
        self.assertEqual(t60_itu_pct(data, "L"), 100.0)
        bands[1]["t60_s"] = 0.50
        self.assertEqual(t60_itu_pct(data, "L"), 87.5)
        for band in bands:
            band["t60_s"] = 1.0
        self.assertEqual(t60_itu_pct(data, "L"), 0.0)
        self.assertIn("0/8 octave centers", t60_itu_cell(data, "L"))
        self.assertIn("#e74c3c", t60_itu_cell(data, "L"))  # 0% -> red.
        for band in bands[:7]:
            band["t60_s"] = 0.25
        bands[7]["t60_s"] = 1.0  # 7/8 inside -> 87.5% -> green.
        self.assertIn("#2ecc71", t60_itu_cell(data, "L"))
        bands[5]["t60_s"] = 1.0
        bands[6]["t60_s"] = 1.0  # 5/8 inside -> 62.5% -> yellow.
        self.assertIn("#f1c40f", t60_itu_cell(data, "L"))
        bands[5]["t60_s"] = 0.25
        bands[6]["t60_s"] = 0.25
        bands[7]["t60_s"] = 0.25
        self.assertIn("T60 within ITU recommendation (%)", summary_table_html(data))
        bands[2].update(t60_s=None, valid=False, fit_range="None", reason="poor fit")
        self.assertIsNone(t60_itu_pct(data, "L"))
        bands[2].update(t60_s=0.25, valid=True, fit_range="T30", reason="")
        data["metadata"] = {}
        self.assertIsNone(t60_itu_pct(data, "L"))
        self.assertIn("Not assessed", t60_itu_cell(data, "L"))

    def test_itu_t60_envelope_uses_volume_and_exact_frequency_breaks(self):
        from scripts.src.acoustic_report import t60_itu_reference
        from scripts.src.figures import create_t60_octaves_figure
        data = {"metadata": {"effective_config": {"recording_config": {
            "room_dimensions": {"length": 5, "width": 4, "height": 5}}}}}
        reference = t60_itu_reference(data)
        self.assertAlmostEqual(reference["tm"], 0.25)
        self.assertIn("100 m³", reference["source"])
        plot = create_t60_octaves_figure("Room", [{"centre_hz": 1000, "t60_s": 0.3}],
                                         itu_reference=reference)["figure"]
        upper, lower = plot["series"][1:]
        self.assertEqual(upper["x"], [63, 200, 4000, 4000, 8000])
        self.assertEqual(lower["x"], [100, 4000, 4000, 8000])
        for actual, expected in zip(upper["y"], [0.55, 0.30, 0.30, 0.35, 0.35]):
            self.assertAlmostEqual(actual, expected)
        for actual, expected in zip(lower["y"], [0.20, 0.20, 0.15, 0.15]):
            self.assertAlmostEqual(actual, expected)

    def test_itu_t60_missing_volume_uses_only_complete_measured_midbands(self):
        from scripts.src.acoustic_report import t60_itu_note, t60_itu_reference
        bands = [{"centre_hz": hz, "t60_s": 0.3 if 200 <= hz <= 4000 else 2.0,
                  "fit_range": "T30", "r2": 0.96, "valid": True, "reason": ""}
                 for hz in [63, 125, 250, 500, 1000, 2000, 4000, 8000, 16000]]
        data = {"channels": {"L": {"t60_octaves": {
            "basis": "measured_room_ir", "min_r2": 0.9, "bands": bands}}}}
        reference = t60_itu_reference(data)
        self.assertAlmostEqual(reference["tm"], 0.3)
        self.assertIn("volume unavailable", reference["source"])
        self.assertIn("relative decay shape only", t60_itu_note(reference))
        bands[3].update(t60_s=None, valid=False, reason="insufficient decay", fit_range="None")
        self.assertIsNone(t60_itu_reference(data))
        self.assertIn("limits unavailable", t60_itu_note(None))

    def test_symmetric_pair_shows_three_traces_with_fifty_db_view(self):
        from scripts.src.figures import create_symmetric_pair_figure
        plot = create_symmetric_pair_figure(
            "L+R", [100.0, 1000.0], [86.0, 87.0], [86.0, 80.0])["figure"]
        self.assertEqual(plot["y"]["max"] - plot["y"]["min"], 50.0)
        self.assertGreater(plot["y"]["max"], 87.0)
        self.assertEqual(
            [s["name"] for s in plot["series"]],
            ["L+R absolute sum", "L+R complex sum", "L+R sum difference"],
        )
        self.assertEqual(plot["series"][2]["y"], [0.0, 7.0])
        self.assertEqual(plot["series"][2]["y_axis"], 1)
        self.assertEqual(plot["y2"]["label"], "Sum difference (dB)")
        self.assertEqual(plot["y2"]["max"], 40.0)
        # Deep cancellations pin to the 40 dB display ceiling.
        deep = create_symmetric_pair_figure(
            "L+R", [100.0], [86.0], [-200.0])["figure"]
        self.assertEqual(deep["series"][2]["y"], [40.0])
        # Without a complex sum only the absolute sum draws.
        solo = create_symmetric_pair_figure(
            "L+R", [100.0, 1000.0], [86.0, 87.0], None)["figure"]
        self.assertEqual(len(solo["series"]), 1)
        self.assertIsNone(solo["y2"])
        self.assertIn("absolute sum only", solo["title"])
        empty = create_symmetric_pair_figure("L+R", [], [], None)["figure"]
        self.assertEqual(empty["y"]["max"] - empty["y"]["min"], 50.0)

    def test_summary_reflection_level_requires_measured_and_valid_emitted_events(self):
        report = {"basis": "measured_room_ir",
                  "method": "bandlimited_early_reflection_table_v1",
                  "band_hz": [1000.0, 8000.0], "threshold_dbfs": -15.0,
                  "direct_reference": "filtered direct peak",
                  "pre": [{"gain_dbfs": -12.0, "time_ms": 5.0,
                           "distance_cm": 171.5, "first_dip_hz": 100.0,
                           "ripple_db": 4.0}],
                  "post": [{"gain_dbfs": -8.0, "time_ms": 3.0,
                            "distance_cm": 102.9, "first_dip_hz": 500.0/3.0,
                            "ripple_db": 7.3},
                           {"gain_dbfs": -11.0, "time_ms": 6.0,
                            "distance_cm": 205.8, "first_dip_hz": 500.0/6.0,
                            "ripple_db": 5.0}]}
        channel = {"early_reflections": report,
                   "final_curve": {"freq": [50.0, 100.0, 200.0, 400.0],
                                   "spl": [60.0, 70.0, 65.0, 68.0]}}
        self.assertEqual(early_reflection_level_db(channel), (-8.0, False))
        html = summary_table_html({"channels": {"L": channel}})
        self.assertIn("-8.0</td>", html)
        self.assertNotIn('needs roomeq field: early_reflections', html)
        details = early_reflections_html(channel, "<L>")
        self.assertIn("&lt;L&gt;", details)
        self.assertIn("First dip (Hz)", details)
        self.assertNotIn('<svg ', details)
        plots = early_reflection_figures(channel, "L", tab="L")
        self.assertEqual(len(plots), 3)
        self.assertTrue(all(p["kind"] == "figure" and p["tab"] == "L" for p in plots))
        self.assertEqual(plots[0]["figure"]["x"]["label"], "Time after direct peak (ms)")
        self.assertEqual(plots[-1]["figure"]["x"]["scale"], "log")
        self.assertTrue(plots[-1]["figure"]["legend"])
        self.assertEqual(len(plots[-1]["figure"]["series"]), 3)
        report["post"] = []
        self.assertIn("No candidates were emitted", early_reflections_html(channel, "L"))
        self.assertEqual(len(early_reflection_figures(channel, "L")), 2)
        report["pre"] = []
        self.assertEqual(len(early_reflection_figures(channel, "L")), 1)
        self.assertNotIn('<table', early_reflections_html(channel, "L"))
        self.assertEqual(early_reflection_level_db(channel), (-15.0, True))
        self.assertIn("≤ -15.0", summary_table_html({"channels": {"L": channel}}))
        report["basis"] = "reconstructed_transfer_ir"
        self.assertIsNone(early_reflection_level_db(channel))
        report["basis"] = "measured_room_ir"
        report["post"] = [{"gain_dbfs": -8.0, "time_ms": 16.0,
                           "distance_cm": 548.8, "first_dip_hz": 31.25,
                           "ripple_db": 7.3}]
        self.assertIsNone(early_reflection_level_db(channel))
        self.assertIn("pending roomeq field early_reflections",
                      early_reflections_html(channel, "L"))

    def test_summary_uses_emitted_shared_reference_early_late_ratio_only_with_full_band(self):
        freq = [1000.0, 2000.0, 4000.0, 8000.0]
        report = {"method": "incoherent_band_energy", "reference": "full_peak_band",
                  "smoothing": "third_octave", "split_ms": 20.0,
                  "full": {"freq": freq, "spl": [0.0] * 4},
                  "early": {"freq": freq, "spl": [-1.0] * 4},
                  "late": {"freq": freq, "spl": [-7.0] * 4}}
        channel = {"early_late_curves": report}
        self.assertAlmostEqual(early_late_ratio_db(channel), 6.0)
        html = summary_table_html({"channels": {"L": channel}})
        self.assertIn("+6.0</td>", html)
        self.assertNotIn('needs roomeq field: early_late_curves', html)
        report["reference"] = "each_segment_peak"
        self.assertIsNone(early_late_ratio_db(channel))
        self.assertIn('needs roomeq field: early_late_curves',
                      summary_table_html({"channels": {"L": channel}}))
        report["reference"] = "full_peak_band"
        report["full"]["freq"] = [2000.0, 3000.0, 4000.0, 8000.0]
        self.assertIsNone(early_late_ratio_db(channel))

    def test_summary_operational_response_uses_emitted_ledger_share(self):
        data = {"channels": {"L": {}, "R": {}, "C": {}},
                "correction_decisions": {"channel_summaries": [
                    {"channel": "L", "operational_response_pct": 100.0,
                     "decided_equalize": 2, "delivered_equalize": 2},
                    {"channel": "R", "operational_response_pct": 50.0,
                     "decided_equalize": 2, "delivered_equalize": 1}]}}
        html = summary_table_html(data)
        self.assertIn(">100.0</td>", html)
        self.assertIn(">50.0</td>", html)
        self.assertIn("background:#2ecc71", html)
        self.assertIn("background:#e74c3c", html)
        self.assertIn("1/2 final EQ scope delivered", html)
        # The channel without decided scope stays pending grey, never green.
        self.assertIn("Not assessed", html)
        # Malformed and nonfinite entries cannot promote a cell.
        data["correction_decisions"]["channel_summaries"] = [
            {"channel": "L", "operational_response_pct": float("nan"),
             "decided_equalize": 1, "delivered_equalize": 1},
            {"channel": "L", "operational_response_pct": "high",
             "decided_equalize": 1, "delivered_equalize": 1},
            {"channel": "", "operational_response_pct": 100.0,
             "decided_equalize": 1, "delivered_equalize": 1},
            "not-a-dict",
        ]
        html = summary_table_html(data)
        self.assertNotIn(">100.0</td>", html)
        self.assertIn("Not assessed", html)
        # No ledger at all: every row pending.
        html = summary_table_html({"channels": {"L": {}}})
        self.assertIn("Not assessed", html)

    def test_summary_operational_response_threshold_boundaries(self):
        for pct, color in [(100.0, "#2ecc71"), (90.1, "#2ecc71"),
                           (90.0, "#f1c40f"), (85.0, "#f1c40f"),
                           (80.0, "#f1c40f"), (79.9, "#e74c3c"),
                           (0.0, "#e74c3c")]:
            data = {"channels": {"L": {}},
                    "correction_decisions": {"channel_summaries": [
                        {"channel": "L", "operational_response_pct": pct,
                         "decided_equalize": 4, "delivered_equalize": 1}]}}
            html = summary_table_html(data)
            self.assertIn(f"background:{color}", html, f"pct={pct}")
            self.assertIn(f">{pct:.1f}</td>", html, f"pct={pct}")

    def test_t60_requires_measured_basis_and_preserves_invalid_band_reason(self):
        from scripts.src.acoustic_report import room_t60_table_html, t60_flatness_window
        from scripts.src.figures import create_t60_octaves_figure
        centers = [63, 125, 250, 500, 1000, 2000, 4000, 8000, 16000]
        bands = [{"centre_hz": hz, "t60_s": 0.4, "fit_range": "T30",
                  "r2": 0.96, "valid": True, "reason": ""} for hz in centers]
        bands[-1] = {"centre_hz": 16000, "t60_s": None, "fit_range": "None",
                     "r2": 0.0, "valid": False, "reason": "above-nyquist <unsafe>"}
        channel = {"t60_octaves": {"basis": "measured_room_ir", "min_r2": 0.9,
                                    "bands": bands}}
        rows = t60_rows(channel)
        self.assertEqual(len(rows), 9)
        self.assertIsNone(rows[-1]["t60_s"])
        mean = room_t60_rows({"channels": {"L": channel, "R": copy.deepcopy(channel)}})
        self.assertEqual(mean[0]["speaker_count"], 2)
        self.assertAlmostEqual(mean[0]["t60_s"], 0.4)
        self.assertEqual(mean[-1]["speaker_count"], 0)
        self.assertIn("above-nyquist &lt;unsafe&gt;", t60_table_html(channel))
        data = {"channels": {"L": channel}}
        self.assertIn("Room mean (s)", room_t60_table_html(data))
        self.assertIn("L T60 (s)", room_t60_table_html(data))
        self.assertIsNone(t60_flatness_window(data))
        plot = create_t60_octaves_figure("L", rows, window=(0.35, 0.45))["figure"]
        self.assertEqual(len(plot["series"]), 3)
        steps = plot["series"][0]
        self.assertEqual(len(steps["x"]), 18)
        self.assertEqual(steps["x"][1], steps["x"][2])
        self.assertEqual(steps["y"][:2], [0.4, 0.4])
        self.assertEqual(steps["y"][-2:], [None, None])
        self.assertEqual(plot["x"]["max"], 20000)
        channel["t60_octaves"]["basis"] = "reconstructed_transfer_ir"
        self.assertIsNone(t60_rows(channel))
        channel["t60_octaves"]["basis"] = "measured_room_ir"
        bands[0]["r2"] = 0.8
        self.assertIsNone(t60_rows(channel))

    def test_t60_flatness_uses_declared_tolerance_or_bs1116_default(self):
        centers = [63, 125, 250, 500, 1000, 2000, 4000, 8000, 16000]
        bands = [{"centre_hz": hz, "t60_s": 0.4, "fit_range": "T30",
                  "r2": 0.96, "valid": True, "reason": ""} for hz in centers]
        report = {"basis": "measured_room_ir", "min_r2": 0.9, "bands": bands}
        channel = {"t60_octaves": report}
        data = {"channels": {"L": channel}, "metadata": {}}
        # No declared tolerance: the BS.1116 midband default applies, labeled.
        self.assertEqual(t60_flatness_tolerance_s(data), (0.05, False))
        self.assertEqual(t60_flatness_pct(data, "L"), 100.0)
        from scripts.src.acoustic_report import t60_flatness_window
        lower, upper = t60_flatness_window(data)
        self.assertAlmostEqual(lower, 0.35)
        self.assertAlmostEqual(upper, 0.45)
        self.assertIn('within default ±0.05 s (ITU-R BS.1116-2',
                      summary_table_html(data))
        data["metadata"]["t60_flatness_tolerance_s"] = 0.05
        self.assertEqual(t60_flatness_tolerance_s(data), (0.05, True))
        self.assertEqual(t60_flatness_pct(data, "L"), 100.0)
        self.assertIn('within declared ±0.05 s of the complete-channel room mean">100.0',
                      summary_table_html(data))
        other = copy.deepcopy(channel)
        for row in other["t60_octaves"]["bands"]:
            row["t60_s"] = 0.8
        data["channels"]["R"] = other
        self.assertEqual(t60_flatness_pct(data, "L"), 0.0)
        self.assertIn('background:#e74c3c33', summary_table_html(data))
        del data["channels"]["R"]
        bands[0]["t60_s"] = 0.6
        self.assertAlmostEqual(t60_flatness_pct(data, "L"), 800.0 / 9.0)
        bands[0] = {"centre_hz": 63, "t60_s": None, "fit_range": None,
                    "r2": 0.0, "valid": False, "reason": "noise floor"}
        self.assertIsNone(t60_flatness_pct(data, "L"))
        self.assertIn('needs roomeq field: nine valid measured t60_octaves fits',
                      summary_table_html(data))
        bands[0] = {"centre_hz": 63, "t60_s": 0.4, "fit_range": "T30",
                    "r2": 0.96, "valid": True, "reason": ""}
        # A malformed explicit value falls back to the labeled default rather
        # than voiding the cell (roomeq rejects such configs before any run).
        data["metadata"]["t60_flatness_tolerance_s"] = -0.05
        self.assertEqual(t60_flatness_tolerance_s(data), (0.05, False))
        self.assertEqual(t60_flatness_pct(data, "L"), 100.0)
        data["metadata"]["t60_flatness_tolerance_s"] = 0.05
        bands[0]["t60_s"] = True
        self.assertIsNone(t60_rows(channel))
        self.assertIsNone(t60_flatness_pct(data, "L"))

    def test_band_mean_and_notch(self):
        curve = {"freq": [100.0, 200.0, 1000.0, 2000.0],
                 "spl": [-12.0, -6.0, 0.0, 0.0]}
        self.assertAlmostEqual(band_mean(curve["freq"], curve["spl"], 500.0, 3000.0), 0.0)
        self.assertAlmostEqual(deepest_notch_db({"final_curve": curve}), -12.0)
        self.assertIsNone(deepest_notch_db({}))

    def test_malformed_curve_cannot_produce_level_or_notch_summary(self):
        complete = {"freq": [100.0, 1000.0, 2000.0],
                    "spl": [-6.0, 0.0, 0.0]}
        malformed = {"freq": [100.0, 1000.0, 2000.0],
                     "spl": [-12.0, 0.0]}
        self.assertIsNone(band_mean(malformed["freq"], malformed["spl"], 500.0, 3000.0))
        self.assertIsNone(deepest_notch_db({"final_curve": malformed}))
        self.assertIsNone(band_mean([1000.0, 500.0], [0.0, 1.0], 500.0, 3000.0))
        self.assertIsNone(band_mean([100.0, float("nan")], [0.0, 1.0], 50.0, 300.0))
        self.assertIsNone(band_mean([1000.0, 2000.0], [1e308, 1e308], 500.0, 3000.0))
        rows = level_compensation({"L": {"initial_curve": complete},
                                   "R": {"initial_curve": malformed}})
        by_name = {row["speaker"]: row for row in rows}
        self.assertEqual(by_name["L"]["comp_db"], 0.0)
        self.assertIsNone(by_name["R"]["comp_db"])

    def test_level_compensation_attenuates_louder_monitor_without_claiming_calibration(self):
        init_l, init_r = _stereo_curves([1.0, 1.0, 1.0, 1.0], [-1.0, -1.0, -1.0, -1.0])
        rows = level_compensation({"L": {"initial_curve": init_l},
                                   "R": {"initial_curve": init_r}})
        by_name = {r["speaker"]: r for r in rows}
        self.assertAlmostEqual(by_name["L"]["comp_db"], -2.0)
        self.assertAlmostEqual(by_name["R"]["comp_db"], 0.0)
        self.assertIsNone(by_name["L"]["residual_db"])

    def test_symmetric_groups_pair_stereo(self):
        groups, unpaired = symmetric_groups({"L": {}, "R": {}, "C": {}})
        self.assertEqual(groups, [("L+R", ["L", "R"])])
        self.assertEqual(unpaired, ["C"])

    def test_pair_sum_difference_grids_must_match(self):
        curve_a = {"freq": [100.0, 200.0], "spl": [0.0, 0.0]}
        curve_b = {"freq": [100.0, 200.0], "spl": [0.0, 0.0]}
        combo = pair_sum_difference(curve_a, curve_b)
        self.assertAlmostEqual(combo["sum_spl"][0], 6.0206, places=3)
        self.assertAlmostEqual(combo["diff_spl"][0], -120.0)
        self.assertIsNone(pair_sum_difference(curve_a, {"freq": [100.0], "spl": [0.0]}))
        self.assertIsNone(pair_sum_difference(curve_a, {"freq": [101.0, 201.0], "spl": [0.0, 0.0]}))
        self.assertIsNone(pair_sum_difference(curve_a, {"freq": [200.0, 100.0], "spl": [0.0, 0.0]}))
        self.assertIsNone(pair_sum_difference(curve_a, {"freq": [100.0, 200.0], "spl": [0.0]}))
        self.assertIsNone(pair_sum_difference(curve_a, {"freq": [100.0, 200.0], "spl": [0.0, 10000.0]}))

    def test_tof_table_and_summary_pending_markers(self):
        metadata = {"timing_diagnostics": {"channels": [
            {"name": "L", "measured_arrival_ms": 0.4, "applied_delay_ms": 0.2,
             "final_arrival_ms": 0.6, "final_offset_from_reference_ms": 0.0},
        ]}}
        rows = tof_table(metadata)
        self.assertEqual(len(rows), 1)
        table = tof_html(metadata)
        self.assertIn("Measured arrival before DSP", table)
        self.assertIn("Calculated arrival after DSP", table)
        self.assertIn("not a post-playback capture", table)
        section = create_tof_figure(rows, after=True)
        self.assertIn("Calculated arrival after DSP", section["chart"]["title"])
        self.assertAlmostEqual(rows[0]["after_ms"], 0.6)
        partial = rows + [{"name": "R", "before_ms": None, "after_ms": None}]
        # The bar schema has no nulls: rows without a measurement are dropped.
        partial_section = create_tof_figure(partial, after=True)
        self.assertEqual(partial_section["chart"]["categories"], ["L"])
        self.assertEqual(partial_section["chart"]["groups"][0]["values"], [0.6])
        self.assertIsNone(create_tof_figure([{"name": "R", "after_ms": float("nan")}], after=True))
        self.assertIsNone(create_tof_figure([{"name": "R", "after_ms": True}], after=True))
        missing = {"timing_diagnostics": {"channels": [
            {"name": "R", "measured_arrival_ms": None, "applied_delay_ms": True,
             "final_arrival_ms": None, "final_offset_from_reference_ms": None},
        ]}}
        self.assertIn("n/a", tof_html(missing))
        html = summary_table_html({"channels": {"L": {"final_curve": {
            "freq": [100.0, 1000.0], "spl": [-5.0, 0.0]}}}})
        self.assertIn("Deepest notch", html)
        self.assertIn("pending", html)
        self.assertIn("early_late_curves", html)

    def test_html_report_contains_feat_report_sections(self):
        init_l, init_r = _stereo_curves([0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0])
        early_late = {
            "method": "incoherent_band_energy", "reference": "full_peak_band",
            "smoothing": "third_octave", "split_ms": 20.0,
            "full": {"freq": [100.0, 125.0], "spl": [0.0, -1.0]},
            "early": {"freq": [100.0, 125.0], "spl": [-1.0, -2.0]},
            "late": {"freq": [100.0, 125.0], "spl": [-7.0, -8.0]},
        }
        t60 = {"basis": "measured_room_ir", "min_r2": 0.9,
               "bands": [{"centre_hz": hz, "t60_s": 0.4, "fit_range": "T30",
                          "r2": 0.96, "valid": True, "reason": ""}
                         for hz in [63, 125, 250, 500, 1000, 2000, 4000, 8000, 16000]]}
        reflections = {
            "basis": "measured_room_ir", "method": "bandlimited_early_reflection_table_v1",
            "band_hz": [1000.0, 8000.0], "threshold_dbfs": -15.0,
            "direct_reference": "filtered direct peak", "pre": [],
            "post": [{"gain_dbfs": -8.0, "time_ms": 5.0,
                      "distance_cm": 171.5, "first_dip_hz": 100.0,
                      "ripple_db": 7.3}],
        }
        data = {"channels": {"L": {"initial_curve": init_l, "final_curve": init_l,
                                  "early_late_curves": early_late,
                                  "early_reflections": reflections,
                                  "t60_octaves": t60},
                             "R": {"initial_curve": init_r, "final_curve": init_r},
                             "C": {}},
                "metadata": {"t60_flatness_tolerance_s": 0.05,
                             "timing_diagnostics": {"channels": [
                    {"name": "L", "measured_arrival_ms": 0.4, "applied_delay_ms": 0.0,
                     "final_arrival_ms": 0.4, "final_offset_from_reference_ms": 0.0},
                    {"name": "R", "measured_arrival_ms": 0.5, "applied_delay_ms": 0.0,
                     "final_arrival_ms": 0.5, "final_offset_from_reference_ms": 0.1},
                ]}}}
        data = RoomEqData(data, Path("."))
        data.symmetric_pairs = {"L+R": {"freq": [100., 1000.],
            "sum_spl": [86., 87.], "diff_spl": [None, None]}}
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "report.html"
            create_html_report(data, output, None)
            html = output.read_text(encoding="utf-8")
        for expected in ("Results summary",
                         "Relative level compensation",
                         "Time of flight",
                         "Details per speaker",
                         "Symmetric pair: L+R",
                         "ITU-R BS.1116-3 upper limit",
                         "ITU-R BS.1116-3 lower limit",
                         "Unpaired channels (not summed): C",
                         "early vs late band energy",
                         "1–8 kHz early reflections: L",
                         "First dip (Hz)",
                         "Early vs late sound: pending roomeq field",
                         "Room mean: measured octave-band T60",
                         "Room T60 contributing speakers by octave: 63 Hz: 1"):
            self.assertIn(expected, html)
        # Removed duplications: no repeated Section 4 heading, no per-channel
        # T60 figure (the room block above carries per-speaker columns).
        self.assertNotIn("Section 4 — Symmetric monitors", html)
        self.assertNotIn("Section 1", html)
        self.assertNotIn("Section 3", html)
        self.assertNotIn("L: measured octave-band T60", html)
        # The flatness cell quotes live inside payload HTML (JSON-escaped on
        sections = report_payload(html)["sections"]
        pairs = [(s["bucket"], s["subhead"]) for s in sections if s.get("bucket")]
        subheads = list(dict.fromkeys(subhead for _, subhead in pairs))
        self.assertEqual(subheads, ["Why this correction?", "Summary",
            "Details per speaker", "Time of Flight",
            "Symmetric monitors", "Time domain analysis", "Section 6: EPA scores",
            "DSP signal flow"])
        self.assertEqual(list(dict.fromkeys(bucket for bucket, _ in pairs)),
                         [BUCKET_DSP, BUCKET_ACOUSTICS, BUCKET_PSYCHOACOUSTIC])
        self.assertNotIn("bucket", sections[0])
        self.assertNotIn("Optimization Summary", html)
        details = [s for s in sections if s.get("subhead") == subheads[2] and s.get("kind") == "figure"]
        self.assertEqual([s["tab"] for s in details], ["L", "R", "C"])
        self.assertTrue(all("Frequency landmarks" not in s["figure"]["title"] for s in details))
        # The flatness cell quotes live inside payload HTML (JSON-escaped on
        # disk), so assert against the decoded section instead of raw text.
        summary = next(
            section["html"] for section in report_payload(html)["sections"]
            if "Results summary" in section.get("html", "")
        )
        self.assertIn(
            'within declared ±0.05 s of the complete-channel room mean">100.0',
            summary,
        )

    def test_engine_early_late_renders_figure_and_ratio_cell(self):
        # Engine shape from attach_measured_acoustics: full contract keys
        # with 1-8 kHz coverage, as emitted for a measured room IR.
        freq = [1000.0, 2000.0, 4000.0, 8000.0]
        curves = {
            "method": "incoherent_band_energy", "reference": "full_peak_band",
            "smoothing": "third_octave", "split_ms": 20.0,
            "basis": "measured_room_ir",
            "direct_reference": "broadband envelope peak",
            "valid_band_hz": [900.0, 9000.0],
            "full": {"freq": list(freq), "spl": [0.0, -1.0, -2.0, -3.0]},
            "early": {"freq": list(freq), "spl": [-0.5, -1.5, -2.5, -3.5]},
            "late": {"freq": list(freq), "spl": [-6.5, -7.5, -8.5, -9.5]},
        }
        data = {"channels": {"L": {"initial_curve": {"freq": list(freq),
                                                      "spl": [0.0] * 4},
                                   "final_curve": {"freq": list(freq),
                                                   "spl": [0.0] * 4},
                                   "early_late_curves": curves}},
                "metadata": {}}
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "report.html"
            create_html_report(data, output, None)
            html = output.read_text(encoding="utf-8")
        payload = report_payload(html)
        titles = [section["figure"]["title"] for section in payload["sections"]
                  if section["kind"] == "figure" and section.get("tab") == "L"]
        self.assertIn("L: early vs late band energy (20 ms split)", titles)
        summary = next(
            section["html"] for section in payload["sections"]
            if "Results summary" in section.get("html", "")
        )
        # The ratio cell flips from pending to its +6 dB value.
        self.assertIn(">+6.0<", summary)
        self.assertNotIn("needs roomeq field: early_late_curves", summary)


class SplPresentationTests(unittest.TestCase):
    def test_band_mean_spl_uses_finite_in_band_samples_only(self):
        curve = {
            "freq": [50.0, 100.0, 1000.0, 10_000.0, 20_000.0],
            "spl": [0.0, 70.0, 80.0, 90.0, 0.0],
        }
        mean = band_mean_spl(curve, 100.0, 10_000.0)
        assert mean is not None
        self.assertAlmostEqual(mean, 80.0)
        self.assertIsNone(band_mean_spl(curve, 200.0, 500.0))
        self.assertIsNone(band_mean_spl({"freq": [100.0], "spl": [80.0, 81.0]},
                                        100.0, 10_000.0))

    def test_shift_target_matches_reference_band_mean(self):
        reference = {
            "freq": [50.0, 100.0, 1000.0, 10_000.0, 20_000.0],
            "spl": [70.0, 72.0, 74.0, 76.0, 78.0],
        }
        target = {
            "freq": [100.0, 1000.0, 10_000.0],
            "spl": [80.0, 80.0, 80.0],
        }
        shifted = shift_target_to_reference_band_mean(target, reference)
        assert shifted is not None
        self.assertEqual(shifted["freq"], reference["freq"])
        shifted_mean = band_mean_spl(shifted, 100.0, 10_000.0)
        reference_mean = band_mean_spl(reference, 100.0, 10_000.0)
        assert shifted_mean is not None and reference_mean is not None
        self.assertAlmostEqual(shifted_mean, reference_mean)
        # Empty reference band: no alignment possible.
        self.assertIsNone(shift_target_to_reference_band_mean(
            target, {"freq": [20.0, 30.0], "spl": [70.0, 71.0]}))

    def test_zoomed_figures_default_to_50_db_span(self):
        curve = {"freq": [20.0, 100.0, 500.0], "spl": [70.0, 72.0, 74.0]}
        fig = create_zoomed_figure("L", curve, curve)
        self.assertAlmostEqual(
            fig["figure"]["y"]["max"] - fig["figure"]["y"]["min"], 50.0)
        fig = create_comparison_zoomed_figure(
            "L", [("iir", {"final_curve": curve})])
        self.assertAlmostEqual(
            fig["figure"]["y"]["max"] - fig["figure"]["y"]["min"], 50.0)


class ComparisonTargetLevelTests(unittest.TestCase):
    def _stereo_mode(self, level_db, target_db):
        freq = [20.0, 50.0, 100.0, 200.0, 500.0, 1000.0,
                2000.0, 5000.0, 10_000.0, 20_000.0]
        curve = {"freq": list(freq), "spl": [level_db] * len(freq)}
        target = {"freq": list(freq), "spl": [target_db] * len(freq)}
        channel = {
            "initial_curve": dict(curve),
            "final_curve": dict(curve),
            "target_curve": dict(target),
        }
        return {"channels": {"L": copy.deepcopy(channel),
                             "R": copy.deepcopy(channel)},
                "metadata": {}}

    def _overlay_target(self, payload, tab):
        figures = [
            section["figure"]
            for section in payload["sections"]
            if section["kind"] == "figure"
            and section.get("tab") == tab
            and "Mode Comparison" in section["figure"]["title"]
        ]
        self.assertEqual(len(figures), 1)
        targets = [s for s in figures[0]["series"] if s["name"] == "Target"]
        self.assertEqual(len(targets), 1)
        return targets[0]

    def test_channel_targets_match_per_channel_mean_and_lr_matches_sum(self):
        data = self._stereo_mode(75.0, 85.0)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "comparison.html"
            create_comparison_html_report([("iir", data)], output)
            payload = report_payload(output.read_text(encoding="utf-8"))
        for tab in ("L", "R"):
            target = self._overlay_target(payload, tab)
            mean = band_mean_spl(
                {"freq": target["x"], "spl": target["y"]}, 100.0, 10_000.0)
            assert mean is not None
            self.assertAlmostEqual(mean, 75.0, places=6)
        # The L+R tab shows the magnitude-only power sum, so its target
        # stays matched to the summed level.
        target = self._overlay_target(payload, "L+R")
        mean = band_mean_spl(
            {"freq": target["x"], "spl": target["y"]}, 100.0, 10_000.0)
        assert mean is not None
        self.assertAlmostEqual(mean, 75.0 + 10.0 * math.log10(2.0), places=6)


class DriverTabT60Tests(unittest.TestCase):
    def test_driver_tabs_keep_their_own_t60(self):
        freq = [20.0, 100.0, 1000.0]
        centers = (63, 125, 250, 500, 1000, 2000, 4000, 8000, 16000)
        t60 = {
            "basis": "measured_room_ir",
            "min_r2": 0.9,
            "bands": [
                {"centre_hz": c, "valid": True, "t60_s": 0.3,
                 "fit_range": "T30", "r2": 0.95, "reason": None}
                for c in centers
            ],
        }
        data = {
            "channels": {
                "LFE": {
                    "drivers": [{
                        "name": "Sub1",
                        "initial_curve": {"freq": list(freq), "spl": [70.0] * 3},
                        "measured_acoustics": {"t60_octaves": t60},
                    }],
                },
            },
            "metadata": {},
        }
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "report.html"
            create_html_report(data, output, None)
            html = output.read_text(encoding="utf-8")
        # Driver-level fits exist nowhere else, so driver tabs keep them
        # while channel tabs reuse the room-level block.
        self.assertIn("Sub1: measured octave-band T60", html)
        self.assertNotIn("LFE: measured octave-band T60", html)


class SymmetricComplexSumReportTests(unittest.TestCase):
    def test_pair_figure_draws_absolute_complex_and_difference(self):
        freq = [100.0, 1000.0]
        abs_level = 80.0 + 20.0 * math.log10(2.0)
        data = {
            "channels": {
                "L": {
                    "initial_curve": {"freq": list(freq), "spl": [80.0, 80.0]},
                    "final_curve": {"freq": list(freq), "spl": [80.0, 80.0],
                                    "phase": [0.0, 0.0]},
                },
                "R": {
                    "initial_curve": {"freq": list(freq), "spl": [80.0, 80.0]},
                    "final_curve": {"freq": list(freq), "spl": [80.0, 80.0],
                                    "phase": [0.0, 90.0]},
                },
            },
            "metadata": {},
        }
        wrapped = RoomEqData(data, Path("."))
        wrapped.symmetric_pairs = {"L+R": {"freq": list(freq), "sum_spl": [abs_level, abs_level], "diff_spl": [-120.0, -120.0]}}
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "report.html"
            create_html_report(wrapped, output, None)
            payload = report_payload(output.read_text(encoding="utf-8"))
        figures = [
            section["figure"]
            for section in payload["sections"]
            if section["kind"] == "figure" and section.get("tab") == "L+R"
            and section["figure"]["title"].startswith("Symmetric pair: L+R")
        ]
        self.assertEqual(len(figures), 1)
        by_name = {s["name"]: s for s in figures[0]["series"]}
        self.assertEqual(
            sorted(by_name),
            ["L+R absolute sum", "L+R complex sum", "L+R sum difference"],
        )
        # In phase at 100 Hz (no cancellation), quadrature at 1 kHz.
        self.assertAlmostEqual(by_name["L+R complex sum"]["y"][0], abs_level)
        self.assertAlmostEqual(
            by_name["L+R complex sum"]["y"][1], 83.01029995663981, places=6)
        self.assertAlmostEqual(by_name["L+R sum difference"]["y"][0], 0.0)
        self.assertAlmostEqual(
            by_name["L+R sum difference"]["y"][1], 3.01029995663981, places=6)

    def test_pair_figure_without_phase_shows_absolute_only(self):
        freq = [100.0, 1000.0]
        data = {
            "channels": {
                side: {
                    "initial_curve": {"freq": list(freq), "spl": [80.0, 80.0]},
                    "final_curve": {"freq": list(freq), "spl": [80.0, 80.0]},
                }
                for side in ("L", "R")
            },
            "metadata": {},
        }
        wrapped = RoomEqData(data, Path("."))
        wrapped.symmetric_pairs = {"L+R": {"freq": list(freq), "sum_spl": [86.0, 86.0], "diff_spl": [-120.0, -120.0]}}
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "report.html"
            create_html_report(wrapped, output, None)
            html = output.read_text(encoding="utf-8")
        payload = report_payload(html)
        figures = [
            section["figure"]
            for section in payload["sections"]
            if section["kind"] == "figure" and section.get("tab") == "L+R"
            and section["figure"]["title"].startswith("Symmetric pair: L+R")
        ]
        self.assertEqual(len(figures), 1)
        self.assertEqual(
            [s["name"] for s in figures[0]["series"]], ["L+R absolute sum"])
        self.assertIn("complex sum unavailable", html)


class WaterfallSurfaceDefaultsTests(unittest.TestCase):
    def test_surface_defaults_to_contours_with_hidden_fill(self):
        section = grid_figure(
            "Waterfall", [20.0, 20000.0], [0.0, 500.0],
            [[0.0, -60.0], [-60.0, -60.0]], surface=True)
        grid = section["grid"]
        self.assertEqual(grid["colormap"], "turbo")
        self.assertIs(grid["show_surface"], False)
        self.assertIs(grid["show_contours"], True)

    def test_surface_display_options_override_defaults(self):
        section = grid_figure(
            "Waterfall", [20.0, 20000.0], [0.0, 500.0],
            [[0.0, -60.0], [-60.0, -60.0]], surface=True,
            colormap="viridis", show_surface=True, show_contours=False)
        grid = section["grid"]
        self.assertEqual(grid["colormap"], "viridis")
        self.assertIs(grid["show_surface"], True)
        self.assertIs(grid["show_contours"], False)

    def test_heatmap_omits_surface_display_keys(self):
        section = grid_figure(
            "Wavelet", [20.0, 20000.0], [-1.0, 15.0],
            [[0.0, -30.0], [-30.0, -30.0]])
        grid = section["grid"]
        self.assertNotIn("colormap", grid)
        self.assertNotIn("show_surface", grid)
        self.assertNotIn("show_contours", grid)


class ReportProvenanceTests(unittest.TestCase):
    def test_roomeq_version_prefers_stamped_producer_version(self):
        self.assertEqual(
            _roomeq_version({"metadata": {"producer_version": "9.9.9"}}), "9.9.9")
        self.assertEqual(
            _roomeq_version({"metadata": {"producer_version": "  "},
                             "roomeq_version": "8.8.8"}), "8.8.8")
        self.assertEqual(_roomeq_version({}), _workspace_roomeq_version())

    def test_workspace_roomeq_version_tracks_cargo_toml(self):
        version = _workspace_roomeq_version()
        assert version is not None
        self.assertRegex(version, r"^\d+\.\d+\.\d+$")
        cargo = (Path(__file__).resolve().parent.parent / "Cargo.toml"
                 ).read_text(encoding="utf-8")
        package = cargo.split("[workspace.package]", 1)[1].split("[", 1)[0]
        self.assertIn(f'version = "{version}"', package)

    def test_report_provenance_carries_version_and_dates(self):
        provenance = _report_provenance(
            [("", {"metadata": {"timestamp": "2026-10-01T10:00:00Z"}})])
        self.assertEqual(provenance["roomeq_version"], _workspace_roomeq_version())
        self.assertEqual(provenance["data_timestamp"], "2026-10-01T10:00:00Z")
        self.assertRegex(provenance["generated_at"], r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")
        mixed = _report_provenance([
            ("a", {"metadata": {"producer_version": "1.0", "timestamp": "t1"}}),
            ("b", {"metadata": {"producer_version": "2.0", "timestamp": "t2"}}),
        ])
        self.assertEqual(mixed["roomeq_version"], "1.0 / 2.0")
        self.assertEqual(mixed["data_timestamp"], "t1 / t2")

    def test_html_report_embeds_provenance_payload(self):
        data = _driver_eq_split_data()
        metadata: dict = data.setdefault("metadata", {})
        metadata["timestamp"] = "2026-10-01T10:00:00Z"
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "report.html"
            create_html_report(data, output, None)
            payload = report_payload(output.read_text(encoding="utf-8"))
        self.assertEqual(payload["provenance"]["roomeq_version"],
                         _workspace_roomeq_version())
        self.assertEqual(payload["provenance"]["data_timestamp"],
                         "2026-10-01T10:00:00Z")
        self.assertIn("generated_at", payload["provenance"])

    def test_shell_renders_provenance_above_renderer_status(self):
        template = (Path(__file__).resolve().parent.parent / "crates"
                    / "autoeq-report-wasm" / "shell" / "template.html"
                    ).read_text(encoding="utf-8")
        self.assertLess(template.index('id="report-provenance"'),
                        template.index('id="renderer-status"'))
        for field in ("provenance.roomeq_version", "provenance.data_timestamp",
                      "provenance.generated_at"):
            self.assertIn(field, template)
        # Text assignment keeps untrusted values out of the HTML parser.
        self.assertIn('getElementById("report-provenance").textContent', template)


class ReportBucketTests(unittest.TestCase):
    def test_single_report_assigns_three_buckets(self):
        data = _driver_eq_split_data()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "report.html"
            create_html_report(data, output, None)
            payload = report_payload(output.read_text(encoding="utf-8"))
        sections = payload["sections"]
        # The flat playback verdict stays unbucketed ahead of the tabs.
        self.assertNotIn("bucket", sections[0])
        self.assertTrue(sections[0].get("flat"))
        self.assertIn("playback-status", sections[0]["html"])
        pairs = [(section.get("bucket"), section.get("subhead"))
                 for section in sections[1:]]
        self.assertTrue(all(bucket for bucket, _ in pairs))
        self.assertTrue(all(subhead for _, subhead in pairs))
        self.assertFalse(any("group" in section for section in sections))
        ordered = list(dict.fromkeys(pairs))
        self.assertEqual(ordered, [
            (BUCKET_DSP, "Why this correction?"),
            (BUCKET_ACOUSTICS, "Summary"),
            (BUCKET_ACOUSTICS, "Details per speaker"),
            (BUCKET_ACOUSTICS, "Time of Flight"),
            (BUCKET_ACOUSTICS, "Symmetric monitors"),
            (BUCKET_ACOUSTICS, "Time domain analysis"),
            (BUCKET_PSYCHOACOUSTIC, "Section 6: EPA scores"),
            (BUCKET_DSP, "DSP signal flow"),
        ])
        self.assertEqual(
            [BUCKET_DSP, BUCKET_ACOUSTICS, BUCKET_PSYCHOACOUSTIC],
            ["DSP analysis", "Acoustics analysis", "Psychoacoustic report"])

    def test_comparison_report_stays_unbucketed(self):
        data = _driver_eq_split_data()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "comparison.html"
            create_comparison_html_report([("iir", data), ("fir", data)], output)
            payload = report_payload(output.read_text(encoding="utf-8"))
        self.assertFalse(any("bucket" in section for section in payload["sections"]))

    def test_relative_levels_lead_the_details_tab(self):
        data = _driver_eq_split_data()
        grid = [100.0, 1000.0, 2000.0, 3000.0]
        data["channels"]["L"]["initial_curve"] = {"freq": grid, "spl": [0.0, 0.0, 0.0, 0.0]}
        data["channels"]["R"]["initial_curve"] = {"freq": grid, "spl": [1.0, 1.0, 1.0, 1.0]}
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "report.html"
            create_html_report(data, output, None)
            payload = report_payload(output.read_text(encoding="utf-8"))
        sections = payload["sections"]
        details = [s for s in sections if s.get("subhead") == "Details per speaker"]
        self.assertTrue(details)
        self.assertIn("Relative level compensation", details[0].get("html", ""))
        self.assertIsNone(details[0].get("tab"))
        summary = [s for s in sections if s.get("subhead") == "Summary"]
        self.assertFalse(any("Relative level compensation" in s.get("html", "")
                             for s in summary))

    def test_shell_renders_centered_bucket_tabs(self):
        template = (Path(__file__).resolve().parent.parent / "crates"
                    / "autoeq-report-wasm" / "shell" / "template.html"
                    ).read_text(encoding="utf-8")
        self.assertIn('["DSP analysis", "Acoustics analysis", "Psychoacoustic report"]',
                      template)
        self.assertIn("buildBucketPage", template)
        bar = template.split(".bucket-bar {", 1)[1].split("}", 1)[0]
        self.assertIn("justify-content: center", bar)
        buttons = template.split(".bucket-bar button {", 1)[1].split("}", 1)[0]
        self.assertIn("border-radius: 8px 8px 0 0", buttons)

    def test_bucket_subheads_render_as_tabs(self):
        template = (Path(__file__).resolve().parent.parent / "crates"
                    / "autoeq-report-wasm" / "shell" / "template.html"
                    ).read_text(encoding="utf-8")
        self.assertIn("buildBucketPage", template)
        self.assertIn("subhead-bar", template)
        self.assertIn("subpage", template)
        self.assertIn(".tabbar, .subhead-bar", template)
        self.assertNotIn("bucket-subhead", template)

    def test_single_card_rule_and_flat_verdict(self):
        template = (Path(__file__).resolve().parent.parent / "crates"
                    / "autoeq-report-wasm" / "shell" / "template.html"
                    ).read_text(encoding="utf-8")
        self.assertIn(".html-section.flat", template)
        self.assertIn(".html-section .filters-section", template)
        self.assertIn(".html-section .plot-container", template)
        self.assertIn('sec.flat ? " flat" : ""', template)

    def test_details_eq_second_axis_spans_50db(self):
        data = _driver_eq_split_data()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "report.html"
            create_html_report(data, output, None)
            payload = report_payload(output.read_text(encoding="utf-8"))
        full = [s["figure"] for s in payload["sections"]
                if s.get("kind") == "figure"
                and s.get("subhead") == "Details per speaker"
                and "(Full Range)" in s["figure"].get("title", "")
                and s["figure"].get("y2")]
        self.assertTrue(full)
        for fig in full:
            self.assertEqual(fig["y2"]["min"], -25.0)
            self.assertEqual(fig["y2"]["max"], 25.0)

    def test_frequency_graphs_stack_presets_and_offer_db_spans(self):
        template = (Path(__file__).resolve().parent.parent / "crates"
                    / "autoeq-report-wasm" / "shell" / "template.html"
                    ).read_text(encoding="utf-8")
        freq = template.split(".freq-presets {", 1)[1].split("}", 1)[0]
        self.assertIn("flex-direction: column", freq)
        self.assertIn(".db-presets", template)
        self.assertIn("setDbSpan", template)
        self.assertIn("for (const span of [20, 50])", template)

    def test_graphs_are_capped_at_800px(self):
        template = (Path(__file__).resolve().parent.parent / "crates"
                    / "autoeq-report-wasm" / "shell" / "template.html"
                    ).read_text(encoding="utf-8")
        card = template.split(".plot-container {", 1)[1].split("}", 1)[0]
        self.assertIn("max-width: 800px", card)
        self.assertIn("margin: 0 auto 20px", card)
        self.assertIn(".html-section .plot-container:not(:has(canvas))", template)
        self.assertIn(".html-section .plot-container:has(canvas)", template)

    def test_correction_explanation_has_no_inline_card(self):
        from scripts.src.correction_explanation import correction_explanation_html
        html = correction_explanation_html({"channels": {}, "metadata": {}})
        self.assertIn('<section class="correction-explanation">', html)
        self.assertNotIn("background:#fff", html)


if __name__ == "__main__":
    unittest.main()
