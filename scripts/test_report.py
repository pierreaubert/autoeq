#!/usr/bin/env python3

import tempfile
import unittest
from pathlib import Path

from scripts.src.data_extract import display_channel_entries
from scripts.src.report import (
    _channel_display_final_curve,
    _comparison_source_label,
    _driver_shaping_summary_html,
    _has_redirected_bass_route,
    _mixed_phase_summary_html,
    create_html_report,
)
from scripts.test_figures import two_sub_overview_data


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

    def test_html_report_renders_per_sub_tabs(self):
        data = two_sub_overview_data()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "report.html"
            create_html_report(data, output, None)
            html = output.read_text(encoding="utf-8")

        self.assertIn(">Left Sub</button>", html)
        self.assertIn(">Right Sub</button>", html)
        self.assertNotIn(">LFE</button>", html)
        self.assertIn("<h2>Channel: Left Sub</h2>", html)
        self.assertIn("Sub DSP Chain", html)
        self.assertIn("EQ: Left Sub", html)
        self.assertIn("EQ: Right Sub", html)


if __name__ == "__main__":
    unittest.main()
