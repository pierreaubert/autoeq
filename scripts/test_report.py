#!/usr/bin/env python3

import unittest

from scripts.src.report import _mixed_phase_summary_html


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


if __name__ == "__main__":
    unittest.main()
