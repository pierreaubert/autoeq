#!/usr/bin/env python3

import math
import unittest

from scripts.src.dsp import (
    biquad_coefficients,
    build_post_dsp_source_curves,
    compute_eq_response,
)


class BiquadParityTests(unittest.TestCase):
    def test_every_rust_biquad_type_has_finite_normalized_coefficients(self):
        filter_types = [
            "lowpass",
            "highpass",
            "highpassvariableq",
            "bandpass",
            "peak",
            "notch",
            "lowshelf",
            "highshelf",
            "allpass",
            "lowshelforf",
            "highshelforf",
            "peakmatched",
        ]

        for filter_type in filter_types:
            with self.subTest(filter_type=filter_type):
                coefficients = biquad_coefficients(
                    filter_type, 1_000.0, 48_000.0, 0.8, 6.0
                )
                self.assertEqual(len(coefficients), 5)
                self.assertTrue(all(math.isfinite(value) for value in coefficients))

    def test_standard_shelves_match_rust_q_independent_convention(self):
        for filter_type in ("lowshelf", "highshelf"):
            with self.subTest(filter_type=filter_type):
                low_q = biquad_coefficients(filter_type, 800.0, 48_000.0, 0.25, 5.0)
                high_q = biquad_coefficients(filter_type, 800.0, 48_000.0, 4.0, 5.0)
                for observed, expected in zip(low_q, high_q):
                    self.assertAlmostEqual(observed, expected, places=14)

    def test_allpass_has_unity_magnitude_and_unknown_types_fail_closed(self):
        response = compute_eq_response(
            [
                {
                    "filter_type": "allpass",
                    "freq": 1_200.0,
                    "q": 0.7,
                    "db_gain": 0.0,
                }
            ],
            [40.0, 400.0, 1_200.0, 8_000.0, 18_000.0],
        )
        for value in response:
            self.assertAlmostEqual(value, 0.0, places=10)

        with self.assertRaisesRegex(ValueError, "unsupported biquad filter type"):
            compute_eq_response(
                [{"filter_type": "future_filter", "freq": 1_000.0, "q": 1.0}],
                [1_000.0],
            )

    def test_peak_and_matched_peak_reach_requested_center_gain(self):
        for filter_type in ("peak", "peakmatched"):
            with self.subTest(filter_type=filter_type):
                response = compute_eq_response(
                    [
                        {
                            "filter_type": filter_type,
                            "freq": 2_000.0,
                            "q": 1.3,
                            "db_gain": 7.0,
                        }
                    ],
                    [2_000.0],
                )
                self.assertAlmostEqual(response[0], 7.0, places=8)


class PostDspSourceCurveTests(unittest.TestCase):
    def test_bass_management_is_reconstructed_per_source(self):
        data = {
            "metadata": {
                "bass_management": {
                    "physical_sub_output": "LFE",
                    "routing_graph": {
                        "routes": [
                            {
                                "source_channel": "L",
                                "route_kind": "redirected_bass_lowpass_to_sub",
                                "crossover_type": "LR24",
                                "low_pass_hz": 80.0,
                                "gain_db": 0.0,
                                "delay_ms": 0.0,
                                "polarity_inverted": False,
                            },
                        {
                            "source_channel": "LFE",
                            "route_kind": "lfe_lowpass_to_sub",
                            "crossover_type": "LR24",
                            # The LFE programme cutoff is independent of the
                            # 80 Hz redirected-bass speaker crossover above.
                            "low_pass_hz": 120.0,
                                "gain_db": -6.0,
                                "delay_ms": 0.0,
                                "polarity_inverted": False,
                            },
                        ]
                    },
                }
            },
            "channels": {
                "L": {
                    "final_curve": {
                        "freq": [80.0],
                        "spl": [50.979400086720375],
                        "phase": [-180.0],
                    }
                },
                "R": {
                    "final_curve": {"freq": [80.0], "spl": [42.0], "phase": [0.0]}
                },
                "LFE": {
                    "initial_curve": {"freq": [80.0], "spl": [60.0], "phase": [0.0]},
                    # Deliberately an aggregate bus value: it must not be reused.
                    "final_curve": {"freq": [80.0], "spl": [99.0], "phase": [0.0]},
                    "plugins": [
                        {
                            "plugin_type": "gain",
                            "parameters": {"gain_db": -3.0, "room_eq_stage": "pre_route"},
                        },
                        {
                            "plugin_type": "gain",
                            "parameters": {
                                "gain_db": 40.0,
                                "room_eq_stage": "route_owned",
                            },
                        },
                    ],
                },
            },
        }

        curves = build_post_dsp_source_curves(data)

        self.assertEqual(set(curves), {"L", "R", "LFE"})
        self.assertAlmostEqual(curves["L"]["spl"][0], 57.0, places=6)
        self.assertAlmostEqual(curves["LFE"]["spl"][0], 49.43433115727133, places=6)
        self.assertEqual(curves["R"]["spl"], [42.0])
        self.assertNotEqual(curves["LFE"]["spl"], [99.0])


if __name__ == "__main__":
    unittest.main()
