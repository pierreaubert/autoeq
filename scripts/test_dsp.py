#!/usr/bin/env python3

import math
import unittest

from scripts.src.dsp import biquad_coefficients, compute_eq_response


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


if __name__ == "__main__":
    unittest.main()
