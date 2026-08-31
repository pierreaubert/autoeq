#!/usr/bin/env python3

import math
import unittest

from scripts.src.dsp import (
    apply_plugins_to_curve,
    biquad_coefficients,
    build_post_dsp_source_curves,
    compute_eq_response,
    compute_group_delay_from_ir,
    wrap_phase,
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


class TemporalResponseTests(unittest.TestCase):
    def test_phase_wrap_uses_report_range(self):
        self.assertEqual(
            wrap_phase([-540.0, -181.0, 180.0, 181.0, 540.0]),
            [-180.0, 179.0, -180.0, -179.0, -180.0],
        )

    def test_group_delay_from_ir_preserves_large_delay(self):
        sample_rate = 48_000.0
        sample_count = 19_200
        delay_ms = 90.0
        delay_sample = round(delay_ms * sample_rate / 1000.0)
        amplitude = [0.0] * sample_count
        amplitude[delay_sample] = 1.0
        ir = {
            "time_ms": [index * 1000.0 / sample_rate for index in range(sample_count)],
            "amplitude": amplitude,
        }

        freq, group_delay_ms = compute_group_delay_from_ir(ir)

        self.assertTrue(freq)
        central = sorted(group_delay_ms)[len(group_delay_ms) // 2]
        self.assertAlmostEqual(central, delay_ms, delta=0.05)

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
    def test_plugin_realization_uses_rust_acoustic_floor(self):
        realized = apply_plugins_to_curve(
            {"freq": [20.0], "spl": [-100.0], "phase": [0.0]},
            [{"plugin_type": "gain", "parameters": {"gain_db": -200.0}}],
            48_000.0,
        )
        self.assertEqual(realized["spl"], [-240.0])

    def test_authoritative_deployed_curves_bypass_python_reconstruction(self):
        deployed = {
            "L": {
                "freq": [80.0],
                "spl": [72.0],
                "phase": [45.0],
            }
        }
        data = {
            "deployed_source_curves": deployed,
            "channels": {"L": {"final_curve": {"freq": [80.0], "spl": [-99.0]}}},
        }

        self.assertIs(build_post_dsp_source_curves(data), deployed)

    def test_pre_route_time_alignment_delay_preserves_crossover_sum(self):
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
                            }
                        ]
                    },
                }
            },
            "channels": {
                "L": {
                    # LR24 high-pass at Fc. The reported final curve already
                    # contains the leading input-alignment delay.
                    "final_curve": {"freq": [80.0], "spl": [53.9794], "phase": [-180.0]},
                    "plugins": [],
                },
                "LFE": {
                    "initial_curve": {"freq": [80.0], "spl": [60.0], "phase": [0.0]},
                    "plugins": [],
                },
            },
        }

        baseline = build_post_dsp_source_curves(data)["L"]["spl"][0]
        delay_ms = 1.0
        data["channels"]["L"]["final_curve"]["phase"][0] -= 360.0 * 80.0 * delay_ms / 1000.0
        data["channels"]["L"]["plugins"] = [
            {
                "plugin_type": "delay",
                "parameters": {"delay_ms": delay_ms, "room_eq_stage": "pre_route"},
            }
        ]
        delayed = build_post_dsp_source_curves(data)["L"]["spl"][0]

        self.assertAlmostEqual(delayed, baseline, places=5)

    def test_redirected_bass_respects_pre_and_post_route_stage_ownership(self):
        data = {
            "metadata": {
                "bass_management": {
                    "physical_sub_output": "LFE",
                    "routing_graph": {
                        "input_trim_db": {"L": -6.0},
                        "routes": [
                            {
                                "source_channel": "L",
                                "route_kind": "redirected_bass_lowpass_to_sub",
                                "crossover_type": "LR24",
                                "low_pass_hz": 80.0,
                                "gain_db": 0.0,
                                "delay_ms": 0.0,
                                "polarity_inverted": False,
                            }
                        ],
                    },
                }
            },
            "channels": {
                "L": {
                    "final_curve": {
                        "freq": [80.0],
                        "spl": [-200.0],
                        "phase": [0.0],
                    },
                    "plugins": [
                        {
                            "plugin_type": "gain",
                            "parameters": {
                                "gain_db": -6.0,
                                "label": "post_dsp_input_level_alignment",
                                "room_eq_stage": "pre_route",
                            },
                        }
                    ]
                },
                "LFE": {
                    "initial_curve": {
                        "freq": [80.0],
                        "spl": [60.0],
                        "phase": [0.0],
                    },
                    "plugins": [
                        {
                            "plugin_type": "gain",
                            "parameters": {
                                "gain_db": -20.0,
                                "room_eq_stage": "pre_route",
                            },
                        },
                        {
                            "plugin_type": "gain",
                            "parameters": {
                                "gain_db": 3.0,
                                "room_eq_stage": "post_route",
                            },
                        },
                    ],
                },
            },
        }

        baseline = build_post_dsp_source_curves(data)["L"]["spl"][0]
        data["channels"]["LFE"]["plugins"][0]["parameters"]["gain_db"] = -40.0
        sub_pre_route_changed = build_post_dsp_source_curves(data)["L"]["spl"][0]
        data["channels"]["LFE"]["plugins"][1]["parameters"]["gain_db"] = 5.0
        sub_post_route_changed = build_post_dsp_source_curves(data)["L"]["spl"][0]

        self.assertAlmostEqual(sub_pre_route_changed, baseline, places=6)
        self.assertAlmostEqual(sub_post_route_changed - baseline, 2.0, places=6)

    def test_bass_management_input_trim_is_applied_once(self):
        data = {
            "metadata": {
                "bass_management": {
                    "physical_sub_output": "LFE",
                    "routing_graph": {
                        "input_trim_db": {},
                        "routes": [
                            {
                                "source_channel": "LFE",
                                "route_kind": "lfe_lowpass_to_sub",
                                "crossover_type": "LR24",
                                "low_pass_hz": 80.0,
                                "gain_db": 0.0,
                                "delay_ms": 0.0,
                                "polarity_inverted": False,
                            }
                        ],
                    },
                }
            },
            "channels": {
                "LFE": {
                    "initial_curve": {"freq": [80.0], "spl": [60.0], "phase": [0.0]},
                    "final_curve": {"freq": [80.0], "spl": [99.0], "phase": [0.0]},
                    "plugins": [
                        {
                            "plugin_type": "gain",
                            "parameters": {
                                "gain_db": -6.0,
                                "label": "post_dsp_input_level_alignment",
                                "room_eq_stage": "pre_route",
                            },
                        }
                    ],
                }
            },
        }

        untrimmed = build_post_dsp_source_curves(data)
        data["metadata"]["bass_management"]["routing_graph"]["input_trim_db"] = {
            "LFE": -6.0
        }
        trimmed = build_post_dsp_source_curves(data)

        self.assertAlmostEqual(
            trimmed["LFE"]["spl"][0] - untrimmed["LFE"]["spl"][0],
            -6.0,
            places=6,
        )

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
        self.assertAlmostEqual(curves["L"]["spl"][0], 58.62888170047014, places=6)
        self.assertAlmostEqual(curves["LFE"]["spl"][0], 49.43433115727133, places=6)
        self.assertEqual(curves["R"]["spl"], [42.0])
        self.assertNotEqual(curves["LFE"]["spl"], [99.0])


if __name__ == "__main__":
    unittest.main()
