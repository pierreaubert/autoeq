#!/usr/bin/env python3

import math
import unittest

from scripts.src.dsp import (
    apply_plugins_to_curve,
    biquad_coefficients,
    build_post_dsp_source_curves,
    compute_eq_response,
    compute_group_delay_from_ir,
    replay_serialized_output,
    resample_spl_onto_grid,
    sum_driver_initial_curves,
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
    def test_independent_serialized_chain_replay_matches_reported_curve(self):
        # This fixture intentionally uses only the serialized chain contract;
        # it does not call a Rust prediction helper or optimizer response.
        curve = {"freq": [100.0, 1_000.0, 10_000.0], "spl": [70.0, 70.0, 70.0]}
        chain = [{"plugin_type": "gain", "parameters": {"gain_db": 6.0}}]
        replay = apply_plugins_to_curve(curve, chain, 48_000.0)
        reported = {"freq": curve["freq"], "spl": [76.0, 76.0, 76.0]}
        self.assertEqual(replay["freq"], reported["freq"])
        for observed, expected in zip(replay["spl"], reported["spl"]):
            self.assertAlmostEqual(observed, expected, places=8)

    def test_serialized_output_replay_matches_deployed_curve(self):
        data = {
            "sample_rate": 48_000.0,
            "deployed_source_curves": {"L": {"freq": [100.0, 1_000.0], "spl": [76.0, 76.0]}},
            "channels": {
                "L": {
                    "initial_curve": {"freq": [100.0, 1_000.0], "spl": [70.0, 70.0]},
                    "plugins": [{"plugin_type": "gain", "parameters": {"gain_db": 6.0}}],
                }
            },
        }
        replayed = replay_serialized_output(data)
        self.assertEqual(replayed["L"], data["deployed_source_curves"]["L"])

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

    def test_mismatched_grids_do_not_drop_main_channels(self):
        # Mains on a full-range grid, sub branch on a sub-only grid: the
        # routed branch is resampled onto the main grid instead of the
        # channel being silently dropped from the corrected-curves row.
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
                                "low_pass_hz": 1000.0,
                                "gain_db": 0.0,
                                "delay_ms": 0.0,
                                "polarity_inverted": False,
                            },
                            {
                                "source_channel": "LFE",
                                "route_kind": "lfe_lowpass_to_sub",
                                "crossover_type": "LR24",
                                "low_pass_hz": 1000.0,
                                "gain_db": 0.0,
                                "delay_ms": 0.0,
                                "polarity_inverted": False,
                            },
                        ],
                    },
                }
            },
            "channels": {
                "L": {
                    "final_curve": {"freq": [20.0, 40.0], "spl": [70.0, 70.0]},
                    "plugins": [],
                },
                "LFE": {
                    "initial_curve": {"freq": [30.0, 60.0], "spl": [60.0, 60.0]},
                    "final_curve": {"freq": [30.0, 60.0], "spl": [60.0, 60.0]},
                    "plugins": [],
                },
            },
        }

        curves = build_post_dsp_source_curves(data)

        self.assertEqual(set(curves), {"L", "LFE"})
        self.assertEqual(curves["L"]["freq"], [20.0, 40.0])
        # Power sum of the 70 dB main and the resampled ~60 dB sub branch
        # (the 1000 Hz low-pass is ~0 dB down at 20-40 Hz).
        expected = 10.0 * math.log10(10.0 ** 7.0 + 10.0 ** 6.0)
        for observed in curves["L"]["spl"]:
            self.assertAlmostEqual(observed, expected, places=3)

    def test_multisub_lfe_replays_drivers_at_acoustic_level(self):
        # The channel aggregate of a multi-sub system is level-relative
        # optimizer state (10 dB here); the corrected LFE curve must replay
        # the ~70 dB driver measurements instead.
        data = {
            "metadata": {
                "bass_management": {
                    "physical_sub_output": "LFE",
                    "routing_graph": {
                        "routes": [
                            {
                                "source_channel": "LFE",
                                "route_kind": "lfe_lowpass_to_sub",
                                "crossover_type": "LR24",
                                "low_pass_hz": 1000.0,
                                "gain_db": 0.0,
                                "delay_ms": 0.0,
                                "polarity_inverted": False,
                            },
                        ],
                    },
                }
            },
            "channels": {
                "LFE": {
                    "initial_curve": {"freq": [50.0], "spl": [10.0]},
                    "final_curve": {"freq": [50.0], "spl": [4.0]},
                    "plugins": [
                        {
                            "plugin_type": "eq",
                            "parameters": {
                                "filters": [
                                    {
                                        "filter_type": "peak",
                                        "freq": 50.0,
                                        "q": 1.0,
                                        "db_gain": -6.0,
                                    }
                                ]
                            },
                        }
                    ],
                    "drivers": [
                        {
                            "name": "sub_1",
                            "initial_curve": {"freq": [50.0], "spl": [70.0]},
                            "plugins": [
                                {
                                    "plugin_type": "gain",
                                    "parameters": {"gain_db": 3.0},
                                }
                            ],
                        },
                        {
                            "name": "sub_2",
                            "initial_curve": {"freq": [50.0], "spl": [70.0]},
                            "plugins": [],
                        },
                    ],
                },
            },
        }

        curves = build_post_dsp_source_curves(data)

        self.assertEqual(set(curves), {"LFE"})
        # Power sum of 73 dB + 70 dB drivers, minus 6 dB shared EQ.
        expected = (
            10.0 * math.log10(10.0 ** 7.3 + 10.0 ** 7.0) - 6.0
        )
        self.assertAlmostEqual(curves["LFE"]["spl"][0], expected, places=2)
        self.assertGreater(curves["LFE"]["spl"][0], 60.0)

    def test_driver_baseline_sums_raw_measurements(self):
        channel = {
            "initial_curve": {"freq": [50.0], "spl": [10.0]},
            "drivers": [
                {"name": "sub_1", "initial_curve": {"freq": [50.0], "spl": [70.0]}},
                {"name": "sub_2", "initial_curve": {"freq": [50.0], "spl": [70.0]}},
            ],
        }

        baseline = sum_driver_initial_curves(channel)

        self.assertIsNotNone(baseline)
        assert baseline is not None
        self.assertEqual(baseline["freq"], [50.0])
        self.assertAlmostEqual(
            baseline["spl"][0], 10.0 * math.log10(2.0 * 10.0 ** 7.0), places=6
        )
        self.assertIsNone(sum_driver_initial_curves({"initial_curve": {}}))
        self.assertIsNone(sum_driver_initial_curves(None))

    def test_redirected_multisub_replays_drivers_at_acoustic_level(self):
        # Redirected bass must use the acoustic per-driver measurements, not
        # the level-relative channel aggregate (10 dB here).
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
                                "low_pass_hz": 1000.0,
                                "gain_db": 0.0,
                                "delay_ms": 0.0,
                                "polarity_inverted": False,
                            },
                            {
                                "source_channel": "LFE",
                                "route_kind": "lfe_lowpass_to_sub",
                                "crossover_type": "LR24",
                                "low_pass_hz": 1000.0,
                                "gain_db": 0.0,
                                "delay_ms": 0.0,
                                "polarity_inverted": False,
                            },
                        ],
                    },
                }
            },
            "channels": {
                "L": {
                    "final_curve": {"freq": [50.0], "spl": [80.0]},
                    "plugins": [],
                },
                "LFE": {
                    "initial_curve": {"freq": [50.0], "spl": [10.0]},
                    "final_curve": {"freq": [50.0], "spl": [4.0]},
                    "plugins": [],
                    "drivers": [
                        {
                            "name": "sub_1",
                            "initial_curve": {"freq": [50.0], "spl": [70.0]},
                            "plugins": [],
                        },
                        {
                            "name": "sub_2",
                            "initial_curve": {"freq": [50.0], "spl": [70.0]},
                            "plugins": [],
                        },
                    ],
                },
            },
        }

        curves = build_post_dsp_source_curves(data)

        # Power sum of two 70 dB drivers (~76 dB) acoustically summed with the
        # 80 dB main; the 10 dB aggregate must not appear.
        expected_sub = 10.0 * math.log10(2.0 * 10.0 ** 7.0)
        expected = 10.0 * math.log10(10.0 ** 8.0 + 10.0 ** (expected_sub / 10.0))
        self.assertAlmostEqual(curves["L"]["spl"][0], expected, places=2)
        self.assertGreater(curves["L"]["spl"][0], 70.0)

    def test_main_highpass_is_realized_before_acoustic_sum(self):
        # With an 80 Hz LR24 splice, bass at 20 Hz must come almost entirely
        # from the sub branch: the full-range main would otherwise
        # double-count it.
        data = {
            "metadata": {
                "bass_management": {
                    "physical_sub_output": "LFE",
                    "routing_graph": {
                        "routes": [
                            {
                                "source_channel": "L",
                                "route_kind": "main_highpass_to_self",
                                "crossover_type": "LR24",
                                "high_pass_hz": 80.0,
                                "delay_ms": 0.0,
                            },
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
                                "low_pass_hz": 1000.0,
                                "gain_db": 0.0,
                                "delay_ms": 0.0,
                                "polarity_inverted": False,
                            },
                        ],
                    },
                }
            },
            "channels": {
                "L": {
                    "final_curve": {"freq": [20.0], "spl": [70.0]},
                    "plugins": [],
                },
                "LFE": {
                    "initial_curve": {"freq": [20.0], "spl": [70.0]},
                    "final_curve": {"freq": [20.0], "spl": [70.0]},
                    "plugins": [],
                },
            },
        }

        curves = build_post_dsp_source_curves(data)

        # At 20 Hz (two octaves below an 80 Hz LR24 splice) the high-passed
        # main is ~48 dB down, so the sum stays within ~1 dB of the sub
        # branch instead of the ~6 dB lift a full-range double-count gives.
        self.assertLess(curves["L"]["spl"][0], 71.5)
        self.assertGreater(curves["L"]["spl"][0], 65.0)

    def test_routed_chain_highpass_is_not_applied_twice(self):
        # Routed executors stamp the route-owned high-pass into the chain;
        # the report must use that realized branch as-is instead of adding
        # the graph transfer a second time. The sub is silent here so the
        # main branch passes through observably: re-applying an LR24
        # high-pass at the probe (its own cutoff) would cost ~6 dB.
        data = {
            "metadata": {
                "bass_management": {
                    "physical_sub_output": "LFE",
                    "routing_graph": {
                        "routes": [
                            {
                                "source_channel": "L",
                                "route_kind": "main_highpass_to_self",
                                "crossover_type": "LR24",
                                "high_pass_hz": 80.0,
                                "delay_ms": 0.0,
                            },
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
                                "low_pass_hz": 1000.0,
                                "gain_db": 0.0,
                                "delay_ms": 0.0,
                                "polarity_inverted": False,
                            },
                        ],
                    },
                }
            },
            "channels": {
                "L": {
                    "final_curve": {"freq": [80.0], "spl": [70.0]},
                    "plugins": [
                        {
                            "plugin_type": "crossover",
                            "parameters": {
                                "type": "LR24",
                                "output": "high",
                                "frequency": 80.0,
                                "room_eq_stage": "route_owned",
                            },
                        }
                    ],
                },
                "LFE": {
                    "initial_curve": {"freq": [80.0], "spl": [-100.0]},
                    "final_curve": {"freq": [80.0], "spl": [-100.0]},
                    "plugins": [],
                },
            },
        }

        curves = build_post_dsp_source_curves(data)

        self.assertAlmostEqual(curves["L"]["spl"][0], 70.0, places=1)

    def test_resample_spl_onto_grid_bridges_display_grids(self):
        # Routed outputs mix driver-measurement and deployed replay grids;
        # display resampling must interpolate in-band and clamp out-of-band.
        out = resample_spl_onto_grid([100.0, 200.0], [70.0, 80.0], [100.0, 200.0])
        self.assertEqual(out, [70.0, 80.0])
        out = resample_spl_onto_grid(
            [100.0, 200.0], [70.0, 80.0], [50.0, 100.0, 141.4213562373095, 200.0, 400.0]
        )
        self.assertEqual(len(out), 5)
        self.assertAlmostEqual(out[0], 70.0)
        self.assertAlmostEqual(out[1], 70.0)
        self.assertAlmostEqual(out[2], 75.0, places=6)
        self.assertAlmostEqual(out[3], 80.0)
        self.assertAlmostEqual(out[4], 80.0)
        self.assertEqual(resample_spl_onto_grid([], [], [100.0]), [])

    def test_sub_without_own_route_falls_back_to_final(self):
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
                        ],
                    },
                }
            },
            "channels": {
                "L": {
                    "final_curve": {"freq": [80.0], "spl": [50.0], "phase": [0.0]},
                    "plugins": [],
                },
                "LFE": {
                    "initial_curve": {"freq": [80.0], "spl": [60.0], "phase": [0.0]},
                    "final_curve": {"freq": [80.0], "spl": [55.0], "phase": [0.0]},
                    "plugins": [],
                },
            },
        }

        curves = build_post_dsp_source_curves(data)

        self.assertEqual(curves["LFE"]["spl"], [55.0])


if __name__ == "__main__":
    unittest.main()
