#!/usr/bin/env python3

import tempfile
import unittest
from pathlib import Path

import plotly.graph_objects as go

from scripts.src.figures import (
    add_channel_response_overlays,
    create_bass_management_routing_figure,
    create_combined_figure,
)


def two_sub_overview_data():
    """Stereo + two-subwoofer fixture with a shared LFE EQ."""
    freq = [20.0, 40.0, 80.0, 160.0]
    main = {
        "initial_curve": {"freq": list(freq), "spl": [70.0] * len(freq)},
        "final_curve": {"freq": list(freq), "spl": [71.0] * len(freq)},
        "eq_response": {"freq": list(freq), "spl": [0.5] * len(freq)},
        "plugins": [],
    }
    sub_freq = [20.0, 40.0, 80.0, 160.0]
    shared_eq = {"filter_type": "peak", "freq": 40.0, "q": 1.0, "db_gain": -3.0}
    lfe = {
        "initial_curve": {"freq": list(sub_freq), "spl": [60.0] * len(sub_freq)},
        "final_curve": {"freq": list(sub_freq), "spl": [61.0] * len(sub_freq)},
        "plugins": [
            {
                "plugin_type": "eq",
                "parameters": {
                    "label": "room_eq_correction",
                    "room_eq_stage": "post_route",
                    "filters": [shared_eq],
                },
            },
        ],
        "drivers": [
            {
                "name": "Two subs_1",
                "initial_curve": {"freq": list(sub_freq), "spl": [70.0] * len(sub_freq)},
                "plugins": [
                    {
                        "plugin_type": "crossover",
                        "parameters": {
                            "frequency": 80.0,
                            "output": "low",
                            "room_eq_stage": "post_route",
                            "type": "LR24",
                        },
                    },
                ],
            },
            {
                "name": "Two subs_2",
                "initial_curve": {"freq": list(sub_freq), "spl": [66.0] * len(sub_freq)},
                "plugins": [
                    {
                        "plugin_type": "gain",
                        "parameters": {"gain_db": -4.0, "room_eq_stage": "post_route"},
                    },
                    {
                        "plugin_type": "crossover",
                        "parameters": {
                            "frequency": 90.0,
                            "output": "low",
                            "room_eq_stage": "post_route",
                            "type": "LR24",
                        },
                    },
                ],
            },
        ],
    }
    routes = [
        {
            "source_channel": "LFE",
            "destination": "Two subs_1",
            "route_kind": "lfe_lowpass_to_sub",
            "crossover_type": "LR24",
            "low_pass_hz": 120.0,
            "gain_db": 14.0,
            "delay_ms": 0.0,
            "polarity_inverted": False,
        },
        {
            "source_channel": "LFE",
            "destination": "Two subs_2",
            "route_kind": "lfe_lowpass_to_sub",
            "crossover_type": "LR24",
            "low_pass_hz": 120.0,
            "gain_db": 10.0,
            "delay_ms": 0.0,
            "polarity_inverted": False,
        },
    ]
    return {
        "channels": {"L": dict(main), "R": dict(main), "LFE": lfe},
        "metadata": {
            "bass_management": {
                "physical_sub_output": "LFE",
                "routing_graph": {"routes": routes},
            },
            "effective_config": {
                "system": {"speakers": {"L": "L", "R": "R", "LFE": "subs"}},
                "speakers": {
                    "subs": {
                        "name": "Two subs",
                        "subwoofers": [{"name": "Left Sub"}, {"name": "Right Sub"}],
                    }
                },
            },
        },
    }


class ChannelOverlayFigureTests(unittest.TestCase):
    def test_all_channels_corrected_row_contains_channel_target(self):
        with tempfile.TemporaryDirectory() as directory:
            target_path = Path(directory) / "target.csv"
            target_path.write_text(
                "frequency,spl\n20,0\n20000,-10\n", encoding="utf-8"
            )
            curve = {
                "freq": [20.0, 200.0, 2_000.0, 20_000.0],
                "spl": [80.0, 77.0, 73.0, 70.0],
            }
            data = {
                "channels": {
                    "L": {
                        "initial_curve": curve,
                        "final_curve": curve,
                        "eq_response": {
                            "freq": curve["freq"],
                            "spl": [0.0] * len(curve["freq"]),
                        },
                    }
                },
                "metadata": {
                    "effective_config": {
                        "target_curve": str(target_path),
                        "optimizer": {"min_freq": 20.0, "max_freq": 16_000.0},
                    }
                },
            }

            fig = create_combined_figure(data)

        self.assertIn("Target: L", [trace.name for trace in fig.data])

    def test_adds_target_and_lfe_plus_channel_traces(self):
        fig = go.Figure()
        target = {"freq": [20.0, 80.0, 20_000.0], "spl": [80.0, 78.0, 70.0]}
        combined = {
            "freq": [20.0, 80.0, 20_000.0],
            "spl": [79.0, 78.0, 70.5],
        }

        add_channel_response_overlays(fig, "L", target, combined)

        self.assertEqual([trace.name for trace in fig.data], ["Target", "LFE + L"])

    def test_lfe_view_can_add_target_without_combined_trace(self):
        fig = go.Figure()
        target = {"freq": [20.0, 120.0], "spl": [80.0, 78.0]}

        add_channel_response_overlays(fig, "LFE", target, None)

        self.assertEqual([trace.name for trace in fig.data], ["Target"])

    def test_corrected_row_keeps_all_channels_on_mismatched_multisub_grids(self):
        # Mains on a full-range grid, multi-sub aggregate and drivers each on
        # their own grid: every logical input must still get a corrected trace
        # at the acoustic level of the driver measurements.
        def route(source, kind, cutoff):
            return {
                "source_channel": source,
                "route_kind": kind,
                "crossover_type": "LR24",
                "low_pass_hz": cutoff,
                "gain_db": 0.0,
                "delay_ms": 0.0,
                "polarity_inverted": False,
            }

        data = {
            "channels": {
                "L": {
                    "initial_curve": {"freq": [20.0, 40.0], "spl": [70.0, 70.0]},
                    "final_curve": {"freq": [20.0, 40.0], "spl": [71.0, 71.0]},
                    "plugins": [],
                },
                "R": {
                    "initial_curve": {"freq": [20.0, 40.0], "spl": [70.0, 70.0]},
                    "final_curve": {"freq": [20.0, 40.0], "spl": [71.0, 71.0]},
                    "plugins": [],
                },
                "LFE": {
                    "initial_curve": {"freq": [30.0, 60.0], "spl": [10.0, 10.0]},
                    "final_curve": {"freq": [30.0, 60.0], "spl": [4.0, 4.0]},
                    "plugins": [],
                    "drivers": [
                        {
                            "name": "sub_1",
                            "initial_curve": {"freq": [20.0, 40.0], "spl": [70.0, 70.0]},
                            "plugins": [],
                        },
                        {
                            "name": "sub_2",
                            "initial_curve": {"freq": [20.0, 40.0], "spl": [66.0, 66.0]},
                            "plugins": [],
                        },
                    ],
                },
            },
            "metadata": {
                "bass_management": {
                    "physical_sub_output": "LFE",
                    "routing_graph": {
                        "routes": [
                            route("L", "redirected_bass_lowpass_to_sub", 80.0),
                            route("R", "redirected_bass_lowpass_to_sub", 80.0),
                            route("LFE", "lfe_lowpass_to_sub", 120.0),
                        ],
                    },
                }
            },
        }

        fig = create_combined_figure(data)
        names = [trace.name for trace in fig.data]

        self.assertIn("Corrected: L", names)
        self.assertIn("Corrected: R", names)
        self.assertIn("Corrected: LFE", names)
        corrected_lfe = next(
            trace for trace in fig.data if trace.name == "Corrected: LFE"
        )
        self.assertGreater(min(corrected_lfe.y), 50.0)

    def test_routing_graph_fans_out_multi_driver_subs(self):
        data = {
            "channels": {
                "LFE": {
                    "drivers": [
                        {
                            "name": "sub_1",
                            "plugins": [
                                {
                                    "plugin_type": "gain",
                                    "parameters": {"gain_db": 6.0},
                                },
                                {
                                    "plugin_type": "delay",
                                    "parameters": {"delay_ms": 2.5},
                                },
                            ],
                        },
                        {"name": "sub_2", "plugins": []},
                    ],
                },
            },
            "metadata": {
                "bass_management": {
                    "physical_sub_output": "LFE",
                    "routing_graph": {
                        "routes": [
                            {
                                "source_channel": "LFE",
                                "destination": "LFE",
                                "route_kind": "lfe_lowpass_to_sub",
                                "crossover_type": "LR24",
                                "low_pass_hz": 120.0,
                                "gain_db": 0.0,
                                "gain_linear": 1.0,
                                "delay_ms": 0.0,
                                "polarity_inverted": False,
                            },
                        ],
                    },
                }
            },
        }

        fig = create_bass_management_routing_figure(data)

        self.assertIsNotNone(fig)
        assert fig is not None
        sankey = fig.data[0]
        labels = list(sankey.node.label)
        self.assertIn("sub: sub_1", labels)
        self.assertIn("sub: sub_2", labels)
        bus = labels.index("out: LFE")
        links = {
            (source, target): hover
            for source, target, hover in zip(
                sankey.link.source, sankey.link.target, sankey.link.customdata
            )
        }
        self.assertIn((bus, labels.index("sub: sub_1")), links)
        self.assertIn((bus, labels.index("sub: sub_2")), links)
        driver_hover = links[(bus, labels.index("sub: sub_1"))]
        self.assertIn("gain: +6.00 dB", driver_hover)
        self.assertIn("delay: 2.500 ms", driver_hover)

    def test_routing_graph_without_drivers_has_no_sub_nodes(self):
        data = {
            "channels": {"LFE": {}},
            "metadata": {
                "bass_management": {
                    "routing_graph": {
                        "routes": [
                            {
                                "source_channel": "LFE",
                                "destination": "LFE",
                                "route_kind": "lfe_lowpass_to_sub",
                                "gain_linear": 1.0,
                            },
                        ],
                    },
                }
            },
        }

        fig = create_bass_management_routing_figure(data)

        self.assertIsNotNone(fig)
        assert fig is not None
        labels = list(fig.data[0].node.label)
        self.assertNotIn("sub: sub_1", labels)
        self.assertFalse(any(label.startswith("sub: ") for label in labels))


class MultiSubEqRowTests(unittest.TestCase):
    def test_eq_row_shows_one_trace_per_subwoofer(self):
        fig = create_combined_figure(two_sub_overview_data())
        eq_names = sorted(
            trace.name for trace in fig.data if trace.name.startswith("EQ: ")
        )

        self.assertEqual(
            eq_names, ["EQ: L", "EQ: Left Sub", "EQ: R", "EQ: Right Sub"]
        )
        self.assertNotIn("EQ: LFE", eq_names)

    def test_per_sub_eq_traces_differ(self):
        fig = create_combined_figure(two_sub_overview_data())
        by_name = {trace.name: trace for trace in fig.data}

        first = list(by_name["EQ: Left Sub"].y)
        second = list(by_name["EQ: Right Sub"].y)
        self.assertEqual(len(first), len(second))
        self.assertGreater(
            max(abs(a - b) for a, b in zip(first, second)), 1.0
        )

    def test_driver_traces_use_lines_and_cross_markers(self):
        fig = create_combined_figure(two_sub_overview_data())
        by_name = {trace.name: trace for trace in fig.data}

        first = by_name["EQ: Left Sub"]
        self.assertEqual(first.mode, "lines")

        second = by_name["EQ: Right Sub"]
        self.assertEqual(second.mode, "lines+markers")
        self.assertEqual(second.marker.symbol, "cross")
        opacity = list(second.marker.opacity)
        self.assertEqual(len(opacity), len(second.y))
        for index, value in enumerate(opacity):
            self.assertEqual(value, 1.0 if index % 10 == 0 else 0.0)

        original_first = by_name["Original: LFE/Two subs_1"]
        self.assertEqual(original_first.mode, "lines")
        original_second = by_name["Original: LFE/Two subs_2"]
        self.assertEqual(original_second.mode, "lines+markers")
        self.assertEqual(original_second.marker.symbol, "cross")

        # Whole-channel traces stay plain lines.
        self.assertEqual(by_name["EQ: L"].mode, "lines")

    def test_single_sub_keeps_collapsed_eq_trace(self):
        data = two_sub_overview_data()
        channel = data["channels"]["LFE"]
        channel["drivers"] = []
        channel["eq_response"] = {
            "freq": [20.0, 40.0],
            "spl": [1.0, -1.0],
        }

        fig = create_combined_figure(data)
        eq_names = sorted(
            trace.name for trace in fig.data if trace.name.startswith("EQ: ")
        )

        self.assertEqual(eq_names, ["EQ: L", "EQ: LFE", "EQ: R"])

    def test_drivers_without_eq_emit_no_eq_trace(self):
        data = two_sub_overview_data()
        channel = data["channels"]["LFE"]
        channel["plugins"] = []
        for driver in channel["drivers"]:
            driver["plugins"] = []

        fig = create_combined_figure(data)
        eq_names = sorted(
            trace.name for trace in fig.data if trace.name.startswith("EQ: ")
        )

        self.assertEqual(eq_names, ["EQ: L", "EQ: R"])


if __name__ == "__main__":
    unittest.main()
