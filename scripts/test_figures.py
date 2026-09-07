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


if __name__ == "__main__":
    unittest.main()
