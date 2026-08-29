#!/usr/bin/env python3

import tempfile
import unittest
from pathlib import Path

import plotly.graph_objects as go

from scripts.src.figures import add_channel_response_overlays, create_combined_figure


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


if __name__ == "__main__":
    unittest.main()
