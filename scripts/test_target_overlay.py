#!/usr/bin/env python3

import tempfile
import unittest
from pathlib import Path

from scripts.src.target_overlay import (
    align_target_to_curve,
    build_target_overlay_curves,
)


class TargetOverlayTests(unittest.TestCase):
    def test_interpolates_target_in_log_frequency_and_aligns_level(self):
        target = {"freq": [20.0, 20_000.0], "spl": [0.0, -10.0]}
        reference = {
            "freq": [20.0, 200.0, 2_000.0, 20_000.0],
            "spl": [80.0, 76.6666666667, 73.3333333333, 70.0],
        }

        aligned = align_target_to_curve(target, reference)

        self.assertIsNotNone(aligned)
        for observed, expected in zip(
            aligned["spl"], reference["spl"], strict=True
        ):
            self.assertAlmostEqual(observed, expected, places=8)

    def test_lfe_stopband_does_not_pull_target_alignment_down(self):
        target = {"freq": [20.0, 20_000.0], "spl": [0.0, -10.0]}
        reference = {
            "freq": [20.0, 40.0, 80.0, 160.0, 1_000.0, 20_000.0],
            "spl": [80.0, 79.0, 77.0, 60.0, -40.0, -120.0],
        }

        aligned = align_target_to_curve(target, reference)

        self.assertIsNotNone(aligned)
        self.assertGreater(aligned["spl"][0], 75.0)

    def test_lfe_target_overlay_includes_programme_lowpass(self):
        with tempfile.TemporaryDirectory() as directory:
            target_path = Path(directory) / "target.csv"
            target_path.write_text(
                "frequency,spl\n20,0\n20000,-10\n", encoding="utf-8"
            )
            data = {
                "metadata": {
                    "effective_config": {"target_curve": str(target_path)},
                    "bass_management": {
                        "routing_graph": {
                            "routes": [
                                {
                                    "source_channel": "LFE",
                                    "route_kind": "lfe_lowpass_to_sub",
                                    "crossover_type": "LR24",
                                    "low_pass_hz": 120.0,
                                }
                            ]
                        }
                    },
                }
            }
            reference = {
                "LFE": {
                    "freq": [40.0, 80.0, 120.0, 160.0, 240.0],
                    "spl": [80.0, 78.0, 73.0, 66.0, 54.0],
                }
            }

            overlay = build_target_overlay_curves(data, reference)["LFE"]

            self.assertGreater(overlay["spl"][1] - overlay["spl"][3], 8.0)

    def test_redirected_main_target_uses_the_optimizer_reference_band(self):
        with tempfile.TemporaryDirectory() as directory:
            target_path = Path(directory) / "target.csv"
            target_path.write_text("frequency,spl\n20,0\n20000,0\n", encoding="utf-8")
            data = {
                "metadata": {
                    "effective_config": {
                        "target_curve": str(target_path),
                        "optimizer": {"min_freq": 20.0, "max_freq": 20_000.0},
                    },
                    "bass_management": {
                        "routing_graph": {
                            "routes": [
                                {
                                    "source_channel": "L",
                                    "route_kind": "redirected_bass_lowpass_to_sub",
                                    "low_pass_hz": 100.0,
                                }
                            ]
                        }
                    },
                }
            }
            reference = {
                "L": {
                    "freq": [50.0, 100.0, 200.0, 400.0, 800.0, 1_600.0],
                    "spl": [60.0, 60.0, 70.0, 69.0, 68.0, 67.0],
                }
            }

            overlay = build_target_overlay_curves(data, reference)["L"]

            self.assertAlmostEqual(overlay["spl"][0], 69.0, places=8)

    def test_loads_effective_config_target_and_builds_each_channel(self):
        with tempfile.TemporaryDirectory() as directory:
            target_path = Path(directory) / "target.csv"
            target_path.write_text(
                "frequency,spl\n20,0\n20000,-10\n", encoding="utf-8"
            )
            data = {
                "metadata": {
                    "effective_config": {
                        "target_curve": str(target_path),
                        "optimizer": {"min_freq": 20.0, "max_freq": 16_000.0},
                    }
                }
            }
            references = {
                "L": {"freq": [20.0, 200.0], "spl": [80.0, 76.0]},
                "R": {"freq": [20.0, 200.0], "spl": [79.0, 75.0]},
            }

            overlays = build_target_overlay_curves(data, references)

        self.assertEqual(set(overlays), {"L", "R"})
        self.assertEqual(overlays["L"]["freq"], references["L"]["freq"])


if __name__ == "__main__":
    unittest.main()
