"""Independent transfer checks for serialized advanced EQ report consumers."""

import cmath
import math
import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from scripts.src.dsp import apply_plugins_to_curve, biquad_coefficients, compute_eq_response


class AdvancedEqTests(unittest.TestCase):
    def test_comparison_report_passes_root_sample_rates(self):
        from scripts.src.figures import create_comparison_eq_overlay_figure
        from scripts.src.report import create_comparison_html_report
        from scripts.test_figures import two_sub_overview_data
        low = two_sub_overview_data()
        high = copy.deepcopy(low)
        low["sample_rate"], high["sample_rate"] = 32000.0, 96000.0
        with tempfile.TemporaryDirectory() as directory, patch(
            "scripts.src.report.create_comparison_eq_overlay_figure",
            wraps=create_comparison_eq_overlay_figure,
        ) as figure:
            output = Path(directory) / "comparison.html"
            create_comparison_html_report([("iir_auto", low), ("fir", high)], output)
            self.assertGreater(figure.call_count, 0)
            for call in figure.call_args_list:
                self.assertEqual(call.kwargs["sample_rates"], {"iir_auto": 32000.0, "fir": 96000.0})
            self.assertIn("EQ sections</th>", output.read_text())

    def test_comparison_eq_uses_each_output_rate(self):
        from scripts.src.figures import create_comparison_eq_overlay_figure
        filt = {"topology": "kautz_filter", "freq": 900.0, "q": 1.0, "db_gain": -0.2}
        channel = {"plugins": [{"plugin_type": "eq", "parameters": {"filters": [filt]}}]}
        rates = {"iir": 32000.0, "fir": 96000.0}
        fig = create_comparison_eq_overlay_figure(
            "L", [(mode, channel) for mode in rates], sample_rates=rates)
        for trace, rate in zip(fig.data, rates.values()):
            self.assertLessEqual(max(trace.x), rate/2)
            self.assertEqual(list(trace.y), compute_eq_response([filt], list(trace.x), rate))
        saved = {"eq_response": {"freq": [100.0, 200.0], "spl": [2.0, -3.0]}}
        fig = create_comparison_eq_overlay_figure("L", [("iir", saved)], sample_rates=rates)
        self.assertEqual(list(fig.data[0].y), [2.0, -3.0])

    def test_summary_counts_sections_and_driver_ownership(self):
        from scripts.src.report import _eq_filter_counts
        bank = {"topology": "kautz_filter", "kautz_sections": [
            {"pole_freq": 200, "q": 1, "gain": 0},
            {"pole_freq": 300, "q": 1, "gain": -0.2}]}
        legacy = {"topology": "kautz_filter", "freq": 100, "q": 1, "db_gain": 0}
        def eq(filters):
            return {"plugin_type": "eq", "parameters": {"filters": filters}}
        channel = {"plugins": [eq([bank])], "drivers": [
            {"plugins": [eq([legacy, {"filter_type": "peak"}])]},
            {"plugins": [eq([{**legacy, "kautz_sections": []}])]}]}
        self.assertEqual(_eq_filter_counts({"channels": {"L": channel, "R": {}}}), [5, 0])
        self.assertEqual(_eq_filter_counts({"channels": [channel]}), [5])
        marker = eq([bank])
        marker["parameters"]["room_eq_stage"] = "route_owned"
        channel["plugins"].append(marker)
        channel["drivers"][0]["plugins"].append(marker)
        self.assertEqual(_eq_filter_counts({"channels": [channel]}), [5])

    def test_report_kautz_weights_are_linear_not_db(self):
        from scripts.src.report import _format_eq_filter_line, _eq_filter_table_html
        filt = {"topology": "kautz_filter", "freq": 900.0, "q": 1.0, "db_gain": -0.2}
        line = _format_eq_filter_line(filt, 1)
        table = _eq_filter_table_html([{"label": None, "filters": [filt]}])
        for html in (line, table):
            self.assertIn("KAUTZ bank", html)
            self.assertIn("weight=-0.2 (linear)", html)
            self.assertNotIn("dB", html)
        escaped = _format_eq_filter_line({"filter_type": "<script>"}, 1)
        self.assertNotIn("<SCRIPT>", escaped)

    def test_figures_use_actual_sample_rate(self):
        from scripts.src.figures import create_eq_figure, create_multipass_eq_figure
        filt = {"topology": "kautz_filter", "freq": 900.0, "q": 1.0, "db_gain": -0.2}
        for rate in (32000.0, 44100.0, 96000.0):
            fig = create_eq_figure("L", [filt], sample_rate=rate)
            expected = compute_eq_response([filt], list(fig.data[0].x), rate)
            self.assertEqual(list(fig.data[0].y), expected)
            self.assertEqual(list(fig.data[1].y), expected)
            self.assertIn("KAUTZ bank", fig.data[1].name)
            self.assertNotIn("dB", fig.data[1].name)
            channel = {"plugins": [{"plugin_type": "eq", "parameters": {"filters": [filt]}}]}
            multipass = create_multipass_eq_figure("L", channel, sample_rate=rate)
            self.assertEqual(list(multipass.data[0].y), expected)

    def check_response(self, filt, rate, frequencies, expected):
        db = compute_eq_response([filt], frequencies, rate)
        curve = {"freq": frequencies, "spl": [0.0] * len(frequencies),
                 "phase": [0.0] * len(frequencies)}
        replay = apply_plugins_to_curve(curve, [
            {"plugin_type": "eq", "parameters": {"filters": [filt]}}
        ], rate)
        for index, value in enumerate(expected):
            self.assertAlmostEqual(db[index], 20 * math.log10(abs(value)), places=7)
            actual = cmath.rect(10 ** (replay["spl"][index] / 20),
                                math.radians(replay["phase"][index]))
            self.assertLess(abs(actual - value), 1e-8)

    def test_kautz_streamed_impulse_multirate(self):
        # Independent sample recurrence, including a zero-weight allpass stage.
        sections = [{"pole_freq": 900.0, "q": 0.7, "gain": 0.0},
                    {"pole_freq": 2100.0, "q": 1.2, "gain": -0.35}]
        for rate in (44100.0, 48000.0, 96000.0):
            impulse = np.zeros(8192)
            impulse[0] = 1.0
            chain = impulse.copy()
            result = impulse.copy()
            for section in sections:
                radius = min(math.exp(-math.pi * section["pole_freq"] /
                                      (section["q"] * rate)), 0.9999)
                a1 = -2 * radius * math.cos(2 * math.pi * section["pole_freq"] / rate)
                a2 = radius * radius
                state = np.zeros_like(chain)
                for n, value in enumerate(chain):
                    state[n] = value - (a1 * state[n-1] if n else 0) - (
                        a2 * state[n-2] if n > 1 else 0)
                result += section["gain"] * (1-a2)**1.5 * state
                chain = a2 * state
                chain[1:] += a1 * state[:-1]
                chain[2:] += state[:-2]
            frequencies = [0.0, 450.0, 900.0, 2100.0, 5000.0, rate/2]
            expected = [np.dot(result, np.exp(-2j * math.pi * f *
                        np.arange(len(result)) / rate)) for f in frequencies]
            self.check_response({"topology": "kautz_filter", "kautz_sections": sections},
                                rate, frequencies, expected)

    def test_warped_allpass_substitution_multirate(self):
        for rate in (44100.0, 48000.0, 96000.0):
            for lam in (-0.4, 0.0, 0.8):
                center = 1700.0
                zc = cmath.exp(-2j * math.pi * center / rate)
                dc = (zc-lam) / (1-lam*zc)
                design = -cmath.phase(dc) * rate / (2*math.pi)
                a1, a2, b0, b1, b2 = biquad_coefficients("peak", design, rate, 1.3, -5.0)
                frequencies = [0.0, 500.0, center, 6000.0, rate/2]
                expected = []
                for f in frequencies:
                    z = cmath.exp(-2j * math.pi * f / rate)
                    d = (z-lam) / (1-lam*z)
                    expected.append((b0+b1*d+b2*d*d)/(1+a1*d+a2*d*d))
                self.check_response({"topology": "warped_biquad", "filter_type": "peak",
                                     "freq": center, "q": 1.3, "db_gain": -5.0,
                                     "lambda": lam}, rate, frequencies, expected)

    def test_invalid_topologies_are_not_peq_fallbacks(self):
        for filt in ({"topology": "unknown"}, {"topology": None},
                     {"topology": "kautz_filter", "kautz_sections": None},
                     {"topology": "kautz_filter", "kautz_sections": [], "sections": []},
                     {"topology": "kautz_filter", "freq": 30000, "q": 1},
                     {"topology": "kautz_filter", "freq": 200, "q": True},
                     {"topology": "warped_biquad", "freq": 200, "q": 1,
                      "db_gain": 0, "lambda": 1.0}):
            with self.subTest(filt=filt), self.assertRaises(ValueError):
                compute_eq_response([filt], [100.0], 48000.0)

    def test_kautz_aliases_defaults_and_legacy_linear_weight(self):
        for key in ("pole_freq", "freq", "frequency", "pole_freq_hz"):
            filt = {"topology": "kautz_filter", "sections": [{key: 200.0, "q": 1.0}]}
            self.assertEqual(compute_eq_response([filt], [100.0, 200.0]), [0.0, 0.0])
        legacy = {"topology": "kautz_filter", "freq": 200.0, "q": 1.0, "db_gain": -0.2}
        canonical = {"topology": "kautz_filter", "kautz_sections": [
            {"pole_freq": 200.0, "q": 1.0, "gain": -0.2}]}
        self.assertEqual(compute_eq_response([legacy], [100.0, 200.0]),
                         compute_eq_response([canonical], [100.0, 200.0]))
