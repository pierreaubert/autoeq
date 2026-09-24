"""Fast validation of capture diagnostics, not acoustic evidence."""

import copy
import unittest

from scripts.src.capture_views import (
    capture_views_html,
    optimization_waterfall_html,
    optimization_wavelet_html,
)
from scripts.src.payload_binding import ALGORITHM, payload_digest


def fixture():
    views = {
        "candidate_graph_id": "candidate", "baseline_graph_id": "baseline",
        "source": "<source>", "seat": "seat", "scope": "<operator scope>",
        "evidence_kind": "synthetic_capture_pair", "unavailable": {"decay": "<missing>"},
        "ir_step": {
            "provenance": {"graph_identity": "candidate"}, "common_reference": "<clock>",
            "times_ms": [0.0, 1.0, 2.0], "pre_ir": [0.0, 0.25, 0.0],
            "post_ir": [0.5, 0.0, 0.0], "pre_step": [0.0, 0.25, 0.25],
            "post_step": [0.5, 0.5, 0.5],
        },
    }
    return {"graph_id": "candidate", "status": "rejected", "exit_code": 1,
            "verified_captures": 1, "detail": "held-out seat failed", "comparisons": [{
        "source": "<source>", "seat": "seat", "capture_views": views,
        "capture_views_binding": {"algorithm": ALGORITHM, "graph_identity": "candidate",
                                  "sha256": payload_digest(views, "candidate")},
    }]}


class CaptureViewsTests(unittest.TestCase):
    def test_recorded_status_is_visible_without_promoting_capture_views(self):
        report = fixture()
        html = capture_views_html(report)
        self.assertIn("Recorded capture verification status: <strong>rejected</strong>", html)
        self.assertIn("held-out seat failed", html)
        self.assertIn("Matched IR evidence: synthetic_capture_pair", html)
        report["status"] = "invented_pass"
        self.assertIn("unsupported verification status", capture_views_html(report))

    def test_room_mean_t60_requires_matching_bound_capture_context(self):
        report = fixture()
        first = report["comparisons"][0]
        first["source"] = "L"
        first["capture_views"]["source"] = "L"
        first["capture_views"].update(settings={"timing_reference_id": "shared"},
                                      valid_band_hz=[20.0, 20000.0],
                                      stimulus_hash="same-stimulus",
                                      sample_rate_hz=48000.0)
        centers = [63, 125, 250, 500, 1000, 2000, 4000, 8000, 16000]
        def rows(value):
            return [{"centre_hz": hz, "t60_s": value, "fit_range": "T30",
                     "r2": 0.98, "valid": True, "reason": ""} for hz in centers]
        first["capture_views"]["octave_t60"] = {
            "method": "octave_schroeder_t30_t20_v1", "min_r2": 0.9,
            "valid_band_hz": [20.0, 20000.0], "scope": "declared",
            "pre": rows(0.4), "post": rows(0.3),
        }
        second = copy.deepcopy(first)
        second["source"] = "R"
        second["capture_views"]["source"] = "R"
        second["capture_views"]["octave_t60"]["pre"] = rows(0.6)
        second["capture_views"]["octave_t60"]["post"] = rows(0.5)
        second["capture_views"]["octave_t60"]["post"][0] = {
            "centre_hz": 63, "t60_s": None, "fit_range": None,
            "r2": 0.0, "valid": False, "reason": "insufficient decay",
        }
        report["comparisons"].append(second)
        def bind():
            for comparison in report["comparisons"]:
                comparison["capture_views_binding"]["sha256"] = payload_digest(
                    comparison["capture_views"], "candidate")
        bind()
        html = capture_views_html(report)
        self.assertIn("Capture room mean octave T60: seat", html)
        self.assertIn("<td>125</td><td>0.500</td><td>2</td><td>0.400</td><td>2</td>", html)
        self.assertIn("<td>63</td><td>0.500</td><td>2</td><td>0.300</td><td>1</td>", html)
        second["capture_views"]["settings"]["timing_reference_id"] = "other"
        bind()
        self.assertNotIn("Capture room mean octave T60", capture_views_html(report))
        second["capture_views"]["settings"]["timing_reference_id"] = "shared"
        second["capture_views"]["stimulus_hash"] = "other-stimulus"
        bind()
        self.assertNotIn("Capture room mean octave T60", capture_views_html(report))
        second["capture_views"]["stimulus_hash"] = "same-stimulus"
        second["capture_views"]["sample_rate_hz"] = 44100.0
        bind()
        self.assertNotIn("Capture room mean octave T60", capture_views_html(report))
        second["capture_views"]["sample_rate_hz"] = 48000.0
        bind()
        second["capture_views"]["octave_t60"]["pre"][0]["t60_s"] = 0.9
        self.assertIn("binding changed", capture_views_html(report))

    def test_bound_wavelet_heatmap_and_invalid_cell(self):
        report = fixture()
        comparison = report["comparisons"][0]
        views = comparison["capture_views"]
        side = {"freqs_hz": [100.0, 200.0, 400.0],
                "times_ms": [-5.0, 0.0, 100.0, 500.0],
                "mags_db": [[-30.0, -20.0, -25.0, -30.0],
                            [-28.0, 0.0, -15.0, -30.0],
                            [-30.0, -10.0, -20.0, -30.0]]}
        views["wavelet"] = {
            "method": "complex_morlet_three_cycle_v1",
            "reference": "each_full_grid_peak", "valid_band_hz": [20.0, 1000.0],
            "cycles": 3.0, "freqs_per_octave": 6.0, "hop_ms": 1.0,
            "display_range_db": [-30.0, 0.0], "scope": "<wavelet scope>",
            "pre": copy.deepcopy(side), "post": copy.deepcopy(side),
        }
        def bind():
            comparison["capture_views_binding"]["sha256"] = payload_digest(views, "candidate")
        bind()
        html = capture_views_html(report)
        self.assertIn("Captured IR three-cycle wavelet", html)
        self.assertIn("&lt;wavelet scope&gt;", html)
        self.assertIn("rgb(", html)
        views["wavelet"]["post"]["mags_db"][0][0] = -3.0
        self.assertIn("binding changed", capture_views_html(report))
        bind()
        views["wavelet"]["post"]["mags_db"][0][0] = 1.0
        bind()
        self.assertIn("invalid capture wavelet samples", capture_views_html(report))

    def test_bound_waterfall_axes_decay_and_invalid_grid(self):
        report = fixture()
        comparison = report["comparisons"][0]
        views = comparison["capture_views"]
        side = {
            "grid": {"times_ms": [-5.0, 60.0, 200.0, 500.0],
                     "freqs_hz": [100.0, 125.0, 250.0],
                     "mags_db": [[-30.0, -10.0, -50.0], [-35.0, -15.0, -55.0],
                                 [-50.0, -40.0, -70.0], [-80.0, -75.0, -90.0]]},
            "resonances": [{"freq_hz": 125.0, "level_db": -15.0,
                            "decay_time_s": 0.35}],
        }
        views["waterfall"] = {
            "method": "hann_stft_waterfall_v1", "reference": "each_full_grid_peak",
            "valid_band_hz": [20.0, 1000.0], "window_ms": 32.0,
            "hop_ms": 2.0, "post_ms": 500.0, "scope": "<capture scope>",
            "pre": copy.deepcopy(side), "post": copy.deepcopy(side),
        }
        def bind():
            comparison["capture_views_binding"]["sha256"] = payload_digest(views, "candidate")
        bind()
        html = capture_views_html(report)
        self.assertIn("Captured IR waterfall and resonance decay", html)
        self.assertIn("&lt;capture scope&gt;", html)
        self.assertIn("own-grid relative level", html)
        self.assertIn("0.35", html)
        views["waterfall"]["pre"]["grid"]["mags_db"][0][0] = -20.0
        self.assertIn("binding changed", capture_views_html(report))
        bind()
        views["waterfall"]["pre"]["grid"]["mags_db"][0][0] = 1.0
        bind()
        self.assertIn("invalid capture waterfall samples", capture_views_html(report))

    def test_band_limited_early_reflections_keep_six_columns_and_binding(self):
        report = fixture()
        comparison = report["comparisons"][0]
        views = comparison["capture_views"]
        event = {"gain_dbfs": -6.02, "time_ms": 5.0,
                 "distance_cm": 171.5, "first_dip_hz": 100.0,
                 "ripple_db": 9.54}
        views["early_reflections"] = {
            "method": "bandlimited_early_reflection_table_v1",
            "band_hz": [1000.0, 8000.0], "threshold_dbfs": -15.0,
            "direct_reference": "<direct peak>", "pre_direct_sample": 480,
            "post_direct_sample": 490, "scope": "<diagnostic scope>",
            "pre": [event], "post": [dict(event, gain_dbfs=-8.0)],
        }
        comparison["capture_views_binding"]["sha256"] = payload_digest(views, "candidate")
        html = capture_views_html(report)
        self.assertIn("Early reflection candidates (1–8 kHz)", html)
        self.assertIn("Comb ripple (dB p-p)", html)
        self.assertIn("&lt;direct peak&gt;", html)
        self.assertIn("&lt;diagnostic scope&gt;", html)
        views["early_reflections"]["post"][0]["time_ms"] = 6.0
        self.assertIn("binding changed", capture_views_html(report))
        comparison["capture_views_binding"]["sha256"] = payload_digest(views, "candidate")
        views["early_reflections"]["post"][0]["first_dip_hz"] = -1.0
        comparison["capture_views_binding"]["sha256"] = payload_digest(views, "candidate")
        self.assertIn("invalid early reflection values", capture_views_html(report))

    def test_bound_shared_reference_early_late_curves(self):
        report = fixture()
        comparison = report["comparisons"][0]
        views = comparison["capture_views"]
        freq = [1000.0, 2000.0, 4000.0, 8000.0]
        def curve(values):
            return {"freq": freq, "spl": values}
        side = {"direct_sample": 480,
                "full": curve([0.0, -1.0, -2.0, -3.0]),
                "early": curve([-1.0, -2.0, -3.0, -4.0]),
                "late": curve([-7.0, -8.0, -9.0, -10.0])}
        views["early_late_curves"] = {
            "method": "incoherent_band_energy", "reference": "full_peak_band",
            "smoothing": "third_octave", "split_ms": 20.0,
            "direct_reference": "broadband envelope peak",
            "valid_band_hz": [80.0, 12000.0], "scope": "<matched captures>",
            "pre": copy.deepcopy(side), "post": copy.deepcopy(side),
        }
        def bind():
            comparison["capture_views_binding"]["sha256"] = payload_digest(views, "candidate")
        bind()
        html = capture_views_html(report)
        self.assertIn("Captured IR early and late energy", html)
        self.assertIn("1–8 kHz mean early minus late: +6.00 dB", html)
        self.assertIn("&lt;matched captures&gt;", html)
        views["early_late_curves"]["post"]["late"]["spl"][0] = -5.0
        self.assertIn("binding changed", capture_views_html(report))
        bind()
        views["early_late_curves"]["reference"] = "late_peak"
        bind()
        self.assertIn("invalid capture early/late analysis contract", capture_views_html(report))

    def test_octave_t60_keeps_invalid_bands_and_capture_binding(self):
        report = fixture()
        comparison = report["comparisons"][0]
        views = comparison["capture_views"]
        centers = [63, 125, 250, 500, 1000, 2000, 4000, 8000, 16000]
        bands = [{"centre_hz": hz, "t60_s": 0.5, "fit_range": "T30",
                  "r2": 0.97, "valid": True, "reason": ""} for hz in centers]
        bands[-1] = {"centre_hz": 16000, "t60_s": None, "fit_range": None,
                     "r2": 0.0, "valid": False, "reason": "above-nyquist <limit>"}
        views["octave_t60"] = {
            "method": "octave_schroeder_t30_t20_v1", "min_r2": 0.9,
            "valid_band_hz": [20.0, 20000.0], "scope": "<declared IR evidence>",
            "pre": copy.deepcopy(bands), "post": copy.deepcopy(bands),
        }
        comparison["capture_views_binding"]["sha256"] = payload_digest(views, "candidate")
        html = capture_views_html(report)
        self.assertIn("Capture IR octave T60 diagnostic", html)
        self.assertIn("&lt;limit&gt;", html)
        self.assertIn("&lt;declared IR evidence&gt;", html)
        self.assertIn("gaps are unavailable fits", html)
        views["octave_t60"]["post"][0]["t60_s"] = 0.8
        self.assertIn("binding changed", capture_views_html(report))
        comparison["capture_views_binding"]["sha256"] = payload_digest(views, "candidate")
        views["octave_t60"]["post"][0]["fit_range"] = "EDT"
        comparison["capture_views_binding"]["sha256"] = payload_digest(views, "candidate")
        self.assertIn("invalid accepted octave T60 fit", capture_views_html(report))

    def test_noise_retains_units_calibration_binding_and_unavailable_bands(self):
        report = fixture()
        comparison = report["comparisons"][0]
        views = comparison["capture_views"]
        views["ambient_noise"] = {
            "evidence_kind": "synthetic_noise_capture", "conditions": "<HVAC state>",
            "analysis": {
                "method": "periodic_hann_welch_pressure_v1", "scope": "<operator declared>",
                "duration_seconds": 2, "bin_spacing_hz": 10, "window_enbw_hz": 15,
                "calibration": {"pascals_per_sample": 2, "self_noise_note": "<not characterized>"},
                "bin_freqs_hz": [990, 1000, 1010], "pressure_psd_pa2_per_hz": [0, 1e-6, 2e-6],
                "spectrum": {"provenance": {"graph_identity": "candidate"},
                             "freqs": [1000], "noise_spl_db": [50]},
                "octave_edges_hz": [[707, 1415]], "octave_bin_centers_hz": [[990, 1010]],
                "unavailable_bands": {"63": "<insufficient support>"}
            }
        }
        def bind():
            comparison["capture_views_binding"]["sha256"] = payload_digest(views, "candidate")
        bind()
        html = capture_views_html(report)
        self.assertIn("Calibrated ambient noise", html)
        self.assertIn("dB re (20 µPa)²/Hz", html)
        self.assertIn("not certified octave filters", html)
        self.assertIn("&lt;HVAC state&gt;", html)
        self.assertIn("&lt;not characterized&gt;", html)
        self.assertIn("&lt;insufficient support&gt;", html)
        noise = views["ambient_noise"]["analysis"]
        noise["calibration"]["pascals_per_sample"] = 20
        self.assertIn("binding changed", capture_views_html(report))
        bind()
        noise["method"] = "unknown"
        bind()
        self.assertIn("analysis method mismatch", capture_views_html(report))
        noise["method"] = "periodic_hann_welch_pressure_v1"
        noise["pressure_psd_pa2_per_hz"] = [0, -1, 0]
        bind()
        self.assertIn("invalid calibrated noise PSD", capture_views_html(report))
        noise["pressure_psd_pa2_per_hz"] = [0, 0, 0]
        noise["spectrum"]["freqs"] = []
        noise["spectrum"]["noise_spl_db"] = []
        noise["octave_edges_hz"] = []
        noise["octave_bin_centers_hz"] = []
        bind()
        self.assertIn("not proof of a noiseless room", capture_views_html(report))

    def test_decay_distinguishes_absolute_normalized_and_unavailable_fit(self):
        report = fixture()
        comparison = report["comparisons"][0]
        views = comparison["capture_views"]
        views["decay"] = {
            "provenance": {"graph_identity": "candidate"},
            "method": "finite_window_octave_schroeder_v1", "scope": "<finite observation>",
            "settings": {"noise_window_ms": [80, 100], "minimum_fit_margin_db": 10,
                         "minimum_r_squared": 0.95},
            "bands": [{"center_hz": 500, "times_ms": [0, 10, 20],
                       "pre_db": [0, -10, -20], "post_db": [-6, -16, -26],
                       "pre_normalized_db": [0, -10, -20], "post_normalized_db": [0, -10, -20],
                       "pre_fit": None, "post_fit": None,
                       "pre_fit_unavailable": "<noise limited>",
                       "post_fit_unavailable": "<noise limited>"}],
        }
        del views["unavailable"]["decay"]
        comparison["capture_views_binding"]["sha256"] = payload_digest(views, "candidate")
        html = capture_views_html(report)
        self.assertIn("Matched octave decay", html)
        self.assertIn("Normalized decay 500 Hz", html)
        self.assertIn("not passive-room RT", html)
        self.assertIn("&lt;noise limited&gt;", html)
        self.assertEqual(html.count("<svg "), 4)
        views["decay"]["bands"][0]["post_db"][0] = -3
        self.assertIn("binding changed", capture_views_html(report))
        views["decay"]["method"] = "unsupported"
        comparison["capture_views_binding"]["sha256"] = payload_digest(views, "candidate")
        self.assertIn("decay graph identity or analysis method mismatch", capture_views_html(report))

    def test_etc_scope_and_common_reference_are_rendered(self):
        report = fixture()
        comparison = report["comparisons"][0]
        views = comparison["capture_views"]
        views["etc"] = {
            "provenance": {"graph_identity": "candidate"}, "method": "hann_analytic_octave_linear_v1",
            "window_ms": [0.0, 40.0], "display_floor_db": -160.0,
            "scope": "<filter spreading>", "filter_half_support_ms": [8.0, 4.0, 2.0, 1.0],
            "bands": [{"center_hz": center, "times_ms": [0.0, 20.0, 40.0],
                       "pre_db": [-20.0, 0.0, -40.0], "post_db": [-14.0, 6.0, -34.0]}
                      for center in [500.0, 1000.0, 2000.0, 4000.0]],
        }
        comparison["capture_views_binding"]["sha256"] = payload_digest(views, "candidate")
        html = capture_views_html(report)
        self.assertEqual(html.count("<svg "), 6)
        self.assertIn("same baseline-band peak", html)
        self.assertIn("&lt;filter spreading&gt;", html)
        views["etc"]["method"] = "periodic-mask"
        comparison["capture_views_binding"]["sha256"] = payload_digest(views, "candidate")
        self.assertIn("analysis method mismatch", capture_views_html(report))

    def test_bound_pair_is_escaped_and_not_promoted(self):
        html = capture_views_html(fixture())
        self.assertEqual(html.count("<svg "), 2)
        self.assertIn("synthetic_capture_pair", html)
        self.assertIn("&lt;source&gt;", html)
        self.assertIn("&lt;clock&gt;", html)
        self.assertIn("&lt;missing&gt;", html)
        self.assertIn("do not establish safety", html)

    def test_mutated_trace_or_graph_refuses(self):
        for field in ("trace", "graph", "seat"):
            report = fixture()
            comparison = report["comparisons"][0]
            if field == "trace":
                comparison["capture_views"]["ir_step"]["post_ir"][0] = 0.75
            elif field == "graph":
                report["graph_id"] = "stale"
            else:
                comparison["seat"] = "wrong"
            html = capture_views_html(report)
            self.assertIn("binding changed", html)
            self.assertNotIn("<svg ", html)

    def test_legacy_has_no_invented_trace(self):
        report = fixture()
        del report["comparisons"][0]["capture_views"]
        html = capture_views_html(report)
        self.assertIn("legacy comparison", html)
        self.assertNotIn("<svg ", html)

    def test_bound_invalid_axis_refuses(self):
        report = copy.deepcopy(fixture())
        comparison = report["comparisons"][0]
        comparison["capture_views"]["ir_step"]["times_ms"] = [0.0, 1.0, 0.5]
        comparison["capture_views_binding"]["sha256"] = payload_digest(comparison["capture_views"], "candidate")
        self.assertIn("invalid common time axis", capture_views_html(report))


def optimization_waterfall_fixture():
    times = [-5.0, 0.0, 60.0, 500.0]
    freqs = [100.0, 200.0, 400.0]
    rows = [[-0.5, 0.0, -1.0], [-1.0, -0.2, -2.0],
            [-6.0, -3.0, -9.0], [-20.0, -12.0, -25.0]]
    waterfall = {
        "basis": "measured_room_ir", "method": "hann_stft_waterfall_v1",
        "reference": "full_grid_peak", "valid_band_hz": [100.0, 400.0],
        "window_ms": 32.0, "hop_ms": 2.0, "post_ms": 500.0,
        "times_ms": times, "freqs_hz": freqs, "mags_db": rows,
        "scope": "<waterfall scope>",
    }
    decays = {
        "basis": "measured_room_ir", "method": "hann_stft_waterfall_v1",
        "reference": "full_grid_peak", "slice_ms": 60.0,
        "decays": [{"freq_hz": 200.0, "level_db": -3.0, "decay_time_s": 0.5}],
    }
    return waterfall, decays


def optimization_wavelet_fixture():
    return {
        "basis": "measured_room_ir", "method": "complex_morlet_three_cycle_v1",
        "reference": "full_grid_peak", "valid_band_hz": [100.0, 200.0],
        "cycles": 3.0, "freqs_per_octave": 6.0, "hop_ms": 1.0,
        "display_range_db": [-30.0, 0.0],
        "freqs_hz": [100.0, 200.0], "times_ms": [-5.0, 0.0, 100.0],
        "mags_db": [[-1.0, 0.0, -5.0], [-3.0, -2.0, -12.0]],
    }


class OptimizationTimeFrequencyTests(unittest.TestCase):
    def test_waterfall_renders_wireframe_and_decay_table(self):
        waterfall, decays = optimization_waterfall_fixture()
        html = optimization_waterfall_html(waterfall, decays)
        self.assertIn("Measured room IR waterfall and resonance decay", html)
        self.assertIn("&lt;waterfall scope&gt;", html)
        self.assertIn("200", html)
        self.assertIn("0.5", html)
        self.assertIn("<svg ", html)

    def test_waterfall_contract_violation_stays_pending(self):
        waterfall, decays = optimization_waterfall_fixture()
        self.assertEqual("", optimization_waterfall_html(None, decays))
        self.assertEqual("", optimization_waterfall_html(waterfall, None))
        bad = copy.deepcopy(waterfall)
        bad["reference"] = "each_full_grid_peak"
        self.assertEqual("", optimization_waterfall_html(bad, decays))
        bad = copy.deepcopy(waterfall)
        bad["mags_db"][0][1] = 1.0
        self.assertEqual("", optimization_waterfall_html(bad, decays))
        bad_decays = copy.deepcopy(decays)
        bad_decays["decays"][0]["decay_time_s"] = -2.0
        self.assertEqual("", optimization_waterfall_html(waterfall, bad_decays))
        bad_decays = copy.deepcopy(decays)
        bad_decays["slice_ms"] = 30.0
        self.assertEqual("", optimization_waterfall_html(waterfall, bad_decays))

    def test_waterfall_empty_decays_still_renders_grid(self):
        waterfall, decays = optimization_waterfall_fixture()
        decays["decays"] = []
        html = optimization_waterfall_html(waterfall, decays)
        self.assertIn("Measured room IR waterfall and resonance decay", html)
        self.assertIn("<svg ", html)

    def test_wavelet_renders_heatmap(self):
        html = optimization_wavelet_html(optimization_wavelet_fixture())
        self.assertIn("Measured room IR three-cycle wavelet", html)
        self.assertIn("<svg ", html)

    def test_wavelet_contract_violation_stays_pending(self):
        self.assertEqual("", optimization_wavelet_html(None))
        self.assertEqual("", optimization_wavelet_html({"basis": "measured_room_ir"}))
        bad = copy.deepcopy(optimization_wavelet_fixture())
        bad["method"] = "stft_v1"
        self.assertEqual("", optimization_wavelet_html(bad))
        bad = copy.deepcopy(optimization_wavelet_fixture())
        bad["mags_db"][0][1] = 0.5
        self.assertEqual("", optimization_wavelet_html(bad))
        bad = copy.deepcopy(optimization_wavelet_fixture())
        bad["times_ms"] = [100.0, 0.0, -5.0]
        self.assertEqual("", optimization_wavelet_html(bad))


if __name__ == "__main__":
    unittest.main()
