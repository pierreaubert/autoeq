"""Fast validation of capture diagnostics, not acoustic evidence."""

import copy
import unittest

from scripts.src.capture_views import capture_views_html
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
    return {"graph_id": "candidate", "comparisons": [{
        "source": "<source>", "seat": "seat", "capture_views": views,
        "capture_views_binding": {"algorithm": ALGORITHM, "graph_identity": "candidate",
                                  "sha256": payload_digest(views, "candidate")},
    }]}


class CaptureViewsTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
