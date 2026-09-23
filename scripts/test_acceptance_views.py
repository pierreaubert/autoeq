"""Fast rendering controls; actual production payloads are tested through Rust."""

import unittest

from scripts.src.acceptance_views import _magnitude_svg, acceptance_views_html


class AcceptanceViewsTests(unittest.TestCase):
    def test_legacy_is_explicitly_unavailable_and_labels_are_escaped(self):
        html = acceptance_views_html({}, '<script>bad</script>')
        self.assertIn("legacy or unfinalized", html)
        self.assertNotIn("<script>", html)
        self.assertIn("&lt;script&gt;", html)

    def test_unknown_version_or_unbound_graph_cannot_render_views(self):
        html = acceptance_views_html({"correction_decisions": {"acceptance_evidence": {
            "version": "future", "payload": {}, "binding": {}}}})
        self.assertIn("Unavailable:", html)
        self.assertNotIn("<svg", html)

    def test_shared_axes_preserve_level_difference(self):
        view = {"freqs": [100.0, 1000.0], "pre_db": [0.0, 0.0], "post_db": [6.0, 6.0]}
        html = _magnitude_svg(view, "<magnitude>")
        self.assertIn("40.00,200.00 760.00,200.00", html)
        self.assertIn("40.00,20.00 760.00,20.00", html)
        self.assertIn("&lt;magnitude&gt;", html)

    def test_malformed_traces_fail_without_silent_truncation(self):
        for freq, pre, post in [
            ([100.0, 1000.0], [0.0], [0.0, 0.0]),
            ([1000.0, 100.0], [0.0, 0.0], [0.0, 0.0]),
            ([100.0, 1000.0], [float("nan"), 0.0], [0.0, 0.0]),
            ([100.0, 1000.0], [-1.7e308, 0.0], [1.7e308, 0.0]),
        ]:
            with self.subTest(freq=freq, pre=pre):
                with self.assertRaises(ValueError):
                    _magnitude_svg({"freqs": freq, "pre_db": pre, "post_db": post}, "test")
