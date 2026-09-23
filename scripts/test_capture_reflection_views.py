"""Captured direction plots must preserve missing and ambiguous evidence."""
import copy
import unittest

from scripts.test_capture_clock_views import fixture
from scripts.src.capture_reflection_views import capture_reflections_html


def reflected_fixture():
    data = fixture()
    event = {"arrival_ms": 15.0, "relative_ms": 10.0, "level_db": -6.0,
             "direction": [1.0, 0.0, 0.0], "mirror_ambiguous": False,
             "residual_samples": 0.1, "band_hz": [300.0, 1000.0],
             "issues": ["conditional plane wave"]}
    for name, source in data["metadata"]["effective_config"]["speakers"].items():
        source["provenance"]["capture"]["reflection_report"] = {
            "source_id": name, "direct_sound": None,
            "early_reflections": [copy.deepcopy(event)], "issues": []}
    return data


class CaptureReflectionViewsTests(unittest.TestCase):
    def test_accepted_direction_is_plotted_without_mutating_input(self):
        data = reflected_fixture()
        before = copy.deepcopy(data)
        html = capture_reflections_html(data)
        self.assertIn('class="capture-arrival-point"', html)
        self.assertIn("Conditional measured direction", html)
        self.assertIn("10.00", html)
        self.assertEqual(data, before)

    def test_ambiguity_bad_timing_and_malformed_vectors_never_get_arrows(self):
        for change in [
            {"mirror_ambiguous": True}, {"direction": [2.0, 0.0, 0.0]},
            {"direction": None}, {"band_hz": [300.0, 3000.0]},
            {"relative_ms": float("nan")}, {"residual_samples": 1.0},
        ]:
            with self.subTest(change=change):
                data = reflected_fixture()
                for source in data["metadata"]["effective_config"]["speakers"].values():
                    source["provenance"]["capture"]["reflection_report"]["early_reflections"][0].update(change)
                self.assertNotIn('class="capture-arrival-point"', capture_reflections_html(data))

    def test_vertical_arrival_does_not_invent_an_azimuth(self):
        data = reflected_fixture()
        for source in data["metadata"]["effective_config"]["speakers"].values():
            source["provenance"]["capture"]["reflection_report"]["early_reflections"][0]["direction"] = [0.0, 0.0, 1.0]
        html = capture_reflections_html(data)
        self.assertNotIn("class=\"capture-arrival-point\"", html)
        self.assertIn("azimuth undefined", html)

    def test_source_identity_mismatch_suppresses_the_plot(self):
        data = reflected_fixture()
        for source in data["metadata"]["effective_config"]["speakers"].values():
            source["provenance"]["capture"]["reflection_report"]["source_id"] = "other-source"
        html = capture_reflections_html(data)
        self.assertNotIn("class=\"capture-arrival-point\"", html)
        self.assertIn("source identity does not match", html)

    def test_missing_clock_and_escaped_notes_are_visible(self):
        data = reflected_fixture()
        for source in data["metadata"]["effective_config"]["speakers"].values():
            source["provenance"]["capture"]["takes"][0]["quality_passed"] = False
            source["provenance"]["capture"]["reflection_report"]["issues"] = ["<script>bad</script>"]
        html = capture_reflections_html(data)
        self.assertNotIn('class="capture-arrival-point"', html)
        self.assertIn("&lt;script&gt;", html)
        self.assertNotIn("<script>", html)
        self.assertEqual(capture_reflections_html({}), "")


if __name__ == "__main__":
    unittest.main()
