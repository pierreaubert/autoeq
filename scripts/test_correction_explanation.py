"""Evidence and presentation regressions for the report's opening explanation."""

import unittest

from scripts.src.correction_explanation import correction_explanation_html


class CorrectionExplanationTests(unittest.TestCase):
    def test_optimizer_band_is_used_when_no_narrower_policy_is_configured(self):
        html = correction_explanation_html({"metadata": {"effective_config": {
            "optimizer": {"min_freq": 30, "max_freq": 400}
        }}})
        self.assertIn("30–400 Hz", html)
        self.assertNotIn("Scope not recorded", html)
        self.assertNotIn("Outside correction scope", html)

    def test_accepted_and_unchanged_outcomes_remain_distinct(self):
        accepted = correction_explanation_html({"metadata": {"correction_acceptance": {
            "outcome": "accepted", "accepted": True, "decision": "accepted"
        }}})
        unchanged = correction_explanation_html({"metadata": {"correction_acceptance": {
            "outcome": "unchanged", "accepted": False, "decision": "identity_fallback"
        }}})
        self.assertIn("final correction passed", accepted)
        self.assertIn("final outcome is unchanged", unchanged)
        self.assertNotIn("final correction passed", unchanged)

    def test_scope_is_not_claimed_as_applied_correction(self):
        html = correction_explanation_html({"metadata": {"effective_config": {
            "optimizer": {"min_freq": 20, "max_freq": 20000,
                          "correction_band": {"min_hz": 35, "max_hz": 300}}
        }}})
        for band in ("20–35 Hz", "35–300 Hz", "300–20000 Hz"):
            self.assertIn(band, html)
        self.assertIn("does not prove that every frequency was corrected", html)
        self.assertIn("filter tails can still affect playback", html)
        self.assertIn("No final correction decision was recorded", html)

    def test_recorded_scope_and_seat_gaps_take_precedence(self):
        html = correction_explanation_html({"metadata": {
            "effective_config": {"optimizer": {
                "correction_band": {"min_hz": 20, "max_hz": 500}}},
            "correction_acceptance": {"outcome": "insufficient_evidence", "acoustic_quality": {
                "correction_band_hz": [40, 300], "evaluated_band_hz": [20, 20000],
                "final_seats": [{"logical_input": "L", "partition": "held_out", "seat_index": 2,
                                 "unassessed_bands_hz": [[20, 40]]}]}}
        }})
        for text in ("40–300 Hz", "20–500 Hz", "300–20000 Hz", "Requested scope",
                     "L / held_out / seat 2", "Not assessed", "insufficient to accept"):
            self.assertIn(text, html)

    def test_nomination_is_advisory_and_center_is_not_an_interval(self):
        html = correction_explanation_html({"metadata": {"audibility_veto": {"L": [{
            "index": 0, "center_hz": 8000, "decision": "Remove", "reason": "HighQAboveGuard",
            "enforced": False, "acceptance": {"confidence": "low"}
        }]}}})
        for text in ("Removal nominated (advisory)", "8000 Hz (filter center)",
                     "high-frequency Q guard", "Confidence: low", "does not establish a frequency interval"):
            self.assertIn(text, html)

    def test_rejection_and_reversion_are_not_overridden_by_applied_history(self):
        html = correction_explanation_html({"metadata": {
            "correction_acceptance": {"outcome": "rejected", "accepted": False,
                                      "reverted_stages": ["eq"], "violations": ["headroom exceeded"]},
            "stage_outcomes": [{"stage": "eq", "status": "applied", "advisories": ["candidate improved"]},
                               {"stage": "phase", "status": "skipped"}]
        }})
        for text in ("final correction was rejected", "Reverted stage: eq", "headroom exceeded",
                     "Stage eq (applied): candidate improved", "phase: skipped; no reason recorded",
                     "final outcome above takes precedence"):
            self.assertIn(text, html)
        self.assertNotIn("final correction passed", html)

    def test_untrusted_evidence_and_mode_labels_are_escaped(self):
        html = correction_explanation_html({"metadata": {
            "correction_acceptance": {"violations": ["<script>bad()</script>"]},
            "stage_outcomes": [{"stage": "<eq>", "checks": [
                {"id": "<check>", "passed": False, "diagnostic": "<diagnostic>"}]}]
        }}, "<mode>")
        for escaped in ("&lt;script&gt;", "&lt;eq&gt;", "&lt;check&gt;", "&lt;diagnostic&gt;", "&lt;mode&gt;"):
            self.assertIn(escaped, html)
        self.assertNotIn("<script>", html)

    def test_missing_invalid_and_inconsistent_evidence_stays_unknown(self):
        for band in ([float("nan"), 300], [300, 20], [True, 300], [0, 300]):
            with self.subTest(band=band):
                html = correction_explanation_html({"metadata": {"correction_acceptance": {
                    "outcome": "accepted", "acoustic_quality": {"correction_band_hz": band}
                }}})
                self.assertIn("Scope not recorded", html)
                self.assertIn("incomplete or inconsistent", html)
                self.assertNotIn("final correction passed", html)
        self.assertIn("No final correction decision", correction_explanation_html({}))


if __name__ == "__main__":
    unittest.main()
