"""Evidence and presentation regressions for the report's opening explanation."""

import json
import os
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


def _record(**overrides):
    record = {
        "decision_id": "d-1",
        "ledger_version": "1.0.0",
        "stage": "final",
        "logical_input": "stereo",
        "physical_output": "main-l",
        "measurement_refs": ["meas-1"],
        "seat_refs": ["seat-a"],
        "frequency_band_hz": [40.0, 400.0],
        "action": "equalize",
        "status": "applied",
        "reason_codes": ["within_limits"],
        "observed": [{"name": "post_p95_abs_residual_db", "value": 3.0, "unit": "db"}],
        "limits": [{"name": "max_post_p95_abs_residual_db", "value": 6.0, "unit": "db"}],
        "evidence_refs": ["ev-1"],
        "confidence": "moderate",
        "final_graph_identity": "graph-final-1",
    }
    record.update(overrides)
    return record


def _ledger_html(decisions, metadata=None):
    data = {"correction_decisions": {"ledger_version": "1.0.0", "decisions": decisions}}
    if metadata is not None:
        data["metadata"] = metadata
    return correction_explanation_html(data)


def _fixture(name):
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(root, "crates", "roomeq-model", "test-data",
                        "decision_ledger", name)
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


class CorrectionDecisionLedgerTests(unittest.TestCase):
    def test_missing_ledger_renders_reason_unavailable_without_crashing(self):
        html = correction_explanation_html({})
        self.assertIn("Reason unavailable", html)
        self.assertIn("No versioned decision ledger was recorded", html)
        html = correction_explanation_html({"correction_decisions": {"decisions": []}})
        self.assertIn("Reason unavailable", html)

    def test_every_k4_status_renders_and_advisory_is_not_removal(self):
        expected = {
            "applied": "Applied",
            "already_acceptable": "Already acceptable",
            "insufficient_evidence": "Insufficient evidence",
            "outside_scope": "Outside scope",
            "constrained": "Constrained",
            "reverted": "Reverted",
            "unresolved": "Unresolved",
            "advisory": "Advisory nomination only",
        }
        for status, text in expected.items():
            with self.subTest(status=status):
                record = _record(status=status)
                if status == "advisory":
                    record["stage"] = "provisional"
                    del record["final_graph_identity"]
                html = _ledger_html([record])
                self.assertIn(text, html)
        advisory = _record(status="advisory", stage="provisional")
        advisory.pop("final_graph_identity", None)
        html = _ledger_html([advisory])
        self.assertIn("Advisory nomination only (nothing applied or removed)", html)
        self.assertNotIn("Removal nominated", html)

    def test_rejected_superseded_and_fallback_records_never_become_delivered(self):
        attempt = _record(decision_id="d-attempt-1", stage="provisional")
        rollback = _record(decision_id="d-rollback-1", status="reverted",
                           supersedes_ids=["d-attempt-1"],
                           final_graph_identity="graph-identity-fallback-1")
        html = _ledger_html([attempt, rollback])
        self.assertIn("Reverted (attempted correction rolled back; not delivered)", html)
        self.assertIn("Supersedes: d-attempt-1", html)
        self.assertIn("The superseded attempt is not delivered correction", html)
        self.assertIn("Provisional record d-attempt-1", html)
        self.assertIn("Final reconciled records above take precedence", html)

    def test_separate_seat_reasons_and_gaps_are_not_collapsed(self):
        applied = _record(decision_id="d-seat-a-1", seat_refs=["seat-a"],
                          measurement_refs=["meas-seat-a"], evidence_refs=["ev-seat-a"])
        gap = _record(decision_id="d-seat-b-1", seat_refs=["seat-b"],
                      measurement_refs=["meas-seat-b"], status="insufficient_evidence",
                      reason_codes=["seat_evidence_gap"], stage="provisional",
                      evidence_refs=[], confidence="unknown")
        gap.pop("final_graph_identity", None)
        html = _ledger_html([applied, gap])
        self.assertIn("seat-a", html)
        self.assertIn("seat-b", html)
        self.assertEqual(html.count("40–400 Hz"), 1 + 1)  # one final row + one history line
        self.assertIn("Provisional record d-seat-b-1", html)

    def test_intervals_and_filter_centers_are_distinguishable(self):
        banded = _record(decision_id="d-band-1")
        centered = _record(decision_id="d-center-1", frequency_band_hz=None,
                           filter_center_hz=120.0, action="prune", status="advisory",
                           stage="provisional")
        centered.pop("final_graph_identity", None)
        html = _ledger_html([banded, centered])
        self.assertIn("40–400 Hz", html)
        self.assertIn("120 Hz (filter center)", html)
        bare = _record(decision_id="d-bare-1", frequency_band_hz=None)
        bare.pop("filter_center_hz", None)
        html = _ledger_html([bare])
        self.assertIn("Frequency unavailable", html)

    def test_constrained_partial_and_remainder_stay_jointly_visible(self):
        applied = _record(decision_id="d-partial-1", frequency_band_hz=[40.0, 200.0],
                          reason_codes=["partial_within_limits"],
                          related_decision_ids=["d-remainder-1"])
        remainder = _record(decision_id="d-remainder-1", frequency_band_hz=[200.0, 400.0],
                            status="constrained", reason_codes=["headroom_limit"],
                            related_decision_ids=["d-partial-1"])
        html = _ledger_html([applied, remainder])
        for text in ("40–200 Hz", "200–400 Hz", "Applied", "Constrained",
                     "Linked record: d-remainder-1", "Linked record: d-partial-1",
                     "constrained remainder stay visible"):
            self.assertIn(text, html)

    def test_graph_identity_mismatch_is_visibly_unverified(self):
        first = _record(decision_id="d-1", final_graph_identity="graph-a")
        second = _record(decision_id="d-2", final_graph_identity="graph-b")
        html = _ledger_html([first, second])
        self.assertIn("disagree on the delivered-graph identity", html)
        self.assertIn("unverified", html)
        delivered = {"delivered_graph_identity": "graph-final-9"}
        html = _ledger_html([_record()], delivered)
        self.assertIn("does not match the recorded delivered graph", html)
        unbound = _record()
        unbound.pop("final_graph_identity", None)
        html = _ledger_html([unbound])
        self.assertIn("without a delivered-graph identity is not a delivery claim", html)

    def test_malformed_records_render_unverified_without_crashing(self):
        html = _ledger_html([{"stage": "final"}])
        self.assertIn("missing decision ID", html)
        html = _ledger_html(["not-a-record"])
        self.assertIn("unreadable entry", html)
        html = _ledger_html([_record(frequency_band_hz=[400.0, 40.0])])
        self.assertIn("Frequency unavailable", html)

    def test_untrusted_k4_strings_are_escaped(self):
        record = _record(logical_input="<L>", physical_output="<sub>",
                         reason_codes=["<reason>"], evidence_refs=["<ev>"],
                         seat_refs=["<seat>"])
        html = _ledger_html([record], {"delivered_graph_identity": "<g>"})
        for escaped in ("&lt;L&gt;", "&lt;sub&gt;", "&lt;reason&gt;", "&lt;ev&gt;", "&lt;seat&gt;"):
            self.assertIn(escaped, html)
        self.assertNotIn("<script>", html)
        html = correction_explanation_html({"metadata": {}}, "<mode>")
        self.assertIn("&lt;mode&gt;", html)

    def test_raw_output_loss_stays_visible(self):
        quality = {"useful_output": [{
            "logical_input": "L", "partition": "held_out", "seat_index": 2,
            "permitted_gain_db": 0.0, "evaluated_band_hz": [20, 20000],
            "mean_level_change_db": -4.5, "unexplained_loss_rms_db": 2.25,
            "worst_unexplained_loss_db": 5.0, "worst_loss_frequency_hz": 90.0,
        }]}
        html = correction_explanation_html(
            {"metadata": {"correction_acceptance": {"acoustic_quality": quality}}})
        for text in ("Recorded raw output loss", "L / held_out / seat 2",
                     "-4.5 dB", "2.25 dB", "5 dB at 90 Hz",
                     "display normalization cannot hide this loss"):
            self.assertIn(text, html)

    def test_verification_kinds_and_signal_levels_stay_separate(self):
        metadata = {"playback_comparisons": [
            {"capture_kind": "simulated_backend", "processing_state": "small_signal",
             "baseline_graph_identity": "graph-base", "candidate_graph_identity": "graph-cand",
             "stimulus_hash": "stim-1", "source_id": "L", "seat_ids": ["seat-a"]},
            {"capture_kind": "stationary_ir", "processing_state": "dynamic",
             "source_id": "L"},
        ], "listening_evidence": {"result": "inconclusive", "protocol_id": "p-1"}}
        html = correction_explanation_html({"metadata": metadata})
        for text in ("exported-backend simulation (not an acoustic recording)",
                     "small_signal", "acoustic capture", "dynamic",
                     "unassessed: insufficient evidence, no promotion",
                     "Recorded listening outcome: inconclusive",
                     "is not listening benefit"):
            self.assertIn(text, html)

    def test_canonical_model_fixtures_agree_with_renderer(self):
        accepted = _fixture("accepted.json")
        html = correction_explanation_html({"correction_decisions": accepted})
        self.assertIn("Applied", html)
        self.assertIn("40–400 Hz", html)
        constrained = _fixture("constrained_partial.json")
        html = correction_explanation_html({"correction_decisions": constrained})
        self.assertIn("Linked record: d-remainder-1", html)
        advisory = _fixture("advisory.json")
        html = correction_explanation_html({"correction_decisions": advisory})
        self.assertIn("Advisory nomination only", html)
        self.assertIn("120 Hz (filter center)", html)
        rejected = _fixture("rejected_with_reversion.json")
        html = correction_explanation_html({"correction_decisions": rejected})
        self.assertIn("Reverted", html)
        self.assertIn("Supersedes: d-attempt-1", html)
        legacy = _fixture("legacy_output.json")
        self.assertNotIn("correction_decisions", legacy)
        html = correction_explanation_html(legacy)
        self.assertIn("Reason unavailable", html)


if __name__ == "__main__":
    unittest.main()
