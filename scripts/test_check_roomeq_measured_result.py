#!/usr/bin/env python3
"""Proof tests for the native-result audit.

The audit must reject unresolved causal delays and decisions whose graph
no longer matches its binding, beyond trusting stored `passed` flags.
"""

import copy
import json
import tempfile
import unittest
from pathlib import Path

import check_roomeq_measured_result
from src.payload_binding import ALGORITHM, payload_digest


def bound_fixture():
    data = {
        "version": "2.1.0",
        "channels": {
            "L": {"plugins": [
                {"plugin_type": "gain", "parameters": {"gain_db": 1.0}},
                {"plugin_type": "delay", "parameters": {"delay_ms": 2.5}},
            ]},
        },
        "metadata": {
            "stage_outcomes": [
                {"stage": "final_graph_sampled_electrical_headroom",
                 "checks": [{"id": "electrical_headroom", "passed": True}]},
            ],
            "effective_config": {
                "optimizer": {"min_freq": 20.0, "max_freq": 20000.0,
                              "finalization": {}},
                "system": {},
                "speakers": {},
            },
            "correction_acceptance": {
                "outcome": "accepted", "accepted": True, "decision": "accepted",
                "violations": [],
                "metrics": {"improvement_db": 1.0,
                            "pre_target_weighted_rms_db": 5.0,
                            "post_target_weighted_rms_db": 4.0},
                "acoustic_quality": {
                    "evaluated_band_hz": [20.0, 20000.0],
                    "final_seats": [
                        {"partition": "main", "logical_input": "L",
                         "seat_index": 0, "improvement_db": 1.0},
                    ],
                    "useful_output": [
                        {"partition": "main", "logical_input": "L",
                         "seat_index": 0, "unexplained_loss_rms_db": 0.5},
                    ],
                },
            },
        },
    }
    payload = {key: value for key, value in data.items() if key != "correction_decisions"}
    identity = "fixture-graph-1"
    data["correction_decisions"] = {
        "ledger_version": "1.0.0",
        "decisions": [{"decision_id": "dec-eq", "stage": "final",
                       "final_graph_identity": identity}],
        "payload_binding": {"algorithm": ALGORITHM, "graph_identity": identity,
                            "sha256": payload_digest(payload, identity)},
    }
    return data


def write_result(data):
    root = Path(tempfile.mkdtemp(prefix="roomeq-audit-proof-"))
    result = root / "dsp-iir.json"
    result.write_text(json.dumps(data), encoding="utf-8")
    files = root / "dsp-iir_files"
    files.mkdir()
    (files / "manifest.json").write_text(
        json.dumps({"status": "complete", "assets_owned": []}), encoding="utf-8")
    return result


class NativeAuditProofTests(unittest.TestCase):
    def test_valid_bound_graph_passes(self):
        row = check_roomeq_measured_result.inspect(write_result(bound_fixture()))
        self.assertEqual(row["outcome"], "accepted")
        self.assertEqual(row["final_seats"], 1)

    def test_negative_channel_delay_is_rejected(self):
        data = bound_fixture()
        data["channels"]["L"]["plugins"][1]["parameters"]["delay_ms"] = -7.644
        with self.assertRaisesRegex(ValueError, "unresolved causal delay"):
            check_roomeq_measured_result.inspect(write_result(data))

    def test_negative_route_delay_is_rejected(self):
        data = bound_fixture()
        data["metadata"]["bass_management"] = {
            "routing_graph": {"routes": [{"destination": "sub",
                                           "delay_ms": -0.701}]}}
        with self.assertRaisesRegex(ValueError, "unresolved causal delay"):
            check_roomeq_measured_result.inspect(write_result(data))

    def test_non_numeric_delay_is_rejected(self):
        data = bound_fixture()
        data["channels"]["L"]["plugins"][1]["parameters"]["delay_ms"] = "late"
        with self.assertRaisesRegex(ValueError, "non-numeric delay"):
            check_roomeq_measured_result.inspect(write_result(data))

    def test_mutated_gain_invalidates_earlier_approval(self):
        data = bound_fixture()
        data["channels"]["L"]["plugins"][0]["parameters"]["gain_db"] = 2.0
        with self.assertRaisesRegex(ValueError, "recorded decisions are stale"):
            check_roomeq_measured_result.inspect(write_result(data))

    def test_mutated_delay_invalidates_earlier_approval(self):
        data = bound_fixture()
        data["channels"]["L"]["plugins"][1]["parameters"]["delay_ms"] = 3.5
        with self.assertRaisesRegex(ValueError, "recorded decisions are stale"):
            check_roomeq_measured_result.inspect(write_result(data))

    def test_mismatched_final_identity_is_rejected(self):
        data = bound_fixture()
        decisions = data["correction_decisions"]["decisions"]
        decisions[0]["final_graph_identity"] = "stale-graph"
        # Rebind the digest so only the final-identity agreement fails.
        payload = {key: value for key, value in data.items()
                   if key != "correction_decisions"}
        binding = data["correction_decisions"]["payload_binding"]
        binding["sha256"] = payload_digest(payload, binding["graph_identity"])
        with self.assertRaisesRegex(ValueError, "does not match"):
            check_roomeq_measured_result.inspect(write_result(data))

    def test_legacy_graph_without_binding_is_audited_on_other_evidence(self):
        data = bound_fixture()
        del data["correction_decisions"]
        row = check_roomeq_measured_result.inspect(write_result(data))
        self.assertEqual(row["outcome"], "accepted")

    def test_fixture_copies_do_not_share_mutation(self):
        first, second = bound_fixture(), bound_fixture()
        first["channels"]["L"]["plugins"][0]["parameters"]["gain_db"] = 9.0
        self.assertEqual(
            second["channels"]["L"]["plugins"][0]["parameters"]["gain_db"], 1.0)


if __name__ == "__main__":
    unittest.main()
