"""Fast selected-output complex-delay contract and mutation checks."""

import copy
import unittest

from check_selected_delay_artifact import check_selected_delay
from src.payload_binding import ALGORITHM, payload_digest


def selected_graph():
    data = {
        "global_plugins": [],
        "channels": {
            "subs": {
                "channel": "subs",
                "plugins": [],
                "drivers": [
                    {"name": "subs_1", "index": 0, "plugins": []},
                    {"name": "subs_2", "index": 1, "plugins": [{
                        "plugin_type": "delay",
                        "parameters": {
                            "delay_ms": 0.5,
                            "label": "joint_physical_drive_delay_refinement",
                            "room_eq_correction_delay": True,
                            "room_eq_stage": "post_route",
                        },
                    }]},
                ],
            }
        },
        "metadata": {"stage_outcomes": [
            {"stage": "final_candidate_objective", "checks": [{"observed": 0.25}]},
            {"stage": "final_correction_selection", "checks": [{
                "id": "joint_drive_delay_subs_1_to_0.500000000",
                "passed": True,
                "observed": 0.25,
            }]},
        ]},
    }
    return bind(data)


def bind(data):
    identity = "selected-delay-fixture"
    payload = {key: value for key, value in data.items() if key != "correction_decisions"}
    data["correction_decisions"] = {
        "ledger_version": "1.0.0",
        "decisions": [],
        "payload_binding": {
            "algorithm": ALGORITHM,
            "graph_identity": identity,
            "sha256": payload_digest(payload, identity),
        },
    }
    return data


class SelectedDelayArtifactTests(unittest.TestCase):
    def test_bound_selected_graph_has_expected_complex_delay(self):
        report = check_selected_delay(selected_graph())
        self.assertEqual(report["selected_driver_index"], 1)
        self.assertEqual(report["target_relative_ms"], 0.5)
        self.assertLess(report["max_complex_error"], 1e-10)

    def test_stale_or_misbound_delay_refuses(self):
        stale = selected_graph()
        stale["channels"]["subs"]["drivers"][1]["plugins"][0]["parameters"]["delay_ms"] = 0.75
        with self.assertRaisesRegex(ValueError, "payload binding failed"):
            check_selected_delay(stale)

        wrong_target = bind(copy.deepcopy(stale))
        with self.assertRaisesRegex(ValueError, "does not uniquely match"):
            check_selected_delay(wrong_target)

        missing = selected_graph()
        missing["channels"]["subs"]["drivers"][1]["plugins"] = []
        bind(missing)
        with self.assertRaisesRegex(ValueError, "no correction-owned physical delay"):
            check_selected_delay(missing)

    def test_wrong_delay_ownership_refuses(self):
        wrong = selected_graph()
        wrong["channels"]["subs"]["drivers"][1]["plugins"][0]["parameters"][
            "room_eq_correction_delay"
        ] = False
        bind(wrong)
        with self.assertRaisesRegex(ValueError, "invalid ownership"):
            check_selected_delay(wrong)

    def test_combined_gain_delay_trial_checks_both_emitted_controls(self):
        combined = selected_graph()
        combined["channels"]["subs"]["drivers"][1]["plugins"].insert(0, {
            "plugin_type": "gain",
            "parameters": {
                "gain_db": -1.0,
                "label": "joint_physical_drive_refinement",
                "room_eq_correction_gain": True,
                "room_eq_stage": "post_route",
            },
        })
        combined["metadata"]["stage_outcomes"][1]["checks"][0]["id"] = (
            "joint_drive_gain_delay_subs_1_to_-1.000000000"
            "__joint_drive_delay_subs_1_to_0.500000000"
        )
        bind(combined)
        report = check_selected_delay(combined)
        self.assertEqual(report["target_relative_ms"], 0.5)
        self.assertLess(report["max_complex_error"], 1e-10)

        wrong_gain = copy.deepcopy(combined)
        wrong_gain["channels"]["subs"]["drivers"][1]["plugins"][0]["parameters"]["gain_db"] = -2.0
        bind(wrong_gain)
        with self.assertRaisesRegex(ValueError, "does not uniquely match"):
            check_selected_delay(wrong_gain)


if __name__ == "__main__":
    unittest.main()
