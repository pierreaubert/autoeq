"""Fast topology and replay checks for the public routed joint-sub contract."""

import copy
from pathlib import Path
import unittest
from unittest.mock import patch

from run_routed_joint_contract import check_routed_joint


ARTIFACT = Path("/Volumes/home_tmp/tmp/routed-joint-test-only.json")
FREQUENCIES = (40.0, 80.0, 120.0)


def routed_graph():
    return {"metadata": {"bass_management": {"routing_graph": {
        "input_channels": ["L", "R"],
        "output_channels": ["L", "R", "Sub1", "RearBass"],
        "routes": [
            {"source_channel": "L", "destination": "Sub1",
             "route_kind": "redirected_bass_lowpass_to_sub"},
            {"source_channel": "R", "destination": "RearBass",
             "route_kind": "redirected_bass_lowpass_to_sub"},
        ],
    }}}}


def transfers():
    return {
        "Sub1": {"L": [0.5 + 0j] * len(FREQUENCIES), "R": [0j] * len(FREQUENCIES)},
        "RearBass": {"L": [0j] * len(FREQUENCIES), "R": [0.5 + 0j] * len(FREQUENCIES)},
    }


class RoutedJointContractTests(unittest.TestCase):
    def test_public_topology_and_independent_replay_are_required(self):
        with (
            patch("run_routed_joint_contract.assess_bound_artifact", return_value={
                "status": "independent_sampled_electrical_match", "peaks": {},
            }),
            patch("run_routed_joint_contract.output_complex_transfers", return_value=transfers()),
        ):
            report = check_routed_joint(routed_graph(), ARTIFACT)
        self.assertEqual(report["inputs"], ["L", "R"])
        self.assertEqual(report["bass_routes"], [["L", "Sub1"], ["R", "RearBass"]])

    def test_swapped_bass_route_refuses_even_if_report_matches(self):
        wrong = copy.deepcopy(routed_graph())
        wrong["metadata"]["bass_management"]["routing_graph"]["routes"][1]["destination"] = "Sub1"
        with self.assertRaisesRegex(ValueError, "bass-route ownership"):
            check_routed_joint(wrong, ARTIFACT)

    def test_cross_input_transfer_refuses_even_if_report_matches(self):
        wrong = transfers()
        wrong["Sub1"]["R"] = [0.1 + 0j] * len(FREQUENCIES)
        with (
            patch("run_routed_joint_contract.assess_bound_artifact", return_value={
                "status": "independent_sampled_electrical_match", "peaks": {},
            }),
            patch("run_routed_joint_contract.output_complex_transfers", return_value=wrong),
            self.assertRaisesRegex(ValueError, "undeclared bass source"),
        ):
            check_routed_joint(routed_graph(), ARTIFACT)


if __name__ == "__main__":
    unittest.main()
