#!/usr/bin/env python3
import unittest
from parameter_crossover_contract import verify_crossover


def fixture(policy=0, phase=1):
    kind = "LR48" if policy == 2 else "LR24"
    frequency = 180.0 if policy == 1 else 160.0
    routes = []
    for source in ["L", "R"]:
        routes.extend([
            dict(route_kind="main_highpass_to_self", source_channel=source,
                 destination=source, high_pass_hz=frequency, group_id="front", crossover_type=kind),
            dict(route_kind="redirected_bass_lowpass_to_sub", source_channel=source,
                 destination="LFE", low_pass_hz=frequency, group_id="front", crossover_type=kind),
        ])
    return {
        "requested_axes": dict(topology=1, crossover=policy, phase=phase),
        "bass_management": {
            "enabled": True, "lfe_channel": "LFE", "physical_sub_output": "LFE",
            "optimization": dict(crossover_type=kind, crossover_range_hz=[120.0, 220.0] if policy == 1 else None,
                                 phase_available=phase == 1, applied=phase == 1,
                                 advisories=[] if phase == 1 else ["missing_phase_crossover_alignment_skipped"]),
            "groups": [dict(group_id="front", selected_crossover_hz=frequency, crossover_type=kind)],
            "routing_graph": dict(input_channels=["L", "R", "LFE"], routes=routes),
        },
    }


class CrossoverContractTests(unittest.TestCase):
    def test_all_three_policies_and_missing_phase(self):
        for policy in range(3):
            for phase in range(3):
                status = verify_crossover(fixture(policy, phase))
                self.assertEqual(status == "unsupported_missing_phase", policy == 1 and phase != 1)

    def test_missing_sub_route_fails(self):
        row = fixture()
        row["bass_management"]["routing_graph"]["routes"].pop()
        with self.assertRaises(AssertionError):
            verify_crossover(row)

    def test_wrong_branch_frequency_fails(self):
        row = fixture()
        row["bass_management"]["routing_graph"]["routes"][1]["low_pass_hz"] = 80.0
        with self.assertRaises(AssertionError):
            verify_crossover(row)

    def test_wrong_crossover_order_fails(self):
        row = fixture(2)
        row["bass_management"]["routing_graph"]["routes"][0]["crossover_type"] = "LR24"
        with self.assertRaises(AssertionError):
            verify_crossover(row)

    def test_claimed_phase_without_execution_fails(self):
        row = fixture(1)
        row["bass_management"]["optimization"]["applied"] = False
        with self.assertRaises(AssertionError):
            verify_crossover(row)

    def test_absent_phase_requires_explicit_skip(self):
        row = fixture(1, 0)
        row["bass_management"]["optimization"]["advisories"] = []
        with self.assertRaises(AssertionError):
            verify_crossover(row)


if __name__ == "__main__":
    unittest.main()
