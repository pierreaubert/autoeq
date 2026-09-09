"""Negative signal-axis controls, independent of executing the matrix."""
from copy import deepcopy
import unittest

from parameter_signal_contract import verify_signal_axes


class SignalContractTests(unittest.TestCase):
    def fixture(self, rate_axis=0, phase_axis=1):
        rate = [44100, 48000, 96000][rate_axis]
        return {
            "requested_axes": {"sample_rate": rate_axis, "phase": phase_axis},
            "sample_rate_hz": rate,
            "effective_config": {"measurements": {
                "L": {"has_phase": phase_axis == 1},
                "R": {"has_phase": phase_axis == 1},
            }},
            "delivered_biquad_sample_rates_hz": {"L": [rate], "R": [rate, rate]},
        }

    def test_all_rate_and_phase_axes(self):
        for rate in range(3):
            for phase in range(3):
                self.assertEqual(verify_signal_axes(self.fixture(rate, phase)),
                                 "biquad_design_rate_verified")

    def test_faults_are_rejected(self):
        row = self.fixture()
        faults = []
        fault = deepcopy(row)
        fault["sample_rate_hz"] = 48000
        faults.append(fault)
        fault = deepcopy(row)
        fault["delivered_biquad_sample_rates_hz"]["R"][1] = 48000
        faults.append(fault)
        fault = deepcopy(row)
        del fault["delivered_biquad_sample_rates_hz"]["R"]
        faults.append(fault)
        fault = deepcopy(row)
        fault["effective_config"]["measurements"]["L"]["has_phase"] = False
        faults.append(fault)
        fault = deepcopy(row)
        fault["effective_config"]["measurements"] = {}
        faults.append(fault)
        for fault in faults:
            with self.subTest(fault=fault), self.assertRaises(AssertionError):
                verify_signal_axes(fault)

    def test_empty_chain_is_not_design_rate_evidence(self):
        row = self.fixture()
        row["delivered_biquad_sample_rates_hz"] = {"L": [], "R": []}
        self.assertEqual(verify_signal_axes(row), "no_delivered_biquad")

    def test_explicit_role_mapping(self):
        row = self.fixture()
        row["effective_config"]["system"] = {"speakers": {"L": "left_capture", "R": "right_capture"}}
        row["effective_config"]["measurements"] = {"left_capture": {"has_phase": True}, "right_capture": {"has_phase": True}}
        self.assertEqual(verify_signal_axes(row), "biquad_design_rate_verified")
        row["effective_config"]["system"]["speakers"]["R"] = "missing_capture"
        with self.assertRaises(AssertionError):
            verify_signal_axes(row)


if __name__ == "__main__":
    unittest.main()
