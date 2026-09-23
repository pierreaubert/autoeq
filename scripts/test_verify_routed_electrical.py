"""Fast independent routed electrical oracle contracts."""

import copy
import cmath
from contextlib import redirect_stdout
import io
import json
import math
from pathlib import Path
import unittest
from unittest.mock import patch

from src.payload_binding import ALGORITHM, payload_digest
from verify_routed_electrical import (
    _lr24,
    _lr48,
    _peak,
    _post_plugins,
    assess_bound_artifact,
    main,
    output_amplitudes,
    output_complex_transfers,
    verify_artifact,
)


RATE = 48_000.0


def route(source, destination="Sub1", inverted=False):
    return {
        "source_channel": source,
        "destination": destination,
        "crossover_type": "LR24",
        "high_pass_hz": None,
        "low_pass_hz": 80.0,
        "gain_db": 0.0,
        "gain_linear": 1.0,
        "matrix_gain": 1.0,
        "delay_ms": 0.0,
        "polarity_inverted": inverted,
    }


def graph(inputs=("L",), routes=None):
    return {
        "global_plugins": [{"plugin_type": "matrix", "parameters": {}}],
        "channels": {
            **{name: {"plugins": []} for name in inputs},
            "Sub1": {"plugins": []},
        },
        "metadata": {
            "bass_management": {
                "routing_graph": {
                    "input_channels": list(inputs),
                    "output_channels": ["Sub1"],
                    "physical_sub_outputs": ["Sub1"],
                    "routes": routes if routes is not None else [route("L")],
                }
            },
            "stage_outcomes": [
                {
                    "stage": "final_graph_sampled_electrical_headroom",
                    "checks": [
                        {
                            "id": "sampled_physical_output:Sub1",
                            "kind": "safety",
                            "passed": True,
                            "observed": 1.0,
                            "limit": 1.0,
                            "diagnostic": '{"output":"Sub1","inputs":["L"],"input_peak_limits":{"L":1.0},"sample_rate_hz":48000.0,"evaluated_band_hz":[0.0,24000.0],"grid_points":8194,"peak_frequency_hz":0.0,"peak_amplitude":1.0}',
                        }
                    ],
                }
            ],
        },
    }


def bound_graph(delivered=None):
    delivered = graph() if delivered is None else delivered
    identity = "synthetic-graph"
    delivered["correction_decisions"] = {
        "ledger_version": "1.0.0",
        "decisions": [],
        "payload_binding": {
            "algorithm": ALGORITHM,
            "graph_identity": identity,
            "sha256": payload_digest(delivered, identity),
        },
    }
    return delivered


class RoutedElectricalOracleTests(unittest.TestCase):
    def test_peak_and_linkwitz_riley_anchors(self):
        peak = {"filter_type": "peak", "freq": 80.0, "q": 2.0, "db_gain": 6.0}
        self.assertAlmostEqual(abs(_peak(peak, 80.0, RATE)), 10 ** (6 / 20), places=10)
        self.assertAlmostEqual(abs(_lr24(0.0, 80.0, RATE, False)), 1.0, places=10)
        self.assertAlmostEqual(abs(_lr24(80.0, 80.0, RATE, False)), 0.5, places=10)
        self.assertAlmostEqual(abs(_lr24(80.0, 80.0, RATE, True)), 0.5, places=10)

        self.assertAlmostEqual(abs(_lr48(80.0, 80.0, RATE, False)), 0.5, places=10)
        self.assertAlmostEqual(abs(_lr48(80.0, 80.0, RATE, True)), 0.5, places=10)
        warped_ratio = math.tan(math.pi * 160.0 / RATE) / math.tan(math.pi * 80.0 / RATE)
        self.assertAlmostEqual(
            abs(_lr48(160.0, 80.0, RATE, False)),
            1 / (1 + warped_ratio ** 8),
            places=10,
        )

    def test_independent_inputs_do_not_cancel_physical_output(self):
        delivered = graph(("L", "R"), [route("L"), route("R", inverted=True)])
        output = output_amplitudes(delivered, [0.0, 80.0], {"L": 1.0, "R": 1.0}, RATE)
        self.assertAlmostEqual(output["Sub1"][0], 2.0, places=10)
        self.assertAlmostEqual(output["Sub1"][1], 1.0, places=10)

        same_input = graph(("L",), [route("L"), route("L", inverted=True)])
        cancelled = output_amplitudes(same_input, [0.0, 80.0], {"L": 1.0}, RATE)
        self.assertAlmostEqual(cancelled["Sub1"][0], 0.0, places=10)
        self.assertAlmostEqual(cancelled["Sub1"][1], 0.0, places=10)

    def test_routed_driver_baked_gain_is_not_applied_twice(self):
        delivered = graph()
        routed = delivered["metadata"]["bass_management"]["routing_graph"]
        routed["routes"][0]["gain_db"] = -6.0
        routed["routes"][0]["gain_linear"] = 10 ** (-6 / 20)
        routed["physical_sub_outputs"] = ["Sub1"]
        driver_gain = {
            "plugin_type": "gain",
            "parameters": {"room_eq_stage": "post_route", "gain_db": -6.0},
        }
        delivered["channels"]["Sub1"]["drivers"] = [
            {"name": "Sub1", "index": 0, "plugins": [driver_gain]}
        ]
        output = output_amplitudes(delivered, [0.0, 80.0], {"L": 1.0}, RATE)
        self.assertAlmostEqual(output["Sub1"][0], 10 ** (-6 / 20), places=10)

        corrected = copy.deepcopy(delivered)
        corrected_gain = corrected["channels"]["Sub1"]["drivers"][0]["plugins"][0]
        corrected_gain["parameters"]["room_eq_correction_gain"] = True
        output = output_amplitudes(corrected, [0.0, 80.0], {"L": 1.0}, RATE)
        self.assertAlmostEqual(output["Sub1"][0], 10 ** (-12 / 20), places=10)

        driver = corrected["channels"]["Sub1"]["drivers"][0]
        driver["plugins"].append({
            "plugin_type": "delay",
            "parameters": {"room_eq_stage": "post_route", "delay_ms": 0.5},
        })
        post = _post_plugins(corrected["channels"], routed, "Sub1")
        self.assertFalse(any(plugin["plugin_type"] == "delay" for plugin in post))
        driver["plugins"][-1]["parameters"]["room_eq_correction_delay"] = True
        post = _post_plugins(corrected["channels"], routed, "Sub1")
        self.assertEqual(sum(plugin["plugin_type"] == "delay" for plugin in post), 1)

    def test_selected_style_physical_advance_has_exact_complex_transfer(self):
        delivered = graph(routes=[route("L", "Sub1"), route("L", "Sub2")])
        routing = delivered["metadata"]["bass_management"]["routing_graph"]
        routing["output_channels"] = ["Sub1", "Sub2"]
        routing["physical_sub_outputs"] = ["Sub1", "Sub2"]
        routing["routes"][1]["delay_ms"] = 2.0
        delivered["channels"]["Sub1"]["drivers"] = [
            {
                "name": "Sub1", "index": 0,
                "plugins": [{"plugin_type": "delay", "parameters": {
                    "delay_ms": 0.5, "room_eq_stage": "post_route",
                    "room_eq_correction_delay": True,
                }}],
            },
            {
                "name": "Sub2", "index": 1,
                "plugins": [{"plugin_type": "delay", "parameters": {
                    "delay_ms": 2.0, "room_eq_stage": "post_route",
                }}],
            },
        ]
        frequencies = [40.0, 80.0, 120.0]
        transfers = output_complex_transfers(delivered, frequencies, RATE)
        for index, frequency in enumerate(frequencies):
            actual = transfers["Sub2"]["L"][index] / transfers["Sub1"]["L"][index]
            expected = cmath.exp(-2j * math.pi * frequency * 0.0015)
            self.assertAlmostEqual(abs(actual - expected), 0.0, places=10)

        # Magnitude-only electrical limits cannot distinguish these delays.
        peaks = output_amplitudes(delivered, frequencies, {"L": 0.1}, RATE)
        for reference, delayed in zip(peaks["Sub1"], peaks["Sub2"]):
            self.assertAlmostEqual(reference, delayed, places=12)
        unmarked = copy.deepcopy(delivered)
        del unmarked["channels"]["Sub1"]["drivers"][0]["plugins"][0]["parameters"][
            "room_eq_correction_delay"
        ]
        unmarked_transfer = output_complex_transfers(unmarked, frequencies, RATE)
        actual = unmarked_transfer["Sub2"]["L"][1] / unmarked_transfer["Sub1"]["L"][1]
        expected = cmath.exp(-2j * math.pi * frequencies[1] * 0.002)
        self.assertAlmostEqual(abs(actual - expected), 0.0, places=10)

        negative = copy.deepcopy(delivered)
        negative["channels"]["Sub1"]["drivers"][0]["plugins"][0]["parameters"][
            "delay_ms"
        ] = -0.5
        with self.assertRaisesRegex(ValueError, "invalid delay"):
            output_complex_transfers(negative, frequencies, RATE)

    def test_lr48_serialized_route_uses_eighth_order_response(self):
        delivered = graph()
        delivered["metadata"]["bass_management"]["routing_graph"]["routes"][0][
            "crossover_type"
        ] = "LR48"
        output = output_amplitudes(delivered, [0.0, 80.0, 160.0], {"L": 1.0}, RATE)
        warped_ratio = math.tan(math.pi * 160.0 / RATE) / math.tan(math.pi * 80.0 / RATE)
        self.assertAlmostEqual(output["Sub1"][0], 1.0, places=10)
        self.assertAlmostEqual(output["Sub1"][1], 0.5, places=10)
        self.assertAlmostEqual(output["Sub1"][2], 1 / (1 + warped_ratio ** 8), places=10)

    def test_mutated_serialized_route_is_not_verified_against_stale_report(self):
        delivered = graph()
        self.assertAlmostEqual(verify_artifact(delivered)["Sub1"], 1.0, places=8)
        changed = copy.deepcopy(delivered)
        changed["metadata"]["bass_management"]["routing_graph"]["routes"][0]["gain_db"] = 6.0
        changed["metadata"]["bass_management"]["routing_graph"]["routes"][0]["gain_linear"] = 10 ** (6 / 20)
        with self.assertRaisesRegex(ValueError, "independent peak"):
            verify_artifact(changed)

        contradictory = copy.deepcopy(delivered)
        contradictory["metadata"]["bass_management"]["routing_graph"]["routes"][0]["gain_linear"] = 2.0
        with self.assertRaisesRegex(ValueError, "contradictory route"):
            verify_artifact(contradictory)

    def test_bound_cli_path_rejects_delay_only_mutation(self):
        delivered = bound_graph()
        self.assertAlmostEqual(
            verify_artifact(delivered, require_binding=True)["Sub1"], 1.0, places=8
        )
        changed = copy.deepcopy(delivered)
        changed["metadata"]["bass_management"]["routing_graph"]["routes"][0]["delay_ms"] = 5.0
        # A single isolated route has the same magnitude with either delay;
        # payload identity must prevent the old report blessing the new graph.
        self.assertAlmostEqual(verify_artifact(changed)["Sub1"], 1.0, places=8)
        with self.assertRaisesRegex(ValueError, "payload binding failed"):
            verify_artifact(changed, require_binding=True)

    def test_verified_over_limit_output_is_not_reported_as_safe(self):
        unsafe = graph()
        unsafe_route = unsafe["metadata"]["bass_management"]["routing_graph"]["routes"][0]
        unsafe_route["gain_db"] = 6.0
        unsafe_route["gain_linear"] = 10 ** (6 / 20)
        check = unsafe["metadata"]["stage_outcomes"][0]["checks"][0]
        peak = unsafe_route["gain_linear"]
        check["observed"] = peak
        check["passed"] = False
        diagnostic = json.loads(check["diagnostic"])
        diagnostic["peak_amplitude"] = peak
        check["diagnostic"] = json.dumps(diagnostic)
        unsafe = bound_graph(unsafe)

        result = assess_bound_artifact(unsafe)
        self.assertEqual(result["status"], "independent_sampled_electrical_over_limit")
        self.assertEqual(result["over_limit_outputs"], ["Sub1"])
        output = io.StringIO()
        with patch("sys.argv", ["verify_routed_electrical.py", "synthetic.json"]), patch(
            "verify_routed_electrical.Path.read_text", return_value=json.dumps(unsafe)
        ), redirect_stdout(output), self.assertRaises(SystemExit) as raised:
            main()
        self.assertEqual(raised.exception.code, 2)
        self.assertEqual(json.loads(output.getvalue())["status"], result["status"])

    def test_report_rate_grid_and_verdict_must_match_evidence(self):
        delivered = graph()
        for field, value, message in (
            ("observed", 0.5, "observation differs"),
            ("passed", False, "verdict differs"),
        ):
            changed = copy.deepcopy(delivered)
            changed["metadata"]["stage_outcomes"][0]["checks"][0][field] = value
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, message):
                verify_artifact(changed)
        changed = copy.deepcopy(delivered)
        changed["sample_rate_hz"] = 44_100.0
        with self.assertRaisesRegex(ValueError, "sample rates differ"):
            verify_artifact(changed)
        changed = copy.deepcopy(delivered)
        changed_check = changed["metadata"]["stage_outcomes"][0]["checks"][0]
        changed_check["diagnostic"] = changed_check["diagnostic"].replace('"grid_points":8194', '"grid_points":8193')
        with self.assertRaisesRegex(ValueError, "frequency grids differ"):
            verify_artifact(changed)

    def test_report_input_provenance_and_unique_final_stage(self):
        delivered = graph()
        changed = copy.deepcopy(delivered)
        check = changed["metadata"]["stage_outcomes"][0]["checks"][0]
        diagnostic = json.loads(check["diagnostic"])
        diagnostic["inputs"] = ["R"]
        check["diagnostic"] = json.dumps(diagnostic)
        with self.assertRaisesRegex(ValueError, "input provenance differs"):
            verify_artifact(changed)

        changed = copy.deepcopy(delivered)
        changed["metadata"]["stage_outcomes"].append(
            copy.deepcopy(changed["metadata"]["stage_outcomes"][0])
        )
        with self.assertRaisesRegex(ValueError, "exactly one final electrical stage"):
            verify_artifact(changed)

        changed = copy.deepcopy(delivered)
        check = changed["metadata"]["stage_outcomes"][0]["checks"][0]
        diagnostic = json.loads(check["diagnostic"])
        diagnostic["peak_frequency_hz"] = 80.0
        check["diagnostic"] = json.dumps(diagnostic)
        with self.assertRaisesRegex(ValueError, "peak frequency has no replayed peak"):
            verify_artifact(changed)

    def test_numerically_tied_highpass_peaks_do_not_require_identical_index(self):
        delivered = graph()
        routed = delivered["metadata"]["bass_management"]["routing_graph"]["routes"][0]
        routed["low_pass_hz"] = None
        routed["high_pass_hz"] = 80.0
        check = delivered["metadata"]["stage_outcomes"][0]["checks"][0]
        diagnostic = json.loads(check["diagnostic"])
        diagnostic["peak_frequency_hz"] = RATE / 2 - RATE / (2 * 8192)
        check["diagnostic"] = json.dumps(diagnostic)
        self.assertAlmostEqual(verify_artifact(delivered)["Sub1"], 1.0, places=8)

    def test_unsupported_topologies_refuse_explicitly(self):
        no_routing = graph()
        del no_routing["metadata"]["bass_management"]
        with self.assertRaisesRegex(ValueError, "unsupported global processing"):
            verify_artifact(no_routing)

        virtual_input = graph()
        virtual_input["metadata"]["bass_management"]["routing_graph"]["routes"][0]["source_channel"] = "LFE"
        with self.assertRaisesRegex(ValueError, "unsupported virtual"):
            verify_artifact(virtual_input)

    def test_declared_virtual_lfe_has_identity_prechain_only(self):
        lfe_route = route("LFE")
        lfe_route["route_kind"] = "lfe_lowpass_to_sub"
        lfe_route["pre_chain_channel"] = "LFE"
        delivered = graph(("L", "LFE"), [route("L"), lfe_route])
        del delivered["channels"]["LFE"]
        output = output_amplitudes(
            delivered, [0.0, 80.0], {"L": 0.1, "LFE": 0.1}, RATE
        )
        self.assertAlmostEqual(output["Sub1"][0], 0.2, places=10)
        self.assertAlmostEqual(output["Sub1"][1], 0.1, places=10)

        for field, value, message in (
            ("route_kind", "redirected_bass_lowpass_to_sub", "unsupported virtual"),
            ("pre_chain_channel", "Sub1", "unsupported virtual"),
            ("source_channel", "unknown", "route endpoint is not declared"),
            ("source_index", 0, "route endpoint index disagrees"),
        ):
            changed = copy.deepcopy(delivered)
            changed["metadata"]["bass_management"]["routing_graph"]["routes"][1][field] = value
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, message):
                output_amplitudes(changed, [0.0, 80.0], {"L": 0.1, "LFE": 0.1}, RATE)

    def test_independent_channel_eq_replays_serialized_output(self):
        peak = 0.1 * 10 ** (6 / 20)
        delivered = {
            "global_plugins": [],
            "channels": {
                "Left": {
                    "channel": "Left",
                    "plugins": [{
                        "plugin_type": "eq",
                        "parameters": {
                            "filters": [{"filter_type": "peak", "freq": 80.0, "q": 2.0, "db_gain": 6.0}]
                        },
                    }],
                }
            },
            "metadata": {
                "stage_outcomes": [{
                    "stage": "final_graph_sampled_electrical_headroom",
                    "checks": [{
                        "id": 'sampled_physical_output:["channel","Left"]',
                        "kind": "safety",
                        "passed": True,
                        "observed": peak,
                        "limit": 1.0,
                        "diagnostic": json.dumps({
                            "output": '["channel","Left"]',
                            "inputs": ["Left"],
                            "input_peak_limits": {"Left": 0.1},
                            "sample_rate_hz": RATE,
                            "evaluated_band_hz": [0.0, RATE / 2],
                            "grid_points": 8194,
                            "peak_frequency_hz": 80.0,
                            "peak_amplitude": peak,
                        }),
                    }],
                }]
            },
        }
        self.assertAlmostEqual(verify_artifact(delivered)['["channel","Left"]'], peak, places=8)
        changed = copy.deepcopy(delivered)
        changed["channels"]["Left"]["plugins"][0]["parameters"]["filters"][0]["db_gain"] = 12.0
        with self.assertRaisesRegex(ValueError, "independent peak"):
            verify_artifact(changed)

    def test_independent_driver_plugins_stay_on_their_physical_outputs(self):
        common_gain = {"plugin_type": "gain", "parameters": {"gain_db": -6.0}}
        driver_eq = {
            "plugin_type": "eq",
            "parameters": {
                "filters": [{"filter_type": "peak", "freq": 80.0, "q": 2.0, "db_gain": 6.0}]
            },
        }
        delivered = {
            "global_plugins": [],
            "channels": {
                "Sub1": {
                    "channel": "Sub1",
                    "plugins": [common_gain],
                    "drivers": [
                        {"index": 0, "name": "A", "plugins": [driver_eq]},
                        {"index": 1, "name": "B", "plugins": []},
                    ],
                }
            },
        }
        outputs = output_amplitudes(delivered, [0.0, 80.0], {"Sub1": 0.1}, RATE)
        a = '["driver","Sub1",0,"A"]'
        b = '["driver","Sub1",1,"B"]'
        self.assertEqual(set(outputs), {a, b})
        self.assertAlmostEqual(outputs[a][1], 0.1, places=10)
        self.assertAlmostEqual(outputs[b][1], 0.1 * 10 ** (-6 / 20), places=10)

        duplicate = copy.deepcopy(delivered)
        duplicate["channels"]["Sub1"]["drivers"][1]["index"] = 0
        with self.assertRaisesRegex(ValueError, "unique indices and names"):
            output_amplitudes(duplicate, [0.0, 80.0], {"Sub1": 0.1}, RATE)

    def test_independent_convolution_is_explicitly_unsupported(self):
        delivered = {
            "global_plugins": [],
            "channels": {
                "Left": {
                    "channel": "Left",
                    "plugins": [{"plugin_type": "convolution", "parameters": {"ir_file": "unused.wav"}}],
                }
            },
        }
        with self.assertRaisesRegex(ValueError, "unsupported serial plugin: convolution"):
            output_amplitudes(delivered, [0.0, 80.0], {"Left": 0.1}, RATE)

    def test_bound_convolution_replays_fir_mix_and_checks_rate(self):
        delivered = {
            "global_plugins": [],
            "channels": {
                "Left": {
                    "channel": "Left",
                    "plugins": [{
                        "plugin_type": "convolution",
                        "parameters": {"ir_file": "left.wav", "mix": 0.5, "gain_db": 6.0},
                    }],
                }
            },
        }
        with patch("verify_routed_electrical.read_fir_wav", return_value=(48_000, [1.0, 0.5])) as reader:
            output = output_amplitudes(
                delivered, [0.0, RATE / 4], {"Left": 1.0}, RATE, Path("/Volumes/home_tmp/tmp")
            )
        self.assertEqual(reader.call_count, 1)
        output_name = '["channel","Left"]'
        gain = 10 ** (6 / 20)
        self.assertAlmostEqual(output[output_name][0], 0.5 + 0.75 * gain, places=10)
        self.assertAlmostEqual(output[output_name][1], abs(0.5 + (1 - 0.5j) * 0.5 * gain), places=10)

        with patch("verify_routed_electrical.read_fir_wav", return_value=(44_100, [1.0])):
            with self.assertRaisesRegex(ValueError, "sidecar sample rate differs"):
                output_amplitudes(
                    delivered, [0.0, 80.0], {"Left": 1.0}, RATE, Path("/Volumes/home_tmp/tmp")
                )

    def test_routed_physical_driver_convolution_is_output_owned(self):
        delivered = graph()
        driver_fir = {
            "plugin_type": "convolution",
            "parameters": {"room_eq_stage": "post_route", "ir_file": "driver.wav"},
        }
        delivered["channels"]["Sub1"]["drivers"] = [
            {"name": "Sub1", "index": 0, "plugins": [driver_fir]}
        ]
        with patch("verify_routed_electrical.read_fir_wav", return_value=(48_000, [0.5, 0.5])):
            output = output_amplitudes(
                delivered, [0.0, 80.0], {"L": 1.0}, RATE, Path("/Volumes/home_tmp/tmp")
            )
        self.assertAlmostEqual(output["Sub1"][0], 1.0, places=10)
        expected_fir = abs(
            0.5
            + 0.5
            * complex(
                math.cos(-2 * math.pi * 80 / RATE),
                math.sin(-2 * math.pi * 80 / RATE),
            )
        )
        self.assertAlmostEqual(output["Sub1"][1], 0.5 * expected_fir, places=10)

        with self.assertRaisesRegex(ValueError, "without saved resource location"):
            output_amplitudes(delivered, [0.0, 80.0], {"L": 1.0}, RATE)

    def test_report_mismatch_and_unsupported_dsp_fail_closed(self):
        delivered = graph()
        self.assertTrue(math.isclose(verify_artifact(delivered)["Sub1"], 1.0, abs_tol=1e-8))
        changed = copy.deepcopy(delivered)
        changed["channels"]["Sub1"]["plugins"].append(
            {
                "plugin_type": "eq",
                "parameters": {
                    "room_eq_stage": "post_route",
                    "filters": [{"filter_type": "peak", "freq": 80.0, "q": 1.0, "db_gain": 12.0}],
                },
            }
        )
        with self.assertRaisesRegex(ValueError, "independent peak"):
            verify_artifact(changed)
        changed["channels"]["Sub1"]["plugins"][0]["plugin_type"] = "convolution"
        with self.assertRaisesRegex(ValueError, "unsupported serial plugin"):
            verify_artifact(changed)

    def test_physical_driver_eq_center_is_part_of_replay_grid(self):
        changed = graph()
        changed["channels"]["Sub1"]["drivers"] = [
            {
                "name": "Sub1",
                "plugins": [{
                    "plugin_type": "eq",
                    "parameters": {
                        "room_eq_stage": "post_route",
                        "filters": [{"filter_type": "peak", "freq": 90.0, "q": 1.0, "db_gain": 12.0}],
                    },
                }],
            }
        ]
        check = changed["metadata"]["stage_outcomes"][0]["checks"][0]
        check["diagnostic"] = check["diagnostic"].replace('"grid_points":8194', '"grid_points":8195')
        with self.assertRaisesRegex(ValueError, "independent peak"):
            verify_artifact(changed)


if __name__ == "__main__":
    unittest.main()
