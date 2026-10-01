"""Tests for the recording wizard (backends + model + app IR)."""

import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from recording_gui import backend as be
from recording_gui import model as mo
from recording_gui._toolkit import ensure_toolkit

try:
    ensure_toolkit()
    TOOLKIT_OK = True
except RuntimeError:
    TOOLKIT_OK = False


class FakeContext:
    """Minimal SessionContext double recording sends/patches/errors."""

    def __init__(self):
        self.acknowledged = []
        self.patches = []
        self.errors = []

    def acknowledge(self, event):
        self.acknowledged.append(event.id)

    def patch(self, ops, request_id=None):
        self.patches.extend(ops)

    def error(self, request_id, code, message):
        self.errors.append((code, message))


def make_event(action, value=None):
    from gpui_toolkit import Event
    return Event(id="e1", sequence=0, node_id="n", event="click",
                 action=action,
                 payload={} if value is None else {"value": value})


class ModelTests(unittest.TestCase):
    def test_init_takes_routes_hardware(self):
        model = mo.WizardModel()
        model.init_takes(["L", "R"])
        self.assertEqual(
            [(t.label, t.out_channel, t.in_channel) for t in model.takes],
            [("L", 0, 0), ("R", 1, 0)])
        model.init_takes(["L"], out_channels=[2], in_channels=[1])
        self.assertEqual(
            (model.takes[0].out_channel, model.takes[0].in_channel), (2, 1))

    def test_step_guards(self):
        model = mo.WizardModel()
        self.assertFalse(model.retreat())
        self.assertTrue(model.advance())
        model.busy = True
        self.assertFalse(model.advance())
        self.assertFalse(model.retreat())
        model.busy = False
        model.step = len(mo.STEPS) - 1
        self.assertFalse(model.advance())

    def test_progress_and_metadata(self):
        model = mo.WizardModel()
        self.assertEqual(model.overall_progress, 0.0)
        model.init_takes(["L", "R"])
        model.takes[0].progress = 1.0
        model.takes[0].state = mo.TAKE_DONE
        self.assertAlmostEqual(model.overall_progress, 0.5)
        self.assertEqual(len(model.pending_takes), 1)
        self.assertTrue(model.metadata_is_valid())
        model.room_width_m = 4.0
        self.assertFalse(model.metadata_is_valid())
        model.room_depth_m = 5.0
        model.room_height_m = 2.7
        self.assertTrue(model.metadata_is_valid())

    def test_signal_amplitude_clips(self):
        self.assertAlmostEqual(mo.SignalConfig().amplitude(), 0.5, places=3)
        self.assertEqual(mo.SignalConfig(level_db=0.0).amplitude(), 1.0)
        self.assertEqual(mo.SignalConfig(level_db=6.0).amplitude(), 1.0)
        self.assertEqual(
            mo.SignalConfig(level_db=float("-inf")).amplitude(), 0.5)

    def test_session_payload_is_json_serializable(self):
        model = mo.WizardModel()
        model.init_takes(["L"])
        payload = model.session_payload()
        self.assertEqual(json.loads(json.dumps(payload))["schema"],
                         "sotf-recording-wizard-v1")

    def test_device_from_cli_is_defensive(self):
        full = mo.DeviceInfo.from_cli({
            "name": "USB", "device_id": "7", "is_default": True,
            "default_config": {"channels": 2, "sample_rate": 48000,
                               "sample_format": "F32"}})
        self.assertEqual((full.channels, full.sample_rate), (2, 48000))
        self.assertIn("default", full.label())
        bare = mo.DeviceInfo.from_cli({})
        self.assertEqual((bare.name, bare.channels), ("?", 0))
        broken = mo.DeviceInfo.from_cli(
            {"name": "X", "default_config": {"channels": "many"}})
        self.assertEqual(broken.channels, 0)


class BackendTests(unittest.TestCase):
    def test_fake_backend_is_deterministic(self):
        fake = be.FakeCaptureBackend()
        inputs, outputs = fake.list_devices()
        self.assertTrue(inputs and outputs)
        first = fake.capture(be.TakeRequest(name="L")).summaries[0]["curve"]
        second = fake.capture(be.TakeRequest(name="L")).summaries[0]["curve"]
        self.assertEqual(first, second)
        other = fake.capture(be.TakeRequest(name="R")).summaries[0]["curve"]
        self.assertNotEqual(first["spl"], other["spl"])
        self.assertEqual(len(fake.requests), 3)
        self.assertEqual(fake.save_session("s.json", {"a": 1}), "s.json")
        self.assertEqual(fake.saved["s.json"], {"a": 1})

    def test_resolve_modes(self):
        backend, demo = be.resolve_backend("fake")
        self.assertTrue(demo)
        self.assertEqual(backend.name, "fake")
        with self.assertRaises(ValueError):
            be.resolve_backend("nope")
        with self.assertRaises(RuntimeError):
            be.resolve_backend("cli", binary="/nonexistent/sotf-capture")
        auto, auto_demo = be.resolve_backend(
            "auto", binary="/nonexistent/sotf-capture")
        self.assertTrue(auto_demo)
        self.assertEqual(auto.name, "fake")

    def test_channel_list(self):
        self.assertEqual(be._channel_list((0, 2)), "0,2")


@unittest.skipUnless(TOOLKIT_OK, "gpui_toolkit not importable")
class WizardAppTests(unittest.TestCase):
    def _app(self, channels=("L", "R")):
        from recording_gui.app import build_app
        fake = be.FakeCaptureBackend()
        model = mo.WizardModel(demo_mode=True)
        inputs, outputs = fake.list_devices()
        model.input_devices = inputs
        model.output_devices = outputs
        model.input_name = inputs[0].name
        model.output_name = outputs[0].name
        model.init_takes(list(channels))
        return build_app(model, fake), model, fake

    def test_sections_and_steppers(self):
        app, _, _ = self._app()
        spec = app.to_spec()
        self.assertEqual(
            [s["id"] for s in spec["sections"]],
            [step_id for step_id, _ in mo.STEPS])
        steppers = []

        def collect(node):
            if isinstance(node, dict):
                if node.get("kind") == "stepper":
                    steppers.append(node)
                for value in node.values():
                    collect(value)
            elif isinstance(node, list):
                for value in node:
                    collect(value)

        collect(spec)
        self.assertEqual(len(steppers), len(mo.STEPS))
        for index, stepper in enumerate(steppers):
            self.assertEqual(stepper["active"], index)
            self.assertEqual(stepper["steps"],
                             [label for _, label in mo.STEPS])

    def test_audio_meters_have_valid_channels(self):
        # The host rejects meters with zero channels or mismatched
        # levels/peaks/channel_names at init ("invalid channels"); the Node
        # constructors do not validate, so pin the contract here.
        app, _, _ = self._app()
        meters = []

        def collect(node):
            if isinstance(node, dict):
                if node.get("kind") in ("audio_level_meter",
                                         "audio_horizontal_meter"):
                    meters.append(node)
                for value in node.values():
                    collect(value)
            elif isinstance(node, list):
                for value in node:
                    collect(value)

        collect(app.to_spec())
        self.assertTrue(meters, "expected at least one audio meter")
        for meter in meters:
            levels = meter.get("levels") or []
            peaks = meter.get("peaks") or []
            names = meter.get("channel_names") or []
            self.assertTrue(1 <= len(levels) <= 128, meter.get("id"))
            self.assertEqual(len(peaks), len(levels), meter.get("id"))
            self.assertEqual(len(names), len(levels), meter.get("id"))

    def test_miniapp_shell_enables_themes(self):
        app, _, _ = self._app()
        miniapp = app.to_spec()["miniapp"]
        self.assertEqual(miniapp["title"], "Recording wizard")
        self.assertEqual(miniapp["app_name"], "Recording wizard")
        self.assertTrue(miniapp["with_theme"])
        self.assertEqual(miniapp["initial_theme"], "dark")

    def test_config_select_options_carry_devices(self):
        app, _, _ = self._app()
        dumped = json.dumps(app.to_spec())
        self.assertIn("Fake Microphone", dumped)
        self.assertIn("Fake Speakers", dumped)
        self.assertIn("wizard_output_device", dumped)

    def test_capture_start_runs_all_takes(self):
        app, model, fake = self._app()
        ctx = FakeContext()
        app.on_action(make_event("capture_start"), ctx)
        self.assertEqual(ctx.acknowledged, ["e1"])
        self.assertTrue(all(t.state == mo.TAKE_DONE for t in model.takes))
        self.assertEqual(len(fake.requests), 2)
        self.assertEqual(fake.requests[0].out_channels, (0,))
        self.assertEqual(fake.requests[1].out_channels, (1,))
        progress = [op["value"] for op in ctx.patches
                    if op.get("id") == "capture-progress"]
        self.assertIn(1.0, progress)
        status = [op["value"] for op in ctx.patches
                  if op.get("id") == "capture-status"]
        self.assertTrue(any("2/2 takes done" in value for value in status))
        self.assertFalse(model.busy)

    def test_capture_cancel_sets_flag_and_status(self):
        app, model, _ = self._app()
        ctx = FakeContext()
        app.on_action(make_event("capture_cancel"), ctx)
        self.assertTrue(model.cancel_requested)
        status = [op["value"] for op in ctx.patches
                  if op.get("id") == "capture-status"]
        self.assertTrue(any("cancel requested" in value for value in status))
        # A fresh run clears the flag (cancel applies between takes).
        app.on_action(make_event("capture_start"), ctx)
        self.assertTrue(all(t.state == mo.TAKE_DONE for t in model.takes))

    def test_spl_probe_bass_and_save_flows(self):
        app, model, fake = self._app()
        ctx = FakeContext()
        app.on_action(make_event("spl_start"), ctx)
        self.assertTrue(model.spl_result["ok"])
        results = [op for op in ctx.patches if op.get("id") == "spl-result"]
        self.assertEqual(len(results), 1)
        app.on_action(make_event("probe_start"), ctx)
        app.on_action(make_event("bass_start"), ctx)
        self.assertEqual(len(model.probe_results), 2)
        self.assertEqual(len(model.bass_results), 2)
        app.on_action(make_event("wizard_save"), ctx)
        self.assertEqual(list(fake.saved), ["./recording.json"])
        status = [op["value"] for op in ctx.patches
                  if op.get("id") == "save-status"]
        self.assertTrue(any("saved" in value for value in status))

    def test_save_rejects_partial_room_dimensions(self):
        app, model, _ = self._app()
        model.room_width_m = 4.0
        ctx = FakeContext()
        app.on_action(make_event("wizard_save"), ctx)
        self.assertTrue(any("room dimensions" in value for value in
                            [op["value"] for op in ctx.patches]))

    def test_control_edits_flow_back_to_model(self):
        app, model, _ = self._app()
        ctx = FakeContext()
        app.on_action(make_event("wizard_signal_type", "pink-noise"), ctx)
        self.assertEqual(model.signal.signal_type, "pink-noise")
        app.on_action(make_event("wizard_signal_type", "nope"), ctx)
        self.assertEqual(model.signal.signal_type, "pink-noise")
        self.assertTrue(ctx.errors)
        app.on_action(make_event("wizard_level", -12.0), ctx)
        self.assertEqual(model.signal.level_db, -12.0)
        app.on_action(make_event("wizard_level", 99.0), ctx)
        self.assertEqual(model.signal.level_db, -12.0)
        app.on_action(make_event("wizard_save_name", "live-room"), ctx)
        self.assertEqual(model.save_name, "live-room")
        app.on_action(make_event("wizard_output_device",
                                 "Fake Speakers"), ctx)
        self.assertEqual(model.output_name, "Fake Speakers")
        app.on_action(make_event("bogus_action"), ctx)
        self.assertEqual(ctx.errors[-1][0], "unknown_action")

    def test_evaluating_charts_from_preloaded_curves(self):
        from recording_gui.app import build_app
        fake = be.FakeCaptureBackend()
        model = mo.WizardModel()
        app = build_app(model, fake, {
            "L": be.synth_take_curve("L"),
            "broken": {"freq": [1.0], "spl": []},
        })
        spec = app.to_spec()
        dumped = json.dumps(spec)
        self.assertIn("chart-eval-L", dumped)
        self.assertEqual(len(app.resources), 1)


if __name__ == "__main__":
    unittest.main()
