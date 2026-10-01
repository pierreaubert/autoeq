from __future__ import annotations
import json, sys, tempfile, unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from roomeq_gui.commands import RoomEqCommand

sibling = (Path(__file__).resolve().parents[3].parent
           / "gpui-toolkit" / "crates" / "gpui-python-runtime" / "python")
if (sibling / "gpui_toolkit" / "__init__.py").is_file():
    sys.path.insert(0, str(sibling))
try:
    from roomeq_gui.app import RoomEqGuiApp
    TOOLKIT_OK = True
except ImportError:
    TOOLKIT_OK = False

SCHEMA = {"type": "object", "properties": {"name": {"type": "string", "default": "new"}}}


def make_app(tmp: Path) -> RoomEqGuiApp:
    result = tmp / "out.result.json"
    result.write_text(json.dumps({
        "version": "1", "pre_score": 2.0, "post_score": 4.0,
        "left": {"frequencies": [100.0, 1000.0, 10000.0],
                 "values": [80.0, 82.0, 79.0]},
    }))
    return RoomEqGuiApp(SCHEMA, {}, RoomEqCommand(None), None, result)


@unittest.skipUnless(TOOLKIT_OK, "gpui_toolkit not importable")
class AppIrTests(unittest.TestCase):
    def test_sections_charts_and_resources(self):
        with tempfile.TemporaryDirectory() as directory:
            app = make_app(Path(directory))
        spec = app.ir()
        self.assertEqual(spec["schema_version"], 1)
        self.assertEqual([s["id"] for s in spec["sections"]],
                         ["workflow", "review"])

        def walk(node, kind):
            found = []
            if isinstance(node, dict):
                if node.get("kind") == kind:
                    found.append(node)
                for value in node.values():
                    found.extend(walk(value, kind))
            elif isinstance(node, list):
                for value in node:
                    found.extend(walk(value, kind))
            return found

        charts = walk(spec, "px_chart_v2")
        self.assertEqual(len(charts), 1)
        self.assertEqual(charts[0]["title"], "left")
        resources = {r.id: r.to_spec() for r in app.resources}
        self.assertEqual(len(resources), 1)
        dataset_id = next(iter(resources))
        self.assertEqual(resources[dataset_id]["row_count"], 3)
        self.assertIn(dataset_id, json.dumps(charts[0]))

    def test_miniapp_shell_enables_themes(self):
        app = RoomEqGuiApp(SCHEMA, {}, RoomEqCommand(None))
        miniapp = app.ir()["miniapp"]
        self.assertEqual(miniapp["title"], "RoomEQ")
        self.assertEqual(miniapp["app_name"], "RoomEQ")
        self.assertTrue(miniapp["with_theme"])
        self.assertEqual(miniapp["initial_theme"], "dark")

    def test_session_child_argv_rebuild(self):
        from roomeq_gui.__main__ import (
            _SESSION_ARGS_ENV, _session_child_argv)
        env = {_SESSION_ARGS_ENV: json.dumps({
            "roomeq": "/tmp/roomeq", "config": "/tmp/room.json",
            "result": None})}
        self.assertEqual(
            _session_child_argv(env),
            ["--roomeq", "/tmp/roomeq", "--config", "/tmp/room.json"])
        self.assertIsNone(_session_child_argv({}))
        self.assertIsNone(_session_child_argv({_SESSION_ARGS_ENV: "{bogus"}))

    def test_no_result_renders_empty_state(self):
        app = RoomEqGuiApp(SCHEMA, {}, RoomEqCommand(None))
        dumped = json.dumps(app.ir())
        self.assertIn("empty_state", dumped)
        self.assertEqual(app.resources, ())


if __name__ == "__main__":
    unittest.main()
