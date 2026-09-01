from __future__ import annotations
import json, tempfile, threading, unittest
from pathlib import Path
from roomeq_gui.commands import RoomEqCommand
from roomeq_gui.document import RoomEqDocument
from roomeq_gui.review import ResultReview, display_series
from roomeq_gui.schema import SchemaEditor

SCHEMA = {"$schema":"https://json-schema.org/draft/2020-12/schema","type":"object","properties":{"name":{"type":"string","default":"new"},"enabled":{"type":["boolean","null"]},"choice":{"oneOf":[{"type":"object","properties":{"kind":{"const":"a"},"a":{"type":"integer","default":1}},"required":["kind"]},{"type":"object","properties":{"kind":{"const":"b"},"b":{"type":"string","default":"x"}},"required":["kind"]}]},"items":{"type":"array","items":{"type":"number"}},"map":{"type":"object","additionalProperties":{"type":"string"}}},"required":["name"]}

class SchemaEditorTests(unittest.TestCase):
    def test_variants_and_unknown_fields(self):
        editor = SchemaEditor(SCHEMA, {"name":"kept","unknown":9,"choice":{"kind":"a","a":3}})
        self.assertEqual(editor.value["unknown"], 9)
        editor.select_variant("/choice", 1); self.assertEqual(editor.value["choice"]["kind"], "b")
        editor.select_variant("/choice", 0); self.assertEqual(editor.value["choice"]["a"], 3)
        editor.edit("/map/a~1b", "ok"); self.assertEqual(editor.value["map"]["a/b"], "ok")
    def test_errors_map_to_pointers(self):
        self.assertIn("/name", SchemaEditor(SCHEMA, {"name":3}).errors())

class DocumentTests(unittest.TestCase):
    def test_atomic_save_and_result_path(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "room.json"; document = RoomEqDocument.create(SCHEMA)
            with self.assertRaises(ValueError): document.result_path()
            document.edit("/name", "saved"); document.save(path)
            self.assertFalse(document.dirty); self.assertEqual(json.loads(path.read_text())["name"], "saved")
            self.assertEqual(document.result_path().name, "room.result.json")

class CommandTests(unittest.TestCase):
    def test_safe_argv_and_cancel(self):
        with tempfile.TemporaryDirectory() as directory:
            binary = Path(directory) / "roomeq"; binary.write_text("#!/bin/sh\nsleep 10\n"); binary.chmod(0o755)
            command = RoomEqCommand(RoomEqCommand.discover(str(binary)))
            self.assertEqual(command.argv(Path("a b.json"), Path("out.json")), [str(binary), "--config", "a b.json", "--output", "out.json"])
            cancel = threading.Event(); threading.Timer(.05, cancel.set).start()
            self.assertTrue(command.run(Path("in"), Path("out"), cancel=cancel).cancelled)

class ReviewTests(unittest.TestCase):
    def test_series_validation_and_metrics(self):
        self.assertEqual(len(display_series("x", list(range(1,3002)), list(range(3001))).x), 2000)
        with self.assertRaises(ValueError): display_series("x", [0], [1])
        review = ResultReview({"version":"1","pre_score":2,"post_score":4,"epa":{"ok":True}})
        self.assertEqual(review.overview()["improvement"], 2); self.assertEqual(review.curves(), [])

class BaselineTests(unittest.TestCase):
    def test_bundled_schemas_match_checked_in_contracts(self):
        root = Path(__file__).resolve().parents[3]
        for kind in ("input", "output"):
            self.assertEqual(json.loads((root / "python/roomeq-gui/roomeq_gui/resources" / f"{kind}_schema.json").read_text()), json.loads((root / "src/bin/roomeq" / f"{kind}_schema.json").read_text()))
