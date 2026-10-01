from __future__ import annotations
import json, os
from pathlib import Path
from typing import Any
from .schema import SchemaEditor

class RoomEqDocument:
    def __init__(self, editor: SchemaEditor, path: Path | None = None): self.editor, self.path, self.dirty = editor, path, False
    @classmethod
    def create(cls, schema: dict[str, Any]) -> "RoomEqDocument": return cls(SchemaEditor(schema))
    @classmethod
    def open(cls, schema: dict[str, Any], path: Path) -> "RoomEqDocument": return cls(SchemaEditor(schema, json.loads(path.read_text())), path)
    def edit(self, pointer: str, value: Any) -> None: self.editor.edit(pointer, value); self.dirty = True
    def save(self, path: Path | None = None) -> Path:
        target = path or self.path
        if target is None: raise ValueError("Save As requires a configuration path")
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + ".tmp")
        temporary.write_text(json.dumps(self.editor.value, indent=2) + "\n")
        os.replace(temporary, target); self.path, self.dirty = target, False
        return target
    def result_path(self, explicit: Path | None = None) -> Path:
        if explicit: return explicit
        if not self.path: raise ValueError("Save the configuration before choosing an output")
        return self.path.with_name(self.path.stem + ".result.json")
