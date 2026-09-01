"""The thin GPUI presentation and coordination layer; no acoustics live here."""
from __future__ import annotations
import json
import os
import sys
from pathlib import Path
from typing import Any
from .commands import RoomEqCommand
from .document import RoomEqDocument
from .review import ResultReview
from .schema import SchemaEditor
from gpui_toolkit import App, section, ui, charts

class RoomEqGuiApp(App):
    def __init__(self, input_schema: dict[str, Any], output_schema: dict[str, Any], command: RoomEqCommand, config: Path | None = None, result: Path | None = None):
        self.command, self.input_schema, self.output_schema = command, input_schema, output_schema
        self.document = RoomEqDocument.open(input_schema, config) if config else RoomEqDocument.create(input_schema)
        self.result = json.loads(result.read_text()) if result else None
        self.schema_warning: str | None = None; self.logs: list[str] = []; self.step = 0
        self.preferences: dict[str, Any] = {"recent_configs": [], "recent_results": [], "selected_binary": str(command.binary) if command.binary else None, "review": {"smoothing": "1/6 octave", "auto_scale": False, "trend": False}}
        try:
            from gpui_toolkit import StateStore
            stored = StateStore("org.autoeq.roomeq-gui").load(version=1, default={})
            if isinstance(stored.state, dict): self.preferences.update(stored.state)
            self._state_store = StateStore("org.autoeq.roomeq-gui")
        except (ImportError, OSError, ValueError, TypeError): self._state_store = None
        super().__init__(title="RoomEQ", sidebar_title="RoomEQ", sections=self._sections())
    def save_preferences(self) -> None:
        """Only navigation/review preferences are persisted; document JSON is never stored here."""
        if self._state_store: self._state_store.save(self.preferences, version=1)
    def remember(self, path: Path, kind: str) -> None:
        key = f"recent_{kind}"; values = [str(path), *[item for item in self.preferences.get(key, []) if item != str(path)]]
        self.preferences[key] = values[:10]
    def validate(self) -> dict[str, list[str]]:
        if self.document.dirty or not self.document.path: raise ValueError("Save the configuration before validation.")
        errors = self.document.editor.errors()
        if errors: return errors
        run = self.command.run(self.document.path, self.document.result_path(), dry_run=True, log=self.logs.append)
        return {} if run.returncode == 0 else {"": [run.output or "RoomEQ dry-run failed"]}
    def optimize(self) -> None:
        if self.document.dirty or not self.document.path: raise ValueError("Save the configuration before optimization.")
        output = self.document.result_path(); run = self.command.run(self.document.path, output, log=self.logs.append)
        if run.returncode == 0 and output.exists():
            self.result = json.loads(output.read_text()); self.remember(output, "results"); self.step = 3
    def review(self) -> ResultReview | None: return ResultReview(self.result) if self.result else None
    def _sections(self) -> list[Any]:
        """Stable native-GPUI IR. Kept small enough for the host session payload."""
        errors = self.document.editor.errors()
        controls = []
        for field in self.document.editor.fields():
            value, schema = field.value, self.document.editor.resolver.resolve(field.schema)
            validation = {"severity": "error", "message": "; ".join(errors.get(field.pointer, []))} if field.pointer in errors else None
            if "enum" in schema: controls.append(ui.select(id=field.node_id, label=field.label, value=str(value), options=[(str(x), str(x)) for x in schema["enum"]], action="edit-field", validation=validation))
            elif schema.get("type") == "boolean": controls.append(ui.toggle(id=field.node_id, label=field.label, value=bool(value), action="edit-field"))
            elif schema.get("type") in ("integer", "number"): controls.append(ui.number_input(id=field.node_id, label=field.label, value=value or 0, action="edit-field", validation=validation))
            elif "path" in field.label.lower() or "file" in field.label.lower(): controls.append(ui.path_input(id=field.node_id, label=field.label, value=str(value or ""), action="edit-field", validation=validation))
            elif schema.get("type") == "object" or "properties" in schema:
                children = [ui.text_input(id=child.node_id, label=child.label, value="" if child.value is None else str(child.value), action="edit-field") for child in self.document.editor.fields(field.pointer)]
                controls.append(ui.accordion(id="accordion" + field.pointer.replace("/", "-"), items=[(field.node_id, field.label, children)], action="accordion"))
            elif schema.get("type") == "array": controls.append(ui.list_editor(id=field.node_id, label=field.label, rows=[], add_action="array-add", remove_action="array-remove"))
            else: controls.append(ui.text_input(id=field.node_id, label=field.label, value="" if value is None else str(value), action="edit-field", validation=validation))
        review = self.review(); review_nodes = []
        if review:
            review_nodes.append(ui.table(id="review-metrics", headers=["Metric", "Value"], rows=[[key, value] for key, value in review.overview().items() if not isinstance(value, (dict, list))]))
            for series in review.curves()[:8]: review_nodes.append(charts.line("chart" + series.id.replace("/", "-"), series.x, series.y, title=series.label, x_label="Hz", y_label="dB"))
        else: review_nodes.append(ui.empty_state("No result loaded", description="Open a DspChainOutput or run optimization to review it."))
        return [
            section("workflow", "Configure", ui.vstack([ui.stepper(id="workflow-stepper", steps=["Configure", "Validate", "Optimize", "Review"], active=self.step, action="step"), ui.accordion(id="config-form", items=[("configuration", "Configuration", controls)], expanded=["configuration"]), ui.button("Save", id="save", action="save"), ui.button("Validate", id="validate", action="validate")])),
            section("review", "Review", ui.vstack(review_nodes)),
        ]
    def ir(self) -> dict[str, Any]: return self.to_spec()
    def run(self) -> None:
        # The native host otherwise prefers its own repository venv.  A
        # console-script installation lives in this interpreter's site-packages,
        # so ensure the supervised child uses the same interpreter.
        os.environ.setdefault("GPUI_PYTHON", sys.executable)
        super().run()
