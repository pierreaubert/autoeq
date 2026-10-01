"""Draft 2020-12 schema resolution and a non-destructive JSON editor model."""
from __future__ import annotations
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Iterable
try:
    from jsonschema import Draft202012Validator
except ImportError:  # Allows source-tree smoke tests before optional dependencies are installed.
    Draft202012Validator = None  # type: ignore[assignment,misc]
from . import pointers

Json = Any

@dataclass(frozen=True)
class Field:
    pointer: str
    label: str
    kind: str
    schema: dict[str, Json]
    required: bool = False
    value: Json = None
    node_id: str = ""

class SchemaResolver:
    def __init__(self, root: dict[str, Json]): self.root = root
    def resolve(self, schema: dict[str, Json]) -> dict[str, Json]:
        seen: set[str] = set()
        while "$ref" in schema:
            ref = schema["$ref"]
            if not isinstance(ref, str) or not ref.startswith("#/") or ref in seen: break
            seen.add(ref); target: Json = self.root
            for part in ref[2:].split("/"): target = target[part.replace("~1", "/").replace("~0", "~")]
            schema = {**target, **{key: value for key, value in schema.items() if key != "$ref"}}
        return schema
    def variants(self, schema: dict[str, Json], value: Json = None) -> list[dict[str, Json]]:
        schema = self.resolve(schema)
        choices = schema.get("oneOf", schema.get("anyOf", []))
        if not choices: return [schema]
        return [self.resolve(choice) for choice in choices]
    def selected_variant(self, schema: dict[str, Json], value: Json) -> int:
        variants = self.variants(schema, value)
        # Tagged variants are common in the RoomEQ schema.  Prefer their
        # discriminator before invoking a standalone validator: a branch can
        # contain local refs whose document root is the full input schema, not
        # the branch itself.
        if isinstance(value, dict):
            for index, candidate in enumerate(variants):
                for name, property_schema in candidate.get("properties", {}).items():
                    property_schema = self.resolve(property_schema)
                    if "const" in property_schema and value.get(name) == property_schema["const"]:
                        return index
        for index, candidate in enumerate(variants):
            try:
                if not self._errors(candidate, value): return index
            # ``jsonschema`` correctly rejects a branch with a root-relative
            # ref when it is validated in isolation.  Declaration order is
            # the documented fallback for imported untagged unions.
            except Exception:
                continue
        return 0
    def default(self, schema: dict[str, Json]) -> Json:
        schema = self.resolve(schema)
        if "default" in schema: return deepcopy(schema["default"])
        if "const" in schema: return deepcopy(schema["const"])
        variants = schema.get("oneOf", schema.get("anyOf"))
        if variants: return self.default(self.resolve(variants[0]))
        types = schema.get("type", [])
        types = [types] if isinstance(types, str) else types
        non_null = next((item for item in types if item != "null"), None)
        if non_null == "object" or "properties" in schema:
            return {name: self.default(child) for name, child in schema.get("properties", {}).items() if name in schema.get("required", []) or "default" in self.resolve(child)}
        if non_null == "array": return []
        if "enum" in schema: return deepcopy(schema["enum"][0])
        return None
    @staticmethod
    def _errors(schema: dict[str, Json], value: Json) -> list[str]:
        if Draft202012Validator is not None: return [error.message for error in Draft202012Validator(schema).iter_errors(value)]
        if schema.get("type") == "object":
            if not isinstance(value, dict): return ["is not of type 'object'"]
            missing = [name for name in schema.get("required", []) if name not in value]
            if missing: return [f"missing required property {missing[0]}"]
            for name, child in schema.get("properties", {}).items():
                if name in value and SchemaResolver._errors(child, value[name]): return [f"{name} is invalid"]
        typ = schema.get("type")
        if typ == "string" and not isinstance(value, str): return ["is not of type 'string'"]
        if typ == "integer" and (not isinstance(value, int) or isinstance(value, bool)): return ["is not of type 'integer'"]
        if typ == "number" and not isinstance(value, (int, float)): return ["is not of type 'number'"]
        if "const" in schema and value != schema["const"]: return ["does not match const"]
        return []

class SchemaEditor:
    """Keeps imported unknown keys and inactive union drafts intact."""
    def __init__(self, schema: dict[str, Json], value: Json | None = None):
        self.schema, self.resolver = schema, SchemaResolver(schema)
        self.value = deepcopy(value) if value is not None else self.resolver.default(schema)
        self._drafts: dict[str, list[Json]] = {}
    def edit(self, pointer: str, value: Json) -> None: self.value = pointers.set(self.value, pointer, value)
    def select_variant(self, pointer: str, index: int) -> None:
        schema = self.schema_at(pointer); variants = self.resolver.variants(schema)
        if index >= len(variants): raise IndexError(index)
        drafts = self._drafts.setdefault(pointer, [None] * len(variants)); current = pointers.get(self.value, pointer)
        old = self.resolver.selected_variant(schema, current); drafts[old] = deepcopy(current)
        chosen = drafts[index] if drafts[index] is not None else self.resolver.default(variants[index])
        self.edit(pointer, chosen)
    def schema_at(self, pointer: str) -> dict[str, Json]:
        schema = self.schema
        for part in pointers.tokens(pointer):
            schema = self.resolver.resolve(schema)
            schema = schema.get("items", {}) if part.isdigit() else schema.get("properties", {}).get(part, schema.get("additionalProperties", {}))
        return self.resolver.resolve(schema)
    def fields(self, pointer: str = "") -> Iterable[Field]:
        schema = self.schema_at(pointer); value = pointers.get(self.value, pointer, self.value if not pointer else None)
        schema = self.resolver.variants(schema, value)[self.resolver.selected_variant(schema, value)]
        if "enum" in schema: yield Field(pointer, pointer.rsplit("/", 1)[-1] or "Configuration", "select", schema, value=value, node_id="field" + pointer.replace("/", "-")); return
        typ = schema.get("type")
        if isinstance(typ, list): typ = next((item for item in typ if item != "null"), "null")
        if typ == "object" or "properties" in schema:
            required = set(schema.get("required", []))
            for name, child in schema.get("properties", {}).items():
                child_pointer = pointers.join(pointer, name)
                yield Field(child_pointer, name, "object" if self.resolver.resolve(child).get("type") == "object" else "field", child, name in required, pointers.get(self.value, child_pointer), "field" + child_pointer.replace("/", "-"))
        else: yield Field(pointer, pointer.rsplit("/", 1)[-1], str(typ or "value"), schema, value=value, node_id="field" + pointer.replace("/", "-"))
    def errors(self) -> dict[str, list[str]]:
        out: dict[str, list[str]] = {}
        if Draft202012Validator is not None:
            for error in Draft202012Validator(self.schema).iter_errors(self.value):
                pointer = "".join("/" + pointers.escape(str(part)) for part in error.absolute_path)
                out.setdefault(pointer, []).append(error.message)
        else:
            for name, child in self.schema.get("properties", {}).items():
                for message in self.resolver._errors(self.resolver.resolve(child), self.value.get(name) if isinstance(self.value, dict) else None):
                    out.setdefault(pointers.join("", name), []).append(message)
        return out
