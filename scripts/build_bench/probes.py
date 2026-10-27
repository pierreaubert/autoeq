"""Reversible, real Rust edits confined to independent source snapshots."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
from pathlib import Path
import re

NUMBER = r"[0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?(?:_[uif][0-9]+)?"
PRIVATE = re.compile(r"\.(?:max|min|saturating_add|saturating_sub|powi|take|skip|unwrap_or)\(\s*(" + NUMBER + r"|&\[\])\s*[,)]")
TEST = re.compile(r"assert_eq!\([^\n;]*,\s*(" + NUMBER + r")\s*[,)]")
PUBLIC_TYPES = {"autoeq-core": "Curve", "autoeq-objective": "ObjectiveData", "autoeq-optim": "OptimParams", "roomeq-model": "RoomConfig", "roomeq-engine": "room_result::RoomOptimizationResult", "roomeq-workflow": "RoomPipelineRequest<'_>", "autoeq-report-wasm": "ReportPayload"}


@dataclass(frozen=True)
class Edit:
    path: Path
    original: bytes
    changed: bytes
    kind: str
    line: int

    def report(self, root: Path) -> dict:
        return {"path": str(self.path.relative_to(root)), "kind": self.kind, "line": self.line, "before_sha256": hashlib.sha256(self.original).hexdigest(), "after_sha256": hashlib.sha256(self.changed).hexdigest()}


def test_ranges(text: str) -> list[tuple[int, int]]:
    # Mask comments/strings before balancing item braces. Preserve positions.
    masked = re.sub(r'//[^\n]*|/\*[\s\S]*?\*/|"(?:\\.|[^"\\])*"',
                    lambda match: " " * len(match.group()), text)
    ranges = []
    for marker in re.finditer(r"#\[(?:cfg\(test\)|test)\]", masked):
        start = marker.end()
        opening = masked.find("{", start)
        terminator = masked.find(";", start)
        if terminator >= 0 and (opening < 0 or terminator < opening):
            ranges.append((marker.start(), terminator + 1))
            continue
        depth = 0
        for position in range(opening, len(masked)):
            depth += (masked[position] == "{") - (masked[position] == "}")
            if depth == 0:
                ranges.append((marker.start(), position + 1))
                break
    return ranges


def literal_edit(path: Path, kind: str, test_source: bool = False) -> Edit | None:
    original = path.read_bytes()
    text = original.decode("utf-8")
    ranges = test_ranges(text)
    pattern = PRIVATE if kind == "private-edit" else TEST
    for match in pattern.finditer(text):
        start, end = match.span(1)
        in_test = test_source or any(low <= start < high for low, high in ranges)
        if (kind == "private-edit" and in_test) or (kind == "test-edit" and not in_test):
            continue
        line_start = text.rfind("\n", 0, start) + 1
        prefix = text[line_start:start]
        if prefix.lstrip().startswith(("//", "*")) or '"' in prefix:
            continue
        literal = text[start:end]
        replacement = "{ let autoeq_build_bench_value = " + literal + "; autoeq_build_bench_value }"
        return Edit(path, original, (text[:start] + replacement + text[end:]).encode(), kind, text.count("\n", 0, start) + 1)
    return None


def select_edit(root: Path, package: dict, kind: str, explicit_file: str | None = None,
                public_type: str | None = None, public_function: bool = False) -> Edit:
    crate = Path(package["manifest_path"]).parent
    if kind == "public-edit":
        if public_function:
            if explicit_file is None:
                raise ValueError("a public function probe requires its owning source file")
            path = root / explicit_file
            addition = "\n#[doc(hidden)]\npub fn autoeq_build_bench_contract_probe() -> bool { true }\n"
        else:
            owned_type = public_type or PUBLIC_TYPES.get(package["name"])
            if owned_type is None:
                raise ValueError(f"no public contract probe defined for {package['name']}")
            lib = next(target for target in package["targets"] if "lib" in target["kind"] or "rlib" in target["kind"])
            path = Path(lib["src_path"])
            addition = f"\nimpl crate::{owned_type} {{\n    #[doc(hidden)]\n    pub fn autoeq_build_bench_contract_probe(&self) -> bool {{ true }}\n}}\n"
        if not path.resolve().is_relative_to(crate.resolve()):
            raise ValueError("public contract probe escapes its owning package")
        original = path.read_bytes()
        if b"autoeq_build_bench_contract_probe" in original:
            raise ValueError("public contract probe already exists")
        return Edit(path, original, original + addition.encode(), kind, len(original.splitlines()) + 1)
    candidates = [root / explicit_file] if explicit_file else sorted((crate / "src").rglob("*.rs"))
    if kind == "test-edit" and not explicit_file:
        candidates.sort(key=lambda path: ("test" not in path.name, str(path)))
    for path in candidates:
        if not path.resolve().is_relative_to(crate.resolve()):
            raise ValueError("edit probe escapes its owning package")
        if kind == "private-edit" and any("test" in part for part in path.relative_to(crate).parts):
            continue
        edit = literal_edit(path, kind, any("test" in part for part in path.relative_to(crate).parts))
        if edit is not None:
            return edit
    raise ValueError(f"no active-code {kind} probe found for {package['name']}; specify --probe-file")


@contextmanager
def applied(edit: Edit, snapshot_root: Path):
    if not edit.path.resolve().is_relative_to(snapshot_root.resolve()):
        raise ValueError("refusing an edit outside the source snapshot")
    if edit.path.read_bytes() != edit.original:
        raise ValueError("probe source changed before the edit")
    edit.path.write_bytes(edit.changed)
    try:
        yield
    finally:
        edit.path.write_bytes(edit.original)
