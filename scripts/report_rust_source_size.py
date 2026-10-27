# /// script
# requires-python = ">=3.12"
# dependencies = ["tree-sitter==0.25.2", "tree-sitter-rust==0.24.0"]
# ///
"""Conservative physical-line inventory; optional audit tool, not a build dependency.

Run with uv run --script scripts/report_rust_source_size.py ROOT OUTPUT.
Ordinary lines may include feature-gated or macro-generated test support. This is
source attribution, not expanded Rust SLOC or evidence of build performance.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

from tree_sitter import Language, Parser
import tree_sitter_rust

PARSER = Parser(Language(tree_sitter_rust.language()))
TOKEN = re.compile(r'"(?:[^"\\]|\\.)*"|[A-Za-z_][A-Za-z_0-9]*|[(),=]')


def cfg_values(expression: str) -> set[bool]:
    """Possible values with test=false and every other cfg left unknown."""
    tokens = TOKEN.findall(expression)
    index = 0

    def parse() -> set[bool]:
        nonlocal index
        name = tokens[index]
        index += 1
        if index < len(tokens) and tokens[index] == "(":
            index += 1
            children = []
            while tokens[index] != ")":
                children.append(parse())
                if tokens[index] == ",":
                    index += 1
            index += 1
            if name == "not" and len(children) == 1:
                return {not value for value in children[0]}
            if name in {"all", "any"}:
                values = {name == "all"}
                for child in children:
                    values = {(a and b) if name == "all" else (a or b)
                              for a in values for b in child}
                return values
            return {True, False}
        if index < len(tokens) and tokens[index] == "=":
            index += 2
            return {True, False}
        return {False} if name == "test" else {True, False}

    try:
        result = parse()
        return result if index == len(tokens) else {True, False}
    except (IndexError, ValueError):
        return {True, False}


def test_only(attributes: list[str]) -> bool:
    for attribute in attributes:
        compact = re.sub(r"\s+", "", attribute)
        if compact in {"#[test]", "#[bench]"}:
            return True
        match = re.fullmatch(r"#!?\[cfg\((.*)\)\]", compact)
        if match and cfg_values(match[1]) == {False}:
            return True
    return False


def inspect(path: Path, crate_root: bool = False) -> dict:
    data = path.read_bytes()
    tree = PARSER.parse(data)
    result = {"lines": len(data.splitlines()), "test_lines": set(),
              "edges": [], "issues": [], "sha256": hashlib.sha256(data).hexdigest()}
    if tree.root_node.has_error:
        result["issues"].append("tree-sitter parse error; review attribution")

    def source(node):
        return data[node.start_byte:node.end_byte].decode("utf-8")

    def walk(parent, module_dir: Path, inherited=False):
        attributes = []
        inner = [source(n) for n in parent.named_children if n.type == "inner_attribute_item"]
        inherited = inherited or test_only(inner)
        if inherited:
            result["test_lines"].update(range(parent.start_point.row, parent.end_point.row + 1))
        for node in parent.named_children:
            if node.type == "attribute_item":
                attributes.append(node)
                continue
            if node.type in {"line_comment", "block_comment", "inner_attribute_item"}:
                continue
            text_attrs = [source(a) for a in attributes]
            only_test = inherited or test_only(text_attrs)
            if only_test:
                start = attributes[0].start_point.row if attributes else node.start_point.row
                end = node.end_point.row + (node.end_point.column > 0)
                result["test_lines"].update(range(start, end))
            if node.type == "mod_item":
                name = source(node.child_by_field_name("name"))
                body = node.child_by_field_name("body")
                path_attribute = next((re.search(r'path\s*=\s*"([^"\\]+)"', a)
                                       for a in text_attrs if re.search(r'path\s*=', a)), None)
                if body is not None:
                    walk(body, module_dir / name, only_test)
                else:
                    candidates = ([path.parent / path_attribute[1]] if path_attribute else
                                  [module_dir / (name + ".rs"), module_dir / name / "mod.rs"])
                    existing = [candidate.resolve() for candidate in candidates if candidate.is_file()]
                    if len(existing) == 1:
                        result["edges"].append((existing[0], only_test))
                    else:
                        result["issues"].append("unresolved or ambiguous module: " + name)
            else:
                # Literal includes carry the surrounding test context into the file.
                node_source = source(node)
                if re.match(r"include!\s*\(", node_source):
                    literal = re.fullmatch(r'include!\s*\(\s*"([^"\\]+)"\s*\)\s*;?', node_source)
                    included = path.parent / literal[1] if literal else None
                    if included and included.is_file():
                        result["edges"].append((included.resolve(), only_test))
                    else:
                        result["issues"].append("nonliteral or missing include! is not expanded")
                for child in node.named_children:
                    if child.type in {"block", "declaration_list"}:
                        walk(child, module_dir, only_test)
            attributes = []

    module_dir = path.parent if crate_root or path.name in {"lib.rs", "main.rs", "mod.rs"} else path.with_suffix("")
    walk(tree.root_node, module_dir)
    result["test_lines"].intersection_update(range(result["lines"]))
    return result


def report(root: Path) -> dict:
    metadata = json.loads(subprocess.check_output(
        ["cargo", "metadata", "--offline", "--locked", "--no-deps", "--format-version", "1"], cwd=root))
    packages = [p for p in metadata["packages"] if p["id"] in metadata["workspace_members"]]
    files, owners, targets = {}, {}, []
    for package in packages:
        directory = Path(package["manifest_path"]).parent
        paths = set()
        for folder in ("src", "tests", "benches", "examples"):
            paths.update((directory / folder).rglob("*.rs"))
        if (directory / "build.rs").is_file():
            paths.add(directory / "build.rs")
        for target in package["targets"]:
            path = Path(target["src_path"]).resolve()
            paths.add(path)
            kind = set(target["kind"])
            role = ("external_test" if "test" in kind else "bench" if "bench" in kind else
                    "example" if "example" in kind else "build" if "custom-build" in kind else "ordinary")
            targets.append((path, role))
        for path in paths:
            path = path.resolve()
            if path in owners and owners[path] != package["name"]:
                raise ValueError(f"multiple owners: {path}")
            owners[path] = package["name"]
            files[path] = inspect(path, path in {entry[0] for entry in targets})
    roles = {path: set() for path in files}
    pending = list(targets)
    while pending:
        path, role = pending.pop()
        if path not in roles or role in roles[path]:
            continue
        roles[path].add(role)
        for child, only_test in files[path]["edges"]:
            pending.append((child, "embedded_test" if only_test and role == "ordinary" else role))
    rows = []
    for path, info in sorted(files.items()):
        reached = roles[path]
        if not reached:
            counts = {"unattributed": info["lines"]}
        elif "ordinary" in reached:
            test_lines = len(info["test_lines"])
            counts = {"ordinary": info["lines"] - test_lines, "embedded_test": test_lines}
        else:
            # Shared support keeps an explicit combined role rather than a false sole owner.
            counts = {"+".join(sorted(reached)): info["lines"]}
        issues = info["issues"] + ([] if reached else ["not reached from a declared Cargo target; unattributed"])
        rows.append({"path": str(path.relative_to(root)), "owner": owners[path],
                     "physical_lines": info["lines"], "line_classes": counts,
                     "target_roles": sorted(reached), "issues": issues, "sha256": info["sha256"]})
    totals, by_owner = {}, {}
    for row in rows:
        owner = by_owner.setdefault(row["owner"], {})
        for role, lines in row["line_classes"].items():
            totals[role] = totals.get(role, 0) + lines
            owner[role] = owner.get(role, 0) + lines
    return {"schema": 1, "root": str(root), "reporter_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "method": "physical Rust lines, conservative cfg(test) attribution",
            "limitations": ["Includes comments and blank lines; not expanded/compiler SLOC.",
                            "All non-test cfg predicates are unknown; potentially production regions stay ordinary; unreachable files stay unattributed.",
                            "Literal include! paths are followed; other macros are not expanded; unresolved/unreachable modules are flagged.",
                            "Only workspace target/source/test/bench/example/build-script Rust files are inventoried."],
            "physical_lines": sum(row["physical_lines"] for row in rows), "line_classes": totals,
            "owners": by_owner, "files": rows}


def self_test():
    assert cfg_values("all(test, feature=\"x\")") == {False}
    assert cfg_values("any(test, feature=\"x\")") == {False, True}
    assert cfg_values("not(test)") == {True}
    assert test_only(["#[cfg(all(test, feature = \"x\"))]"])
    assert not test_only(["#[cfg(any(test, feature = \"x\"))]"])
    assert not test_only(["#[cfg_attr(test, test)]"])
    import tempfile
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory).resolve()
        path = root / "lib.rs"
        path.write_text("pub fn production() {}\n#[cfg(test)]\nmod tests {\n #[test] fn check() {}\n}\n#[cfg(any(test, feature=\"x\"))]\nfn shared() {}\n#[cfg(all(test, feature=\"x\"))]\nmod external;\ninclude!(\"included.rs\");\n")
        (root / "external.rs").write_text("fn helper() {}\n")
        (root / "included.rs").write_text("fn included() {}\n")
        parsed = inspect(path, True)
        assert parsed["test_lines"] == {1, 2, 3, 4, 7, 8}
        assert set(parsed["edges"]) == {(root / "external.rs", True), (root / "included.rs", False)}
        assert not parsed["issues"]
        path.write_text("#![cfg(test)]\nfn helper() {}\n")
        assert inspect(path, True)["test_lines"] == {0, 1}
    print("cfg, nested/out-of-line module, literal include and line attribution checks passed")


if __name__ == "__main__":
    if sys.argv[1:] == ["--self-test"]:
        self_test()
    else:
        root = Path(sys.argv[1]).resolve()
        output = report(root)
        assert output["physical_lines"] == sum(output["line_classes"].values())
        Path(sys.argv[2]).write_text(json.dumps(output, indent=2) + "\n")
        print(json.dumps({key: output[key] for key in ("physical_lines", "line_classes", "owners")}))
