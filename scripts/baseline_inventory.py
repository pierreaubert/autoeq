#!/usr/bin/env python3
"""Phase 0 baseline inventory for refactor-20261025.

Records the declared workspace shape (packages, targets by kind, features,
workspace-internal dependency edges), the QA oracle/man registry identities,
Python test files, and production/test LOC split. Everything is derived from
`cargo metadata` and checked-in manifests: no compilation is performed, so
this inventory describes declared targets, not built artifacts. Compiled
test-identity listings (nextest JSON) are a Phase 2 deliverable.

Usage:
    python3 scripts/baseline_inventory.py [--output-dir DIR]

Writes `inventory.json` and `inventory.md` under
`docs/baselines/refactor-20261025/` by default.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tomllib
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = REPO / "docs" / "baselines" / "refactor-20261025"


def run(cmd: list[str], cwd: Path = REPO) -> str:
    return subprocess.run(
        cmd, cwd=cwd, check=True, capture_output=True, text=True
    ).stdout.strip()


def git_rev(path: Path) -> str | None:
    try:
        return run(["git", "rev-parse", "HEAD"], cwd=path)
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def tool_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name, cmd in [
        ("rustc", ["rustc", "--version"]),
        ("cargo", ["cargo", "--version"]),
        ("cargo-nextest", ["cargo", "nextest", "--version"]),
        ("just", ["just", "--version"]),
    ]:
        try:
            versions[name] = run(cmd).splitlines()[0]
        except (subprocess.CalledProcessError, FileNotFoundError, IndexError):
            versions[name] = None
    return versions


def count_loc(root: Path) -> int:
    total = 0
    for path in root.rglob("*.rs"):
        total += len(path.read_text(errors="replace").splitlines())
    return total


def crate_loc() -> dict[str, dict[str, int]]:
    """Production (src/) vs test-tree (tests/) LOC per package directory.

    Inline `#[cfg(test)]` modules stay in the production count; this is a
    declared-tree split, recorded as such in the output.
    """
    result: dict[str, dict[str, int]] = {}
    packages = [("autoeq (root)", REPO)]
    packages += sorted(
        (p.name, p) for p in (REPO / "crates").iterdir() if p.is_dir()
    )
    for name, root in packages:
        entry = {
            "production_loc": count_loc(root / "src") if (root / "src").is_dir() else 0,
            "test_tree_loc": count_loc(root / "tests") if (root / "tests").is_dir() else 0,
        }
        result[name] = entry
    return result


def qa_manifest() -> dict[str, object]:
    path = REPO / "crates" / "autoeq-qa" / "validation-manifest.toml"
    data = tomllib.loads(path.read_text())
    cases = data.get("case", [])
    return {
        "path": str(path.relative_to(REPO)),
        "case_count": len(cases),
        "case_ids": [c["id"] for c in cases],
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def roomeq_registry() -> dict[str, object]:
    path = REPO / "crates" / "roomeq-qa" / "src" / "registry.json"
    data = json.loads(path.read_text())
    sections: dict[str, object] = {
        key: len(value) if isinstance(value, (list, dict)) else value
        for key, value in data.items()
    }
    return {
        "path": str(path.relative_to(REPO)),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "sections": sections,
    }


def python_tests() -> list[str]:
    found: list[str] = []
    for pattern_root in [REPO / "scripts", REPO / "tests", REPO / "ui"]:
        if pattern_root.is_dir():
            found += [
                str(p.relative_to(REPO))
                for p in pattern_root.rglob("test_*.py")
            ]
    return sorted(found)


def cargo_inventory() -> dict[str, object]:
    raw = run(
        ["cargo", "metadata", "--locked", "--format-version", "1"]
    )
    meta = json.loads(raw)
    member_ids = set(meta["workspace_members"])
    workspace_names = {
        p["name"] for p in meta["packages"] if p["id"] in member_ids
    }
    packages: dict[str, object] = {}
    totals: dict[str, int] = {}
    for pkg in sorted(meta["packages"], key=lambda p: p["name"]):
        if pkg["id"] not in member_ids:
            continue
        targets: dict[str, list[str]] = {}
        for target in pkg["targets"]:
            for kind in target["kind"]:
                targets.setdefault(kind, []).append(target["name"])
                totals[kind] = totals.get(kind, 0) + 1
        deps = {
            kind: sorted(
                d["name"]
                for d in pkg["dependencies"]
                if d["name"] in workspace_names and (d["kind"] or "normal") == kind
            )
            for kind in ("normal", "dev", "build")
        }
        packages[pkg["name"]] = {
            "version": pkg["version"],
            "manifest": str(Path(pkg["manifest_path"]).relative_to(REPO)),
            "features": sorted(pkg["features"]),
            "targets": {k: sorted(v) for k, v in sorted(targets.items())},
            "workspace_dependencies": deps,
        }
    return {
        "workspace_members": sorted(workspace_names),
        "resolved_package_count": len(meta["packages"]),
        "target_kind_totals": dict(sorted(totals.items())),
        "packages": packages,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    inventory: dict[str, object] = {
        "schema": "refactor-20261025/baseline-inventory/v1",
        "revisions": {
            "autoeq": git_rev(REPO),
            "math-audio": git_rev(REPO.parent / "math-audio"),
        },
        "lockfile_sha256": hashlib.sha256(
            (REPO / "Cargo.lock").read_bytes()
        ).hexdigest(),
        "tools": tool_versions(),
        "cargo": cargo_inventory(),
        "qa_manifest": qa_manifest(),
        "roomeq_registry": roomeq_registry(),
        "python_test_files": python_tests(),
        "loc": crate_loc(),
        "notes": [
            "Declared-target inventory from cargo metadata; no compilation.",
            "Compiled nextest test-identity listing is deferred to Phase 2.",
            "LOC split counts tests/ trees as test LOC; inline #[cfg(test)] "
            "modules remain in production LOC.",
        ],
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "inventory.json").write_text(
        json.dumps(inventory, indent=2, sort_keys=True) + "\n"
    )
    cargo = inventory["cargo"]  # type: ignore[assignment]
    lines = [
        "# Baseline inventory — refactor-20261025 (Phase 0)",
        "",
        f"- autoeq revision: `{inventory['revisions']['autoeq']}`",  # type: ignore[index]
        f"- math-audio revision: `{inventory['revisions']['math-audio']}`",  # type: ignore[index]
        f"- workspace members: {len(cargo['workspace_members'])}",  # type: ignore[index]
        f"- resolved packages: {cargo['resolved_package_count']}",  # type: ignore[index]
        f"- QA oracle cases: {inventory['qa_manifest']['case_count']}",  # type: ignore[index]
        f"- Python test files: {len(inventory['python_test_files'])}",  # type: ignore[arg-type]
        "",
        "Target-kind totals (declared, from `cargo metadata`):",
        "",
    ]
    for kind, count in cargo["target_kind_totals"].items():  # type: ignore[index]
        lines.append(f"- `{kind}`: {count}")
    lines += ["", "Full detail: [inventory.json](inventory.json).", ""]
    (args.output_dir / "inventory.md").write_text("\n".join(lines))
    print(f"wrote {args.output_dir / 'inventory.json'}")
    print(f"wrote {args.output_dir / 'inventory.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
