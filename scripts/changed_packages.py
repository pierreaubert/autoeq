#!/usr/bin/env python3
"""Affected-package selection for `just test-pr` (refactor-20261025 Phase 1).

Maps changed paths to owning workspace packages via `cargo metadata`,
then widens to reverse dependencies over normal+dev+build workspace
edges (feature edges follow the same dependency declarations). When in
doubt the selection widens: changes to shared build configuration
(workspace manifest, lockfile, Cargo config, Justfile lanes, nextest
config, partition policy) or to paths outside any package select the
whole default-member library set. This selector only bounds Cargo
compilation for a development lane; `just ntest` remains the complete
inventory.

Usage:
    python3 scripts/changed_packages.py [--base REF] [--format flags|names]

Prints `-p <package>` flags (default) or bare names, space-separated.
Prints nothing when no package-owned path changed.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

WIDEN_PATHS = {
    "Cargo.toml",
    "Cargo.lock",
    "Justfile",
    "rust-toolchain.toml",
    "scripts/crate_partition_policy.json",
    "scripts/check_crate_partition.py",
    ".config/nextest.toml",
}
WIDEN_PREFIXES = (".cargo/", "builds/")


def run(cmd: list[str]) -> str:
    return subprocess.run(
        cmd, cwd=REPO, check=True, capture_output=True, text=True
    ).stdout


def changed_paths(base: str) -> list[str]:
    committed = run(["git", "diff", "--name-only", f"{base}...HEAD"])
    staged = run(["git", "diff", "--name-only", "--cached"])
    unstaged = run(["git", "diff", "--name-only"])
    untracked = run(["git", "ls-files", "--others", "--exclude-standard"])
    seen: list[str] = []
    for block in (committed, staged, unstaged, untracked):
        for line in block.splitlines():
            if line and line not in seen:
                seen.append(line)
    return seen


def workspace_model() -> tuple[dict[str, Path], dict[str, set[str]], list[str]]:
    meta = json.loads(
        run(["cargo", "metadata", "--format-version", "1"])
    )
    member_ids = set(meta["workspace_members"])
    id_to_name = {p["id"]: p["name"] for p in meta["packages"]}
    roots: dict[str, Path] = {}
    reverse: dict[str, set[str]] = {}
    for pkg in meta["packages"]:
        if pkg["id"] not in member_ids:
            continue
        roots[pkg["name"]] = Path(pkg["manifest_path"]).parent
        reverse.setdefault(pkg["name"], set())
    for pkg in meta["packages"]:
        if pkg["id"] not in member_ids:
            continue
        for dep in pkg["dependencies"]:
            if dep["name"] in roots:
                reverse[dep["name"]].add(pkg["name"])
    defaults = sorted(
        id_to_name[i] for i in meta["workspace_default_members"]
    )
    return roots, reverse, defaults


def owner_of(path: str, roots: dict[str, Path]) -> str | None:
    absolute = REPO / path
    best: tuple[int, str] | None = None
    for name, root in roots.items():
        try:
            absolute.relative_to(root)
        except ValueError:
            continue
        depth = len(root.parts)
        if best is None or depth > best[0]:
            best = (depth, name)
    return best[1] if best else None


def reverse_closure(selected: set[str], reverse: dict[str, set[str]]) -> set[str]:
    result = set(selected)
    queue = list(selected)
    while queue:
        current = queue.pop()
        for dependent in reverse.get(current, set()):
            if dependent not in result:
                result.add(dependent)
                queue.append(dependent)
    return result


def select(paths: list[str], roots: dict[str, Path],
           reverse: dict[str, set[str]], defaults: list[str]) -> list[str]:
    selected: set[str] = set()
    for path in paths:
        if path in WIDEN_PATHS or path.startswith(WIDEN_PREFIXES):
            return list(defaults)
        owner = owner_of(path, roots)
        if owner is None:
            # Outside every package (repo-root trees such as tests/ belong
            # to the root package whose manifest dir is REPO, so owner is
            # found; a genuine miss widens rather than guessing).
            return list(defaults)
        selected.add(owner)
    if not selected:
        return []
    return sorted(reverse_closure(selected, reverse))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="main")
    parser.add_argument("--format", choices=["flags", "names"], default="flags")
    args = parser.parse_args()

    roots, reverse, defaults = workspace_model()
    selected = select(changed_paths(args.base), roots, reverse, defaults)
    if not selected:
        return 0
    if args.format == "flags":
        print(" ".join(f"-p {name}" for name in selected))
    else:
        print(" ".join(selected))
    return 0


if __name__ == "__main__":
    sys.exit(main())
