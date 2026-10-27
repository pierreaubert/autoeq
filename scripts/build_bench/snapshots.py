"""Create independent source trees while preserving exact revision identities."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

EXCLUDED = {".git", ".worktrees", ".tokensave", "target", "target-static", "venv", ".venv", "node_modules", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache", ".DS_Store", ".aws", ".codex", ".agents", ".claude"}


RUNTIME_ROOTS = (Path("data_tests"), Path("crates/autoeq-report-wasm/dist"))
RUNTIME_EXCLUDED = EXCLUDED - {"target", "target-static"}


def capture(command: list[str], cwd: Path | None = None) -> str:
    return subprocess.run(command, cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()



def excluded_path(root: Path, relative: Path) -> bool:
    """Exclude caches and recognized output directories, preserving authored names."""
    if relative.name == ".env":
        return True
    runtime = any(relative.is_relative_to(directory) for directory in RUNTIME_ROOTS)
    excluded = RUNTIME_EXCLUDED if runtime else EXCLUDED
    if any(part in excluded for part in relative.parts):
        return True
    if runtime:
        return False
    parts = relative.parts
    # Only the harness's report directories are outputs; bench-prefixed source
    # files and fixture directories remain source inputs.
    if (len(parts) >= 4 and parts[:2] == ("docs", "baselines")
            and parts[3].startswith("bench-")
            and (len(parts) > 4 or (root / relative).is_dir())):
        return True
    for ancestor in (relative, *relative.parents):
        path = root / ancestor
        if (ancestor.name.startswith("target-") and path.is_dir()
                and ((path / ".rustc_info.json").is_file() or (path / "CACHEDIR.TAG").is_file())):
            return True
    return False


def identity(root: Path) -> dict:
    patch = subprocess.check_output(["git", "diff", "--binary", "HEAD"], cwd=root)
    untracked = capture(["git", "ls-files", "--others", "--exclude-standard"], root)
    return {
        "revision": capture(["git", "rev-parse", "HEAD"], root),
        "tracked_patch_sha256": hashlib.sha256(patch).hexdigest(),
        "untracked_source_sha256": {
            name: digest(root / name) for name in untracked.splitlines()
            if (root / name).is_file() and not excluded_path(root, Path(name))
        },
        "lockfile_sha256": digest(root / "Cargo.lock") if (root / "Cargo.lock").is_file() else None,
        "runtime_inputs_sha256": {
            str(path.relative_to(root)): digest(path)
            for directory in (root / name for name in RUNTIME_ROOTS)
            if directory.is_dir() for path in sorted(directory.rglob("*"))
            if path.is_file() and not excluded_path(root, path.relative_to(root))
        },
    }


def ignored(root: Path):
    def ignore(directory: str, names: list[str]) -> set[str]:
        relative = Path(directory).relative_to(root)
        return {name for name in names if excluded_path(root, relative / name)}
    return ignore


def copy_repository(source: Path, destination: Path, reference: str | None = None) -> dict:
    """Clone metadata, then materialize the requested source without hard links."""
    source = source.resolve()
    before = identity(source)
    subprocess.run(["git", "clone", "--quiet", "--shared", "--no-checkout", str(source), str(destination)], check=True)
    revision = capture(["git", "rev-parse", reference or "HEAD"], source)
    if reference:
        subprocess.run(["git", "checkout", "--quiet", "--detach", revision], cwd=destination, check=True)
        # Runtime inputs ignored by Git remain inputs, never shared writable trees.
        extra = capture(["git", "ls-files", "--others", "--ignored", "--exclude-standard", "--", "data_tests", "crates/autoeq-report-wasm/dist"], source)
        for name in extra.splitlines():
            original = source / name
            if original.is_file() and not excluded_path(source, Path(name)):
                copied = destination / name
                copied.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(original, copied)
    else:
        # An independent index reproduces the original dirty state. No checkout
        # populates files deleted by the source's uncommitted patch.
        subprocess.run(["git", "read-tree", revision], cwd=destination, check=True)
        listed = subprocess.check_output(["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"], cwd=source)
        for encoded in listed.split(b"\0"):
            if not encoded:
                continue
            relative = Path(os.fsdecode(encoded))
            if excluded_path(source, relative):
                continue
            original = source / relative
            copied = destination / relative
            if original.is_dir():
                shutil.copytree(original, copied, dirs_exist_ok=True, symlinks=False, ignore=ignored(source))
            elif original.is_file():
                copied.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(original, copied)
        for relative in RUNTIME_ROOTS:
            original = source / relative
            if original.is_dir():
                shutil.copytree(original, destination / relative, dirs_exist_ok=True, symlinks=False, ignore=ignored(source))
    after = identity(source)
    if after != before:
        raise RuntimeError(f"source changed while snapshotting: {source}")
    return {"origin": str(source), "snapshot": str(destination), "requested_reference": reference, "origin_identity": before, "snapshot_identity": identity(destination)}


@dataclass(frozen=True)
class Toolchain:
    cargo: Path
    rustc: Path
    rustdoc: Path
    description: str
    host: str
    hashes: dict[str, str]

    @classmethod
    def resolve(cls, requested: str | None = None) -> Toolchain:
        flags = ["--toolchain", requested] if requested else []
        paths = {name: Path(capture(["rustup", "which", *flags, name])).resolve() for name in ("cargo", "rustc", "rustdoc")}
        description = capture([str(paths["rustc"]), "--version", "--verbose"])
        host = next(line.split(": ", 1)[1] for line in description.splitlines() if line.startswith("host: "))
        return cls(**paths, description=description, host=host, hashes={name: digest(path) for name, path in paths.items()})

    def environment(self) -> dict[str, str]:
        self.verify()
        env = dict(os.environ)
        env.update(CARGO=str(self.cargo), RUSTC=str(self.rustc), RUSTDOC=str(self.rustdoc), CARGO_NET_OFFLINE="true", PYTHONDONTWRITEBYTECODE="1")
        env.pop("CARGO_TARGET_DIR", None)
        return env

    def verify(self) -> None:
        for name in ("cargo", "rustc", "rustdoc"):
            if digest(getattr(self, name)) != self.hashes[name]:
                raise RuntimeError(f"pinned toolchain binary changed during measurement: {name}")

    def report(self) -> dict:
        return {"description": self.description, "host": self.host, "paths": {name: str(getattr(self, name)) for name in self.hashes}, "sha256": self.hashes}


class Storage:
    """Own the only directory tree this harness is allowed to remove."""
    def __init__(self, parent: Path, sources: list[Path]):
        parent = parent.resolve()
        if any(parent == source.resolve() or parent.is_relative_to(source.resolve()) for source in sources):
            raise ValueError("benchmark scratch must be outside source repositories")
        parent.mkdir(parents=True, exist_ok=True)
        self.root = Path(tempfile.mkdtemp(prefix="autoeq-build-bench-", dir=parent))
        self.marker = self.root / ".autoeq-build-bench-owned"
        self.marker.write_text(str(self.root))

    def remove(self, path: Path) -> None:
        resolved = path.resolve()
        if not self.marker.is_file() or resolved == self.root or not resolved.is_relative_to(self.root):
            raise ValueError(f"refusing to remove unowned benchmark path: {path}")
        if path.exists():
            shutil.rmtree(path)

    def close(self) -> None:
        if self.marker.is_file() and self.marker.read_text() == str(self.root):
            shutil.rmtree(self.root)


def metadata(root: Path, toolchain: Toolchain) -> dict:
    return json.loads(subprocess.run([str(toolchain.cargo), "metadata", "--offline", "--locked", "--format-version", "1"], cwd=root, env=toolchain.environment(), check=True, capture_output=True, text=True).stdout)


def dependency_roots(source: Path, toolchain: Toolchain) -> set[Path]:
    original = metadata(source, toolchain)
    siblings = set()
    for package in original["packages"]:
        if package["source"] is not None:
            continue
        manifest = Path(package["manifest_path"]).resolve()
        if manifest.is_relative_to(source.resolve()):
            continue
        sibling = Path(capture(["git", "rev-parse", "--show-toplevel"], manifest.parent)).resolve()
        siblings.add(sibling)
    return siblings


def snapshot(source: Path, destination: Path, toolchain: Toolchain, reference: str | None = None) -> tuple[Path, dict]:
    destination.mkdir(parents=True)
    root = destination / "autoeq"
    report = {"autoeq": copy_repository(source, root, reference)}
    siblings = dependency_roots(source, toolchain)
    for sibling in sorted(siblings):
        report[sibling.name] = copy_repository(sibling, destination / sibling.name)
    # Metadata is also a proof that every relative path resolves in the copy.
    copied_metadata = metadata(root, toolchain)
    for package in copied_metadata["packages"]:
        if package["source"] is None and not Path(package["manifest_path"]).resolve().is_relative_to(destination.resolve()):
            raise RuntimeError(f"path dependency escaped the source snapshot: {package['manifest_path']}")
    return root, report
