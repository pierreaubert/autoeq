"""Unit tests for scripts/changed_packages.py selection logic."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import changed_packages as cp  # noqa: E402


ROOTS = {
    "autoeq": cp.REPO,
    "autoeq-core": cp.REPO / "crates" / "autoeq-core",
    "roomeq-engine": cp.REPO / "crates" / "roomeq-engine",
}
REVERSE = {
    "autoeq-core": {"roomeq-engine", "autoeq"},
    "roomeq-engine": {"autoeq"},
    "autoeq": set(),
}
DEFAULTS = ["autoeq-core", "roomeq-engine"]


class SelectTests(unittest.TestCase):
    def test_owned_path_widens_to_reverse_dependencies(self) -> None:
        selected = cp.select(
            ["crates/autoeq-core/src/lib.rs"], ROOTS, REVERSE, DEFAULTS
        )
        self.assertEqual(selected, ["autoeq", "autoeq-core", "roomeq-engine"])

    def test_shared_config_widens_to_default_members(self) -> None:
        for path in ["Cargo.toml", ".cargo/config.toml", "builds/qa/qa-roomeq.just"]:
            self.assertEqual(
                cp.select([path], ROOTS, REVERSE, DEFAULTS), DEFAULTS, path
            )

    def test_no_changes_selects_nothing(self) -> None:
        self.assertEqual(cp.select([], ROOTS, REVERSE, DEFAULTS), [])

    def test_root_package_owns_root_tests(self) -> None:
        selected = cp.select(["tests/some_test.rs"], ROOTS, REVERSE, DEFAULTS)
        self.assertEqual(selected, ["autoeq"])


if __name__ == "__main__":
    unittest.main()
