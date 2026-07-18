#!/usr/bin/env python3
"""Focused tests for the crate-partition fitness checker."""

from __future__ import annotations

import pathlib
import tempfile
import unittest

from scripts import check_crate_partition as checker


def package(name: str, dependencies: list[str]) -> dict:
    return {
        "id": f"{name} 0.1.0 (path+file:///tmp/{name})",
        "name": name,
        "manifest_path": f"/tmp/{name}/Cargo.toml",
        "dependencies": [{"name": dependency} for dependency in dependencies],
    }


def policy() -> dict:
    return {
        "root_package": "app",
        "terminal_consumers": ["app"],
        "allowed_direct_dependencies": {
            "core": [],
            "engine": ["core"],
        },
        "temporary_exceptions": [],
        "metric_budgets": {
            "root_rust_loc": 100,
            "root_binary_rust_loc": 20,
            "root_unit_tests": 3,
        },
    }


class DependencyPolicyTests(unittest.TestCase):
    def test_accepts_allowed_edges_and_terminal_consumer(self) -> None:
        packages = {
            "app": package("app", ["engine"]),
            "core": package("core", []),
            "engine": package("engine", ["core"]),
        }
        edges, errors = checker.check_dependency_policy(packages, policy())
        self.assertEqual(
            edges, {("app", "engine"), ("engine", "core")}
        )
        self.assertEqual(errors, [])

    def test_rejects_forbidden_edge(self) -> None:
        packages = {
            "app": package("app", []),
            "core": package("core", []),
            "engine": package("engine", ["app"]),
        }
        _, errors = checker.check_dependency_policy(packages, policy())
        self.assertIn(
            "workspace crate depends on root facade: engine -> app", errors
        )

    def test_requires_live_documented_exception(self) -> None:
        current_policy = policy()
        current_policy["temporary_exceptions"] = [
            {
                "from": "engine",
                "to": "core",
                "remove_by": "WP2",
                "reason": "fixture",
            },
            {
                "from": "core",
                "to": "engine",
                "remove_by": "WP2",
                "reason": "fixture",
            },
        ]
        packages = {
            "app": package("app", []),
            "core": package("core", []),
            "engine": package("engine", ["core"]),
        }
        _, errors = checker.check_dependency_policy(packages, current_policy)
        self.assertIn(
            "temporary exception is already allowed: engine -> core", errors
        )
        self.assertIn(
            "stale temporary exception must be removed: core -> engine", errors
        )

    def test_detects_cycle_even_when_edges_are_allowed(self) -> None:
        current_policy = policy()
        current_policy["allowed_direct_dependencies"]["core"] = ["engine"]
        packages = {
            "app": package("app", []),
            "core": package("core", ["engine"]),
            "engine": package("engine", ["core"]),
        }
        _, errors = checker.check_dependency_policy(packages, current_policy)
        self.assertTrue(
            any(error.startswith("workspace dependency cycle:") for error in errors)
        )


class RatchetTests(unittest.TestCase):
    def test_exception_and_budget_lists_only_shrink(self) -> None:
        baseline = policy()
        current = policy()
        current["temporary_exceptions"] = [
            {
                "from": "engine",
                "to": "app",
                "remove_by": "WP2",
                "reason": "fixture",
            }
        ]
        current["metric_budgets"]["root_rust_loc"] = 101
        errors = checker.check_monotonic_ratchets(current, baseline)
        self.assertIn(
            "temporary exception list may only shrink: added engine -> app", errors
        )
        self.assertIn(
            "metric budget may only shrink: root_rust_loc 100 -> 101", errors
        )

    def test_metric_values_cannot_exceed_budgets(self) -> None:
        metrics = {
            "root_rust_loc": 101,
            "root_binary_rust_loc": 20,
            "root_unit_tests": 3,
        }
        self.assertEqual(
            checker.check_metric_budgets(metrics, policy()),
            ["metric increased: root_rust_loc=101, budget=100"],
        )


class OwnershipTests(unittest.TestCase):
    def test_detects_whitespace_only_cross_owner_copy(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = pathlib.Path(temporary_directory)
            root_source = root / "src" / "feature.rs"
            crate_source = root / "crates" / "feature" / "src" / "feature.rs"
            root_source.parent.mkdir(parents=True)
            crate_source.parent.mkdir(parents=True)
            statement = "pub fn copied() -> usize { 42 }\n" * 12
            root_source.write_text(statement, encoding="utf-8")
            crate_source.write_text(statement.replace(" ", "  "), encoding="utf-8")
            duplicates = checker.duplicate_source_ownership(root)
            self.assertEqual(len(duplicates), 1)
            self.assertEqual(set(duplicates[0]), {root_source, crate_source})

    def test_extracts_multiline_facade_declarations(self) -> None:
        source = """
#[macro_export]
macro_rules! visible { () => {}; }
pub mod model;
pub use model::{
    Config,
    Result,
};
#[cfg(test)]
mod tests;
"""
        self.assertEqual(
            checker.extract_root_public_api(source),
            [
                "macro visible",
                "pub mod model;",
                "pub use model::{ Config, Result, };",
            ],
        )


if __name__ == "__main__":
    unittest.main()
