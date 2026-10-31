#!/usr/bin/env python3
"""Focused tests for the crate-partition fitness checker."""

from __future__ import annotations

import pathlib
import tempfile
import unittest

from scripts import check_crate_partition as checker
from scripts import run_crate_test_matrix as matrix_runner


def package(name: str, dependencies: list[str | tuple[str, str]]) -> dict:
    return {
        "id": f"{name} 0.1.0 (path+file:///tmp/{name})",
        "name": name,
        "manifest_path": f"/tmp/{name}/Cargo.toml",
        "dependencies": [
            {
                "name": dependency[0] if isinstance(dependency, tuple) else dependency,
                "kind": dependency[1] if isinstance(dependency, tuple) else None,
            }
            for dependency in dependencies
        ],
    }


def policy() -> dict:
    return {
        "root_package": "app",
        "terminal_consumers": ["app"],
        "allowed_direct_dependencies": {
            "core": [],
            "engine": ["core"],
        },
        "allowed_dev_dependencies": {},
        "allowed_build_dependencies": {},
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
            edges,
            {("app", "engine", "normal"), ("engine", "core", "normal")},
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

    def test_accepts_refactor_phase_remove_by(self) -> None:
        current_policy = policy()
        current_policy["allowed_direct_dependencies"]["engine"] = []
        current_policy["temporary_exceptions"] = [
            {
                "from": "engine",
                "to": "core",
                "remove_by": "phase-3d",
                "reason": "scheduled dependency cut",
            }
        ]
        packages = {
            "app": package("app", []),
            "core": package("core", []),
            "engine": package("engine", ["core"]),
        }
        _, errors = checker.check_dependency_policy(packages, current_policy)
        self.assertEqual(errors, [])

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
            any(
                error.startswith("workspace dependency cycle (normal/build):")
                for error in errors
            )
        )

    def test_build_dependency_cycles_are_rejected(self) -> None:
        current_policy = policy()
        current_policy["allowed_direct_dependencies"]["core"] = ["engine"]
        current_policy["allowed_build_dependencies"]["engine"] = ["core"]
        packages = {
            "app": package("app", []),
            "core": package("core", [("engine", "normal")]),
            "engine": package("engine", [("core", "build")]),
        }

        _, errors = checker.check_dependency_policy(packages, current_policy)

        self.assertTrue(
            any(
                error.startswith("workspace dependency cycle (normal/build):")
                for error in errors
            )
        )

    def test_dev_only_cycle_is_not_a_runtime_cycle(self) -> None:
        current_policy = policy()
        current_policy["allowed_direct_dependencies"]["core"] = ["engine"]
        current_policy["allowed_dev_dependencies"]["engine"] = ["core"]
        packages = {
            "app": package("app", []),
            "core": package("core", [("engine", "normal")]),
            "engine": package("engine", [("core", "dev")]),
        }

        edges, errors = checker.check_dependency_policy(packages, current_policy)

        self.assertEqual(
            edges,
            {("core", "engine", "normal"), ("engine", "core", "dev")},
        )
        self.assertEqual(errors, [])

    def test_unlisted_dev_edge_is_rejected_by_kind(self) -> None:
        packages = {
            "app": package("app", []),
            "core": package("core", []),
            "engine": package("engine", [("core", "dev")]),
        }

        _, errors = checker.check_dependency_policy(packages, policy())

        self.assertIn("forbidden dev workspace edge: engine -> core", errors)

    def test_required_external_edges_are_checked(self) -> None:
        current_policy = policy()
        current_policy["required_external_dependencies"] = {
            "engine": {"normal": ["sha2", "tempfile"]}
        }
        packages = {
            "app": package("app", []),
            "core": package("core", []),
            "engine": package("engine", [("core", "normal"), "sha2"]),
        }

        _, errors = checker.check_dependency_policy(packages, current_policy)

        self.assertIn(
            "required external dependency is missing: engine -> tempfile (normal)",
            errors,
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
        errors = checker.check_monotonic_ratchets(
            current, baseline, {"root_rust_loc": 100}
        )
        self.assertIn(
            "temporary exception list may only shrink: added engine -> app", errors
        )
        self.assertIn(
            "metric budget increase exceeds measured base source: root_rust_loc 100 -> 101, base source 100",
            errors,
        )

    def test_allowed_edge_reclassified_as_exception_is_a_tightening(self) -> None:
        baseline = policy()
        current = policy()
        # engine -> core was allowed in the baseline; demoting it to a
        # temporary exception with a removal phase must not fail the
        # ratchet, while an exception for a forbidden edge still does.
        current["allowed_direct_dependencies"]["engine"] = []
        current["temporary_exceptions"] = [
            {
                "from": "engine",
                "to": "core",
                "remove_by": "phase-3c",
                "reason": "scheduled dependency cut",
            }
        ]
        self.assertEqual(
            checker.check_monotonic_ratchets(
                current, baseline, {"root_rust_loc": 100}
            ),
            [],
        )
        current["temporary_exceptions"].append(
            {
                "from": "core",
                "to": "app",
                "remove_by": "phase-3c",
                "reason": "previously forbidden",
            }
        )
        self.assertIn(
            "temporary exception list may only shrink: added core -> app",
            checker.check_monotonic_ratchets(
                current, baseline, {"root_rust_loc": 100}
            ),
        )

    def test_budget_increase_cannot_exceed_actual_baseline_source(self) -> None:
        baseline = policy()
        baseline["metric_budgets"]["root_rust_loc"] = 100
        current = policy()
        current["metric_budgets"]["root_rust_loc"] = 103
        self.assertEqual(
            checker.check_monotonic_ratchets(
                current, baseline, {"root_rust_loc": 103}
            ),
            [],
        )

        current["metric_budgets"]["root_rust_loc"] = 104
        self.assertIn(
            "metric budget increase exceeds measured base source: root_rust_loc 100 -> 104, base source 103",
            checker.check_monotonic_ratchets(
                current, baseline, {"root_rust_loc": 103}
            ),
        )

    def test_rejects_duplicate_policy_json_keys(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = pathlib.Path(temporary_directory) / "duplicate.json"
            path.write_text(
                '{"roomeq-qa": [], "roomeq-qa": ["core"]}',
                encoding="utf-8",
            )

            with self.assertRaisesRegex(
                ValueError, "duplicate JSON object key: roomeq-qa"
            ):
                checker.load_json(path)

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


class BuildLaneTests(unittest.TestCase):
    def lanes_policy(self) -> dict:
        current = policy()
        current["build_lanes"] = {
            "test": {
                "selection": "default-members",
                "packages": ["core", "engine"],
                "cargo_profile": "test",
                "nextest_profile": "fast",
            },
            "ntest": {
                "selection": "workspace",
                "cargo_profile": "qa",
                "nextest_profile": "full",
            },
        }
        return current

    def test_accepts_lanes_matching_default_members(self) -> None:
        self.assertEqual(
            checker.check_build_lanes(
                {"app", "core", "engine"},
                ["core", "engine"],
                self.lanes_policy(),
            ),
            [],
        )

    def test_rejects_default_member_drift_and_absent_package(self) -> None:
        current = self.lanes_policy()
        current["build_lanes"]["test"]["packages"] = ["core", "renamed"]
        errors = checker.check_build_lanes(
            {"app", "core", "engine"}, ["core", "engine"], current
        )
        self.assertIn(
            "build lane test names absent package: renamed", errors
        )
        self.assertIn(
            "build lane test packages do not match workspace default-members",
            errors,
        )

    def test_missing_build_lanes_is_a_bootstrap_noop(self) -> None:
        self.assertEqual(
            checker.check_build_lanes({"app"}, [], policy()), []
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


class FocusedTestMatrixTests(unittest.TestCase):
    def test_lists_and_builds_policy_commands(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            policy_path = pathlib.Path(temporary_directory) / "policy.json"
            policy_path.write_text(
                '{"focused_tests":{"core":"cargo test -p core --lib"}}',
                encoding="utf-8",
            )

            tests = matrix_runner.focused_tests(policy_path)

        self.assertEqual(tests, {"core": "cargo test -p core --lib"})
        self.assertEqual(
            matrix_runner.command_for_package(tests, "core", release=True),
            ["cargo", "test", "-p", "core", "--lib", "--release"],
        )

    def test_rejects_unknown_package(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown focused-test package"):
            matrix_runner.command_for_package({}, "missing", release=False)

    def test_rejects_nonexistent_feature_in_focused_recipe(self) -> None:
        package_data = {
            "core": {
                "features": {"default": []},
                "targets": [{"kind": ["lib"]}],
            }
        }
        self.assertIn(
            "focused test command for core names unknown feature plotly",
            checker.check_focused_tests(
                package_data,
                {
                    "focused_tests": {
                        "core": "cargo test -p core --features plotly"
                    }
                },
            ),
        )

    def test_recognizes_rlib_output_as_a_library_target(self) -> None:
        package_data = {
            "gpui": {
                "features": {"default": []},
                "targets": [
                    {"kind": ["cdylib", "rlib"], "crate_types": ["cdylib", "rlib"]}
                ],
            }
        }

        errors = checker.check_focused_tests(
            package_data,
            {"focused_tests": {"gpui": "cargo test -p gpui --lib"}},
        )

        self.assertEqual(errors, [])


class VirtualWorkspaceTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = pathlib.Path(self.directory.name)
        (self.root / "Cargo.toml").write_text('[workspace]\nmembers = []\n')
        self.packages = {"core": package("core", []), "cli": package("cli", ["core"])}
        self.packages["cli"]["publish"] = []
        self.policy = {"virtual_workspace": True, "root_package": "autoeq", "terminal_packages": ["cli"]}

    def check(self):
        return checker.check_virtual_workspace(self.root, self.packages, self.policy)

    def test_accepts_virtual_workspace_with_terminal_ownership(self):
        self.assertEqual(self.check(), [])

    def test_rejects_root_package_resurrection(self):
        (self.root / "Cargo.toml").write_text('[workspace]\n[package]\nname = "autoeq"\n')
        self.assertTrue(any("retired root package" in error for error in self.check()))

    def test_rejects_root_source_resurrection(self):
        (self.root / "src").mkdir()
        (self.root / "src/lib.rs").write_text('pub use core::*;')
        self.assertTrue(any("facade sources" in error for error in self.check()))

    def test_rejects_production_to_terminal_edge(self):
        self.packages["core"] = package("core", ["cli"])
        self.assertTrue(any("production package depends on terminal" in error for error in self.check()))

    def test_rejects_publishable_terminal_package(self):
        self.packages["cli"]["publish"] = None
        self.assertTrue(any("publish = false" in error for error in self.check()))

    def test_rejects_new_dev_cycle(self):
        current = policy()
        current["virtual_workspace"] = True
        current["allowed_dev_dependencies"] = {"core": ["engine"]}
        packages = {"app": package("app", []), "core": package("core", [("engine", "dev")]), "engine": package("engine", ["core"])}
        errors = checker.check_dependency_policy(packages, current)[1]
        self.assertTrue(any("normal/build/dev" in error for error in errors))


class DependencyClosureTests(unittest.TestCase):
    def setUp(self):
        self.metadata = {
            "packages": [{"id": name, "name": name} for name in ("objective", "helper", "backend")],
            "resolve": {"nodes": [
                {"id": "objective", "deps": [{"pkg": "helper", "dep_kinds": [{"kind": None}]}]},
                {"id": "helper", "deps": [{"pkg": "backend", "dep_kinds": [{"kind": None}]}]},
                {"id": "backend", "deps": []},
            ]},
        }
        self.policy = {"forbidden_dependency_closures": {"objective": ["backend"]}}

    def test_rejects_transitive_backend_dependency(self):
        self.assertIn("forbidden dependency closure: objective -> backend",
                      checker.check_forbidden_dependency_closures(self.metadata, self.policy))

    def test_dev_only_dependency_is_not_a_production_closure(self):
        self.metadata["resolve"]["nodes"][1]["deps"][0]["dep_kinds"] = [{"kind": "dev"}]
        self.assertEqual(checker.check_forbidden_dependency_closures(self.metadata, self.policy), [])

    def test_missing_graph_cannot_pass(self):
        self.metadata["resolve"] = None
        self.assertTrue(checker.check_forbidden_dependency_closures(self.metadata, self.policy))

    def test_missing_owner_node_cannot_pass(self):
        self.metadata["resolve"]["nodes"].pop(0)
        self.assertTrue(checker.check_forbidden_dependency_closures(self.metadata, self.policy))



class ExplicitLibraryExportTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.source = pathlib.Path(self.temporary.name) / "lib.rs"
        self.packages = {"owner": {"targets": [{"kind": ["rlib", "cdylib"], "src_path": str(self.source)}]}}

    def test_multiline_and_grouped_wildcards_fail(self):
        for statement in ("pub use source::*;", "pub use source::{\n A, *\n};"):
            self.source.write_text(statement)
            self.assertTrue(checker.check_explicit_library_exports(self.packages, {"explicit_library_exports": True}))

    def test_explicit_names_and_doc_examples_pass(self):
        self.source.write_text("//! pub use source::*;\npub use source::{A, B};\n")
        self.assertEqual(checker.check_explicit_library_exports(self.packages, {"explicit_library_exports": True}), [])

    def test_legacy_policy_does_not_apply_new_ratchet(self):
        self.source.write_text("pub use source::*;\n")
        self.assertEqual(checker.check_explicit_library_exports(self.packages, {}), [])


class TerminalDependencyTests(unittest.TestCase):
    def check(self, kind):
        current = policy()
        current["forbidden_direct_external_dependencies"] = {"engine": ["clap"]}
        packages = {"app": package("app", []), "core": package("core", []),
                    "engine": package("engine", ["core", ("clap", kind)])}
        return checker.check_dependency_policy(packages, current)[1]

    def test_normal_parser_dependency_is_rejected(self):
        self.assertTrue(any("forbidden direct normal" in error for error in self.check("normal")))

    def test_build_parser_dependency_is_rejected(self):
        self.assertTrue(any("forbidden direct build" in error for error in self.check("build")))

    def test_example_and_test_parser_dependency_is_allowed(self):
        self.assertEqual(self.check("dev"), [])

if __name__ == "__main__":
    unittest.main()
