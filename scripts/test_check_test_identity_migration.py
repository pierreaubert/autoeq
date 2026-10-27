"""Failure controls for the test relocation identity gate."""
import copy
import unittest
from scripts.check_test_identity_migration import compare, compare_full_inventory


class IdentityMigrationTests(unittest.TestCase):
    def setUp(self):
        self.before = {"rust-suites": {"autoeq::target": {
            "binary-name": "target", "testcases": {
                "regular": {"kind": "test", "ignored": False},
                "opt_in": {"kind": "test", "ignored": True},
            },
        }}}
        self.after = {"rust-suites": {"owner::target": copy.deepcopy(
            self.before["rust-suites"]["autoeq::target"])}}

    def check(self):
        return compare(self.before, self.after, {"target": "owner"})[0]

    def test_exact_relocation_passes(self):
        self.assertEqual(self.check(), [])

    def test_dropped_ignored_case_fails(self):
        del self.after["rust-suites"]["owner::target"]["testcases"]["opt_in"]
        self.assertTrue(any("missing test" in error for error in self.check()))

    def test_changed_ignore_state_fails(self):
        self.after["rust-suites"]["owner::target"]["testcases"]["regular"]["ignored"] = True
        self.assertTrue(any("changed ignored" in error for error in self.check()))

    def test_duplicate_old_target_fails(self):
        self.after["rust-suites"].update(self.before["rust-suites"])
        self.assertTrue(any("old target still compiled" in error for error in self.check()))

    def test_extra_case_fails(self):
        self.after["rust-suites"]["owner::target"]["testcases"]["extra"] = {"kind": "test", "ignored": False}
        self.assertTrue(any("unmapped added test" in error for error in self.check()))


class PackagingRetirementTests(unittest.TestCase):
    def setUp(self):
        self.before = {"rust-suites": {"autoeq::packaging_excludes": {
            "package-name": "autoeq", "binary-name": "packaging_excludes",
            "testcases": {"published_package_excludes_qa_scripts_backups_and_mdat": {"kind": "test", "ignored": False}},
        }}}
        self.owners = {"packaging_excludes": {"retired": "Root package removed"}}

    def test_retirement_requires_root_package_absence(self):
        errors, _ = compare(self.before, self.before, self.owners)
        self.assertTrue(any("cannot retire" in error for error in errors))

    def test_only_obsolete_packaging_contract_can_retire(self):
        self.before["rust-suites"]["autoeq::packaging_excludes"]["binary-name"] = "substantive_test"
        errors, _ = compare(self.before, {"rust-suites": {}}, {"substantive_test": {"retired": "excuse"}})
        self.assertTrue(any("unsupported test retirement" in error for error in errors))

    def test_documents_retired_contract_when_root_is_removed(self):
        errors, rows = compare(self.before, {"rust-suites": {}}, self.owners)
        self.assertEqual(errors, [])
        self.assertEqual(len(rows), 1)
        self.assertTrue(rows[0][2].startswith("<retired"))


class FullInventoryMigrationTests(unittest.TestCase):
    def setUp(self):
        case = {"kind": "test", "ignored": False}
        self.before = {"rust-suites": {
            "backend": {"testcases": {"cli::parse": dict(case), "solver": dict(case)}},
            "terminal": {"testcases": {"command": dict(case)}},
        }}
        self.after = copy.deepcopy(self.before)
        moved = self.after["rust-suites"]["backend"]["testcases"].pop("cli::parse")
        self.after["rust-suites"]["terminal"]["testcases"].update({"cli::parse": moved, "defaults": dict(case)})
        self.manifest = {"relocations": [{"from_suite": "backend", "name": "cli::parse", "to_suite": "terminal"}],
                         "additions": [{"suite": "terminal", "name": "defaults"}]}

    def check(self):
        return compare_full_inventory(self.before, self.after, self.manifest)[0]

    def test_complete_inventory_preserves_moved_and_unmoved_cases(self):
        self.assertEqual(self.check(), [])

    def test_unrelated_solver_cannot_disappear(self):
        del self.after["rust-suites"]["backend"]["testcases"]["solver"]
        self.assertTrue(any("missing test" in error for error in self.check()))

    def test_retaining_duplicate_parser_is_rejected(self):
        self.after["rust-suites"]["backend"]["testcases"]["cli::parse"] = {"kind": "test", "ignored": False}
        self.assertTrue(any("unmapped added" in error for error in self.check()))

    def test_ignore_state_cannot_change_during_move(self):
        self.after["rust-suites"]["terminal"]["testcases"]["cli::parse"]["ignored"] = True
        self.assertTrue(any("changed ignored" in error for error in self.check()))

    def test_declared_addition_must_exist(self):
        del self.after["rust-suites"]["terminal"]["testcases"]["defaults"]
        self.assertTrue(any("missing declared addition" in error for error in self.check()))

    def test_relocation_destination_cannot_merge_distinct_tests(self):
        self.manifest["relocations"].append({"from_suite": "backend", "name": "solver", "to_suite": "terminal", "to_name": "cli::parse"})
        self.assertTrue(any("duplicate relocation destination" in error for error in self.check()))
