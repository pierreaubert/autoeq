"""Regression checks for matching benchmark probes and executed test identities."""
import os
from pathlib import Path
import sys
import tempfile
import unittest

from scripts.build_bench.plans import cases
from scripts.build_bench.probes import select_edit
from scripts.build_bench.scopes import execute_matching_tests, resolve_identities, scope_for_compiler_profile, validate_listing, validate_result


class ScopeTests(unittest.TestCase):
    def test_backend_edit_does_not_follow_legacy_evaluator_file_order(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for label, include_evaluator in [('before', True), ('after', False)]:
                checkout = root / label
                crate = checkout / 'crates/autoeq-optim'
                backend = crate / 'src/optim/de/tests.rs'
                backend.parent.mkdir(parents=True)
                backend.write_text('#[test] fn backend_case() { assert_eq!(2, 2); }')
                if include_evaluator:
                    evaluator = crate / 'src/loss/epa/score/tests.rs'
                    evaluator.parent.mkdir(parents=True)
                    evaluator.write_text('#[test] fn evaluator_case() { assert_eq!(1, 1); }')
                case = next(c for c in cases({'autoeq-optim': {}}, None, True) if c.name == 'optimizer')
                package = {'name': case.package, 'manifest_path': str(crate / 'Cargo.toml')}
                probe = select_edit(checkout, package, 'test-edit', case.test_probe_file)
                self.assertEqual(probe.path, backend)
                self.assertIn(b'backend_case', probe.changed)

    def test_public_probes_follow_logical_contract_owners(self):
        for packages, expected_objective, expected_dsp in [
            ({'autoeq-optim': {}}, 'autoeq-optim', 'roomeq-engine'),
            ({'autoeq-objective': {}, 'roomeq-dsp': {}}, 'autoeq-objective', 'roomeq-dsp'),
        ]:
            selected = {c.name: c for c in cases(packages, None, True)}
            self.assertEqual(selected['objective'].package, expected_objective)
            self.assertEqual(selected['objective'].public_type, 'ObjectiveData')
            with tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                case = selected['dsp']
                source = root / case.public_function_file
                source.parent.mkdir(parents=True)
                source.write_text('pub fn resolve() {}')
                package = {'name': expected_dsp, 'manifest_path': str(root / 'crates' / expected_dsp / 'Cargo.toml')}
                probe = select_edit(root, package, 'public-edit', case.public_function_file, public_function=True)
                self.assertIn(b'pub fn autoeq_build_bench_contract_probe()', probe.changed)
                self.assertNotIn(b'impl crate::', probe.changed)

    def test_scope_guards_reject_missing_extra_duplicate_and_unexecuted_cases(self):
        conditional = {'tests': {'kept': False, 'release_only': False},
                       'profile_conditions': {'release_only': {'debug_assertions': False}}}
        development = scope_for_compiler_profile(conditional, {'debug_assertions': True})
        self.assertEqual(development['tests'], {'kept': False})
        self.assertEqual(development['profile_excluded_tests'], conditional['profile_conditions'])
        self.assertEqual(scope_for_compiler_profile(conditional, {'debug_assertions': False})['tests'], conditional['tests'])
        for profile in [None, {}, {'debug_assertions': 'false'}]:
            with self.assertRaises(ValueError):
                scope_for_compiler_profile(conditional, profile)
        expected = {'kept': False, 'ignored': True}
        validate_listing('ignored: test\nkept: test\n', expected)
        validate_result('test result: ok. 1 passed; 0 failed; 1 ignored; 0 measured; 5 filtered out;\n', expected)
        for listing in ['kept: test\n', 'ignored: test\nkept: test\nextra: test\n',
                        'ignored: test\nkept: test\nkept: test\n']:
            with self.assertRaises(RuntimeError):
                validate_listing(listing, expected)
        with self.assertRaises(ValueError):
            validate_listing('', {})
        for result in ['test result: ok. 0 passed; 0 failed; 0 ignored;',
                       'test result: ok. 2 passed; 0 failed; 0 ignored;']:
            with self.assertRaises(RuntimeError):
                validate_result(result, expected)

    def test_explicit_relocations_are_bijective_and_fail_closed(self):
        expected = {"modern": False}
        aliases = {"modern": ["legacy"]}
        self.assertEqual(resolve_identities(["legacy"], expected, aliases), {"modern": "legacy"})
        self.assertEqual(resolve_identities(["modern"], expected, aliases), {"modern": "modern"})
        for available in [["unrelated"], ["modern", "legacy"]]:
            with self.assertRaises(RuntimeError):
                resolve_identities(available, expected, aliases)
        with self.assertRaises(RuntimeError):
            resolve_identities(["legacy"], {"first": False, "second": False},
                               {"first": ["legacy"], "second": ["legacy"]})

    @unittest.skipIf(os.name == 'nt', 'feedback measurement uses POSIX child accounting')
    def test_execution_uses_exact_matching_identities_and_preserves_ignored_count(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            binary = root / 'tests'
            binary.write_text('#!' + sys.executable + '\n'
                              'import sys\n'
                              'if "--exact" not in sys.argv:\n'
                              ' print("ignored: test\\nkept: test\\nunrelated: test"); raise SystemExit(0)\n'
                              'assert sys.argv[1:5] == ["--test-threads=1", "--exact", "ignored", "kept"]\n'
                              'if "--list" in sys.argv: print("ignored: test\\nkept: test")\n'
                              'else: print("test result: ok. 1 passed; 0 failed; 1 ignored; 0 measured; 7 filtered out;")\n')
            binary.chmod(0o755)
            result = execute_matching_tests(str(binary), {'tests': {'kept': False, 'ignored': True}},
                                            root, dict(os.environ), root / 'measured')
            self.assertTrue(result['exact_test_scope_verified'])
            self.assertEqual(result['logical_test_count'], 2)
            self.assertEqual(result['logical_ignored_count'], 1)


if __name__ == '__main__':
    unittest.main()
