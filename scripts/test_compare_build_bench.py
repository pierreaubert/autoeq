import copy
import unittest
from scripts.compare_build_bench import compare, freeze


def report(scale=1):
    return {'status': 'passed', 'tools': {'rustc': 'pinned'}, 'host': {'jobs': 8},
            'environment': {}, 'sources': {'autoeq': {'snapshot_identity': {'revision': 'baseline'}},
                                          'math-audio': {'snapshot_identity': {'revision': 'dependency'}}},
            'results': [{'case': 'owner', 'scenario': scenario, 'repetition': repetition,
                         'wall_seconds': (10 + repetition * .01) * scale,
                         'max_process_rss_bytes': 100, 'target_dir_bytes': 1000}
                        for scenario in ('private-edit', 'test-build', 'full-native') for repetition in range(3)]}


class BudgetTests(unittest.TestCase):
    def test_meaningful_improvements_pass(self):
        baseline = report()
        result = compare(baseline, report(.7), freeze(baseline, 'hash'), 'hash')
        self.assertEqual(result['status'], 'passed')

    def test_equal_workload_cannot_claim_improvement(self):
        baseline = report()
        result = compare(baseline, report(), freeze(baseline, 'hash'), 'hash')
        self.assertEqual(len(result['errors']), 3)

    def test_dependency_change_invalidates_comparison(self):
        baseline, candidate = report(), report(.7)
        candidate['sources']['math-audio']['snapshot_identity']['revision'] = 'different'
        with self.assertRaisesRegex(ValueError, 'identities differ'):
            compare(baseline, candidate, freeze(baseline, 'hash'), 'hash')

    def test_nightly_compiler_change_invalidates_comparison(self):
        baseline, candidate = report(), report(.7)
        baseline['nightly_toolchain'] = {'sha256': {'rustc': 'compiler-a'}}
        candidate['nightly_toolchain'] = {'sha256': {'rustc': 'compiler-b'}}
        with self.assertRaisesRegex(ValueError, 'identities differ'):
            compare(baseline, candidate, freeze(baseline, 'hash'), 'hash')

    def test_effective_build_flags_change_invalidates_comparison(self):
        baseline, candidate = report(), report(.7)
        baseline['effective_build_environment'] = {'CARGO_ENCODED_RUSTFLAGS': '-Ctarget-cpu=native'}
        candidate['effective_build_environment'] = {'CARGO_ENCODED_RUSTFLAGS': '-Ctarget-cpu=generic'}
        with self.assertRaisesRegex(ValueError, 'identities differ'):
            compare(baseline, candidate, freeze(baseline, 'hash'), 'hash')

    def test_non_measurement_runtime_asset_change_invalidates_comparison(self):
        baseline, candidate = report(), report(.7)
        bundle = 'crates/autoeq-report-wasm/dist/report2d.wasm'
        baseline['sources']['autoeq']['snapshot_identity']['runtime_inputs_sha256'] = {bundle: 'bundle-a'}
        candidate['sources']['autoeq']['snapshot_identity']['runtime_inputs_sha256'] = {bundle: 'bundle-b'}
        with self.assertRaisesRegex(ValueError, 'identities differ'):
            compare(baseline, candidate, freeze(baseline, 'hash'), 'hash')

    def test_baseline_cannot_change_after_freeze(self):
        baseline = report()
        with self.assertRaisesRegex(ValueError, 'changed after budget freeze'):
            compare(baseline, report(.7), freeze(baseline, 'hash'), 'different')

    def test_reduced_scope_and_duplicate_repetitions_are_rejected(self):
        baseline = report()
        candidate = report(.7)
        candidate['results'] = [row for row in candidate['results'] if row['scenario'] != 'full-native']
        with self.assertRaisesRegex(ValueError, 'scopes differ'):
            compare(baseline, candidate, freeze(baseline, 'hash'), 'hash')
        baseline['results'][1]['repetition'] = 0
        with self.assertRaisesRegex(ValueError, 'independent repetitions'):
            freeze(baseline, 'hash')

    def test_memory_regression_is_reported_even_with_faster_wall_time(self):
        baseline, candidate = report(), report(.7)
        candidate['results'][0]['max_process_rss_bytes'] = 1000
        candidate['results'][1]['max_process_rss_bytes'] = 1000
        result = compare(baseline, candidate, freeze(baseline, 'hash'), 'hash')
        self.assertIn('max_process_rss_bytes', result['errors'][0])

    def test_observed_variance_prevents_noise_claim(self):
        baseline, candidate = report(), report(.9)
        candidate['results'][0]['wall_seconds'] = 1
        result = compare(baseline, candidate, freeze(baseline, 'hash'), 'hash')
        self.assertFalse(result['required_improvements'][0]['passed'])


if __name__ == '__main__':
    unittest.main()
