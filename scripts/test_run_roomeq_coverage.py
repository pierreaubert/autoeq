import unittest
from scripts.run_roomeq_coverage import PACKAGES, included, inventory, lcov_totals


class CoverageOwnershipTests(unittest.TestCase):
    def test_scope_follows_moved_production_sources(self):
        for path in ('roomeq-dsp/src/physical_routing.rs',
                     'autoeq-optim-types/src/reference_registry.rs',
                     'autoeq-optim-types/src/evidence.rs',
                     'roomeq-workflow/src/room_optimization.rs'):
            self.assertTrue(included('/workspace/crates/' + path), path)

    def test_fixtures_and_unrelated_contracts_are_excluded(self):
        for path in ('roomeq-workflow/tests/fixture.rs', 'roomeq-qa/src/registry.rs',
                     'autoeq-optim-types/src/config.rs', 'autoeq-optim-types/src/q_adjustment.rs', 'autoeq-optim-types/src/loss.rs',
                     'roomeq-cli/src/roomeq.rs'):
            self.assertFalse(included('/workspace/crates/' + path), path)

    def test_missing_owner_fails_closed(self):
        with self.assertRaisesRegex(ValueError, 'no source inventory'):
            inventory({'data': [{'files': []}]})

    def test_unexpected_report_file_fails_closed(self):
        with self.assertRaisesRegex(ValueError, 'out-of-scope'):
            inventory({'data': [{'files': [{'filename': '/workspace/crates/roomeq-qa/src/lib.rs'}]}]})

    def test_each_owner_has_line_mapping(self):
        files = [{'filename': '/workspace/crates/' + package + '/src/' +
                  ('evidence.rs' if package == 'autoeq-optim-types' else 'lib.rs'),
                  'summary': {'lines': {'count': 10, 'covered': 9, 'percent': 90}}}
                 for package in PACKAGES]
        result = inventory({'data': [{'files': files}]})
        self.assertEqual(len(result['files']), len(PACKAGES))

    def test_llvm_summaries_are_preserved_even_when_da_rows_differ(self):
        lcov = 'SF:/workspace/crates/roomeq-model/src/lib.rs\nDA:2,3\nLF:3\nLH:2\nend_of_record\n'
        self.assertEqual(lcov_totals(lcov), {'/workspace/crates/roomeq-model/src/lib.rs': {'count': 3, 'covered': 2}})

    def test_missing_line_summary_fails_closed(self):
        with self.assertRaisesRegex(ValueError, 'missing LCOV'):
            lcov_totals('SF:/workspace/crates/roomeq-model/src/lib.rs\nDA:2,3\nend_of_record\n')

    def test_optimizer_helper_cannot_enter_original_scope(self):
        with self.assertRaisesRegex(ValueError, 'out-of-scope'):
            lcov_totals('SF:/workspace/crates/autoeq-optim-types/src/q_adjustment.rs\nLF:3\nLH:3\nend_of_record\n')


if __name__ == '__main__':
    unittest.main()
