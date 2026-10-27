"""Failure controls for compiler-driven public export expansion."""
import json
from pathlib import Path
import tempfile
import unittest

from scripts.expand_root_exports import Catalog


def item(identifier, name, inner, visibility='public', span=None):
    return {'id': identifier, 'name': name, 'inner': inner, 'visibility': visibility, 'span': span}


class RootExportTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / 'crates/autoeq-fixture/src/lib.rs'
        self.source.parent.mkdir(parents=True)
        self.source.write_text('mod score { pub fn score() {} }\npub use score::*;\n')
        span = {'filename': 'crates/autoeq-fixture/src/lib.rs', 'begin': [2, 1], 'end': [2, 18]}
        self.data = {'root': 0, 'includes_private': True, 'paths': {}, 'index': {
            '0': item(0, 'autoeq_fixture', {'module': {'items': [1, 3]}}),
            '1': item(1, 'score', {'module': {'items': [2]}}, visibility='default'),
            '2': item(2, 'score', {'function': {}}),
            '3': item(3, None, {'use': {'is_glob': True, 'id': 1, 'name': 'score', 'source': 'score'}}, span=span),
        }}

    def catalog(self):
        (self.root / 'autoeq_fixture.json').write_text(json.dumps(self.data))
        return Catalog(self.root)

    def test_incomplete_metadata_is_rejected(self):
        self.data['includes_private'] = False
        with self.assertRaisesRegex(ValueError, 'private and hidden'):
            self.catalog()

    def test_private_module_does_not_hide_same_named_public_function(self):
        catalog = self.catalog()
        self.assertIn('score#value', catalog.exports('autoeq_fixture', '0'))
        self.assertNotIn('score#type', catalog.exports('autoeq_fixture', '0'))
        edits, _ = catalog.root_edits(self.root)
        self.assertEqual(edits[self.source.resolve()][0][3], 'pub use score::{score};')

    def test_hidden_public_item_is_preserved_from_compiler_metadata(self):
        self.data['index']['2']['attrs'] = ['#[doc(hidden)]']
        self.assertIn('score#value', self.catalog().exports('autoeq_fixture', '0'))

    def test_stale_source_span_is_rejected(self):
        self.source.write_text('// changed source\n' + self.source.read_text())
        with self.assertRaisesRegex(ValueError, 'stale or unsupported'):
            self.catalog().root_edits(self.root)

    def test_source_outside_repository_is_rejected(self):
        self.data['index']['3']['span']['filename'] = str(self.root.parent / 'outside/src/lib.rs')
        with self.assertRaisesRegex(ValueError, 'escapes repository'):
            self.catalog().root_edits(self.root)

    def test_tuple_constructor_is_public_only_when_all_fields_are_public(self):
        self.data['index']['2'] = item(2, 'score', {'struct': {'kind': {'tuple': [4]}}})
        self.data['index']['4'] = item(4, '0', {'struct_field': {}}, visibility='default')
        self.assertEqual(self.catalog().slots('autoeq_fixture', '2', 'score'), {'score#type'})
        self.data['index']['4']['visibility'] = 'public'
        self.assertEqual(self.catalog().slots('autoeq_fixture', '2', 'score'), {'score#type', 'score#value'})
        with self.assertRaisesRegex(ValueError, 'partially shadowed'):
            self.catalog().root_edits(self.root)

    def test_unavailable_module_metadata_fails_instead_of_dropping_exports(self):
        self.data['index']['3']['inner']['use']['id'] = 99
        with self.assertRaisesRegex(ValueError, 'unresolved module'):
            self.catalog().exports('autoeq_fixture', '0')
