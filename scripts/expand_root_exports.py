#!/usr/bin/env python3
"""Audit or expand library-root globs using compiler-generated rustdoc JSON.

Generate metadata with nightly cargo doc, --document-hidden-items and
--document-private-items. This command previews by default. --apply changes only
src/lib.rs statements whose source spans still match the compiler's input.
Review conditional exports separately and check minimal feature configurations;
all-feature metadata describes only its selected compilation configuration.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re


class Catalog:
    def __init__(self, directory: Path):
        self.docs = {p.stem: json.loads(p.read_text()) for p in directory.glob('*.json')
                     if p.stem.startswith(('autoeq_', 'roomeq_'))}
        if not self.docs or any(not doc.get('includes_private') for doc in self.docs.values()):
            raise ValueError('export audit requires rustdoc JSON with private and hidden items enabled')
        self.module_paths = {}
        self.by_path = {}
        for name, doc in self.docs.items():
            self._modules(name, str(doc['root']), (name,))

    def _modules(self, name, identifier, path):
        key = (name, identifier)
        if key in self.module_paths:
            return
        self.module_paths[key] = path
        self.by_path[path] = key
        doc = self.docs[name]
        for child_id in doc['index'][identifier]['inner']['module']['items']:
            child = doc['index'][str(child_id)]
            if 'module' in child['inner']:
                self._modules(name, str(child_id), (*path, child['name']))

    def module(self, name, identifier):
        identifier = str(identifier)
        item = self.docs[name]['index'].get(identifier)
        if item and 'module' in item['inner']:
            return name, identifier
        path = self.docs[name]['paths'].get(identifier)
        if path is None or path['kind'] != 'module':
            raise ValueError(f'unresolved module: {name}:{identifier}')
        key = self.by_path.get(tuple(path['path']))
        if key is None:
            raise ValueError(f'missing module metadata: {path}')
        return key

    def symbol(self, name, identifier, seen=frozenset()):
        identifier = str(identifier)
        key = (name, identifier)
        if key in seen:
            raise ValueError(f'cyclic export alias: {key}')
        doc = self.docs[name]
        item = doc['index'].get(identifier)
        if item:
            use = item['inner'].get('use')
            if use and use['id'] is not None:
                return self.symbol(name, use['id'], seen | {key})
        path = doc['paths'].get(identifier)
        if path:
            return '/'.join(path['path']) + ':' + path['kind']
        if item:
            span = item.get('span')
            return f"{span['filename'] if span else name}:{item['name']}:{next(iter(item['inner']))}"
        raise ValueError(f'unresolved export identity: {key}')

    def slots(self, name, identifier, exported):
        identifier = str(identifier)
        doc = self.docs[name]
        item = doc['index'].get(identifier)
        if item and 'use' in item['inner']:
            return self.slots(name, item['inner']['use']['id'], exported)
        kind = next(iter(item['inner'])) if item else doc['paths'][identifier]['kind']
        if kind == 'struct':
            if item is None:
                path = doc['paths'][identifier]['path']
                provider = self.docs.get(path[0])
                if provider:
                    own = next((key for key, value in provider['paths'].items() if value['path'] == path), None)
                    item = provider['index'].get(own) if own else None
                    doc = provider
            if item is None:
                raise ValueError(f'cannot determine constructor namespace: {name}:{identifier}')
            shape = item['inner']['struct']['kind']
            public_constructor = shape == 'unit' or (isinstance(shape, dict) and 'tuple' in shape
                and all(field is not None and doc['index'][str(field)]['visibility'] == 'public' for field in shape['tuple']))
            namespaces = ['type', 'value'] if public_constructor else ['type']
        elif kind in ('function', 'constant', 'static', 'variant'):
            namespaces = ['value']
        elif kind in ('macro', 'proc_macro', 'proc_derive', 'proc_attribute'):
            namespaces = ['macro']
        else:
            namespaces = ['type']
        return {exported + '#' + namespace for namespace in namespaces}

    def exports(self, name, identifier, active=frozenset()):
        name, identifier = self.module(name, identifier)
        key = (name, identifier)
        if key in active:
            raise ValueError(f'cyclic glob export: {key}')
        doc = self.docs[name]
        children = [doc['index'][str(child)] for child in doc['index'][identifier]['inner']['module']['items']]
        output = {}
        explicit_names = set()
        globs = []
        for child in children:
            use = child['inner'].get('use')
            if use and use['is_glob']:
                if child['visibility'] == 'public':
                    globs.append(use)
                continue
            exported = use['name'] if use else child['name']
            if exported is None:
                continue
            # Private declarations also shadow names imported by a glob.
            slots = self.slots(name, child['id'], exported)
            explicit_names.update(slots)
            if child['visibility'] == 'public':
                for slot in slots:
                    output[slot] = self.symbol(name, child['id'])
        for use in globs:
            for exported, identity in self.exports(name, use['id'], active | {key}).items():
                if exported in explicit_names:
                    continue
                if exported in output and output[exported] != identity:
                    raise ValueError(f'ambiguous public glob: {key}::{exported}')
                output[exported] = identity
        return output

    def root_edits(self, root: Path):
        edits = {}
        scopes = {}
        for (name, identifier), path in sorted(self.module_paths.items()):
            doc = self.docs[name]
            parent = doc['index'][identifier]
            children = [doc['index'][str(child)] for child in parent['inner']['module']['items']]
            if not any(child['inner'].get('use', {}).get('is_glob')
                       and child['visibility'] == 'public' and child.get('span')
                       and child['span']['filename'].endswith('/src/lib.rs') for child in children):
                continue
            globs = []
            claimed = set()
            for child in children:
                use = child['inner'].get('use')
                if use and use['is_glob']:
                    if child['visibility'] == 'public':
                        globs.append(child)
                else:
                    exported = use['name'] if use else child['name']
                    if exported is not None:
                        claimed.update(self.slots(name, child['id'], exported))
            scoped = []
            for child in sorted(globs, key=lambda item: item['span']['begin'] if item['span'] else [0, 0]):
                span = child.get('span')
                if not span or not span['filename'].endswith('/src/lib.rs'):
                    continue
                source = (root / span['filename']).resolve()
                if not source.is_relative_to(root.resolve()):
                    raise ValueError(f'export source escapes repository: {source}')
                text = source.read_text()
                lines = text.splitlines(keepends=True)
                start_line, start_col = span['begin']
                end_line, end_col = span['end']
                start = sum(map(len, lines[:start_line - 1])) + start_col - 1
                end = sum(map(len, lines[:end_line - 1])) + end_col - 1
                original = text[start:end]
                match = re.fullmatch(r'pub use ([A-Za-z_][A-Za-z_0-9:]*)::\*;', original)
                if not match:
                    raise ValueError(f'stale or unsupported export span: {source}:{start_line}: {original!r}')
                exported = self.exports(name, child['inner']['use']['id'])
                selected_slots = set(exported) - claimed
                selected = sorted({slot.rsplit('#', 1)[0] for slot in selected_slots})
                for exported_name in selected:
                    namespaces = {slot for slot in exported if slot.rsplit('#', 1)[0] == exported_name}
                    if namespaces - selected_slots:
                        raise ValueError(f'partially shadowed multi-namespace export: {path}::{exported_name}')
                shadowed = sorted(set(exported) - selected_slots)
                claimed.update(exported)
                replacement = 'pub use ' + match.group(1) + '::{' + ', '.join(selected) + '};' if selected else ''
                edits.setdefault(source, []).append((start, end, original, replacement))
                scoped.append({'line': start_line, 'source': match.group(1), 'named_exports': selected,
                               'already_shadowed': shadowed})
            if scoped:
                scopes['::'.join(path)] = {'exports': self.exports(name, identifier), 'statements': scoped}
        return edits, scopes

    def scope_exports(self, scopes):
        return {path: self.exports(*self.by_path[tuple(path.split('::'))]) for path in scopes}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rustdoc-dir', required=True, type=Path)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output', type=Path)
    parser.add_argument('--compare', type=Path)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    catalog = Catalog(args.rustdoc_dir)
    if args.compare:
        expected = json.loads(args.compare.read_text())
        actual = catalog.scope_exports(expected)
        differences = [scope for scope in expected if expected[scope]['exports'] != actual[scope]]
        if differences:
            raise SystemExit('public export changes: ' + ', '.join(differences))
        print(f'Public exports preserved in {len(expected)} affected module scopes')
        return
    edits, scopes = catalog.root_edits(args.root)
    if args.output:
        args.output.write_text(json.dumps(scopes, indent=2, sort_keys=True) + '\n')
    if args.apply:
        for source, changes in edits.items():
            text = source.read_text()
            for start, end, original, replacement in sorted(changes, reverse=True):
                if text[start:end] != original:
                    raise ValueError(f'export source changed during audit: {source}')
                text = text[:start] + replacement + text[end:]
            source.write_text(text)
    print(f'{"Expanded" if args.apply else "Previewed"} {sum(map(len, edits.values()))} globs in {len(edits)} library roots')


if __name__ == '__main__':
    main()
