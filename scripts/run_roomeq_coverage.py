#!/usr/bin/env python3
"""Accumulate RoomEQ coverage once and retain the audited source inventory."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys

ROOM_PACKAGES = ('roomeq-model', 'roomeq-analysis', 'roomeq-quality', 'roomeq-engine',
                 'roomeq-export', 'roomeq-workflow', 'roomeq-dsp')
PACKAGES = (*ROOM_PACKAGES, 'autoeq-optim-types')
# Match complete source paths: relocated integration/QA fixtures never enter
# the denominator. Neutral config/loss originated outside the six RoomEQ crates.
SOURCE_PATTERN = (r'/crates/(' + '|'.join(ROOM_PACKAGES) + r')/src/.*\.rs$'
                  r'|/crates/autoeq-optim-types/src/(?:lib|evidence|reference_registry)\.rs$')


def included(filename: str) -> bool:
    return re.search(SOURCE_PATTERN, filename.replace('\\', '/')) is not None


def inventory(report: dict, packages=PACKAGES) -> dict:
    rows = []
    for entry in report['data'][0]['files']:
        filename = entry['filename']
        if not included(filename):
            raise ValueError(f'out-of-scope file in coverage report: {filename}')
        relative = 'crates/' + filename.replace('\\', '/').split('/crates/', 1)[1]
        if '/roomeq-dsp/' in relative:
            origin = ('crates/roomeq-engine/src/fir/apply.rs' if relative.endswith('/fir.rs')
                      else 'crates/roomeq-engine/src/' + relative.split('/src/', 1)[1]
                      if not relative.endswith('/lib.rs') else 'new shared DSP module declarations')
        elif relative.endswith('/reference_registry.rs') and '/autoeq-optim-types/' in relative:
            origin = 'roomeq-model/src/reference_registry.rs'
        elif '/autoeq-optim-types/' in relative:
            origin = 'roomeq-model/src/report_contracts.rs'
        else:
            origin = relative
        rows.append({'path': relative, 'origin': origin, **entry['summary']['lines']})
    present = {row['path'].split('/')[1] for row in rows}
    missing = set(packages) - present
    if missing:
        raise ValueError(f'coverage has no source inventory for {sorted(missing)}')
    return {'scope': 'RoomEQ src files, including pre-existing embedded unit-test lines; external test and QA fixture files excluded',
            'excluded_contract_sources': ['config.rs', 'loss.rs', 'q_adjustment.rs'],
            'files': sorted(rows, key=lambda row: row['path'])}


def lcov_totals(lcov: str) -> dict:
    """Use LLVM LF/LH summaries, matching its JSON line-count semantics."""
    files = {}
    current = None
    for line in lcov.splitlines():
        if line.startswith('SF:'):
            current = line[3:].replace('\\', '/')
            if not included(current):
                raise ValueError(f'out-of-scope LCOV source: {current}')
            if current in files:
                raise ValueError(f'duplicate LCOV source record: {current}')
            files[current] = {}
        elif line.startswith(('LF:', 'LH:')):
            if current is None:
                raise ValueError('LCOV summary without source')
            files[current]['count' if line.startswith('LF:') else 'covered'] = int(line[3:])
        elif line == 'end_of_record':
            current = None
    if not files or any(set(row) != {'count', 'covered'} for row in files.values()):
        raise ValueError('missing LCOV source summaries')
    return files


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--cargo', default='cargo')
    parser.add_argument('--resume-contributors', action='store_true', help='retain completed library profiles and run integration contributors')
    parser.add_argument('--report-only', action='store_true', help='audit existing release profiles without rerunning tests')
    parser.add_argument('--output', type=Path, default=Path('target/qa'))
    args = parser.parse_args()
    if not os.environ.get('AUTOEQ_REPORT_PRODUCER'):
        parser.error('run through scripts/run_with_report_producer.py --profile release')
    cargo = shlex.split(args.cargo)
    root = args.source_root.resolve()
    metadata = json.loads(subprocess.check_output([*cargo, 'metadata', '--no-deps', '--offline', '--format-version', '1'], cwd=root))
    names = {package['name'] for package in metadata['packages'] if package['id'] in metadata['workspace_members']}
    packages = tuple(package for package in PACKAGES if package in names)
    expected = ROOM_PACKAGES[:-1] if 'autoeq' in names else PACKAGES
    if set(packages) != set(expected):
        raise ValueError('unexpected coverage package ownership')
    package_args = [arg for name in packages for arg in ('-p', name)]
    base = [*cargo, 'llvm-cov']
    run = lambda command: subprocess.run(command, cwd=root, check=True)
    if not args.report_only:
        # --no-report already retains profiles (and conflicts with --no-clean).
        # Follow cargo-llvm-cov's documented clean/accumulate/report sequence.
        if not args.resume_contributors:
            run([*base, 'clean', '--workspace'])
            run([*base, *package_args, '--lib', '--no-report', '--release'])
        contributors = [
            ('roomeq-workflow', ['dba_physical_outputs', 'joint_home_cinema', 'roomeq_pipeline_test', 'system_config_test']),
            ('roomeq-cli', ['roomeq_integration_test']),
            ('roomeq-qa', ['roomeq_generated_data_test']),
        ]
        if 'autoeq' in names:
            contributors = [('roomeq-workflow', ['dba_physical_outputs', 'joint_home_cinema']),
                            ('autoeq', ['roomeq_integration_test', 'roomeq_pipeline_test',
                                        'system_config_test', 'roomeq_generated_data_test'])]
        for package, tests in contributors:
            run([*base, '-p', package, *(['--all-features'] if package == 'autoeq' else []),
                 *(arg for name in tests for arg in ('--test', name)), '--no-report', '--release'])
    args.output.mkdir(parents=True, exist_ok=True)
    outside = sorted(path.name for path in (root / 'crates').iterdir()
                     if path.is_dir() and path.name not in packages)
    ignore = (r'/rustc/[^/]+/library/|/rustlib/src/rust/library/|/(tests|benches|examples)/|/crates/(' + '|'.join(map(re.escape, outside))
              + r')/|/crates/autoeq-optim-types/src/(config|loss|q_adjustment)\.rs$')
    # Dependencies outside selected workspace packages are ignored by
    # cargo-llvm-cov. inventory() rejects any unexpected remaining source.
    report_args = [*base, 'report', '--release', *package_args, '--ignore-filename-regex', ignore]
    report_path = args.output / 'roomeq-coverage.json'
    run([*report_args, '--json', '--summary-only', '--skip-functions', '--output-path', str(report_path)])
    data = json.loads(report_path.read_text())
    # Inlined standard-library/dependency source can use canonical paths that
    # cargo-llvm-cov's default ignore regex misses on symlinked toolchains.
    # Filter only files outside the explicit responsibility allowlist.
    outside_sources = sorted({entry['filename'] for entry in data['data'][0]['files']
                              if not included(entry['filename'])})
    if outside_sources:
        ignore += '|' + '|'.join(re.escape(path) + '$' for path in outside_sources)
        report_args = [*base, 'report', '--release', *package_args, '--ignore-filename-regex', ignore]
        run([*report_args, '--json', '--summary-only', '--skip-functions', '--output-path', str(report_path)])
        data = json.loads(report_path.read_text())
    ownership = inventory(data, packages)
    ownership['excluded_non_owner_sources'] = outside_sources
    run([*report_args, '--lcov', '--output-path', str(args.output / 'roomeq-coverage.lcov')])
    lcov_files = lcov_totals((args.output / 'roomeq-coverage.lcov').read_text())
    json_files = {entry['filename']: {metric: entry['summary']['lines'][metric] for metric in ('count', 'covered')}
                  for entry in data['data'][0]['files']}
    if lcov_files != json_files:
        raise ValueError('LCOV LF/LH and JSON per-file line inventories disagree')
    count = sum(row['count'] for row in lcov_files.values())
    covered = sum(row['covered'] for row in lcov_files.values())
    responsibility = {'count': count, 'covered': covered, 'percent': 100 * covered / count if count else 0}
    ownership['original_responsibility'] = responsibility
    (args.output / 'roomeq-coverage-ownership.json').write_text(json.dumps(ownership, indent=2) + '\n')
    print(f"Original RoomEQ responsibility: {responsibility['percent']:.2f}% (required 90%)", flush=True)
    subsystem = subprocess.run([sys.executable, str(Path(__file__).with_name('check_roomeq_subsystem_coverage.py')), str(report_path)], cwd=root, check=False)
    aggregate = subprocess.run([*report_args, '--summary-only', '--fail-under-lines', '90'], cwd=root, check=False)
    return int(responsibility['percent'] < 90 or subsystem.returncode != 0 or aggregate.returncode != 0)



if __name__ == '__main__':
    raise SystemExit(main())
