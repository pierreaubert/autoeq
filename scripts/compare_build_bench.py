#!/usr/bin/env python3
"""Freeze baseline budgets before comparing controlled build measurements."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import statistics


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def groups(report):
    if report['status'] != 'passed':
        raise ValueError('only successful reports can establish acceptance')
    grouped = {}
    for row in report['results']:
        if not row.get('diagnostic'):
            grouped.setdefault((row['case'], row['scenario']), []).append(row)
    if not grouped:
        raise ValueError('report has no normal measurement rows')
    for key, rows in grouped.items():
        if len(rows) < 3 or len({row['repetition'] for row in rows}) != len(rows):
            raise ValueError(f'{key}: requires at least three independent repetitions')
    return grouped


def identity(report):
    return {'harness': report.get('harness_sha256'), 'tools': report['tools'],
            'nightly_toolchain': report.get('nightly_toolchain'),
            'host': report['host'], 'environment': report['environment'],
            'effective_build_environment': report.get('effective_build_environment'),
            'fixtures': report['sources']['autoeq']['snapshot_identity'].get('runtime_inputs_sha256', {}),
            'dependencies': {name: record['snapshot_identity'] for name, record in report['sources'].items()
                             if name != 'autoeq'}}


def freeze(report, source_digest):
    rows = []
    for (case, scenario), samples in sorted(groups(report).items()):
        walls = [sample['wall_seconds'] for sample in samples]
        median = statistics.median(walls)
        spread = max(walls) - min(walls)
        rows.append({'case': case, 'scenario': scenario, 'baseline_median_seconds': median,
                     'baseline_spread_seconds': spread,
                     # Fifteen percent is an investigation ceiling, not a
                     # speedup claim. Improvements must exceed observed noise.
                     'max_wall_seconds': median + max(0.15 * median, spread),
                     'max_process_rss_bytes': 1.20 * statistics.median(sample['max_process_rss_bytes'] for sample in samples),
                     'max_target_dir_bytes': 1.20 * statistics.median(sample['target_dir_bytes'] for sample in samples)})
    return {'schema': 'refactor-20261025/budgets/v1', 'baseline_report_sha256': source_digest,
            'comparison_identity': identity(report), 'baseline_source': report['sources']['autoeq']['snapshot_identity'],
            'policy': {'wall_investigation_fraction': 0.15, 'memory_and_target_fraction': 0.20,
                       'minimum_improvement_fraction': 0.05,
                       'required_improvement_workloads': ['private-edit', 'test-build', 'full-native']},
            'rows': rows}


def compare(baseline, candidate, budgets, baseline_digest):
    if budgets['baseline_report_sha256'] != baseline_digest:
        raise ValueError('baseline changed after budget freeze')
    if budgets['comparison_identity'] != identity(baseline) or identity(baseline) != identity(candidate):
        raise ValueError('toolchain, host, environment, or dependency identities differ')
    before, after = groups(baseline), groups(candidate)
    if set(before) != set(after):
        raise ValueError('baseline/candidate workload scopes differ')
    limits = {(row['case'], row['scenario']): row for row in budgets['rows']}
    if set(limits) != set(before):
        raise ValueError('budget workload scopes differ')
    rows, errors = [], []
    for key in sorted(before):
        a, b = before[key], after[key]
        old = statistics.median(row['wall_seconds'] for row in a)
        new = statistics.median(row['wall_seconds'] for row in b)
        limit = limits[key]
        regressions = []
        for metric, maximum in (('wall_seconds', 'max_wall_seconds'),
                                ('max_process_rss_bytes', 'max_process_rss_bytes'),
                                ('target_dir_bytes', 'max_target_dir_bytes')):
            value = statistics.median(row[metric] for row in b)
            if value > limit[maximum]:
                regressions.append(metric)
        if regressions:
            errors.append(f'{key}: investigate material regressions: {regressions}')
        rows.append({'case': key[0], 'scenario': key[1], 'before_seconds': old, 'after_seconds': new,
                     'ratio': new / old, 'investigate': regressions})
    improvements = []
    for scenario in budgets['policy']['required_improvement_workloads']:
        selected = [key for key in before if key[1] == scenario]
        if not selected:
            errors.append(f'missing required improvement workload: {scenario}')
            continue
        def totals(group):
            repetitions = set.intersection(*(set(row['repetition'] for row in group[key]) for key in selected))
            if len(repetitions) < 3:
                raise ValueError('composite workload has fewer than three complete repetitions')
            return [sum(next(row['wall_seconds'] for row in group[key] if row['repetition'] == repetition)
                        for key in selected) for repetition in sorted(repetitions)]
        a, b = totals(before), totals(after)
        old, new = statistics.median(a), statistics.median(b)
        noise = max(max(a) - min(a), max(b) - min(b), old * budgets['policy']['minimum_improvement_fraction'])
        passed = old - new > noise
        improvements.append({'scenario': scenario, 'before_seconds': old, 'after_seconds': new,
                             'observed_noise_floor_seconds': noise, 'passed': passed})
        if not passed:
            errors.append(f'{scenario}: improvement does not exceed observed noise and 5% floor')
    return {'status': 'passed' if not errors else 'failed', 'errors': errors,
            'required_improvements': improvements, 'rows': rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', required=True, type=Path)
    parser.add_argument('--candidate', type=Path)
    parser.add_argument('--budgets', required=True, type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    baseline = json.loads(args.baseline.read_text())
    if args.candidate is None:
        if args.budgets.exists():
            parser.error('refusing to overwrite frozen acceptance budgets')
        args.budgets.write_text(json.dumps(freeze(baseline, digest(args.baseline)), indent=2) + '\n')
        return 0
    if args.output is None:
        parser.error('--output is required for a comparison')
    result = compare(baseline, json.loads(args.candidate.read_text()), json.loads(args.budgets.read_text()), digest(args.baseline))
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(result['status'])
    for error in result['errors']:
        print(error)
    return int(bool(result['errors']))


if __name__ == '__main__':
    raise SystemExit(main())
