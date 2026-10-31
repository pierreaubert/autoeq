"""Prepare a profile-matched report producer inside a full-native timed command."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import shutil
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.build_bench.execution import compiler_artifacts


def file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--profile', required=True)
    parser.add_argument('--cargo', required=True)
    parser.add_argument('--producer-record', type=Path, required=True)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    root = args.root.resolve()
    if not command or not (root / 'Cargo.toml').is_file():
        parser.error('a source workspace and native command are required')
    if not os.environ.get('CARGO_TARGET_DIR'):
        parser.error('full-native requires an explicit owned CARGO_TARGET_DIR')
    target = Path(os.environ['CARGO_TARGET_DIR']).resolve()
    record_path = args.producer_record.resolve()
    record_path.parent.mkdir(parents=True, exist_ok=True)
    stdout = record_path.with_suffix('.build.stdout')
    stderr = record_path.with_suffix('.build.stderr')
    build_command = [args.cargo, 'build', '--offline',
                     '-p', 'autoeq-report-wasm', '--bin', 'report-producer',
                     '--profile', args.profile, '--target-dir', str(target),
                     '--timings', '--message-format=json-render-diagnostics']
    built = subprocess.run(build_command, cwd=root, capture_output=True, text=True, check=False)
    stdout.write_text(built.stdout)
    stderr.write_text(built.stderr)
    sys.stderr.write(built.stderr)
    if built.returncode:
        return built.returncode
    artifacts = [item for item in compiler_artifacts(stdout)
                 if item['target']['name'] == 'report-producer'
                 and 'bin' in item['target'].get('kind', [])
                 and item['profile'].get('test') is False and item['executable']]
    if len(artifacts) != 1:
        raise RuntimeError('Cargo did not identify one ordinary report-producer executable')
    artifact = artifacts[0]
    producer = Path(artifact['executable']).resolve()
    if not producer.is_file() or not producer.is_relative_to(target):
        raise RuntimeError('report-producer executable is missing or outside the owned target')
    # The original native recipe prebuilds workspace binaries and may replace
    # Cargo's output while feature sets or build contexts unify. Bind tests to
    # an independent copy of the exact ordinary artifact selected above. This
    # preparation stays inside the measured command, and the copy is immutable
    # from Cargo's perspective; its own digest must still survive the lane.
    cargo_producer = producer
    cargo_digest = file_digest(cargo_producer)
    prepared_dir = Path(tempfile.mkdtemp(prefix='native-report-producer-', dir=target))
    producer = prepared_dir / cargo_producer.name
    shutil.copy2(cargo_producer, producer)
    independent = (producer.stat().st_dev, producer.stat().st_ino) != (
        cargo_producer.stat().st_dev, cargo_producer.stat().st_ino)
    if not independent or file_digest(producer) != cargo_digest:
        raise RuntimeError('report-producer preparation did not create an independent exact copy')
    record = {'root': str(root), 'requested_profile': args.profile,
              'target_dir': str(target), 'build_command': build_command,
              'build_stdout': str(stdout), 'build_stderr': str(stderr),
              'artifact': artifact, 'executable': str(producer),
              'executable_sha256': file_digest(producer),
              'cargo_executable': str(cargo_producer),
              'cargo_executable_sha256_at_prepare': cargo_digest,
              'independent_copy_verified': independent,
              'producer_build_included_in_measurement': True}
    record_path.write_text(json.dumps(record, indent=2) + '\n')
    environment = dict(os.environ, AUTOEQ_REPORT_PRODUCER=str(producer),
                       AUTOEQ_WORKSPACE_ROOT=str(root))
    result = subprocess.run(command, cwd=root, env=environment, check=False)
    after_digest = file_digest(producer) if producer.is_file() else None
    record.update(exit_code=result.returncode, after_lane_sha256=after_digest,
                  cargo_executable_after_lane_sha256=(
                      file_digest(cargo_producer) if cargo_producer.is_file() else None))
    record_path.write_text(json.dumps(record, indent=2) + '\n')
    if result.returncode == 0 and after_digest != record['executable_sha256']:
        raise RuntimeError('native lane changed its prepared report-producer artifact')
    return result.returncode


if __name__ == '__main__':
    raise SystemExit(main())
