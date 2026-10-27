#!/usr/bin/env python3
"""Prepare the canonical producer in the test profile, then run a test command."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', default='qa')
    parser.add_argument('--cargo', default='cargo')
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command:
        parser.error('a test command is required after --')
    root = Path(__file__).resolve().parents[1]
    built = subprocess.run([*shlex.split(args.cargo), 'build', '--locked',
                            '-p', 'autoeq-report-wasm', '--bin', 'report-producer',
                            '--profile', args.profile, '--message-format=json-render-diagnostics'],
                           cwd=root, stdout=subprocess.PIPE, text=True, check=False)
    producers = []
    for line in built.stdout.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            print(line)
            continue
        if event.get('reason') == 'compiler-message':
            sys.stderr.write(event['message'].get('rendered') or '')
        if (event.get('reason') == 'compiler-artifact'
                and event['target']['name'] == 'report-producer' and event.get('executable')):
            producers.append(event['executable'])
    if built.returncode:
        return built.returncode
    if len(producers) != 1 or not Path(producers[0]).is_file():
        raise RuntimeError('Cargo did not identify the native report-producer executable')
    environment = dict(os.environ, AUTOEQ_REPORT_PRODUCER=producers[0], AUTOEQ_WORKSPACE_ROOT=str(root))
    return subprocess.run(command, cwd=root, env=environment, check=False).returncode


if __name__ == '__main__':
    raise SystemExit(main())
