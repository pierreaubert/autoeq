#!/usr/bin/env python3
"""Record per-unit compiler phase diagnostics in instrumented builds only."""
from __future__ import annotations

import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time


def main() -> int:
    directory = Path(os.environ['AUTOEQ_BENCH_COMPILE_LOG'])
    directory.mkdir(parents=True, exist_ok=True)
    command = sys.argv[1:]
    stamp = f'compile-{os.getpid()}-{time.time_ns()}'
    start = time.perf_counter()
    process = subprocess.run(command, capture_output=True)
    wall = time.perf_counter() - start
    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    sys.stdout.buffer.write(process.stdout)
    sys.stderr.buffer.write(process.stderr)
    stderr = directory / f'{stamp}.stderr'
    stderr.write_bytes(process.stderr)
    phases = []
    for line in process.stderr.decode(errors='replace').splitlines():
        if line.startswith('time: {'):
            try:
                phases.append(json.loads(line[len('time: '):]))
            except json.JSONDecodeError:
                continue
    def argument(name):
        return command[command.index(name) + 1] if name in command else None
    record = {'command': command, 'crate_name': argument('--crate-name'),
              'crate_type': argument('--crate-type'), 'test': '--test' in command,
              'compiler_wall_seconds': wall, 'exit_code': process.returncode,
              'max_process_rss_bytes': usage.ru_maxrss if sys.platform == 'darwin' else usage.ru_maxrss * 1024,
              'stderr': str(stderr), 'phases': phases}
    (directory / f'{stamp}.json').write_text(json.dumps(record, indent=2) + '\n')
    return process.returncode


if __name__ == '__main__':
    raise SystemExit(main())
