#!/usr/bin/env python3
"""Native linker diagnostic shim; used only in explicitly instrumented samples."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import resource
import subprocess
import sys
import time


def main() -> int:
    actual = os.environ["AUTOEQ_BENCH_REAL_LINKER"]
    directory = Path(os.environ["AUTOEQ_BENCH_LINK_LOG"])
    directory.mkdir(parents=True, exist_ok=True)
    arguments = sys.argv[1:]
    if os.environ.get("AUTOEQ_BENCH_APPLE_LINK_STATS") == "1":
        arguments = [*arguments, "-Wl,-print_statistics"]
    command = [actual, *arguments]
    stamp = f"link-{os.getpid()}-{time.time_ns()}"
    start = time.perf_counter()
    process = subprocess.run(command, capture_output=True)
    wall = time.perf_counter() - start
    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    sys.stdout.buffer.write(process.stdout)
    sys.stderr.buffer.write(process.stderr)
    (directory / f"{stamp}.stderr").write_bytes(process.stderr)
    (directory / f"{stamp}.argv.json").write_text(json.dumps(command) + "\n")
    timers = {}
    for line in process.stderr.decode(errors="replace").splitlines():
        match = re.match(r"^\s*([\w.-]+)\s+(\d+)ms\s+(\d+)ms\s+(\d+)ms\s+(\d+)ms\s*$", line)
        if match:
            timers[match[1]] = dict(zip(("max_ms", "total_ms", "user_ms", "sys_ms"), map(int, match.groups()[1:])))
    output = None
    for index, argument in enumerate(arguments[:-1]):
        if argument == "-o":
            output = arguments[index + 1]
    record = {
        "command_sha256": hashlib.sha256(json.dumps(command).encode()).hexdigest(),
        "output": output, "driver_wall_seconds": wall, "exit_code": process.returncode,
        "max_process_rss_bytes": usage.ru_maxrss if sys.platform == "darwin" else usage.ru_maxrss * 1024,
        "apple_linker_timers": timers,
        "lto_timers": {name: values for name, values in timers.items() if re.search("lto|bitcode|codegen", name, re.I)},
        "raw_stderr": str(directory / f"{stamp}.stderr"),
        "arguments": str(directory / f"{stamp}.argv.json"),
    }
    (directory / f"{stamp}.json").write_text(json.dumps(record, indent=2) + "\n")
    return process.returncode


if __name__ == "__main__":
    raise SystemExit(main())
