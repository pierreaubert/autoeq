#!/usr/bin/env python3
"""Capture exact executable help/version contracts before and after relocation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

BINARIES = (
    "autoeq", "benchmark-autoeq-speaker", "autoeq-download-speakers",
    "roomeq", "convert-recording", "roomeq-fuzzer", "roomeq-qa-quality",
    "roomeq-qa-coverage", "roomeq-qa-features", "roomeq-qa-synthetic",
    "roomeq-qa-acoustic", "migrate_provenance",
)


def capture(bin_dir: Path) -> dict:
    records = {}
    for binary in BINARIES:
        records[binary] = {}
        # These two hand-parsed runners ignore unknown flags and would run QA.
        # Capture only their declared display options.
        if binary == "roomeq-qa-quality":
            flags = ("--help", "--program-version")
        elif binary == "roomeq-qa-synthetic":
            flags = ("--help",)
        else:
            flags = ("--help", "--version", "--program-version")
        for flag in flags:
            result = subprocess.run(
                [str(bin_dir.resolve() / binary), flag],
                capture_output=True, timeout=30, check=False,
            )
            records[binary][flag] = {
                "exit_code": result.returncode,
                "stdout": result.stdout.decode("utf-8"),
                "stderr": result.stderr.decode("utf-8"),
            }
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bin-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--compare", type=Path)
    args = parser.parse_args()
    records = capture(args.bin_dir)
    args.output.write_text(json.dumps(records, indent=2, sort_keys=True) + "\n")
    if args.compare:
        expected = json.loads(args.compare.read_text())
        if expected != records:
            for binary in sorted(expected.keys() | records.keys()):
                if expected.get(binary) != records.get(binary):
                    print(f"CLI contract changed: {binary}")
            return 1
        print(f"CLI contracts identical: {len(records)} executables")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
