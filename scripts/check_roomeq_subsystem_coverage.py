#!/usr/bin/env python3
"""Enforce RoomEQ line-coverage floors per subsystem from llvm-cov JSON."""

import json
import pathlib
import sys


THRESHOLDS = {
    "src/roomeq/acoustic_qa/": 90.0,
    "src/roomeq/eq/": 85.0,
    "src/roomeq/optimize/": 80.0,
    "src/roomeq/workflows/": 75.0,
}


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: check_roomeq_subsystem_coverage.py <llvm-cov.json>", file=sys.stderr)
        return 2
    report = json.loads(pathlib.Path(sys.argv[1]).read_text())
    files = report["data"][0]["files"]
    failed = False
    for prefix, threshold in THRESHOLDS.items():
        covered = 0
        count = 0
        for file_report in files:
            filename = file_report["filename"].replace("\\", "/")
            marker = filename.find(prefix)
            if marker < 0:
                continue
            lines = file_report["summary"]["lines"]
            covered += lines["covered"]
            count += lines["count"]
        percent = 100.0 * covered / count if count else 0.0
        state = "PASS" if percent >= threshold else "FAIL"
        print(f"{state} {prefix}: {percent:.2f}% (required {threshold:.2f}%)")
        failed |= percent < threshold
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())

