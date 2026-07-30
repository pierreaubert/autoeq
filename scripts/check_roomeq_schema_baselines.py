#!/usr/bin/env python3
"""Compare generated RoomEQ schemas with the committed compatibility baselines."""

from __future__ import annotations

import difflib
import argparse
import json
import os
import pathlib
import subprocess
import sys


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
SCHEMAS = {
    "input": REPO_ROOT / "src" / "bin" / "roomeq" / "input_schema.json",
    "output": REPO_ROOT / "src" / "bin" / "roomeq" / "output_schema.json",
}


def canonical_json(value: object) -> str:
    return json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def generate_schema(kind: str) -> object:
    environment = os.environ.copy()
    environment["CARGO_TERM_COLOR"] = "never"
    completed = subprocess.run(
        [
            "cargo",
            "run",
            "--quiet",
            "--locked",
            "-p",
            "autoeq",
            "--features",
            "cli",
            "--bin",
            "roomeq",
            "--",
            "--schema",
            kind,
        ],
        cwd=REPO_ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode:
        raise RuntimeError(
            f"failed to generate {kind} schema:\n{completed.stderr.rstrip()}"
        )
    try:
        return json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError(
            f"roomeq emitted invalid {kind} schema JSON: {error}\n"
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        ) from error


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--update", action="store_true", help="replace baselines with generated schemas")
    args = parser.parse_args()
    failed = False
    for kind, baseline_path in SCHEMAS.items():
        expected = json.loads(baseline_path.read_text(encoding="utf-8"))
        actual = generate_schema(kind)
        if args.update:
            baseline_path.write_text(canonical_json(actual), encoding="utf-8")
            print(f"UPDATED RoomEQ {kind} schema: {baseline_path.relative_to(REPO_ROOT)}")
            continue
        if actual == expected:
            print(f"PASS RoomEQ {kind} schema: {baseline_path.relative_to(REPO_ROOT)}")
            continue
        failed = True
        diff = difflib.unified_diff(
            canonical_json(expected).splitlines(),
            canonical_json(actual).splitlines(),
            fromfile=f"accepted-{kind}-schema",
            tofile=f"generated-{kind}-schema",
            lineterm="",
        )
        print(f"FAIL RoomEQ {kind} schema", file=sys.stderr)
        print("\n".join(diff), file=sys.stderr)
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
