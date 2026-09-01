from __future__ import annotations
import json, os, shutil, signal, subprocess, threading, time, select
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

Log = Callable[[str], None]
@dataclass
class RunResult: returncode: int; cancelled: bool = False; output: str = ""
@dataclass
class RoomEqCommand:
    binary: Path | None = None
    @staticmethod
    def discover(explicit: str | None = None, repository: Path | None = None) -> Path | None:
        candidates = [explicit, os.environ.get("ROOMEQ_BIN"), shutil.which("roomeq")]
        if repository: candidates += [str(repository / "target/release/roomeq"), str(repository / "target/debug/roomeq")]
        return next((Path(path) for path in candidates if path and Path(path).is_file() and os.access(path, os.X_OK)), None)
    def argv(self, config: Path, output: Path, dry_run: bool = False) -> list[str]:
        if not self.binary: raise FileNotFoundError("RoomEQ binary not found. Build it with `just prod-roomeq` or pass --roomeq PATH.")
        return [str(self.binary), "--config", str(config), "--output", str(output), *(["--dry-run"] if dry_run else [])]
    def schema(self, kind: str) -> dict:
        if not self.binary: raise FileNotFoundError("RoomEQ binary is unavailable")
        completed = subprocess.run([str(self.binary), "--schema", kind], text=True, capture_output=True, check=True)
        return json.loads(completed.stdout)
    def run(self, config: Path, output: Path, *, dry_run: bool = False, log: Log | None = None, cancel: threading.Event | None = None) -> RunResult:
        process = subprocess.Popen(self.argv(config, output, dry_run), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, start_new_session=True)
        lines: list[str] = []; cancelled = False
        while process.poll() is None:
            readable, _, _ = select.select([process.stdout], [], [], .05) if process.stdout else ([], [], [])
            line = process.stdout.readline() if readable and process.stdout else ""
            if line: lines.append(line); log and log(line.rstrip())
            if cancel and cancel.is_set():
                cancelled = True
                try: os.killpg(process.pid, signal.SIGTERM)
                except PermissionError: process.terminate()
                try: process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    try: os.killpg(process.pid, signal.SIGKILL)
                    except PermissionError: process.kill()
        if process.stdout:
            remainder = process.stdout.read(); lines.append(remainder)
            for line in remainder.splitlines(): log and log(line)
        if process.stdout: process.stdout.close()
        return RunResult(process.wait(), cancelled, "".join(lines))
