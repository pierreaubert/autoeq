"""Per-command clocks, memory observations, compiler artifacts, and link diagnostics."""
from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time


def process_tree_rss(root_pid: int) -> int | None:
    try:
        result = subprocess.run(["ps", "-axo", "pid=,ppid=,rss="], capture_output=True, text=True, check=True)
        records = [tuple(map(int, line.split())) for line in result.stdout.splitlines() if len(line.split()) == 3]
    except (OSError, ValueError, subprocess.CalledProcessError):
        return None
    descendants = {root_pid}
    while True:
        expanded = descendants | {pid for pid, parent, _rss in records if parent in descendants}
        if expanded == descendants:
            break
        descendants = expanded
    values = [rss * 1024 for pid, _parent, rss in records if pid in descendants]
    return sum(values) if values else None


def timed(command: list[str], cwd: Path, env: dict[str, str], output: Path) -> dict:
    """Measure a fresh child; never reuse cumulative RUSAGE_CHILDREN maxima."""
    output.parent.mkdir(parents=True, exist_ok=True)
    stdout = output.with_suffix(".stdout")
    stderr = output.with_suffix(".stderr")
    observations: list[int] = []
    stop = threading.Event()
    load_before = os.getloadavg() if hasattr(os, "getloadavg") else None
    start = time.perf_counter()
    with stdout.open("wb") as out, stderr.open("wb") as err:
        process = subprocess.Popen(command, cwd=cwd, env=env, stdout=out, stderr=err, start_new_session=True)

        def monitor() -> None:
            while not stop.is_set():
                rss = process_tree_rss(process.pid)
                if rss is not None:
                    observations.append(rss)
                stop.wait(0.1)

        sampler = threading.Thread(target=monitor, daemon=True)
        sampler.start()
        try:
            _pid, status, usage = os.wait4(process.pid, 0)
            process.returncode = os.waitstatus_to_exitcode(status)
            finished = time.perf_counter()
        except BaseException:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            process.wait()
            raise
        finally:
            stop.set()
            sampler.join()
    elapsed = finished - start
    return {
        "command": command, "cwd": str(cwd), "wall_seconds": elapsed,
        "exit_code": process.returncode,
        "max_process_rss_bytes": usage.ru_maxrss if sys.platform == "darwin" else usage.ru_maxrss * 1024,
        "sampled_tree_peak_rss_bytes": max(observations) if observations else None,
        "rss_samples": len(observations), "rss_sample_interval_seconds": 0.1,
        "load_average_before": load_before,
        "load_average_after": os.getloadavg() if hasattr(os, "getloadavg") else None,
        "stdout": str(stdout), "stderr": str(stderr),
    }


def compiler_artifacts(path: Path) -> list[dict]:
    result = []
    for line in path.read_text(errors="replace").splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(event, dict) and event.get("reason") == "compiler-artifact":
            result.append({key: event[key] for key in ("package_id", "target", "profile", "features", "filenames", "executable", "fresh")})
    return result


def fingerprints(target: Path) -> dict[str, tuple[int, int]]:
    """Observe compile fingerprints; execution logs are deliberately excluded."""
    return {
        str(path.relative_to(target)): (path.stat().st_mtime_ns, path.stat().st_size)
        for path in target.rglob("*") if path.is_file() and ".fingerprint" in path.parts
    }


def dir_size_bytes(path: Path) -> int:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def collect_timings(target: Path, destination: Path) -> list[str]:
    import shutil
    directory = target / "cargo-timings"
    destination.mkdir(parents=True, exist_ok=True)
    copied = []
    if directory.is_dir():
        for source in directory.glob("*.html"):
            copied.append(str(shutil.copy2(source, destination / source.name)))
    return copied
