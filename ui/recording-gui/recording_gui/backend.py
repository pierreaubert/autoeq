"""Capture backends (toolkit-free).

``FakeCaptureBackend`` serves scripted devices and deterministic synthetic
takes for tests and demos. ``CliCaptureBackend`` shells out to the
``sotf-capture`` binary (``devices --json`` / ``capture --json``) for real
measurement takes.
"""
from __future__ import annotations

import json
import math
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from .model import DeviceInfo

FAKE_CURVE_POINTS = 64


@dataclass(frozen=True)
class TakeRequest:
    """One stimulus take (mirrors ``sotf-capture capture`` flags)."""

    signal: str = "sweep"
    duration_s: float = 5.0
    sample_rate: int = 48000
    out_channels: tuple[int, ...] = (0,)
    in_channels: tuple[int, ...] = (0,)
    name: str = "take"
    output_dir: str | None = None
    device: str | None = None
    start_hz: float = 20.0
    end_hz: float = 20000.0
    amplitude: float = 0.5
    freq: float | None = None
    mic_calibration: str | None = None


@dataclass
class TakeResult:
    """Outcome of one take request."""

    ok: bool
    message: str = ""
    wav_paths: list[str] = field(default_factory=list)
    csv_paths: list[str] = field(default_factory=list)
    summaries: list[dict] = field(default_factory=list)


class CaptureBackend(Protocol):
    """What the wizard needs from a capture engine."""

    @property
    def name(self) -> str: ...

    def list_devices(
        self,
    ) -> tuple[list[DeviceInfo], list[DeviceInfo]]:
        """Return (inputs, outputs)."""
        ...

    def capture(self, request: TakeRequest) -> TakeResult:
        """Run one take to completion (blocking; handlers run off-thread)."""
        ...

    def save_session(self, path: str, payload: dict) -> str:
        """Persist the wizard session JSON; return the written path."""
        ...


def synth_take_curve(label: str) -> dict[str, list[float]]:
    """Deterministic fake measurement curve for one take label."""
    seed = sum(ord(ch) for ch in label) % 97
    freq = [
        20.0 * (20_000.0 / 20.0) ** (i / (FAKE_CURVE_POINTS - 1))
        for i in range(FAKE_CURVE_POINTS)
    ]
    spl = []
    for f in freq:
        logf = math.log10(f / 1000.0)
        room = 4.0 * math.sin(logf * 5.1 + seed) * math.exp(-abs(logf))
        tilt = -2.0 * max(0.0, logf) - 6.0 * max(0.0, -logf - 1.0)
        bump = 3.0 * math.exp(-((logf + 0.7) ** 2) / 0.02)
        spl.append(80.0 + room + tilt + bump)
    return {"freq": freq, "spl": spl}


class FakeCaptureBackend:
    """Scripted backend: instant takes, no audio hardware, no files."""

    name = "fake"

    def __init__(
        self,
        inputs: list[DeviceInfo] | None = None,
        outputs: list[DeviceInfo] | None = None,
    ) -> None:
        self._inputs = inputs if inputs is not None else [
            DeviceInfo("Fake Microphone", channels=2, sample_rate=48000,
                       is_default=True),
        ]
        self._outputs = outputs if outputs is not None else [
            DeviceInfo("Fake Speakers", channels=2, sample_rate=48000,
                       is_default=True),
        ]
        self.requests: list[TakeRequest] = []
        self.saved: dict[str, dict] = {}

    def list_devices(self) -> tuple[list[DeviceInfo], list[DeviceInfo]]:
        return (list(self._inputs), list(self._outputs))

    def capture(self, request: TakeRequest) -> TakeResult:
        self.requests.append(request)
        return TakeResult(
            ok=True,
            message=f"fake {request.signal} take '{request.name}'",
            summaries=[{
                "backend": "fake",
                "name": request.name,
                "curve": synth_take_curve(request.name),
            }],
        )

    def save_session(self, path: str, payload: dict) -> str:
        self.saved[path] = json.loads(json.dumps(payload))
        return path


def _channel_list(values: tuple[int, ...]) -> str:
    return ",".join(str(int(v)) for v in values)


class CliCaptureBackend:
    """Real takes via the ``sotf-capture`` binary (blocking subprocess)."""

    name = "cli"

    def __init__(self, binary: str) -> None:
        self.binary = binary

    def _run(self, argv: list[str]) -> subprocess.CompletedProcess[str]:
        try:
            return subprocess.run(
                argv, capture_output=True, text=True, check=False,
            )
        except OSError as error:
            raise RuntimeError(
                f"cannot run {' '.join(argv)}: {error}"
            ) from error

    def list_devices(self) -> tuple[list[DeviceInfo], list[DeviceInfo]]:
        proc = self._run([self.binary, "devices", "--json"])
        if proc.returncode != 0:
            raise RuntimeError(
                f"sotf-capture devices failed: {proc.stderr.strip()}"
            )
        try:
            raw = json.loads(proc.stdout)
        except json.JSONDecodeError as error:
            raise RuntimeError(
                f"sotf-capture devices printed invalid JSON: {error}"
            ) from error
        inputs = [
            DeviceInfo.from_cli(entry)
            for entry in raw.get("input", []) if isinstance(entry, dict)
        ]
        outputs = [
            DeviceInfo.from_cli(entry)
            for entry in raw.get("output", []) if isinstance(entry, dict)
        ]
        return (inputs, outputs)

    def capture(self, request: TakeRequest) -> TakeResult:
        argv = [
            self.binary, "capture",
            "--signal", request.signal,
            "--duration", str(request.duration_s),
            "--sample-rate", str(request.sample_rate),
            "--hwaudio-send-to", _channel_list(request.out_channels),
            "--hwaudio-record-from", _channel_list(request.in_channels),
            "--name", request.name,
            "--start-freq", str(request.start_hz),
            "--end-freq", str(request.end_hz),
            "--amp", str(request.amplitude),
            "--json",
        ]
        if request.output_dir:
            argv += ["--output-dir", request.output_dir]
        if request.device:
            argv += ["--device", request.device]
        if request.freq is not None:
            argv += ["--freq", str(request.freq)]
        if request.mic_calibration:
            argv += ["--microphone-compensation", request.mic_calibration]
        before = set()
        if request.output_dir:
            out_dir = Path(request.output_dir)
            if out_dir.is_dir():
                before = {p.name for p in out_dir.iterdir()}
        proc = self._run(argv)
        summaries: list[dict] = []
        for line in proc.stdout.splitlines():
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                parsed = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                summaries.append(parsed)
        wav_paths: list[str] = []
        csv_paths: list[str] = []
        if request.output_dir:
            out_dir = Path(request.output_dir)
            if out_dir.is_dir():
                for path in sorted(out_dir.iterdir()):
                    if path.name in before or not path.is_file():
                        continue
                    suffix = path.suffix.lower()
                    if suffix == ".wav":
                        wav_paths.append(str(path))
                    elif suffix == ".csv":
                        csv_paths.append(str(path))
        if proc.returncode != 0:
            detail = proc.stderr.strip() or proc.stdout.strip()
            return TakeResult(ok=False, message=detail or "capture failed",
                              wav_paths=wav_paths, csv_paths=csv_paths,
                              summaries=summaries)
        return TakeResult(ok=True, message=f"captured '{request.name}'",
                          wav_paths=wav_paths, csv_paths=csv_paths,
                          summaries=summaries)

    def save_session(self, path: str, payload: dict) -> str:
        target = Path(path)
        if target.suffix.lower() != ".json":
            target = target.with_suffix(".json")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return str(target)


def find_cli_binary(explicit: str | None = None) -> str | None:
    """Resolve the ``sotf-capture`` binary path (explicit or PATH)."""
    if explicit:
        candidate = Path(explicit)
        if candidate.is_file():
            return str(candidate)
        return None
    for name in ("sotf-capture", "sotf_capture"):
        found = shutil.which(name)
        if found:
            return found
    return None


def resolve_backend(
    mode: str,
    binary: str | None = None,
) -> tuple[CaptureBackend, bool]:
    """Build the backend for ``fake`` / ``cli`` / ``auto``.

    Returns (backend, demo_mode). ``auto`` uses the CLI when the binary
    resolves, else the fake backend in demo mode (with a stderr warning).
    ``cli`` raises when the binary is missing instead of silently demoing.
    """
    if mode == "fake":
        return (FakeCaptureBackend(), True)
    resolved = find_cli_binary(binary)
    if mode == "cli":
        if resolved is None:
            raise RuntimeError(
                "sotf-capture binary not found (tried PATH entries "
                "'sotf-capture' and 'sotf_capture'); pass --bin or use "
                "--backend fake."
            )
        return (CliCaptureBackend(resolved), False)
    if mode == "auto":
        if resolved is None:
            print("warning: sotf-capture binary not found; using fake "
                  "devices (demo mode). Pass --bin or --backend fake to "
                  "silence this warning.", file=sys.stderr)
            return (FakeCaptureBackend(), True)
        return (CliCaptureBackend(resolved), False)
    raise ValueError(f"unknown backend mode: {mode!r}")
