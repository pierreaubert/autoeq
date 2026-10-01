"""Wizard domain state (toolkit-free mirror of RecordingScreenModel).

Only the fields the Python wizard drives are mirrored: step position,
device selection, signal configuration, take list + progress, per-step
results, and save metadata. Defaults match ``sotf-capture``'s
``RecordingScreenModel::default``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

#: (step id, step label) in workflow order.
STEPS: tuple[tuple[str, str], ...] = (
    ("config", "Devices & setup"),
    ("spl", "SPL calibration"),
    ("capture", "Capture"),
    ("probe", "Probe"),
    ("bass", "Bass anchor"),
    ("evaluating", "Evaluating"),
    ("saving", "Saving"),
)

#: dBFS reference shared by capture/probe/bass anchor (sotf-capture).
DEFAULT_SIGNAL_LEVEL_DB = -6.0206
DEFAULT_SWEEP_START_HZ = 20.0
DEFAULT_SWEEP_END_HZ = 20_000.0
DEFAULT_NUM_SWEEPS = 4

TAKE_PENDING = "pending"
TAKE_RECORDING = "recording"
TAKE_DONE = "done"
TAKE_FAILED = "failed"


@dataclass(frozen=True)
class DeviceInfo:
    """One audio device (input or output)."""

    name: str
    device_id: str | None = None
    channels: int = 0
    sample_rate: int | None = None
    sample_format: str | None = None
    is_default: bool = False

    @classmethod
    def from_cli(cls, raw: dict) -> "DeviceInfo":
        """Parse one ``sotf-capture devices --json`` entry (defensive)."""
        config = raw.get("default_config") or {}
        rate = config.get("sample_rate")
        channels = config.get("channels", 0)
        try:
            channels = int(channels)
        except (TypeError, ValueError):
            channels = 0
        try:
            rate = int(rate) if rate is not None else None
        except (TypeError, ValueError):
            rate = None
        return cls(
            name=str(raw.get("name", "?")),
            device_id=raw.get("device_id"),
            channels=channels,
            sample_rate=rate,
            sample_format=config.get("sample_format"),
            is_default=bool(raw.get("is_default", False)),
        )

    def label(self) -> str:
        bits = self.name
        if self.channels:
            bits += f" ({self.channels} ch"
            if self.sample_rate:
                bits += f", {self.sample_rate} Hz"
            bits += ")"
        if self.is_default:
            bits += " — default"
        return bits


@dataclass
class SignalConfig:
    """Stimulus configuration (capture step)."""

    signal_type: str = "sweep"
    duration_s: float = 5.0
    level_db: float = DEFAULT_SIGNAL_LEVEL_DB
    start_hz: float = DEFAULT_SWEEP_START_HZ
    end_hz: float = DEFAULT_SWEEP_END_HZ
    num_sweeps: int = DEFAULT_NUM_SWEEPS
    sample_rate: int = 48000
    pre_silence_s: float = 2.0
    bass_octave_s: float = 3.0

    def amplitude(self) -> float:
        """Linear amplitude in (0, 1] (clipped, never silent)."""
        try:
            amp = 10.0 ** (float(self.level_db) / 20.0)
        except (OverflowError, ValueError):
            amp = 0.5
        if not math.isfinite(amp) or amp <= 0.0:
            return 0.5
        return min(amp, 1.0)


@dataclass
class TakeState:
    """One capture take (channel × mic × position, hardware-routed)."""

    key: str
    label: str
    channel: str
    out_channel: int
    in_channel: int
    state: str = TAKE_PENDING
    progress: float = 0.0
    message: str = ""
    wav_path: str | None = None
    csv_path: str | None = None
    summary: dict | None = None


@dataclass
class WizardModel:
    """Mutable wizard session state (single-threaded action handlers)."""

    step: int = 0
    input_devices: list[DeviceInfo] = field(default_factory=list)
    output_devices: list[DeviceInfo] = field(default_factory=list)
    input_name: str | None = None
    output_name: str | None = None
    mic_calibration_path: str | None = None
    output_dir: str | None = None
    signal: SignalConfig = field(default_factory=SignalConfig)
    takes: list[TakeState] = field(default_factory=list)
    busy: bool = False
    cancel_requested: bool = False
    status: str = "idle"
    spl_result: dict | None = None
    probe_results: list[dict] = field(default_factory=list)
    bass_results: list[dict] = field(default_factory=list)
    save_name: str = "recording"
    room_width_m: float = 0.0
    room_depth_m: float = 0.0
    room_height_m: float = 0.0
    room_unit: str = "metric"
    setup_description: str = ""
    demo_mode: bool = False

    @property
    def step_id(self) -> str:
        return STEPS[self.step][0]

    @property
    def pending_takes(self) -> list[TakeState]:
        return [t for t in self.takes if t.state == TAKE_PENDING]

    @property
    def done_takes(self) -> list[TakeState]:
        return [t for t in self.takes if t.state == TAKE_DONE]

    @property
    def overall_progress(self) -> float:
        if not self.takes:
            return 0.0
        return sum(t.progress for t in self.takes) / len(self.takes)

    def advance(self) -> bool:
        """Move one step forward unless busy or at the end."""
        if self.busy or self.step >= len(STEPS) - 1:
            return False
        self.step += 1
        return True

    def retreat(self) -> bool:
        """Move one step back unless busy or at the start."""
        if self.busy or self.step <= 0:
            return False
        self.step -= 1
        return True

    def init_takes(
        self,
        channels: list[str],
        out_channels: list[int] | None = None,
        in_channels: list[int] | None = None,
    ) -> None:
        """Build one take per channel with hardware routing.

        ``out_channels`` routes playback (default: 0..n); ``in_channels``
        routes recording (default: shared input 0, like the CLI).
        """
        self.takes = []
        for index, channel in enumerate(channels):
            out = out_channels[index] if out_channels else index
            shared = in_channels[index] if in_channels else 0
            self.takes.append(TakeState(
                key=f"take-{index}",
                label=channel,
                channel=channel,
                out_channel=int(out),
                in_channel=int(shared),
            ))

    def metadata_is_valid(self) -> bool:
        """Room dimensions are all-zero (unspecified) or all positive."""
        values = (self.room_width_m, self.room_depth_m, self.room_height_m)
        if all(v == 0.0 for v in values):
            return True
        return all(
            isinstance(v, (int, float)) and math.isfinite(v) and v > 0.0
            for v in values
        )

    def session_payload(self) -> dict:
        """JSON-serializable session summary written by Save."""
        return {
            "schema": "sotf-recording-wizard-v1",
            "save_name": self.save_name,
            "input_device": self.input_name,
            "output_device": self.output_name,
            "mic_calibration": self.mic_calibration_path,
            "signal": {
                "type": self.signal.signal_type,
                "duration_s": self.signal.duration_s,
                "level_db": self.signal.level_db,
                "start_hz": self.signal.start_hz,
                "end_hz": self.signal.end_hz,
                "sweeps": self.signal.num_sweeps,
                "sample_rate": self.signal.sample_rate,
            },
            "takes": [
                {
                    "label": t.label,
                    "channel": t.channel,
                    "out_channel": t.out_channel,
                    "in_channel": t.in_channel,
                    "state": t.state,
                    "wav": t.wav_path,
                    "csv": t.csv_path,
                }
                for t in self.takes
            ],
            "spl": self.spl_result,
            "probe": self.probe_results,
            "bass_anchor": self.bass_results,
            "room_m": {
                "width": self.room_width_m,
                "depth": self.room_depth_m,
                "height": self.room_height_m,
            },
            "setup_description": self.setup_description,
        }
