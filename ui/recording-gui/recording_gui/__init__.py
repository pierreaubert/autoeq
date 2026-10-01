"""Recording wizard: measurement capture workflow on gpui-toolkit.

Python mirror of the ``sotf-capture`` recording workflow
(``RecordingScreenModel`` + the app-gpui recording steps): one sidebar
section per step, a stepper in each section, and a pluggable capture
backend (``FakeCaptureBackend`` for tests/demos, ``CliCaptureBackend``
driving the ``sotf-capture`` binary for real takes).
"""

from .backend import (
    CliCaptureBackend,
    FakeCaptureBackend,
    TakeRequest,
    TakeResult,
    resolve_backend,
)
from .model import STEPS, DeviceInfo, SignalConfig, TakeState, WizardModel

__all__ = [
    "STEPS",
    "CliCaptureBackend",
    "DeviceInfo",
    "FakeCaptureBackend",
    "SignalConfig",
    "TakeRequest",
    "TakeResult",
    "WizardApp",
    "WizardModel",
    "build_app",
    "resolve_backend",
]


def __getattr__(name: str):
    # The App layer needs gpui_toolkit; keep model/backend importable
    # without it so pure-logic tests run anywhere.
    if name in ("WizardApp", "build_app"):
        from .app import WizardApp, build_app

        return {"WizardApp": WizardApp, "build_app": build_app}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
