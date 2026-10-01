from __future__ import annotations
import argparse, csv, json, os, sys
from pathlib import Path
from ._toolkit import ensure_toolkit
from .app import build_app
from .backend import resolve_backend
from .model import WizardModel


def _default_device(devices):
    for device in devices:
        if device.is_default:
            return device.name
    return devices[0].name if devices else None


def _read_curve_csv(path: Path) -> dict:
    freq, spl = [], []
    with path.open("r", newline="") as handle:
        for row in csv.DictReader(handle):
            freq.append(float(row["freq"]))
            spl.append(float(row["spl"]))
    if not freq:
        raise ValueError(f"no curve rows: {path}")
    return {"freq": freq, "spl": spl}


# The native host relaunches [python, script] with no user arguments for the
# supervised session, so the parent forwards its resolved CLI config through
# this variable and the session child restores it (paths as absolute).
_SESSION_ARGS_ENV = "RECORDING_GUI_SESSION_ARGS"


def _forward_session_args(args) -> None:
    os.environ[_SESSION_ARGS_ENV] = json.dumps({
        "backend": args.backend,
        "bin": (str(Path(args.bin).expanduser().resolve())
                if args.bin else None),
        "channels": args.channels,
        "output_dir": (str(Path(args.output_dir).expanduser().resolve())
                       if args.output_dir else None),
        "curves": [str(Path(p).expanduser().resolve()) for p in args.curves],
    })


def _session_child_argv(env=None) -> list[str] | None:
    """Rebuild argv from the parent's forwarded config (None if absent)."""
    raw = (env if env is not None else os.environ).get(_SESSION_ARGS_ENV, "")
    try:
        saved = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(saved, dict):
        return None
    argv: list[str] = []
    if saved.get("backend"):
        argv += ["--backend", str(saved["backend"])]
    if saved.get("bin"):
        argv += ["--bin", str(saved["bin"])]
    if saved.get("channels"):
        argv += ["--channels", str(saved["channels"])]
    if saved.get("output_dir"):
        argv += ["--output-dir", str(saved["output_dir"])]
    curves = saved.get("curves") or []
    if curves:
        argv += ["--curves", *[str(p) for p in curves]]
    return argv


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="recording-gui",
        description="Recording wizard (measurement capture workflow).")
    parser.add_argument("--backend", choices=("auto", "fake", "cli"),
                        default="auto")
    parser.add_argument("--bin", default=None)
    parser.add_argument("--channels", default="L,R")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--curves", nargs="*", default=[])
    if argv is None and os.environ.get("GPUI_TOOLKIT_SESSION") == "1":
        forwarded = _session_child_argv()
        if forwarded is not None:
            argv = forwarded
    args = parser.parse_args(argv)
    if os.environ.get("GPUI_TOOLKIT_SESSION") != "1":
        _forward_session_args(args)
    ensure_toolkit()
    backend, demo_mode = resolve_backend(args.backend, args.bin)
    inputs, outputs = backend.list_devices()
    binary = getattr(backend, "binary", None)
    print(f"recording-gui: backend={backend.name}"
          + (f" ({binary})" if binary else "")
          + f" inputs={len(inputs)} outputs={len(outputs)}"
          + (" (demo mode)" if demo_mode else ""),
          file=sys.stderr)
    model = WizardModel(demo_mode=demo_mode)
    model.input_devices = inputs
    model.output_devices = outputs
    model.input_name = _default_device(inputs)
    model.output_name = _default_device(outputs)
    model.output_dir = args.output_dir
    channels = [c.strip() for c in args.channels.split(",") if c.strip()]
    model.init_takes(channels or ["L"])
    eval_curves = {}
    for csv_path in args.curves:
        path = Path(csv_path)
        try:
            eval_curves[path.stem] = _read_curve_csv(path)
        except (OSError, ValueError) as error:
            print(f"warning: skipping {csv_path}: {error}", file=sys.stderr)
    app = build_app(model, backend, eval_curves)
    if os.environ.get("GPUI_TOOLKIT_DUMP_IR") == "1":
        print(json.dumps(app.to_spec(), indent=2))
        return 0
    app.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
