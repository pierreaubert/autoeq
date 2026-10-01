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
    args = parser.parse_args(argv)
    ensure_toolkit()
    backend, demo_mode = resolve_backend(args.backend, args.bin)
    inputs, outputs = backend.list_devices()
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
