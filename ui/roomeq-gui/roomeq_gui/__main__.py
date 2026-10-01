from __future__ import annotations
import argparse, json, os
from importlib.resources import files
from pathlib import Path
from .app import RoomEqGuiApp
from .commands import RoomEqCommand

# The native host relaunches [python, script] with no user arguments for the
# supervised session, so the parent forwards its resolved CLI config through
# this variable and the session child restores it (paths as absolute).
_SESSION_ARGS_ENV = "ROOMEQ_GUI_SESSION_ARGS"

def bundled(kind: str) -> dict: return json.loads(files("roomeq_gui.resources").joinpath(f"{kind}_schema.json").read_text())
def existing_config(value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_file(): raise argparse.ArgumentTypeError(f"RoomEQ configuration does not exist: {value}")
    return path
def _abspath(value) -> str | None:
    return str(Path(value).expanduser().resolve()) if value else None
def _forward_session_args(args) -> None:
    os.environ[_SESSION_ARGS_ENV] = json.dumps({
        "roomeq": _abspath(args.roomeq),
        "config": _abspath(args.config),
        "result": _abspath(args.result),
    })
def _session_child_argv(env=None) -> list[str] | None:
    """Rebuild argv from the parent's forwarded config (None if absent)."""
    try:
        saved = json.loads((env if env is not None else os.environ).get(_SESSION_ARGS_ENV, ""))
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(saved, dict):
        return None
    argv: list[str] = []
    for flag in ("roomeq", "config", "result"):
        if saved.get(flag):
            argv += [f"--{flag}", str(saved[flag])]
    return argv
def main(argv=None) -> None:
    parser = argparse.ArgumentParser(prog="roomeq-gui", description="Configure and review RoomEQ in a native GPUI application.")
    parser.add_argument("--roomeq", metavar="PATH", help="RoomEQ executable to use")
    parser.add_argument("--config", metavar="FILE", type=existing_config, help="load an existing RoomEQ JSON configuration at startup")
    parser.add_argument("--result", metavar="FILE", type=Path, help="open an existing RoomEQ result at startup")
    if argv is None and os.environ.get("GPUI_TOOLKIT_SESSION") == "1":
        forwarded = _session_child_argv()
        if forwarded is not None:
            argv = forwarded
    args = parser.parse_args(argv)
    if os.environ.get("GPUI_TOOLKIT_SESSION") != "1":
        _forward_session_args(args)
    root = Path(__file__).resolve().parents[3]
    command = RoomEqCommand(RoomEqCommand.discover(args.roomeq, root))
    warning = None
    try: input_schema, output_schema = command.schema("input"), command.schema("output")
    except Exception: input_schema, output_schema, warning = bundled("input"), bundled("output"), "Using bundled schemas; select a RoomEQ binary to verify compatibility."
    app = RoomEqGuiApp(input_schema, output_schema, command, args.config, args.result); app.schema_warning = warning
    if os.environ.get("GPUI_TOOLKIT_DUMP_IR") == "1": print(json.dumps(app.ir())); return
    app.run()

if __name__ == "__main__": main()
