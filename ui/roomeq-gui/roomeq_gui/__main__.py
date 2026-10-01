from __future__ import annotations
import argparse, json
from importlib.resources import files
from pathlib import Path
from .app import RoomEqGuiApp
from .commands import RoomEqCommand

def bundled(kind: str) -> dict: return json.loads(files("roomeq_gui.resources").joinpath(f"{kind}_schema.json").read_text())
def existing_config(value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_file(): raise argparse.ArgumentTypeError(f"RoomEQ configuration does not exist: {value}")
    return path
def main() -> None:
    parser = argparse.ArgumentParser(prog="roomeq-gui", description="Configure and review RoomEQ in a native GPUI application.")
    parser.add_argument("--roomeq", metavar="PATH", help="RoomEQ executable to use")
    parser.add_argument("--config", metavar="FILE", type=existing_config, help="load an existing RoomEQ JSON configuration at startup")
    parser.add_argument("--result", metavar="FILE", type=Path, help="open an existing RoomEQ result at startup")
    args = parser.parse_args(); root = Path(__file__).resolve().parents[3]
    command = RoomEqCommand(RoomEqCommand.discover(args.roomeq, root))
    warning = None
    try: input_schema, output_schema = command.schema("input"), command.schema("output")
    except Exception: input_schema, output_schema, warning = bundled("input"), bundled("output"), "Using bundled schemas; select a RoomEQ binary to verify compatibility."
    app = RoomEqGuiApp(input_schema, output_schema, command, args.config, args.result); app.schema_warning = warning
    if __import__("os").environ.get("GPUI_TOOLKIT_DUMP_IR") == "1": print(json.dumps(app.ir())); return
    app.run()

if __name__ == "__main__": main()
