#!/usr/bin/env python3
"""Generate and independently replay the public routed joint-sub graph."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import tempfile

from src.loaders import RoomEqData
from verify_routed_electrical import assess_bound_artifact, output_complex_transfers


TEST_NAME = "roadmap_correction_admission_routed_joint_sub_named_outputs_reach_refinement"
SCRATCH_ROOT = Path(os.environ.get("ROOMEQ_QA_SCRATCH_ROOT", "/Volumes/home_tmp/tmp"))
FREQUENCIES_HZ = (40.0, 80.0, 120.0)


def check_routed_joint(data: dict, artifact_path: Path) -> dict:
    """Verify binding, routed peaks, and the fixture's source/output ownership."""
    routing = (data.get("metadata", {}).get("bass_management") or {}).get("routing_graph") or {}
    if routing.get("input_channels") != ["L", "R"]:
        raise ValueError("routed joint fixture needs separate L/R inputs")
    if not {"Sub1", "RearBass"}.issubset(routing.get("output_channels") or []):
        raise ValueError("routed joint fixture lost named physical outputs")
    bass_routes = {
        (route["source_channel"], route["destination"])
        for route in routing.get("routes") or []
        if route.get("route_kind") == "redirected_bass_lowpass_to_sub"
    }
    if bass_routes != {("L", "Sub1"), ("R", "RearBass")}:
        raise ValueError("routed joint fixture changed its bass-route ownership")

    bound = RoomEqData(data, artifact_path.resolve().parent)
    report = assess_bound_artifact(bound)
    if report["status"] != "independent_sampled_electrical_match":
        raise ValueError("routed joint fixture did not match within-limit electrical replay")

    transfers = output_complex_transfers(bound, FREQUENCIES_HZ, 48_000.0)
    for output, source, other in [("Sub1", "L", "R"), ("RearBass", "R", "L")]:
        if not all(abs(value) > 1e-12 for value in transfers[output][source]):
            raise ValueError(f"{output} lost its declared bass source")
        if any(abs(value) > 1e-12 for value in transfers[output][other]):
            raise ValueError(f"{output} acquired an undeclared bass source")
    report["inputs"] = routing["input_channels"]
    report["bass_routes"] = sorted([list(route) for route in bass_routes])
    report["frequencies_hz"] = FREQUENCIES_HZ
    return report


def run(artifact_path: Path) -> dict:
    if artifact_path.exists():
        raise ValueError(f"refusing to overwrite routed joint artifact: {artifact_path}")
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[1]
    environment = dict(os.environ, TMPDIR=str(SCRATCH_ROOT))
    environment["ROOMEQ_ROUTED_JOINT_ARTIFACT"] = str(artifact_path)
    command = [
        "cargo", "test", "-p", "autoeq", "--test", "roomeq_admission_correction",
        TEST_NAME, "--", "--exact",
    ]
    completed = subprocess.run(
        command, cwd=root, env=environment, capture_output=True, text=True,
        timeout=600, check=False,
    )
    if completed.returncode:
        raise RuntimeError(f"routed joint public producer failed:\n{completed.stdout}\n{completed.stderr}")
    if not artifact_path.is_file():
        raise RuntimeError("public producer did not save the routed final graph")
    report = check_routed_joint(json.loads(artifact_path.read_text()), artifact_path)
    report["producer_test"] = TEST_NAME
    report["artifact"] = str(artifact_path)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, help="Persist at a new path; default is temporary")
    arguments = parser.parse_args()
    if arguments.artifact is not None:
        report = run(arguments.artifact.resolve())
        report["artifact_retained"] = True
        print(json.dumps(report, indent=2))
        return
    if not SCRATCH_ROOT.is_dir():
        raise RuntimeError(f"required scratch directory is missing: {SCRATCH_ROOT}")
    with tempfile.TemporaryDirectory(prefix="roomeq-routed-joint-", dir=SCRATCH_ROOT) as directory:
        report = run(Path(directory) / "selected-output.json")
    report.pop("artifact")
    report["artifact_retained"] = False
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
