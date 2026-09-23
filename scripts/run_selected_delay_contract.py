#!/usr/bin/env python3
"""Generate and independently verify a public selected physical-delay graph."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import tempfile

from check_selected_delay_artifact import check_selected_delay


TEST_NAME = "roadmap_correction_joint_drive_pair_reaches_public_finalization"
COMBINED_TEST_NAME = "roadmap_correction_joint_drive_search_reaches_final_graph_trials"
SCRATCH_ROOT = Path("/Volumes/home_tmp/tmp")


def run(artifact: Path, *, combined: bool = False) -> dict:
    root = Path(__file__).resolve().parents[1]
    if artifact.exists():
        raise ValueError(f"refusing to overwrite selected-delay artifact: {artifact}")
    artifact.parent.mkdir(parents=True, exist_ok=True)
    test_name = COMBINED_TEST_NAME if combined else TEST_NAME
    artifact_variable = (
        "ROOMEQ_SELECTED_GAIN_DELAY_ARTIFACT" if combined else "ROOMEQ_SELECTED_DELAY_ARTIFACT"
    )
    command = [
        "cargo", "test", "-p", "autoeq", "--test", "roomeq_admission_correction",
        test_name, "--", "--exact",
    ]
    environment = dict(os.environ, TMPDIR=str(SCRATCH_ROOT))
    environment[artifact_variable] = str(artifact)
    completed = subprocess.run(command, cwd=root, env=environment,
                               capture_output=True, text=True, timeout=600, check=False)
    if completed.returncode:
        raise RuntimeError(f"selected-delay public producer failed:\n{completed.stdout}\n{completed.stderr}")
    if not artifact.is_file():
        raise RuntimeError("public producer did not save its finalized output")
    report = check_selected_delay(json.loads(artifact.read_text()))
    if combined and not report["selected_trial"].startswith("joint_drive_gain_delay_"):
        raise ValueError("public producer did not select a combined gain-and-delay trial")
    report["producer_test"] = test_name
    report["artifact"] = str(artifact)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path,
                        help="Persist the selected graph at a new path; default is temporary")
    parser.add_argument("--combined", action="store_true",
                        help="Verify the selected public gain-and-delay graph")
    arguments = parser.parse_args()
    if arguments.artifact is not None:
        report = run(arguments.artifact.resolve(), combined=arguments.combined)
        report["artifact_retained"] = True
        print(json.dumps(report, indent=2))
        return
    if not SCRATCH_ROOT.is_dir():
        raise RuntimeError(f"required scratch directory is missing: {SCRATCH_ROOT}")
    with tempfile.TemporaryDirectory(prefix="roomeq-selected-delay-", dir=SCRATCH_ROOT) as directory:
        report = run(Path(directory) / "selected-output.json", combined=arguments.combined)
    report.pop("artifact")
    report["artifact_retained"] = False
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
