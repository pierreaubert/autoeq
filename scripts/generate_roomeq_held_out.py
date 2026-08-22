#!/usr/bin/env python3
"""Generate deterministic held-out response positions for bundled QA recordings."""

import csv
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATASETS = ("2.0_8361a", "2.0_d3v", "2.0_t7v")
FEM_DATASETS = {
    "large_multi_seat_2_1": ("left", "right"),
    "medium_multi_seat": ("left", "right"),
    "medium_stereo_2_1": ("left", "right"),
    "small_stereo_2_2_mso": ("left", "right"),
    "medium_surround_5_1": ("left", "right", "center", "surround_left", "surround_right"),
}
SEEDS = [1, 7, 42, 424242, 8675309]


def generate(source: Path, destination: Path, position: int, channel: str) -> None:
    with source.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames
        rows = list(reader)
    if fieldnames is None:
        raise ValueError(f"missing CSV header in {source}")
    frequency_key = "frequency_hz" if "frequency_hz" in fieldnames else "freq"
    spl_key = "spl_db" if "spl_db" in fieldnames else "spl"
    phase_key = "phase_deg" if "phase_deg" in fieldnames else "phase"
    channel_phase = 0.7 if channel == "R" else 0.0
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            frequency = float(row[frequency_key])
            octave = math.log2(max(frequency, 20.0) / 20.0)
            magnitude_delta = (
                0.32 * math.sin(octave * (1.1 + 0.13 * position) + channel_phase)
                + 0.11 * math.cos(octave * 2.3 + position)
            )
            phase_delta = 2.5 * math.sin(octave * 0.9 + position + channel_phase)
            row[frequency_key] = f"{frequency:.2f}"
            row[spl_key] = f"{float(row[spl_key]) + magnitude_delta:.4f}"
            row[phase_key] = f"{float(row[phase_key]) + phase_delta:.4f}"
            writer.writerow(row)


def generate_interpolated(
    sources: list[Path], destination: Path, position: int, channel: str
) -> None:
    """Synthesize an unseen seat from every available simulated training seat."""
    source_rows: list[list[dict[str, str]]] = []
    fieldnames: list[str] | None = None
    for source in sources:
        with source.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise ValueError(f"missing CSV header in {source}")
            if fieldnames is None:
                fieldnames = reader.fieldnames
            elif reader.fieldnames != fieldnames:
                raise ValueError(f"CSV columns differ in {source}")
            source_rows.append(list(reader))

    if fieldnames is None or not source_rows:
        raise ValueError(f"no FEM training measurements for {destination}")
    if len({len(rows) for rows in source_rows}) != 1:
        raise ValueError(f"CSV row counts differ for {destination}")

    frequency_key = "frequency_hz" if "frequency_hz" in fieldnames else "freq"
    spl_key = "spl_db" if "spl_db" in fieldnames else "spl"
    phase_key = "phase_deg" if "phase_deg" in fieldnames else "phase"
    channel_phase = 0.7 if channel == "R" else 0.0
    # Two distinct convex combinations represent positions between the simulated
    # training seats. Keeping all weights non-zero prevents either held-out curve
    # from being a disguised copy of a training measurement.
    patterns = ([0.50, 0.30, 0.20], [0.20, 0.35, 0.45])
    raw_weights = patterns[position - 1]
    weights = raw_weights[: len(source_rows)]
    weight_sum = sum(weights)
    weights = [weight / weight_sum for weight in weights]

    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row_index, rows in enumerate(zip(*source_rows)):
            frequency = float(rows[0][frequency_key])
            if any(abs(float(row[frequency_key]) - frequency) > 1e-6 for row in rows[1:]):
                raise ValueError(f"CSV frequency grids differ for {destination}")
            octave = math.log2(max(frequency, 20.0) / 20.0)
            spl = sum(weight * float(row[spl_key]) for weight, row in zip(weights, rows))
            phase_x = sum(
                weight * math.cos(math.radians(float(row[phase_key])))
                for weight, row in zip(weights, rows)
            )
            phase_y = sum(
                weight * math.sin(math.radians(float(row[phase_key])))
                for weight, row in zip(weights, rows)
            )
            phase = math.degrees(math.atan2(phase_y, phase_x))
            spl += 0.18 * math.sin(octave * (1.1 + 0.13 * position) + channel_phase)
            spl += 0.07 * math.cos(octave * 2.3 + position)
            phase += 1.5 * math.sin(octave * 0.9 + position + channel_phase)
            row = dict(rows[0])
            row[frequency_key] = f"{frequency:.2f}"
            row[spl_key] = f"{spl:.4f}"
            row[phase_key] = f"{phase:.4f}"
            writer.writerow(row)


def main() -> None:
    measured = ROOT / "data_tests" / "roomeq" / "measured"
    for dataset in DATASETS:
        directory = measured / dataset
        for channel in ("L", "R"):
            for position in (1, 2):
                generate(
                    directory / f"{channel}.csv",
                    directory / f"{channel}_heldout_{position}.csv",
                    position,
                    channel,
                )

    fem = ROOT / "data_tests" / "roomeq" / "generate" / "fem"
    for scenario, channels in FEM_DATASETS.items():
        directory = fem / scenario
        for channel in channels:
            sources = sorted(directory.glob(f"{channel}_lp*.csv"))
            for position in (1, 2):
                generate_interpolated(
                    sources,
                    directory / f"{channel}_heldout_{position}.csv",
                    position,
                    "R" if channel == "right" else "L",
                )

    manifest_path = ROOT / "data_tests" / "roomeq" / "acoustic_corpus" / "manifest.json"
    with manifest_path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    held_out = {
        "fem_large_multiseat_21": [
            (c, f"../generate/fem/large_multi_seat_2_1/{name}_heldout_{p}.csv")
            for c, name in (("L", "left"), ("R", "right"))
            for p in (1, 2)
        ],
        "fem_medium_multiseat_20": [
            (c, f"../generate/fem/medium_multi_seat/{name}_heldout_{p}.csv")
            for c, name in (("L", "left"), ("R", "right"))
            for p in (1, 2)
        ],
        "measured_stereo_8361a": [(c, f"../measured/2.0_8361a/{c}_heldout_{p}.csv") for c in ("L", "R") for p in (1, 2)],
        "fem_medium_stereo_21": [(c, f"../generate/fem/medium_stereo_2_1/{name}_heldout_{p}.csv") for c, name in (("L", "left"), ("R", "right")) for p in (1, 2)],
        "measured_stereo_d3v": [(c, f"../measured/2.0_d3v/{c}_heldout_{p}.csv") for c in ("L", "R") for p in (1, 2)],
        "measured_stereo_t7v": [(c, f"../measured/2.0_t7v/{c}_heldout_{p}.csv") for c in ("L", "R") for p in (1, 2)],
        "fem_small_stereo_22_mso": [
            (c, f"../generate/fem/small_stereo_2_2_mso/{name}_heldout_{p}.csv")
            for c, name in (("L", "left"), ("R", "right"))
            for p in (1, 2)
        ],
        "fem_medium_home_cinema_51": [(c, f"../generate/fem/medium_surround_5_1/{name}_heldout_{p}.csv") for c, name in (("L", "left"), ("R", "right"), ("C", "center"), ("SL", "surround_left"), ("SR", "surround_right")) for p in (1, 2)],
    }
    for scenario in manifest["scenarios"]:
        if scenario["id"] in held_out:
            scenario["held_out"] = [
                {"channel": channel, "path": path}
                for channel, path in held_out[scenario["id"]]
            ]
        scenario["robustness"] = {
            "seeds": SEEDS,
            "noise_peak_db": 0.2,
            "coherence_floor": 0.8,
        }
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")
if __name__ == "__main__":
    main()
