#!/usr/bin/env python3
"""Report RoomEQ measured-suite status and cross-mode comparability.

Scans a suite output directory (as produced by
``scripts/test_roomeq_measured.sh``) and writes two machine-readable
reports:

* ``summary.json``: every expected scenario/mode combination with a
  terminal status (``accepted``/``unchanged``/``rejected``/``error``),
  the roomeq exit code when known, and the headline metrics.
* ``comparison.json``: per scenario, whether the pre-correction metrics
  agree across modes and an explicit diff of the modes' effective
  configurations. A mode must not look better because its baseline was
  conditioned differently.

With ``--check``, exits non-zero when any combination is missing its
result, finite pre-metrics, or effective configuration is missing, or shared
playback controls differ. Default mode only writes reports and always exits zero.

With ``--expect PATH``, additionally compares every combination against
a frozen expectations file (see
``data_tests/roomeq/measured/expected_outcomes.json``) and exits
non-zero on any drift: status and outcome must match exactly, and
``improvement_db`` must agree within 1e-4 dB (regression signal, not
bit-exactness proof across platforms).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import re
import sys

FAILED_RE = re.compile(r"FAILED RoomEQ measured: .* \(roomeq exit (\d+)\)")


def reject_nonfinite_json(token):
    raise ValueError(f"nonfinite JSON number: {token}")


def load_json(path: pathlib.Path):
    try:
        value = json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_nonfinite_json)
        return value if isinstance(value, dict) else None
    except (OSError, ValueError):
        return None


def combo_status(out_dir: pathlib.Path, scenario: str, mode: str) -> dict:
    """Derive one combination's terminal status from its artifacts."""
    combo = out_dir / scenario / mode
    result = combo / f"dsp-{mode}.json"
    entry: dict = {
        "scenario": scenario,
        "mode": mode,
        "status": "error",
        "roomeq_exit": None,
        "outcome": None,
        "improvement_db": None,
        "pre_target_weighted_rms_db": None,
        "post_target_weighted_rms_db": None,
        "dsp_json": None,
        "notes": [],
    }
    run_log = combo / "run.log"
    if run_log.is_file():
        try:
            text = run_log.read_text(encoding="utf-8", errors="replace")
        except OSError:
            text = ""
        match = FAILED_RE.search(text)
        if match:
            entry["roomeq_exit"] = int(match.group(1))
        elif result.is_file():
            entry["roomeq_exit"] = 0
    data = load_json(result)
    if data is None:
        if not combo.is_dir():
            entry["notes"].append("combo directory missing")
        elif not result.is_file():
            entry["notes"].append("no top-level DSP result")
        else:
            entry["notes"].append("DSP result unreadable")
        return entry
    entry["dsp_json"] = str(result.relative_to(out_dir))
    metadata = data.get("metadata")
    acceptance = metadata.get("correction_acceptance") if isinstance(metadata, dict) else None
    if not isinstance(acceptance, dict):
        entry["notes"].append("correction acceptance metadata unavailable")
        return entry
    outcome = acceptance.get("outcome")
    entry["outcome"] = outcome
    metrics = acceptance.get("metrics")
    if not isinstance(metrics, dict):
        metrics = {}
    for key in (
        "improvement_db",
        "pre_target_weighted_rms_db",
        "post_target_weighted_rms_db",
    ):
        value = metrics.get(key)
        if finite_metric(value):
            entry[key] = value
    if outcome in ("accepted", "unchanged", "rejected"):
        entry["status"] = outcome
    else:
        entry["notes"].append(f"unrecognized outcome: {outcome!r}")
    if entry["roomeq_exit"] not in (None, 0) and entry["status"] == "rejected":
        entry["notes"].append("rejected diagnostic saved despite failing exit")
    return entry


def diff_effective(a, b, path=""):
    """Yield leaf paths where two effective configs differ."""
    diffs = []
    if isinstance(a, dict) and isinstance(b, dict):
        for key in sorted(set(a) | set(b)):
            child = f"{path}.{key}" if path else str(key)
            if key not in a:
                diffs.append((child, "<absent>", b[key]))
            elif key not in b:
                diffs.append((child, a[key], "<absent>"))
            else:
                diffs.extend(diff_effective(a[key], b[key], child))
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            diffs.append((path or "<root>", f"<len {len(a)}>", f"<len {len(b)}>"))
        else:
            for index, (x, y) in enumerate(zip(a, b)):
                diffs.extend(diff_effective(x, y, f"{path}[{index}]"))
    elif a != b:
        diffs.append((path or "<root>", a, b))
    return diffs


def finite_metric(value) -> bool:
    """A metric must be a finite number rather than a boolean or missing value."""
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def shared_playback_controls(config: dict) -> dict:
    """Select the declared input, topology, target, and finalization controls.

    Mode-specific FIR/hybrid design settings remain in the complete diff. This
    equality check does not validate external measurement bytes or establish
    equal optimizer budgets, filter families, or physical capture conditions.
    """
    keys = (
        "version", "system", "speakers", "crossovers", "target_curve",
        "provenance", "recording_config", "measured_impulse_responses", "ctc",
    )
    controls = {key: config.get(key) for key in keys}
    optimizer = config.get("optimizer")
    if isinstance(optimizer, dict):
        controls["optimizer"] = {
            key: optimizer.get(key) for key in ("finalization",)
        }
    else:
        controls["optimizer"] = optimizer
    return controls


def compare_scenario(out_dir: pathlib.Path, scenario: str, modes: list[str]) -> dict:
    """Check complete pre-metrics and declared shared playback controls."""
    present = {}
    report = {
        "scenario": scenario,
        "modes_requested": modes,
        "modes_compared": [],
        "mode_results_complete": False,
        "pre_metrics_equal": None,
        "pre_metrics": {},
        "shared_playback_controls_equal": None,
        "shared_playback_control_diffs": [],
        "effective_config_diffs": [],
        "verdict": "fail",
        "notes": [],
    }
    if len(modes) < 2 or len(set(modes)) != len(modes):
        report["notes"].append("comparison requires at least two distinct requested modes")
        return report
    for mode in modes:
        data = load_json(out_dir / scenario / mode / f"dsp-{mode}.json")
        if data is not None:
            present[mode] = data
    report["modes_compared"] = sorted(present)
    missing = sorted(set(modes) - set(present))
    report["mode_results_complete"] = not missing
    if missing:
        report["notes"].append(f"mode results unavailable: {', '.join(missing)}")

    configs = {}
    for mode, data in sorted(present.items()):
        metadata = data.get("metadata")
        if not isinstance(metadata, dict):
            metadata = {}
        acceptance = metadata.get("correction_acceptance")
        metrics = acceptance.get("metrics") if isinstance(acceptance, dict) else None
        value = metrics.get("pre_target_weighted_rms_db") if isinstance(metrics, dict) else None
        valid_pre_metric = finite_metric(value) and value >= 0.0
        report["pre_metrics"][mode] = value if valid_pre_metric else None
        if not valid_pre_metric:
            report["notes"].append(f"finite pre-metric unavailable for {mode}")
        config = metadata.get("effective_config")
        optimizer = config.get("optimizer") if isinstance(config, dict) else None
        finalization = optimizer.get("finalization") if isinstance(optimizer, dict) else None
        if (not isinstance(config, dict)
                or not isinstance(config.get("version"), str)
                or not config.get("version")
                or not isinstance(config.get("speakers"), dict)
                or not config.get("speakers")
                or not isinstance(finalization, dict)
                or not isinstance(finalization.get("subwoofer_limiter"), bool)
                or not finite_metric(finalization.get("output_ceiling_dbfs"))):
            report["notes"].append(f"effective shared controls unavailable for {mode}")
        else:
            configs[mode] = config

    values = list(report["pre_metrics"].values())
    if not missing and all(finite_metric(v) for v in values):
        report["pre_metrics_equal"] = all(v == values[0] for v in values)
        if not report["pre_metrics_equal"]:
            report["notes"].append("pre-correction metrics differ across modes")

    if len(configs) == len(modes):
        baseline_mode = sorted(configs)[0]
        baseline = configs[baseline_mode]
        for mode in sorted(configs)[1:]:
            cfg = configs[mode]
            for path, left, right in diff_effective(baseline, cfg):
                report["effective_config_diffs"].append(
                    {"path": path, baseline_mode: left, mode: right}
                )
            for path, left, right in diff_effective(
                shared_playback_controls(baseline), shared_playback_controls(cfg)
            ):
                report["shared_playback_control_diffs"].append(
                    {"path": path, baseline_mode: left, mode: right}
                )
        report["shared_playback_controls_equal"] = not report["shared_playback_control_diffs"]
        if not report["shared_playback_controls_equal"]:
            report["notes"].append("declared shared playback controls differ across modes")
    if report["effective_config_diffs"]:
        report["notes"].append(
            f"{len(report['effective_config_diffs'])} effective-config leaf diffs; "
            "differences are recorded without assuming they are intentional"
        )
    if report["pre_metrics_equal"] and report["shared_playback_controls_equal"]:
        report["verdict"] = "pass"
    report["notes"].append(
        "config equality does not verify external measurement bytes, optimizer "
        "budget parity, filter-family equivalence, or physical capture conditions"
    )
    return report


# Mirror scripts/test_roomeq_measured.sh so the expected matrix is what was
# requested, not every fixture present (e.g. 2.1_sigberg2 is not requested).
DEFAULT_SCENARIOS = (
    "2.2_unknown 2.2_sigberg1 2.2_sigberg2 2.2_sigberg3 2.2_genelec "
    "2.0_8361a 2.0_d3v 2.0_fidelia 2.0_t7v_2024 2.0_t7v_2026 "
    "5.0_genelec 5.1_kef 5.1.4_genelec 2.0_ascilab1 2.0_ascilab2 2.0_ascilab3"
)
DEFAULT_MODES = "iir fir mixed mixed-phase"


def expected_matrix(args) -> tuple[list[str], list[str]]:
    scenarios = (args.scenarios or os.environ.get("SCENARIOS") or DEFAULT_SCENARIOS).split()
    modes = (args.modes or os.environ.get("MODES") or DEFAULT_MODES).split()
    return scenarios, modes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out_dir", help="suite output directory to scan")
    parser.add_argument("--in-dir", default="data_tests/roomeq/measured")
    parser.add_argument("--scenarios", default="")
    parser.add_argument("--modes", default="")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--expect", default="", help="frozen expectations JSON to compare against")
    args = parser.parse_args()
    out_dir = pathlib.Path(args.out_dir)
    in_dir = pathlib.Path(args.in_dir)
    if not out_dir.is_dir():
        print(f"missing output directory: {out_dir}", file=sys.stderr)
        return 2
    scenarios, modes = expected_matrix(args)
    entries = [combo_status(out_dir, s, m) for s in scenarios for m in modes]
    counts: dict[str, int] = {}
    for entry in entries:
        counts[entry["status"]] = counts.get(entry["status"], 0) + 1
    summary = {
        "generated_by": "scripts/roomeq_suite_report.py",
        "out_dir": str(out_dir),
        "in_dir": str(in_dir),
        "scenarios": scenarios,
        "modes": modes,
        "counts": counts,
        "combinations": {(e["scenario"], e["mode"]): e for e in entries},
    }
    # JSON keys must be strings; flatten the pair key.
    summary["combinations"] = {
        f"{e['scenario']}/{e['mode']}": e for e in entries
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    comparisons = [compare_scenario(out_dir, s, modes) for s in scenarios]
    comparison = {
        "generated_by": "scripts/roomeq_suite_report.py",
        "scenarios": {c["scenario"]: c for c in comparisons},
    }
    (out_dir / "comparison.json").write_text(
        json.dumps(comparison, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"scanned {len(entries)} combinations: {counts}")
    fails = [c["scenario"] for c in comparisons if c["verdict"] == "fail"]
    if fails:
        print(f"comparison evidence failed in: {', '.join(fails)}")
    if args.expect:
        frozen = load_json(pathlib.Path(args.expect))
        if frozen is None:
            print(f"unreadable expectations file: {args.expect}", file=sys.stderr)
            return 2
        expected = frozen.get("combinations") or {}
        drifts = []
        for key, entry in sorted(summary["combinations"].items()):
            want = expected.get(key)
            if want is None:
                drifts.append(f"{key}: no frozen expectation")
                continue
            if entry["status"] != want.get("status"):
                drifts.append(
                    f"{key}: status {want.get('status')} -> {entry['status']}"
                )
            if entry.get("outcome") != want.get("outcome"):
                drifts.append(
                    f"{key}: outcome {want.get('outcome')} -> {entry.get('outcome')}"
                )
            want_imp = want.get("improvement_db")
            got_imp = entry.get("improvement_db")
            if isinstance(want_imp, (int, float)) and isinstance(
                got_imp, (int, float)
            ):
                if abs(got_imp - want_imp) > 1e-4:
                    drifts.append(
                        f"{key}: improvement_db {want_imp} -> {got_imp}"
                    )
            elif (want_imp is None) != (got_imp is None):
                drifts.append(
                    f"{key}: improvement_db {want_imp} -> {got_imp}"
                )
        for key in sorted(set(expected) - set(summary["combinations"])):
            drifts.append(f"{key}: missing from scan")
        if drifts:
            print(f"EXPECT FAIL: {len(drifts)} drifts vs {args.expect}")
            for drift in drifts:
                print(f"  {drift}")
            return 2
        print(f"EXPECT PASS vs {args.expect}")
    if not args.check:
        return 0
    missing = [k for k, e in summary["combinations"].items() if e["status"] == "error"]
    if missing or fails:
        print(f"CHECK FAIL: {len(missing)} errors, {len(fails)} mismatches")
        return 2
    print("CHECK PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
