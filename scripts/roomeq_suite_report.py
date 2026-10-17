#!/usr/bin/env python3
"""Report RoomEQ measured-suite status and cross-mode comparability.

Scans a suite output directory (as produced by
``scripts/test_roomeq_measured.sh``) and writes two machine-readable
reports:

* ``summary.json``: every expected scenario/mode combination with a
  terminal status (``accepted``/``unchanged``/``safe_fallback``/``rejected``/``error``),
  the roomeq exit code when known, and the headline metrics.
* ``comparison.json``: per scenario, whether the frozen baseline/target/grid
  receipts agree across modes, separate native safety pre-metrics, and an explicit diff of the modes' effective
  configurations. A mode must not look better because its baseline was
  conditioned differently.

With ``--check``, exits non-zero when any recorded pipeline stage failed, a combination is missing its
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
import hashlib
import math
import os
import pathlib
import re
import statistics
import sys

from epa_evidence import inspect_level_matched_epa, inspect_epa_spectral_flatness

REPORT_SOURCE_SHA256 = hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()

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
        "score_basis": None,
        "native_marginal_metrics": None,
        "optimizer_training_scores": None,
        "dsp_json": None,
        "diagnostic_artifact": load_json(combo / "diagnostic-artifact.json"),
        "stages": load_json(combo / "stages.json"),
        "level_matched_epa": None,
        "epa_spectral_flatness": None,
        "pre_ringing_audible_db": None,
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
    if entry["stages"] is not None:
        entry["roomeq_exit"] = entry["stages"].get("roomeq")
        for stage, code in entry["stages"].items():
            if code not in (None, 0):
                entry["notes"].append(f"{stage} stage failed with exit {code}")
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
    try:
        entry["level_matched_epa"] = inspect_level_matched_epa(metadata)
        entry["epa_spectral_flatness"] = inspect_epa_spectral_flatness(metadata)
    except ValueError as error:
        entry["notes"].append(str(error))
        return entry
    temporal = (acceptance.get("acoustic_quality") or {}).get("temporal") or {}
    if finite_metric(temporal.get("pre_ringing_audible_db")):
        entry["pre_ringing_audible_db"] = temporal["pre_ringing_audible_db"]
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
    entry["native_marginal_metrics"] = {
        key: entry[key] for key in ("improvement_db", "pre_target_weighted_rms_db",
                                   "post_target_weighted_rms_db")
    }
    entry["optimizer_training_scores"] = {
        key: metadata.get(key) if finite_metric(metadata.get(key)) else None
        for key in ("pre_score", "post_score")
    }
    entry["score_basis"] = "native_marginal_legacy"
    frozen = frozen_grid_scores(acceptance)
    if frozen is not None:
        pre = frozen["pre_weighted_rms_mean_db"]
        post = frozen["post_weighted_rms_mean_db"]
        entry.update(pre_target_weighted_rms_db=pre, post_target_weighted_rms_db=post,
                     improvement_db=pre-post,
                     score_basis="configured_reference_training_mean" if all(
                         row.get("configured_reference_method") == "configured_routing_unprocessed_captures_v1"
                         for row in frozen["basis"]
                     ) else "frozen_grid_training_mean")
    if outcome in ("accepted", "unchanged", "safe_fallback", "rejected"):
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


def frozen_grid_scores(acceptance):
    """Validate complete fixed-grid receipts; never extrapolate missing support."""
    quality = acceptance.get("acoustic_quality") if isinstance(acceptance, dict) else None
    training = quality.get("training") if isinstance(quality, dict) else None
    if not isinstance(training, dict):
        return None
    receipts = training.get("frozen_grid_residual")
    count = training.get("curve_count")
    configured = training.get("configured_baseline")
    seats = None
    if configured is not None:
        if (not isinstance(configured, dict)
                or configured.get("method") != "configured_routing_unprocessed_captures_v1"
                or configured.get("unavailable") != []
                or not isinstance(configured.get("playback_graph_sha256"), str)
                or re.fullmatch(r"[0-9a-f]{64}", configured["playback_graph_sha256"]) is None
                or not isinstance(configured.get("residuals"), list)):
            return None
        seats = configured["residuals"]
        if any(not isinstance(seat, dict) or not isinstance(seat.get("score"), dict)
               or not isinstance(seat.get("seat_index"), int) or isinstance(seat["seat_index"], bool)
               or seat["seat_index"] < 0 or not isinstance(seat.get("physical_outputs"), list)
               or not seat["physical_outputs"]
               or any(not isinstance(output, str) or not output for output in seat["physical_outputs"])
               or len(set(seat["physical_outputs"])) != len(seat["physical_outputs"])
               for seat in seats):
            return None
        identities = [(seat["score"].get("logical_input"), seat["seat_index"]) for seat in seats]
        if any(not isinstance(name, str) or not name for name, _ in identities) or len(set(identities)) != len(identities):
            return None
        receipts = [seat["score"] for seat in seats]
    if (not isinstance(receipts, list) or not receipts
            or not isinstance(count, int) or isinstance(count, bool)
            or count != len(receipts) or not all(isinstance(row, dict) for row in receipts)):
        return None
    indices = [row.get("curve_index") for row in receipts]
    if (any(not isinstance(index, int) or isinstance(index, bool) for index in indices)
            or sorted(indices) != list(range(count))):
        return None
    receipts = sorted(receipts, key=lambda row: row["curve_index"])
    basis = []
    for row in receipts:
        if any(not isinstance(row.get(key), str)
               or re.fullmatch(r"[0-9a-f]{64}", row[key]) is None
               for key in ("frequency_grid_sha256", "baseline_samples_sha256", "target_samples_sha256")):
            return None
        band = row.get("evaluated_band_hz")
        samples = row.get("sample_count")
        if (not isinstance(band, list) or len(band) != 2
                or not all(finite_metric(value) for value in band)
                or not 0 < band[0] < band[1]
                or not isinstance(samples, int) or isinstance(samples, bool) or samples < 2
                or not isinstance(row.get("normalized_to_pre"), bool)
                or row.get("post_covers_baseline_grid") is not True
                or row.get("logical_input") is not None and not isinstance(row["logical_input"], str)):
            return None
        if any(not finite_metric(row.get(key)) for key in (
                "level_offset_db", "post_reference_gain_db", "pre_weighted_rms_db",
                "post_weighted_rms_db", "improvement_db")):
            return None
        if (row["pre_weighted_rms_db"] < 0 or row["post_weighted_rms_db"] < 0
                or not math.isclose(row["improvement_db"],
                                    row["pre_weighted_rms_db"] - row["post_weighted_rms_db"],
                                    rel_tol=0, abs_tol=1e-9)):
            return None
        reference = {key: row.get(key) for key in (
            "logical_input", "curve_index", "frequency_grid_sha256",
            "baseline_samples_sha256", "target_samples_sha256", "sample_count",
            "evaluated_band_hz", "normalized_to_pre", "level_offset_db",
            "post_reference_gain_db", "pre_weighted_rms_db",
        )}
        if configured is not None:
            seat = next(seat for seat in seats if seat["score"]["curve_index"] == row["curve_index"])
            reference.update({"configured_reference_method": configured["method"],
                              "reference_graph_sha256": configured["playback_graph_sha256"],
                              "seat_index": seat["seat_index"],
                              "physical_outputs": sorted(seat["physical_outputs"])})
        relative = row.get("relative_target_reference")
        reference["target_reference_method"] = "raw_target_levels_v1"
        if relative is not None:
            if (row["normalized_to_pre"] is not True or not isinstance(relative, dict)
                    or relative.get("method") != "first_baseline_frequency_target_level_nanodb_v1"
                    or not finite_metric(relative.get("target_level_removed_db"))
                    or not finite_metric(relative.get("level_offset_db"))
                    or not isinstance(relative.get("target_shape_sha256"), str)
                    or re.fullmatch(r"[0-9a-f]{64}", relative["target_shape_sha256"]) is None
                    or not math.isclose(relative["level_offset_db"],
                                        row["level_offset_db"] + relative["target_level_removed_db"],
                                        rel_tol=0, abs_tol=1e-8)):
                return None
            # The original target hash/offset remain in the saved DSP receipt.
            # Relative scoring cancels their constant level; compare the
            # explicitly recorded zero-first-sample shape and pre level plane.
            reference["target_reference_method"] = relative["method"]
            reference["target_samples_sha256"] = relative["target_shape_sha256"]
            reference["level_offset_db"] = relative["level_offset_db"]
        basis.append(reference)
    pre = statistics.median(row["pre_weighted_rms_db"] for row in receipts)
    post = statistics.median(row["post_weighted_rms_db"] for row in receipts)
    return {"basis": basis, "pre_weighted_rms_median_db": pre,
            "post_weighted_rms_median_db": post, "improvement_db": pre - post,
            "pre_weighted_rms_mean_db": statistics.mean(row["pre_weighted_rms_db"] for row in receipts),
            "post_weighted_rms_mean_db": statistics.mean(row["post_weighted_rms_db"] for row in receipts)}


# Numerical tie tolerance only, matching native finalization ranking. This is
# not the 0.25 dB native seat-regression budget or an audibility allowance.
HYBRID_PART_TIE_EPS_DB = 1e-9


def hybrid_best_part_check(present, report):
    """Judge corrected hybrid against eligible standalone parts on one frozen basis."""
    check = {"status": "unavailable", "score_basis": "configured_reference_training_mean",
             "required_for_accepted_hybrid": False, "passed": None,
             "hybrid_post_mean_db": None, "best_part": None,
             "best_part_post_mean_db": None, "signed_regression_db": None,
             "tolerance_db": HYBRID_PART_TIE_EPS_DB}
    if "mixed" not in report["modes_requested"]:
        check.update(status="not_requested", reason="hybrid not requested")
        return check
    metadata = (present.get("mixed") or {}).get("metadata")
    acceptance = metadata.get("correction_acceptance") if isinstance(metadata, dict) else None
    check["required_for_accepted_hybrid"] = isinstance(acceptance, dict) and acceptance.get("outcome") == "accepted"
    scores = report["frozen_grid_scores"]
    parts = {}
    for mode in ("iir", "fir", "mixed"):
        data = present.get(mode)
        if data is None or mode not in report["modes_requested"]:
            check["reason"] = f"standalone comparison result unavailable: {mode}"
            return check
        metadata = data.get("metadata")
        acceptance = metadata.get("correction_acceptance") if isinstance(metadata, dict) else None
        if (not isinstance(acceptance, dict)
                or acceptance.get("outcome") not in ("accepted", "unchanged", "safe_fallback")):
            check["reason"] = f"approved playback result unavailable: {mode}"
            return check
        score = scores.get(mode)
        if (score is None or not score["basis"]
                or any(row.get("configured_reference_method") != "configured_routing_unprocessed_captures_v1"
                       for row in score["basis"])):
            check["reason"] = f"configured-reference evidence unavailable: {mode}"
            return check
        parts[mode] = score
    if (report["shared_playback_controls_equal"] is not True
            or any(parts[mode]["basis"] != parts["mixed"]["basis"] for mode in ("iir", "fir"))):
        check["reason"] = "standalone and hybrid score bases or playback controls differ"
        return check
    best = min(("iir", "fir"), key=lambda mode: (parts[mode]["post_weighted_rms_mean_db"], mode))
    hybrid = parts["mixed"]["post_weighted_rms_mean_db"]
    best_score = parts[best]["post_weighted_rms_mean_db"]
    margin = hybrid - best_score
    passed = margin <= HYBRID_PART_TIE_EPS_DB
    check.update(status="pass" if passed else "regression", passed=passed,
                 hybrid_post_mean_db=hybrid, best_part=best,
                 best_part_post_mean_db=best_score, signed_regression_db=margin)
    if not check["required_for_accepted_hybrid"]:
        check["status"] = "fallback_diagnostic"
        check["reason"] = "flagged fallback is not an accepted hybrid correction"
    return check


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
        "pre_metric_basis": "native_marginal_safety_diagnostic",
        "comparison_pre_metrics": {},
        "comparison_pre_metrics_equal": None,
        "frozen_grid_basis_equal": None,
        "frozen_grid_scores": {},
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
    declared_frozen = False
    for mode, data in sorted(present.items()):
        metadata = data.get("metadata")
        if not isinstance(metadata, dict):
            metadata = {}
        acceptance = metadata.get("correction_acceptance")
        report["frozen_grid_scores"][mode] = frozen_grid_scores(acceptance)
        quality = acceptance.get("acoustic_quality") if isinstance(acceptance, dict) else None
        training = quality.get("training") if isinstance(quality, dict) else None
        declared_frozen |= isinstance(training, dict) and "frozen_grid_residual" in training
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
    frozen = list(report["frozen_grid_scores"].values())
    has_frozen = declared_frozen or any(score is not None for score in frozen)
    if has_frozen:
        if not missing and all(score is not None for score in frozen):
            report["frozen_grid_basis_equal"] = all(score["basis"] == frozen[0]["basis"] for score in frozen)
            report["comparison_pre_metrics"] = {
                mode: score["pre_weighted_rms_mean_db"]
                for mode, score in report["frozen_grid_scores"].items()
            }
            values = list(report["comparison_pre_metrics"].values())
            report["comparison_pre_metrics_equal"] = all(value == values[0] for value in values)
            if not report["frozen_grid_basis_equal"]:
                report["notes"].append("frozen baseline, target, grid, or level references differ across modes")
        else:
            report["notes"].append("complete supported frozen-grid evidence unavailable for some modes")
        comparable = report["frozen_grid_basis_equal"] and report["comparison_pre_metrics_equal"]
    else:
        comparable = report["pre_metrics_equal"]
        report["notes"].append("legacy metric comparison; no frozen-grid basis verified")
    if (comparable and report["shared_playback_controls_equal"]
            and report["pre_metrics_equal"] is not None):
        report["verdict"] = "pass"
    report["hybrid_best_part"] = hybrid_best_part_check(present, report)
    hybrid = report["hybrid_best_part"]
    if hybrid["required_for_accepted_hybrid"] and hybrid["passed"] is not True:
        report["verdict"] = "fail"
        report["notes"].append("accepted hybrid does not prove target error at least as good as the best standalone part")
    report["notes"].append(
        "config equality does not verify external measurement bytes, optimizer "
        "budget parity, filter-family equivalence, or physical capture conditions"
    )
    return report


# Mirror scripts/test_roomeq_measured.sh so the expected matrix is what was
# requested, not every fixture present. A saved run manifest takes precedence.
DEFAULT_SCENARIOS = (
    "2.2_unknown 2.2_sigberg1 2.2_sigberg2 2.2_sigberg3 2.2_genelec "
    "2.0_8361a 2.0_d3v 2.0_fidelia 2.0_t7v_2024 2.0_t7v_2026 "
    "5.0_genelec 5.1_kef 5.1.4_genelec 5.0.4_genelec 2.0_ascilab1 2.0_ascilab2 2.0_ascilab3 "
    "2.0_gen8340a_p1_badroom 2.0_kefls60_p1_badroom 2.1_kefls60_p2_badroom "
    "2.2_genelec8340a_4m_p1_badroom 2.2_genelec8340a_4m_p2_badroom 2.2_kefls60_p2_badroom"
)
DEFAULT_MODES = "iir fir mixed mixed-phase"


def expected_matrix(args) -> tuple[list[str], list[str]]:
    manifest = load_json(pathlib.Path(args.out_dir) / "manifest.json") or {}

    def requested(key, default):
        explicit = getattr(args, key) or os.environ.get(key.upper())
        if explicit:
            return explicit.split()
        saved = manifest.get(key)
        if saved is not None:
            if (not isinstance(saved, list) or not saved
                    or any(not isinstance(v, str) or not v for v in saved)):
                raise ValueError(f"invalid {key} in run manifest")
            return saved
        return default.split()

    scenarios = requested("scenarios", DEFAULT_SCENARIOS)
    modes = requested("modes", DEFAULT_MODES)
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
    try:
        scenarios, modes = expected_matrix(args)
    except ValueError as error:
        print(str(error), file=sys.stderr)
        return 2
    entries = [combo_status(out_dir, s, m) for s in scenarios for m in modes]
    counts: dict[str, int] = {}
    for entry in entries:
        counts[entry["status"]] = counts.get(entry["status"], 0) + 1
    summary = {
        "generated_by": "scripts/roomeq_suite_report.py",
        "generated_by_sha256": REPORT_SOURCE_SHA256,
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
        "generated_by_sha256": REPORT_SOURCE_SHA256,
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
            want_basis = want.get("score_basis", "native_marginal_legacy")
            if want_basis == "native_marginal_legacy":
                got_imp = (entry.get("native_marginal_metrics") or {}).get("improvement_db")
            elif want_basis == entry.get("score_basis"):
                got_imp = entry.get("improvement_db")
            else:
                drifts.append(f"{key}: score_basis {want_basis} -> {entry.get('score_basis')}")
                continue
            want_imp = want.get("improvement_db")
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
    stage_failures = [k for k, e in summary["combinations"].items()
                      if any(code not in (None, 0) for code in (e["stages"] or {}).values())]
    if missing or fails or stage_failures:
        print(f"CHECK FAIL: {len(missing)} errors, {len(fails)} mismatches, "
              f"{len(stage_failures)} combinations with stage failures")
        return 2
    print("CHECK PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
