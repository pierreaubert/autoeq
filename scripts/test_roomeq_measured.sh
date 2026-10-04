#!/usr/bin/env bash
set -euo pipefail

# Use `mbx` when installed, else plain `cargo` (mirrors the justfile).
if [ -z "${CARGO:-}" ]; then
    if command -v mbx >/dev/null 2>&1; then CARGO=mbx; else CARGO=cargo; fi
fi

# Run from the repository root even when invoked from another directory.
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
IN=${IN:-./data_tests/roomeq/measured}
OUT=${OUT:-./data_generated/roomeq/measured}
LOG=${LOG:-warn}
# Space-separated subsets support the plan's one-case-at-a-time audit.
SCENARIOS=${SCENARIOS:-'2.2_unknown 2.2_sigberg1 2.2_sigberg2 2.2_sigberg3 2.2_genelec 2.0_8361a 2.0_d3v 2.0_fidelia 2.0_t7v_2024 2.0_t7v_2026  5.0_genelec 5.1_kef 5.1.4_genelec 2.0_ascilab1 2.0_ascilab2 2.0_ascilab3'}
MODES=${MODES:-'iir fir mixed mixed-phase'}
read -r -a scenarios <<< "$SCENARIOS"
read -r -a modes <<< "$MODES"

# Fresh run directories: refuse to mix new results with stale artifacts
# unless reuse is explicitly allowed.
if [ -d "$OUT" ] && [ -n "$(ls -A "$OUT" 2>/dev/null)" ] && [ -z "${ROOMEQ_ALLOW_REUSE_OUT:-}" ]; then
    echo "Refusing to reuse non-empty OUT=$OUT (stale artifacts look like current failures)." >&2
    echo "Set ROOMEQ_ALLOW_REUSE_OUT=1 to append, or point OUT at a fresh directory." >&2
    exit 1
fi
mkdir -p "$OUT"

if [[ -z ${PYTHON:-} ]]; then
    # A linked Git worktree can reuse the checkout's existing plot environment.
    shared_venv="$(git rev-parse --git-common-dir)/../venv/bin/python3"
    if [[ -x ./venv/bin/python3 ]]; then
        PYTHON=./venv/bin/python3
    elif [[ -x "$shared_venv" ]]; then
        PYTHON="$shared_venv"
    else
        PYTHON=python3
    fi
fi
"$PYTHON" -c 'import numpy' || {
    echo 'RoomEQ plots require numpy; set PYTHON to an interpreter with it.' >&2
    exit 1
}

# Validate the entire requested subset before expensive optimization.
for scenario in "${scenarios[@]}"; do
    [[ "$scenario" =~ ^[a-zA-Z0-9_.-]+$ ]] && [[ "$scenario" != .* ]] || {
        echo "Invalid measured scenario: $scenario" >&2; exit 1;
    }
    for mode in "${modes[@]}"; do
        case "$mode" in iir|fir|mixed|mixed-phase) ;; *)
            echo "Invalid measured mode: $mode" >&2; exit 1 ;; esac
        for config in "$IN/$scenario/recordings.json" "$IN/$scenario/optimiser-$mode.json"; do
            [[ -f "$config" ]] || { echo "Missing measured config: $config" >&2; exit 1; }
        done
    done
done

$CARGO build --release --locked --features cli --bin roomeq
BIN=${CARGO_TARGET_DIR:-./target}/release/roomeq

# WP1 run manifest preamble: identify this run before any optimization.
RUN_ID=$(date -u +%Y%m%dT%H%M%SZ)-$$
if command -v sha256sum >/dev/null 2>&1; then
    BIN_SHA=$(sha256sum "$BIN" | awk '{print $1}')
else
    BIN_SHA=$(shasum -a 256 "$BIN" | awk '{print $1}')
fi
GIT_REV=$(git rev-parse HEAD 2>/dev/null || echo unknown)
if [ -z "$(git status --porcelain 2>/dev/null)" ]; then GIT_DIRTY=false; else GIT_DIRTY=true; fi
STARTED_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)
RUN_ID="$RUN_ID" BIN_SHA="$BIN_SHA" GIT_REV="$GIT_REV" GIT_DIRTY="$GIT_DIRTY" \
STARTED_UTC="$STARTED_UTC" SCENARIOS="$SCENARIOS" MODES="$MODES" BIN_PATH="$BIN" \
    "$PYTHON" - "$OUT/manifest.json" "$0" "$@" <<'PYEOF'
import json, os, sys
manifest = {
    "run_id": os.environ["RUN_ID"],
    "started_utc": os.environ["STARTED_UTC"],
    "command": sys.argv[2:],
    "scenarios": os.environ["SCENARIOS"].split(),
    "modes": os.environ["MODES"].split(),
    "roomeq_bin": os.environ["BIN_PATH"],
    "roomeq_sha256": os.environ["BIN_SHA"],
    "git_revision": os.environ["GIT_REV"],
    "git_dirty": os.environ["GIT_DIRTY"] == "true",
    "finished_utc": None,
    "suite_exit": None,
}
with open(sys.argv[1], "w", encoding="utf-8") as handle:
    json.dump(manifest, handle, indent=2, sort_keys=True)
    handle.write("\n")
print(f"RoomEQ run manifest: {sys.argv[1]} (run {manifest['run_id']})")
PYEOF
# REW captures are tracked, but their derived CSV directories are ignored.
# Prepare missing derivatives in fresh worktrees without overwriting an
# existing capture export (including any user edits to that export).
for scenario in "${scenarios[@]}"; do
    for capture in "$IN/$scenario"/*.mdat; do
        [[ -f "$capture" ]] || continue
        "$PYTHON" ./utils/mdat2csv.py "$capture" --no-clobber
    done
    for mode in "${modes[@]}"; do
        mode_out="$OUT/$scenario/$mode"
        mkdir -p "$mode_out"
        if ! RUST_LOG="$LOG" "$BIN" \
            --config "$IN/$scenario/recordings.json" \
            --override-config "$IN/$scenario/optimiser-$mode.json" \
            --output "$mode_out/dsp-$mode.json" --dry-run \
            > "$mode_out/config-validation.log" 2>&1; then
            cat "$mode_out/config-validation.log" >&2
            exit 1
        fi
    done
done

failures=()
for scenario in "${scenarios[@]}"; do
    results=()
    for mode in "${modes[@]}"; do
        # Each mode owns its WAVs and manifest; later modes cannot overwrite
        # sidecars referenced by earlier exported graphs.
        mode_out="$OUT/$scenario/$mode"
        mkdir -p "$mode_out"
        result="$mode_out/dsp-$mode.json"
        echo "=== RoomEQ measured: $scenario / $mode ==="
        # A rejected (or otherwise failing) mode must not abort the
        # remaining runs: record it and continue to the end.
        run_status=0
        RUST_LOG="$LOG" "$BIN" \
            --config "$IN/$scenario/recordings.json" \
            --override-config "$IN/$scenario/optimiser-$mode.json" \
            --output "$result" 2>&1 | tee "$mode_out/run.log" || run_status=${PIPESTATUS[0]}
        if (( run_status != 0 )); then
            echo "--- FAILED RoomEQ measured: $scenario / $mode (roomeq exit $run_status) ---" | tee -a "$mode_out/run.log"
            failures+=("$scenario/$mode: roomeq exit $run_status")
            continue
        fi
        if ! "$PYTHON" ./ui/display-roomeq "$result" \
            --output "$mode_out/dsp-$mode.html"; then
            failures+=("$scenario/$mode: display failed")
            continue
        fi
        if ! "$PYTHON" ./scripts/check_roomeq_measured_result.py "$result" \
            | tee "$mode_out/audit.jsonl"; then
            failures+=("$scenario/$mode: audit failed")
            continue
        fi
        results+=("$result")
    done
    if (( ${#results[@]} > 1 )); then
        # Recheck all earlier FIRs after the final mode has written its assets.
        if ! "$PYTHON" ./scripts/check_roomeq_measured_result.py "${results[@]}"; then
            failures+=("$scenario: cross-mode recheck failed")
        fi
        if ! "$PYTHON" ./ui/display-roomeq --compare "${results[@]}" \
            --output "$OUT/$scenario/compare.html"; then
            failures+=("$scenario: compare failed")
        fi
    fi
done
# WP1: always publish machine-readable reports, even on failure.
if ! "$PYTHON" ./scripts/roomeq_suite_report.py "$OUT"; then
    failures+=("suite: report generation failed")
fi
FINISHED_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)
if (( ${#failures[@]} > 0 )); then SUITE_EXIT=1; else SUITE_EXIT=0; fi
"$PYTHON" - "$OUT/manifest.json" "$FINISHED_UTC" "$SUITE_EXIT" <<'PYEOF'
import json, sys
with open(sys.argv[1], encoding="utf-8") as handle:
    manifest = json.load(handle)
manifest["finished_utc"] = sys.argv[2]
manifest["suite_exit"] = int(sys.argv[3])
with open(sys.argv[1], "w", encoding="utf-8") as handle:
    json.dump(manifest, handle, indent=2, sort_keys=True)
    handle.write("\n")
PYEOF
if (( ${#failures[@]} > 0 )); then
    echo "=== RoomEQ measured failures (${#failures[@]}) ===" >&2
    for failure in "${failures[@]}"; do
        echo "  - $failure" >&2
    done
    exit 1
fi
echo "=== RoomEQ measured: all runs passed ==="
