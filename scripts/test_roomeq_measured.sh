#!/usr/bin/env bash
set -euo pipefail

# Run from the repository root even when invoked from another directory.
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
IN=${IN:-./data_tests/roomeq/measured}
OUT=${OUT:-./data_generated/roomeq/measured}
LOG=${LOG:-warn}
# Space-separated subsets support the plan's one-case-at-a-time audit.
SCENARIOS=${SCENARIOS:-'2.2_unknown 2.2_sigberg1 2.2_sigberg2 2.2_sigberg3 2.2_genelec 2.0_8361a 2.0_d3v 2.0_fidelia 2.0_t7v 5.0_genelec 5.1_kef 5.1.4_genelec'}
MODES=${MODES:-'iir fir mixed mixed-phase'}
read -r -a scenarios <<< "$SCENARIOS"
read -r -a modes <<< "$MODES"

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
"$PYTHON" -c 'import numpy, plotly' || {
    echo 'RoomEQ plots require numpy and plotly; set PYTHON to an interpreter with both.' >&2
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

cargo build --release --locked --features cli --bin roomeq
BIN=${CARGO_TARGET_DIR:-./target}/release/roomeq
# REW captures are tracked, but their derived CSV directories are ignored.
# Prepare missing derivatives in fresh worktrees without overwriting an
# existing capture export (including any user edits to that export).
for scenario in "${scenarios[@]}"; do
    for capture in "$IN/$scenario"/*.mdat; do
        [[ -f "$capture" ]] || continue
        "$PYTHON" ./scripts/mdat2csv.py "$capture" --no-clobber
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

for scenario in "${scenarios[@]}"; do
    results=()
    for mode in "${modes[@]}"; do
        # Each mode owns its WAVs and manifest; later modes cannot overwrite
        # sidecars referenced by earlier exported graphs.
        mode_out="$OUT/$scenario/$mode"
        mkdir -p "$mode_out"
        result="$mode_out/dsp-$mode.json"
        echo "=== RoomEQ measured: $scenario / $mode ==="
        RUST_LOG="$LOG" "$BIN" \
            --config "$IN/$scenario/recordings.json" \
            --override-config "$IN/$scenario/optimiser-$mode.json" \
            --output "$result" 2>&1 | tee "$mode_out/run.log"
        "$PYTHON" ./scripts/display-roomeq.py "$result" \
            --output "$mode_out/dsp-$mode.html"
        "$PYTHON" ./scripts/check_roomeq_measured_result.py "$result" \
            | tee "$mode_out/audit.jsonl"
        results+=("$result")
    done
    if (( ${#results[@]} > 1 )); then
        # Recheck all earlier FIRs after the final mode has written its assets.
        "$PYTHON" ./scripts/check_roomeq_measured_result.py "${results[@]}"
        "$PYTHON" ./scripts/display-roomeq.py --compare "${results[@]}" \
            --output "$OUT/$scenario/compare.html"
    fi
done
