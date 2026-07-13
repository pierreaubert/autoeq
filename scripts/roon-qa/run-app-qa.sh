#!/usr/bin/env bash
set -euo pipefail

[[ "$(uname -s)" == "Darwin" ]] || { echo "Roon application QA requires macOS." >&2; exit 1; }
qa_home="${ROOMEQ_ROON_QA_HOME:-$HOME/Library/Application Support/SotF/RoonExportQA}"
config="$qa_home/config.env"
[[ -f "$config" ]] || { echo "Run 'just qa-export-roon-setup' first." >&2; exit 1; }
source "$config"
manifest="${ROOMEQ_ROON_MANIFEST:-}"
[[ -f "$manifest" ]] || { echo "Set ROOMEQ_ROON_MANIFEST to an exported Roon JSON manifest." >&2; exit 1; }

repo_root="$(cd "$(dirname "$0")/../.." && pwd)"
python="${ROOMEQ_PYTHON:-$repo_root/venv/bin/python}"
export NODE_PATH="$qa_home/extension/node_modules"
transport=(node "$repo_root/scripts/roon-qa/roon_transport.js")
ui="$qa_home/bin/roon-ui"
capture_dir="$qa_home/captures/current"
diagnostic="$qa_home/diagnostics/roon-qa-$(date -u +%Y%m%dT%H%M%SZ).json"
readback_file="$qa_home/tmp/ui-readback.json"
input="${ROOMEQ_BLACKHOLE_FFMPEG_INPUT:-:BlackHole 64ch}"

cleanup() {
  "${transport[@]}" stop >/dev/null 2>&1 || true
  if [[ -n "${ROOMEQ_ROON_UI_APPLY_CMD:-}" ]]; then
    "$ROOMEQ_ROON_UI_APPLY_CMD" --zone "$ROOMEQ_ROON_QA_ZONE" --mode cleanup \
      --manifest "$manifest" --convolution "$(dirname "$manifest")/room_eq_convolution.zip" \
      >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT INT TERM

[[ -n "${ROOMEQ_ROON_UI_APPLY_CMD:-}" ]] || {
  echo "Set ROOMEQ_ROON_UI_APPLY_CMD to the private, version-calibrated UI action executable." >&2
  exit 2
}
"${transport[@]}" zones >/dev/null
"$ui" assert-zone "$ROOMEQ_ROON_QA_ZONE" >/dev/null
mkdir -p "$capture_dir"
chmod 700 "$capture_dir"

capture_mode() {
  local mode="$1"
  "$ROOMEQ_ROON_UI_APPLY_CMD" --zone "$ROOMEQ_ROON_QA_ZONE" --mode "$mode" \
    --manifest "$manifest" --convolution "$(dirname "$manifest")/room_eq_convolution.zip"
  "$ui" assert-zone "$ROOMEQ_ROON_QA_ZONE" >/dev/null
  "${transport[@]}" stop >/dev/null 2>&1 || true
  ffmpeg -hide_banner -loglevel error -y -f avfoundation -i "$input" \
    -t 10 -ar 48000 -ac 2 -c:a pcm_f32le "$capture_dir/$mode.wav" &
  local capture_pid=$!
  sleep 1
  "${transport[@]}" play >/dev/null
  wait "$capture_pid"
  "${transport[@]}" stop >/dev/null
}

for mode in baseline iir fir combined; do capture_mode "$mode"; done
"$ui" readback "$ROOMEQ_ROON_QA_ZONE" > "$readback_file"
roon_version="$(mdls -raw -name kMDItemVersion /Applications/Roon.app)"
"$python" "$repo_root/scripts/roon-qa/verify_capture.py" \
  --manifest "$manifest" --captures "$capture_dir" --output "$diagnostic" \
  --roon-version "$roon_version" --zone "$ROOMEQ_ROON_QA_ZONE" \
  --ui-readback "$readback_file"
chmod 600 "$diagnostic"
cleanup
trap - EXIT INT TERM
echo "Roon application QA passed. Private textual diagnostics: $diagnostic"
