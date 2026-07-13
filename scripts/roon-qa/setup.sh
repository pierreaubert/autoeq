#!/usr/bin/env bash
set -euo pipefail

[[ "$(uname -s)" == "Darwin" ]] || { echo "Roon application QA requires macOS." >&2; exit 1; }
qa_home="${ROOMEQ_ROON_QA_HOME:-$HOME/Library/Application Support/SotF/RoonExportQA}"
qa_zone="${ROOMEQ_ROON_QA_ZONE:-SotF Roon QA}"
repo_root="$(cd "$(dirname "$0")/../.." && pwd)"
python="${ROOMEQ_PYTHON:-$repo_root/venv/bin/python}"

for path in /Applications/Roon.app /Applications/Xcode.app; do
  [[ -e "$path" ]] || { echo "Missing required application: $path" >&2; exit 1; }
done
for command in ffmpeg node npm swiftc; do
  command -v "$command" >/dev/null || { echo "Missing required command: $command" >&2; exit 1; }
done
[[ -x "$python" ]] || { echo "Python environment not found: $python" >&2; exit 1; }
"$python" -c 'import numpy, scipy' || { echo "QA Python needs numpy and scipy." >&2; exit 1; }
system_profiler SPAudioDataType 2>/dev/null | grep -q "BlackHole 64ch" || {
  echo "BlackHole 64ch was not found by macOS." >&2
  exit 1
}

umask 077
mkdir -p "$qa_home"/{bin,captures,diagnostics,extension,filters,music,tmp}
chmod 700 "$qa_home" "$qa_home"/*
printf 'ROOMEQ_ROON_QA_HOME=%q\nROOMEQ_ROON_QA_ZONE=%q\n' "$qa_home" "$qa_zone" > "$qa_home/config.env"

ffmpeg -hide_banner -loglevel error -y -f lavfi \
  -i "aevalsrc=exprs=0.005*sin(2*PI*30*(exp(log(600)*t/8)-1)/(log(600)/8)):s=48000:d=8" \
  -ac 2 -c:a pcm_f32le "$qa_home/music/sotf-roon-qa-sweep.wav"

npm install --ignore-scripts --no-audit --no-fund \
  --prefix "$qa_home/extension" "$repo_root/scripts/roon-qa" >/dev/null
swiftc "$repo_root/scripts/roon-qa/roon_ui.swift" -o "$qa_home/bin/roon-ui" \
  -framework Cocoa -framework ApplicationServices -framework Vision -framework ImageIO

cat <<EOF
Private QA state prepared at:
  $qa_home

Manual one-time steps (credentials and license state stay inside Roon):
  1. Open Roon and sign in or activate it manually.
  2. Create a zone named exactly '$qa_zone', backed by BlackHole 64ch at 48 kHz.
  3. Add '$qa_home/music' as a watched folder and queue sotf-roon-qa-sweep.wav.
  4. From '$qa_home', run the following and enable the extension in Roon when asked:
     NODE_PATH='$qa_home/extension/node_modules' ROOMEQ_ROON_QA_ZONE='$qa_zone' \\
       node '$repo_root/scripts/roon-qa/roon_transport.js' zones
  5. Grant Accessibility and Screen Recording permission to your terminal.

The authorization state remains below this mode-0700 directory. Login, license,
tokens, and screenshots are never written to the repository or printed.
EOF
