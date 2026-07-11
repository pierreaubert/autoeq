#!/usr/bin/env bash
set -euo pipefail

config="${1:?expected PipeWire filter-chain configuration path}"
runtime_dir="$(mktemp -d)"
config_dir="$(mktemp -d)"
log_file="$(mktemp)"
pipewire_pid=""

cleanup() {
    if [[ -n "$pipewire_pid" ]]; then
        kill "$pipewire_pid" 2>/dev/null || true
        wait "$pipewire_pid" 2>/dev/null || true
    fi
    rm -rf "$runtime_dir" "$config_dir" "$log_file"
}
trap cleanup EXIT

export XDG_RUNTIME_DIR="$runtime_dir"
# The export is a pipewire.conf.d fragment, not a complete daemon config. Use
# PipeWire's stock configuration for its protocol/core modules and add the
# generated filter-chain as a temporary drop-in.
mkdir -p "$config_dir/pipewire.conf.d"
cp /usr/share/pipewire/pipewire.conf "$config_dir/pipewire.conf"
cp "$config" "$config_dir/pipewire.conf.d/90-roomeq.conf"
export PIPEWIRE_CONFIG_DIR="$config_dir"
pipewire >"$log_file" 2>&1 &
pipewire_pid=$!

for _ in $(seq 1 50); do
    # The daemon needs PIPEWIRE_CONFIG_DIR for our temporary drop-in, but the
    # client needs its stock client.conf rather than that daemon-only directory.
    if env -u PIPEWIRE_CONFIG_DIR pw-cli info 0 >/dev/null 2>&1; then
        exit 0
    fi
    if ! kill -0 "$pipewire_pid" 2>/dev/null; then
        cat "$log_file" >&2
        exit 1
    fi
    sleep 0.1
done

echo "PipeWire did not accept the generated filter-chain configuration." >&2
cat "$log_file" >&2
exit 1
