#!/usr/bin/env bash
set -euo pipefail

# `cargo test` treats an empty filter selection as success. Probe the exact
# library selection first so this Linux integration recipe cannot go green if
# the PipeWire tests are renamed or removed. Keep Cargo failures distinct from
# an empty selection, especially when a container's dependency download fails.
test_list="$(mktemp)"
trap 'rm -f "$test_list"' EXIT
if ! cargo test -p autoeq --lib pipewire -- --list >"$test_list"; then
    echo "Could not discover PipeWire export tests." >&2
    exit 1
fi
if ! grep -q ': test$' "$test_list"; then
    echo "No PipeWire export tests matched the expected library selection." >&2
    exit 1
fi

cargo test -p autoeq --lib pipewire -- --nocapture
