#!/usr/bin/env bash
set -euo pipefail

# Use `mbx` when installed, else plain `cargo` (mirrors the justfile).
if [ -z "${CARGO:-}" ]; then
    if command -v mbx >/dev/null 2>&1; then CARGO=mbx; else CARGO=cargo; fi
fi

# `cargo test` treats an empty filter selection as success. Probe the exact
# library selection first so this Linux integration recipe cannot go green if
# the PipeWire tests are renamed or removed. Keep Cargo failures distinct from
# an empty selection, especially when a container's dependency download fails.
test_list="$(mktemp)"
trap 'rm -f "$test_list"' EXIT
if ! $CARGO test --release -p roomeq-export --lib pipewire -- --list >"$test_list"; then
    echo "Could not discover PipeWire export tests." >&2
    exit 1
fi
if ! grep -q ': test$' "$test_list"; then
    echo "No PipeWire export tests matched the expected library selection." >&2
    exit 1
fi

$CARGO test --release -p roomeq-export --lib pipewire -- --nocapture
