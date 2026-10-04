#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)
sibling_root=$(dirname "$repo_root")
export GIT_TERMINAL_PROMPT=0

checkout_sibling() {
    local name=$1
    local revision=$2
    local destination="$sibling_root/$name"

    if [[ -L "$destination" ]]; then
        echo "Sibling checkout is a symlink: $destination" >&2
        return 1
    fi
    if [[ -e "$destination" ]]; then
        if [[ ! -d "$destination" ]] ||
            [[ $(git -C "$destination" rev-parse --show-toplevel 2>/dev/null || true) != "$destination" ]] ||
            [[ $(git -C "$destination" remote get-url origin 2>/dev/null || true) != "https://github.com/pierreaubert/$name.git" ]]; then
            echo "Unexpected sibling checkout: $destination" >&2
            return 1
        fi
    else
        git init -q "$destination"
        git -C "$destination" remote add origin "https://github.com/pierreaubert/$name.git"
        git -C "$destination" fetch --depth=1 origin "$revision"
        git -C "$destination" checkout --detach -q FETCH_HEAD
    fi

    if [[ $(git -C "$destination" rev-parse HEAD) != "$revision" ]]; then
        echo "Sibling revision mismatch: $name" >&2
        return 1
    fi
    if [[ -n $(git -C "$destination" status --porcelain) ]]; then
        echo "Sibling checkout is dirty: $name" >&2
        return 1
    fi
    echo "$name pinned at $revision"
}

checkout_sibling math-audio 4b434f999d47d625fceafed8a46c87d47f4817de
checkout_sibling sofa-reader 15e78a7af254914787c6e66c617373d77d72e313
checkout_sibling gpui-toolkit 29d29a9465510ba4925e84e9c0bc88b1257caf0c
