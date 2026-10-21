#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)
sibling_root=$(dirname "$repo_root")
export GIT_TERMINAL_PROMPT=0

# Gitea provides the GitHub-compatible server/owner environment variables.
# Private feature revisions exist on that server; public CI retains its pins.
ci_server=${GITHUB_SERVER_URL:-https://github.com}
ci_owner=${GITHUB_REPOSITORY_OWNER:-pierreaubert}
if [[ "$ci_server" == https://github.com ]]; then
    sibling_remote_base=https://github.com/pierreaubert
    math_revision=4b434f999d47d625fceafed8a46c87d47f4817de
    sofa_revision=15e78a7af254914787c6e66c617373d77d72e313
    toolkit_revision=29d29a9465510ba4925e84e9c0bc88b1257caf0c
else
    if [[ ! "$ci_server" =~ ^https?://[^/]+/?$ ]] ||
        [[ ! "$ci_owner" =~ ^[A-Za-z0-9_.-]+$ ]]; then
        echo "Invalid private CI sibling origin" >&2
        exit 1
    fi
    sibling_remote_base=${ci_server%/}/$ci_owner
    # Includes the reusable FIR designer API required by RoomEQ.
    math_revision=d2907882a52adbbeaf9f9749e8a6cd12b0847c22
    sofa_revision=9b290c6b9384831bf2e6b76e005beff755fe32ca
    toolkit_revision=a1989afc2b070081763f9f888735ec51ee8a82b0
fi

checkout_sibling() {
    local name=$1
    local revision=$2
    local destination="$sibling_root/$name"
    local remote="$sibling_remote_base/$name.git"

    if [[ -L "$destination" ]]; then
        echo "Sibling checkout is a symlink: $destination" >&2
        return 1
    fi
    if [[ -e "$destination" ]]; then
        if [[ ! -d "$destination" ]] ||
            [[ $(git -C "$destination" rev-parse --show-toplevel 2>/dev/null || true) != "$destination" ]] ||
            [[ $(git -C "$destination" remote get-url origin 2>/dev/null || true) != "$remote" ]]; then
            echo "Unexpected sibling checkout: $destination" >&2
            return 1
        fi
    else
        git init -q "$destination"
        git -C "$destination" remote add origin "$remote"
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

checkout_sibling math-audio "$math_revision"
checkout_sibling sofa-reader "$sofa_revision"
checkout_sibling gpui-toolkit "$toolkit_revision"
