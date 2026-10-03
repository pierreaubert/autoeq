"""Require the Gitea CI workflow body to mirror GitHub Actions."""

from __future__ import annotations

import pathlib
import sys


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
GITHUB_WORKFLOW = REPO_ROOT / ".github/workflows/ci.yml"
GITEA_WORKFLOW = REPO_ROOT / ".gitea/workflows/ci.yml"
GITEA_HEADER = (
    "# Mirror of .github/workflows/ci.yml for Gitea Actions.",
    "# Keep the job/step bodies in sync; only the header below is Gitea-specific.",
    "# Requires on the Gitea host: an Actions runner with the `ubuntu-latest` label",
    "# (plus a self-hosted `macos-15` runner for the macOS job), `actions/*` fetched",
    "# from github.com, and cache + artifact storage enabled (Gitea >= 1.22).",
)


def normalized_lines(contents: str) -> list[str]:
    return [line.rstrip() for line in contents.splitlines()]


def check_workflow_parity(
    github_contents: str,
    gitea_contents: str,
) -> list[str]:
    github_lines = normalized_lines(github_contents)
    gitea_lines = normalized_lines(gitea_contents)
    if tuple(gitea_lines[: len(GITEA_HEADER)]) != GITEA_HEADER:
        return ["Gitea workflow does not start with the documented mirror header"]
    gitea_body = gitea_lines[len(GITEA_HEADER) :]
    if github_lines == gitea_body:
        return []

    limit = min(len(github_lines), len(gitea_body))
    mismatch = next(
        (index for index in range(limit) if github_lines[index] != gitea_body[index]),
        limit,
    )
    github_line = github_lines[mismatch] if mismatch < len(github_lines) else "<EOF>"
    gitea_line = gitea_body[mismatch] if mismatch < len(gitea_body) else "<EOF>"
    return [
        "GitHub/Gitea workflow bodies differ at line "
        f"{mismatch + 1}: GitHub={github_line!r}, Gitea={gitea_line!r}"
    ]


def main() -> int:
    errors = check_workflow_parity(
        GITHUB_WORKFLOW.read_text(encoding="utf-8"),
        GITEA_WORKFLOW.read_text(encoding="utf-8"),
    )
    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    print("GitHub and Gitea workflow bodies match")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
