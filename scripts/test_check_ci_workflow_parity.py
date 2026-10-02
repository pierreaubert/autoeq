from __future__ import annotations

import unittest

from scripts.check_ci_workflow_parity import check_workflow_parity


GITEA_HEADER = """# Mirror of .github/workflows/ci.yml for Gitea Actions.
# Keep the job/step bodies in sync; only the header below is Gitea-specific.
# Requires on the Gitea host: an Actions runner with the `ubuntu-latest` label
# (plus a self-hosted `macos-15` runner for the macOS job), `actions/*` fetched
# from github.com, and cache + artifact storage enabled (Gitea >= 1.22).
"""


class WorkflowParityTests(unittest.TestCase):
    def test_accepts_matching_body_and_gitea_header(self) -> None:
        self.assertEqual(check_workflow_parity("name: CI\njobs:\n", GITEA_HEADER + "name: CI\njobs:\n"), [])

    def test_ignores_trailing_whitespace_only(self) -> None:
        self.assertEqual(check_workflow_parity("name: CI  \n", GITEA_HEADER + "name: CI\n"), [])

    def test_rejects_different_body(self) -> None:
        errors = check_workflow_parity("jobs:\n  qa: {}\n", GITEA_HEADER + "jobs:\n")
        self.assertTrue(errors)
        self.assertIn("line 2", errors[0])

    def test_rejects_missing_header(self) -> None:
        errors = check_workflow_parity("name: CI\n", "name: CI\n")
        self.assertEqual(len(errors), 1)
        self.assertIn("documented mirror header", errors[0])


if __name__ == "__main__":
    unittest.main()
