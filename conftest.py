"""Pytest bootstrapping: make the repository root importable.

The console-script `pytest` entry point does not put the invocation
directory on `sys.path`, so the `from scripts.…` imports used across
`scripts/test_*.py` fail under bare `pytest`. The `unittest`-based
recipes and `python -m pytest` already have the root on the path, where
this insert is a harmless no-op.
"""

import sys
from pathlib import Path

ROOT = str(Path(__file__).resolve().parent)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
