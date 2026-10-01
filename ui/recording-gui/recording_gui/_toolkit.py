"""Locate the gpui-toolkit Python package.

Prefers an installed ``gpui_toolkit`` wheel; falls back to the sibling
``gpui-toolkit`` checkout (``../gpui-toolkit/crates/gpui-python-runtime/python``)
so the GUIs run straight from an ``all_of_sotf`` checkout without install.
"""
from __future__ import annotations

import sys
from pathlib import Path


def sibling_toolkit_dir() -> Path | None:
    """Return the sibling checkout's ``python/`` dir when it exists."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = (
            parent / "gpui-toolkit" / "crates" / "gpui-python-runtime" / "python"
        )
        if (candidate / "gpui_toolkit" / "__init__.py").is_file():
            return candidate
    return None


def ensure_toolkit() -> None:
    """Make ``import gpui_toolkit`` work or raise with install help."""
    try:
        import gpui_toolkit  # noqa: F401
        return
    except ImportError:
        pass
    sibling = sibling_toolkit_dir()
    if sibling is not None:
        sys.path.insert(0, str(sibling))
    try:
        import gpui_toolkit  # noqa: F401
    except ImportError as error:
        raise RuntimeError(
            "gpui_toolkit is not importable. Install the wheel "
            "(`pip install gpui-toolkit`) or run from an all_of_sotf "
            "checkout with the sibling ../gpui-toolkit repository present."
        ) from error
