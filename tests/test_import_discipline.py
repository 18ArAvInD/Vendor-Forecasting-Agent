"""Import-discipline guard: ``streamlit`` may live ONLY in ui/app.py.

This is a lightweight, defensive regression guard. It scans the package source
files as plain text (it does NOT import ``app.py`` or ``streamlit``, and does not
require streamlit to be installed). It asserts that the only module referencing
``import streamlit`` is ``ui/app.py``, and that ``ui/app.py`` never imports
``boto3``. If the source tree cannot be located for any reason, the test skips
cleanly rather than failing spuriously.
"""

import re
from pathlib import Path

import pytest

_SRC_ROOT = Path(__file__).resolve().parents[1] / "src" / "vendor_forecasting_agent"
_STREAMLIT_RE = re.compile(r"^\s*(?:import\s+streamlit|from\s+streamlit\b)", re.MULTILINE)
_BOTO3_IMPORT_RE = re.compile(r"^\s*(?:import\s+boto3|from\s+boto3\b)", re.MULTILINE)


def _python_files() -> list[Path]:
    if not _SRC_ROOT.is_dir():
        pytest.skip(f"source tree not found at {_SRC_ROOT}")
    return sorted(_SRC_ROOT.rglob("*.py"))


def test_streamlit_imported_only_in_app() -> None:
    """No module other than ui/app.py may import streamlit."""
    offenders: list[str] = []
    for path in _python_files():
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue
        if not _STREAMLIT_RE.search(text):
            continue
        rel = path.relative_to(_SRC_ROOT).as_posix()
        if rel != "ui/app.py":
            offenders.append(rel)

    assert offenders == [], (
        "streamlit must be imported only in ui/app.py; found in: " + ", ".join(offenders)
    )


def test_app_does_not_import_boto3() -> None:
    """ui/app.py must never import boto3 (it stays an optional AWS dependency)."""
    app_path = _SRC_ROOT / "ui" / "app.py"
    if not app_path.is_file():
        pytest.skip(f"app.py not found at {app_path}")
    text = app_path.read_text(encoding="utf-8")
    assert _BOTO3_IMPORT_RE.search(text) is None, "app.py must not import boto3"
