"""Bootstrap helpers for Streamlit pages.

Streamlit executes page files from the webapp directory, so make the project root
importable before importing src.* modules.
"""
from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
WEBAPP_ROOT = PROJECT_ROOT / "webapp"


def ensure_project_imports() -> Path:
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    return PROJECT_ROOT
