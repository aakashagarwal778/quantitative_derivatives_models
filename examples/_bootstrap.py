"""Allow running example scripts directly from the repository root or file path.

This file prepends the local ``src`` directory to ``sys.path`` when the package
has not yet been installed in editable mode.
"""
from __future__ import annotations
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / 'src'
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
