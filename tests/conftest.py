"""Shared fixtures and path setup for tests."""

import sys
from pathlib import Path

# Add the repo root to sys.path so 'motion_analytics' is importable
_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))
