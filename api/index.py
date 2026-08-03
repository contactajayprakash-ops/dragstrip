"""Vercel entry point — the whole app is the FastAPI instance in server.py."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import env_local  # noqa: F401 — deployment-only credentials, absent in git
except ImportError:
    pass

from server import app  # noqa: E402,F401
