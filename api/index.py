"""Vercel entry point — the whole app is the FastAPI instance in server.py.

Vercel's rewrite hands the function its own path (/api/index) instead of the
one the browser asked for, so the rewrite smuggles the original path through
a __path query param and this shim puts it back before FastAPI routes.
"""

import sys
from pathlib import Path
from urllib.parse import parse_qsl, urlencode

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import env_local  # noqa: F401 — deployment-only credentials, absent in git
except ImportError:
    pass

from server import app as fastapi_app  # noqa: E402


class RestoreOriginalPath:
    def __init__(self, inner):
        self.inner = inner

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            pairs = parse_qsl(scope.get("query_string", b"").decode(),
                              keep_blank_values=True)
            path, rest = None, []
            for k, v in pairs:
                if k == "__path":
                    path = v
                else:
                    rest.append((k, v))
            if path is not None:
                scope = dict(scope)
                scope["path"] = path or "/"
                scope["raw_path"] = (path or "/").encode()
                scope["query_string"] = urlencode(rest).encode()
        await self.inner(scope, receive, send)


app = RestoreOriginalPath(fastapi_app)
