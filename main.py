"""Vercel entrypoint: credentials first (deploy-only file), then the app."""

try:
    import env_local  # noqa: F401 — absent in git; present in deployments
except ImportError:
    pass

from server import app  # noqa: E402,F401
