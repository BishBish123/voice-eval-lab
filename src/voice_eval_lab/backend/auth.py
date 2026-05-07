"""Bearer-token middleware for the voice-eval-lab backend.

When ``BACKEND_AUTH_TOKEN`` is set, every request must supply a matching
``Authorization: Bearer <token>`` header.  When unset, auth is bypassed with a
startup warning (development mode).
"""

from __future__ import annotations

import logging
import os

from fastapi import HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

logger = logging.getLogger(__name__)

_bearer = HTTPBearer(auto_error=False)

_AUTH_TOKEN: str | None = None
_AUTH_ENABLED: bool = False


def configure_auth(token: str | None) -> None:
    """Configure the auth middleware from the given token value.

    Called once at app startup.  When *token* is falsy, auth is disabled with a
    log warning.
    """
    global _AUTH_TOKEN, _AUTH_ENABLED
    if token:
        _AUTH_TOKEN = token
        _AUTH_ENABLED = True
    else:
        _AUTH_TOKEN = None
        _AUTH_ENABLED = False
        logger.warning(
            "BACKEND_AUTH_TOKEN is not set — authentication is DISABLED. "
            "Set BACKEND_AUTH_TOKEN before deploying to a shared environment."
        )


async def require_auth(request: Request) -> None:
    """FastAPI dependency that enforces bearer-token auth when enabled.

    When auth is disabled (no ``BACKEND_AUTH_TOKEN``), this is a no-op.
    When auth is enabled, returns 401 on missing/invalid token.
    """
    if not _AUTH_ENABLED:
        return

    credentials: HTTPAuthorizationCredentials | None = await _bearer(request)
    if credentials is None or credentials.credentials != _AUTH_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid or missing bearer token")


def load_auth_from_env() -> str | None:
    """Read ``BACKEND_AUTH_TOKEN`` from the environment."""
    return os.environ.get("BACKEND_AUTH_TOKEN") or None


__all__ = ["configure_auth", "load_auth_from_env", "require_auth"]
