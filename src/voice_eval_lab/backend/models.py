"""Pydantic models for the voice-eval-lab backend API."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel


class CreateSessionRequest(BaseModel):
    """Request body for POST /sessions."""

    user_id: str
    scenario_id: str | None = None


class SessionResponse(BaseModel):
    """Response body for POST /sessions."""

    session_id: str
    livekit_token: str | None
    expires_at: str  # ISO-8601 timestamp


class SessionState(BaseModel):
    """Full session state returned by GET /sessions/{id}."""

    session_id: str
    user_id: str
    scenario_id: str | None
    started_at: datetime
    ended_at: datetime | None


__all__ = ["CreateSessionRequest", "SessionResponse", "SessionState"]
