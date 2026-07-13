from __future__ import annotations

from pydantic import BaseModel, Field


class EventIn(BaseModel):
    event_id: str
    site_id: str
    label: str
    track_id: int
    started_at: str
    detected_at: str
    org_id: str = "default"
    description: str | None = None
    severity: str | None = None


class EventOut(EventIn):
    assigned: bool = Field(default=False)
