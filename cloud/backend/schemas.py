from __future__ import annotations

from pydantic import BaseModel, Field


class EventIn(BaseModel):
    event_id: str
    site_id: str
    label: str
    track_id: int
    started_at: str
    detected_at: str
    # Phase 2.5 multi-tenant field — every event belongs to an org, defaulting
    # to "default" until a real multi-tenant deployment assigns sites to orgs.
    org_id: str = "default"
    # Phase 2.4 reasoning fields — populated by reasoning/describe.py on the
    # edge; optional so older/simpler senders (e.g. curl, tests) still work.
    description: str | None = None
    severity: str | None = None


class EventOut(EventIn):
    assigned: bool = Field(default=False)
