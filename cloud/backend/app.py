"""Local-first cloud backend (DECISIONS.md D8).

Replaces two things from the original MVP at once:

1. `Control Station/app.py`, which polled `requests.get('Replace with your API
   URL')` — a literal placeholder — every 10 seconds against a Lambda that
   doesn't exist yet in dev. This app *is* the API; nothing to poll, nothing to
   configure before it runs.
2. AWS API Gateway, whose free tier is 12-months-only and bills from request one
   after that (see DECISIONS.md D8). FastAPI run locally is the free substitute;
   Lambda/DynamoDB stay available behind SENTINEL_STORAGE_BACKEND=dynamodb
   since both of *those* are permanently free at dev scale.

Also fixes the original `/assign/<cid>` endpoint, which had zero
authentication — anyone reaching the Flask app could mark evidence as
assigned. Here, write endpoints require a bearer token when
SENTINEL_API_TOKEN is set.
"""

from __future__ import annotations

from fastapi import Depends, FastAPI, Header, HTTPException

from cloud.backend.db import EventRecord, make_store
from cloud.backend.schemas import EventIn, EventOut
from config import settings

app = FastAPI(title="Sentinel local backend")
_store = make_store()


def require_token(authorization: str | None = Header(default=None)) -> None:
    if not settings.api_token:
        return  # no token configured — fine for local dev, set one before exposing this
    expected = f"Bearer {settings.api_token}"
    if authorization != expected:
        raise HTTPException(status_code=401, detail="Missing or invalid bearer token")


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "site_id": settings.site_id, "storage_backend": settings.storage_backend}


@app.post("/events", response_model=EventOut, dependencies=[Depends(require_token)])
def create_event(event: EventIn) -> EventOut:
    record = EventRecord(**event.model_dump())
    _store.save(record)
    return EventOut(**event.model_dump(), assigned=False)


@app.get("/events", response_model=list[EventOut])
def list_events(limit: int = 100) -> list[EventOut]:
    return [EventOut(**vars(r)) for r in _store.list_recent(limit=limit)]


@app.post("/events/{event_id}/assign", dependencies=[Depends(require_token)])
def assign_event(event_id: str) -> dict:
    ok = _store.assign(event_id)
    if not ok:
        raise HTTPException(status_code=404, detail=f"No event with id {event_id}")
    return {"status": "assigned", "event_id": event_id}
