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

from pathlib import Path

from fastapi import Depends, FastAPI, File, HTTPException, Header, UploadFile
from fastapi.responses import HTMLResponse

from cloud.backend.db import EventRecord, make_store
from cloud.backend.schemas import EventIn, EventOut
from cloud.backend.vision import detect_and_annotate
from config import settings

app = FastAPI(title="Sentinel local backend")
_store = make_store()
_INDEX_HTML = (Path(__file__).parent / "static" / "index.html").read_text()


def require_token(authorization: str | None = Header(default=None)) -> None:
    if not settings.api_token:
        return  # no token configured — fine for local dev, set one before exposing this
    expected = f"Bearer {settings.api_token}"
    if authorization != expected:
        raise HTTPException(status_code=401, detail="Missing or invalid bearer token")


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "site_id": settings.site_id, "storage_backend": settings.storage_backend}


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    """Light-mode test UI: upload a photo, run the configured detector, see
    the annotated result. A demo/testing surface, not the live edge pipeline —
    see edge/main.py for the continuous-camera version.
    """
    return _INDEX_HTML


@app.post("/detect")
async def detect(file: UploadFile = File(...), conf: float = 0.4) -> dict:
    image_bytes = await file.read()
    try:
        return detect_and_annotate(image_bytes, conf=conf)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


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


@app.post("/heartbeat")
def heartbeat(payload: dict) -> dict:
    """A site pings this periodically (edge/cloud_client.py send_heartbeat).
    A site that stops pinging is flagged `silent` by /sites/status — VISION.md's
    "silence is an alarm": camera unplugged, tampered, or powered off all look
    identical from here, and all three deserve the same alert.
    """
    site_id = payload.get("site_id")
    if not site_id:
        raise HTTPException(status_code=400, detail="site_id is required")
    if not hasattr(_store, "record_heartbeat"):
        raise HTTPException(status_code=501, detail="Heartbeat tracking requires the sqlite backend")
    _store.record_heartbeat(site_id)
    return {"status": "ok", "site_id": site_id}


@app.get("/sites/status")
def sites_status() -> list[dict]:
    if not hasattr(_store, "site_statuses"):
        raise HTTPException(status_code=501, detail="Heartbeat tracking requires the sqlite backend")
    return _store.site_statuses(settings.heartbeat_silent_threshold_s)
