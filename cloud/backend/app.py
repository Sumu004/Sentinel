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

import asyncio
import json
import subprocess
import sys
import threading
import time
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import HTMLResponse, Response, StreamingResponse

from cloud.backend.db import EventRecord, make_store
from cloud.backend.notifications import NotificationEngine, NotificationPayload
from cloud.backend.schemas import EventIn, EventOut
from config import settings


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    yield
    with _pipeline_lock:
        if _pipeline_process is not None and _pipeline_process.poll() is None:
            _pipeline_process.terminate()


app = FastAPI(title="Sentinel local backend", lifespan=_lifespan)
_store = make_store()
_notifier = NotificationEngine()
_subscribers: list[asyncio.Queue] = []
_DASHBOARD_HTML = (Path(__file__).parent / "static" / "dashboard.html").read_text()

_pipeline_lock = threading.Lock()
_pipeline_process: subprocess.Popen | None = None
_pipeline_log_lines: deque[str] = deque(maxlen=500)


def _drain_pipeline_output(proc: subprocess.Popen) -> None:
    """Runs on its own thread for the lifetime of the pipeline subprocess —
    continuously reads stdout into a bounded deque so GET /pipeline/logs can
    do a non-destructive tail read at any time, instead of a one-shot drain
    that would lose lines between polls.
    """
    if proc.stdout is None:
        return
    for line in proc.stdout:
        _pipeline_log_lines.append(line.rstrip("\n"))

_latest_frame_lock = threading.Lock()
_latest_frame: bytes | None = None
_latest_frame_at: float = 0.0


def require_token(authorization: str | None = Header(default=None)) -> None:
    if not settings.api_token:
        return
    expected = f"Bearer {settings.api_token}"
    if authorization != expected:
        raise HTTPException(status_code=401, detail="Missing or invalid bearer token")


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "site_id": settings.site_id, "storage_backend": settings.storage_backend}


@app.get("/", response_class=HTMLResponse)
def dashboard() -> str:
    """Single live dashboard: a browser webcam preview (visual only — real
    detection runs from edge/main.py on the machine with the trained model
    and physical camera) plus real-time events/descriptions/logs streamed
    from GET /events and GET /events/stream.
    """
    return _DASHBOARD_HTML


@app.post("/events", response_model=EventOut, dependencies=[Depends(require_token)])
def create_event(event: EventIn) -> EventOut:
    record = EventRecord(**event.model_dump())
    _store.save(record)
    out = EventOut(**event.model_dump(), assigned=False)

    _notifier.notify(
        NotificationPayload(
            event_id=event.event_id,
            site_id=event.site_id,
            label=event.label,
            severity=event.severity or "medium",
            description=event.description or f"{event.label} detected at {event.site_id}",
        )
    )

    for queue in _subscribers:
        queue.put_nowait(out.model_dump())

    return out


@app.get("/events", response_model=list[EventOut])
def list_events(limit: int = 100, org_id: str | None = None) -> list[EventOut]:
    return [EventOut(**vars(r)) for r in _store.list_recent(limit=limit, org_id=org_id)]


@app.get("/events/stream")
async def stream_events():
    """Server-Sent Events — a connected client gets pushed new events as they
    happen, instead of polling. Replaces the original MVP's
    `apscheduler`-driven 10-second poll against a placeholder URL.
    """

    async def event_generator():
        queue: asyncio.Queue = asyncio.Queue()
        _subscribers.append(queue)
        try:
            while True:
                event_data = await queue.get()
                yield f"data: {json.dumps(event_data)}\n\n"
        finally:
            _subscribers.remove(queue)

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.post("/events/{event_id}/assign", dependencies=[Depends(require_token)])
def assign_event(event_id: str) -> dict:
    ok = _store.assign(event_id)
    if not ok:
        raise HTTPException(status_code=404, detail=f"No event with id {event_id}")
    return {"status": "assigned", "event_id": event_id}


@app.patch("/events/{event_id}/description", dependencies=[Depends(require_token)])
def update_event_description(event_id: str, payload: dict) -> dict:
    """Backs the async VLM-enrichment path (edge/description_worker.py): the
    event already exists (created with a fast template description so
    alerting isn't blocked), and this arrives later with a richer
    description once the slower VLM call finishes.
    """
    description = payload.get("description")
    severity = payload.get("severity")
    if not description or not severity:
        raise HTTPException(status_code=400, detail="description and severity are required")
    if not hasattr(_store, "update_description"):
        raise HTTPException(status_code=501, detail="Description updates require the sqlite backend")

    ok = _store.update_description(event_id, description, severity)
    if not ok:
        raise HTTPException(status_code=404, detail=f"No event with id {event_id}")

    for queue in _subscribers:
        queue.put_nowait({"event_id": event_id, "description": description, "severity": severity, "updated": True})

    return {"status": "updated", "event_id": event_id}


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


@app.put("/live-frame", dependencies=[Depends(require_token)])
async def put_live_frame(request: Request) -> dict:
    """edge/live_frame_streamer.py PUTs the latest JPEG-encoded, detection-box
    -annotated frame here at a throttled rate — see that module for why this
    is decoupled from the frame loop (the same reason edge/description_worker.py
    exists: nothing that can stall or be slow may sit inline in _process_frame).
    """
    global _latest_frame, _latest_frame_at
    body = await request.body()
    if not body:
        raise HTTPException(status_code=400, detail="Empty frame body")
    with _latest_frame_lock:
        _latest_frame = body
        _latest_frame_at = time.time()
    return {"status": "ok"}


@app.get("/live-frame")
def get_live_frame() -> Response:
    """Single latest annotated frame as a plain JPEG — useful for a quick
    manual check (curl/browser) without needing MJPEG support.
    """
    with _latest_frame_lock:
        frame = _latest_frame
    if frame is None:
        raise HTTPException(status_code=404, detail="No live frame yet — is edge.main running with live-frame streaming enabled?")
    return Response(content=frame, media_type="image/jpeg")


@app.get("/live-frame/stream")
async def stream_live_frame():
    """MJPEG stream (multipart/x-mixed-replace) — the dashboard's <img> tag
    consumes this directly, no client-side JS needed to decode frames.
    Polls the in-memory latest frame at a fixed rate; stale frames (older
    than a few seconds — no edge pipeline currently pushing) are skipped so
    the stream visibly stops rather than freezing on an ancient frame.
    """

    async def frame_generator():
        last_sent_at = 0.0
        while True:
            with _latest_frame_lock:
                frame, frame_at = _latest_frame, _latest_frame_at
            fresh = frame is not None and (time.time() - frame_at) < 5.0
            if fresh and frame_at != last_sent_at:
                last_sent_at = frame_at
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n"
                    b"Content-Length: " + str(len(frame)).encode() + b"\r\n\r\n"
                    + frame + b"\r\n"
                )
            await asyncio.sleep(0.08)

    return StreamingResponse(frame_generator(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.post("/pipeline/start", dependencies=[Depends(require_token)])
def start_pipeline() -> dict:
    """Spawns `python -m edge.main --no-preview` as a child of this server
    process, so the dashboard's Start button replaces a manual terminal
    command. See the module-level note above on the camera-permission
    limit this inherits.
    """
    global _pipeline_process
    with _pipeline_lock:
        if _pipeline_process is not None and _pipeline_process.poll() is None:
            return {"status": "already_running", "pid": _pipeline_process.pid}

        _pipeline_log_lines.clear()
        repo_root = Path(__file__).resolve().parent.parent.parent
        _pipeline_process = subprocess.Popen(
            [sys.executable, "-m", "edge.main", "--no-preview"],
            cwd=repo_root,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        threading.Thread(target=_drain_pipeline_output, args=(_pipeline_process,), daemon=True).start()
        return {"status": "started", "pid": _pipeline_process.pid}


@app.post("/pipeline/stop", dependencies=[Depends(require_token)])
def stop_pipeline() -> dict:
    global _pipeline_process
    with _pipeline_lock:
        if _pipeline_process is None or _pipeline_process.poll() is not None:
            return {"status": "not_running"}
        _pipeline_process.terminate()
        try:
            _pipeline_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            _pipeline_process.kill()
        return {"status": "stopped"}


@app.get("/pipeline/status")
def pipeline_status() -> dict:
    with _pipeline_lock:
        if _pipeline_process is None:
            return {"running": False}
        running = _pipeline_process.poll() is None
        return {"running": running, "pid": _pipeline_process.pid if running else None}


@app.get("/pipeline/logs")
def pipeline_logs(lines: int = 50) -> dict:
    """Tail of the pipeline subprocess's stdout/stderr — lets the dashboard
    show real errors (e.g. camera permission, missing model file) instead of
    a silent "not running" with no explanation. Non-destructive: reads from
    the bounded deque _drain_pipeline_output fills continuously, so repeated
    polls don't lose lines the way a one-shot pipe drain would.
    """
    return {"lines": list(_pipeline_log_lines)[-lines:]}


@app.post("/recordings/clear", dependencies=[Depends(require_token)])
def clear_recordings() -> dict:
    """Deletes recorded clip files (and their signed manifests) from
    settings.clips_dir, and clears the pipeline log buffer. Deliberately
    does NOT touch the events database or evidence/custody.py's
    chain-of-custody log — those are the tamper-evident integrity record,
    not disposable UI state, and clearing them silently would defeat the
    point of a custody chain. Refuses to run while the pipeline is actively
    recording, to avoid deleting a clip mid-write.
    """
    with _pipeline_lock:
        if _pipeline_process is not None and _pipeline_process.poll() is None:
            raise HTTPException(
                status_code=409,
                detail="Stop the pipeline before clearing recordings — it may be writing a clip right now.",
            )

    deleted = 0
    if settings.clips_dir.exists():
        for f in settings.clips_dir.iterdir():
            if f.is_file():
                f.unlink()
                deleted += 1

    _pipeline_log_lines.clear()

    return {"status": "cleared", "files_deleted": deleted}
