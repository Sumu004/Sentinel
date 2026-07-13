"""Posts events from the edge to the cloud backend (cloud/backend/app.py).

Talks to a plain local FastAPI route, not AWS API Gateway — see DECISIONS.md D8
on why API Gateway's free tier is the one to avoid. The request shape is
identical regardless of what's running behind SENTINEL_API_HOST/PORT, so
pointing this at a real deployed backend later is a config change.

Phase 2.3: a failed send now queues in edge/outbox.py instead of being
dropped — see send_event_or_queue and edge/pipeline.py's periodic retry.
"""

from __future__ import annotations

import logging

import requests

from config import settings
from edge.events import Event
from edge.outbox import Outbox

logger = logging.getLogger(__name__)


def _base_url() -> str:
    return f"http://{settings.api_host}:{settings.api_port}"


def _headers() -> dict:
    if settings.api_token:
        return {"Authorization": f"Bearer {settings.api_token}"}
    return {}


def send_payload(payload: dict) -> bool:
    """Posts a raw event payload dict. Returns False on any network failure —
    never raises, since a network blip must not crash the edge loop.
    """
    try:
        response = requests.post(
            f"{_base_url()}/events", json=payload, headers=_headers(), timeout=5
        )
        response.raise_for_status()
        return True
    except requests.RequestException as exc:
        logger.warning("Failed to send event %s to backend: %s", payload.get("event_id"), exc)
        return False


def send_event(event: Event) -> bool:
    payload = {
        "event_id": event.event_id,
        "site_id": event.site_id,
        "label": event.label,
        "track_id": event.track_id,
        "started_at": event.started_at,
        "detected_at": event.detected_at,
        "description": event.description,
        "severity": event.severity,
    }
    return send_payload(payload)


def send_event_or_queue(event: Event, outbox: Outbox) -> bool:
    """Try to send immediately; if that fails, queue it in the outbox so a
    later retry_pending() call picks it up instead of the event being lost.
    """
    if send_event(event):
        return True
    outbox.enqueue(event)
    logger.info("Event %s queued to outbox (network unavailable)", event.event_id)
    return False


def send_description_update(event_id: str, description: str, severity: str) -> bool:
    """PATCHes a richer description onto an already-sent event. Used by
    edge/description_worker.py once a slow VLM call finishes — the event
    itself was already sent immediately with a fast template description so
    alerting isn't blocked on this.
    """
    try:
        response = requests.patch(
            f"{_base_url()}/events/{event_id}/description",
            json={"description": description, "severity": severity},
            headers=_headers(),
            timeout=5,
        )
        response.raise_for_status()
        return True
    except requests.RequestException as exc:
        logger.warning("Failed to send description update for event %s: %s", event_id, exc)
        return False


def send_heartbeat() -> bool:
    """Pings the backend so it knows this site is alive. A stopped heartbeat
    is itself an alarm (VISION.md "silence is an alarm") — see
    cloud/backend/app.py's /heartbeat and /sites/status endpoints.
    """
    try:
        response = requests.post(
            f"{_base_url()}/heartbeat",
            json={"site_id": settings.site_id},
            headers=_headers(),
            timeout=5,
        )
        response.raise_for_status()
        return True
    except requests.RequestException as exc:
        logger.warning("Heartbeat failed: %s", exc)
        return False
