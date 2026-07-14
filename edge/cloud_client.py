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
    if send_event(event):
        return True
    outbox.enqueue(event)
    logger.info("Event %s queued to outbox (network unavailable)", event.event_id)
    return False


def send_description_update(event_id: str, description: str, severity: str) -> bool:
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
