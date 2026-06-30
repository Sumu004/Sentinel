"""Posts events from the edge to the cloud backend (cloud/backend/app.py).

Talks to a plain local FastAPI route, not AWS API Gateway — see DECISIONS.md D8
on why API Gateway's free tier is the one to avoid. The request shape is
identical regardless of what's running behind SENTINEL_API_HOST/PORT, so
pointing this at a real deployed backend later is a config change.
"""

from __future__ import annotations

import logging

import requests

from config import settings
from edge.events import Event

logger = logging.getLogger(__name__)


def _base_url() -> str:
    return f"http://{settings.api_host}:{settings.api_port}"


def _headers() -> dict:
    if settings.api_token:
        return {"Authorization": f"Bearer {settings.api_token}"}
    return {}


def send_event(event: Event) -> bool:
    """Best-effort send. Returns False on failure instead of raising — a
    network blip must not crash the edge loop. Phase 2.3 replaces this with a
    real store-and-forward queue so failed sends are retried, not dropped.
    """
    try:
        response = requests.post(
            f"{_base_url()}/events",
            json={
                "event_id": event.event_id,
                "site_id": event.site_id,
                "label": event.label,
                "track_id": event.track_id,
                "started_at": event.started_at,
                "detected_at": event.detected_at,
            },
            headers=_headers(),
            timeout=5,
        )
        response.raise_for_status()
        return True
    except requests.RequestException as exc:
        logger.warning("Failed to send event %s to backend: %s", event.event_id, exc)
        return False
