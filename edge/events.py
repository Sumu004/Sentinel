"""Turns persistent tracks into debounced events.

A track only becomes an Event once it has existed for SENTINEL_EVENT_MIN_DURATION_S
(default 3s) and only emits once per track — this is what keeps "one intruder for
8 seconds" from becoming hundreds of duplicate alerts.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone

from config import settings
from edge.tracker import Track


@dataclass(frozen=True)
class Event:
    event_id: str
    site_id: str
    label: str
    track_id: int
    started_at: str
    detected_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


def debounce(tracks: list[Track]) -> list[Event]:
    """Call once per frame with the current tracks. Returns newly-qualified events
    (a track that just crossed the minimum-duration threshold) and marks them
    emitted so they fire exactly once.
    """
    new_events: list[Event] = []
    for track in tracks:
        if track.emitted:
            continue
        if track.duration_s < settings.event_min_duration_s:
            continue
        track.emitted = True
        new_events.append(
            Event(
                event_id=str(uuid.uuid4()),
                site_id=settings.site_id,
                label=track.label,
                track_id=track.track_id,
                started_at=datetime.fromtimestamp(track.first_seen, tz=timezone.utc).isoformat(),
            )
        )
    return new_events
