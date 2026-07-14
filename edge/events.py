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
    description: str | None = None
    severity: str | None = None


def debounce(tracks: list[Track]) -> list[Event]:
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
