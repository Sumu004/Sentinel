"""Multi-object tracking + event debouncing.

D3 in DECISIONS.md targets ByteTrack. ByteTrack isn't pip-installable as a clean
library (it ships as a research repo) so it's deferred to Phase 2.1 alongside the
real detector. This module ships a small, dependency-free centroid tracker that
satisfies the same need right now: turn N raw detections per frame into a
*track* that persists across frames, so "one person for 8 seconds" becomes one
event instead of ~240 raw detections.

This directly fixes the bug in the original MVP's main.py, which appended every
matching frame individually with no tracking or debouncing at all.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

from config import settings
from edge.detector import Detection


@dataclass
class Track:
    track_id: int
    label: str
    centroid: tuple[float, float]
    first_seen: float
    last_seen: float
    age_frames_missed: int = 0
    emitted: bool = False

    @property
    def duration_s(self) -> float:
        return self.last_seen - self.first_seen


@dataclass
class CentroidTracker:
    """Nearest-centroid tracker. Good enough to debounce events; swap for
    ByteTrack in Phase 2.1 without touching anything downstream — both produce
    a list[Track] with the same fields used by events.py.
    """

    max_distance: float = 75.0
    tracks: dict[int, Track] = field(default_factory=dict)
    _next_id: int = 0

    def update(self, detections: list[Detection]) -> list[Track]:
        now = time.time()
        unmatched_tracks = set(self.tracks.keys())
        unmatched_detections = list(range(len(detections)))

        for det_idx in list(unmatched_detections):
            det = detections[det_idx]
            cx, cy = self._centroid(det)
            best_id, best_dist = None, self.max_distance
            for track_id in unmatched_tracks:
                tx, ty = self.tracks[track_id].centroid
                dist = math.hypot(cx - tx, cy - ty)
                if dist < best_dist:
                    best_id, best_dist = track_id, dist
            if best_id is not None:
                track = self.tracks[best_id]
                track.centroid = (cx, cy)
                track.last_seen = now
                track.age_frames_missed = 0
                unmatched_tracks.discard(best_id)
                unmatched_detections.remove(det_idx)

        for det_idx in unmatched_detections:
            det = detections[det_idx]
            cx, cy = self._centroid(det)
            track_id = self._next_id
            self._next_id += 1
            self.tracks[track_id] = Track(
                track_id=track_id, label=det.label, centroid=(cx, cy), first_seen=now, last_seen=now
            )

        for track_id in unmatched_tracks:
            self.tracks[track_id].age_frames_missed += 1

        stale = [
            tid for tid, t in self.tracks.items() if t.age_frames_missed > settings.track_max_age_frames
        ]
        for tid in stale:
            del self.tracks[tid]

        return list(self.tracks.values())

    @staticmethod
    def _centroid(det: Detection) -> tuple[float, float]:
        x, y, w, h = det.box
        return (x + w / 2, y + h / 2)
