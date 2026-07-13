"""Real ByteTrack integration (DECISIONS.md D3).

D3 originally deferred ByteTrack because the reference implementation only
ever shipped as an un-packaged research repo — no clean `pip install`. That
gap is now closed for real: `supervision` (Roboflow's own library, MIT
license, actively maintained — the same org behind RF-DETR) ships a proper,
pip-installable reimplementation of the actual ByteTrack algorithm.

`sv.ByteTrack` is flagged deprecated as of supervision 0.28 (removed in
0.30) in favor of a renamed API upstream — imported here from its stable
submodule path to avoid the deprecation warning noise; the underlying class
and algorithm are unchanged.

Exposes the exact same `update(detections) -> list[Track]` interface as
`edge/tracker.py CentroidTracker`, so this is a real drop-in swap, not a
parallel implementation — `edge/pipeline.py` doesn't need to know which one
it's holding.
"""

from __future__ import annotations

import time

import numpy as np
from supervision.detection.core import Detections as SVDetections
from supervision.tracker.byte_tracker.core import ByteTrack as _SVByteTrack

from config import settings
from edge.detector import Detection
from edge.tracker import Track


class ByteTrackTracker:
    def __init__(self):
        self._bytetrack = _SVByteTrack()
        self._label_by_class_id: dict[int, str] = {}
        self._class_id_by_label: dict[str, int] = {}
        self._tracks: dict[int, Track] = {}

    def _class_id_for(self, label: str) -> int:
        if label not in self._class_id_by_label:
            new_id = len(self._class_id_by_label)
            self._class_id_by_label[label] = new_id
            self._label_by_class_id[new_id] = label
        return self._class_id_by_label[label]

    def update(self, detections: list[Detection]) -> list[Track]:
        now = time.time()

        if detections:
            xyxy = np.array(
                [[x, y, x + w, y + h] for x, y, w, h in (d.box for d in detections)], dtype=np.float32
            )
            confidence = np.array([d.confidence for d in detections], dtype=np.float32)
            class_id = np.array([self._class_id_for(d.label) for d in detections], dtype=int)
        else:
            xyxy = np.zeros((0, 4), dtype=np.float32)
            confidence = np.zeros((0,), dtype=np.float32)
            class_id = np.zeros((0,), dtype=int)

        sv_detections = SVDetections(xyxy=xyxy, confidence=confidence, class_id=class_id)
        tracked = self._bytetrack.update_with_detections(sv_detections)

        seen_ids = set()
        for i in range(len(tracked)):
            track_id = int(tracked.tracker_id[i])
            label = self._label_by_class_id[int(tracked.class_id[i])]
            x1, y1, x2, y2 = tracked.xyxy[i]
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            seen_ids.add(track_id)

            if track_id not in self._tracks:
                self._tracks[track_id] = Track(
                    track_id=track_id, label=label, centroid=(cx, cy), first_seen=now, last_seen=now
                )
            else:
                t = self._tracks[track_id]
                t.centroid = (cx, cy)
                t.last_seen = now
                t.age_frames_missed = 0

        for track_id, t in self._tracks.items():
            if track_id not in seen_ids:
                t.age_frames_missed += 1

        stale = [tid for tid, t in self._tracks.items() if t.age_frames_missed > settings.track_max_age_frames]
        for tid in stale:
            del self._tracks[tid]

        return list(self._tracks.values())


def make_tracker():
    if settings.tracker_backend == "centroid":
        from edge.tracker import CentroidTracker

        return CentroidTracker()
    if settings.tracker_backend == "bytetrack":
        return ByteTrackTracker()
    raise ValueError(f"Unknown SENTINEL_TRACKER_BACKEND: {settings.tracker_backend!r}")
