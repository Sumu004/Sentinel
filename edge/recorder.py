"""Clip recording with a pre-event ring buffer.

Captures SENTINEL_PRE_EVENT_SECONDS before the event fired, plus
SENTINEL_POST_EVENT_SECONDS after.
"""

from __future__ import annotations

from collections import deque
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from config import settings


class RingBufferRecorder:
    def __init__(self):
        maxlen = max(1, int(settings.capture_fps * settings.pre_event_seconds))
        self._buffer: deque[np.ndarray] = deque(maxlen=maxlen)
        self._post_frames_needed = 0
        self._writer: cv2.VideoWriter | None = None
        self._writer_path: Path | None = None
        self._last_finished: Path | None = None

    def push_frame(self, frame: np.ndarray) -> None:
        """Call once per frame, every frame, regardless of whether an event is active."""
        self._buffer.append(frame.copy())
        if self._writer is not None:
            self._writer.write(frame)
            self._post_frames_needed -= 1
            if self._post_frames_needed <= 0:
                self._finish()

    def trigger(self, label: str) -> None:
        """Call when an event fires. Flushes the pre-event buffer into a new clip
        and keeps recording for SENTINEL_POST_EVENT_SECONDS more frames.
        """
        if self._writer is not None:
            self._post_frames_needed = int(settings.capture_fps * settings.post_event_seconds)
            return

        settings.clips_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = settings.clips_dir / f"{label}_{timestamp}.mp4"

        h, w = self._buffer[-1].shape[:2] if self._buffer else (480, 640)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._writer = cv2.VideoWriter(str(path), fourcc, settings.capture_fps, (w, h))
        for buffered_frame in self._buffer:
            self._writer.write(buffered_frame)
        self._writer_path = path
        self._post_frames_needed = int(settings.capture_fps * settings.post_event_seconds)

    def _finish(self) -> None:
        assert self._writer is not None and self._writer_path is not None
        self._writer.release()
        self._last_finished = self._writer_path
        self._writer = None
        self._writer_path = None

    def pop_finished_clip(self) -> Path | None:
        """Returns the path of a just-completed clip exactly once, then None
        until the next clip finishes.
        """
        path, self._last_finished = self._last_finished, None
        return path

    @property
    def is_recording(self) -> bool:
        return self._writer is not None
