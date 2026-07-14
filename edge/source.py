"""Video source abstraction — webcam or RTSP/ONVIF IP camera.

Swap via SENTINEL_SOURCE_KIND.
"""

from __future__ import annotations

from typing import Iterator

import cv2
import numpy as np

from config import settings


class VideoSource:
    """Common interface every source implements: iterate BGR frames."""

    def frames(self) -> Iterator[np.ndarray]:
        raise NotImplementedError

    def release(self) -> None:
        raise NotImplementedError


class OpenCVSource(VideoSource):
    """Wraps cv2.VideoCapture for both webcam indices and RTSP URLs."""

    def __init__(self, target: int | str):
        self._target = target
        self._cap = cv2.VideoCapture(target)
        if not self._cap.isOpened():
            raise RuntimeError(f"Could not open video source: {target!r}")

    def frames(self) -> Iterator[np.ndarray]:
        while True:
            ok, frame = self._cap.read()
            if not ok:
                break
            yield frame

    def release(self) -> None:
        self._cap.release()


def make_source() -> VideoSource:
    """Factory driven entirely by config — see SENTINEL_SOURCE_KIND in .env.example."""
    if settings.source_kind == "rtsp":
        if not settings.rtsp_url:
            raise ValueError("SENTINEL_SOURCE_KIND=rtsp requires SENTINEL_RTSP_URL to be set")
        return OpenCVSource(settings.rtsp_url)
    if settings.source_kind == "webcam":
        return OpenCVSource(settings.webcam_index)
    raise ValueError(f"Unknown SENTINEL_SOURCE_KIND: {settings.source_kind!r}")
