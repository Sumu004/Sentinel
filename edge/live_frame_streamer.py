"""Streams the annotated frame to the dashboard on a background thread.

Holds only the single most recent frame (no queue) and pushes it to
PUT /live-frame at its own pace, independent of the camera's frame rate.
"""

from __future__ import annotations

import logging
import threading
import time

import cv2
import numpy as np
import requests

from config import settings

logger = logging.getLogger(__name__)


class LiveFrameStreamer:
    def __init__(self, max_fps: float = 8.0):
        self._min_interval_s = 1.0 / max_fps
        self._lock = threading.Lock()
        self._latest_frame: np.ndarray | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True, name="live-frame-streamer")
        self._thread.start()

    def update(self, frame: np.ndarray) -> None:
        """Called from the frame loop — must stay O(1), never do I/O here."""
        with self._lock:
            self._latest_frame = frame

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)

    def _run(self) -> None:
        base_url = f"http://{settings.api_host}:{settings.api_port}"
        headers = {"Authorization": f"Bearer {settings.api_token}"} if settings.api_token else {}

        while not self._stop.is_set():
            start = time.time()
            with self._lock:
                frame = self._latest_frame

            if frame is not None:
                ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                if ok:
                    try:
                        requests.put(
                            f"{base_url}/live-frame",
                            data=buf.tobytes(),
                            headers={**headers, "Content-Type": "application/octet-stream"},
                            timeout=2,
                        )
                    except requests.RequestException as exc:
                        logger.debug("Live-frame push failed (non-fatal): %s", exc)

            elapsed = time.time() - start
            time.sleep(max(0.0, self._min_interval_s - elapsed))
