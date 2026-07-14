from __future__ import annotations

import logging
import queue
import threading

import numpy as np

from edge.cloud_client import send_description_update
from edge.events import Event
from reasoning.describe import Describer

logger = logging.getLogger(__name__)


class DescriptionWorker:
    def __init__(self, describer: Describer, maxsize: int = 100, stop_timeout_s: float = 20.0):
        self._describer = describer
        self._queue: queue.Queue[tuple[Event, float, np.ndarray] | None] = queue.Queue(maxsize=maxsize)
        self._thread: threading.Thread | None = None
        self._stop_timeout_s = stop_timeout_s

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True, name="description-worker")
        self._thread.start()

    def enqueue(self, event: Event, duration_s: float, frame: np.ndarray) -> None:
        try:
            self._queue.put_nowait((event, duration_s, frame))
        except queue.Full:
            logger.warning("Description worker queue full — dropping enrichment for event %s", event.event_id)

    def stop(self) -> None:
        self._queue.put_nowait(None)
        if self._thread is not None:
            self._thread.join(timeout=self._stop_timeout_s)
            if self._thread.is_alive():
                logger.warning(
                    "Description worker still running an enrichment after %.0fs shutdown wait — "
                    "that one description will be lost.",
                    self._stop_timeout_s,
                )

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                return
            event, duration_s, frame = item
            try:
                result = self._describer.describe(event.label, duration_s, frame=frame)
                send_description_update(event.event_id, result.text, result.severity)
            except Exception:
                logger.exception("Description enrichment failed for event %s", event.event_id)
