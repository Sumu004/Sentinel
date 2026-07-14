import time
from unittest.mock import patch

import numpy as np

from edge.description_worker import DescriptionWorker
from edge.events import Event
from reasoning.describe import Describer, EventDescription


def _event() -> Event:
    return Event(
        event_id="e1",
        site_id="dev-site-01",
        label="person",
        track_id=1,
        started_at="2026-01-01T00:00:00+00:00",
        detected_at="2026-01-01T00:00:03+00:00",
    )


class _SlowFakeDescriber(Describer):
    def __init__(self):
        self.calls = []

    def describe(self, label, duration_s, context_reason="", frame=None):
        self.calls.append((label, duration_s, frame is not None))
        return EventDescription(text="a rich real description", severity="high", backend="qwen-local")


def test_enqueue_does_not_block_caller():
    describer = _SlowFakeDescriber()
    worker = DescriptionWorker(describer)
    worker.start()
    try:
        start = time.time()
        worker.enqueue(_event(), 5.0, np.zeros((10, 10, 3), dtype=np.uint8))
        elapsed = time.time() - start
        assert elapsed < 0.1
    finally:
        worker.stop()


def test_worker_calls_describer_and_sends_update():
    describer = _SlowFakeDescriber()
    worker = DescriptionWorker(describer)
    worker.start()

    with patch("edge.description_worker.send_description_update") as mock_send:
        worker.enqueue(_event(), 5.0, np.zeros((10, 10, 3), dtype=np.uint8))
        worker.stop()

    assert len(describer.calls) == 1
    assert describer.calls[0][0] == "person"
    assert describer.calls[0][2] is True
    mock_send.assert_called_once_with("e1", "a rich real description", "high")


def test_worker_survives_describer_exception():
    class _BrokenDescriber(Describer):
        def describe(self, label, duration_s, context_reason="", frame=None):
            raise RuntimeError("Ollama unreachable")

    worker = DescriptionWorker(_BrokenDescriber())
    worker.start()
    with patch("edge.description_worker.send_description_update") as mock_send:
        worker.enqueue(_event(), 5.0, np.zeros((10, 10, 3), dtype=np.uint8))
        worker.stop()

    mock_send.assert_not_called()


def test_queue_full_drops_without_raising():
    describer = _SlowFakeDescriber()
    worker = DescriptionWorker(describer, maxsize=0)
    worker.enqueue(_event(), 5.0, np.zeros((10, 10, 3), dtype=np.uint8))
