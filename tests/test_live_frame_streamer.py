"""LiveFrameStreamer must never block the frame loop — update() is O(1), and
the actual HTTP push happens on its own thread at its own pace. Mocks
requests.put (same pattern as the Ollama/DynamoDB tests) so this stays a
fast, offline unit test.
"""

import time
from unittest.mock import patch

import numpy as np

from edge.live_frame_streamer import LiveFrameStreamer


def test_update_does_not_block_caller():
    streamer = LiveFrameStreamer(max_fps=2.0)
    with patch("edge.live_frame_streamer.requests.put"):
        streamer.start()
        try:
            start = time.time()
            streamer.update(np.zeros((10, 10, 3), dtype=np.uint8))
            elapsed = time.time() - start
            assert elapsed < 0.05
        finally:
            streamer.stop()


def test_streamer_pushes_encoded_jpeg_bytes():
    streamer = LiveFrameStreamer(max_fps=20.0)
    with patch("edge.live_frame_streamer.requests.put") as mock_put:
        streamer.start()
        streamer.update(np.zeros((10, 10, 3), dtype=np.uint8))
        time.sleep(0.2)
        streamer.stop()

    assert mock_put.called
    _, kwargs = mock_put.call_args
    assert kwargs["data"][:2] == b"\xff\xd8"


def test_streamer_survives_request_exception():
    import requests

    streamer = LiveFrameStreamer(max_fps=20.0)
    with patch("edge.live_frame_streamer.requests.put", side_effect=requests.ConnectionError("refused")):
        streamer.start()
        streamer.update(np.zeros((10, 10, 3), dtype=np.uint8))
        time.sleep(0.2)
        streamer.stop()
