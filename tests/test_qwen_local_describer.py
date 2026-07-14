"""QwenLocalDescriber calls a real Ollama server over HTTP. Tests mock
requests.post.
"""

import base64
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

from reasoning.describe import QwenLocalDescriber, _encode_frame_jpeg_base64


def _frame() -> np.ndarray:
    return np.zeros((64, 64, 3), dtype=np.uint8)


def test_describe_requires_a_frame():
    describer = QwenLocalDescriber(endpoint="http://127.0.0.1:11434", model="qwen2.5vl:3b")
    with pytest.raises(ValueError, match="requires a frame"):
        describer.describe("person", 5.0, frame=None)


def test_describe_posts_expected_payload_and_parses_response():
    describer = QwenLocalDescriber(endpoint="http://127.0.0.1:11434", model="qwen2.5vl:3b")
    mock_response = MagicMock()
    mock_response.json.return_value = {"response": "A person stands near the front door at night."}
    mock_response.raise_for_status.return_value = None

    with patch("reasoning.describe.requests.post", return_value=mock_response) as mock_post:
        result = describer.describe("person", 5.0, frame=_frame())

    assert result.text == "A person stands near the front door at night."
    assert result.backend == "qwen-local"
    assert result.severity == "medium"

    called_url, called_kwargs = mock_post.call_args[0][0], mock_post.call_args[1]
    assert called_url == "http://127.0.0.1:11434/api/generate"
    payload = called_kwargs["json"]
    assert payload["model"] == "qwen2.5vl:3b"
    assert payload["stream"] is False
    assert len(payload["images"]) == 1
    assert isinstance(payload["images"][0], str) and len(payload["images"][0]) > 0
    assert payload["options"]["num_predict"] == 80
    assert payload["options"]["num_ctx"] == 2048
    assert payload["keep_alive"] == "30m"


def test_encode_frame_downscales_large_frames():
    large_frame = np.zeros((1080, 810, 3), dtype=np.uint8)
    encoded = _encode_frame_jpeg_base64(large_frame, max_dim=512)
    decoded = cv2.imdecode(np.frombuffer(base64.b64decode(encoded), dtype=np.uint8), cv2.IMREAD_COLOR)
    assert max(decoded.shape[:2]) <= 512


def test_encode_frame_leaves_small_frames_unchanged():
    small_frame = np.zeros((100, 80, 3), dtype=np.uint8)
    encoded = _encode_frame_jpeg_base64(small_frame, max_dim=512)
    decoded = cv2.imdecode(np.frombuffer(base64.b64decode(encoded), dtype=np.uint8), cv2.IMREAD_COLOR)
    assert decoded.shape[:2] == (100, 80)


def test_describe_raises_clearly_when_ollama_unreachable():
    import requests

    describer = QwenLocalDescriber(endpoint="http://127.0.0.1:11434", model="qwen2.5vl:3b")
    with patch("reasoning.describe.requests.post", side_effect=requests.ConnectionError("refused")):
        with pytest.raises(RuntimeError, match="could not reach Ollama"):
            describer.describe("person", 5.0, frame=_frame())


def test_describe_raises_on_empty_response():
    describer = QwenLocalDescriber(endpoint="http://127.0.0.1:11434", model="qwen2.5vl:3b")
    mock_response = MagicMock()
    mock_response.json.return_value = {"response": ""}
    mock_response.raise_for_status.return_value = None

    with patch("reasoning.describe.requests.post", return_value=mock_response):
        with pytest.raises(RuntimeError, match="empty response"):
            describer.describe("person", 5.0, frame=_frame())
