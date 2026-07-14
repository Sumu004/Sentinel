"""Event description.

Three backends behind one interface (SENTINEL_VLM_BACKEND):

- "template" (default) — deterministic, rule-based description from the
  event's own fields.
- "qwen-local" — Qwen2.5-VL via a local Ollama server.
- "frontier" — a hosted VLM (Claude/GPT-4o-class) for the escalation
  tier; requires an API key, not called automatically.

`Describer.describe()` takes an optional frame — qwen-local requires it.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
from datetime import datetime, timezone

import cv2
import numpy as np
import requests

from config import settings


@dataclass(frozen=True)
class EventDescription:
    text: str
    severity: str
    backend: str


class Describer:
    def describe(
        self, label: str, duration_s: float, context_reason: str = "", frame: np.ndarray | None = None
    ) -> EventDescription:
        raise NotImplementedError


_SEVERITY_BY_LABEL = {
    "person": "medium",
    "vehicle": "medium",
    "car": "medium",
    "bus": "medium",
    "package": "low",
    "animal": "low",
    "dog": "low",
    "cat": "low",
    "motion": "low",
}


class TemplateDescriber(Describer):
    """Deterministic, no dependencies — a rule-based description generator
    that works with zero setup.
    """

    def describe(
        self, label: str, duration_s: float, context_reason: str = "", frame: np.ndarray | None = None
    ) -> EventDescription:
        hour = datetime.now(timezone.utc).hour
        time_of_day = "overnight" if hour < 6 or hour >= 22 else ("daytime" if 6 <= hour < 18 else "evening")
        severity = _SEVERITY_BY_LABEL.get(label, "medium")

        if context_reason:
            text = f"{label.capitalize()} detected ({time_of_day}) — suppressed: {context_reason}."
            severity = "low"
        else:
            text = f"{label.capitalize()} detected and tracked for {duration_s:.0f}s ({time_of_day})."

        return EventDescription(text=text, severity=severity, backend="template")


def _encode_frame_jpeg_base64(frame: np.ndarray, max_dim: int = 512) -> str:
    """Downscales before encoding — the biggest lever on VLM latency. Capping
    the longer side at 512px cuts visual tokens substantially with no
    visible loss for scene-description tasks.
    """
    h, w = frame.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        frame = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    if not ok:
        raise ValueError("Failed to JPEG-encode frame for VLM request")
    return base64.b64encode(buf.tobytes()).decode("ascii")


class QwenLocalDescriber(Describer):
    """Qwen2.5-VL inference via a local Ollama server. Requires `ollama serve`
    running with the model pulled (`ollama pull qwen2.5vl:3b`). Raises
    clearly if the server/model isn't reachable.
    """

    def __init__(self, endpoint: str | None = None, model: str | None = None, timeout_s: float = 60.0):
        self._endpoint = endpoint or settings.vlm_endpoint
        self._model = model or settings.vlm_model
        self._timeout_s = timeout_s

    def describe(
        self, label: str, duration_s: float, context_reason: str = "", frame: np.ndarray | None = None
    ) -> EventDescription:
        if frame is None:
            raise ValueError(
                "QwenLocalDescriber requires a frame — a VLM with no image is just a "
                "worse TemplateDescriber. Pass the event's captured frame."
            )
        severity = _SEVERITY_BY_LABEL.get(label, "medium")
        prompt = (
            f"This is a security camera frame where a '{label}' was detected and tracked "
            f"for {duration_s:.0f} seconds. In one short sentence, describe exactly what "
            "you see in the image (setting, position, notable behavior). Do not speculate "
            "beyond what's visible."
        )

        try:
            response = requests.post(
                f"{self._endpoint}/api/generate",
                json={
                    "model": self._model,
                    "prompt": prompt,
                    "images": [_encode_frame_jpeg_base64(frame)],
                    "stream": False,
                    "options": {"num_predict": 80, "num_ctx": 2048},
                    "keep_alive": "30m",
                },
                timeout=self._timeout_s,
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            raise RuntimeError(
                f"QwenLocalDescriber could not reach Ollama at {self._endpoint} "
                f"(model={self._model}). Is `ollama serve` running with the model pulled? "
                f"Underlying error: {exc}"
            ) from exc

        text = response.json().get("response", "").strip()
        if not text:
            raise RuntimeError(f"QwenLocalDescriber got an empty response from Ollama (model={self._model})")

        return EventDescription(text=text, severity=severity, backend="qwen-local")


class FrontierDescriber(Describer):
    """A hosted frontier VLM. Requires an API key and is a paid,
    consent-gated action; never called automatically.
    """

    def __init__(self, api_key: str | None = None):
        if not api_key:
            raise ValueError(
                "FrontierDescriber requires an API key. This is a paid service — "
                "set one explicitly and consciously, it is never enabled by default."
            )
        self._api_key = api_key

    def describe(
        self, label: str, duration_s: float, context_reason: str = "", frame: np.ndarray | None = None
    ) -> EventDescription:
        raise NotImplementedError("Frontier VLM escalation is not wired up in this environment.")


def make_describer() -> Describer:
    backend = settings.vlm_backend
    if backend in ("none", "template"):
        return TemplateDescriber()
    if backend == "qwen-local":
        return QwenLocalDescriber()
    if backend == "frontier":
        raise ValueError("SENTINEL_VLM_BACKEND=frontier requires passing an API key explicitly — see FrontierDescriber.")
    raise ValueError(f"Unknown SENTINEL_VLM_BACKEND: {backend!r}")
