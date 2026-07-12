"""Event description (DECISIONS.md D4, VISION.md L2 "understands the place").

Three backends behind one interface (SENTINEL_VLM_BACKEND):

- "template" (default, free, always works) — a deterministic, rule-based
  description from the event's own fields (label, duration, time of day,
  context-engine suppression reason if any). No download, no API key, no
  GPU. This is genuinely what ships today.
- "qwen-local" — Qwen2.5-VL running locally via Ollama/vLLM (D4). NOT wired
  up this session: real local VLM inference needs either a GPU-backed
  runtime (e.g. moondream's "Photon" backend, or an Ollama server) that has
  to be set up and running separately — that's a real infrastructure step
  for you to do, not something to fake here.
- "frontier" — a hosted VLM (Claude/GPT-4o-class) for the escalation tier
  (D4). Requires an API key and is a paid, consent-gated action — not
  something this module will call on its own.

Calling make_describer() with "qwen-local" or "frontier" before that
infrastructure exists raises clearly, the same pattern edge/detector.py's
ModelDetector used before a trained model existed.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from config import settings


@dataclass(frozen=True)
class EventDescription:
    text: str
    severity: str  # "low" | "medium" | "high"
    backend: str


class Describer:
    def describe(self, label: str, duration_s: float, context_reason: str = "") -> EventDescription:
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
    """Free, deterministic, no dependencies. Real for today's pipeline —
    not a placeholder for a VLM, a genuine rule-based description generator
    that works with zero setup.
    """

    def describe(self, label: str, duration_s: float, context_reason: str = "") -> EventDescription:
        hour = datetime.now(timezone.utc).hour
        time_of_day = "overnight" if hour < 6 or hour >= 22 else ("daytime" if 6 <= hour < 18 else "evening")
        severity = _SEVERITY_BY_LABEL.get(label, "medium")

        if context_reason:
            text = f"{label.capitalize()} detected ({time_of_day}) — suppressed: {context_reason}."
            severity = "low"
        else:
            text = f"{label.capitalize()} detected and tracked for {duration_s:.0f}s ({time_of_day})."

        return EventDescription(text=text, severity=severity, backend="template")


class QwenLocalDescriber(Describer):
    """Phase 2.4 target — requires a running local VLM server (Ollama/vLLM
    serving Qwen2.5-VL, per DECISIONS.md D4). Not set up this session.
    """

    def __init__(self, endpoint: str = "http://127.0.0.1:11434"):
        self._endpoint = endpoint

    def describe(self, label: str, duration_s: float, context_reason: str = "") -> EventDescription:
        raise NotImplementedError(
            "QwenLocalDescriber requires a running local VLM server (Ollama or vLLM "
            "serving Qwen2.5-VL) at SENTINEL_VLM_ENDPOINT — not set up in this environment. "
            "Use SENTINEL_VLM_BACKEND=template until that infrastructure exists."
        )


class FrontierDescriber(Describer):
    """Phase 2.4 escalation tier — a hosted frontier VLM. Requires an API key
    and is a paid, consent-gated action; never called automatically.
    """

    def __init__(self, api_key: str | None = None):
        if not api_key:
            raise ValueError(
                "FrontierDescriber requires an API key. This is a paid service — "
                "set one explicitly and consciously, it is never enabled by default."
            )
        self._api_key = api_key

    def describe(self, label: str, duration_s: float, context_reason: str = "") -> EventDescription:
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
