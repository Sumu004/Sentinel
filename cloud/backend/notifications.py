"""Notification engine (DECISIONS.md D8, VISION.md L6 "notification is the
product"). Pluggable channels behind one interface — a real, free default
(console/log) plus stubs for the paid channels (SMS, push, webhook) that need
credentials this environment doesn't have.

For an unattended site there's no operator watching a wall of monitors — the
alert *is* the value delivered, so this interface exists even before a real
SMS/push provider is wired up: it's what every future channel plugs into.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

_SEVERITY_ORDER = {"low": 0, "medium": 1, "high": 2}


@dataclass(frozen=True)
class NotificationPayload:
    event_id: str
    site_id: str
    label: str
    severity: str
    description: str


class NotificationChannel:
    def send(self, payload: NotificationPayload) -> bool:
        raise NotImplementedError


class ConsoleChannel(NotificationChannel):
    """Free, always works, zero setup — logs the alert. This is the real
    default channel today; SMS/push/webhook are documented interfaces for
    when real provider credentials exist, not simulated here.
    """

    def send(self, payload: NotificationPayload) -> bool:
        logger.warning(
            "ALERT [%s] site=%s %s: %s", payload.severity.upper(), payload.site_id, payload.label, payload.description
        )
        return True


class WebhookChannel(NotificationChannel):
    """Posts the alert to any URL the user configures — Slack, Discord,
    Teams, or a custom endpoint. Free (no vendor account needed on Sentinel's
    side) but requires the user to already have a webhook URL to send to.
    """

    def __init__(self, url: str):
        if not url:
            raise ValueError("WebhookChannel requires a URL — set SENTINEL_WEBHOOK_URL")
        self._url = url

    def send(self, payload: NotificationPayload) -> bool:
        import requests

        try:
            response = requests.post(
                self._url,
                json={
                    "text": f"[{payload.severity.upper()}] {payload.site_id}: {payload.description}",
                    "event_id": payload.event_id,
                },
                timeout=5,
            )
            response.raise_for_status()
            return True
        except requests.RequestException as exc:
            logger.warning("Webhook notification failed: %s", exc)
            return False


class SMSChannel(NotificationChannel):
    """Phase 2.5 target — requires a paid SMS provider (Twilio et al.). Not
    wired up: needs an account and credentials only the user can provide.
    """

    def __init__(self, api_key: str | None = None):
        if not api_key:
            raise ValueError("SMSChannel requires a provider API key — this is a paid service, not enabled by default.")
        self._api_key = api_key

    def send(self, payload: NotificationPayload) -> bool:
        raise NotImplementedError("SMS notification is not wired up in this environment.")


class NotificationEngine:
    """Routes to every configured channel; a channel failing doesn't block
    the others. Escalation policy (severity routing, cooldowns to prevent
    alert fatigue) is a natural next layer on top of this — this is the
    dispatch mechanism it would sit on.
    """

    def __init__(self, channels: list[NotificationChannel] | None = None, min_severity: str = "low"):
        self._channels = channels if channels is not None else [ConsoleChannel()]
        self._min_severity = _SEVERITY_ORDER.get(min_severity, 0)

    def notify(self, payload: NotificationPayload) -> dict[str, bool]:
        if _SEVERITY_ORDER.get(payload.severity, 0) < self._min_severity:
            return {}

        results = {}
        for channel in self._channels:
            try:
                results[type(channel).__name__] = channel.send(payload)
            except Exception:
                logger.exception("Notification channel %s failed", type(channel).__name__)
                results[type(channel).__name__] = False
        return results
