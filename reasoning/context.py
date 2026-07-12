"""Context engine (DECISIONS.md D4/D6, VISION.md L2).

Knows the site's *normal* — schedules, expected vehicles, roles — so an alert
only fires when reality contradicts expectation. This is the single biggest
false-alarm killer described in VISION.md: "the cleaner comes Tuesday 6pm"
should suppress that exact event, not require retraining a model.

Fully rule-based and free — no ML, no API, no download. This is the layer
that runs *before* anything gets to a VLM or a human: cheap, deterministic,
and it's where most of the false-alarm reduction actually happens in
practice, not in the detector.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, time, timezone


@dataclass(frozen=True)
class ScheduleRule:
    """Suppresses events matching a label during a recurring time window on
    specific days. `days` uses Python's weekday() convention: 0=Monday.
    """

    label: str
    days: tuple[int, ...]
    start: time
    end: time
    reason: str = ""

    def matches(self, label: str, when: datetime) -> bool:
        if label != self.label:
            return False
        if when.weekday() not in self.days:
            return False
        current = when.time()
        if self.start <= self.end:
            return self.start <= current <= self.end
        return current >= self.start or current <= self.end  # window crosses midnight


@dataclass
class ContextEngine:
    schedule_rules: list[ScheduleRule] = field(default_factory=list)

    def add_schedule(self, rule: ScheduleRule) -> None:
        self.schedule_rules.append(rule)

    def should_suppress(self, label: str, when: datetime | None = None) -> tuple[bool, str]:
        """Returns (suppressed, reason). An event that matches a schedule rule
        is expected, not a threat — suppress it and say why, so the operator
        can audit *why* nothing fired, not just trust a black box.
        """
        when = when or datetime.now(timezone.utc)
        for rule in self.schedule_rules:
            if rule.matches(label, when):
                return True, rule.reason or f"matches schedule rule for {label}"
        return False, ""
