from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, time, timezone


@dataclass(frozen=True)
class ScheduleRule:
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
        return current >= self.start or current <= self.end


@dataclass
class ContextEngine:
    schedule_rules: list[ScheduleRule] = field(default_factory=list)

    def add_schedule(self, rule: ScheduleRule) -> None:
        self.schedule_rules.append(rule)

    def should_suppress(self, label: str, when: datetime | None = None) -> tuple[bool, str]:
        when = when or datetime.now(timezone.utc)
        for rule in self.schedule_rules:
            if rule.matches(label, when):
                return True, rule.reason or f"matches schedule rule for {label}"
        return False, ""
