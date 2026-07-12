from datetime import datetime, time, timezone

from reasoning.context import ContextEngine, ScheduleRule
from reasoning.describe import EventDescription, TemplateDescriber


def test_schedule_rule_suppresses_matching_window():
    engine = ContextEngine()
    engine.add_schedule(
        ScheduleRule(label="person", days=(1,), start=time(18, 0), end=time(19, 0), reason="scheduled cleaner")
    )
    tuesday_630pm = datetime(2026, 7, 14, 18, 30, tzinfo=timezone.utc)
    suppressed, reason = engine.should_suppress("person", tuesday_630pm)
    assert suppressed is True
    assert reason == "scheduled cleaner"


def test_schedule_rule_does_not_suppress_outside_window():
    engine = ContextEngine()
    engine.add_schedule(
        ScheduleRule(label="person", days=(1,), start=time(18, 0), end=time(19, 0), reason="scheduled cleaner")
    )
    tuesday_8pm = datetime(2026, 7, 14, 20, 0, tzinfo=timezone.utc)
    suppressed, _ = engine.should_suppress("person", tuesday_8pm)
    assert suppressed is False


def test_schedule_rule_does_not_suppress_wrong_day():
    engine = ContextEngine()
    engine.add_schedule(
        ScheduleRule(label="person", days=(1,), start=time(18, 0), end=time(19, 0), reason="scheduled cleaner")
    )
    wednesday_630pm = datetime(2026, 7, 15, 18, 30, tzinfo=timezone.utc)
    suppressed, _ = engine.should_suppress("person", wednesday_630pm)
    assert suppressed is False


def test_template_describer_normal_event():
    d = TemplateDescriber().describe("person", 5.0)
    assert isinstance(d, EventDescription)
    assert d.severity == "medium"
    assert d.backend == "template"
    assert "5s" in d.text


def test_template_describer_suppressed_event_is_low_severity():
    d = TemplateDescriber().describe("person", 5.0, context_reason="scheduled visit")
    assert d.severity == "low"
    assert "scheduled visit" in d.text


def test_template_describer_animal_is_low_severity():
    d = TemplateDescriber().describe("animal", 3.0)
    assert d.severity == "low"
