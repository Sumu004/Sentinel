from cloud.backend.notifications import ConsoleChannel, NotificationEngine, NotificationPayload


def _payload(severity: str = "medium") -> NotificationPayload:
    return NotificationPayload(
        event_id="e1", site_id="site-01", label="person", severity=severity, description="test event"
    )


def test_console_channel_sends():
    channel = ConsoleChannel()
    assert channel.send(_payload()) is True


def test_engine_routes_to_all_channels():
    calls = []

    class RecordingChannel:
        def send(self, payload):
            calls.append(payload.event_id)
            return True

    engine = NotificationEngine(channels=[RecordingChannel(), RecordingChannel()])
    results = engine.notify(_payload())
    assert len(calls) == 2
    assert all(results.values())


def test_engine_respects_min_severity():
    class RecordingChannel:
        def __init__(self):
            self.called = False

        def send(self, payload):
            self.called = True
            return True

    channel = RecordingChannel()
    engine = NotificationEngine(channels=[channel], min_severity="high")

    engine.notify(_payload(severity="low"))
    assert channel.called is False

    engine.notify(_payload(severity="high"))
    assert channel.called is True


def test_engine_survives_a_failing_channel():
    class BrokenChannel:
        def send(self, payload):
            raise RuntimeError("channel is down")

    engine = NotificationEngine(channels=[BrokenChannel()])
    results = engine.notify(_payload())
    assert results["BrokenChannel"] is False
