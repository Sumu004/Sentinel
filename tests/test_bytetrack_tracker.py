"""Tests for ByteTrackTracker against the real supervision ByteTrack
implementation, not mocked.
"""

from edge.bytetrack_tracker import ByteTrackTracker, make_tracker
from edge.detector import Detection
from edge.tracker import CentroidTracker, Track


def test_bytetrack_assigns_same_id_to_nearby_detection():
    tracker = ByteTrackTracker()
    det_a = Detection(label="person", confidence=0.9, box=(100, 100, 40, 80))
    tracks_1 = tracker.update([det_a])
    assert len(tracks_1) == 1
    track_id = tracks_1[0].track_id
    assert tracks_1[0].label == "person"

    det_b = Detection(label="person", confidence=0.9, box=(104, 102, 40, 80))
    tracks_2 = tracker.update([det_b])
    assert len(tracks_2) == 1
    assert tracks_2[0].track_id == track_id


def test_bytetrack_tracks_multiple_labels_independently():
    tracker = ByteTrackTracker()
    tracks = tracker.update(
        [
            Detection(label="person", confidence=0.9, box=(0, 0, 40, 80)),
            Detection(label="vehicle", confidence=0.9, box=(300, 300, 100, 60)),
        ]
    )
    assert len(tracks) == 2
    labels = {t.label for t in tracks}
    assert labels == {"person", "vehicle"}
    assert len({t.track_id for t in tracks}) == 2


def test_bytetrack_returns_track_dataclass_compatible_with_events_debounce():
    tracker = ByteTrackTracker()
    tracks = tracker.update([Detection(label="person", confidence=0.9, box=(10, 10, 40, 80))])
    assert len(tracks) == 1
    assert isinstance(tracks[0], Track)
    assert tracks[0].duration_s >= 0


def test_bytetrack_eventually_drops_stale_tracks():
    tracker = ByteTrackTracker()
    tracker.update([Detection(label="person", confidence=0.9, box=(0, 0, 40, 80))])
    assert len(tracker._tracks) == 1

    for _ in range(20):
        tracker.update([])

    assert len(tracker._tracks) == 0


def _with_tracker_backend(value: str):
    """Settings is a frozen dataclass, so object.__setattr__ is needed to
    patch it in tests — same pattern as tests/test_clear_recordings.py.
    """
    from config import settings

    original = settings.tracker_backend
    object.__setattr__(settings, "tracker_backend", value)
    return settings, original


def test_make_tracker_returns_centroid_by_default():
    settings, original = _with_tracker_backend("centroid")
    try:
        assert isinstance(make_tracker(), CentroidTracker)
    finally:
        object.__setattr__(settings, "tracker_backend", original)


def test_make_tracker_returns_bytetrack_when_configured():
    settings, original = _with_tracker_backend("bytetrack")
    try:
        assert isinstance(make_tracker(), ByteTrackTracker)
    finally:
        object.__setattr__(settings, "tracker_backend", original)


def test_make_tracker_raises_on_unknown_backend():
    settings, original = _with_tracker_backend("nonsense")
    try:
        try:
            make_tracker()
            assert False, "expected ValueError"
        except ValueError as e:
            assert "nonsense" in str(e)
    finally:
        object.__setattr__(settings, "tracker_backend", original)
