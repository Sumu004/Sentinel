import time

from edge.detector import Detection
from edge.events import debounce
from edge.tracker import CentroidTracker


def test_tracker_assigns_same_id_to_nearby_detection():
    tracker = CentroidTracker()
    det_a = Detection(label="motion", confidence=1.0, box=(100, 100, 20, 20))
    tracks_1 = tracker.update([det_a])
    assert len(tracks_1) == 1
    track_id = tracks_1[0].track_id

    det_b = Detection(label="motion", confidence=1.0, box=(105, 102, 20, 20))  # small move
    tracks_2 = tracker.update([det_b])
    assert len(tracks_2) == 1
    assert tracks_2[0].track_id == track_id


def test_tracker_drops_stale_tracks():
    tracker = CentroidTracker()
    det = Detection(label="motion", confidence=1.0, box=(0, 0, 10, 10))
    tracker.update([det])
    assert len(tracker.tracks) == 1

    for _ in range(20):  # exceeds default track_max_age_frames
        tracker.update([])

    assert len(tracker.tracks) == 0


def test_event_does_not_fire_before_min_duration():
    tracker = CentroidTracker()
    det = Detection(label="motion", confidence=1.0, box=(0, 0, 10, 10))
    tracks = tracker.update([det])
    events = debounce(tracks)
    assert events == []  # just started — hasn't met SENTINEL_EVENT_MIN_DURATION_S yet


def test_event_fires_once_after_min_duration():
    tracker = CentroidTracker()
    det = Detection(label="motion", confidence=1.0, box=(0, 0, 10, 10))
    tracks = tracker.update([det])

    # simulate time passing past the debounce threshold
    tracks[0].first_seen -= 10
    tracks[0].last_seen = time.time()

    events_first = debounce(tracks)
    assert len(events_first) == 1

    events_second = debounce(tracks)
    assert events_second == []  # already emitted — must not fire twice
