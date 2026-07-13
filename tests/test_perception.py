import numpy as np

from perception.anomaly import StatisticalAnomalyDetector
from perception.audio import RMSLoudSoundDetector
from perception.fire_smoke import HSVFireSmokeDetector
from perception.pose import analyze_fall
from perception.reid import Gallery, HistogramReID, cosine_similarity


def test_analyze_fall_detects_flattened_collapsed_pose():
    standing = np.zeros((17, 2))
    standing[5] = (40, 10)
    standing[6] = (60, 10)
    standing[11] = (40, 90)
    standing[12] = (60, 90)
    assert bool(analyze_fall(standing)) is False

    fallen = np.zeros((17, 2))
    fallen[5] = (10, 50)
    fallen[6] = (30, 55)
    fallen[11] = (70, 52)
    fallen[12] = (90, 54)
    assert bool(analyze_fall(fallen)) is True


def test_analyze_fall_ignores_missing_keypoints():
    sparse = np.zeros((17, 2))
    sparse[5] = (10, 10)
    assert analyze_fall(sparse) is False


def test_reid_histogram_matches_same_color_crop():
    embedder = HistogramReID()
    red_crop_a = np.full((20, 20, 3), (0, 0, 200), dtype=np.uint8)
    red_crop_b = np.full((20, 20, 3), (0, 0, 210), dtype=np.uint8)
    blue_crop = np.full((20, 20, 3), (200, 0, 0), dtype=np.uint8)

    sim_same = cosine_similarity(embedder.embed(red_crop_a), embedder.embed(red_crop_b))
    sim_diff = cosine_similarity(embedder.embed(red_crop_a), embedder.embed(blue_crop))
    assert sim_same > sim_diff


def test_reid_gallery_recognizes_reappearing_identity():
    gallery = Gallery(embedder=HistogramReID(), threshold=0.9)
    red_crop = np.full((20, 20, 3), (0, 0, 200), dtype=np.uint8)
    blue_crop = np.full((20, 20, 3), (200, 0, 0), dtype=np.uint8)

    id1, is_new1 = gallery.match_or_register(red_crop)
    id2, is_new2 = gallery.match_or_register(blue_crop)
    id3, is_new3 = gallery.match_or_register(red_crop)

    assert is_new1 is True
    assert is_new2 is True
    assert is_new3 is False
    assert id3 == id1
    assert id2 != id1


def test_statistical_anomaly_needs_warmup_before_flagging():
    detector = StatisticalAnomalyDetector(min_samples=5)
    for _ in range(4):
        detector.observe("person", speed=1.0, duration_s=3.0)
    assert detector.is_anomalous("person", speed=50.0, duration_s=3.0) is False


def test_statistical_anomaly_flags_outlier_after_warmup():
    detector = StatisticalAnomalyDetector(min_samples=5, z_threshold=3.0)
    normal_speeds = [0.9, 1.0, 1.1, 0.95, 1.05, 1.0, 0.9, 1.1, 1.0, 0.95]
    for speed in normal_speeds:
        detector.observe("person", speed=speed, duration_s=3.0)
    assert detector.is_anomalous("person", speed=1.0, duration_s=3.0) is False
    assert detector.is_anomalous("person", speed=50.0, duration_s=3.0) is True


def test_rms_loud_sound_detector_thresholds_correctly():
    detector = RMSLoudSoundDetector(threshold=0.2)
    quiet = np.random.uniform(-0.01, 0.01, 4000).astype(np.float32)
    loud = np.random.uniform(-0.9, 0.9, 4000).astype(np.float32)

    assert detector.analyze(quiet).is_loud is False
    assert detector.analyze(loud).is_loud is True


def test_hsv_fire_smoke_detector_flags_fire_colored_region():
    detector = HSVFireSmokeDetector()
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    frame[:, :] = (0, 100, 255)

    result = detector.analyze(frame)
    assert result.fire_detected is True


def test_hsv_fire_smoke_detector_ignores_plain_dark_frame():
    detector = HSVFireSmokeDetector()
    frame = np.zeros((100, 100, 3), dtype=np.uint8)

    result = detector.analyze(frame)
    assert result.fire_detected is False
    assert result.smoke_detected is False
