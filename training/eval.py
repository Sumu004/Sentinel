"""Bake-off evaluation (DECISIONS.md D1, training/README.md methodology).

Two parts:
1. Held-out mAP — delegates to each library's own validator, runs today.
2. False-alarms-per-camera-per-day — the metric that actually matters for an
   unattended site. This needs hours of real, event-free footage from a
   deployed camera, which doesn't exist yet — `count_false_alarms` is a real,
   runnable function, but it's only as good as the footage you point it at.
   Wire it up once the Phase 2.0 pipeline has been recording for a few days.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from config import settings
from edge.detector import Detector
from edge.events import debounce
from edge.tracker import CentroidTracker


def eval_yolo12(weights: Path, data_yaml: Path) -> dict:
    from ultralytics import YOLO

    model = YOLO(str(weights))
    metrics = model.val(data=str(data_yaml))
    return {"mAP50": metrics.box.map50, "mAP50_95": metrics.box.map}


def eval_rfdetr(checkpoint: Path, dataset_dir: Path) -> dict:
    from rfdetr import RFDETRBase

    model = RFDETRBase(pretrain_weights=str(checkpoint))
    metrics = model.eval(dataset_dir=str(dataset_dir))
    return metrics


def count_false_alarms(detector: Detector, footage_path: Path, hours_of_footage: float) -> float:
    """Runs `detector` over event-free footage and returns alarms/camera/day.

    `footage_path` should be a video known to contain *no* real events — its
    only purpose is measuring the spurious-detection rate. Reuses the exact
    tracker/debounce path from the live pipeline (edge/tracker.py,
    edge/events.py) so the number reflects what would actually alert someone,
    not raw per-frame false positives.
    """
    import cv2

    cap = cv2.VideoCapture(str(footage_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open footage: {footage_path}")

    tracker = CentroidTracker()
    alarm_count = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        detections = detector.detect(frame)
        tracks = tracker.update(detections)
        alarm_count += len(debounce(tracks))
    cap.release()

    if hours_of_footage <= 0:
        raise ValueError("hours_of_footage must be > 0")
    return alarm_count * (24.0 / hours_of_footage)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_yolo = sub.add_parser("yolo12-map")
    p_yolo.add_argument("--weights", type=Path, required=True)
    p_yolo.add_argument("--data", type=Path, required=True)

    p_fa = sub.add_parser("false-alarms")
    p_fa.add_argument("--footage", type=Path, required=True)
    p_fa.add_argument("--hours", type=float, required=True)

    args = parser.parse_args()

    if args.command == "yolo12-map":
        print(eval_yolo12(args.weights, args.data))
    elif args.command == "false-alarms":
        from edge.detector import make_detector

        rate = count_false_alarms(make_detector(), args.footage, args.hours)
        print(f"False alarms per camera per day: {rate:.2f}")


if __name__ == "__main__":
    main()
