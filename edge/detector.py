"""Detection backends.

D1 in DECISIONS.md targets a fine-tuned RF-DETR/YOLO12 bake-off. That needs a
trained model and a labelled dataset (Phase 2.1) — neither exists yet. Rather than
fake it, this module ships a real, zero-download, zero-dependency default
(background-subtraction motion detection) that proves the full pipeline
(source -> detect -> track -> debounce -> event -> evidence) end to end today,
plus a `ModelDetector` stub with the exact interface RF-DETR/YOLO12 will fill in
Phase 2.1. Swapping is a one-line config change (SENTINEL_DETECTOR_BACKEND).
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from config import settings


@dataclass(frozen=True)
class Detection:
    label: str
    confidence: float
    box: tuple[int, int, int, int]  # x, y, w, h


class Detector:
    def detect(self, frame: np.ndarray) -> list[Detection]:
        raise NotImplementedError


class MotionDetector(Detector):
    """Background-subtraction detector. Free, no model weights, no internet.

    Not a substitute for the trained model in D1 — it answers "did something
    move" not "what is it." Good enough to validate the rest of the pipeline
    (tracking, debouncing, evidence chain, cloud sync) before Phase 2.1 lands.
    """

    def __init__(self, min_area: int = 1500):
        self._bg = cv2.createBackgroundSubtractorMOG2(detectShadows=True)
        self._min_area = min_area

    def detect(self, frame: np.ndarray) -> list[Detection]:
        mask = self._bg.apply(frame)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        mask = cv2.dilate(mask, np.ones((5, 5), np.uint8), iterations=2)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detections = []
        for c in contours:
            area = cv2.contourArea(c)
            if area < self._min_area:
                continue
            x, y, w, h = cv2.boundingRect(c)
            detections.append(Detection(label="motion", confidence=1.0, box=(x, y, w, h)))
        return detections


class ModelDetector(Detector):
    """Loads a fine-tuned YOLO/RF-DETR model (DECISIONS.md D1) and runs real
    inference. Accepts an Ultralytics `.pt` (the bake-off winner from
    training/, or a COCO-pretrained yolo12n.pt to start) — `ultralytics`
    transparently loads both fine-tuned and pretrained checkpoints.

    Maps model output to the same `Detection` dataclass the motion detector
    emits, so the tracker/debounce/recorder pipeline downstream is identical
    regardless of backend. Honours SENTINEL_DETECT_CLASSES — only the
    configured classes raise events, everything else is ignored (e.g. `animal`
    can be detected for the suppression logic without alarming).
    """

    def __init__(self, model_path: str, conf: float = 0.35):
        if not model_path:
            raise ValueError(
                "SENTINEL_DETECTOR_BACKEND=model requires SENTINEL_DETECTOR_MODEL_PATH "
                "to point at trained weights (a fine-tuned .pt from training/, or "
                "yolo12n.pt to start). See training/README.md."
            )
        from ultralytics import YOLO  # local import: heavy dep, only loaded for this backend

        self._model = YOLO(model_path)
        self._conf = conf
        self._wanted = set(settings.detect_classes) if settings.detect_classes else None

    def detect(self, frame: np.ndarray) -> list[Detection]:
        results = self._model.predict(frame, conf=self._conf, verbose=False)
        detections: list[Detection] = []
        for result in results:
            names = result.names
            for box in result.boxes:
                label = names[int(box.cls)]
                if self._wanted is not None and label not in self._wanted:
                    continue
                x1, y1, x2, y2 = (int(v) for v in box.xyxy[0])
                detections.append(
                    Detection(
                        label=label,
                        confidence=float(box.conf),
                        box=(x1, y1, x2 - x1, y2 - y1),
                    )
                )
        return detections


def make_detector() -> Detector:
    if settings.detector_backend == "motion":
        return MotionDetector()
    if settings.detector_backend == "model":
        return ModelDetector(settings.detector_model_path or "yolo12n.pt")
    raise ValueError(f"Unknown SENTINEL_DETECTOR_BACKEND: {settings.detector_backend!r}")
