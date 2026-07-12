"""On-demand detection for the test UI (cloud/backend/static/index.html).

Loads the configured model once (lazy singleton) and runs inference on an
uploaded image, returning both the raw detections and an annotated JPEG ready
to embed as a data URI. This is a demo/testing surface, separate from the live
edge pipeline in edge/detector.py — same underlying model, different entry
point (a single uploaded photo vs. a continuous camera stream).
"""

from __future__ import annotations

import base64
from functools import lru_cache

import cv2
import numpy as np

from config import settings


@lru_cache(maxsize=1)
def _load_model():
    from ultralytics import YOLO

    model_path = settings.detector_model_path or "yolo12n.pt"
    return YOLO(model_path)


def detect_and_annotate(image_bytes: bytes, conf: float = 0.4) -> dict:
    """Runs detection on raw image bytes. Returns detections plus a base64
    JPEG with boxes drawn, ready for `<img src="data:image/jpeg;base64,...">`.
    """
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("Could not decode image — is it a valid image file?")

    model = _load_model()
    results = model.predict(frame, conf=conf, verbose=False)
    result = results[0]

    detections = []
    for box in result.boxes:
        x1, y1, x2, y2 = (round(float(v), 1) for v in box.xyxy[0])
        detections.append(
            {
                "label": result.names[int(box.cls)],
                "confidence": round(float(box.conf), 3),
                "box": [x1, y1, x2, y2],
            }
        )

    annotated = result.plot()  # BGR numpy array with boxes drawn
    ok, buf = cv2.imencode(".jpg", annotated)
    if not ok:
        raise RuntimeError("Failed to encode annotated image")

    return {
        "detections": detections,
        "image_base64": base64.b64encode(buf.tobytes()).decode("ascii"),
    }
