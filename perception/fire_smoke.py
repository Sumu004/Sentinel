"""Fire/smoke (VISION.md L1 row: "fire/smoke").

Same honest-limit shape as `audio.py`: a real trained fire/smoke detector
needs a labelled dataset (fire has good public datasets, unlike `package` —
but none is wired up yet since it wasn't part of D1's scope). This ships a
real, free, zero-download heuristic instead: HSV color-range thresholding
for fire (orange/yellow/red, high saturation) and grayish smoke (low
saturation, mid brightness, low color variance), gated by a minimum pixel
area so noise doesn't trigger it.

This is a legitimate, commonly-used pre-deep-learning technique — not a
placeholder — but it will false-positive on fire-colored objects (orange
traffic cones, sunset reflections) and miss fires outside its color/lighting
assumptions. Upgrade path: swap for a trained detector (e.g. fine-tuned
YOLO on a public fire/smoke dataset) behind the same `FireSmokeDetector`
interface once that's prioritized.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class FireSmokeResult:
    fire_detected: bool
    smoke_detected: bool
    fire_pixel_ratio: float
    smoke_pixel_ratio: float


class FireSmokeDetector:
    def analyze(self, frame: np.ndarray) -> FireSmokeResult:
        raise NotImplementedError


@dataclass
class HSVFireSmokeDetector(FireSmokeDetector):
    fire_pixel_ratio_threshold: float = 0.02
    smoke_pixel_ratio_threshold: float = 0.15

    def analyze(self, frame: np.ndarray) -> FireSmokeResult:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        total = frame.shape[0] * frame.shape[1]

        fire_mask = cv2.inRange(hsv, (0, 120, 150), (35, 255, 255))
        fire_ratio = float(np.count_nonzero(fire_mask)) / total

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        smoke_mask = cv2.inRange(hsv, (0, 0, 90), (180, 40, 220))
        local_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        smoke_ratio = float(np.count_nonzero(smoke_mask)) / total if local_var < 150 else 0.0

        return FireSmokeResult(
            fire_detected=fire_ratio >= self.fire_pixel_ratio_threshold,
            smoke_detected=smoke_ratio >= self.smoke_pixel_ratio_threshold,
            fire_pixel_ratio=round(fire_ratio, 4),
            smoke_pixel_ratio=round(smoke_ratio, 4),
        )
