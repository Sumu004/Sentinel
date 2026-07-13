"""Pose/action — fall detection (VISION.md L1, DECISIONS.md targets RTMPose edge / ViTPose cloud).

RTMPose needs the mmpose stack (heavy, awkward to pip-install cleanly — the
same shape of problem D3 hit with ByteTrack). Rather than ship nothing, this
splits the concern in two:

- `YoloPoseEstimator` — real keypoint inference using Ultralytics' pose models
  (`yolo11n-pose.pt`, auto-downloaded, Apache-2.0, free). Not RTMPose, but a
  real, working, free pose backbone today.
- `analyze_fall(keypoints)` — pure logic, no model dependency, so it's fully
  unit-testable without a network call or GPU. A fall is inferred from body
  aspect ratio + hip-below-shoulder inversion, a standard free heuristic
  (not a trained fall classifier — upgrade path is a fine-tuned action model
  once real fall footage exists to validate against).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PoseResult:
    keypoints: np.ndarray
    box: tuple[int, int, int, int]


class PoseEstimator:
    def estimate(self, frame: np.ndarray) -> list[PoseResult]:
        raise NotImplementedError


class YoloPoseEstimator(PoseEstimator):
    """Real keypoint inference. `model_path` defaults to Ultralytics' free
    COCO-pretrained pose model — no fine-tuning required to get real keypoints.
    """

    def __init__(self, model_path: str = "yolo11n-pose.pt", conf: float = 0.35):
        from ultralytics import YOLO

        self._model = YOLO(model_path)
        self._conf = conf

    def estimate(self, frame: np.ndarray) -> list[PoseResult]:
        results = self._model.predict(frame, conf=self._conf, verbose=False)
        out = []
        for result in results:
            if result.keypoints is None:
                continue
            for kpts, box in zip(result.keypoints.xy, result.boxes):
                x1, y1, x2, y2 = (int(v) for v in box.xyxy[0])
                out.append(PoseResult(keypoints=kpts.cpu().numpy(), box=(x1, y1, x2 - x1, y2 - y1)))
        return out


_L_SHOULDER, _R_SHOULDER = 5, 6
_L_HIP, _R_HIP = 11, 12


def analyze_fall(keypoints: np.ndarray, aspect_ratio_threshold: float = 1.0) -> bool:
    """Free heuristic fall detector: a standing person's bounding box is
    taller than wide and hips sit below shoulders; a fallen person's box
    flattens out (wide-vs-tall inverts) and hip/shoulder height collapses
    toward the same level. Both conditions must hold to reduce false
    positives from someone merely bending over.

    `keypoints` is (17, 2) in COCO order; a (0, 0) entry means "not detected"
    and is ignored.
    """
    valid = keypoints[(keypoints[:, 0] > 0) | (keypoints[:, 1] > 0)]
    if valid.shape[0] < 4:
        return False

    xs, ys = valid[:, 0], valid[:, 1]
    width = xs.max() - xs.min()
    height = ys.max() - ys.min()
    if height <= 0:
        return False
    is_flattened = (width / height) > aspect_ratio_threshold

    shoulder_ys = [y for y in (keypoints[_L_SHOULDER, 1], keypoints[_R_SHOULDER, 1]) if y > 0]
    hip_ys = [y for y in (keypoints[_L_HIP, 1], keypoints[_R_HIP, 1]) if y > 0]
    if not shoulder_ys or not hip_ys:
        return is_flattened

    shoulder_y = sum(shoulder_ys) / len(shoulder_ys)
    hip_y = sum(hip_ys) / len(hip_ys)
    torso_span = abs(hip_y - shoulder_y)
    is_collapsed = torso_span < (height * 0.15)

    return is_flattened and is_collapsed
