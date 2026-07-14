"""Audio — loud-sound detection.

An RMS-energy-based detector: answers "something loud just happened",
not "what kind of sound". Upgrade path: a trained classifier over
mel-spectrograms behind the same AudioEventDetector interface.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AudioEvent:
    is_loud: bool
    rms: float


class AudioEventDetector:
    def analyze(self, samples: np.ndarray) -> AudioEvent:
        raise NotImplementedError


@dataclass
class RMSLoudSoundDetector(AudioEventDetector):
    """`threshold` is on RMS amplitude of normalized float32 samples in
    [-1, 1] — a sudden bang/crash/shout pushes RMS well above steady
    ambient noise.
    """

    threshold: float = 0.2

    def analyze(self, samples: np.ndarray) -> AudioEvent:
        if samples.size == 0:
            return AudioEvent(is_loud=False, rms=0.0)
        rms = float(np.sqrt(np.mean(np.square(samples, dtype=np.float64))))
        return AudioEvent(is_loud=rms >= self.threshold, rms=rms)
