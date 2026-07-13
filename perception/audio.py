"""Audio (VISION.md L1 row: "audio" — glass break, gunshot, raised voices).

Honest limit, stated up front: there is no free way to *classify* sound
events (glass break vs. gunshot vs. shouting) without either a paid API or a
labelled audio dataset to train on — neither exists for this project yet, the
same gap `training/README.md` documents for `package` images. Rather than
fake a classifier, this ships what *is* real and free: an RMS-energy-based
loud-sound detector. It answers "something loud just happened" — a real,
useful signal (a loud crash/bang worth flagging) — not "what kind of sound."

Upgrade path: swap `RMSLoudSoundDetector` for a trained classifier (e.g. a
small CNN over mel-spectrograms, once real site audio exists to label) behind
the same `AudioEventDetector` interface.
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
    """Free, real, no model weights. `threshold` is on RMS amplitude of
    normalized float32 samples in [-1, 1] — a sudden bang/crash/shout pushes
    RMS well above steady ambient noise.
    """

    threshold: float = 0.2

    def analyze(self, samples: np.ndarray) -> AudioEvent:
        if samples.size == 0:
            return AudioEvent(is_loud=False, rms=0.0)
        rms = float(np.sqrt(np.mean(np.square(samples, dtype=np.float64))))
        return AudioEvent(is_loud=rms >= self.threshold, rms=rms)
