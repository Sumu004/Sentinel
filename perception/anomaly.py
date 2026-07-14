from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class _RunningStats:
    count: int = 0
    mean: float = 0.0
    m2: float = 0.0

    def update(self, value: float) -> None:
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        delta2 = value - self.mean
        self.m2 += delta * delta2

    @property
    def std(self) -> float:
        if self.count < 2:
            return 0.0
        return math.sqrt(self.m2 / (self.count - 1))


class AnomalyDetector:
    def score(self, label: str, speed: float, duration_s: float) -> float:
        raise NotImplementedError

    def is_anomalous(self, label: str, speed: float, duration_s: float) -> bool:
        raise NotImplementedError


@dataclass
class StatisticalAnomalyDetector(AnomalyDetector):
    z_threshold: float = 3.0
    min_samples: int = 8
    _speed_stats: dict[str, _RunningStats] = field(default_factory=dict)
    _duration_stats: dict[str, _RunningStats] = field(default_factory=dict)

    def observe(self, label: str, speed: float, duration_s: float) -> None:
        self._speed_stats.setdefault(label, _RunningStats()).update(speed)
        self._duration_stats.setdefault(label, _RunningStats()).update(duration_s)

    def score(self, label: str, speed: float, duration_s: float) -> float:
        speed_z = self._z(self._speed_stats.get(label), speed)
        duration_z = self._z(self._duration_stats.get(label), duration_s)
        return max(speed_z, duration_z)

    def is_anomalous(self, label: str, speed: float, duration_s: float) -> bool:
        stats = self._speed_stats.get(label)
        if stats is None or stats.count < self.min_samples:
            return False
        return self.score(label, speed, duration_s) >= self.z_threshold

    @staticmethod
    def _z(stats: _RunningStats | None, value: float) -> float:
        if stats is None or stats.count < 2 or stats.std == 0:
            return 0.0
        return abs(value - stats.mean) / stats.std
