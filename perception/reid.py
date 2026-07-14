from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np


class ReIDEmbedder:
    def embed(self, crop: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class HistogramReID(ReIDEmbedder):
    def __init__(self, bins: tuple[int, int, int] = (8, 8, 8)):
        self._bins = bins

    def embed(self, crop: np.ndarray) -> np.ndarray:
        if crop.size == 0:
            return np.zeros(int(np.prod(self._bins)), dtype=np.float32)
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1, 2], None, list(self._bins), [0, 180, 0, 256, 0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        return hist.astype(np.float32)


class OSNetReID(ReIDEmbedder):
    _MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    _STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __init__(self, model_name: str = "osnet_x0_25"):
        import torch
        import torchreid

        self._torch = torch
        self._model = torchreid.reid.models.build_model(name=model_name, num_classes=1, pretrained=True)
        self._model.eval()

    def embed(self, crop: np.ndarray) -> np.ndarray:
        if crop.size == 0:
            return np.zeros(512, dtype=np.float32)

        resized = cv2.resize(crop, (128, 256))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        normalized = (rgb - self._MEAN) / self._STD
        tensor = self._torch.from_numpy(normalized.transpose(2, 0, 1)).unsqueeze(0).float()

        with self._torch.no_grad():
            embedding = self._model(tensor)

        return embedding.squeeze(0).cpu().numpy().astype(np.float32)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


@dataclass
class Gallery:
    embedder: ReIDEmbedder
    threshold: float = 0.85
    _identities: dict[str, np.ndarray] = field(default_factory=dict)

    def match_or_register(self, crop: np.ndarray) -> tuple[str, bool]:
        embedding = self.embedder.embed(crop)
        best_id, best_score = None, 0.0
        for identity_id, known in self._identities.items():
            score = cosine_similarity(embedding, known)
            if score > best_score:
                best_id, best_score = identity_id, score

        if best_id is not None and best_score >= self.threshold:
            self._identities[best_id] = embedding
            return best_id, False

        new_id = f"id-{len(self._identities)}"
        self._identities[new_id] = embedding
        return new_id, True
